"""Single-node or torchrun distributed fine-tuning for HGGT."""

import contextlib
import json
import logging
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from hggt.training.checkpoint import (
    load_file, load_model_weights, restore_rng, save_checkpoint, validate_resume,
)


def learning_rate(step, total_steps, peak, minimum, warmup_fraction):
    progress = min(max(step / total_steps, 0.0), 1.0)
    if progress < warmup_fraction:
        return minimum + (peak - minimum) * progress / warmup_fraction
    progress = (progress - warmup_fraction) / (1 - warmup_fraction)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return value


def all_finite(value):
    flag = torch.isfinite(value.detach()).all().to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def check_batch(batch):
    required = ("images", "extrinsics", "intrinsics", "hand_pose", "betas", "keypoints_3d", "keypoints_2d")
    for key in required:
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise ValueError(f"Batch is missing tensor '{key}'")
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"Batch contains non-finite values in '{key}'")
    images = batch["images"]
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [batch, views, 3, height, width]")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("images must be in [0, 1]")
    batch_size, views = images.shape[:2]
    if batch["extrinsics"].shape != (batch_size, views, 3, 4):
        raise ValueError("extrinsics must have shape [batch, views, 3, 4]")
    if batch["intrinsics"].shape != (batch_size, views, 3, 3):
        raise ValueError("intrinsics must have shape [batch, views, 3, 3]")


def check_data(cfg):
    """Check both streams before allocating the transformer or using a GPU."""
    torch.set_num_threads(2)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = instantiate(cfg.data.train, _recursive_=False)
    loader = iter(data.get_loader(epoch=0))
    for _ in range(2):
        try:
            batch, name, _ = next(loader)
        except StopIteration as exc:
            raise ValueError("A training stream is too short to produce a batch") from exc
        check_batch(batch)
        print(f"{name}: images={tuple(batch['images'].shape)}, finite tensors, valid camera shapes")


class Trainer:
    def __init__(self, cfg, model, geometry, criterion, optimizer, scaler, device, rank=0, world_size=1):
        self.cfg = cfg
        self.model = model
        self.raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        self.geometry = geometry
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.global_step = 0
        self.dtype = getattr(torch, cfg.optim.amp_dtype)
        self.total_steps = cfg.train.epochs * cfg.train.batches_per_epoch // cfg.train.accum_steps
        self.output = Path(cfg.logging.output_dir).expanduser()
        self.writer = None
        if rank == 0 and cfg.logging.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.output / "tensorboard"))

    def compute_loss(self, batch):
        batch = self.geometry.normalize_batch(batch)
        batch = to_device(batch, self.device)
        autocast = (torch.autocast(device_type=self.device.type, dtype=self.dtype)
                    if self.dtype != torch.float32 else contextlib.nullcontext())
        with autocast:
            predictions = self.model(images=batch["images"])
        # MANO and camera geometry use float32 even when the backbone uses AMP.
        with torch.autocast(device_type=self.device.type, enabled=False):
            predictions = self.geometry.decode_predictions(predictions, batch)
            losses = self.criterion(predictions, batch)
        return losses

    def log_losses(self, losses, phase, source, epoch, batch_idx):
        names = sorted(losses)
        values = torch.stack([losses[key].detach().float() for key in names])
        if dist.is_initialized():
            dist.all_reduce(values)
            values /= self.world_size
        if self.rank != 0:
            return
        metrics = dict(zip(names, values.cpu().tolist()))
        record = dict(phase=phase, source=source, epoch=epoch + 1, batch=batch_idx, step=self.global_step,
                      lr=self.optimizer.param_groups[0]["lr"], **metrics)
        with (self.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        logging.info("%s epoch=%d batch=%d step=%d %s objective=%.6f lr=%.3g",
                     phase, epoch + 1, batch_idx, self.global_step, source, metrics["objective"], record["lr"])
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{source}/{key}", value,
                                       epoch * self.cfg.train.batches_per_epoch + batch_idx)
            self.writer.add_scalar("optim/lr", record["lr"], self.global_step)

    def train_epoch(self, loader, epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        accumulation = self.cfg.train.accum_steps
        for index in range(self.cfg.train.batches_per_epoch):
            batch, source, position = self.next_batch(iterator)
            if position != index % accumulation:
                raise ValueError("Data loader accumulation position does not match train.accum_steps")
            update = (index + 1) % accumulation == 0
            sync = self.model.no_sync() if isinstance(self.model, DistributedDataParallel) and not update else contextlib.nullcontext()
            with sync:
                losses = self.compute_loss(batch)
                if not all_finite(losses["objective"]):
                    raise FloatingPointError(f"Non-finite {source} loss at epoch {epoch + 1}, batch {index}")
                self.scaler.scale(losses["objective"] / accumulation).backward()
            if update:
                lr = learning_rate(self.global_step, self.total_steps, self.cfg.optim.lr,
                                   self.cfg.optim.min_lr, self.cfg.optim.warmup_fraction)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.scaler.unscale_(self.optimizer)
                # The recipe clips the three modules independently.
                norms = []
                for name in ("aggregator", "camera_head", "mano_head"):
                    module = getattr(self.raw_model, name)
                    norms.append(torch.nn.utils.clip_grad_norm_(module.parameters(), self.cfg.optim.grad_clip))
                if not all_finite(torch.stack(norms)):
                    raise FloatingPointError(f"Non-finite gradients at optimizer step {self.global_step}")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            # Log both data streams at the selected interval of optimizer steps.
            if (index // accumulation) % self.cfg.logging.log_every == 0:
                self.log_losses(losses, "train", source, epoch, index)
            del batch, losses

    def next_batch(self, iterator, allow_end=False):
        batch = next(iterator, None)
        available = torch.tensor(int(batch is not None), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not available.item():
            if allow_end:
                return None
            raise ValueError("A data stream ended before the configured batch limit; reduce the limit or add data")
        return batch

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        iterator = iter(loader)
        totals = {}
        for index in range(self.cfg.val.batches):
            item = self.next_batch(iterator, allow_end=True)
            if item is None:
                break
            batch, source, _ = item
            losses = self.compute_loss(batch)
            if not all_finite(losses["objective"]):
                raise FloatingPointError(f"Non-finite validation loss from {source}")
            count, sums = totals.setdefault(source, [0, {key: torch.zeros_like(value) for key, value in losses.items()}])
            totals[source][0] = count + 1
            for key, value in losses.items():
                sums[key] += value
        if not totals:
            raise ValueError("Validation streams did not produce any batches; add data or reduce max_img_per_gpu")
        for source, (count, sums) in totals.items():
            self.log_losses({key: value / count for key, value in sums.items()}, "val", source, epoch,
                            self.cfg.train.batches_per_epoch)


def run(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("HGGT fine-tuning requires a CUDA GPU; --check-data runs on CPU")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if cfg.optim.amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; set optim.amp_dtype=float16")
    if world_size > 1:
        dist.init_process_group(backend=cfg.distributed.backend, timeout=timedelta(minutes=cfg.distributed.timeout_minutes))
    trainer = None
    try:
        seed = cfg.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = cfg.cuda.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.cuda.allow_tf32
        torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cuda.cudnn_deterministic
        output = Path(cfg.logging.output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        if not cfg.checkpoint.resume_path and (output / "checkpoints/checkpoint.pt").exists():
            raise FileExistsError("Output already contains a checkpoint; set resume_path or choose a new logging.output_dir")
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                            format="%(asctime)s %(levelname)s %(message)s")
        plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        train_data = instantiate(cfg.data.train, _recursive_=False)
        val_data = instantiate(cfg.data.val, _recursive_=False) if cfg.val.batches else None
        checkpoint = load_file(cfg.checkpoint.resume_path or cfg.checkpoint.init_path)
        if cfg.checkpoint.resume_path:
            validate_resume(checkpoint, plain_cfg, world_size)
        model = instantiate(cfg.model)
        load_model_weights(model, checkpoint)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
                                      eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.amp_dtype == "float16")
        start_epoch, global_step, saved_rng = 0, 0, None
        if cfg.checkpoint.resume_path:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["next_epoch"]
            global_step = checkpoint["global_step"]
            saved_rng = checkpoint["rng_states"][rank]
        del checkpoint
        from hggt.training.hand import HandGeometry
        from hggt.training.loss import HandCameraLoss

        geometry = HandGeometry(cfg.paths.mano_models, device=device)
        criterion = HandCameraLoss(**OmegaConf.to_container(cfg.loss, resolve=True))
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                            find_unused_parameters=cfg.distributed.find_unused_parameters,
                                            gradient_as_bucket_view=cfg.distributed.gradient_as_bucket_view,
                                            bucket_cap_mb=cfg.distributed.bucket_cap_mb,
                                            broadcast_buffers=cfg.distributed.broadcast_buffers)
        trainer = Trainer(cfg, model, geometry, criterion, optimizer, scaler, device, rank, world_size)
        trainer.global_step = global_step
        if saved_rng:
            restore_rng(saved_rng)
        if rank == 0:
            OmegaConf.save(cfg, output / ("resume_config.yaml" if cfg.checkpoint.resume_path else "config.yaml"))
            logging.info("Full-model fine-tuning: %d processes, %d updates planned", world_size, trainer.total_steps)
        for epoch in range(start_epoch, cfg.train.epochs):
            trainer.train_epoch(train_data.get_loader(epoch), epoch)
            if val_data and (epoch + 1) % cfg.val.every == 0:
                trainer.validate(val_data.get_loader(epoch), epoch)
            save_checkpoint(trainer.raw_model, optimizer, scaler, plain_cfg, epoch, trainer.global_step,
                            output / "checkpoints", rank, world_size)
            if dist.is_initialized():
                dist.barrier()
    finally:
        if trainer is not None and trainer.writer:
            trainer.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
