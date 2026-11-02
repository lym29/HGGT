"""Weight initialization and epoch-boundary training checkpoints."""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


FORMAT = "hggt-training-v1"


def load_file(path):
    """Load a local weight file. Only load trusted PyTorch pickle checkpoints."""
    path = Path(path).expanduser()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors

        return load_safetensors(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_weights(model, checkpoint):
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Expected a state dictionary or a checkpoint with a 'model' state dictionary")
    # DDP wrapping changes the prefix, but VGGT-derived HGGT checkpoints use the
    # same aggregator/camera_head/mano_head parameter names as the public model.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith(("depth_head.", "point_head."))}
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Checkpoint does not match this HGGT architecture. "
            f"Missing: {missing[:8]}; unexpected: {unexpected[:8]}; shape mismatch: {mismatched[:8]}. "
            "Use a compatible HGGT checkpoint and match model.mano_qk_norm."
        )
    model.load_state_dict(state, strict=True)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(model, optimizer, scaler, cfg, epoch, global_step, directory, rank=0, world_size=1):
    local_rng = rng_state()
    states = [local_rng]
    if dist.is_initialized():
        states = [None] * world_size
        dist.all_gather_object(states, local_rng)
    if rank != 0:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "next_epoch": epoch + 1, "global_step": global_step,
        "config": cfg, "rng_states": states, "world_size": world_size,
    }
    temporary = directory / "checkpoint.pt.tmp"
    latest = directory / "checkpoint.pt"
    torch.save(payload, temporary)
    os.replace(temporary, latest)
    if (epoch + 1) % cfg["train"]["save_every"] == 0:
        snapshot = directory / f"checkpoint_{epoch + 1}.pt"
        shutil.copyfile(latest, str(snapshot) + ".tmp")
        os.replace(str(snapshot) + ".tmp", snapshot)


def validate_resume(checkpoint, cfg, world_size):
    required = {"model", "optimizer", "scaler", "next_epoch", "global_step", "config", "rng_states", "world_size"}
    if checkpoint.get("format") != FORMAT or not required.issubset(checkpoint):
        raise ValueError("Full resume requires a checkpoint saved by this trainer; use init_path for other weights")
    if checkpoint["world_size"] != world_size:
        raise ValueError("Resume with the same number of processes as the saved run")
    # Preserve the schedule and batch semantics of a resumed run. Paths and log
    # locations may change, while the optimizer and training plan must match.
    for key in ("model", "optim", "train", "val", "max_img_per_gpu", "seed", "num_workers", "loss", "cuda"):
        if checkpoint["config"][key] != cfg[key]:
            raise ValueError(f"Resume configuration differs at '{key}'; use init_path to start a new fine-tune")
    def data_signature(value):
        if isinstance(value, dict):
            return {key: data_signature(item) for key, item in value.items()
                    if key not in ("hamer_data_dir", "HOI_DATA_DIR", "mano_model_path")}
        if isinstance(value, list):
            return [data_signature(item) for item in value]
        return value
    if data_signature(checkpoint["config"]["data"]) != data_signature(cfg["data"]):
        raise ValueError("Resume dataset/sampler/augmentation settings differ; use init_path for a new fine-tune")
