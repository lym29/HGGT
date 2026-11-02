"""Command-line entry point for HGGT fine-tuning."""

import argparse
import math
from pathlib import Path

from omegaconf import OmegaConf


def validate_config(cfg):
    """Validate the recipe without importing CUDA or opening datasets."""
    positive = {
        "train.epochs": cfg.train.epochs,
        "train.batches_per_epoch": cfg.train.batches_per_epoch,
        "train.accum_steps": cfg.train.accum_steps,
        "train.save_every": cfg.train.save_every,
        "val.every": cfg.val.every,
        "logging.log_every": cfg.logging.log_every,
        "max_img_per_gpu": cfg.max_img_per_gpu,
        "img_size": cfg.img_size,
        "patch_size": cfg.patch_size,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.train.accum_steps % 2:
        raise ValueError("train.accum_steps must be even to balance the two data streams")
    if cfg.train.batches_per_epoch % cfg.train.accum_steps:
        raise ValueError("train.batches_per_epoch must be divisible by train.accum_steps")
    for name, value in {"num_workers": cfg.num_workers, "val.batches": cfg.val.batches, "seed": cfg.seed}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if cfg.val.batches % 2:
        raise ValueError("val.batches must be even")
    for name in ("lr", "min_lr", "warmup_fraction", "weight_decay", "eps", "grad_clip"):
        value = cfg.optim[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"optim.{name} must be finite")
    if cfg.optim.grad_clip <= 0 or cfg.optim.eps <= 0 or cfg.optim.weight_decay < 0:
        raise ValueError("grad_clip and eps must be positive; weight_decay must be nonnegative")
    if len(cfg.optim.betas) != 2 or not all(0 <= beta < 1 for beta in cfg.optim.betas):
        raise ValueError("optim.betas must contain two values in [0, 1)")
    if not 0 < cfg.optim.warmup_fraction < 1:
        raise ValueError("optim.warmup_fraction must be between zero and one")
    if not 0 <= cfg.optim.min_lr <= cfg.optim.lr or cfg.optim.lr <= 0:
        raise ValueError("Learning rates must satisfy 0 <= min_lr <= lr and lr > 0")
    if cfg.optim.amp_dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("optim.amp_dtype must be bfloat16, float16, or float32")
    if cfg.checkpoint.init_path and cfg.checkpoint.resume_path:
        raise ValueError("Set only checkpoint.init_path or checkpoint.resume_path")
    if cfg.model._target_ != "hggt.models.hggt.HGGT":
        raise ValueError("This recipe requires hggt.models.hggt.HGGT")
    if not cfg.model.enable_camera or not cfg.model.enable_mano:
        raise ValueError("Camera and MANO heads are required")
    if cfg.model.enable_point or cfg.model.enable_depth:
        raise ValueError("This recipe supervises only the camera and MANO heads")
    if cfg.patch_size != 14 or cfg.img_size < 14 or cfg.img_size % cfg.patch_size:
        raise ValueError("img_size must be a positive multiple of the HGGT patch size (14)")
    for phase in (cfg.data.train, cfg.data.val):
        lo, hi = phase.common_config2.img_nums
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (lo, hi)):
            raise ValueError("View limits must be integers")
        if not 2 <= lo <= hi <= cfg.max_img_per_gpu:
            raise ValueError("Multi-view limits must satisfy 2 <= min <= max <= max_img_per_gpu")


def check_paths(cfg):
    for name in ("hamer_data", "synthetic_data", "mano_models"):
        path = Path(cfg.paths[name]).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"paths.{name} is not a directory: {path}")
    mano = Path(cfg.paths.mano_models).expanduser() / "MANO_RIGHT.pkl"
    if not mano.is_file():
        raise FileNotFoundError(f"Download the licensed MANO model first: {mano}")


def load_config(path, overrides=()):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    # Expand home paths before resolving dataset interpolations so that the
    # preflight checks and actual loaders use exactly the same directories.
    for name in cfg.paths:
        cfg.paths[name] = str(Path(cfg.paths[name]).expanduser())
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/finetune.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Resolve and validate configuration only")
    mode.add_argument("--check-data", action="store_true", help="Read one CPU batch from each training stream")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides, e.g. train.epochs=1")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.check_config:
        print(OmegaConf.to_yaml(cfg))
        return
    check_paths(cfg)
    if args.check_data:
        from hggt.training.trainer import check_data

        check_data(cfg)
    else:
        if not (cfg.checkpoint.init_path or cfg.checkpoint.resume_path):
            parser.error("Provide checkpoint.init_path for fine-tuning or checkpoint.resume_path to resume")
        from hggt.training.trainer import run

        run(cfg)


if __name__ == "__main__":
    main()
