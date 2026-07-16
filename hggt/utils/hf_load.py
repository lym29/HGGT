"""HGGT checkpoint / Hugging Face loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hggt.models.hggt import HGGT

# Official Hub release: https://huggingface.co/catmint123/HGGT
DEFAULT_HF_REPO = "catmint123/HGGT"


def _load_hggt_from_state_dict(
    model_state: dict[str, Any],
    img_size: int,
    device: str | torch.device,
) -> HGGT:
    mano_qk_norm = any("cross_attn_trunk" in key and "q_norm" in key for key in model_state)
    model = HGGT(
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=False,
        enable_depth=False,
        enable_mano=True,
        mano_qk_norm=mano_qk_norm,
    )
    model.load_state_dict(model_state, strict=False)
    return model.eval().to(device)


def load_hggt(
    checkpoint_path: str,
    img_size: int,
    device: str | torch.device,
    *,
    prefer_from_pretrained: bool = False,
) -> HGGT:
    """Load HGGT from a local checkpoint file or Hugging Face ``from_pretrained``.

    Args:
        checkpoint_path: Local weight file, or Hub repo id such as ``catmint123/HGGT``.
        img_size: Used when constructing from a raw training state dict.
        device: Target device.
        prefer_from_pretrained: If True (or path is not a local file), use
            ``HGGT.from_pretrained``.
    """
    path = Path(checkpoint_path)
    if path.is_file() and not prefer_from_pretrained:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        return _load_hggt_from_state_dict(model_state, img_size, device)

    model = HGGT.from_pretrained(checkpoint_path)
    return model.eval().to(device)
