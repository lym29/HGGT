#!/usr/bin/env python3
"""Single-frame multi-view hand mesh demo on pre-cropped images.

Assumes each input image is already a hand-centric square crop (as used during
HGGT training / POEM preprocessing). Hand detection is not included in this
demo; a detection-based pipeline will be added later.

Example::

    python demo/demo_multiview_images.py \\
        --image_folder examples/multiview/Arctic/sample_0000 \\
        --output_dir outputs/demo_arctic_0000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from hggt.utils.hf_load import DEFAULT_HF_REPO, load_hggt
from hggt.utils.mano_utils import decode_mano_and_cameras, load_mano_model, overlay_hand_on_image
from hggt.utils.multiview_io import build_mosaic, load_multiview_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HGGT demo: pre-cropped multi-view hand images (no detection)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_folder", type=str, help="Folder of pre-cropped views")
    group.add_argument("--images", type=str, nargs="+", help="Explicit image paths")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repo id (default: {DEFAULT_HF_REPO})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local .pt/.bin. If set, skips from_pretrained.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(ROOT / "assets" / "mano_v1_2" / "models"),
        help="MANO model directory (optional; enables mesh overlay)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "demo_multiview"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--proxy", type=str, default=None, help="Optional HTTP(S) proxy")
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=None,
        help="Optional HF_ENDPOINT override, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()


def _apply_hub_env(proxy: str | None, hf_endpoint: str | None) -> None:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
    if proxy:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[key] = proxy
    if Path("/.dockerenv").exists():
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            value = os.environ.get(key)
            if value and ("127.0.0.1" in value or "localhost" in value):
                os.environ[key] = value.replace("127.0.0.1", "172.17.0.1").replace(
                    "localhost", "172.17.0.1"
                )


def main() -> None:
    args = parse_args()
    _apply_hub_env(args.proxy, args.hf_endpoint)
    device = args.device
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
        else torch.float32
    )

    print(f"Loading images...")
    images_tensor, images_rgb, paths = load_multiview_folder(
        image_folder=args.image_folder,
        image_paths=args.images,
        device=device,
        img_size=args.img_size,
    )
    num_views = images_tensor.shape[1]
    print(f"  {num_views} views from {[p.name for p in paths]}")

    if args.checkpoint:
        model_ref = args.checkpoint
        prefer_hub = False
        print(f"Loading HGGT from local checkpoint: {model_ref}")
    else:
        model_ref = args.repo_id
        prefer_hub = True
        print(f"Loading HGGT via from_pretrained({model_ref!r})")

    model = load_hggt(
        model_ref,
        args.img_size,
        device,
        prefer_from_pretrained=prefer_hub,
    )

    print("Running inference...")
    device_type = device.split(":")[0]
    with torch.no_grad(), torch.autocast(
        device_type=device_type, dtype=dtype, enabled=(device_type != "cpu")
    ):
        preds = model(images_tensor)

    os.makedirs(args.output_dir, exist_ok=True)
    mosaic = build_mosaic(images_rgb)
    mosaic_path = os.path.join(args.output_dir, "input_mosaic.jpg")
    cv2.imwrite(mosaic_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved {mosaic_path}")

    save_dict: dict = {
        "num_views": np.array(num_views, dtype=np.int32),
        "image_paths": np.array([str(p) for p in paths]),
    }
    if "mano_params" in preds:
        save_dict["mano_hand_pose"] = preds["mano_params"]["hand_pose"][0].detach().cpu().numpy()
        save_dict["mano_betas"] = preds["mano_params"]["betas"][0].detach().cpu().numpy()
        save_dict["mano_transl"] = preds["mano_params"]["transl"][0].detach().cpu().numpy()
    if "pose_enc" in preds:
        save_dict["pose_enc"] = preds["pose_enc"][0].detach().cpu().numpy()

    mano_path = Path(args.mano_model_path)
    if mano_path.is_dir() and (mano_path / "MANO_RIGHT.pkl").is_file():
        print(f"Loading MANO from {mano_path}")
        mano_model, faces = load_mano_model(str(mano_path), device)
        decoded = decode_mano_and_cameras(preds, mano_model, args.img_size)
        save_dict["pred_verts"] = decoded["pred_verts"]
        save_dict["pred_joints"] = decoded["pred_joints"]
        save_dict["faces"] = faces
        if "pred_extrinsics" in decoded:
            save_dict["pred_extrinsics"] = decoded["pred_extrinsics"]
            save_dict["pred_intrinsics"] = decoded["pred_intrinsics"]
            overlays: list[np.ndarray] = []
            for view_idx in range(num_views):
                overlays.append(
                    overlay_hand_on_image(
                        images_rgb[view_idx],
                        decoded["pred_verts"],
                        faces,
                        decoded["pred_extrinsics"][view_idx],
                        decoded["pred_intrinsics"][view_idx],
                    )
                )
            overlay_mosaic = build_mosaic(overlays)
            mosaic_out = os.path.join(args.output_dir, "overlay_mosaic.jpg")
            cv2.imwrite(
                mosaic_out,
                cv2.cvtColor(overlay_mosaic, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            print(f"Saved {mosaic_out}")
    else:
        print(
            f"MANO not found at {mano_path} (expected MANO_RIGHT.pkl). "
            "Skipping mesh overlay; result.npz will still contain mano_params."
        )

    npz_path = os.path.join(args.output_dir, "result.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved {npz_path}")
    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
