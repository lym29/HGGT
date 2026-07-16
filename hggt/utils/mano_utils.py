"""MANO decoding and simple 2D mesh overlay helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from hggt.training.train_utils.projection import compute_mano_output
from hggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_mano_model(mano_model_path: str, device: str | torch.device):
    """Load a right-hand MANO model (smplx)."""
    import smplx

    mano_model = smplx.MANO(
        model_path=mano_model_path,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=True,
        create_transl=False,
    ).to(device)
    mano_model.eval()
    faces = np.array(mano_model.faces, dtype=np.int32)
    return mano_model, faces


def decode_mano_and_cameras(
    preds: dict[str, Any],
    mano_model,
    img_size: int,
) -> dict[str, Any]:
    """Decode MANO verts/joints and per-view cameras from HGGT predictions."""
    with torch.no_grad():
        verts, joints = compute_mano_output(
            mano_model,
            hand_pose=preds["mano_params"]["hand_pose"],
            shape=preds["mano_params"]["betas"],
            transl=preds["mano_params"]["transl"],
            is_right=True,
        )

    result: dict[str, Any] = {
        "pred_verts": verts[0].cpu().numpy(),
        "pred_joints": joints[0].cpu().numpy(),
        "mano_hand_pose": preds["mano_params"]["hand_pose"][0].cpu().numpy(),
        "mano_betas": preds["mano_params"]["betas"][0].cpu().numpy(),
        "mano_transl": preds["mano_params"]["transl"][0].cpu().numpy(),
    }

    if "pose_enc" in preds:
        extr_all, intr_all = pose_encoding_to_extri_intri(preds["pose_enc"], (img_size, img_size))
        result["pred_extrinsics"] = extr_all[0].cpu().numpy()  # (S, 3, 4)
        result["pred_intrinsics"] = intr_all[0].cpu().numpy()  # (S, 3, 3)

    return result


def project_points(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) world points to pixel (u, v, z)."""
    extr = extrinsic[:3, :] if extrinsic.shape[0] >= 4 else extrinsic
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    cam = (extr @ np.concatenate([points, ones], axis=1).T).T
    z = cam[:, 2]
    u = intrinsic[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + intrinsic[0, 2]
    v = intrinsic[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + intrinsic[1, 2]
    return np.stack([u, v, z], axis=1).astype(np.float32)


def overlay_mesh_wireframe(
    image: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int] = (0, 180, 255),
    alpha: float = 0.65,
) -> np.ndarray:
    """Draw a simple projected mesh wireframe on an RGB uint8 image."""
    pts = project_points(verts, extrinsic, intrinsic)
    h, w = image.shape[:2]
    canvas = image.copy()
    overlay = image.copy()

    for face in faces:
        tri = pts[face]
        if np.any(tri[:, 2] <= 0):
            continue
        poly = tri[:, :2].astype(np.int32)
        if np.any(poly[:, 0] < -50) or np.any(poly[:, 0] >= w + 50):
            continue
        if np.any(poly[:, 1] < -50) or np.any(poly[:, 1] >= h + 50):
            continue
        cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
