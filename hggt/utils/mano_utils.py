"""MANO decoding and pyrender mesh overlay helpers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from hggt.training.train_utils.projection import compute_mano_output
from hggt.utils.pose_enc import pose_encoding_to_extri_intri

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


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


def create_raymond_lights():
    """Create raymond light nodes for a pyrender scene."""
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp], dtype=np.float64)
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0], dtype=np.float64)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


def render_hand_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    image: np.ndarray,
    extrinsic: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Render a solid hand mesh onto an image (same style as HOIrecon ``demo_hand_poem``).

    Args:
        vertices: (778, 3) mesh in the reference / first-camera frame used by HGGT.
        faces: (F, 3) MANO faces.
        image: (H, W, 3) RGB float in ``[0, 1]`` or uint8.
        extrinsic: (3, 4) or (4, 4) world-to-camera.
        fx, fy, cx, cy: Intrinsics in pixel units.

    Returns:
        (H, W, 3) RGB uint8 overlay.
    """
    import pyrender
    import trimesh

    if image.dtype == np.uint8:
        image_f = image.astype(np.float32) / 255.0
    else:
        image_f = image.astype(np.float32)
        if image_f.max() > 1.0:
            image_f = image_f / 255.0

    img_h, img_w = image_f.shape[:2]
    extr = extrinsic[:3, :] if extrinsic.shape[0] >= 4 else extrinsic

    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h,
        point_size=1.0,
    )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(1.0, 0.9, 0.8, 1.0),
    )

    rotation = extr[:, :3]
    translation = extr[:, 3]
    vertices_cam = (vertices @ rotation.T) + translation
    vertices_gl = vertices_cam.copy()
    vertices_gl[:, 1] *= -1
    vertices_gl[:, 2] *= -1

    mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices_gl, faces.copy()),
        material=material,
    )
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, "mesh")
    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=img_h - cy,
        zfar=1e12,
    )
    scene.add(camera, pose=np.eye(4))
    for node in create_raymond_lights():
        scene.add_node(node)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    color = color.astype(np.float32) / 255.0
    valid_mask = color[:, :, -1:] > 0
    composited = color[:, :, :3] * valid_mask + (1.0 - valid_mask) * image_f
    return (composited * 255.0).clip(0, 255).astype(np.uint8)


def overlay_hand_on_image(
    image: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: overlay mesh using a 3x3 intrinsic matrix."""
    return render_hand_mesh(
        verts,
        faces,
        image,
        extrinsic=extrinsic,
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
    )
