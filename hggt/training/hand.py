# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Training geometry for the public HGGT MANO and camera heads."""

from pathlib import Path

import torch
from torch import nn

from hggt.training.train_utils.normalization import normalize_cam_worldpoints_hand_batch
from hggt.training.train_utils.projection import compute_mano_output, project_keypoints_to_2d


class HandGeometry(nn.Module):
    """Normalize targets and differentiably decode HGGT's MANO predictions.

    MANO uses a right-hand model, full axis-angle pose (3 global + 45 local),
    ten shape coefficients, and ``flat_hand_mean=False``. Its learned hand
    mean is added by smplx; HGGT predicts the pose offsets expected by smplx.
    A caller-supplied MANO-compatible module permits testing without licensed
    assets. The model's own parameters are frozen, not its input gradients.
    """

    def __init__(self, mano_model_path=None, device="cpu", *, mano_model=None):
        super().__init__()
        if mano_model is None:
            if mano_model_path is None or not Path(mano_model_path).exists():
                raise FileNotFoundError("Provide the directory containing licensed MANO model files")
            from smplx import MANO
            mano_model = MANO(
                model_path=str(mano_model_path), use_pca=False,
                flat_hand_mean=False, is_rhand=True, create_transl=False,
            )
        self.mano_model = mano_model.to(device).eval()
        self.mano_model.requires_grad_(False)

    @torch.no_grad()
    def normalize_batch(self, batch: dict) -> dict:
        """Return normalized targets; leave the input mapping/tensors untouched.

        The training recipe transforms the MANO global orientation and the
        annotated XYZ joints. It does not supervise GT MANO translation or
        transform 2D annotations, which already include image augmentation.
        """
        normalized = dict(batch)
        extrinsics, _, pose, _, _, xyz = normalize_cam_worldpoints_hand_batch(
            extrinsics=batch["extrinsics"], hand_pose=batch["hand_pose"],
            hand_keypoints_3d=batch["keypoints_3d"][..., :3],
        )
        normalized["extrinsics"] = extrinsics
        normalized["hand_pose"] = pose
        normalized["keypoints_3d"] = torch.cat((xyz, batch["keypoints_3d"][..., 3:]), dim=-1)
        return normalized

    def decode_predictions(self, predictions: dict, batch: dict) -> dict:
        """Return stage dictionaries with decoded joints and final reprojection.

        Final-stage hand reprojection uses GT extrinsics and detached predicted
        intrinsics. The loss separately projects detached final hand joints
        through every predicted camera, retaining camera-head gradients there.
        """
        mano_stages = predictions["mano_params_list"]
        camera_stages = predictions["pose_enc_list"]
        if not mano_stages or len(mano_stages) != len(camera_stages):
            raise ValueError("Hand and camera heads must produce the same nonzero number of stages")
        decoded = dict(predictions)
        stages = [dict(stage) for stage in mano_stages]
        batch_size = stages[0]["hand_pose"].shape[0]
        vertices, joints = compute_mano_output(
            self.mano_model,
            hand_pose=torch.cat([stage["hand_pose"] for stage in stages]),
            shape=torch.cat([stage["betas"] for stage in stages]),
            transl=torch.cat([stage["transl"] for stage in stages]),
        )
        for index, stage in enumerate(stages):
            selection = slice(index * batch_size, (index + 1) * batch_size)
            stage["hand_verts"] = vertices[selection]
            stage["keypoints_3d"] = joints[selection]
        camera = camera_stages[-1]
        views = camera.shape[1]
        count = joints.shape[1]
        expanded = stages[-1]["keypoints_3d"][:, None].expand(-1, views, -1, -1)
        extrinsics = batch.get("extrinsics")
        projected, depth = project_keypoints_to_2d(
            expanded.reshape(batch_size * views, count, 3),
            camera.detach().reshape(batch_size * views, -1),
            batch["images"].shape[-2:],
            gt_extrinsics=None if extrinsics is None else extrinsics.reshape(batch_size * views, 3, 4),
        )
        stages[-1]["pred_keypoints_2d"] = projected.reshape(batch_size, views, count, 3)
        stages[-1]["pred_keypoints_z_cam"] = depth.reshape(batch_size, views, count)
        decoded["mano_params_list"] = stages
        decoded["mano_params"] = stages[-1]
        return decoded
