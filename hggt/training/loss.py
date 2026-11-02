# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Camera and MANO supervision for HGGT's iterative prediction heads."""

from collections.abc import Mapping

import torch
from torch import nn

from hggt.training.train_utils.projection import project_keypoints_to_2d
from hggt.utils.pose_enc import extri_intri_to_pose_encoding


def _clip_finite(value: torch.Tensor) -> torch.Tensor:
    """Retain the recipe's finite outlier clipping without hiding NaN or Inf."""
    return torch.where(torch.isfinite(value), value.clamp(-100.0, 100.0), value)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _multiview_mask(batch, reference):
    mask = batch.get("is_multiview")
    if mask is None:
        return torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
    return mask.to(device=reference.device, dtype=torch.bool).reshape(reference.shape[0])


class HandCameraLoss(nn.Module):
    """Compute the released camera/hand objective from decoded HGGT outputs.

    Call ``HandGeometry.decode_predictions`` before this loss. Parameter and
    root-relative 3D losses supervise every MANO stage. The hand reprojection
    term uses only the final stage; a separate multiview term supervises every
    camera stage using detached final-stage hand joints. Stage-weighted sums
    are divided by the number of stages, not the sum of the stage weights.
    """

    def __init__(self, camera: Mapping | None = None, hand: Mapping | None = None):
        super().__init__()
        self.camera = {"weight": 5.0, "loss_type": "l1", **dict(camera or {})}
        self.hand = {
            "weight": 1.0, "loss_type": "l2", "gamma": 0.8,
            "weight_pose": 0.1, "weight_beta": 0.05,
            "weight_keypoints_3d": 5.0, "weight_keypoints_2d": 1.0,
            "weight_multiview_keypoints_2d": 1.0, "weight_negative_depth": 1.0,
            **dict(hand or {}),
        }

    def forward(self, predictions: dict, batch: dict) -> dict[str, torch.Tensor]:
        camera_options = {k: v for k, v in self.camera.items() if k != "weight"}
        hand_options = {k: v for k, v in self.hand.items() if k != "weight"}
        result = compute_camera_loss(predictions, batch, **camera_options)
        result.update(compute_hand_loss(predictions, batch, **hand_options))
        result["objective"] = (
            self.camera["weight"] * result["loss_camera"]
            + self.hand["weight"] * result["loss_hand"]
        )
        return result


def compute_camera_loss(
    predictions, batch, loss_type="l1", gamma=0.6,
    weight_trans=1.0, weight_rot=1.0, weight_focal=0.5,
    pose_encoding_type="absT_quaR_FoV",
):
    stages = predictions["pose_enc_list"]
    if not stages:
        raise ValueError("Camera supervision requires at least one prediction stage")
    if loss_type not in ("l1", "l2"):
        raise ValueError(f"Unknown loss type: {loss_type}")
    zero = sum((_zero(stage) for stage in stages), _zero(stages[0]))
    mask = _multiview_mask(batch, stages[0])
    terms = {"loss_T": zero, "loss_R": zero, "loss_FL": zero}
    if mask.any():
        intrinsics = batch.get("intrinsics")
        gt = extri_intri_to_pose_encoding(
            batch["extrinsics"][mask],
            None if intrinsics is None else intrinsics[mask],
            batch["images"].shape[-2:], pose_encoding_type=pose_encoding_type,
        )
        components = [("loss_T", slice(0, 3)), ("loss_R", slice(3, 7))]
        if intrinsics is not None:
            components.append(("loss_FL", slice(7, 9)))
        for index, stage in enumerate(stages):
            stage_weight = gamma ** (len(stages) - index - 1) / len(stages)
            for name, columns in components:
                difference = stage[mask][..., columns] - gt[..., columns]
                error = difference.abs() if loss_type == "l1" else difference.norm(dim=-1)
                terms[name] = terms[name] + _clip_finite(error).mean() * stage_weight
        terms = {name: _clip_finite(value) for name, value in terms.items()}
    terms["loss_camera"] = (
        weight_trans * terms["loss_T"] + weight_rot * terms["loss_R"]
        + weight_focal * terms["loss_FL"]
    )
    return terms


def _keypoints_2d_loss(predicted, target, image_hw, loss_type):
    """Confidence-weighted reprojection, normalized by valid joints per sample."""
    height, width = image_hw
    size = predicted.new_tensor([width, height])
    xy = predicted[..., :2].clamp(min=-0.5 * size, max=1.5 * size)
    confidence = target[..., 2:3] * predicted[..., 2:3]
    difference = _clip_finite(confidence * (xy / size - target[..., :2] / size))
    if loss_type == "l2":
        error = difference.norm(dim=-1).sum(dim=(1, 2))
        count = confidence.squeeze(-1).sum(dim=(1, 2)).clamp(min=1.0)
    elif loss_type == "l1":
        error = difference.abs().sum(dim=(1, 2, 3))
        count = (2 * confidence.sum(dim=(1, 2, 3))).clamp(min=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return _clip_finite((error / count).mean())


def hand_loss_single(predicted, batch, loss_type="l2"):
    if loss_type not in ("l1", "l2"):
        raise ValueError(f"Unknown loss type: {loss_type}")
    zero = _zero(predicted["hand_pose"]) + _zero(predicted["keypoints_3d"])
    terms = {name: zero for name in (
        "loss_pose", "loss_beta", "loss_keypoints_3d",
        "loss_keypoints_2d", "loss_negative_depth",
    )}
    mask = (batch["has_hand_pose"].reshape(-1) > 0.5).to(predicted["hand_pose"])
    for key, name in (("hand_pose", "loss_pose"), ("betas", "loss_beta")):
        difference = _clip_finite(mask[:, None] * (predicted[key] - batch[key]))
        error = difference.norm(dim=-1) if loss_type == "l2" else difference.abs()
        # The mean includes samples without parameter GT, matching the recipe.
        terms[name] = _clip_finite(error.mean())
    joints = predicted["keypoints_3d"]
    target = batch["keypoints_3d"]
    relative_prediction = joints - joints[:, :1]
    relative_target = target[..., :3] - target[:, :1, :3]
    difference = _clip_finite(target[..., 3:4] * (relative_prediction - relative_target))
    error = difference.norm(dim=-1) if loss_type == "l2" else difference.abs()
    terms["loss_keypoints_3d"] = _clip_finite(error.mean())
    if "pred_keypoints_2d" in predicted and "keypoints_2d" in batch:
        terms["loss_keypoints_2d"] = _keypoints_2d_loss(
            predicted["pred_keypoints_2d"], batch["keypoints_2d"],
            batch["images"].shape[-2:], loss_type,
        )
        depth = _clip_finite(predicted["pred_keypoints_z_cam"])
        terms["loss_negative_depth"] = _clip_finite(torch.relu(-depth).square().mean())
    return terms


def compute_multiview_keypoints_loss(
    pred_keypoints_3d, pred_pose_enc, gt_keypoints_2d, img_size, loss_type="l2",
):
    batch_size, views = pred_pose_enc.shape[:2]
    if views <= 1:
        return _zero(pred_pose_enc)
    joints = pred_keypoints_3d.shape[1]
    expanded = pred_keypoints_3d[:, None].expand(-1, views, -1, -1)
    projected, _ = project_keypoints_to_2d(
        expanded.reshape(batch_size * views, joints, 3),
        pred_pose_enc.reshape(batch_size * views, -1), img_size,
    )
    return _keypoints_2d_loss(
        projected.reshape(batch_size, views, joints, 3), gt_keypoints_2d,
        img_size, loss_type,
    )


def compute_hand_loss(
    predictions, batch, loss_type="l2", gamma=0.8,
    weight_pose=0.1, weight_beta=0.05, weight_keypoints_3d=5.0,
    weight_keypoints_2d=1.0, weight_multiview_keypoints_2d=1.0,
    weight_negative_depth=1.0,
):
    stages = predictions["mano_params_list"]
    cameras = predictions["pose_enc_list"]
    if not stages or len(cameras) != len(stages):
        raise ValueError("Hand and camera heads must produce the same nonzero number of stages")
    weights = {
        "loss_pose": weight_pose, "loss_beta": weight_beta,
        "loss_keypoints_3d": weight_keypoints_3d,
        "loss_keypoints_2d": weight_keypoints_2d,
        "loss_multiview_keypoints_2d": weight_multiview_keypoints_2d,
        "loss_negative_depth": weight_negative_depth,
    }
    zero = _zero(stages[0]["keypoints_3d"])
    terms = {name: zero for name in weights}
    for index, stage in enumerate(stages):
        factor = gamma ** (len(stages) - index - 1) / len(stages)
        for name, value in hand_loss_single(stage, batch, loss_type).items():
            terms[name] = terms[name] + factor * value
    mask = _multiview_mask(batch, cameras[0])
    if "keypoints_2d" in batch and mask.any():
        # Camera consistency must not update the hand prediction through 3D joints.
        fixed_joints = stages[-1]["keypoints_3d"].detach()[mask]
        for index, camera in enumerate(cameras):
            factor = gamma ** (len(stages) - index - 1) / len(stages)
            terms["loss_multiview_keypoints_2d"] = terms["loss_multiview_keypoints_2d"] + factor * compute_multiview_keypoints_loss(
                fixed_joints, camera[mask], batch["keypoints_2d"][mask],
                batch["images"].shape[-2:], loss_type,
            )
    terms = {name: _clip_finite(value) for name, value in terms.items()}
    terms["loss_hand"] = _clip_finite(sum(weights[name] * value for name, value in terms.items()))
    return terms
