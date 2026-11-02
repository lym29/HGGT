# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""First-camera coordinate normalization without changing metric scale."""

import torch

from hggt.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle


def _transform_points(points, rotation, translation):
    batch_size = rotation.shape[0]
    transformed = torch.bmm(points.reshape(batch_size, -1, 3), rotation.transpose(-1, -2))
    return (transformed + translation[:, None]).reshape_as(points)


def transform_hand_batch(hand_pose, hand_transl, extri_R, extri_t, mano_model):
    """Transform MANO pose/translation, correcting for MANO's rotation center.

    The zero-shape vertex reference is retained from the training recipe.
    This helper is for callers that also need a transformed MANO translation;
    training supervises root-relative joints and does not use this translation.
    """
    rotation = extri_R @ axis_angle_to_matrix(hand_pose[:, :3])
    pose = torch.cat((matrix_to_axis_angle(rotation), hand_pose[:, 3:]), dim=-1)
    shape = hand_pose.new_zeros((hand_pose.shape[0], 10))
    original = mano_model(global_orient=hand_pose[:, :3], hand_pose=hand_pose[:, 3:], betas=shape)
    transformed = mano_model(global_orient=pose[:, :3], hand_pose=pose[:, 3:], betas=shape)
    vertex = original.vertices[:, :1] + hand_transl[:, None]
    target_vertex = _transform_points(vertex, extri_R, extri_t)[:, 0]
    return pose, target_vertex - transformed.vertices[:, 0]


def normalize_cam_worldpoints_hand_batch(
    extrinsics, world_points=None, hand_pose=None, transl=None,
    mano_model=None, hand_verts=None, hand_keypoints_3d=None,
):
    """Express world-to-camera matrices and geometry in the first camera frame.

    ``extrinsics`` is [B,S,3,4], pose is [B,48], and points end in XYZ.
    No scale normalization is applied. To transform MANO translation as well,
    supply both ``transl`` and ``mano_model``. Device and dtype are preserved.
    """
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("extrinsics must have shape [B,S,3,4]")
    first_rotation = extrinsics[:, 0, :3, :3]
    first_translation = extrinsics[:, 0, :3, 3]
    rotation = extrinsics[..., :3, :3] @ first_rotation[:, None].transpose(-1, -2)
    translation = extrinsics[..., :3, 3:] - rotation @ first_translation[:, None, :, None]
    new_extrinsics = torch.cat((rotation, translation), dim=-1)
    new_pose = new_translation = None
    if hand_pose is not None:
        global_rotation = first_rotation @ axis_angle_to_matrix(hand_pose[:, :3])
        new_pose = torch.cat((matrix_to_axis_angle(global_rotation), hand_pose[:, 3:]), dim=-1)
        if transl is not None:
            if mano_model is None:
                raise ValueError("Transforming MANO translation requires mano_model")
            new_pose, new_translation = transform_hand_batch(
                hand_pose, transl, first_rotation, first_translation, mano_model,
            )
    transform = lambda points: None if points is None else _transform_points(points, first_rotation, first_translation)
    return (
        new_extrinsics, transform(world_points), new_pose, new_translation,
        transform(hand_verts), transform(hand_keypoints_3d),
    )
