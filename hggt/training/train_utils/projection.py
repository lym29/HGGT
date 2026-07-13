# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional

from hggt.utils.pose_enc import pose_encoding_to_extri_intri

# MANO joint indices for 5 fingertips (to extend 16 joints to 21)
# These vertex indices correspond to the tip of each finger
MANO_FINGERTIP_VERTEX_IDS = {
    'thumb': 745,
    'index': 317,
    'middle': 445,
    'ring': 556,
    'pinky': 673
}

# Reordering to match OpenPose format (21 joints)
MANO_TO_OPENPOSE_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def perspective_projection_with_intrinsics(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Computes the perspective projection of 3D points using camera extrinsics and intrinsics.
    
    Args:
        points (torch.Tensor): 3D points in world coordinates (B, N, 3)
        extrinsics (torch.Tensor): Camera extrinsics [R|t] (B, 3, 4)
        intrinsics (torch.Tensor): Camera intrinsics K (B, 3, 3)
        return_depth (bool): If True, also return camera-space Z values separately
        return_with_confidence (bool): If True, return points as (B, N, 3) with confidence as last dim
    
    Returns:
        torch.Tensor: 2D projected points with confidence (B, N, 3) where last dim is confidence
        OR (if return_depth=True)
        tuple: (2D projected points with confidence (B, N, 3), camera Z values (B, N))
    """
    # Extract R and t from extrinsics
    R = extrinsics[..., :3, :3]  # (B, 3, 3)
    t = extrinsics[..., :3, 3]   # (B, 3)
    
    # Transform points to camera coordinates: points_cam = R @ points + t
    points_cam = torch.einsum('bij,bnj->bni', R, points) + t.unsqueeze(1)  # (B, N, 3)
    
    # Project to image plane using intrinsics: points_2d = K @ points_cam
    points_2d_homog = torch.einsum('bij,bnj->bni', intrinsics, points_cam)  # (B, N, 3)
    
    # Normalize by depth (z-coordinate)
    z = points_2d_homog[..., 2:3]  # (B, N, 1)
    
    # Create confidence mask: points with z > 0 are valid (in front of camera)
    # Points with z <= 0 are invalid (behind or at camera) -> confidence = 0
    confidence = (z > 0).float()  # (B, N, 1)
    
    # For numerical stability in division, use a small epsilon where z <= 0
    # These invalid points will be masked out by confidence = 0
    z_safe = torch.where(z > 0, z, torch.ones_like(z) * 1e-2)
    
    
    points_2d = points_2d_homog[..., :2] / z_safe  # (B, N, 2)
    points_2d_with_conf = torch.cat([points_2d, confidence], dim=-1)  # (B, N, 3)
    
    return points_2d_with_conf, z.squeeze(-1)  # (B, N, 3), (B, N)


def get_camera_matrices_from_pose_enc(pose_enc: torch.Tensor, img_size):
    """
    Convert pose encoding to camera extrinsics and intrinsics using VGGT's native function.
    
    Args:
        pose_enc (torch.Tensor): Pose encoding (B, 9)
            [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
        img_size (tuple, list, or int): Image size (H, W) or single value if square
    
    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): (B, 3, 4) - Camera extrinsics [R|t]
            - intrinsics (torch.Tensor): (B, 3, 3) - Camera intrinsics K
    """
    # Handle img_size format
    if isinstance(img_size, (tuple, list)):
        image_size_hw = img_size  # (H, W)
    else:
        image_size_hw = (img_size, img_size)  # Square image
    
    # Add sequence dimension if needed: (B, 9) → (B, 1, 9)
    if pose_enc.dim() == 2:
        pose_enc = pose_enc.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Use VGGT's native function to convert pose encoding
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding=pose_enc,
        image_size_hw=image_size_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True
    )
    
    # Remove sequence dimension if we added it
    if squeeze_output:
        extrinsics = extrinsics.squeeze(1)  # (B, 3, 4)
        intrinsics = intrinsics.squeeze(1)  # (B, 3, 3)
    
    return extrinsics, intrinsics


def compute_mano_output(
    mano_model,
    hand_pose: torch.Tensor,
    shape: torch.Tensor,
    transl: torch.Tensor,
    is_right: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both hand mesh vertices and keypoints from MANO parameters in a single forward pass.
    
    Extends MANO's 16 base joints to 21 joints by adding 5 fingertip vertices,
    similar to HaMeR's implementation.
    
    Args:
        mano_model: MANO model instance
        hand_pose (torch.Tensor): Hand pose parameters (B, 48)
            First 3 are global_orient, remaining 45 are hand_pose
        shape (torch.Tensor): Shape parameters (B, 10)
        transl (torch.Tensor): Translation parameters (B, 3)
        is_right (bool): Whether it's a right hand. If False, mirrors X coordinates. Default: True.
    
    Returns:
        tuple: (vertices, keypoints)
            - vertices (torch.Tensor): 3D hand vertices (B, 778, 3)
            - keypoints (torch.Tensor): 3D hand keypoints/joints (B, 21, 3)
    """
    # Run MANO model once
    mano_output = mano_model(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:],
        betas=shape
    )
    
    # Get vertices and apply translation
    verts = mano_output.vertices + transl.unsqueeze(1)  # (B, 778, 3)
    
    # Get base joints (16 joints) and apply translation
    base_joints = mano_output.joints + transl.unsqueeze(1)  # (B, 16, 3)
    
    # Extract 5 fingertip vertices to extend to 21 joints (like HaMeR)
    fingertip_indices = torch.tensor(
        list(MANO_FINGERTIP_VERTEX_IDS.values()),
        dtype=torch.long,
        device=verts.device
    )  # [5]
    
    fingertip_joints = torch.index_select(verts, 1, fingertip_indices)  # (B, 5, 3)
    
    # Concatenate base joints (16) + fingertips (5) = 21 joints
    joints = torch.cat([base_joints, fingertip_joints], dim=1)  # (B, 21, 3)
    
    # Reorder to OpenPose format (optional, for compatibility)
    joint_map = torch.tensor(MANO_TO_OPENPOSE_ORDER, dtype=torch.long, device=joints.device)
    joints = joints[:, joint_map, :]  # (B, 21, 3)
    
    # Mirror X coordinates if left hand
    if not is_right:
        verts[:, :, 0] = -verts[:, :, 0]
        joints[:, :, 0] = -joints[:, :, 0]

    
    return verts, joints


def compute_hand_vertices(
    mano_model,
    hand_pose: torch.Tensor,
    shape: torch.Tensor,
    transl: torch.Tensor,
    is_right: bool = True
) -> torch.Tensor:
    """
    Compute 3D hand mesh vertices from MANO parameters.
    
    Note: If you need both vertices and keypoints, use compute_mano_output() instead
    to avoid running MANO model twice.
    
    Args:
        mano_model: MANO model instance
        hand_pose (torch.Tensor): Hand pose parameters (B, 48)
        shape (torch.Tensor): Shape parameters (B, 10)
        transl (torch.Tensor): Translation parameters (B, 3)
        is_right (bool): Whether it's a right hand. Default: True.
    
    Returns:
        torch.Tensor: 3D hand vertices (B, 778, 3)
    """
    verts, _ = compute_mano_output(mano_model, hand_pose, shape, transl, is_right)
    return verts


def compute_keypoints_from_mano(
    mano_model,
    hand_pose: torch.Tensor,
    shape: torch.Tensor,
    transl: torch.Tensor
) -> torch.Tensor:
    """
    Compute 3D hand keypoints/joints from MANO parameters.
    
    Note: If you need both vertices and keypoints, use compute_mano_output() instead
    to avoid running MANO model twice.
    
    Args:
        mano_model: MANO model instance
        hand_pose (torch.Tensor): Hand pose parameters (B, 48)
        shape (torch.Tensor): Shape parameters (B, 10)
        transl (torch.Tensor): Translation parameters (B, 3)
    
    Returns:
        torch.Tensor: 3D keypoints/joints (B, 21, 3)
    """
    _, joints = compute_mano_output(mano_model, hand_pose, shape, transl, is_right=True)
    return joints


def project_keypoints_to_2d(
    keypoints_3d: torch.Tensor,
    pred_pose_enc: torch.Tensor,
    img_size,
    gt_extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project 3D keypoints to 2D using VGGT's camera model.
    
    Args:
        keypoints_3d (torch.Tensor): 3D keypoints in world coordinates (B, N, 3)
        pose_enc (torch.Tensor): VGGT pose encoding (B, 9)
        img_size (tuple, list, or int): Image size (H, W) or single value if square
        gt_extrinsics (torch.Tensor, optional): Ground truth extrinsics (B, 3, 4). 
            If provided, will be used instead of extrinsics computed from pose_enc.
            Default: None
        return_depth (bool): If True, also return camera-space Z values separately
        return_with_confidence (bool): If True, return with confidence as last dimension (B, N, 3)
    
    Returns:
        torch.Tensor: 2D keypoints with confidence (B, N, 3) where last dim is confidence
        OR (if return_depth=True)
        tuple: (2D keypoints with confidence (B, N, 3), camera Z values (B, N))
    """
    # Get camera matrices from pose encoding using VGGT's native function
    extrinsics, intrinsics = get_camera_matrices_from_pose_enc(pred_pose_enc, img_size)
    
    # Use gt_extrinsics if provided, otherwise use computed extrinsics
    if gt_extrinsics is not None:
        extrinsics = gt_extrinsics
    
    # Project using the camera matrices
    result = perspective_projection_with_intrinsics(
        points=keypoints_3d,
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )
    
    return result

