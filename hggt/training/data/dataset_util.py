# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Image geometry and MANO helpers used by the HGGT training datasets."""

import logging
import math
from typing import List

import cv2
import numpy as np
from PIL import Image

lanczos = Image.Resampling.LANCZOS
bicubic = Image.Resampling.BICUBIC

def crop_image_depth_and_intrinsic_by_pp(image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False, masks=None, keypoints_2d=None):
    """Crop around the principal point and translate pixel-space annotations."""
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)
    if original_size[0] < target_shape[0]:
        error_message = f'Height check failed: original height {original_size[0]} is less than target height {target_shape[0]}.'
        raise ValueError(f'{filepath}: {error_message}')
    if original_size[1] < target_shape[1]:
        error_message = f'Width check failed: original width {original_size[1]} is less than target width {target_shape[1]}.'
        raise ValueError(f'{filepath}: {error_message}')
    cx = intrinsic[1, 2]
    cy = intrinsic[0, 2]
    if strict:
        half_x = min(target_shape[0] / 2, cx)
        half_y = min(target_shape[1] / 2, cy)
    else:
        half_x = min(target_shape[0] / 2, cx, original_size[0] - cx)
        half_y = min(target_shape[1] / 2, cy, original_size[1] - cy)
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)
    if start_x < 0 or start_y < 0 or half_x < 1 or half_y < 1:
        raise ValueError(f"{filepath}: principal point does not permit a nonempty crop")
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)
    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        if depth_map.ndim == 2:
            depth_map = depth_map[start_x:end_x, start_y:end_y]
        elif depth_map.ndim == 3:
            depth_map = depth_map[start_x:end_x, start_y:end_y, :]
        else:
            raise ValueError(f'Unsupported depth_map dimensions: {depth_map.ndim}')
    cropped_masks = None
    if masks is not None:
        cropped_masks = []
        for mask in masks:
            cropped_mask = mask[start_x:end_x, start_y:end_y]
            cropped_masks.append(cropped_mask)
    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y
    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y
    if keypoints_2d is not None:
        keypoints_2d = keypoints_2d.copy()
        keypoints_2d[:, 1] = keypoints_2d[:, 1] - start_x
        keypoints_2d[:, 0] = keypoints_2d[:, 0] - start_y
    if strict:
        if (image.shape[:2] != target_shape).any():
            current_h, current_w = image.shape[:2]
            target_h, target_w = (target_shape[0], target_shape[1])
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(f'The cropped image is bigger than the target shape: cropped=({current_h},{current_w}), target=({target_h},{target_w}).')
            image = np.pad(image, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            if depth_map is not None:
                if depth_map.ndim == 2:
                    depth_map = np.pad(depth_map, pad_width=((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                elif depth_map.ndim == 3:
                    depth_map = np.pad(depth_map, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            if cropped_masks is not None:
                for i, mask in enumerate(cropped_masks):
                    cropped_masks[i] = np.pad(mask, pad_width=((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return (image, depth_map, intrinsic, track, keypoints_2d, cropped_masks)

def resize_image_depth_and_intrinsic(image, depth_map, intrinsic, target_shape, original_size, track=None, pixel_center=True, safe_bound=4, rescale_aug=True, masks=None, keypoints_2d=None, min_output_resolution=64, sanity_check_threshold=48):
    """Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,"""
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()
    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)
    if np.any(original_size <= 0):
        raise ValueError(f'Invalid original_size: {original_size}. Image may be corrupted.')
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    if np.any(input_resolution <= 0):
        raise ValueError(f'Invalid input image size: {input_resolution}. Image may be corrupted.')
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    if np.any(output_resolution <= 0) or max_resize_scale <= 0:
        logging.warning(f'Invalid resize scale {max_resize_scale} or output resolution {output_resolution}. Clamping to minimum {min_output_resolution}x{min_output_resolution}.')
        output_resolution = np.maximum(output_resolution, min_output_resolution)
    elif np.any(output_resolution < sanity_check_threshold):
        logging.warning(f'Output resolution suspiciously small: {output_resolution} (max_resize_scale={max_resize_scale}, original_size={original_size}). Clamping to minimum {min_output_resolution}x{min_output_resolution}.')
        output_resolution = np.maximum(output_resolution, min_output_resolution)
    else:
        output_resolution = np.maximum(output_resolution, min_output_resolution)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)
    if depth_map is not None:
        if depth_map.ndim == 3:
            num_layers = depth_map.shape[2]
            resized_layers = []
            for layer_idx in range(num_layers):
                layer_resized = cv2.resize(depth_map[:, :, layer_idx], output_resolution, fx=max_resize_scale, fy=max_resize_scale, interpolation=cv2.INTER_NEAREST)
                resized_layers.append(layer_resized)
            depth_map = np.stack(resized_layers, axis=2)
        else:
            depth_map = cv2.resize(depth_map, output_resolution, fx=max_resize_scale, fy=max_resize_scale, interpolation=cv2.INTER_NEAREST)
    resized_masks = None
    if masks is not None:
        resized_masks = []
        for mask in masks:
            resized_mask = cv2.resize(mask, output_resolution, fx=max_resize_scale, fy=max_resize_scale, interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized_mask)
    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)
    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5
    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale
    if track is not None:
        track = track * actual_resize_scale
    if keypoints_2d is not None:
        keypoints_2d = keypoints_2d.copy()
        keypoints_2d[:, :2] = keypoints_2d[:, :2] * actual_resize_scale
    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5
    if depth_map is not None:
        assert image.shape[:2] == depth_map.shape[:2], f'Image shape {image.shape[:2]} != depth shape {depth_map.shape[:2]}'
    return (image, depth_map, intrinsic, track, keypoints_2d, resized_masks)

def threshold_depth_map(depth_map: np.ndarray, max_percentile: float=99, min_percentile: float=1, max_depth: float=-1) -> np.ndarray:
    """Thresholds a depth map using percentile-based limits and optional maximum depth clamping."""
    if depth_map is None:
        return None
    depth_map = depth_map.astype(float, copy=True)
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0
    depth_max_thres = np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    depth_min_thres = np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0
    return depth_map

def init_mano_model(mano_model_path: str):
    """Initialize MANO model for precomputation."""
    from smplx import MANO
    mano_model = MANO(model_path=mano_model_path, use_pca=False, flat_hand_mean=False, create_transl=False, is_rhand=True)
    mano_model.eval()
    return mano_model

def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """Flip 2D or 3D keypoints."""
    joints = joints.copy()
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]
    return joints

def precompute_mano_outputs(mano_model, hand_pose: np.ndarray, shape: np.ndarray, transl: np.ndarray):
    """Precompute MANO vertices and 3D keypoints."""
    import torch
    with torch.no_grad():
        hand_pose_t = torch.from_numpy(hand_pose).unsqueeze(0).float()
        shape_t = torch.from_numpy(shape).unsqueeze(0).float()
        transl_t = torch.from_numpy(transl).unsqueeze(0).float()
        mano_output = mano_model(global_orient=hand_pose_t[:, :3], hand_pose=hand_pose_t[:, 3:], betas=shape_t)
        verts = (mano_output.vertices[0] + transl_t[0]).cpu().numpy()
        base_joints = (mano_output.joints[0] + transl_t[0]).cpu().numpy()
        fingertip_indices = [745, 317, 445, 556, 673]
        fingertip_joints = verts[fingertip_indices]
        keypoints_3d = np.concatenate([base_joints, fingertip_joints], axis=0)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        keypoints_3d = keypoints_3d[mano_to_openpose]
    return (verts.astype(np.float32), keypoints_3d.astype(np.float32))

def fliplr_mano_params(hand_pose: np.ndarray) -> np.ndarray:
    """Flip MANO hand_pose parameters when flipping the image for left hands."""
    hand_pose_flipped = hand_pose.copy()
    hand_pose_flipped[1::3] *= -1
    hand_pose_flipped[2::3] *= -1
    return hand_pose_flipped.astype(np.float32)
