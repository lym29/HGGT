# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Shared principal-point crop and resize operations for HGGT training."""

import numpy as np
from torch.utils.data import Dataset

from hggt.training.data.dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    resize_image_depth_and_intrinsic,
)


class BaseDataset(Dataset):
    def __init__(self, common_conf):
        self.training = common_conf.training
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        if self.img_size <= 0 or self.img_size % self.patch_size:
            raise ValueError("img_size must be a positive multiple of patch_size")

    def __getitem__(self, index):
        seq_index, img_per_seq, aspect_ratio = index
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_data(self, **kwargs):
        raise NotImplementedError

    def get_target_shape(self, aspect_ratio):
        height = int(self.img_size * aspect_ratio) // self.patch_size * self.patch_size
        if height < self.patch_size:
            raise ValueError(f"Aspect ratio {aspect_ratio} gives an empty image")
        return np.array([height, self.img_size])

    def process_one_image(self, image, depth_map, extrinsic, intrinsic,
                          target_shape, *, masks=None, keypoints_2d=None, filepath=None):
        image = np.array(image, copy=True)
        original_size = np.array(image.shape[:2])
        if image.ndim != 3 or image.shape[2] != 3 or np.any(original_size <= 0):
            raise ValueError(f"Invalid RGB image: {filepath}")
        intrinsic = np.array(intrinsic, dtype=np.float32, copy=True)
        extrinsic = np.array(extrinsic, dtype=np.float32, copy=True)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"Invalid camera intrinsics: {filepath}")
        if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
            raise ValueError(f"Invalid camera extrinsics: {filepath}")
        depth_map = None if depth_map is None else np.array(depth_map, copy=True)
        masks = None if masks is None else [np.array(mask, copy=True) for mask in masks]
        aug_size = original_size
        if self.training and self.aug_scale:
            scale = np.minimum(np.random.uniform(*self.aug_scale, size=2), 1.0)
            aug_size = np.maximum(1, (original_size * scale).astype(np.int32))
        image, depth_map, intrinsic, _, keypoints_2d, masks = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsic, aug_size, filepath=filepath,
            masks=masks, keypoints_2d=keypoints_2d,
        )
        if self.rescale:
            image, depth_map, intrinsic, _, keypoints_2d, masks = resize_image_depth_and_intrinsic(
                image, depth_map, intrinsic, target_shape, np.array(image.shape[:2]),
                safe_bound=4, rescale_aug=self.rescale_aug, masks=masks, keypoints_2d=keypoints_2d,
            )
        image, depth_map, intrinsic, _, keypoints_2d, masks = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsic, target_shape, filepath=filepath,
            strict=True, masks=masks, keypoints_2d=keypoints_2d,
        )
        return image, depth_map, extrinsic, intrinsic, masks, keypoints_2d
