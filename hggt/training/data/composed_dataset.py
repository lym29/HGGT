# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Combine real datasets by their effective lengths and prepare HGGT tensors."""

import bisect
import random

from hydra.utils import instantiate
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from hggt.training.data.augmentation import get_image_augmentation


class TupleConcatDataset(ConcatDataset):
    def __init__(self, datasets, common_config):
        super().__init__(datasets)
        if not self.datasets or any(len(dataset) == 0 for dataset in self.datasets):
            raise ValueError("All composed datasets must contain samples")
        self.inside_random = common_config.inside_random

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise ValueError("Dataset index must be (index, view_count, aspect_ratio)")
        sample_index, views, aspect = index
        if self.inside_random:
            sample_index = random.randrange(len(self))
        if sample_index < 0:
            sample_index += len(self)
        if not 0 <= sample_index < len(self):
            raise IndexError(sample_index)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, sample_index)
        offset = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        dataset = self.datasets[dataset_index]
        result = dataset[(sample_index - offset, views, aspect)]
        result["is_multiview"] = dataset.is_multiview
        return result


class ComposedDataset(Dataset):
    def __init__(self, dataset_configs, common_config):
        self.base_dataset = TupleConcatDataset(
            [instantiate(config, common_conf=common_config) for config in dataset_configs], common_config,
        )
        self.config = common_config
        self.image_aug = get_image_augmentation(
            common_config.augs.color_jitter, common_config.augs.gray_scale, common_config.augs.gau_blur,
        )

    def __len__(self):
        return len(self.base_dataset)

    def set_epoch(self, epoch):
        for dataset in self.base_dataset.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __getitem__(self, index):
        if self.config.fix_img_num > 0:
            index = (index[0], self.config.fix_img_num, self.config.fix_aspect_ratio)
        raw = self.base_dataset[index]
        images = torch.from_numpy(np.stack(raw["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255)
        if self.config.training and self.image_aug:
            if self.config.augs.cojitter and random.random() > self.config.augs.cojitter_ratio:
                images = self.image_aug(images)
            else:
                images = torch.stack([self.image_aug(image) for image in images])
        sample = {"seq_name": raw["seq_name"], "images": images.contiguous(),
                  "ids": torch.as_tensor(raw["ids"], dtype=torch.long),
                  "num_views": torch.tensor(len(images)),
                  "is_multiview": torch.tensor(raw["is_multiview"]),
                  "has_intrinsics": torch.tensor(raw["has_intrinsics"]),
                  "has_hand_verts": torch.tensor(raw["has_hand_verts"])}
        for key in ("extrinsics", "intrinsics", "has_hand_pose", "has_betas", "hand_pose",
                    "betas", "keypoints_3d", "keypoints_2d", "hand_verts"):
            sample[key] = torch.from_numpy(np.asarray(raw[key], dtype=np.float32))
        sample["point_masks"] = torch.from_numpy(np.asarray(raw["point_masks"], dtype=bool))
        if "depths" in raw and self.config.load_depth:
            sample["depths"] = torch.from_numpy(np.asarray(raw["depths"], dtype=np.float32))
        return sample
