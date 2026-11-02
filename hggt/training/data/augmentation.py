# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Photometric augmentation shared by the training datasets."""

from torchvision import transforms


def get_image_augmentation(color_jitter=None, gray_scale=False, gau_blur=False):
    operations = []
    if color_jitter:
        options = {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.1}
        options.update({key: value for key, value in color_jitter.items() if key != "p"})
        operations.append(transforms.RandomApply(
            [transforms.ColorJitter(**options)], p=color_jitter.get("p", 0.9),
        ))
    if gray_scale:
        operations.append(transforms.RandomGrayscale(p=0.05))
    if gau_blur:
        operations.append(transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05))
    return transforms.Compose(operations) if operations else None
