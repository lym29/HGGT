# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for HGGT evaluation / training-style preprocessing.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            img_per_seq (int, optional): Number of images per sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
        masks=None,
        keypoints_2d=None,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.
            masks (list, optional): Optional list of mask arrays to be processed alongside the image.
            keypoints_2d (numpy.ndarray, optional): Optional 2D keypoints array to be processed alongside the image.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information,
                processed_masks (list, optional): Processed masks if provided,
                keypoints_2d (numpy.ndarray): Processed 2D keypoints if provided
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        if depth_map is not None:
            depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)
        if masks is not None:
            masks = [np.copy(mask) for mask in masks]

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track, keypoints_2d, cropped_masks = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath, masks=masks,
            keypoints_2d=keypoints_2d,
        )
        
        masks = cropped_masks

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intri_opencv, track, keypoints_2d, resized_masks = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug,
                masks=masks,
                keypoints_2d=keypoints_2d,
            )
            masks = resized_masks
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track, keypoints_2d, final_cropped_masks = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True, masks=masks,
            keypoints_2d=keypoints_2d,
        )
        
        masks = final_cropped_masks

        # Convert depth to world and camera coordinates
        # If depth_map is multi-layer [H, W, L], only convert the first layer [H, W]
        
        if depth_map is not None and depth_map.ndim < 3:
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
            )
        else:
            world_coords_points, cam_coords_points, point_mask = None, None, None

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
            masks,
            keypoints_2d,  # 2D data transformed (resize/crop/rotate)
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: add the function to sample the ids by pose similarity ranking.

        Sample a set of IDs from a sequence close to a given start index.

        You can specify the range either as a ratio of the number of input IDs
        or as a fixed integer window.


        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands
                around the start index. Default is 2.0 if neither expand_ratio nor
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids
