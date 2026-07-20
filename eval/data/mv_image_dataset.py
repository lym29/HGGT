"""
Multi-view image (mv-image) WebDataset loader for HGGT evaluation.

Loads multi-view hand WebDataset tars (HO3D / DexYCB / Arctic / Interhand /
Oakink / Freihand). Only hand-related fields are used (joints_3d, verts_3d,
camera parameters).

Each sample contains one or more camera views of a hand. This dataset:
1. Streams data from WebDataset tar files
2. Crops each view around the hand region using bbox annotations
3. Processes through BaseDataset's pipeline (resize; no training augmentation in eval)
4. Outputs multi-view batches compatible with HGGT

Author: Yumeng Liu (lym29@connect.hku.hk)
"""

import glob
import logging
import os
import random

import cv2
import numpy as np
from PIL import Image as PILImage

from .base_dataset import BaseDataset

try:
    import webdataset as wds
except ImportError:
    wds = None

# Datasets that require extrinsic inversion
INV_EXTR_DATASETS = ['Interhand', 'Arctic', 'Oakink', 'Oakink2']

# Datasets who need to subtract MANO hand_mean_pose
# so that the pose matches MANO's expected convention (deviation from mean).
SUBSET_MEAN_POSE_DATASETS = ['HO3D', 'DexYCB', 'Freihand']


def _get_affine_trans_no_rot(center, scale, res):
    """Compute affine transform for bbox crop without rotation.

    Creates a 3x3 affine matrix that crops a square region of size ``scale``
    centered at ``center`` and resizes it to resolution ``res``.

    Affine crop without rotation (hand-centric square crop).

    Args:
        center: Crop center (2,) in pixel coordinates.
        scale: Side length of the crop region (scalar).
        res: Output resolution (width, height) tuple.

    Returns:
        3x3 affine transform matrix (np.ndarray).
    """
    affinet = np.zeros((3, 3))
    scale_ratio = float(res[0]) / float(res[1])
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale * scale_ratio
    affinet[0, 2] = res[0] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[1] * (-float(center[1]) / scale * scale_ratio + 0.5)
    affinet[2, 2] = 1
    return affinet


class MVImageDataset(BaseDataset):
    """
    Multi-view image WebDataset wrapper for HGGT evaluation.

    Streams multi-view tar files and converts hand data to HGGT's
    training format.  Only hand-related data is loaded:

    * 3D hand joints  (``joints_3d``)
    * 3D hand vertices (``verts_3d``)
    * Camera intrinsics & extrinsics
    * Per-view images with hand-centric bbox crop

    Object depth is **not** loaded.

    The "world" coordinate frame equals the master camera frame (first
    selected view).  ``hand_verts`` and ``keypoints_3d`` are expressed in this
    frame, and per-view extrinsics are master-relative.

    Directory structure expected (WebDataset tar files):
        /path/to/mv_image_data/
        ├── HO3D_mv_train/
        │   ├── HO3D_mv_train-000000.tar
        │   ├── HO3D_mv_train-000001.tar
        │   └── ...
        ├── DexYCB_mv/
        │   └── ...
        └── ...
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        data_root: str = None,
        url_pattern: str = None,
        epoch_size: int = None,
        shuffle: int = 1000,
        random_n_views: bool = True,
        view_range: tuple = (2, 8),
        max_samples: int = -1,
        bbox_crop_size: int = None,
        mano_model_path: str = None,
    ):
        """
        Initialize MVImageDataset.

        Args:
            common_conf: Configuration object with common settings
                (img_size, patch_size, augs, rescale, etc.).
            split: Dataset split ('train' or 'test').
            data_root: Root directory containing multi-view image WebDataset tar files.
            url_pattern: URL pattern for tar files relative to *data_root*,
                e.g. ``"HO3D_mv_train/HO3D_mv_train-{000000..000010}.tar"``.
            epoch_size: Number of samples per epoch.  When set, the
                WebDataset iterator will cycle after this many samples.
            shuffle: Shuffle buffer size (0 = no shuffle).
            random_n_views: If True, randomly shuffle cameras and select a
                random number of views per sample.
            view_range: ``(min_views, max_views)`` for random view selection.
                Only used when *random_n_views* is True.
            max_samples: Hard cap on dataset length (-1 = use epoch_size).
            bbox_crop_size: Resolution for the hand bbox crop.  Defaults to
                ``common_conf.img_size`` (518).
            mano_model_path: Path to MANO model dir (containing MANO_RIGHT.pkl).
                If set and dataset is HO3D/DexYCB/Freihand, hand_pose[3:48] is
                adjusted by subtracting hands_mean so MANO shape matches GT verts.
        """
        super().__init__(common_conf=common_conf)

        self.training = common_conf.training
        self.data_root = data_root
        self.url_pattern = url_pattern
        self.epoch_size = epoch_size
        self.shuffle_buffer = shuffle
        self.random_n_views = random_n_views
        self.view_range = view_range
        self.split = split

        # Size for hand bbox crop (before process_one_image resize)
        self.bbox_crop_size = bbox_crop_size or self.img_size

        if data_root is None:
            raise ValueError("data_root must be specified")
        if url_pattern is None:
            raise ValueError("url_pattern must be specified")

        if self.random_n_views:
            assert self.view_range is not None and self.view_range[0] >= 1, \
                "view_range must be specified with min >= 1 when random_n_views=True"

        # Detect dataset name for inv_extr handling
        # e.g. "Arctic_mv/Arctic_mv_val_p1-{...}.tar" -> "Arctic"
        self.dataset_name = url_pattern.split("/")[-1].split("_")[0]
        self.inv_extr = self.dataset_name in INV_EXTR_DATASETS

        # Load MANO hand mean pose for HO3D/DexYCB/Freihand (pose stored as absolute)
        self.hand_mean_pose = None
        if mano_model_path and self.dataset_name in SUBSET_MEAN_POSE_DATASETS:
            pkl_path = os.path.join(mano_model_path, "MANO_RIGHT.pkl")
            if os.path.isfile(pkl_path):
                mano_data = np.load(pkl_path, allow_pickle=True, encoding='latin1')
                self.hand_mean_pose = np.array(mano_data['hands_mean'], dtype=np.float32).reshape(-1)[:45]
                logging.info(f"  hand_mean_pose loaded for {self.dataset_name} (subtract from hand_pose[3:48])")
            else:
                logging.warning(f"  MANO_RIGHT.pkl not found at {pkl_path}, skipping hand_mean_pose")

        # Create streaming WebDataset
        self.webdataset = self._create_streaming_dataset()
        self._iterator = None
        self._closed = False

        # Dataset length
        if max_samples > 0:
            self.len_data = max_samples
        elif epoch_size is not None:
            self.len_data = epoch_size
        else:
            self.len_data = 50000

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: {self.dataset_name} multi-view image dataset")
        logging.info(f"  Data root: {data_root}")
        logging.info(f"  URL pattern: {url_pattern}")
        logging.info(f"  inv_extr: {self.inv_extr}")
        logging.info(f"  Dataset length: {self.len_data}")
        logging.info(f"  Random n_views: {random_n_views}, range: {view_range}")

    # ------------------------------------------------------------------
    # WebDataset creation
    # ------------------------------------------------------------------

    def _create_streaming_dataset(self):
        """Create streaming WebDataset pipeline."""
        if wds is None:
            raise ImportError(
                "webdataset is required. Install with: pip install webdataset"
            )

        full_url = os.path.join(self.data_root, self.url_pattern)

        # Try glob first (for patterns without braces); fall back to direct URL
        tar_files = sorted(glob.glob(full_url))
        if tar_files:
            urls = tar_files
            logging.info(f"Found {len(tar_files)} tar files via glob")
        else:
            urls = full_url
            logging.info(f"Using URL pattern directly: {full_url}")

        dataset = wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,
            workersplitter=None,
            shardshuffle=False,
            resampled=False,
            empty_check=False,  # Allow fewer shards than workers
        ).decode("rgb8")

        if self.training and self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)

        dataset = dataset.select(self._filter_valid)

        # Set epoch size if specified
        if self.epoch_size is not None:
            dataset = dataset.with_epoch(self.epoch_size)

        return dataset

    @staticmethod
    def _filter_valid(sample):
        """Filter out samples without valid hand data."""
        labels = sample.get("label.pyd", {})
        if not labels:
            return False
        if "joints_3d" not in labels or "verts_3d" not in labels:
            return False
        n_imgs = sum(1 for k in sample.keys() if k.startswith("image"))
        return n_imgs >= 1

    # ------------------------------------------------------------------
    # Hand-centric bbox crop helpers
    # ------------------------------------------------------------------

    def _crop_image_with_bbox(self, img, bbox_center, bbox_scale):
        """Crop image around hand using affine warp.

        Args:
            img: Input image (H, W, 3), uint8, RGB.
            bbox_center: Center of bounding box (2,).
            bbox_scale: Scale (side length) of bounding box, scalar.

        Returns:
            img_cropped: Cropped image (crop_size, crop_size, 3), uint8.
            affine: 3x3 affine transform matrix.
        """
        out_res = (self.bbox_crop_size, self.bbox_crop_size)
        affine = _get_affine_trans_no_rot(bbox_center, float(bbox_scale), out_res)

        affine_2x3 = affine[:2, :].astype(np.float32)
        img_cropped = cv2.warpAffine(
            img,
            affine_2x3,
            (self.bbox_crop_size, self.bbox_crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return img_cropped, affine.astype(np.float32)

    def _transform_cam_intr(self, cam_intr, bbox_center, bbox_scale):
        """Transform camera intrinsics after bbox crop.

        ``target_cam_intr = affine_postrot @ cam_intr``

        Args:
            cam_intr: Camera intrinsic matrix (3, 3).
            bbox_center: Center of bounding box (2,).
            bbox_scale: Scale of bounding box, scalar.

        Returns:
            Transformed camera intrinsic matrix (3, 3), float32.
        """
        out_res = (self.bbox_crop_size, self.bbox_crop_size)
        affine_postrot = _get_affine_trans_no_rot(
            bbox_center, float(bbox_scale), out_res
        )
        target_cam_intr = affine_postrot.dot(cam_intr)
        return target_cam_intr.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.len_data
    
    def __del__(self):
        """Cleanup resources when dataset is deleted."""
        if not self._closed:
            self.close()
    
    def close(self):
        """Explicitly close iterator and release resources."""
        if self._iterator is not None and hasattr(self._iterator, 'close'):
            try:
                self._iterator.close()
            except:
                pass
        self._iterator = None
        self._closed = True

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve the next multi-view sample from the stream.

        *seq_index* is ignored (streaming mode).

        Args:
            seq_index: Ignored.
            img_per_seq: Number of views to return.  If more views are
                requested than available in the sample, views are duplicated.
            seq_name: Ignored.
            ids: Ignored.
            aspect_ratio: Target aspect ratio.

        Returns:
            dict: Batch dict compatible with HGGT training pipeline.
        """
        if self._iterator is None:
            self._iterator = iter(self.webdataset)

        max_retries = 10
        for _ in range(max_retries):
            # resampled=True (training): infinite stream, no StopIteration expected
            # resampled=False (validation): let StopIteration propagate to the
            # DataLoader / AlternatingDataLoaderWrapper so it knows data is exhausted
            sample = next(self._iterator)

            try:
                batch = self._process_to_batch(sample, img_per_seq, aspect_ratio)
                if batch is not None:
                    return batch
            except Exception as e:
                logging.warning(f"Failed to process sample: {e}")
                continue

        logging.error("Failed to get valid sample after retries")
        return self._create_dummy_batch(img_per_seq or 2, aspect_ratio)

    # ------------------------------------------------------------------
    # Core sample processing
    # ------------------------------------------------------------------

    def _process_to_batch(self, sample, img_per_seq, aspect_ratio):
        """
        Convert a multi-view WebDataset sample into a HGGT batch.

        Steps per view:
        1. Handle request_flip (horizontal flip)
        2. Virtual camera rotation so the optical axis points toward the
           hand's bbox centre, then hand-centric bbox crop
        3. Update intrinsics for the crop
        4. Compute master-relative extrinsic with rotation correction
        5. Resize to target shape and scale intrinsics
        6. Project master_joints_3d to 2D keypoints using final params

        HGGT's pose encoding (``pose_enc.py``) assumes ``cx=W/2, cy=H/2``,
        so pp must be at image centre.  The virtual camera rotation is a
        pure rotation about the optical centre (exact, depth-independent)
        that brings the hand to the image centre.  After rotation, the
        bbox crop is centred on bbox_center ≈ pp, so pp ends up at the
        crop centre naturally — no pixel shift, no black borders.

        Args:
            sample: Raw WebDataset sample dict.
            img_per_seq: Target number of views.
            aspect_ratio: Target aspect ratio.

        Returns:
            Batch dict or None on failure.
        """
        key = sample.get("__key__", "unknown")
        labels = sample.get("label.pyd", {})
        if not labels:
            return None

        # ---- collect image keys ----
        n_view_imgs = {
            k: sample[k] for k in sample.keys() if k.startswith("image")
        }
        n_cams = len(n_view_imgs)
        if n_cams == 0:
            return None

        img_type = "png" if any("png" in k for k in n_view_imgs) else "jpg"

        # ---- extrinsic inversion for certain datasets ----
        if self.inv_extr and "cam_extr" in labels:
            labels['cam_extr'] = [
                np.linalg.inv(labels['cam_extr'][i]) for i in range(n_cams)
            ]

        # ---- view selection ----
        indices = list(range(n_cams))
        if self.random_n_views and self.training:
            random.shuffle(indices)
            if img_per_seq is not None:
                n = min(img_per_seq, n_cams)
            else:
                n = int(round(random.gauss(4, 2)))
                n = min(max(self.view_range[0], n), self.view_range[1])
                n = min(n, n_cams)
            indices_keep = indices[:n]
        else:
            indices_keep = indices
            if img_per_seq is not None and img_per_seq < n_cams:
                indices_keep = indices[:img_per_seq]

        # Pad with duplicates if needed
        if img_per_seq is not None:
            while len(indices_keep) < img_per_seq:
                indices_keep.append(
                    indices_keep[random.randint(0, len(indices_keep) - 1)]
                )

        # ---- master view ----
        master_id = indices_keep[0]
        master_joints_3d = np.array(
            labels["joints_3d"][master_id], dtype=np.float32
        )  # (21, 3)
        master_verts_3d = np.array(
            labels["verts_3d"][master_id], dtype=np.float32
        )  # (778, 3)

        # ---- MANO parameters (from master view) ----
        has_mano_pose = "mano_pose" in labels
        has_mano_shape = "mano_shape" in labels

        if has_mano_pose and has_mano_shape:
            raw_pose = np.array(labels["mano_pose"][master_id], dtype=np.float32)
            master_hand_pose = raw_pose.reshape(-1)[:48].copy()  # (48,)
            # HO3D/DexYCB/Freihand: pose stored in absolute space; subtract MANO mean
            if self.hand_mean_pose is not None:
                master_hand_pose[3:48] -= self.hand_mean_pose
            master_betas = np.array(
                labels["mano_shape"][master_id], dtype=np.float32
            ).reshape(-1)[:10].copy()  # (10,)
            has_hand_pose = 1.0
            has_betas = 1.0
        else:
            master_hand_pose = np.zeros(48, dtype=np.float32)
            master_betas = np.zeros(10, dtype=np.float32)
            has_hand_pose = 0.0
            has_betas = 0.0

        # cam_extr is camera-to-world (C2W) after inv_extr processing.
        # HGGT expects master-to-camera (M2C) extrinsics.
        # M2C = W2C_ind @ C2W_master = inv(C2W_ind) @ C2W_master
        T_master = (
            np.array(labels["cam_extr"][master_id], dtype=np.float64)
            if "cam_extr" in labels
            else np.eye(4, dtype=np.float64)
        )

        target_image_shape = self.get_target_shape(aspect_ratio)

        # ---- per-view processing ----
        images = []
        extrinsics_list = []
        intrinsics_list = []
        original_sizes = []
        keypoints_2d_list = []
        keypoints_depth_rel_list = []
        # _debug_raw_views = []  # for visualization only

        for ind in indices_keep:
            # Resolve image key
            img_key = f"image_{ind}.{img_type}"
            if img_key not in n_view_imgs:
                alt_type = "png" if img_type == "jpg" else "jpg"
                img_key = f"image_{ind}.{alt_type}"
                if img_key not in n_view_imgs:
                    continue

            img = n_view_imgs[img_key]
            if hasattr(img, 'mode'):  # PIL Image
                img = np.array(img)

            # Step 1: request_flip
            if labels.get("request_flip", False):
                cam_intr_raw = labels["cam_intr"][ind]
                raw_size = labels["raw_size"][ind]
                cam_center = np.array(
                    [cam_intr_raw[0, 2], cam_intr_raw[1, 2]]
                )
                M = np.array(
                    [[-1, 0, 2 * cam_center[0]], [0, 1, 0]],
                    dtype=np.float32,
                )
                img = cv2.warpAffine(img, M, tuple(raw_size))

            # Step 2: virtual camera rotation + bbox crop.
            #
            # HGGT's pose encoding assumes cx=W/2, cy=H/2 (see pose_enc.py),
            # so pp MUST be at the image centre.  Crops are around
            # bbox_center (hand centre), which generally differs from pp.
            # Cropping around bbox_center then shifting to centre pp causes
            # black borders; cropping around pp instead causes the hand to
            # be off-centre.
            #
            # Solution: rotate the camera so that its optical axis points
            # toward the bbox_center.  This is a pure rotation about the
            # camera's optical centre — it does not depend on depth and is
            # therefore exact.  After rotation the hand projects near the
            # new pp (image centre), and we can crop around pp = bbox_center
            # in the rotated image without conflict.
            #
            # Steps:
            #   a) Compute R_corr that maps the direction toward bbox_center
            #      to the optical axis [0, 0, 1].
            #   b) Warp the original image by the homography K @ R_corr @ K^-1.
            #   c) Crop the warped image with the affine (centred on
            #      bbox_center).  Since the hand now projects near
            #      bbox_center ≈ pp in the warped image, pp ends up at
            #      crop centre.
            #   d) Update extrinsics: R_new = R_corr @ R_old, t unchanged
            #      (rotation about optical centre does not affect t).

            cam_intr = np.array(labels["cam_intr"][ind], dtype=np.float32)
            bbox_center = labels["bbox_center"][ind]
            bbox_scale = labels["bbox_scale"][ind]
            fx = cam_intr[0, 0]
            fy = cam_intr[1, 1]
            cx_orig = cam_intr[0, 2]
            cy_orig = cam_intr[1, 2]

            # 2a. Direction from optical centre toward bbox_center.
            d = np.array([
                (bbox_center[0] - cx_orig) / fx,
                (bbox_center[1] - cy_orig) / fy,
                1.0,
            ], dtype=np.float64)
            d /= np.linalg.norm(d)
            # R_corr rotates d → [0, 0, 1]
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(d, z)
            s = np.linalg.norm(v)
            c = np.dot(d, z)
            if s < 1e-8:
                R_corr = np.eye(3, dtype=np.float64)
            else:
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0],
                ], dtype=np.float64)
                R_corr = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

            # 2b. Warp original image by homography H = K @ R_corr @ K^{-1}.
            K = cam_intr.astype(np.float64)
            K_inv = np.linalg.inv(K)
            H = K @ R_corr @ K_inv
            orig_h, orig_w = img.shape[:2]
            img_warped = cv2.warpPerspective(
                img, H.astype(np.float32), (orig_w, orig_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # After rotation the hand projects near pp = (cx, cy).
            # Crop the warped image centred on pp so that pp ends up at
            # the crop centre (= image centre after resize).
            pp = np.array([cx_orig, cy_orig])
            img_cropped, _ = self._crop_image_with_bbox(
                img_warped, pp, bbox_scale
            )

            # Step 3: intrinsics after crop centred on pp.
            # Since crop centre = pp, the new cx' = R/2 exactly.
            adjusted_intr = self._transform_cam_intr(
                cam_intr, pp, bbox_scale
            )

            # Step 4: master-relative extrinsic with rotation correction.
            if "cam_extr" in labels:
                T_cam = np.array(
                    labels["cam_extr"][ind], dtype=np.float64
                )
                T_cam_inv = np.linalg.inv(T_cam)         # W2C_ind
                T_rel = T_cam_inv @ T_master              # M2C
                extr_3x4 = T_rel[:3, :4].astype(np.float64)
                # Apply rotation correction to extrinsics
                R_old = extr_3x4[:3, :3]
                t_old = extr_3x4[:3, 3]
                R_new = R_corr @ R_old
                t_new = R_corr @ t_old
                extr_3x4 = np.hstack([R_new, t_new[:, None]]).astype(np.float32)
            else:
                extr_3x4 = np.eye(4, dtype=np.float32)[:3, :4]

            # Step 5: resize to target shape and scale intrinsics.
            tgt_h, tgt_w = int(target_image_shape[0]), int(target_image_shape[1])
            crop_h, crop_w = img_cropped.shape[:2]
            scale_h = tgt_h / crop_h
            scale_w = tgt_w / crop_w
            pil_img = PILImage.fromarray(img_cropped)
            resample = (PILImage.LANCZOS
                        if (scale_h < 1 or scale_w < 1)
                        else PILImage.BICUBIC)
            pil_img = pil_img.resize((tgt_w, tgt_h), resample)
            image_out = np.array(pil_img)

            final_intr = adjusted_intr.copy()
            final_intr[0, 0] *= scale_w   # fx
            final_intr[0, 2] *= scale_w   # cx
            final_intr[1, 1] *= scale_h   # fy
            final_intr[1, 2] *= scale_h   # cy

            # Step 6: project master_joints_3d to 2D keypoints in this view
            # using the final (post-resize) intrinsics and extrinsics
            joints_homog = np.concatenate(
                [master_joints_3d, np.ones((21, 1), dtype=np.float32)],
                axis=1,
            )  # (21, 4)
            T_rel_4x4 = np.eye(4, dtype=np.float32)
            T_rel_4x4[:3, :4] = extr_3x4
            joints_cam = (T_rel_4x4 @ joints_homog.T).T[:, :3]  # (21, 3)
            kp2d_homog = (final_intr @ joints_cam.T).T  # (21, 3)
            kp2d_z = kp2d_homog[:, 2:3].copy()
            kp2d_z[np.abs(kp2d_z) < 1e-6] = 1e-6
            kp2d_xy = kp2d_homog[:, :2] / kp2d_z
            kp2d = np.concatenate(
                [kp2d_xy, np.ones((21, 1), dtype=np.float32)], axis=1
            )  # (21, 3) with confidence=1

            # Root-relative depth: z_rel[i] = z_cam[i] - z_cam[wrist]
            depth_rel = (joints_cam[:, 2] - joints_cam[0, 2]).astype(np.float32)  # (21,)

            images.append(image_out)
            extrinsics_list.append(extr_3x4)
            intrinsics_list.append(final_intr)
            original_sizes.append(
                np.array([target_image_shape[0], target_image_shape[1]])
            )
            keypoints_2d_list.append(kp2d)
            keypoints_depth_rel_list.append(depth_rel)

            # # Debug: store raw view info for visualization
            # if "cam_extr" in labels:
            #     T_cam_dbg = np.array(labels["cam_extr"][ind], dtype=np.float64)
            #     T_rel_dbg = np.linalg.inv(T_cam_dbg) @ T_master
            #     orig_extr = T_rel_dbg[:3, :4].astype(np.float32)
            # else:
            #     orig_extr = np.eye(4, dtype=np.float32)[:3, :4]
            # _debug_raw_views.append({
            #     "raw_image": img.copy(),
            #     "cam_intr": cam_intr.copy(),
            #     "cam_extr_m2c": orig_extr.copy(),
            #     "bbox_center": np.array(bbox_center).copy(),
            #     "bbox_scale": float(bbox_scale),
            #     "view_idx": ind,
            # })

        if len(images) == 0:
            return None

        # ---- Canonicalize: make master camera = identity ----
        # After R_corr is applied in Step 4, the master extrinsic becomes
        # [R_corr_master | 0] instead of identity.  master_joints_3d /
        # master_verts_3d are still in the *original* master camera space
        # (before R_corr), so the 3D points and extrinsics are inconsistent.
        #
        # Fix: absorb R_corr_master into the 3D GT points and right-multiply
        # it out of all extrinsic rotations.  This is a pure change of basis:
        #   p_new  = R_corr_master @ p_old          (rotate 3D points)
        #   R_new  = R_old @ R_corr_master^T        (right-multiply extrinsics)
        #   t_new  = t_old                           (translation unchanged)
        # After this:
        #   master extrinsic → R_corr_master @ R_corr_master^T = I  ✓
        #   other extrinsics → consistent with new 3D frame
        #   2D keypoint projections → algebraically unchanged
        if len(extrinsics_list) > 0:
            R_corr_master = extrinsics_list[0][:3, :3].copy()  # (3, 3)
            master_joints_3d = (R_corr_master @ master_joints_3d.T).T.astype(np.float32)
            master_verts_3d  = (R_corr_master @ master_verts_3d.T).T.astype(np.float32)
            for i in range(len(extrinsics_list)):
                extr = extrinsics_list[i].copy()
                extr[:3, :3] = extr[:3, :3] @ R_corr_master.T
                extrinsics_list[i] = extr

        # ---- keypoints_3d with confidence channel ----
        keypoints_3d = np.concatenate(
            [master_joints_3d, np.ones((21, 1), dtype=np.float32)], axis=1
        )  # (21, 4)

        n_views = len(images)
        img_h, img_w = int(target_image_shape[0]), int(target_image_shape[1])

        batch = {
            # Metadata
            "seq_name": f"mv_{self.dataset_name}_{key}",
            "ids": np.arange(n_views, dtype=np.int32),
            "frame_num": n_views,
            # MANO parameters (from master view)
            "has_hand_pose": np.array(has_hand_pose, dtype=np.float32),
            "has_betas": np.array(has_betas, dtype=np.float32),
            "hand_pose": master_hand_pose,
            "betas": master_betas,
            "transl": np.zeros(3, dtype=np.float32),
            # Hand GT (in master camera space)
            "hand_verts": master_verts_3d,
            "keypoints_3d": keypoints_3d,
            "keypoints_2d": keypoints_2d_list,
            "keypoints_depth_rel": keypoints_depth_rel_list,  # Per-view root-relative camera-space depth [21]
            "has_keypoints_depth_rel": np.array(1.0, dtype=np.float32),  # GT camera params → accurate depths
            # Multi-view images
            "images": images,
            # Camera parameters
            "extrinsics": extrinsics_list,
            "intrinsics": intrinsics_list,
            "has_intrinsics": True,
            "has_hand_verts": True,
            "original_sizes": original_sizes,
            # View ordering metadata
            "view_indices": list(indices_keep),   # [int, ...] which camera indices were used, in model input order
            "master_view_idx": int(master_id),    # index of the master (first) view fed to the model
            # Depth/mask/point-cloud placeholders
            "depths": [np.zeros((img_h, img_w), dtype=np.float32)] * n_views,
            "hand_masks": [np.zeros((img_h, img_w), dtype=np.uint8)] * n_views,
            "object_masks": [np.zeros((img_h, img_w), dtype=np.uint8)] * n_views,
            "point_masks": [np.zeros((img_h, img_w), dtype=np.bool_)] * n_views,
            # # Debug: raw view info for visualization (not used in training)
            # "_debug_raw_views": _debug_raw_views,
        }

        return batch

    def _create_dummy_batch(self, img_per_seq, aspect_ratio):
        """Create a dummy batch for error fallback."""
        target_shape = self.get_target_shape(aspect_ratio)
        img_h, img_w = target_shape
        n = max(img_per_seq, 1)

        dummy_img = np.zeros((img_h, img_w, 3), dtype=np.float32)
        dummy_extr = np.eye(4, dtype=np.float32)[:3, :4]
        dummy_intr = np.array(
            [[500, 0, img_w / 2], [0, 500, img_h / 2], [0, 0, 1]],
            dtype=np.float32,
        )

        return {
            "seq_name": "mv_dummy",
            "ids": np.arange(n, dtype=np.int32),
            "frame_num": n,
            "has_hand_pose": np.array(0.0, dtype=np.float32),
            "has_betas": np.array(0.0, dtype=np.float32),
            "hand_pose": np.zeros(48, dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "transl": np.zeros(3, dtype=np.float32),
            "hand_verts": np.zeros((778, 3), dtype=np.float32),
            "keypoints_3d": np.zeros((21, 4), dtype=np.float32),
            "keypoints_2d": [np.zeros((21, 3), dtype=np.float32)] * n,
            "keypoints_depth_rel": [np.zeros(21, dtype=np.float32)] * n,
            "has_keypoints_depth_rel": np.array(0.0, dtype=np.float32),
            "images": [dummy_img] * n,
            "extrinsics": [dummy_extr] * n,
            "intrinsics": [dummy_intr] * n,
            "has_intrinsics": False,
            "has_hand_verts": False,
            "original_sizes": [np.array([img_h, img_w])] * n,
            # Properly-shaped zero placeholders (no real depth/mask data)
            "depths": [np.zeros((img_h, img_w), dtype=np.float32)] * n,
            "hand_masks": [np.zeros((img_h, img_w), dtype=np.uint8)] * n,
            "object_masks": [np.zeros((img_h, img_w), dtype=np.uint8)] * n,
            "point_masks": [np.zeros((img_h, img_w), dtype=np.bool_)] * n,
        }


# ======================================================================
# Standalone test
# ======================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(
        description="Test MVImageDataset loading"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing multi-view image tar files",
    )
    parser.add_argument(
        "--url_pattern", type=str, required=True,
        help='URL pattern, e.g. "HO3D_mv_train/HO3D_mv_train-{000000..000002}.tar"',
    )
    parser.add_argument(
        "--epoch_size", type=int, default=100,
        help="Number of samples per epoch",
    )
    parser.add_argument(
        "--img_per_seq", type=int, default=4,
        help="Number of views per sample",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./mv_image_dataset_vis",
        help="Output directory for visualization",
    )
    args = parser.parse_args()

    class MockAugs:
        scales = [0.9, 1.1]
        cojitter = False
        cojitter_ratio = 0.5
        color_jitter = False
        gray_scale = False
        gau_blur = False

    class MockConfig:
        training = True
        debug = False
        img_size = 518
        patch_size = 14
        rescale = True
        rescale_aug = False
        landscape_check = False
        augs = MockAugs()
        inside_random = False
        load_depth = False
        allow_duplicate_img = True
        fix_img_num = 0
        fix_aspect_ratio = 1
        load_track = False
        track_num = 0

    common_conf = MockConfig()

    dataset = MVImageDataset(
        common_conf=common_conf,
        split="train",
        data_root=args.data_root,
        url_pattern=args.url_pattern,
        epoch_size=args.epoch_size,
        shuffle=100,
        random_n_views=True,
        view_range=(2, 6),
    )

    # MANO hand skeleton connectivity (21 joints)
    HAND_BONES = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),# ring
        (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    ]
    FINGER_COLORS = [
        (0, 0, 255),    # thumb  - red
        (0, 165, 255),  # index  - orange
        (0, 255, 0),    # middle - green
        (255, 255, 0),  # ring   - cyan
        (255, 0, 0),    # pinky  - blue
    ]

    def draw_hand_skeleton(canvas, kp2d, img_h, img_w):
        """Draw hand skeleton with colored bones and joint dots on canvas."""
        for bone_idx, (j1, j2) in enumerate(HAND_BONES):
            color = FINGER_COLORS[bone_idx // 4]
            c1, c2 = kp2d[j1, 2], kp2d[j2, 2]
            x1, y1 = int(kp2d[j1, 0]), int(kp2d[j1, 1])
            x2, y2 = int(kp2d[j2, 0]), int(kp2d[j2, 1])
            in1 = c1 > 0.5 and 0 <= x1 < img_w and 0 <= y1 < img_h
            in2 = c2 > 0.5 and 0 <= x2 < img_w and 0 <= y2 < img_h
            if in1 and in2:
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        for j in range(21):
            if kp2d[j, 2] > 0.5:
                x, y = int(kp2d[j, 0]), int(kp2d[j, 1])
                if 0 <= x < img_w and 0 <= y < img_h:
                    color = FINGER_COLORS[min((j - 1) // 4, 4)] if j > 0 else (255, 255, 255)
                    cv2.circle(canvas, (x, y), 3, color, -1)
                    cv2.circle(canvas, (x, y), 3, (0, 0, 0), 1)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Testing {args.num_samples} samples...\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        batch = dataset.get_data(
            img_per_seq=args.img_per_seq, aspect_ratio=1.0
        )

        n_views = batch['frame_num']
        seq_name = batch['seq_name']
        print(f"Sample {i}: {seq_name}")
        print(f"  frame_num:    {n_views}")
        print(f"  images:       {len(batch['images'])} x {batch['images'][0].shape}")
        print(f"  extrinsics:   {len(batch['extrinsics'])} x {np.array(batch['extrinsics'][0]).shape}")
        print(f"  intrinsics:   {len(batch['intrinsics'])} x {np.array(batch['intrinsics'][0]).shape}")
        print(f"  keypoints_3d: {batch['keypoints_3d'].shape}")
        print(f"  hand_verts:   {batch['hand_verts'].shape}")
        print(f"  keypoints_2d: {len(batch['keypoints_2d'])} x {batch['keypoints_2d'][0].shape}")
        print(f"  has_hand_pose: {batch['has_hand_pose']}")
        print(f"  has_betas:     {batch['has_betas']}")

        # --- Per-view visualization ---
        view_canvases = []
        for v in range(n_views):
            img_v = batch['images'][v].copy()
            if img_v.max() <= 1.0:
                img_v = (img_v * 255).astype(np.uint8)
            bgr = cv2.cvtColor(img_v, cv2.COLOR_RGB2BGR)
            img_h, img_w = bgr.shape[:2]

            kp2d = batch['keypoints_2d'][v]  # (21, 3) [x, y, conf]
            extr = batch['extrinsics'][v]    # (3, 4)
            intr = batch['intrinsics'][v]    # (3, 3)

            draw_hand_skeleton(bgr, kp2d, img_h, img_w)

            # Annotate view index and whether it is the master view
            tag = f"View {v}" + (" [master]" if v == 0 else "")
            cv2.putText(bgr, tag, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Print extrinsic R (first row) and t to quickly spot identity
            R = extr[:3, :3]
            t = extr[:3, 3]
            is_identity = np.allclose(R, np.eye(3), atol=1e-4) and np.allclose(t, 0, atol=1e-4)
            extr_tag = "E=I" if is_identity else f"t=[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]"
            cv2.putText(bgr, extr_tag, (5, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Count how many keypoints are inside the image
            n_inside = 0
            for j in range(21):
                if kp2d[j, 2] > 0.5:
                    x, y = kp2d[j, 0], kp2d[j, 1]
                    if 0 <= x < img_w and 0 <= y < img_h:
                        n_inside += 1
            kp_tag = f"KP inside: {n_inside}/21"
            cv2.putText(bgr, kp_tag, (5, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Print focal length
            fx, fy = intr[0, 0], intr[1, 1]
            cv2.putText(bgr, f"f=[{fx:.0f},{fy:.0f}]", (5, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            view_canvases.append(bgr)

            # Also save individual view
            view_path = os.path.join(
                args.output_dir, f"sample_{i:03d}_view{v}.jpg"
            )
            cv2.imwrite(view_path, bgr)

        # Tile all views into a single image (2 rows max)
        n_cols = min(n_views, 4)
        n_rows = (n_views + n_cols - 1) // n_cols
        tile_h, tile_w = view_canvases[0].shape[:2]
        grid = np.zeros((n_rows * tile_h, n_cols * tile_w, 3), dtype=np.uint8)
        for v, canvas in enumerate(view_canvases):
            r, c = divmod(v, n_cols)
            grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = canvas
        grid_path = os.path.join(args.output_dir, f"sample_{i:03d}_grid.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"  Saved grid: {grid_path}")

        # --- Reprojection verification ---
        # Re-project master_joints_3d with stored extrinsics/intrinsics and
        # compare with stored kp2d to verify consistency.
        kp3d = batch['keypoints_3d'][:, :3]  # (21, 3)  master-camera-space
        for v in range(n_views):
            extr = batch['extrinsics'][v]
            intr = batch['intrinsics'][v]
            kp2d_stored = batch['keypoints_2d'][v]

            pts_h = np.concatenate([kp3d, np.ones((21, 1), dtype=np.float32)], axis=1)
            E4 = np.eye(4, dtype=np.float32)
            E4[:3, :4] = extr
            pts_cam = (E4 @ pts_h.T).T[:, :3]
            proj = (intr @ pts_cam.T).T
            z = proj[:, 2:3].copy()
            z[np.abs(z) < 1e-6] = 1e-6
            proj_xy = proj[:, :2] / z

            err = np.linalg.norm(proj_xy - kp2d_stored[:, :2], axis=1)
            print(f"  View {v}: reproj err mean={err.mean():.2f}px  max={err.max():.2f}px")

        print()

    print("Done!")
