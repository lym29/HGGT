"""Stream the single-view hand annotations distributed with HaMeR.

The annotation format and confidence filter follow HaMeR (https://github.com/geopavlakos/hamer).
"""

import logging
from pathlib import Path
import pickle
import random

import numpy as np
from torch.utils.data import get_worker_info

from hggt.training.data.base_dataset import BaseDataset
from hggt.training.data.dataset_util import fliplr_keypoints, fliplr_mano_params
from hggt.training.data.worker_fn import get_rank, get_world_size


class HaMeRImageDataset(BaseDataset):
    is_multiview = False
    DATASETS = ("freihand", "ho3d", "interhand26m", "h2o3d", "dex", "rhd", "mtc", "cocow", "halpe", "mpiinzsl")

    def __init__(self, common_conf, split="train", hamer_data_dir=None,
                 dataset_name="freihand", mano_model_path=None, len_train=50000,
                 len_test=5000, max_samples=-1, hamer_img_size=256,
                 hamer_focal_length=5000.0, epoch_size=None):
        super().__init__(common_conf)
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown HaMeR dataset {dataset_name!r}; choose from {self.DATASETS}")
        if hamer_data_dir is None or mano_model_path is None:
            raise ValueError("hamer_data_dir and mano_model_path are required")
        shard_dir = Path(hamer_data_dir) / "dataset_tars" / f"{self.dataset_name}-{split}"
        self.shards = sorted(str(path) for path in shard_dir.glob("*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"No HaMeR tar shards found: {shard_dir}/*.tar")
        mano_file = Path(mano_model_path) / "MANO_RIGHT.pkl"
        if not mano_file.is_file():
            raise FileNotFoundError(f"Missing licensed MANO model: {mano_file}")
        with mano_file.open("rb") as handle:
            self.hand_mean_pose = np.asarray(pickle.load(handle, encoding="latin1")["hands_mean"], dtype=np.float32).reshape(45)
        self.length = int(epoch_size if epoch_size is not None else (len_train if split == "train" else len_test))
        if max_samples >= 0:
            self.length = min(self.length, int(max_samples))
        if self.length <= 0:
            raise ValueError("HaMeR epoch length must be positive")
        self.hamer_focal_length = float(hamer_focal_length)
        self.rank, self.world_size = get_rank(), get_world_size()
        self._iterator = None
        self._epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        self._iterator = None

    def _create_stream(self):
        import webdataset as wds
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        global_worker = self.rank * num_workers + worker_id
        total_workers = self.world_size * num_workers
        shards = self.shards[global_worker::total_workers]
        if not shards:
            # Small datasets may have fewer shards than distributed workers.
            shards = [self.shards[global_worker % len(self.shards)]]
            logging.warning("HaMeR %s has %d shards for %d workers; worker %d repeats a shard",
                            self.dataset_name, len(self.shards), total_workers, global_worker)
        pipeline = wds.WebDataset(shards, shardshuffle=False, nodesplitter=None, workersplitter=None)
        pipeline = pipeline.decode("rgb8").rename(jpg="jpg;jpeg;png")
        pipeline = pipeline.compose(self._split_annotations).select(self._has_enough_keypoints)
        if self.training:
            pipeline = pipeline.shuffle(100, rng=random.Random(random.getrandbits(64)))
        return iter(pipeline)

    @staticmethod
    def _split_annotations(source):
        for item in source:
            key = item.get("__key__", "unknown")
            if "jpg" not in item or "data.pyd" not in item:
                raise ValueError(f"HaMeR sample {key}: expected RGB image and data.pyd annotations")
            annotations = item["data.pyd"]
            if not isinstance(annotations, (list, tuple)):
                raise ValueError(f"HaMeR sample {key}: data.pyd must contain a list of annotations")
            for index, annotation in enumerate(annotations):
                yield {"__key__": f"{key}_{index}", "image": item["jpg"], "annotation": annotation}

    @staticmethod
    def _has_enough_keypoints(item):
        joints = np.asarray(item["annotation"].get("keypoints_2d", np.zeros((21, 3))))
        if joints.shape != (21, 3):
            raise ValueError(f"HaMeR sample {item['__key__']}: keypoints_2d must have shape (21, 3)")
        return np.count_nonzero(joints[:, 2] > 0.3) >= 4

    def _next_sample(self):
        # Retry one exhausted pass. Empty/fully filtered shards must fail, not recurse.
        for _ in range(2):
            if self._iterator is None:
                self._iterator = self._create_stream()
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = None
        raise RuntimeError(f"HaMeR {self.dataset_name}: assigned shards contain no samples with four confident keypoints")

    @staticmethod
    def _crop_image_bbox(image, center, scale, min_size=32):
        bbox_size = float(np.max(scale)) * 200.0
        if not np.isfinite(center).all() or not np.isfinite(bbox_size) or bbox_size <= 0:
            raise ValueError("HaMeR annotation has a nonfinite center or nonpositive bounding box")
        cx, cy = map(int, center)
        half_size = int(bbox_size / 2)
        x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
        x2, y2 = min(image.shape[1], cx + half_size), min(image.shape[0], cy + half_size)
        if x2 <= 0 or y2 <= 0 or x1 >= image.shape[1] or y1 >= image.shape[0]:
            raise ValueError("HaMeR bounding box does not overlap its image")
        if x2 - x1 < min_size:
            expand = (min_size - (x2 - x1)) // 2
            x1, x2 = max(0, x1 - expand), min(image.shape[1], x2 + expand)
            if x2 - x1 < min_size:
                if x1 == 0:
                    x2 = min(image.shape[1], min_size)
                else:
                    x1 = max(0, x2 - min_size)
        if y2 - y1 < min_size:
            expand = (min_size - (y2 - y1)) // 2
            y1, y2 = max(0, y1 - expand), min(image.shape[0], y2 + expand)
            if y2 - y1 < min_size:
                if y1 == 0:
                    y2 = min(image.shape[0], min_size)
                else:
                    y1 = max(0, y2 - min_size)
        return image[y1:y2, x1:x2].copy(), (x1, y1)

    def get_data(self, seq_index=None, img_per_seq=1, seq_name=None, ids=None, aspect_ratio=1.0):
        if img_per_seq != 1:
            raise ValueError("HaMeR provides one view per sample; set the single-view sampler to one view")
        sample = self._next_sample()
        data, key = sample["annotation"], sample["__key__"]
        try:
            image = np.asarray(sample["image"])
            center = np.asarray(data["center"], dtype=np.float32).reshape(2).copy()
            scale = np.asarray(data["scale"], dtype=np.float32)
            joints2d = np.asarray(data["keypoints_2d"], dtype=np.float32).reshape(21, 3).copy()
            joints3d = np.asarray(data.get("keypoints_3d", np.zeros((21, 4))), dtype=np.float32).reshape(21, 4).copy()
            has_mano = float(data.get("has_hand_pose", 0)) > 0 and float(data.get("has_betas", 0)) > 0
            pose = np.asarray(data["hand_pose"] if has_mano else np.zeros(48), dtype=np.float32).reshape(48).copy()
            betas = np.asarray(data["betas"] if has_mano else np.zeros(10), dtype=np.float32).reshape(10).copy()
            if not all(np.isfinite(value).all() for value in (joints2d, joints3d, pose, betas)):
                raise ValueError("nonfinite keypoints or MANO parameters")
            flip = float(data.get("right", 1)) < 0.5
            if flip:
                image = np.fliplr(image).copy()
                center[0] = image.shape[1] - center[0]
                joints2d = fliplr_keypoints(joints2d, image.shape[1], list(range(21)))
                joints3d = fliplr_keypoints(joints3d, 1, list(range(21)))
                if has_mano:
                    pose = fliplr_mano_params(pose)
            image, offset = self._crop_image_bbox(image, center, scale)
            joints2d[:, :2] -= np.asarray(offset)
            height, width = image.shape[:2]
            intrinsic = np.array([[self.hamer_focal_length, 0, width / 2],
                                  [0, self.hamer_focal_length, height / 2], [0, 0, 1]], dtype=np.float32)
            extrinsic = np.eye(4, dtype=np.float32)[:3]
            image, _, extrinsic, intrinsic, _, joints2d = self.process_one_image(
                image, None, extrinsic, intrinsic, self.get_target_shape(aspect_ratio),
                keypoints_2d=joints2d, filepath=key,
            )
            if has_mano:
                pose[3:] -= self.hand_mean_pose
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid HaMeR {self.dataset_name} sample {key}: {error}") from error
        return {"seq_name": f"hamer_{self.dataset_name}_{key}", "ids": np.array([0]),
                "images": [image], "extrinsics": [extrinsic], "intrinsics": [intrinsic],
                "depths": [np.zeros(image.shape[:2], dtype=np.float32)],
                "point_masks": [np.zeros(image.shape[:2], dtype=bool)],
                "hand_pose": pose, "betas": betas, "keypoints_2d": [joints2d], "keypoints_3d": joints3d,
                "hand_verts": np.zeros((778, 3), dtype=np.float32),
                "has_hand_pose": np.float32(has_mano), "has_betas": np.float32(has_mano),
                "has_intrinsics": False, "has_hand_verts": False}
