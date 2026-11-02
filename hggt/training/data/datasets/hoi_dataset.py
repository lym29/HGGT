"""Synthetic HGGT multi-view sequences with world-coordinate MANO supervision."""

from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import random

import cv2
import numpy as np

from hggt.training.data.base_dataset import BaseDataset
from hggt.training.data.dataset_util import init_mano_model, precompute_mano_outputs, threshold_depth_map


class HOIDataset(BaseDataset):
    is_multiview = True

    def __init__(self, common_conf, split="train", HOI_DATA_DIR=None,
                 min_num_images=8, max_samples=-1, mano_model_path=None, split_dir="splits"):
        super().__init__(common_conf)
        if HOI_DATA_DIR is None or mano_model_path is None:
            raise ValueError("HOI_DATA_DIR and mano_model_path are required")
        self.root = Path(HOI_DATA_DIR)
        self.mano_model_path = str(mano_model_path)
        if not (Path(mano_model_path) / "MANO_RIGHT.pkl").is_file():
            raise FileNotFoundError(f"Missing licensed MANO model: {Path(mano_model_path) / 'MANO_RIGHT.pkl'}")
        self.load_depth = common_conf.load_depth
        self.depth_scale = float(getattr(common_conf, "depth_scale", 1.0))
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.min_num_images = int(min_num_images)
        self.mano_model = None
        split_file = self.root / split_dir / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing HGGT split: {split_file}; extract the dataset and retain its splits directory")
        self.sequences = self._load_split(split_file)
        if not self.sequences:
            raise ValueError(f"No synthetic sequences listed in {split_file}")
        self.length = len(self.sequences) * 10 if max_samples < 0 else int(max_samples)
        if self.length <= 0:
            raise ValueError("Synthetic epoch length must be positive")

    def _load_split(self, split_file):
        sequences = []
        category_roots = {}
        for category in ("small", "medium", "large"):
            roots = [base for base in (self.root, self.root / "data") if (base / category).is_dir()]
            if len(roots) > 1:
                raise ValueError(f"Ambiguous extracted category {category!r}: found both {roots}")
            if roots:
                category_roots[category] = roots[0]
        for line_number, line in enumerate(split_file.read_text().splitlines(), 1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            path = PurePosixPath(entry)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid relative path at {split_file}:{line_number}: {entry}")
            # Resolve sequence entries without stat-ing every sequence on a network filesystem.
            if path.name.startswith("sequence_") and (len(path.parts) == 4 or (len(path.parts) == 5 and path.parts[0] == "data")):
                if path.parts[0] == "data" and len(path.parts) == 5:
                    path = PurePosixPath(*path.parts[1:])
                category = path.parts[0]
                if category not in category_roots:
                    raise FileNotFoundError(f"No extracted category {category!r} under {self.root} or its data directory")
                sequences.append(str((category_roots[category] / path).relative_to(self.root)))
                continue
            candidates = [self.root / path, self.root / "data" / path]
            if len(path.parts) == 1:
                candidates = [base / category / path for base in (self.root, self.root / "data")
                              for category in ("small", "medium", "large")]
            found = [candidate for candidate in candidates if candidate.is_dir()]
            if not found:
                raise FileNotFoundError(f"Split entry {entry!r} is not extracted under {self.root} or its data directory")
            if len(found) > 1:
                raise ValueError(f"Ambiguous split entry {entry!r}: found both {found}; keep one extracted copy")
            directory = found[0]
            if len(path.parts) in (1, 2):
                children = sorted(directory.glob("*/sequence_*"))
                if not children:
                    raise ValueError(f"No motion/sequence_* directories under split object {directory}")
                sequences.extend(str(child.relative_to(self.root)) for child in children if child.is_dir())
            else:
                raise ValueError(f"Unsupported HGGT split entry {entry!r}; expected category/object/motion/sequence or object ID")
        return list(dict.fromkeys(sequences))

    def __len__(self):
        return self.length

    @staticmethod
    def _array(value, shape, label):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{label} must be a finite array with shape {shape}; got {array.shape}")
        return array

    @lru_cache(maxsize=128)
    def _sequence_metadata(self, name):
        directory = self.root / name
        info_file = directory / "info.json"
        if not info_file.is_file():
            raise FileNotFoundError(f"Missing synthetic sequence metadata: {info_file}")
        info = json.loads(info_file.read_text())
        if "mano_params" not in info:
            raise ValueError(f"{info_file} must contain portable mano_params (global_orient, hand_pose, betas, transl)")
        params = info["mano_params"]
        try:
            pose = np.concatenate([self._array(params["global_orient"], (3,), "global_orient"),
                                   self._array(params["hand_pose"], (45,), "hand_pose")])
            betas = self._array(params["betas"], (10,), "betas")
            transl = self._array(params["transl"], (3,), "transl")
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid MANO parameters in {info_file}: {error}") from error
        if self.mano_model is None:
            self.mano_model = init_mano_model(self.mano_model_path)
        vertices, joints = precompute_mano_outputs(self.mano_model, pose, betas, transl)
        intrinsic_file = directory / "camera" / "intrinsics.txt"
        if not intrinsic_file.is_file():
            raise FileNotFoundError(f"Missing camera intrinsics: {intrinsic_file}")
        intrinsic = self._array(np.loadtxt(intrinsic_file), (3, 3), str(intrinsic_file))
        frames = sorted(path for path in (directory / "images").iterdir()
                        if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(frames) < self.min_num_images:
            raise ValueError(f"{directory}: expected at least {self.min_num_images} views, found {len(frames)}")
        return pose, betas, vertices, joints, intrinsic, frames

    @staticmethod
    def _read_image(path, *, mask=False):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {'mask' if mask else 'RGB image'}: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing or unreadable {'mask' if mask else 'RGB image'}: {path}")
        return image if mask else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_data(self, seq_index=None, img_per_seq=1, seq_name=None, ids=None, aspect_ratio=1.0):
        if seq_name is None:
            index = random.randrange(len(self.sequences)) if self.inside_random else int(seq_index or 0) % len(self.sequences)
            seq_name = self.sequences[index]
        elif seq_name not in self.sequences:
            raise KeyError(f"Sequence is not in this split: {seq_name}")
        pose, betas, vertices, joints, intrinsic, frames = self._sequence_metadata(seq_name)
        if ids is None:
            if not self.allow_duplicate_img and img_per_seq > len(frames):
                raise ValueError(f"Requested {img_per_seq} distinct views from {seq_name}, which has {len(frames)}")
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(frames)):
            raise ValueError(f"Invalid view indices for {seq_name}: {ids}")
        result = {key: [] for key in ("images", "intrinsics", "extrinsics", "depths", "point_masks", "keypoints_2d")}
        directory = self.root / seq_name
        for frame_id in ids:
            image_path = frames[int(frame_id)]
            image = self._read_image(image_path)
            extrinsic_path = directory / "camera" / "extrinsics" / f"{image_path.stem}.txt"
            if not extrinsic_path.is_file():
                raise FileNotFoundError(f"Missing camera extrinsics: {extrinsic_path}")
            extrinsic = self._array(np.loadtxt(extrinsic_path)[:3, :4], (3, 4), str(extrinsic_path))
            masks = [self._read_image(directory / "masks" / f"{image_path.stem}_{name}_mask0001.png", mask=True)
                     for name in ("hand_right", "object")]
            if any(mask.shape != image.shape[:2] for mask in masks):
                raise ValueError(f"Mask resolution differs from its RGB image: {image_path}")
            depth_path = directory / "depths" / f"{image_path.stem}.npy"
            depth = None
            if self.load_depth and depth_path.is_file():
                depth = np.load(depth_path, allow_pickle=False)
                if depth.shape != image.shape[:2]:
                    raise ValueError(f"Depth resolution differs from its RGB image: {depth_path}")
                depth = threshold_depth_map(depth, min_percentile=-1, max_percentile=98) * self.depth_scale
                depth[~((masks[0] > 0) | (masks[1] > 0))] = 100.0
            camera_joints = joints @ extrinsic[:, :3].T + extrinsic[:, 3]
            projected = camera_joints @ intrinsic.T
            if np.any(np.abs(projected[:, 2]) < 1e-8):
                raise ValueError(f"Cannot project MANO joints at zero camera depth: {image_path}")
            joints2d = np.concatenate([projected[:, :2] / projected[:, 2:3], np.ones((21, 1))], axis=1).astype(np.float32)
            image, depth, extrinsic, camera, masks, joints2d = self.process_one_image(
                image, depth, extrinsic, intrinsic, self.get_target_shape(aspect_ratio),
                masks=masks, keypoints_2d=joints2d, filepath=str(image_path),
            )
            result["images"].append(image)
            result["intrinsics"].append(camera)
            result["extrinsics"].append(extrinsic)
            result["point_masks"].append((masks[0] > 0) | (masks[1] > 0))
            result["keypoints_2d"].append(joints2d)
            if self.load_depth:
                result["depths"].append(np.zeros(image.shape[:2], dtype=np.float32) if depth is None else depth.astype(np.float32))
        result.update(seq_name=f"synthetic_{seq_name}", ids=ids, hand_pose=pose.copy(), betas=betas.copy(),
                      hand_verts=vertices.copy(), keypoints_3d=np.concatenate([joints, np.ones((21, 1), dtype=np.float32)], axis=1),
                      has_hand_pose=np.float32(1), has_betas=np.float32(1), has_intrinsics=True, has_hand_verts=True)
        return result
