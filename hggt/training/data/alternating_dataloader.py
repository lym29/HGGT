"""Alternate real single-view and synthetic multi-view HGGT batches."""

import math
import random
from collections.abc import Mapping

from hydra.utils import instantiate
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from hggt.training.data.worker_fn import get_rank, get_world_size, get_worker_init_fn


def flexible_collate_fn(batch):
    """Stack common fields; retain sample identifiers and other strings as lists."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    common = set.intersection(*(set(sample) for sample in batch))
    result = {}
    for key in sorted(common):
        values = [sample[key] for sample in batch]
        first = values[0]
        try:
            if isinstance(first, (torch.Tensor, np.ndarray)):
                result[key] = torch.stack([torch.as_tensor(value) for value in values])
            elif isinstance(first, (int, float, bool, np.number)):
                result[key] = torch.as_tensor(values)
            elif isinstance(first, Mapping):
                result[key] = flexible_collate_fn(values)
            else:
                result[key] = values
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Cannot collate field {key!r}; samples must have the same view count and shape") from exc
    return result


class FixedBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=True):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.sampler, self.batch_size, self.drop_last = sampler, batch_size, drop_last

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append((index, 1, 1.0))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size if self.drop_last else math.ceil(len(self.sampler) / self.batch_size)


class DynamicBatchSamplerForMultiview(Sampler):
    """Choose one view count/aspect per batch, with B*S <= max_img_per_gpu."""
    def __init__(self, sampler, aspect_ratio_range, image_num_range, max_img_per_gpu=32, seed=42):
        self.sampler = sampler
        self.aspect_ratio_range = tuple(aspect_ratio_range)
        self.image_num_range = tuple(image_num_range)
        self.max_img_per_gpu, self.seed, self.epoch = max_img_per_gpu, seed, 0
        lo, hi = self.image_num_range
        if not 1 <= lo <= hi <= max_img_per_gpu:
            raise ValueError("View range must satisfy 1 <= min <= max <= max_img_per_gpu")
        if not 0 < self.aspect_ratio_range[0] <= self.aspect_ratio_range[1]:
            raise ValueError("Aspect ratio range must be positive and ordered")

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def _batch_parameters(self):
        rng = random.Random(self.epoch * 100)
        choices = list(range(self.image_num_range[0], self.image_num_range[1] + 1))
        remaining = len(self.sampler)
        while remaining:
            views = rng.choices(choices)[0]
            aspect = round(rng.uniform(*self.aspect_ratio_range), 2)
            count = min(remaining, self.max_img_per_gpu // views)
            yield count, views, aspect
            remaining -= count

    def __iter__(self):
        indices = iter(self.sampler)
        for count, views, aspect in self._batch_parameters():
            yield [(next(indices), views, aspect) for _ in range(count)]

    def __len__(self):
        return sum(1 for _ in self._batch_parameters())


class AlternatingDataLoaderWrapper:
    """Yield (batch, dataset_name, accumulation_index), stopping at complete pairs."""
    def __init__(self, dataloader1, dataloader2, accumulation_steps=4,
                 dataset1_name="single", dataset2_name="multiple"):
        if accumulation_steps < 2 or accumulation_steps % 2:
            raise ValueError("accumulation_steps must be a positive even number")
        self.dataloader1, self.dataloader2 = dataloader1, dataloader2
        self.accumulation_steps = accumulation_steps
        self.dataset1_name, self.dataset2_name = dataset1_name, dataset2_name

    def __iter__(self):
        for pair_index, (single, multiple) in enumerate(zip(self.dataloader1, self.dataloader2)):
            step = 2 * pair_index
            yield single, self.dataset1_name, step % self.accumulation_steps
            yield multiple, self.dataset2_name, (step + 1) % self.accumulation_steps

    def __len__(self):
        return 2 * min(len(self.dataloader1), len(self.dataloader2))


class AlternatingMultiDataLoader:
    def __init__(self, dataset1, dataset2, common_config1, common_config2,
                 num_workers=4, shuffle=True, pin_memory=True, drop_last=True,
                 collate_fn=None, worker_init_fn=None, persistent_workers=False,
                 seed=42, max_img_per_gpu=32, accumulation_steps=4,
                 dataset1_name="single", dataset2_name="multiple"):
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if accumulation_steps < 2 or accumulation_steps % 2:
            raise ValueError("accumulation_steps must be positive and even")
        self.dataset1 = instantiate(dataset1, common_config=common_config1, _recursive_=False)
        self.dataset2 = instantiate(dataset2, common_config=common_config2, _recursive_=False)
        self.seed, self.num_workers = seed, num_workers
        self.worker_init_fn = worker_init_fn
        self.loader_options = dict(num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=persistent_workers and num_workers > 0,
                                   collate_fn=collate_fn or flexible_collate_fn)
        if num_workers > 0:
            # CUDA/NCCL is initialized before iteration; workers must not fork it.
            self.loader_options["multiprocessing_context"] = "spawn"
        distributed = dict(num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle, seed=seed)
        self.sampler1 = DistributedSampler(self.dataset1, **distributed)
        self.sampler2 = DistributedSampler(self.dataset2, **distributed)
        self.batch_sampler1 = FixedBatchSampler(self.sampler1, max_img_per_gpu, drop_last)
        self.batch_sampler2 = DynamicBatchSamplerForMultiview(
            self.sampler2, common_config2.augs.aspects, common_config2.img_nums, max_img_per_gpu, seed,
        )
        self.wrapper_options = dict(accumulation_steps=accumulation_steps,
                                    dataset1_name=dataset1_name, dataset2_name=dataset2_name)

    def get_loader(self, epoch):
        self.sampler1.set_epoch(epoch)
        self.batch_sampler2.set_epoch(epoch + self.seed)
        for dataset in (self.dataset1, self.dataset2):
            dataset.set_epoch(epoch)
        initialize = get_worker_init_fn(self.seed, self.num_workers, epoch, self.worker_init_fn)
        loaders = [DataLoader(dataset, batch_sampler=sampler, worker_init_fn=initialize,
                              **self.loader_options)
                   for dataset, sampler in ((self.dataset1, self.batch_sampler1),
                                             (self.dataset2, self.batch_sampler2))]
        return AlternatingDataLoaderWrapper(*loaders, **self.wrapper_options)
