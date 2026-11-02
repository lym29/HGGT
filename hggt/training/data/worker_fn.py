# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the terms in licenses/VGGT.txt.
# Modified for HGGT.

"""Seed workers using the parent rank, including multiprocessing spawn workers."""

from functools import partial
import random

import numpy as np
import torch
import torch.distributed as dist


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def default_worker_init_fn(worker_id, *, num_workers, epoch, seed, rank, world_size):
    worker_seed = seed + rank * max(1, num_workers) + worker_id + world_size + epoch * 12345
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    torch.manual_seed(worker_seed)


def get_worker_init_fn(seed, num_workers, epoch, worker_init_fn=None):
    if worker_init_fn is not None:
        return worker_init_fn
    return partial(default_worker_init_fn, num_workers=num_workers, epoch=epoch,
                   seed=seed, rank=get_rank(), world_size=get_world_size())
