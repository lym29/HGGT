#!/usr/bin/env python3
"""
Multi-GPU multi-view image (mv-image) hand pose evaluation for HGGT.

Features:
- Uses MVImageDataset (identical preprocessing to training, with R_corr)
- Evaluates multi-view image datasets (HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand)
- Multi-GPU via torchrun + PyTorch DDP (sample-level round-robin across ranks)
- Optional per-sample camera dumps; aggregates metrics on rank 0

Launch (single GPU)::

    python eval/eval_mv_image.py --data_root /path/to/mv_image_data --output_dir outputs/eval

Launch (multi-GPU)::

    torchrun --nproc_per_node=4 eval/eval_mv_image.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --data_root /path/to/mv_image_data \\
        --output_dir outputs/eval

Author: Yumeng Liu (lym29@connect.hku.hk)
"""

from __future__ import annotations

import argparse
import datetime
import glob as glob_module
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from smplx import MANO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.data.mv_image_dataset import MVImageDataset
from eval.metrics import Joint3DPCK, PAEval, Vert3DPCK
from hggt.training.train_utils.projection import compute_mano_output
from hggt.utils.hf_load import DEFAULT_HF_REPO, load_hggt
from hggt.utils.pose_enc import pose_encoding_to_extri_intri


# ==============================================================================
# Dataset Metadata (test/validation splits)
# ==============================================================================
DATASET_META = {
    "HO3D": {
        "url": "HO3D_mv_test/HO3D_mv_test-{000000..000002}.tar",
        "max_view": 5,
        "epoch_size": 2706,      # 2 × 3 × 11 × 41
    },
    "DexYCB": {
        "url": "DexYCB_mv/DexYCB_mv_test-{000000..000003}.tar",
        "max_view": 8,
        "epoch_size": 4950,      # 2 × 9 × 25 × 11
    },
    "Arctic": {
        "url": "Arctic_mv/Arctic_mv_val_p1-{000000..000045}.tar",
        "max_view": 8,
        "epoch_size": 17392,     # 2^4 × 1087
    },
    "Interhand": {
        "url": "Interhand_mv/Interhand_mv_val-{000000..000022}.tar",
        "max_view": 8,
        "epoch_size": 85255,     # 5 × 17051
    },
    "Oakink": {
        "url": "Oakink_mv/Oakink_mv_test-{000000..000045}.tar",
        "max_view": 4,
        "epoch_size": 21351,     # 3 × 7117
    },
    "Freihand": {
        "url": "Freihand_mv/Freihand_mv_test-{000000..000000}.tar",
        "max_view": 1,
        "epoch_size": 3960,      # 8 × 9 × 5 × 11
    },
}

ALL_DATASETS = list(DATASET_META.keys())


# ==============================================================================
# Mock config for MVImageDataset (no training augmentation)
# ==============================================================================
class _EvalAugs:
    scales = [1.0, 1.0]
    cojitter = False
    cojitter_ratio = 0.5
    color_jitter = False
    gray_scale = False
    gau_blur = False


class _EvalConf:
    """Minimal config compatible with MVImageDataset / BaseDataset."""
    training = False
    debug = False
    img_size = 518          # overridden per run from args
    patch_size = 14
    rescale = True
    rescale_aug = False
    landscape_check = False
    inside_random = False
    load_depth = False
    allow_duplicate_img = True
    fix_img_num = 0
    fix_aspect_ratio = 1
    load_track = False
    track_num = 0
    augs = _EvalAugs()


# ==============================================================================
# DDP Utilities
# ==============================================================================
def init_distributed():
    """Initialize distributed process group if torchrun environment is set."""
    if "RANK" not in os.environ:
        return False, 0, 1  # single-process mode
    # Long timeout (4h) so barrier/all_reduce don't timeout when one rank is still
    # in a long per-dataset loop (e.g. Arctic ~2h); default 10min is too short.
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=4),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True, rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool):
    if distributed:
        dist.barrier()


def all_gather_records(local_records: list, distributed: bool) -> list:
    """Gather per-sample records from all ranks to rank 0."""
    if not distributed:
        return local_records
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_records)
    # Flatten list-of-lists
    flat = []
    for part in gathered:
        flat.extend(part)
    return flat


# ==============================================================================
# Metric helpers
# ==============================================================================
_pa_eval_instance = PAEval(cfg=None, mesh_score=True)

_pck_config_50mm = {"VAL_MIN": 0.0, "VAL_MAX": 0.05, "STEPS": 100}
_pck_config_20mm = {"VAL_MIN": 0.0, "VAL_MAX": 0.02, "STEPS": 100}


def _make_pck_evaluators():
    """Create a fresh set of PCK evaluators for one dataset evaluation run."""
    def _j(cfg): c = cfg.copy(); c["EVAL_TYPE"] = "joints_3d"; return Joint3DPCK(**c)
    def _v(cfg): c = cfg.copy(); c["EVAL_TYPE"] = "verts_3d";  return Vert3DPCK(**c)

    return {
        # Original — 50mm / 20mm
        "j50":  _j(_pck_config_50mm), "v50":  _v(_pck_config_50mm),
        "j20":  _j(_pck_config_20mm), "v20":  _v(_pck_config_20mm),
        # PA — 50mm / 20mm
        "pa_j50": _j(_pck_config_50mm), "pa_v50": _v(_pck_config_50mm),
        "pa_j20": _j(_pck_config_20mm), "pa_v20": _v(_pck_config_20mm),
        # RR — 50mm / 20mm
        "rr_j50": _j(_pck_config_50mm), "rr_v50": _v(_pck_config_50mm),
        "rr_j20": _j(_pck_config_20mm), "rr_v20": _v(_pck_config_20mm),
        # PA-orig — 50mm / 20mm
        "pa_orig_j50": _j(_pck_config_50mm), "pa_orig_v50": _v(_pck_config_50mm),
        "pa_orig_j20": _j(_pck_config_20mm), "pa_orig_v20": _v(_pck_config_20mm),
        # RR-orig — 50mm / 20mm
        "rr_orig_j50": _j(_pck_config_50mm), "rr_orig_v50": _v(_pck_config_50mm),
        "rr_orig_j20": _j(_pck_config_20mm), "rr_orig_v20": _v(_pck_config_20mm),
        # CA — 50mm / 20mm
        "ca_j50": _j(_pck_config_50mm), "ca_v50": _v(_pck_config_50mm),
        "ca_j20": _j(_pck_config_20mm), "ca_v20": _v(_pck_config_20mm),
        # CA-PA — 50mm / 20mm
        "ca_pa_j50": _j(_pck_config_50mm), "ca_pa_v50": _v(_pck_config_50mm),
        "ca_pa_j20": _j(_pck_config_20mm), "ca_pa_v20": _v(_pck_config_20mm),
        # CA-RR — 50mm / 20mm
        "ca_rr_j50": _j(_pck_config_50mm), "ca_rr_v50": _v(_pck_config_50mm),
        "ca_rr_j20": _j(_pck_config_20mm), "ca_rr_v20": _v(_pck_config_20mm),
    }


def _cam_align_rigid_shift(pred_points, K_pred, K_gt):
    """Rigidly shift 3D points from predicted camera space to GT camera space (centroid-based)."""
    centroid = pred_points.mean(axis=0)
    z = float(centroid[2])
    if abs(z) < 1e-8:
        return pred_points.copy().astype(np.float32), np.zeros(3, dtype=np.float32)
    depth_scale = float(K_gt[0, 0] / K_pred[0, 0])
    u = K_pred[0, 0] * centroid[0] / z + K_pred[0, 2]
    v = K_pred[1, 1] * centroid[1] / z + K_pred[1, 2]
    z_new = z * depth_scale
    x_new = (u - K_gt[0, 2]) * z_new / K_gt[0, 0]
    y_new = (v - K_gt[1, 2]) * z_new / K_gt[1, 1]
    delta = np.array([x_new - centroid[0], y_new - centroid[1], z_new - z], dtype=np.float32)
    return (pred_points + delta).astype(np.float32), delta


def _transform_pred_to_gt_camera(points_3d, K_pred, K_gt):
    """Per-point re-projection from K_pred to K_gt camera frame."""
    Z = points_3d[:, 2]
    depth_scale = float(K_gt[0, 0] / K_pred[0, 0])
    u = K_pred[0, 0] * points_3d[:, 0] / Z + K_pred[0, 2]
    v = K_pred[1, 1] * points_3d[:, 1] / Z + K_pred[1, 2]
    Z_new = Z * depth_scale
    X_new = (u - K_gt[0, 2]) * Z_new / K_gt[0, 0]
    Y_new = (v - K_gt[1, 2]) * Z_new / K_gt[1, 1]
    return np.stack([X_new, Y_new, Z_new], axis=1).astype(np.float32)


def compute_metrics_for_records(records: list):
    """
    Compute all metrics (MPJPE, MPVPE, PCK, PA, RR, CA, ...) from gathered per-sample records.

    Each record is a dict with keys:
        pred_joints: (21, 3) float32
        pred_verts:  (778, 3) float32
        gt_joints:   (21, 3) float32
        gt_verts:    (778, 3) float32
        K_pred:      (3, 3) float32  — predicted master-view intrinsics (or None)
        K_gt:        (3, 3) float32  — GT master-view intrinsics (or None)

    Returns:
        dict of scalar metric values.
    """
    pck = _make_pck_evaluators()
    orig_mpjpe_list, orig_mpvpe_list = [], []

    for rec in records:
        pj = rec["pred_joints"][np.newaxis]   # (1, 21, 3)
        pv = rec["pred_verts"][np.newaxis]    # (1, 778, 3)
        gj = rec["gt_joints"][np.newaxis]     # (1, 21, 3)
        gv = rec["gt_verts"][np.newaxis]      # (1, 778, 3)
        K_pred = rec.get("K_pred")
        K_gt   = rec.get("K_gt")

        # --- Intrinsic correction ---
        if K_pred is not None and K_gt is not None:
            pj_corr = _transform_pred_to_gt_camera(pj[0], K_pred, K_gt)[np.newaxis]
            pv_corr = _transform_pred_to_gt_camera(pv[0], K_pred, K_gt)[np.newaxis]
        else:
            pj_corr, pv_corr = pj.copy(), pv.copy()

        # Original (before correction)
        orig_mpjpe_list.append(float(_pa_eval_instance.get_dist(pj, gj)[0]))
        orig_mpvpe_list.append(float(_pa_eval_instance.get_dist(pv, gv)[0]))

        # --- PA (corrected) ---
        pj_pa, _ = _pa_eval_instance.align_w_scale(gj[0], pj_corr[0])
        pv_pa, _ = _pa_eval_instance.align_w_scale(gv[0], pv_corr[0])

        # --- RR (corrected) ---
        root_corr = pj_corr[:, 0:1, :]
        gt_root   = gj[:, 0:1, :]
        pj_rr = pj_corr - root_corr
        pv_rr = pv_corr - root_corr
        gj_rr = gj - gt_root
        gv_rr = gv - gt_root

        # --- PA-orig (original pred, no correction) ---
        pj_pa_orig, _ = _pa_eval_instance.align_w_scale(gj[0], pj[0])
        pv_pa_orig, _ = _pa_eval_instance.align_w_scale(gv[0], pv[0])

        # --- RR-orig ---
        root_orig = pj[:, 0:1, :]
        pj_rr_orig = pj - root_orig
        pv_rr_orig = pv - root_orig

        # --- CA (camera-aligned rigid shift, original pred) ---
        if K_pred is not None and K_gt is not None:
            pv_ca, delta = _cam_align_rigid_shift(pv[0], K_pred, K_gt)
            pj_ca = (pj[0] + delta).astype(np.float32)
        else:
            pv_ca = pv[0].copy()
            pj_ca = pj[0].copy()

        # --- CA-PA ---
        pj_ca_pa, _ = _pa_eval_instance.align_w_scale(gj[0], pj_ca)
        pv_ca_pa, _ = _pa_eval_instance.align_w_scale(gv[0], pv_ca)

        # --- CA-RR ---
        root_ca = pj_ca[0:1]
        pj_ca_rr = pj_ca - root_ca
        pv_ca_rr = pv_ca - root_ca

        def _feed(tag_j, tag_v, preds_j, preds_v, tgt_j, tgt_v):
            pd = {
                "pred_joints_3d": torch.from_numpy(np.array(preds_j)[np.newaxis]).float(),
                "pred_verts_3d":  torch.from_numpy(np.array(preds_v)[np.newaxis]).float(),
            }
            td = {
                "master_joints_3d": torch.from_numpy(np.array(tgt_j)[np.newaxis]).float(),
                "master_verts_3d":  torch.from_numpy(np.array(tgt_v)[np.newaxis]).float(),
            }
            pck[tag_j].feed(pd, td)
            pck[tag_v].feed(pd, td)

        _feed("j50", "v50", pj_corr[0], pv_corr[0], gj[0], gv[0])
        _feed("j20", "v20", pj_corr[0], pv_corr[0], gj[0], gv[0])
        _feed("pa_j50", "pa_v50", pj_pa, pv_pa, gj[0], gv[0])
        _feed("pa_j20", "pa_v20", pj_pa, pv_pa, gj[0], gv[0])
        _feed("rr_j50", "rr_v50", pj_rr[0], pv_rr[0], gj_rr[0], gv_rr[0])
        _feed("rr_j20", "rr_v20", pj_rr[0], pv_rr[0], gj_rr[0], gv_rr[0])
        _feed("pa_orig_j50", "pa_orig_v50", pj_pa_orig, pv_pa_orig, gj[0], gv[0])
        _feed("pa_orig_j20", "pa_orig_v20", pj_pa_orig, pv_pa_orig, gj[0], gv[0])
        _feed("rr_orig_j50", "rr_orig_v50", pj_rr_orig[0], pv_rr_orig[0], gj_rr[0], gv_rr[0])
        _feed("rr_orig_j20", "rr_orig_v20", pj_rr_orig[0], pv_rr_orig[0], gj_rr[0], gv_rr[0])
        _feed("ca_j50", "ca_v50", pj_ca, pv_ca, gj[0], gv[0])
        _feed("ca_j20", "ca_v20", pj_ca, pv_ca, gj[0], gv[0])
        _feed("ca_pa_j50", "ca_pa_v50", pj_ca_pa, pv_ca_pa, gj[0], gv[0])
        _feed("ca_pa_j20", "ca_pa_v20", pj_ca_pa, pv_ca_pa, gj[0], gv[0])
        _feed("ca_rr_j50", "ca_rr_v50", pj_ca_rr, pv_ca_rr, gj_rr[0], gv_rr[0])
        _feed("ca_rr_j20", "ca_rr_v20", pj_ca_rr, pv_ca_rr, gj_rr[0], gv_rr[0])

    def _m(tag):  return pck[tag].get_measures()
    def _auc(tag): return _m(tag)["auc_all"] * 100
    def _epe(tag): return _m(tag)["epe_mean_all"] * 1000
    def _pck20(tag): return pck[tag].get_pck_all(0.02) * 100

    metrics = {
        "orig_mpjpe": np.mean(orig_mpjpe_list) * 1000 if orig_mpjpe_list else 0.0,
        "orig_mpvpe": np.mean(orig_mpvpe_list) * 1000 if orig_mpvpe_list else 0.0,
        # With IC
        "mpjpe": _epe("j50"), "mpvpe": _epe("v50"),
        "auc50_joints": _auc("j50"), "auc50_verts": _auc("v50"),
        "auc20_joints": _auc("j20"), "auc20_verts": _auc("v20"),
        "pck@20mm": _pck20("j50"), "pckv@20mm": _pck20("v50"),
        # PA (IC)
        "pa_mpjpe": _epe("pa_j50"), "pa_mpvpe": _epe("pa_v50"),
        "pa_auc50_joints": _auc("pa_j50"), "pa_auc50_verts": _auc("pa_v50"),
        "pa_auc20_joints": _auc("pa_j20"), "pa_auc20_verts": _auc("pa_v20"),
        "pa_pck@20mm": _pck20("pa_j50"), "pa_pckv@20mm": _pck20("pa_v50"),
        # RR (IC)
        "rr_mpjpe": _epe("rr_j50"), "rr_mpvpe": _epe("rr_v50"),
        "rr_auc50_joints": _auc("rr_j50"), "rr_auc50_verts": _auc("rr_v50"),
        "rr_auc20_joints": _auc("rr_j20"), "rr_auc20_verts": _auc("rr_v20"),
        "rr_pck@20mm": _pck20("rr_j50"), "rr_pckv@20mm": _pck20("rr_v50"),
        # PA-orig
        "pa_orig_mpjpe": _epe("pa_orig_j50"), "pa_orig_mpvpe": _epe("pa_orig_v50"),
        "pa_orig_auc50_joints": _auc("pa_orig_j50"), "pa_orig_auc50_verts": _auc("pa_orig_v50"),
        "pa_orig_auc20_joints": _auc("pa_orig_j20"), "pa_orig_auc20_verts": _auc("pa_orig_v20"),
        "pa_orig_pck@20mm": _pck20("pa_orig_j50"), "pa_orig_pckv@20mm": _pck20("pa_orig_v50"),
        # RR-orig
        "rr_orig_mpjpe": _epe("rr_orig_j50"), "rr_orig_mpvpe": _epe("rr_orig_v50"),
        "rr_orig_auc50_joints": _auc("rr_orig_j50"), "rr_orig_auc50_verts": _auc("rr_orig_v50"),
        "rr_orig_auc20_joints": _auc("rr_orig_j20"), "rr_orig_auc20_verts": _auc("rr_orig_v20"),
        "rr_orig_pck@20mm": _pck20("rr_orig_j50"), "rr_orig_pckv@20mm": _pck20("rr_orig_v50"),
        # CA
        "ca_mpjpe": _epe("ca_j50"), "ca_mpvpe": _epe("ca_v50"),
        "ca_auc50_joints": _auc("ca_j50"), "ca_auc50_verts": _auc("ca_v50"),
        "ca_auc20_joints": _auc("ca_j20"), "ca_auc20_verts": _auc("ca_v20"),
        "ca_pck@20mm": _pck20("ca_j50"), "ca_pckv@20mm": _pck20("ca_v50"),
        # CA-PA
        "ca_pa_mpjpe": _epe("ca_pa_j50"), "ca_pa_mpvpe": _epe("ca_pa_v50"),
        "ca_pa_auc50_joints": _auc("ca_pa_j50"), "ca_pa_auc50_verts": _auc("ca_pa_v50"),
        "ca_pa_auc20_joints": _auc("ca_pa_j20"), "ca_pa_auc20_verts": _auc("ca_pa_v20"),
        "ca_pa_pck@20mm": _pck20("ca_pa_j50"), "ca_pa_pckv@20mm": _pck20("ca_pa_v50"),
        # CA-RR
        "ca_rr_mpjpe": _epe("ca_rr_j50"), "ca_rr_mpvpe": _epe("ca_rr_v50"),
        "ca_rr_auc50_joints": _auc("ca_rr_j50"), "ca_rr_auc50_verts": _auc("ca_rr_v50"),
        "ca_rr_auc20_joints": _auc("ca_rr_j20"), "ca_rr_auc20_verts": _auc("ca_rr_v20"),
        "ca_rr_pck@20mm": _pck20("ca_rr_j50"), "ca_rr_pckv@20mm": _pck20("ca_rr_v50"),
        "n_samples": len(records),
    }
    return metrics


# ==============================================================================
# Result Formatting
# ==============================================================================
def format_metrics(dataset_name: str, checkpoint: str, metrics: dict) -> str:
    """Format metrics dict into a human-readable string."""
    n = metrics.get("n_samples", "?")
    lines = [
        "=" * 80,
        f"Dataset: {dataset_name}   Checkpoint: {checkpoint}   Samples: {n}",
        "=" * 80,
        "",
        "[1. Original - Before Intrinsic Correction]",
        f"  MPJPE (orig):  {metrics['orig_mpjpe']:.2f} mm",
        f"  MPVPE (orig):  {metrics['orig_mpvpe']:.2f} mm",
        "",
        "[2. With Intrinsic Correction]",
        f"  MPJPE:  {metrics['mpjpe']:.2f} mm",
        f"  MPVPE:  {metrics['mpvpe']:.2f} mm",
        f"  AUC@50mm (joints): {metrics['auc50_joints']:.2f}%",
        f"  AUC@50mm (verts):  {metrics['auc50_verts']:.2f}%",
        f"  AUC@20mm (joints): {metrics['auc20_joints']:.2f}%",
        f"  AUC@20mm (verts):  {metrics['auc20_verts']:.2f}%",
        f"  PCK@20mm:  {metrics['pck@20mm']:.2f}%",
        f"  PCKV@20mm: {metrics['pckv@20mm']:.2f}%",
        "",
        "[3. PA - Procrustes Aligned (IC corrected)]",
        f"  PA-MPJPE:  {metrics['pa_mpjpe']:.2f} mm",
        f"  PA-MPVPE:  {metrics['pa_mpvpe']:.2f} mm",
        f"  PA-AUC@50mm (joints): {metrics['pa_auc50_joints']:.2f}%",
        f"  PA-AUC@50mm (verts):  {metrics['pa_auc50_verts']:.2f}%",
        f"  PA-AUC@20mm (joints): {metrics['pa_auc20_joints']:.2f}%",
        f"  PA-AUC@20mm (verts):  {metrics['pa_auc20_verts']:.2f}%",
        f"  PA-PCK@20mm:  {metrics['pa_pck@20mm']:.2f}%",
        f"  PA-PCKV@20mm: {metrics['pa_pckv@20mm']:.2f}%",
        "",
        "[4. RR - Root Relative (IC corrected)]",
        f"  RR-MPJPE:  {metrics['rr_mpjpe']:.2f} mm",
        f"  RR-MPVPE:  {metrics['rr_mpvpe']:.2f} mm",
        f"  RR-AUC@50mm (joints): {metrics['rr_auc50_joints']:.2f}%",
        f"  RR-AUC@50mm (verts):  {metrics['rr_auc50_verts']:.2f}%",
        f"  RR-AUC@20mm (joints): {metrics['rr_auc20_joints']:.2f}%",
        f"  RR-AUC@20mm (verts):  {metrics['rr_auc20_verts']:.2f}%",
        f"  RR-PCK@20mm:  {metrics['rr_pck@20mm']:.2f}%",
        f"  RR-PCKV@20mm: {metrics['rr_pckv@20mm']:.2f}%",
        "",
        "[5. PA - Procrustes Aligned (original, no IC)]",
        f"  PA-MPJPE (orig):  {metrics['pa_orig_mpjpe']:.2f} mm",
        f"  PA-MPVPE (orig):  {metrics['pa_orig_mpvpe']:.2f} mm",
        f"  PA-AUC@50mm (joints): {metrics['pa_orig_auc50_joints']:.2f}%",
        f"  PA-AUC@50mm (verts):  {metrics['pa_orig_auc50_verts']:.2f}%",
        f"  PA-AUC@20mm (joints): {metrics['pa_orig_auc20_joints']:.2f}%",
        f"  PA-AUC@20mm (verts):  {metrics['pa_orig_auc20_verts']:.2f}%",
        f"  PA-PCK@20mm:  {metrics['pa_orig_pck@20mm']:.2f}%",
        f"  PA-PCKV@20mm: {metrics['pa_orig_pckv@20mm']:.2f}%",
        "",
        "[6. RR - Root Relative (original, no IC)]",
        f"  RR-MPJPE (orig):  {metrics['rr_orig_mpjpe']:.2f} mm",
        f"  RR-MPVPE (orig):  {metrics['rr_orig_mpvpe']:.2f} mm",
        f"  RR-AUC@50mm (joints): {metrics['rr_orig_auc50_joints']:.2f}%",
        f"  RR-AUC@50mm (verts):  {metrics['rr_orig_auc50_verts']:.2f}%",
        f"  RR-AUC@20mm (joints): {metrics['rr_orig_auc20_joints']:.2f}%",
        f"  RR-AUC@20mm (verts):  {metrics['rr_orig_auc20_verts']:.2f}%",
        f"  RR-PCK@20mm:  {metrics['rr_orig_pck@20mm']:.2f}%",
        f"  RR-PCKV@20mm: {metrics['rr_orig_pckv@20mm']:.2f}%",
        "",
        "[7. CA - Camera-Aligned (rigid centroid shift, K_pred→K_gt)]",
        f"  CA-MPJPE:  {metrics['ca_mpjpe']:.2f} mm",
        f"  CA-MPVPE:  {metrics['ca_mpvpe']:.2f} mm",
        f"  CA-AUC@50mm (joints): {metrics['ca_auc50_joints']:.2f}%",
        f"  CA-AUC@50mm (verts):  {metrics['ca_auc50_verts']:.2f}%",
        f"  CA-AUC@20mm (joints): {metrics['ca_auc20_joints']:.2f}%",
        f"  CA-AUC@20mm (verts):  {metrics['ca_auc20_verts']:.2f}%",
        f"  CA-PCK@20mm:  {metrics['ca_pck@20mm']:.2f}%",
        f"  CA-PCKV@20mm: {metrics['ca_pckv@20mm']:.2f}%",
        "",
        "[8. CA-PA - Procrustes Aligned on Camera-Aligned]",
        f"  CA-PA-MPJPE:  {metrics['ca_pa_mpjpe']:.2f} mm",
        f"  CA-PA-MPVPE:  {metrics['ca_pa_mpvpe']:.2f} mm",
        f"  CA-PA-AUC@50mm (joints): {metrics['ca_pa_auc50_joints']:.2f}%",
        f"  CA-PA-AUC@50mm (verts):  {metrics['ca_pa_auc50_verts']:.2f}%",
        f"  CA-PA-AUC@20mm (joints): {metrics['ca_pa_auc20_joints']:.2f}%",
        f"  CA-PA-AUC@20mm (verts):  {metrics['ca_pa_auc20_verts']:.2f}%",
        f"  CA-PA-PCK@20mm:  {metrics['ca_pa_pck@20mm']:.2f}%",
        f"  CA-PA-PCKV@20mm: {metrics['ca_pa_pckv@20mm']:.2f}%",
        "",
        "[9. CA-RR - Root-Relative on Camera-Aligned]",
        f"  CA-RR-MPJPE:  {metrics['ca_rr_mpjpe']:.2f} mm",
        f"  CA-RR-MPVPE:  {metrics['ca_rr_mpvpe']:.2f} mm",
        f"  CA-RR-AUC@50mm (joints): {metrics['ca_rr_auc50_joints']:.2f}%",
        f"  CA-RR-AUC@50mm (verts):  {metrics['ca_rr_auc50_verts']:.2f}%",
        f"  CA-RR-AUC@20mm (joints): {metrics['ca_rr_auc20_joints']:.2f}%",
        f"  CA-RR-AUC@20mm (verts):  {metrics['ca_rr_auc20_verts']:.2f}%",
        f"  CA-RR-PCK@20mm:  {metrics['ca_rr_pck@20mm']:.2f}%",
        f"  CA-RR-PCKV@20mm: {metrics['ca_rr_pckv@20mm']:.2f}%",
        "",
    ]
    return "\n".join(lines)


def format_summary_row(dataset_name: str, metrics: dict) -> str:
    """One-line compact summary row for the combined table."""
    return (
        f"{dataset_name:<12} "
        f"MPJPE={metrics['mpjpe']:6.2f}  PA={metrics['pa_mpjpe']:6.2f}  "
        f"RR={metrics['rr_mpjpe']:6.2f}  CA={metrics['ca_mpjpe']:6.2f}  "
        f"CA-PA={metrics['ca_pa_mpjpe']:6.2f}  CA-RR={metrics['ca_rr_mpjpe']:6.2f}  "
        f"n={metrics['n_samples']}"
    )


# ==============================================================================
# Per-dataset Evaluation Loop
# ==============================================================================
def evaluate_dataset(
    dataset_name: str,
    model,
    mano_model,
    args,
    device: str,
    rank: int,
    world_size: int,
    distributed: bool,
    dtype,
) -> list:
    """
    Run inference over one multi-view image dataset.

    Uses sample-level round-robin distribution: rank i processes samples where
    global_index % world_size == rank. This avoids NCCL timeout when shard count
    < world_size, and ensures perfect load balance regardless of shard count.
    Returns a list of per-sample record dicts (local to this rank).
    Each record contains raw predictions + GT needed to compute any metric.
    """
    meta = DATASET_META[dataset_name]

    # Build a fresh _EvalConf with the right img_size
    conf = _EvalConf()
    conf.img_size = args.img_size

    # Build dataset for its preprocessing pipeline (_process_to_batch).
    # We do NOT use dataset.webdataset for iteration (it has split_by_node which
    # causes NCCL timeout when shard count < world_size). Instead we build raw_wds
    # below with nodesplitter=None and do sample-level round-robin ourselves.
    logging.info(f"[Rank {rank}] Creating MVImageDataset for {dataset_name}...")
    dataset = MVImageDataset(
        common_conf=conf,
        split="test",
        data_root=args.data_root,
        url_pattern=meta["url"],
        epoch_size=meta["epoch_size"],
        shuffle=0,           # deterministic
        random_n_views=False,  # use all available views per sample
        mano_model_path=args.mano_model_path,
    )

    # Build a fresh webdataset WITHOUT node splitting so all ranks see all shards.
    # We do sample-level round-robin below to distribute work evenly.
    full_url = os.path.join(args.data_root, meta["url"])
    tar_files = sorted(glob_module.glob(full_url))
    urls = tar_files if tar_files else full_url
    raw_wds = wds.WebDataset(
        urls,
        nodesplitter=None,   # disable shard-level splitting
        shardshuffle=False,
        resampled=False,
        empty_check=False,
    ).decode("rgb8")

    local_records = []
    sample_count = 0  # samples processed by this rank
    max_samples = args.max_samples  # -1 means no limit
    batch_size = getattr(args, 'batch_size', 1)  # default to 1 if not specified

    desc = f"[Rank {rank}] {dataset_name}"
    pbar = tqdm(desc=desc, disable=(rank != 0))

    # Batch accumulation buffers
    batch_buffer = []
    batch_keys = []

    def process_batch_buffer():
        """Process accumulated samples in batch_buffer."""
        nonlocal sample_count
        if not batch_buffer:
            return

        # Stack images from all samples: (B, S, 3, H, W)
        imgs_list = [item["imgs_tensor"] for item in batch_buffer]
        imgs_batch = torch.cat(imgs_list, dim=0).to(device)  # (B, S, 3, H, W)

        # ----- Model inference -----
        with torch.no_grad():
            if device != "cpu":
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(imgs_batch)
            else:
                predictions = model(imgs_batch)

        # ----- Process each sample in the batch -----
        for i, item in enumerate(batch_buffer):
            sample_key = item["sample_key"]
            gt_joints = item["gt_joints"]
            gt_verts = item["gt_verts"]
            gt_intrinsics = item["gt_intrinsics"]
            gt_extrinsics = item["gt_extrinsics"]
            K_gt = item["K_gt"]
            view_indices = item["view_indices"]
            master_view_idx = item["master_view_idx"]

            if "mano_params" not in predictions:
                logging.warning(f"[Rank {rank}] No MANO params for {sample_key}, skipping.")
                continue

            mano_params_i = {
                k: v[i : i + 1] if v is not None else None
                for k, v in predictions["mano_params"].items()
            }
            with torch.no_grad():
                pred_verts_t, pred_joints_t = compute_mano_output(
                    mano_model,
                    hand_pose=mano_params_i["hand_pose"],
                    shape=mano_params_i["betas"],
                    transl=mano_params_i["transl"],
                    is_right=True,
                )
            pred_joints = pred_joints_t[0].cpu().numpy().astype(np.float32)  # (21, 3)
            pred_verts = pred_verts_t[0].cpu().numpy().astype(np.float32)  # (778, 3)

            # ----- Extract predicted camera params -----
            pred_extrinsics = None
            pred_intrinsics_all = None
            K_pred = None

            if "pose_enc" in predictions:
                img_h, img_w = imgs_batch.shape[-2:]
                # Extract pose_enc for sample i
                pose_enc_i = predictions["pose_enc"][i:i+1]
                pred_extr_t, pred_intr_t = pose_encoding_to_extri_intri(
                    pose_enc_i, (img_h, img_w)
                )
                # Make relative to first camera
                pred_extr_np = pred_extr_t[0].cpu().numpy()   # (S, 3, 4) absolute
                pred_intr_np = pred_intr_t[0].cpu().numpy()   # (S, 3, 3)

                # Convert to relative: first cam = identity
                first_4x4 = np.eye(4, dtype=np.float64)
                first_4x4[:3, :] = pred_extr_np[0]
                cam0_to_world = np.linalg.inv(first_4x4)
                pred_extr_rel = np.zeros_like(pred_extr_np)
                for s in range(pred_extr_np.shape[0]):
                    if s == 0:
                        pred_extr_rel[s] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)
                    else:
                        e4 = np.eye(4, dtype=np.float64)
                        e4[:3, :] = pred_extr_np[s]
                        pred_extr_rel[s] = (e4 @ cam0_to_world)[:3, :].astype(np.float32)

                pred_extrinsics = pred_extr_rel.astype(np.float32)   # (S, 3, 4)
                pred_intrinsics_all = pred_intr_np.astype(np.float32) # (S, 3, 3)
                K_pred = pred_intrinsics_all[0]

            # ----- Build per-sample record -----
            record = {
                "sample_key":     sample_key,
                "dataset":        dataset_name,
                "pred_joints":    pred_joints,
                "pred_verts":     pred_verts,
                "gt_joints":      gt_joints,
                "gt_verts":       gt_verts,
                "K_pred":         K_pred,
                "K_gt":           K_gt,
                "view_indices":   view_indices,
                "master_view_idx": master_view_idx,
                "pred_extrinsics":     pred_extrinsics,
                "pred_intrinsics_all": pred_intrinsics_all,
                "gt_intrinsics":       gt_intrinsics,
                "gt_extrinsics":       gt_extrinsics,
            }
            local_records.append(record)
            sample_count += 1
            pbar.update(1)

        # Clear buffer
        batch_buffer.clear()
        batch_keys.clear()

    for global_idx, raw_sample in enumerate(raw_wds):
        # Sample-level round-robin: skip samples not assigned to this rank
        if global_idx % world_size != rank:
            continue

        if max_samples > 0 and sample_count >= max_samples:
            break

        sample_key = raw_sample.get("__key__", f"sample_{sample_count:06d}")

        batch = dataset._process_to_batch(raw_sample, img_per_seq=None, aspect_ratio=1.0)
        if batch is None:
            raise RuntimeError(f"[Rank {rank}] _process_to_batch returned None for sample {sample_key}")

        # ----- Convert images to (1, S, 3, H, W) float32 tensor -----
        imgs_np = np.stack(batch["images"])                           # (S, H, W, 3) uint8
        imgs_tensor = torch.from_numpy(
            imgs_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        ).unsqueeze(0)                                                # (1, S, 3, H, W)

        # ----- Ground truth -----
        gt_joints = batch["keypoints_3d"][:, :3].astype(np.float32)  # (21, 3)
        gt_verts  = batch["hand_verts"].astype(np.float32)            # (778, 3)

        # GT camera params (post-crop + R_corr, same space as model input)
        gt_intrinsics = np.stack(batch["intrinsics"]).astype(np.float32)  # (S, 3, 3)
        K_gt = gt_intrinsics[0]  # master view

        # Build (S, 4, 4) GT extrinsics (master-relative) for saving
        extr_list = []
        for ext3x4 in batch["extrinsics"]:
            e4 = np.eye(4, dtype=np.float32)
            e4[:3, :] = ext3x4
            extr_list.append(e4)
        gt_extrinsics = np.stack(extr_list)  # (S, 4, 4)

        # Add to batch buffer
        batch_buffer.append({
            "sample_key": sample_key,
            "imgs_tensor": imgs_tensor,
            "gt_joints": gt_joints,
            "gt_verts": gt_verts,
            "gt_intrinsics": gt_intrinsics,
            "gt_extrinsics": gt_extrinsics,
            "K_gt": K_gt,
            "view_indices": batch["view_indices"],
            "master_view_idx": batch["master_view_idx"],
        })
        batch_keys.append(sample_key)

        # Process batch when buffer is full
        if len(batch_buffer) >= batch_size:
            process_batch_buffer()

    # Process remaining samples in buffer
    if batch_buffer:
        process_batch_buffer()

    pbar.close()
    dataset.close()
    logging.info(f"[Rank {rank}] {dataset_name}: processed {sample_count} samples.")
    return local_records


# ==============================================================================
# Camera NPZ Saving
# ==============================================================================
def save_camera_npz(records: list, output_dir: str, dataset_name: str):
    """Save all samples' camera parameters as a dict keyed by sample_id.

    Output: cameras/<dataset_name>/cameras.pkl

    Structure:
      {
        "000000000": {
            "view_indices":    np.int32   (S,),   camera indices in model input order
            "master_view_idx": int,                camera index of the master (first) view
            "gt_intrinsics":   np.float32 (S,3,3), GT intrinsics in view_indices order
            "gt_extrinsics":   np.float32 (S,4,4), GT extrinsics (master-relative w2c)
            "pred_extrinsics": np.float32 (S,3,4), predicted extrinsics (rel to cam0)
            "pred_intrinsics": np.float32 (S,3,3), predicted intrinsics
        },
        "000000001": { ... },
        ...
      }
    """
    cam_dir = os.path.join(output_dir, "cameras", dataset_name)
    os.makedirs(cam_dir, exist_ok=True)
    fpath = os.path.join(cam_dir, "cameras.pkl")

    cameras = {}
    for rec in records:
        sid = rec["sample_key"]
        entry = {
            "view_indices":    np.array(rec["view_indices"], dtype=np.int32),
            "master_view_idx": int(rec["master_view_idx"]),
            "gt_intrinsics":   rec["gt_intrinsics"],    # (S, 3, 3)
            "gt_extrinsics":   rec["gt_extrinsics"],    # (S, 4, 4)
        }
        if rec["pred_extrinsics"] is not None:
            entry["pred_extrinsics"] = rec["pred_extrinsics"]      # (S, 3, 4)
        if rec["pred_intrinsics_all"] is not None:
            entry["pred_intrinsics"] = rec["pred_intrinsics_all"]  # (S, 3, 3)
        cameras[sid] = entry

    with open(fpath, "wb") as f:
        pickle.dump(cameras, f)
    logging.info(f"  Saved {len(cameras)} samples' cameras to {fpath}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU multi-view image evaluation for HGGT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Local checkpoint (.pt) or Hugging Face repo id (default: {DEFAULT_HF_REPO})",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(ROOT / "assets" / "mano_v1_2" / "models"),
        help="Path to MANO model directory (containing MANO_RIGHT.pkl)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of multi-view image WebDataset tar files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DATASET",
        help=f"Datasets to evaluate (default: all). Choices: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=518,
        help="Input image size for HGGT (default: 518)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (samples per step, default: 1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max samples per dataset per rank (-1 = all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for result files and optional cameras",
    )
    parser.add_argument(
        "--save_cameras",
        action="store_true",
        default=False,
        help="Save predicted camera params as pickle files per dataset",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path",
    )

    args = parser.parse_args()

    distributed, rank, world_size = init_distributed()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file and is_main_process(rank):
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    if is_main_process(rank):
        logging.info("=" * 80)
        logging.info("HGGT multi-view image evaluation")
        logging.info("=" * 80)
        logging.info(f"  Checkpoint:  {args.checkpoint}")
        logging.info(f"  Data root:   {args.data_root}")
        logging.info(f"  Datasets:    {args.datasets}")
        logging.info(f"  Output dir:  {args.output_dir}")
        logging.info(f"  Device:      {device}  |  World size: {world_size}")
        logging.info(f"  Save cameras: {args.save_cameras}")
        logging.info("")

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process(rank):
        logging.info(f"Loading HGGT from {args.checkpoint}...")
    model = load_hggt(args.checkpoint, img_size=args.img_size, device=device)

    if is_main_process(rank):
        logging.info(f"Loading MANO from {args.mano_model_path}...")
    mano_model = MANO(
        model_path=args.mano_model_path,
        use_pca=False,
        flat_hand_mean=False,
        create_transl=False,
        is_rhand=True,
    ).to(device)
    mano_model.eval()

    all_dataset_metrics = {}

    for dataset_name in args.datasets:
        if is_main_process(rank):
            logging.info("")
            logging.info("=" * 60)
            logging.info(f"  Evaluating: {dataset_name}")
            logging.info("=" * 60)

        barrier(distributed)

        local_records = evaluate_dataset(
            dataset_name=dataset_name,
            model=model,
            mano_model=mano_model,
            args=args,
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            dtype=dtype,
        )

        barrier(distributed)

        all_records = all_gather_records(local_records, distributed)

        if is_main_process(rank):
            logging.info(f"Total samples gathered for {dataset_name}: {len(all_records)}")

            metrics = compute_metrics_for_records(all_records)
            all_dataset_metrics[dataset_name] = metrics

            result_str = format_metrics(dataset_name, args.checkpoint, metrics)
            logging.info("\n" + result_str)

            result_file = os.path.join(args.output_dir, f"{dataset_name}_results.txt")
            with open(result_file, "w") as f:
                f.write(result_str)
            logging.info(f"  Results saved to: {result_file}")

            if args.save_cameras:
                logging.info(f"  Saving cameras for {dataset_name}...")
                save_camera_npz(all_records, args.output_dir, dataset_name)

        barrier(distributed)

    if is_main_process(rank) and all_dataset_metrics:
        logging.info("")
        logging.info("=" * 80)
        logging.info("COMBINED SUMMARY (all datasets, MPJPE in mm)")
        logging.info("=" * 80)
        header = (
            f"{'Dataset':<12} {'MPJPE':>8} {'PA-MPJPE':>10} {'RR-MPJPE':>10} "
            f"{'CA-MPJPE':>10} {'CA-PA':>8} {'CA-RR':>8} {'n':>8}"
        )
        logging.info(header)
        logging.info("-" * len(header))

        summary_lines = [header, "-" * len(header)]
        for ds, m in all_dataset_metrics.items():
            row = (
                f"{ds:<12} {m['mpjpe']:8.2f} {m['pa_mpjpe']:10.2f} {m['rr_mpjpe']:10.2f} "
                f"{m['ca_mpjpe']:10.2f} {m['ca_pa_mpjpe']:8.2f} {m['ca_rr_mpjpe']:8.2f} "
                f"{m['n_samples']:8d}"
            )
            logging.info(row)
            summary_lines.append(row)

        summary_file = os.path.join(args.output_dir, "all_results_summary.txt")
        with open(summary_file, "w") as f:
            f.write("HGGT multi-view image evaluation Summary\n")
            f.write(f"Checkpoint: {args.checkpoint}\n\n")
            f.write("\n".join(summary_lines) + "\n")
        logging.info(f"\nSummary saved to: {summary_file}")

    barrier(distributed)
    if distributed:
        dist.destroy_process_group()

    if is_main_process(rank):
        logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
