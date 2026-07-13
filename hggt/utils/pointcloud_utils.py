"""
Utilities for point cloud processing and normal estimation using WNNC.
Used for converting multi-layer depth maps to point clouds with normals.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# Add WNNC to path
WNNC_PATH = os.path.join(os.path.dirname(__file__), "../../../third_party/WNNC")
if os.path.exists(WNNC_PATH) and WNNC_PATH not in sys.path:
    sys.path.insert(0, WNNC_PATH)

from hggt.utils.geometry import unproject_depth_map_to_point_map


def estimate_normals_wnnc(
    points: torch.Tensor,
    width_config: str = 'l1',
    iterations: int = 40,
) -> torch.Tensor:
    """
    Estimate normals using WNNC (Winding Number Normal Consistency).

    This is a robust method for normal estimation that works well with noisy point clouds.
    Reference: https://github.com/jsnln/WNNC

    Args:
        points: Point cloud tensor of shape (N, 3), should be on GPU for best performance
        width_config: Width configuration for WNNC. Options:
            - 'l0': [0.002, 0.016] - noise level 0, for uniform noise-free points
            - 'l1': [0.01, 0.04] - noise level 1, for real scans (default)
            - 'l2': [0.02, 0.08] - noise level 2, for sigma=0.25%
            - 'l3': [0.03, 0.12] - noise level 3, for sigma=0.5%
            - 'l4': [0.04, 0.16] - noise level 4, for sigma=1%
            - 'l5': [0.05, 0.2] - noise level 5, for sparse points and 3D sketches
        iterations: Number of iterations for WNNC optimization. Default: 40

    Returns:
        Estimated normals of shape (N, 3) - unit vectors
    """
    device = points.device

    # Normalize points to [-1, 1] range for WNNC
    bbox_scale = 1.1
    bbox_center = (points.min(0)[0] + points.max(0)[0]) / 2.0
    bbox_len = (points.max(0)[0] - points.min(0)[0]).max()
    points_normalized = (points - bbox_center) * (2.0 / (bbox_len * bbox_scale))

    # Initialize normals and other variables
    normals = torch.zeros_like(points_normalized).contiguous()
    b = torch.ones(points_normalized.shape[0], 1, device=device) * 0.5
    widths = torch.ones(points_normalized.shape[0], device=device)

    # Width presets
    preset_widths = {
        'l0': [0.002, 0.016],
        'l1': [0.01, 0.04],
        'l2': [0.02, 0.08],
        'l3': [0.03, 0.12],
        'l4': [0.04, 0.16],
        'l5': [0.05, 0.2],
    }
    wsmin, wsmax = preset_widths.get(width_config, [0.01, 0.04])

    # Import WNNC
    try:
        import wn_treecode
    except ImportError:
        raise ImportError(
            "WNNC (wn_treecode) not found. Please install it from: "
            "https://github.com/jsnln/WNNC\n"
            "Run: cd third_party/WNNC/ext && pip install -e ."
        )

    # Build tree for winding number computation
    wn_func = wn_treecode.WindingNumberTreecode(points_normalized)

    # WNNC optimization loop
    with torch.no_grad():
        for i in range(iterations):
            width_scale = wsmin + ((iterations - 1 - i) / (iterations - 1)) * (wsmax - wsmin)

            # Gradient step
            A_mu = wn_func.forward_A(normals, widths * width_scale)
            AT_A_mu = wn_func.forward_AT(A_mu, widths * width_scale)
            r = wn_func.forward_AT(b, widths * width_scale) - AT_A_mu
            A_r = wn_func.forward_A(r, widths * width_scale)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            normals = normals + alpha * r

            # WNNC step
            out_normals = wn_func.forward_G(normals, widths * width_scale)

            # Rescale and normalize
            out_normals = F.normalize(out_normals, dim=-1).contiguous()
            normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
            normals = out_normals.clone() * normals_len

    # Final normalization
    out_normals = F.normalize(normals, dim=-1)

    return out_normals


def estimate_normals_from_merged_pointcloud(
    points: np.ndarray,
    width_config: str = 'l1',
    iterations: int = 40,
) -> np.ndarray:
    """
    Estimate normals from a merged point cloud using WNNC.

    This function is designed to estimate normals from a complete point cloud that has been
    merged from multiple depth layers. This provides more context for accurate normal estimation
    compared to estimating normals per-layer independently.

    Args:
        points: Point cloud of shape (N, 3)
        width_config: WNNC width configuration. Default: 'l1' (for real scans)
        iterations: Number of WNNC iterations. Default: 40

    Returns:
        Estimated normals of shape (N, 3) - unit vectors
    """
    # Convert to torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_torch = torch.from_numpy(points).float().to(device)

    # Use WNNC for robust normal estimation
    normals_torch = estimate_normals_wnnc(
        points_torch,
        width_config=width_config,
        iterations=iterations,
    )

    # Convert back to numpy
    normals = normals_torch.cpu().numpy()

    return normals


def unproject_multi_depth_to_pointcloud_with_normals(
    multi_depth: np.ndarray,
    opacity: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    opacity_threshold: float = 0.5,
    width_config: str = 'l1',
    wnnc_iterations: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject multi-layer depth maps to point cloud and estimate normals using WNNC.

    This function:
    1. Unprojects each depth layer to 3D points
    2. Merges all valid points from all layers into a single point cloud
    3. Estimates normals for the merged point cloud using WNNC
    4. Maps normals back to the original multi-layer structure

    This approach provides better normal estimation than per-layer estimation because
    it uses the complete geometric context from all layers.

    Args:
        multi_depth: Multi-layer depth of shape (H, W, L)
        opacity: Opacity values of shape (H, W, L)
        extrinsics: Camera extrinsic matrix of shape (3, 4)
        intrinsics: Camera intrinsic matrix of shape (3, 3)
        opacity_threshold: Threshold for masking invalid points. Default: 0.5
        width_config: WNNC width configuration. Default: 'l1'
        wnnc_iterations: Number of WNNC iterations. Default: 40

    Returns:
        Tuple of (pointmap, normalmap):
            - pointmap: Multi-layer point map of shape (H, W, L, 3)
            - normalmap: Multi-layer normal map of shape (H, W, L, 3)
    """
    H, W, L = multi_depth.shape
    pointmap = np.zeros((H, W, L, 3), dtype=np.float32)
    normalmap = np.zeros((H, W, L, 3), dtype=np.float32)

    # Step 1: Unproject all layers to 3D points
    for layer_idx in range(L):
        depth_layer = multi_depth[:, :, layer_idx]  # (H, W)

        # Unproject to world coordinates (add batch dimension)
        world_points = unproject_depth_map_to_point_map(
            depth_map=depth_layer[None, :, :, None],  # (1, H, W, 1)
            extrinsics_cam=extrinsics[None],  # (1, 3, 4)
            intrinsics_cam=intrinsics[None],  # (1, 3, 3)
        )  # (1, H, W, 3)

        pointmap[:, :, layer_idx, :] = world_points[0]

    # Step 2: Merge all valid points from all layers
    valid_mask = opacity > opacity_threshold  # (H, W, L)

    # Flatten and collect all valid points with their indices
    all_points = []
    point_to_layer_map = []  # Maps each point back to (h, w, layer)

    for layer_idx in range(L):
        layer_valid_mask = valid_mask[:, :, layer_idx]
        layer_points = pointmap[:, :, layer_idx, :]

        # Get valid points for this layer
        valid_points = layer_points[layer_valid_mask]

        if len(valid_points) > 0:
            all_points.append(valid_points)

            # Store mapping: (h, w, layer) for each valid point
            h_indices, w_indices = np.where(layer_valid_mask)
            for h, w in zip(h_indices, w_indices):
                point_to_layer_map.append((h, w, layer_idx))

    # Check if we have enough points for normal estimation
    if len(all_points) == 0:
        print("Warning: No valid points found for normal estimation")
        return pointmap, normalmap

    # Merge all points into single array
    merged_points = np.concatenate(all_points, axis=0)  # (N_total, 3)

    # Check minimum points requirement
    if len(merged_points) < 10:
        print(f"Warning: Only {len(merged_points)} points available, too few for normal estimation")
        return pointmap, normalmap

    # Step 3: Estimate normals for the merged point cloud using WNNC
    try:
        merged_normals = estimate_normals_from_merged_pointcloud(
            points=merged_points,
            width_config=width_config,
            iterations=wnnc_iterations,
        )  # (N_total, 3)
    except Exception as e:
        print(f"Warning: Failed to estimate normals: {e}")
        import traceback
        traceback.print_exc()
        return pointmap, normalmap

    # Step 4: Map normals back to multi-layer structure
    for point_idx, (h, w, layer_idx) in enumerate(point_to_layer_map):
        normalmap[h, w, layer_idx, :] = merged_normals[point_idx]

    # Step 5: Mask invalid points (where opacity is low)
    mask = opacity < opacity_threshold
    pointmap[mask] = 0.0
    normalmap[mask] = 0.0

    return pointmap, normalmap


def unproject_multi_depth_to_pointmap(
    multi_depth: np.ndarray,
    opacity: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    opacity_threshold: float = 0.5,
) -> np.ndarray:
    """
    Unproject multi-layer depth maps to multi-layer point maps (without normals).

    This is a simpler version that only converts depth to points without normal estimation.

    Args:
        multi_depth: Multi-layer depth of shape (H, W, L)
        opacity: Opacity values of shape (H, W, L)
        extrinsics: Camera extrinsic matrix of shape (3, 4)
        intrinsics: Camera intrinsic matrix of shape (3, 3)
        opacity_threshold: Threshold for masking invalid points

    Returns:
        Multi-layer point map of shape (H, W, L, 3)
    """
    H, W, L = multi_depth.shape
    pointmap = np.zeros((H, W, L, 3), dtype=np.float32)

    # Unproject each layer
    for layer_idx in range(L):
        depth_layer = multi_depth[:, :, layer_idx]  # (H, W)

        # Unproject to world coordinates (add batch dimension)
        world_points = unproject_depth_map_to_point_map(
            depth_map=depth_layer[None, :, :, None],  # (1, H, W, 1)
            extrinsics_cam=extrinsics[None],  # (1, 3, 4)
            intrinsics_cam=intrinsics[None],  # (1, 3, 3)
        )  # (1, H, W, 3)

        pointmap[:, :, layer_idx, :] = world_points[0]

    # Mask invalid points (where opacity is low)
    mask = opacity < opacity_threshold
    pointmap[mask] = 0.0

    return pointmap
