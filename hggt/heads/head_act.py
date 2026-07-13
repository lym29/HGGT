# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


def activate_head_multi_depth(out, num_layers=4, depth_activation="exp", opacity_activation="sigmoid", conf_activation=None, depth_scale=0.001):
    """
    Process network output for multi-layered depth prediction.
    Each layer has [depth, opacity] pair, optionally with [depth, opacity, confidence].

    Args:
        out: Network output tensor (B, C, H, W) where C = num_layers * 2 or num_layers * 3
        num_layers: Number of depth layers
        depth_activation: Activation type for depth values (e.g., "exp", "relu", "linear", "softplus_scaled")
        opacity_activation: Activation type for opacity values (e.g., "sigmoid", "expp1")
        conf_activation: Activation type for confidence values (e.g., "expp1", "sigmoid", None)
                        If None, confidence is not predicted.
        depth_scale: Scale factor for scaled activations (e.g., softplus_scaled). Default: 0.01

    Returns:
        Tuple of (multi_depth, multi_opacity) or (multi_depth, multi_opacity, multi_conf)
        - multi_depth: (B, H, W, num_layers) - depth values for each layer
        - multi_opacity: (B, H, W, num_layers) - opacity values for each layer (0-1)
        - multi_conf: (B, H, W, num_layers) - confidence values for each layer (optional)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B, H, W, C

    # Check if confidence is predicted
    has_confidence = conf_activation is not None
    channels_per_layer = 3 if has_confidence else 2

    # Reshape to separate layers: (B, H, W, num_layers, channels_per_layer)
    fmap = fmap.view(*fmap.shape[:3], num_layers, channels_per_layer)

    # Split depth and opacity for each layer
    depth_logits = fmap[..., 0]  # (B, H, W, num_layers)
    opacity_logits = fmap[..., 1]  # (B, H, W, num_layers)

    # Apply depth activation
    if depth_activation == "exp":
        multi_depth = torch.exp(depth_logits)
    elif depth_activation == "relu":
        multi_depth = F.relu(depth_logits)
    elif depth_activation == "softplus":
        # Softplus activation ensures positive values: softplus(x) = log(1 + exp(x))
        # Good for delta depth (incremental depth) that must be > 0
        multi_depth = F.softplus(depth_logits)
    elif depth_activation == "softplus_scaled":
        multi_depth = F.softplus(depth_logits) * depth_scale
    elif depth_activation == "linear":
        multi_depth = depth_logits
    elif depth_activation == "inv_log":
        multi_depth = inverse_log_transform(depth_logits)
    else:
        raise ValueError(f"Unknown depth_activation: {depth_activation}")

    # Apply opacity activation
    if opacity_activation == "sigmoid":
        multi_opacity = torch.sigmoid(opacity_logits)
    else:
        raise ValueError(f"Unknown opacity_activation: {opacity_activation}")

    # Apply confidence activation if enabled
    if has_confidence:
        conf_logits = fmap[..., 2]  # (B, H, W, num_layers)

        if conf_activation == "expp1":
            multi_conf = 1 + conf_logits.exp()
        elif conf_activation == "expp0":
            multi_conf = conf_logits.exp()
        elif conf_activation == "sigmoid":
            multi_conf = torch.sigmoid(conf_logits)
        else:
            raise ValueError(f"Unknown conf_activation: {conf_activation}")

        return multi_depth, multi_opacity, multi_conf
    else:
        return multi_depth, multi_opacity


def activate_head_pointmap(out, num_layers=4, point_activation="tanh_scaled", normal_activation="tanh", opacity_activation="sigmoid", point_scale=2.0):
    """
    Process network output for multi-layered object point map and normal map prediction.
    Each layer has [xyz, normal_xyz, opacity] = 7 channels.

    Args:
        out: Network output tensor (B, C, H, W) where C = num_layers * 7
        num_layers: Number of point map layers
        point_activation: Activation type for xyz coordinates (e.g., "tanh_scaled", "linear", "inv_log")
        normal_activation: Activation type for normal vectors (e.g., "tanh", "linear")
        opacity_activation: Activation type for opacity values (e.g., "sigmoid")
        point_scale: Scale factor for point coordinates (default: 2.0 for [-2, 2] range)

    Returns:
        Tuple of (multi_pointmap, multi_normalmap, multi_opacity)
        - multi_pointmap: (B, H, W, num_layers, 3) - xyz coordinates for each layer
        - multi_normalmap: (B, H, W, num_layers, 3) - normal vectors for each layer (unit vectors)
        - multi_opacity: (B, H, W, num_layers) - opacity values for each layer (0-1)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B, H, W, C

    channels_per_layer = 7  # xyz (3) + normal_xyz (3) + opacity (1)

    # Reshape to separate layers: (B, H, W, num_layers, channels_per_layer)
    fmap = fmap.view(*fmap.shape[:3], num_layers, channels_per_layer)

    # Split into components
    point_logits = fmap[..., :3]  # (B, H, W, num_layers, 3)
    normal_logits = fmap[..., 3:6]  # (B, H, W, num_layers, 3)
    opacity_logits = fmap[..., 6]  # (B, H, W, num_layers)

    # Apply point activation
    if point_activation == "tanh_scaled":
        multi_pointmap = torch.tanh(point_logits) * point_scale
    elif point_activation == "tanh":
        multi_pointmap = torch.tanh(point_logits)
    elif point_activation == "linear":
        multi_pointmap = point_logits
    elif point_activation == "inv_log":
        multi_pointmap = inverse_log_transform(point_logits)
    elif point_activation == "sigmoid_scaled":
        # Map to [0, 1] then scale to [-point_scale, point_scale]
        multi_pointmap = (torch.sigmoid(point_logits) - 0.5) * 2 * point_scale
    else:
        raise ValueError(f"Unknown point_activation: {point_activation}")

    # Apply normal activation and normalize to unit vectors
    if normal_activation == "tanh":
        multi_normalmap = torch.tanh(normal_logits)
    elif normal_activation == "linear":
        multi_normalmap = normal_logits
    else:
        raise ValueError(f"Unknown normal_activation: {normal_activation}")

    # Normalize normals to unit vectors
    multi_normalmap = F.normalize(multi_normalmap, dim=-1, eps=1e-8)

    # Apply opacity activation
    if opacity_activation == "sigmoid":
        multi_opacity = torch.sigmoid(opacity_logits)
    else:
        raise ValueError(f"Unknown opacity_activation: {opacity_activation}")

    return multi_pointmap, multi_normalmap, multi_opacity
