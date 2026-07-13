# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from hggt.models.aggregator import Aggregator
from hggt.heads.camera_head import CameraHead
from hggt.heads.dpt_head import DPTHead

from hggt.heads.mano_head_crossattn import MANOHead



class HGGT(nn.Module, PyTorchModelHubMixin):
    """
    HGGT (Hand Geometry Grounded Transformer) for multi-view hand mesh reconstruction.
    
    Built on the VGGT backbone.
    """
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_mano=True,
        mano_qk_norm: bool = False,   # QK-norm in cross-attn trunk; set False for old ckpts
    ):
        """
        Initialize HGGT model.

        Args:
            img_size (int): Input image size. Default: 518.
            patch_size (int): Vision transformer patch size. Default: 14.
            embed_dim (int): Embedding dimension. Default: 1024.
            enable_camera (bool): Enable camera head. Default: True.
            enable_point (bool): Enable point head for 3D points. Default: True.
            enable_depth (bool): Enable original DPT depth head. Default: True.
            enable_mano (bool): Enable MANO head for hand pose. Default: True.
            mano_qk_norm (bool): QK-norm in cross-attn trunk. Default: False.
        """
        super().__init__()
        
        # Store configuration for HOI-specific functionality
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Initialize the backbone aggregator
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # ========== Prediction Heads ==========
        # Camera pose estimation
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        
        # 3D point reconstruction  
        self.point_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=4, 
            activation="inv_log", 
            conf_activation="expp1"
        ) if enable_point else None
        
        # Single-layer depth estimation
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=2, 
            activation="exp", 
            conf_activation="expp1"
        ) if enable_depth else None
        
        # Hand pose estimation (MANO parameters)
        self.mano_head = MANOHead(dim_in=2 * embed_dim, qk_norm=mano_qk_norm) if enable_mano else None

    def forward(self, images: torch.Tensor, **kwargs):
        """
        Forward pass of the HGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            **kwargs: Additional arguments for specific heads (e.g., diffusion inference steps)

        Returns:
            dict: A dictionary containing predictions from various heads:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9]
                - depth (torch.Tensor): Original DPT depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for original depth predictions
                - world_points (torch.Tensor): 3D world coordinates for each pixel
                - world_points_conf (torch.Tensor): Confidence scores for world points
                - images (torch.Tensor): Original input images (preserved for visualization)

                MANO-specific outputs:
                - mano_params (torch.Tensor): Predicted MANO parameters
                - mano_params_list (list): List of MANO parameters from all layers
        """        
        # Handle batch dimension
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # Get aggregated tokens from the backbone
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        
        # Process with mixed precision for efficiency
        with torch.cuda.amp.autocast(enabled=False):
            # IMPORTANT: Run mano_head first to update camera tokens via cross-attention
            if self.mano_head is not None:
                mano_params_list, aggregated_tokens_list, _ = self.mano_head(
                    aggregated_tokens_list
                )
                predictions["mano_params"] = mano_params_list[-1]
                predictions["mano_params_list"] = mano_params_list
            
            # Use updated tokens from mano_head for camera_head and other heads
            # Camera tokens have been refined via cross-attention with hand tokens
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # Store images for visualization during inference
        if not self.training:
            predictions["images"] = images

        return predictions
