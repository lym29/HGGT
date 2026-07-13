# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hggt.layers import Mlp
from hggt.layers.block import Block
from hggt.heads.head_act import activate_pose


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block where queries attend to key-value pairs.

    Optionally applies per-head QK LayerNorm (not L2-normalize) on Q and K,
    matching the VGGT Attention design. This stabilizes attention under bf16
    when the block is stacked many times.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        # Separate projections for queries and key-values
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Pre-attention LayerNorms (pre-LN style)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        # Per-head QK normalization — mirrors VGGT Attention: norm_layer(head_dim)
        # elementwise_affine=True (default) matches the original implementation exactly
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=proj_drop)

        # Layer scale (optional)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, N_q, C] - tokens to be updated
            keys_values: [B, N_kv, C] - tokens to attend to

        Returns:
            updated_queries: [B, N_q, C]
        """
        B, N_q, C = queries.shape
        N_kv = keys_values.shape[1]

        # Pre-LN normalization
        q = self.norm_q(queries)
        kv = self.norm_kv(keys_values)

        # Project to Q, K, V
        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # [B, H, N_q, D]
        kv = self.kv_proj(kv).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [2, B, H, N_kv, D]
        k, v = kv[0], kv[1]  # Each: [B, H, N_kv, D]

        # Per-head QK LayerNorm for numerical stability under bf16
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )  # [B, H, N_q, D]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, C)  # [B, N_q, C]
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        # Residual connection with layer scale
        queries = queries + self.ls1(attn_output)

        # MLP with residual
        queries = queries + self.ls2(self.mlp(self.norm_mlp(queries)))

        return queries


class LayerScale(nn.Module):
    """Layer scale module for stabilizing training."""
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MANOHead(nn.Module):
    """
    Multi-view MANO Head with Cross-Attention among Multiple Token Types.
    
    Token roles:
    
    1. View tokens from aggregated_tokens_list[-1] (used as keys/values):
       - camera_tokens: one per view (also used as queries; updated and returned)
       - register_tokens: num_register_tokens per view (typically 4; KV only)
       - patch_tokens: image features per view (KV only)
    
    2. Hand parameter tokens (learnable; used as both queries and keys/values):
       - hand_pose_token: hand pose (48 parameters)
       - shape_token: hand shape (10 parameters)
       - transl_token: hand translation (3 parameters)
    
    Cross-attention updates only queries = camera tokens + hand tokens; all view
    tokens plus hand tokens provide keys/values for context. Updated camera tokens
    are written back into the token list for camera_head; hand tokens are decoded
    to MANO parameters.
    
    Args:
        dim_in (int): Input dimension from the backbone. Default: 2048.
        trunk_depth (int): Number of transformer layers for refinement. Default: 4.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (int): MLP expansion ratio. Default: 4.
        init_values (float): Layer scale initialization value. Default: 0.01.
        qk_norm (bool): Per-head QK LayerNorm in CrossAttentionBlock. Default: True.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,  
        num_heads: int = 16,    
        mlp_ratio: int = 4,    
        init_values: float = 0.01,
        qk_norm: bool = True,
    ):
        super().__init__()
        
        # MANO parameter dimensions
        self.hand_pose_dim = 48              # 16 joints * 3 rotation parameters
        self.shape_dim = 10                  # PCA shape parameters (beta)
        self.translation_dim = 3             # translation parameters
        
        self.trunk_depth = trunk_depth
        self.dim_in = dim_in

        # Token normalization for input tokens
        self.token_norm = nn.LayerNorm(dim_in)
        
        # ==============================================================
        # Learnable tokens for MANO parameters (initialized small)
        # Camera tokens are extracted from aggregated_tokens_list, not learned here
        # ==============================================================
        self.hand_pose_token = nn.Parameter(torch.randn(1, 1, self.hand_pose_dim) * 0.02)
        self.shape_token = nn.Parameter(torch.randn(1, 1, self.shape_dim) * 0.02)
        self.transl_token = nn.Parameter(torch.randn(1, 1, self.translation_dim) * 0.02)
        
        # ==============================================================
        # Embedding layers to project MANO tokens to unified dimension (dim_in)
        # ==============================================================
        self.embed_hand_pose = nn.Linear(self.hand_pose_dim, dim_in)
        self.embed_shape = nn.Linear(self.shape_dim, dim_in)
        self.embed_transl = nn.Linear(self.translation_dim, dim_in)
        
        # ==============================================================
        # Modulation modules (similar to camera_head.py)
        # Generate shift, scale, and gate parameters for adaptive layer norm
        # ==============================================================
        self.hand_pose_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.shape_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.transl_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        
        # Adaptive layer normalization without affine parameters
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        
        # ==============================================================
        # Cross-attention transformer trunk (optimized)
        # Only hand tokens + camera tokens are queries (updated)
        # All tokens are keys/values (for context)
        # ==============================================================
        self.cross_attn_trunk = nn.ModuleList(
            [
                CrossAttentionBlock(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values, qk_norm=qk_norm)
                for _ in range(trunk_depth)
            ]
        )
        
        # Output normalization after cross-attention
        self.cross_attn_norm = nn.LayerNorm(dim_in)
        
        # ==============================================================
        # Projection heads to decode MANO tokens back to parameter space
        # ==============================================================
        self.hand_pose_head = Mlp(in_features=dim_in, hidden_features=dim_in // 4, out_features=self.hand_pose_dim, drop=0)
        self.shape_head = Mlp(in_features=dim_in, hidden_features=dim_in // 4, out_features=self.shape_dim, drop=0)
        self.transl_head = Mlp(in_features=dim_in, hidden_features=dim_in // 4, out_features=self.translation_dim, drop=0)

    def forward(
        self, 
        aggregated_tokens_list: list,
        num_iterations: int = 4,
    ):
        """
        Forward pass with iterative cross-attention refinement.
        
        Process:
        1. Extract all tokens from aggregated_tokens_list[-1]
           (camera, register, and patch tokens per view)
        2. Iteratively refine via cross-attention: queries are camera + hand tokens;
           keys/values are all view tokens + hand tokens
        3. Write updated camera tokens back into the token list for camera_head
        4. Decode hand tokens to MANO parameters at each iteration

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction. 
                Shape: [B, S, N, C] where S is number of views, 
                N = 1 (camera) + num_register_tokens + num_patches
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            tuple: (mano_params_list, updated_aggregated_tokens_list, refined_hand_tokens)
                - mano_params_list: List of MANO parameter dicts from each iteration
                - updated_aggregated_tokens_list: Token list with updated camera tokens
                - refined_hand_tokens: Refined hand tokens [B, 3, C] containing [hand_pose, shape, transl]
        """
        # ==============================================================
        # Step 1: Extract all tokens from the last aggregator layer
        # ==============================================================
        tokens = aggregated_tokens_list[-1]  # [B, S, N, C] where S is number of views
        tokens = self.token_norm(tokens)
        B, S, N, C = tokens.shape
        
        # Flatten all tokens from all views: [B, S, N, C] -> [B, S*N, C]
        # This includes camera tokens, register tokens, and patch tokens
        all_view_tokens = tokens.reshape(B, S * N, C)
        
        # ==============================================================
        # Step 2: Perform iterative cross-attention refinement
        # ==============================================================
        mano_params_list, updated_view_tokens, refined_hand_tokens = self.trunk_fn(all_view_tokens, num_iterations, B, S, N, C)
        
        # ==============================================================
        # Step 3: Reconstruct updated aggregated_tokens_list
        # ==============================================================
        # Reshape updated tokens back to [B, S, N, C]
        updated_tokens = updated_view_tokens.reshape(B, S, N, C)
        
        # Create updated aggregated_tokens_list by replacing the last element
        updated_aggregated_tokens_list = aggregated_tokens_list[:-1] + [updated_tokens]
        
        return mano_params_list, updated_aggregated_tokens_list, refined_hand_tokens

    def trunk_fn(self, all_view_tokens: torch.Tensor, num_iterations: int, B: int, S: int, N: int, C: int) -> tuple:
        """
        Iteratively refine MANO predictions using cross-attention.
        
        Per iteration:
        1. Embed (and modulate) hand tokens from learned init or previous predictions
        2. Build queries = [camera tokens (S), hand_pose, shape, transl]
           and keys/values = [all_view_tokens (M), hand_pose, shape, transl]
        3. Run cross-attention trunk (only queries are updated)
        4. Write updated camera tokens back into all_view_tokens
        5. Decode hand tokens to MANO parameter deltas and accumulate

        Token Flow:
        Flattened view tokens: [B, S*N, C]
        where N = 1 (camera) + 4 (register) + P (patches)
            ↓
        For each iteration:
        ├─ Queries:  [camera(S), hand_pose(1), shape(1), transl(1)]
        ├─ KV:       [all_view_tokens(M), hand_pose(1), shape(1), transl(1)]
        ├─ Cross-attention updates queries only
        ├─ Write updated cameras back into all_view_tokens
        └─ Decode hand tokens → accumulate MANO deltas
            ↓
        Output: (list of MANO dicts, updated view tokens [B, S*N, C], refined hand tokens)
        
        Args:
            all_view_tokens (torch.Tensor): All tokens from all views [B, M, C]
                where M = S*N (camera, register, and patch tokens)
            num_iterations (int): Number of refinement iterations.
            B (int): Batch size
            S (int): Number of views
            N (int): Number of tokens per view (from VGGT aggregator)
            C (int): Channel dimension

        Returns:
            tuple: (mano_dict_list, updated_view_tokens, refined_hand_tokens)
                - mano_dict_list: List of MANO parameter dicts from each iteration
                - updated_view_tokens: View tokens with updated camera tokens [B, S*N, C]
                - refined_hand_tokens: Refined hand tokens [B, 3, C] from last iteration
        """
        # Initialize MANO predictions (None for first iteration)
        pred_hand_pose = None
        pred_shape = None 
        pred_transl = None
        
        mano_dict_list = []

        for iteration_idx in range(num_iterations):
            # ==============================================================
            # Step 1: Prepare embedded MANO tokens for this iteration
            # ==============================================================
            if pred_hand_pose is None:
                # First iteration: use learned initial tokens
                hand_pose_input = self.embed_hand_pose(self.hand_pose_token.expand(B, 1, -1))
                shape_input = self.embed_shape(self.shape_token.expand(B, 1, -1))
                transl_input = self.embed_transl(self.transl_token.expand(B, 1, -1))
            else:
                # Subsequent iterations: use previous predictions (detached to save memory)
                pred_hand_pose = pred_hand_pose.detach()
                pred_shape = pred_shape.detach()
                pred_transl = pred_transl.detach()
                
                hand_pose_input = self.embed_hand_pose(pred_hand_pose.unsqueeze(1))
                shape_input = self.embed_shape(pred_shape.unsqueeze(1))
                transl_input = self.embed_transl(pred_transl.unsqueeze(1))
            
            # ==============================================================
            # Step 1.5: Apply modulation to hand tokens (similar to camera_head.py)
            # Generate shift, scale, and gate parameters for adaptive layer norm
            # ==============================================================
            # Modulate hand pose token
            shift_hp, scale_hp, gate_hp = self.hand_pose_modulation(hand_pose_input).chunk(3, dim=-1)
            hand_pose_modulated = gate_hp * modulate(self.adaln_norm(hand_pose_input), shift_hp, scale_hp)
            hand_pose_modulated = hand_pose_modulated + hand_pose_input  # Residual connection
            
            # Modulate shape token
            shift_sh, scale_sh, gate_sh = self.shape_modulation(shape_input).chunk(3, dim=-1)
            shape_modulated = gate_sh * modulate(self.adaln_norm(shape_input), shift_sh, scale_sh)
            shape_modulated = shape_modulated + shape_input  # Residual connection
            
            # Modulate translation token
            shift_tr, scale_tr, gate_tr = self.transl_modulation(transl_input).chunk(3, dim=-1)
            transl_modulated = gate_tr * modulate(self.adaln_norm(transl_input), shift_tr, scale_tr)
            transl_modulated = transl_modulated + transl_input  # Residual connection

            # ==============================================================
            # Step 2: Extract camera tokens (first token from each view)
            # all_view_tokens: [B, S*N, C] where N tokens per view
            # Camera token positions: 0, N, 2N, ..., (S-1)*N
            # ==============================================================
            M = all_view_tokens.shape[1]  # S*N
            camera_indices = torch.arange(0, M, N, device=all_view_tokens.device)  # [S]
            camera_tokens = all_view_tokens[:, camera_indices, :]  # [B, S, C]
            
            # ==============================================================
            # Step 3: Prepare queries (only tokens to be updated)
            # Queries: camera tokens + hand tokens = S + 3 tokens
            # Keys/Values: all view tokens + hand tokens = M + 3 tokens
            # ==============================================================
            queries = torch.cat([
                camera_tokens,         # [B, S, C] - camera tokens to be updated
                hand_pose_modulated,   # [B, 1, C] - hand pose token
                shape_modulated,       # [B, 1, C] - shape token
                transl_modulated,      # [B, 1, C] - translation token
            ], dim=1)  # [B, S+3, C]
            
            keys_values = torch.cat([
                all_view_tokens,       # [B, M, C] - all tokens from all views
                hand_pose_modulated,   # [B, 1, C]
                shape_modulated,       # [B, 1, C]
                transl_modulated,      # [B, 1, C]
            ], dim=1)  # [B, M+3, C]
            
            # ==============================================================
            # Step 4: Cross-attention - queries attend to keys_values
            # Only camera and hand tokens are updated
            # ==============================================================
            for layer in self.cross_attn_trunk:
                queries = layer(queries, keys_values)  # [B, S+3, C]
            
            queries = self.cross_attn_norm(queries)
            
            # ==============================================================
            # Step 5: Extract updated tokens
            # ==============================================================
            updated_camera_tokens = queries[:, :S, :]         # [B, S, C]
            hand_pose_output = queries[:, S, :]               # [B, C]
            shape_output = queries[:, S+1, :]                 # [B, C]
            transl_output = queries[:, S+2, :]                # [B, C]
            
            # ==============================================================
            # Step 6: Write updated camera tokens back into all_view_tokens
            # Register and patch tokens are unchanged (they were KV-only)
            # ==============================================================
            updated_view_tokens = all_view_tokens.clone()
            updated_view_tokens[:, camera_indices, :] = updated_camera_tokens
            all_view_tokens = updated_view_tokens[:, :S*N, :]
            
            # ==============================================================
            # Step 7: Decode MANO tokens to parameter space and compute deltas
            # ==============================================================
            hand_pose_delta = self.hand_pose_head(hand_pose_output)  # [B, 48]
            shape_delta = self.shape_head(shape_output)              # [B, 10]
            transl_delta = self.transl_head(transl_output)           # [B, 3]

            # ==============================================================
            # Step 8: Update MANO predictions (accumulate deltas)
            # ==============================================================
            if pred_hand_pose is None:
                # First iteration: initialize predictions
                pred_hand_pose = hand_pose_delta
                pred_shape = shape_delta
                pred_transl = transl_delta
            else:
                # Subsequent iterations: accumulate deltas
                pred_hand_pose = pred_hand_pose + hand_pose_delta
                pred_shape = pred_shape + shape_delta
                pred_transl = pred_transl + transl_delta

            # ==============================================================
            # Step 9: Store MANO predictions for this iteration
            # ==============================================================
            mano_dict = {
                "hand_pose": pred_hand_pose,
                "betas": pred_shape,
                "transl": pred_transl
            }
            mano_dict_list.append(mano_dict)

        # Final refined hand tokens [B, 3, C] from the last iteration
        refined_hand_tokens = torch.stack([
            hand_pose_output,  # [B, C]
            shape_output,      # [B, C]
            transl_output      # [B, C]
        ], dim=1)  # [B, 3, C]
        
        # Return MANO parameters and updated view tokens (camera tokens refined for camera_head)
        return mano_dict_list, all_view_tokens, refined_hand_tokens




def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
