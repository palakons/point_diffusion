import time
import torch
import sys, subprocess
import tensorflow as tf
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from man_ddpm import MANDataset, chamfer_distance
from geomloss import SamplesLoss
from torch_cluster import knn_graph
from pytorch3d.loss import chamfer_distance   as  pt3d_chamfer_distance # Import here for clarity
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
import argparse
# from pytorch3d.loss import chamfer_distance

sys.path.insert(0, "/home/palakons/PointNeXt")
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional

# import F
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_l2_with_attr_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    w_xyz: float = 1.0,
    w_velocity: float = 0.1,
    w_rcs: float = 0.01,
    xyz_slice=slice(0, 3),
    vel_slice=slice(3, 6),
    rcs_slice=slice(6, 7),
    use_greedy: bool = False,
):
    """
    One-to-one matching loss using Hungarian assignment.

    Matching is done using xyz only.
    After matching, compute xyz + velocity + rcs losses on the matched pairs.
    
    Args:
        use_greedy: If True, use fast greedy matching instead of Hungarian (10-50x faster)
                   For N > 200 points, this is highly recommended
    """
    assert pred.dim() == 3 and gt.dim() == 3
    B, Np, Dp = pred.shape
    Bg, Ng, Dg = gt.shape
    assert B == Bg, (pred.shape, gt.shape)
    assert (
        Np == Ng
    ), f"Hungarian matching requires same number of points, got {Np} vs {Ng}"
    assert Dp >= 7 and Dg >= 7

    total_loss = pred.new_zeros(())
    xyz_loss_total = 0.0
    vel_loss_total = 0.0
    rcs_loss_total = 0.0

    # Extract point coordinates and attributes once (avoid repeated slicing)
    pred_xyz = pred[:, :, xyz_slice]  # (B, N, 3)
    pred_vel = pred[:, :, vel_slice]  # (B, N, 3)
    pred_rcs = pred[:, :, rcs_slice]  # (B, N, 1)
    
    gt_xyz = gt[:, :, xyz_slice]      # (B, N, 3)
    gt_vel = gt[:, :, vel_slice]      # (B, N, 3)
    gt_rcs = gt[:, :, rcs_slice]      # (B, N, 1)

    for b in range(B):
        if use_greedy:
            row_ind, col_ind = _greedy_matching_gpu(pred_xyz[b], gt_xyz[b])
        else:
            # Standard Hungarian algorithm
            xyz_cost = torch.cdist(pred_xyz[b], gt_xyz[b], p=2) ** 2
            
            # OPTIMIZATION: Use float32 for assignment if needed
            if xyz_cost.dtype == torch.float64:
                xyz_cost = xyz_cost.float()
            
            # Transfer to CPU only once
            cost_np = xyz_cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            row_ind = torch.as_tensor(row_ind, device=pred.device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)

        # Compute losses on matched pairs
        xyz_l = (pred_xyz[b, row_ind] - gt_xyz[b, col_ind]).pow(2).mean()
        vel_l = (pred_vel[b, row_ind] - gt_vel[b, col_ind]).pow(2).mean()
        rcs_l = (pred_rcs[b, row_ind] - gt_rcs[b, col_ind]).pow(2).mean()

        loss_b = (w_xyz * xyz_l) + (w_velocity * vel_l) + (w_rcs * rcs_l)
        total_loss = total_loss + loss_b

        # Avoid repeated CPU transfers - accumulate as tensors first
        xyz_loss_total += xyz_l.detach().item()
        vel_loss_total += vel_l.detach().item()
        rcs_loss_total += rcs_l.detach().item()

    total_loss = total_loss / B
    details = {
        "xyz": xyz_loss_total / B,
        "velocity": vel_loss_total / B,
        "rcs": rcs_loss_total / B,
    }
    return total_loss, details


def _greedy_matching_gpu(pred_xyz, gt_xyz):
    """
    PROPER greedy matching that enforces one-to-one assignment.
    
    Algorithm:
    1. Compute all pairwise distances
    2. Repeatedly pick the smallest distance pair
    3. Remove both matched points
    4. Repeat until all matched
    
    This is O(N^3) in worst case (same as Hungarian) BUT:
    - Simpler implementation
    - Still GPU-native
    - More intuitive
    
    Returns one-to-one assignment like Hungarian.
    """
    N = pred_xyz.shape[0]
    
    # Compute all pairwise distances: (N, N)
    dist = torch.cdist(pred_xyz, gt_xyz, p=2)  # L2 distance
    
    row_ind = []
    col_ind = []
    
    # Track which points are already matched
    unmatched_pred = set(range(N))
    unmatched_gt = set(range(N))
    
    # O(N^2) greedy matching: sort all distances, assign pairs greedily
    # 1. Flatten distance matrix and sort
    pairs = [(i, j) for i in range(N) for j in range(N)]
    flat_dist = dist.flatten()
    sorted_indices = torch.argsort(flat_dist)
    assigned_pred = set()
    assigned_gt = set()
    row_ind = []
    col_ind = []
    for idx in sorted_indices:
        i = idx // N
        j = idx % N
        if i not in assigned_pred and j not in assigned_gt:
            row_ind.append(i)
            col_ind.append(j)
            assigned_pred.add(i)
            assigned_gt.add(j)
            if len(row_ind) == N:
                break
    
    row_ind = torch.tensor(row_ind, device=pred_xyz.device, dtype=torch.long)
    col_ind = torch.tensor(col_ind, device=pred_xyz.device, dtype=torch.long)
    
    return row_ind, col_ind


def chamfer_with_attr_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    device: str,
    attr_slice=slice(3, 7),
    w_xyz: float = 1.0,
    w_velocity: float = 0.1,
    w_rcs: float = 0.01,
):
    """
    pred, gt: (B, N, D) and (B, M, D); D>=7 assumed
    Uses Chamfer on xyz plus attribute loss on NN-matched pairs.
    """
    pred_xyz = pred[:, :, :3]
    gt_xyz = gt[:, :, :3]

    # (B, N, M)
    dist = torch.cdist(pred_xyz, gt_xyz)

    # forward: each pred -> nearest gt
    nn_gt_idx = dist.argmin(dim=2)  # (B, N)
    forward_xyz = dist.min(dim=2).values.mean()

    # backward: each gt -> nearest pred
    nn_pred_idx = dist.argmin(dim=1)  # (B, M)
    backward_xyz = dist.min(dim=1).values.mean()

    xyz_cd = forward_xyz + backward_xyz

    # attributes matched using the same NN assignment
    pred_attr = pred[:, :, attr_slice]
    gt_attr = gt[:, :, attr_slice]

    # gather matched attributes
    gt_attr_for_pred = gt_attr.gather(
        dim=1,
        index=nn_gt_idx.unsqueeze(-1).expand(-1, -1, gt_attr.shape[-1]),
    )  # (B, N, A)
    pred_attr_for_gt = pred_attr.gather(
        dim=1,
        index=nn_pred_idx.unsqueeze(-1).expand(-1, -1, pred_attr.shape[-1]),
    )  # (B, M, A)

    # --- SPLIT: velocity (first 3) and RCS (last 1) ---
    pred_vel = pred_attr[:, :, :3]
    gt_vel_for_pred = gt_attr_for_pred[:, :, :3]
    gt_vel = gt_attr[:, :, :3]
    pred_vel_for_gt = pred_attr_for_gt[:, :, :3]

    pred_rcs = pred_attr[:, :, 3:4]
    gt_rcs_for_pred = gt_attr_for_pred[:, :, 3:4]
    gt_rcs = gt_attr[:, :, 3:4]
    pred_rcs_for_gt = pred_attr_for_gt[:, :, 3:4]

    vel_fwd = (pred_vel - gt_vel_for_pred).pow(2).mean()
    vel_bwd = (gt_vel - pred_vel_for_gt).pow(2).mean()
    vel_loss = 0.5 * (vel_fwd + vel_bwd)

    rcs_fwd = (pred_rcs - gt_rcs_for_pred).pow(2).mean()
    rcs_bwd = (gt_rcs - pred_rcs_for_gt).pow(2).mean()
    rcs_loss = 0.5 * (rcs_fwd + rcs_bwd)

    total = (w_xyz * xyz_cd) + (w_velocity * vel_loss) + (w_rcs * rcs_loss)
    return total, {
        "xyz": xyz_cd.item(),
        "velocity": vel_loss.item(),
        "rcs": rcs_loss.item(),
    }


class TransformerPointAE(nn.Module):
    """
    A full transformer-based autoencoder for point clouds.
    Uses TransformerPointEncoder and TransformerPointDecoder.
    """

    def __init__(
        self,
        d_model=1024,
        seq_length=256,
        output_points=500,
        num_encoder_layers=4,
        num_decoder_layers=6,  # Added decoder layers
        device="cuda",
    ):
        super().__init__()
        self.encoder = TransformerPointEncoder(
            latent_dim=d_model,
            seq_length=seq_length,
            num_encoder_layers=num_encoder_layers,
        ).to(device)
        self.decoder = TransformerPointDecoder(  # Using the full transformer decoder
            latent_dim=d_model,
            output_points=output_points,
            num_decoder_layers=num_decoder_layers,
        ).to(device)

    def encode(self, uvz):
        return self.encoder(uvz)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, uvz):
        """Full autoencoder pass for reconstruction."""
        latent = self.encode(uvz)
        predicted_uvz, confidence = self.decode(latent)
        return predicted_uvz, confidence, latent


class TransformerPointEncoder(nn.Module):
    """
    Encodes a variable-length point cloud into a fixed-size latent representation.
    Mirrors the structure of the AttentionPointDecoder.
    """

    def __init__(self, latent_dim=1024, seq_length=256, num_encoder_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # 1. Input projection: Project 3D coordinates to the latent dimension
        self.input_proj = nn.Linear(3, latent_dim)

        # 2. Learnable latent queries: These will "ask" the point cloud for information
        # and form the basis of our fixed-size latent representation.
        self.latent_queries = nn.Parameter(torch.randn(1, seq_length, latent_dim))

        # 3. A few layers of self-attention for the point cloud features to interact
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
        )
        self.self_attn_layers = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4. Cross-attention: Latent queries attend to the processed point features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, uvz):
        """
        Args:
            uvz: (B, N, 3) - Input point cloud

        Returns:
            latent: (B, seq_length, latent_dim) - Fixed-size latent representation
        """
        B, N, _ = uvz.shape

        # 1. Project input points into the feature space
        point_features = self.input_proj(uvz)  # (B, N, latent_dim)

        # 2. Let point features interact via self-attention
        processed_features = self.self_attn_layers(point_features)  # (B, N, latent_dim)

        # 3. Expand latent queries for the batch
        queries = self.latent_queries.expand(B, -1, -1)  # (B, seq_length, latent_dim)

        # 4. Cross-attention: queries attend to the processed point features
        # This distills the information from N points into `seq_length` latent vectors.
        latent, _ = self.cross_attn(
            query=queries, key=processed_features, value=processed_features
        )

        # 5. Apply layer normalization for stability
        latent = self.norm(latent)

        return latent


class TransformerAttentionPointAE(nn.Module):
    """
    Encodes a variable-length point cloud into a fixed-size latent representation.
    Mirrors the structure of the AttentionPointDecoder.
    """

    def __init__(
        self,
        latent_dim=1024,
        seq_length=256,
        output_points=500,
        num_encoder_layers=4,
        device="cuda",
    ):
        super().__init__()
        self.encoder = TransformerPointEncoder(
            latent_dim=latent_dim,
            seq_length=seq_length,
            num_encoder_layers=num_encoder_layers,
        ).to(device)
        self.decoder = AttentionPointDecoder(
            latent_dim=latent_dim, output_points=output_points
        ).to(device)

    def encode(self, uvz):
        return self.encoder(uvz)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, uvz):
        """Full autoencoder pass for reconstruction."""
        latent = self.encode(uvz)
        predicted_uvz, confidence = self.decode(latent)
        return predicted_uvz, confidence, latent


class TransformerPointDecoder(nn.Module):
    """
    Encodes a variable-length point cloud into a fixed-size latent representation.
    Mirrors the structure of the AttentionPointDecoder.
    """

    def __init__(
        self,
        latent_dim=1024,
        output_points=500,
        num_decoder_layers=6,
        nhead=8,
        hidden_dim=None,
        output_dim=3,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim * 4

        self.latent_dim = latent_dim
        self.output_points = output_points

        # Learnable queries that will be transformed into points
        self.point_queries = nn.Parameter(torch.randn(1, output_points, latent_dim))

        # Standard Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )

        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Final projection heads to get coordinates and confidence
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, output_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        """
        Args:
            latent: (B, seq_length, latent_dim) - Encoder output

        Returns:
            radar_7d: (B, output_points, output_dim)
            confidence: (B, output_points, 1)
        """
        B = latent.shape[0]
        # Start with the learnable point queries
        queries = self.point_queries.expand(B, -1, -1)

        # Pass through the stack of transformer decoder layers
        # `latent` is the memory (K, V) for cross-attention
        # `queries` is the target (Q) for both self- and cross-attention
        decoded_features = self.transformer_decoder(tgt=queries, memory=latent)

        # Project to final outputs
        radar_7d = self.coord_head(decoded_features)
        confidence = self.confidence_head(decoded_features)

        return radar_7d, confidence


class AttentionPointDecoder(nn.Module):
    """
    Decode latent embeddings to UVZ point cloud coordinates.
    Uses cross-attention mechanism similar to DETR/Point-E.
    """

    def __init__(
        self,
        latent_dim=1024,
        output_points=500,
        hidden_dim=512,
        output_dim=3,
        num_attention_heads=8,
        attention_dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_points = output_points
        self.hidden_dim = hidden_dim

        # Learnable point queries (one per output point)
        # These act like "slots" that will be filled with point information
        self.point_queries = nn.Parameter(
            torch.randn(1, output_points, latent_dim) * 0.02
        )

        # Cross-attention: queries attend to latent features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feedforward network to refine features
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
        )

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),  # Output (u, v, z)  + etc.
        )

        # Optional: confidence head (for weighted loss during training)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Confidence score [0, 1]
        )

    def forward(self, latent):
        """
        Decode latent embeddings to UVZ point coordinates.

        Args:
            latent: (B, num_latent_points, latent_dim)
                    e.g., (B, 256, 1024) - denoised latent from diffusion

        Returns:
            radar_7d: (B, output_points, output_dim) - UVZ coordinates
            confidence: (B, output_points, 1) - optional confidence scores
        """
        B = latent.shape[0]

        # Expand learnable queries for batch
        queries = self.point_queries.expand(B, -1, -1)  # (B, output_points, latent_dim)

        # Cross-attention: queries attend to latent features
        # This allows each point query to "gather" relevant information
        # from the diffusion latent
        attn_output, attn_weights = self.cross_attn(
            query=queries,  # (B, output_points, latent_dim)
            key=latent,  # (B, num_latent_points, latent_dim)
            value=latent,  # (B, num_latent_points, latent_dim)
        )

        # Residual connection + norm
        queries = self.norm1(queries + attn_output)  # (B, output_points, latent_dim)

        # Feedforward refinement
        refined = self.ffn(queries)  # (B, output_points, hidden_dim)
        refined = self.norm2(refined)

        # Predict UVZ coordinates
        radar_7d = self.coord_head(refined)  # (B, output_points, output_dim)

        # Optional: predict confidence scores
        confidence = self.confidence_head(refined)  # (B, output_points, 1)

        return radar_7d, confidence


class DA3DepthTokenizer(nn.Module):
    def __init__(self, out_dim=1024, grid_h=16, grid_w=32, in_ch=1, mid_ch=128):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w

        # light conv stem + downsample
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.SiLU(),
        )

        # project to out_dim (1024)
        self.proj = nn.Conv2d(mid_ch, out_dim, kernel_size=1)

        # learnable 2D positional embedding at target grid
        self.pos = nn.Parameter(torch.zeros(1, out_dim, grid_h, grid_w))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, depth_map):
        """
        depth_map: (B,1,238,504) from DA3 (or whatever size)
        returns: tokens (B, L_depth, 1024) where L_depth = grid_h*grid_w
        """

        x = self.stem(depth_map)  # (B, 128, h,w)
        x = F.adaptive_avg_pool2d(x, (self.grid_h, self.grid_w))  # (B, 128, 16, 32)
        x = self.proj(x)  # (B, 1024, 16, 32)
        x = x + self.pos  # add 2D pos
        tokens = x.flatten(2).transpose(1, 2)  # (B, 512, 1024)
        return tokens


class TransformerDenoiser(nn.Module):
    """
    Custom transformer-based denoiser model.
    the x input is of shape (B, <seq length>, in_channels)
    the condition input is of shape (B, <cond length>, condition_channels)
    """

    def __init__(
        self,
        in_channels,
        condition_channels,
        num_channels=256,
        num_heads=8,
        num_layers=12,
        max_seq_len=256,
        max_cond_len=769,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Timestep embedding (similar to DiT)
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.SiLU(),
            nn.Linear(num_channels * 4, num_channels),
        )

        self.input_proj = nn.Linear(in_channels, num_channels)
        self.condition_proj = nn.Linear(condition_channels, num_channels)

        # Learnable positional embeddings - THIS WAS MISSING!
        self.pos_embed_x = nn.Parameter(
            torch.randn(1, max_seq_len, num_channels) * 0.02
        )
        self.pos_embed_cond = nn.Parameter(
            torch.randn(1, max_cond_len, num_channels) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_channels,
            nhead=num_heads,
            dim_feedforward=num_channels * 4,
            activation="gelu",
            batch_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(num_channels, in_channels)

        # Layer norm for stability
        self.norm = nn.LayerNorm(num_channels)

    def timestep_embedding(self, timesteps, dim):
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: (B,) tensor of timestep indices
            dim: embedding dimension

        Returns:
            (B, dim) tensor of embeddings
        """
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if dim % 2 == 1:  # zero pad if odd dimension
            emb = F.pad(emb, (0, 1))

        return emb

    def forward(self, x, timesteps, encoder_hidden_states):
        """
        x: (B, <seq length>, in_channels)
        timesteps: (B,) - not used in this simple implementation
        encoder_hidden_states: (B, <cond length>, condition_channels)
        """
        batch_size, seq_len, _ = x.shape

        # Encode timesteps
        t_emb = self.timestep_embedding(
            timesteps, self.num_channels
        )  # (B, num_channels)
        t_emb = self.time_embed(t_emb)  # (B, num_channels)

        # Project inputs
        x = self.input_proj(x)  # (B, <seq length>, num_channels)
        cond = self.condition_proj(
            encoder_hidden_states
        )  # (B, <cond length>, num_channels)

        # Add positional embeddings
        x = x + self.pos_embed_x[:, :seq_len, :]
        cond = cond + self.pos_embed_cond[:, : encoder_hidden_states.shape[1], :]

        # Add timestep embedding to input (broadcast across sequence)
        x = x + t_emb.unsqueeze(1)  # (B, seq_len, num_channels)

        # Concatenate x and cond along sequence dimension
        combined = torch.cat(
            [x, cond], dim=1
        )  # (B, <seq length> + <cond length>, num_channels)

        # Pass through transformer
        transformed = self.transformer(
            combined
        )  # (B, <seq length> + <cond length>, num_channels)

        x_transformed = transformed[:, :seq_len, :]  # Extract transformed x part
        # Normalize and project back to in_channels
        x_transformed = self.norm(x_transformed)
        output = self.output_proj(x_transformed)  # (B, seq_len, in_channels)

        return output


class PretrainedPointNeXtEncoderPointAE(nn.Module):
    """
    Encapsulates the PointNeXt Encoder, projection, and Decoder.
    Handles the conversion between UVZ point clouds and the latent space.
    """

    def __init__(
        self,
        d_model=1024,
        output_points=500,
        seq_length=256,
        query_latent_pool_nhead=8,
        query_latent_pool_dropout=0.1,
        device="cuda",
        pointnext_home="/home/palakons/PointNeXt",
        decoder_model="attention",  # "transformer" or "attention", "mlp"s
        num_decoder_layers=6,
        num_decoder_head=8,
        decoder_dropout=0.1,
        pointnext_config="shapenetpart/pointnext-s_c64.yaml",
        output_dim=3,
        ball_query_nsample=48,
        ball_query_radius=8,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_points = output_points
        self.seq_length = seq_length
        self.device = device

        # --- 1. Load PointNeXt Encoder ---
        cfg = EasyConfig()
        cfg.load(f"{pointnext_home}/cfgs/{pointnext_config}", recursive=True)

        cfg.model.encoder_args.nsample = ball_query_nsample  # was 48
        cfg.model.encoder_args.radius = (
            ball_query_radius  # example; tune to your xyz scale
        )

        full_model = build_model_from_cfg(cfg.model)
        if False:  # only pointnext_config="shapenetpart/pointnext-s_c64.yaml"
            pretrained_path = f"{pointnext_home}/pretrained/shapenetpart/pointnext-s-c64/checkpoint/shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth"
            load_checkpoint(full_model, pretrained_path=pretrained_path)

        self.pointnext_encoder = full_model.encoder.to(device)
        # print("model: ", self.pointnext_encoder)

        # (encoder): PointNextEncoder(
        #     (encoder): Sequential(
        #     (0): Sequential(
        #         (0): SetAbstraction(
        #         (convs): Sequential(
        #             (0): Sequential(
        #             (0): Conv1d(7, 32, kernel_size=(1,), stride=(1,))
        #             )
        #         )
        #         )
        #     )
        #     (1): Sequential(
        #         (0): SetAbstraction(
        #         (convs): Sequential(
        #             (0): Sequential(
        #             (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (1): Sequential(
        #             (0): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (2): Sequential(
        #             (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #         )
        #         (grouper): QueryAndGroup()
        #         )
        #     )
        #     (2): Sequential(
        #         (0): SetAbstraction(
        #         (convs): Sequential(
        #             (0): Sequential(
        #             (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (1): Sequential(
        #             (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (2): Sequential(
        #             (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #         )
        #         (grouper): QueryAndGroup()
        #         )
        #     )
        #     (3): Sequential(
        #         (0): SetAbstraction(
        #         (convs): Sequential(
        #             (0): Sequential(
        #             (0): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (1): Sequential(
        #             (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (2): Sequential(
        #             (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #         )
        #         (grouper): QueryAndGroup()
        #         )
        #     )
        #     (4): Sequential(
        #         (0): SetAbstraction(
        #         (convs): Sequential(
        #             (0): Sequential(
        #             (0): Conv2d(259, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (1): Sequential(
        #             (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #             (2): Sequential(
        #             (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #             (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #             (2): ReLU(inplace=True)
        #             )
        #         )
        #         (grouper): QueryAndGroup()
        #         )
        #     )
        #     )
        # )

        # singularity/home/palakons/PointNeXt/cfgs/scannet/pointnext-s.yaml
        encoder_dim = cfg.model.encoder_args.width * (  # 32
            2 ** (len(cfg.model.encoder_args.blocks) - 1)  # [1, 1, 1, 1, 1]
        )

        # --- 2. Define Trainable Components ---
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        # NEW: learnable latent queries + cross-attention pooling to fixed seq_length
        self.latent_queries = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=query_latent_pool_nhead,
            dropout=query_latent_pool_dropout,
            batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(d_model)

        if decoder_model == "transformer":
            self.pointcloud_decoder = TransformerPointDecoder(
                latent_dim=d_model,
                output_points=output_points,
                num_decoder_layers=num_decoder_layers,
                output_dim=output_dim,
            )
        elif decoder_model == "attention":
            self.pointcloud_decoder = AttentionPointDecoder(
                latent_dim=d_model,
                output_points=output_points,
                output_dim=output_dim,
                num_attention_heads=num_decoder_head,
                attention_dropout=decoder_dropout,
            )

    def encode(self, radar_7d_data, training=True):
        """Converts a radar_7d_data to a latent representation."""

        # print("radar_7d_data shape:", radar_7d_data.shape)
        # print("5 samples of radar_7d_data:", radar_7d_data[0, :5, :])
        point_data = {
            "pos": radar_7d_data[:, :, :3].contiguous(),  # (B,N,3)
            "x": radar_7d_data[:, :, 3:]
            .transpose(1, 2)
            .contiguous(),  # (B,4,N): vx,vy,vz,rcs
        }

        encoder_output = self.pointnext_encoder(point_data)
        # (positions_per_stage, features_per_stage),

        features = encoder_output[1][-1]
        # for i, layer in enumerate(encoder_output[1]):
        #     print("layer shape:", i, layer.shape)
        # layer shape: 0 torch.Size([1, 7, 800])
        # layer shape: 1 torch.Size([1, 32, 800])
        # layer shape: 2 torch.Size([1, 64, 200])
        # layer shape: 3 torch.Size([1, 128, 50])
        # layer shape: 4 torch.Size([1, 256, 12])
        # layer shape: 5 torch.Size([1, 512, 3])

        features = features.transpose(1, 2)  # [1, 3, 512]

        # Project and resample to the correct sequence length
        tokens = self.encoder_proj(features)  # (B, N, d_model)

        # Cross-attention pooling: (B, N, d_model) -> (B, seq_length, d_model)
        B, N, _ = tokens.shape
        if N == 0:
            # safety fallback (shouldn't happen in normal PointNeXt outputs)
            latent = tokens.new_zeros((B, self.seq_length, self.d_model))
            return latent

        queries = self.latent_queries.expand(B, -1, -1)  # (B, seq_length, d_model)
        latent, _ = self.cross_attn(query=queries, key=tokens, value=tokens)
        latent = self.latent_norm(latent)
        # print("latent shape:", latent.shape) #([1, 64, 768])

        return latent

    def decode(self, latent):
        """Converts a latent representation back to a UVZ point cloud."""
        return self.pointcloud_decoder(latent)

    def forward(self, uvz):
        """Full autoencoder pass for reconstruction."""
        latent = self.encode(uvz, training=self.training)
        predicted_radar_7d, confidence = self.decode(latent)
        # print("predicted_radar_7d shape:", predicted_radar_7d.shape)
        # print("confidence shape:", confidence.shape)
        # print("latent shape:", latent.shape)
        return predicted_radar_7d, confidence, latent

    def resample_to_seq_length(self, pointnext_embedding, training=True):
        """
        Resample point features to target sequence length.

        Args:
            pointnext_embedding: (B, N, C) where N might be != seq_length
            training: bool, if True use random sampling, else interpolation

        Returns:
            resampled: (B, seq_length, C)
        """
        B, N, C = pointnext_embedding.shape

        if N == self.seq_length:
            return pointnext_embedding

        if training:
            # TRAINING: Random sampling for data augmentation
            if N < self.seq_length:
                # Upsample by random sampling with replacement
                indices = torch.randint(0, N, (self.seq_length,), device=self.device)
                resampled = pointnext_embedding[:, indices, :]
            else:
                # Downsample by random sampling without replacement
                indices = torch.randperm(N, device=self.device)[: self.seq_length]
                resampled = pointnext_embedding[:, indices, :]

            return resampled

        else:
            # INFERENCE: Deterministic interpolation for stability
            if N < self.seq_length:
                # Upsample via interpolation
                resampled = F.interpolate(
                    pointnext_embedding.transpose(1, 2),  # (B, C, N)
                    size=self.seq_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(
                    1, 2
                )  # (B, seq_length, C)
            else:
                # Downsample via interpolation
                resampled = F.interpolate(
                    pointnext_embedding.transpose(1, 2),
                    size=self.seq_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            return resampled


# def chamfer_distance(pred_uvz, gt_uvz):
#     print("shape of pred_uvz:", pred_uvz.shape)
#     print("shape of gt_uvz:", gt_uvz.shape)
#     """
#     Compute bidirectional Chamfer Distance.

#     Args:
#         pred_uvz: (B, N, 3) - predicted UVZ
#         gt_uvz: (B, M, 3) - ground truth UVZ

#     Returns:
#         chamfer_loss: scalar tensor
#     """
#     # Compute pairwise squared distances
#     # pred: (B, N, 1, 3), gt: (B, 1, M, 3)
#     pred_expanded = pred_uvz.unsqueeze(2)  # (B, N, 1, 3)
#     gt_expanded = gt_uvz.unsqueeze(1)  # (B, 1, M, 3)

#     # dist[b, i, j] = ||pred[b,i] - gt[b,j]||^2
#     dist = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # (B, N, M)

#     # Forward: nearest GT for each predicted point
#     try:
#         min_dist_pred_to_gt, _ = torch.min(dist, dim=2)  # (B, N)
#         forward_loss = min_dist_pred_to_gt.mean()
#     except:
#         print("pred_expanded shape:", pred_expanded.shape)
#         print(
#             "gt_expanded shape:", gt_expanded.shape
#         )  # gt_expanded shape: torch.Size([1, 1, 0, 3])
#         print(f"dist shape: {dist.shape}")
#         return None

#     # Backward: nearest predicted for each GT point
#     min_dist_gt_to_pred, _ = torch.min(dist, dim=1)  # (B, M)
#     backward_loss = min_dist_gt_to_pred.mean()

#     # Bidirectional Chamfer
#     chamfer_loss = forward_loss + backward_loss

#     return chamfer_loss


class DitDDPM:
    """
    Simple DDPM trainer for conditional pixel-depth occupancy grid generation
    """

    def __init__(
        self,  # sequence length (N)
        autoencoder: PretrainedPointNeXtEncoderPointAE,  # Pass the AE as an argument
        num_train_timesteps=1000,
        # image_size=(16, 32),
        device="cuda",
        # use_pointnext=True,
        # geometric_loss_weight:int=0,
        d_model: int = 1024,
        # output_points=500,
        seq_length=256,
        # pointnext_home='/home/palakons/PointNeXt',
    ):
        # self.pointnext_home=pointnext_home
        self.device = device
        # self.geometric_loss_weight = geometric_loss_weight
        # self.image_size = image_size
        # self.use_pointnext = use_pointnext
        # self.output_points = output_points
        self.seq_length = seq_length
        # if self.use_pointnext:
        #     print("Using PointNeXt encoder and decoder from ShapeNetPart.")

        #     # Load ShapeNetPart config (has both encoder and decoder)
        #     cfg = EasyConfig()
        #     cfg.load(f'{self.pointnext_home}/cfgs/shapenetpart/pointnext-s_c64.yaml', recursive=True)

        #     # cfg.model.encoder_args.in_channels = 3
        #     # Build full model
        #     full_model = build_model_from_cfg(cfg.model)

        #     pretrained_path = (
        #         f'{self.pointnext_home}/pretrained/shapenetpart/pointnext-s-c64/checkpoint/shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth'
        #     )

        #     if os.path.exists(pretrained_path):
        #         print(f"✅ Loading pretrained checkpoint from:")
        #         print(f"   {pretrained_path}")

        #         # Use PointNeXt's load_checkpoint function
        #         # Returns: (best_epoch, best_val_metric)
        #         best_epoch, best_val = load_checkpoint(
        #             full_model,
        #             pretrained_path=pretrained_path
        #         )

        #         print(f"✅ Successfully loaded pretrained weights!")
        #         print(f"   Best epoch: {best_epoch}")
        #         print(f"   Best validation metric: {best_val}")

        #     else:
        #         raise FileNotFoundError(f"❌ Pretrained checkpoint not found at: {pretrained_path}")

        #     # Extract encoder
        #     self.pointnext_encoder = full_model.encoder.to(device)
        #     # Freeze encoder parameters
        #     for param in self.pointnext_encoder.parameters():
        #         param.requires_grad = False

        #     trainable = sum(p.numel() for p in self.pointnext_encoder.parameters() if p.requires_grad)
        #     total = sum(p.numel() for p in self.pointnext_encoder.parameters())
        #     print(f"   Encoder: {trainable:,} / {total:,} trainable (should be 0)")
        #     assert trainable == 0, "Encoder parameters are not frozen!"

        #     # Get encoder output dimension
        #     encoder_dim = cfg.model.encoder_args.width * (2 ** (len(cfg.model.encoder_args.blocks) - 1))
        #     print(f"   Encoder output dim: {encoder_dim}")

        #     # Projection layers
        #     self.encoder_proj = nn.Linear(encoder_dim, d_model).to(device)
        #     proj_params = sum(p.numel() for p in self.encoder_proj.parameters())

        #     # Add a final projection to convert decoder output to UVZ coordinates
        #     # The decoder outputs features, we need to project to (x, y, z)

        #     self.pointcloud_decoder = PointCloudDecoder(
        #         latent_dim=d_model,
        #         output_points=output_points
        #     ).to(device)
        #     decoder_params = sum(p.numel() for p in self.pointcloud_decoder.parameters())
        #     print(f"   Encoder projection parameters: {proj_params:,}")
        #     print(f"   Decoder parameters: {decoder_params:,}")
        #     # For interpolating to desired output_points
        #     self.output_points = output_points

        # else:
        #     self.pointnext_encoder = None
        #     self.encoder_proj = None
        #     self.pointcloud_decoder = None

        self.autoencoder = autoencoder.to(device)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        # Create model
        self.model = TransformerDenoiser(
            in_channels=d_model,  # PointNeXt embedding dim
            condition_channels=d_model,  # ViT + Depth
            num_channels=256,  # Example: number of transformer channels
            num_heads=8,  # Example: number of attention heads
            num_layers=12,  # Example: number of transformer layers
            max_seq_len=self.seq_length,
            max_cond_len=512 + 257,  # Depth tokens + ViT features = 769,
        ).to(device)

        # Add DA3DepthTokenizer for depth tokenization
        self.depth_tokenizer = DA3DepthTokenizer(
            out_dim=d_model, grid_h=16, grid_w=512 // 16
        ).to(device)

        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,  # problem
            clip_sample_range=1,
            prediction_type="epsilon",
        )

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    # def resample_to_seq_length(self, pointnext_embedding, training=True):
    #     """
    #     Resample point features to target sequence length.

    #     Args:
    #         pointnext_embedding: (B, N, C) where N might be != seq_length
    #         training: bool, if True use random sampling, else interpolation

    #     Returns:
    #         resampled: (B, seq_length, C)
    #     """
    #     B, N, C = pointnext_embedding.shape

    #     if N == self.seq_length:
    #         return pointnext_embedding

    #     if training:
    #         # TRAINING: Random sampling for data augmentation
    #         if N < self.seq_length:
    #             # Upsample by random sampling with replacement
    #             indices = torch.randint(0, N, (self.seq_length,), device=self.device)
    #             resampled = pointnext_embedding[:, indices, :]
    #         else:
    #             # Downsample by random sampling without replacement
    #             indices = torch.randperm(N, device=self.device)[:self.seq_length]
    #             resampled = pointnext_embedding[:, indices, :]

    #         return resampled

    #     else:
    #         # INFERENCE: Deterministic interpolation for stability
    #         if N < self.seq_length:
    #             # Upsample via interpolation
    #             resampled = F.interpolate(
    #                 pointnext_embedding.transpose(1, 2),  # (B, C, N)
    #                 size=self.seq_length,
    #                 mode='linear',
    #                 align_corners=False
    #             ).transpose(1, 2)  # (B, seq_length, C)
    #         else:
    #             # Downsample via interpolation
    #             resampled = F.interpolate(
    #                 pointnext_embedding.transpose(1, 2),
    #                 size=self.seq_length,
    #                 mode='linear',
    #                 align_corners=False
    #             ).transpose(1, 2)

    #         return resampled
    def prepare_condition(self, depth_image, clip_feature):
        """
        Prepare condition from depth images, and precomputed ViT features.

        Args:
            depth_image: (B, 1, H, W) or (B, H, W) - Depth image.
            clip_feature: (B, 257, 1024) - Precomputed ViT features.

        Returns:
            condition: (B, 512 + 257, 1024) - Concatenated depth tokens and ViT features.
        """
        # Ensure depth has channel dimension
        if depth_image.ndim == 3:
            depth_image = depth_image.unsqueeze(1)  # (B, 1, H, W)

        # Tokenize depth image
        depth_tokens = self.depth_tokenizer(depth_image)  # (B, 512, 1024)
        # Concatenate depth tokens with ViT features
        condition = torch.cat(
            [depth_tokens, clip_feature], dim=1
        )  # (B, 512 + 257 = 769, 1024)
        return condition

    def train_step(self, batch, optimizer):
        """
        Single training step

        Args:
            "uvz": torch.stack(uvz_points),
            'rgb': torch.stack(rgb_images),
            'depth': torch.stack(depth_images),
            'clip_feature': torch.stack(clip_features),  # Add CLIP features to batch

        Returns:
            loss: scalar tensor
        """
        rgb = batch["camera_front"].to(self.device)
        depth = batch["depth_image"].to(self.device)
        uvz = batch["uvz"].to(self.device)
        clip_feature = batch["clip_feature"].to(self.device)

        # print("rgb shape",rgb.shape)
        # print("depth shape",depth.shape)
        # print("uvz shape",uvz.shape)
        # print("clip_feature shape",clip_feature.shape)
        # rgb shape torch.Size([4, 3, 16, 32])
        # depth shape torch.Size([4, 16, 32])
        # uvz shape torch.Size([4, 500, 3])
        # clip_feature shape torch.Size([4, 257, 1024])

        batch_size = rgb.shape[0]

        # Prepare condition
        condition = self.prepare_condition(depth, clip_feature)

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        # apply PointNeXt ecoder to get DiT embedding of dim (B, 256, 1024)

        if self.use_pointnext:
            uvz_padded = torch.cat(
                [
                    uvz,  # (B, 500, 3) - xyz coordinates
                    torch.zeros(
                        uvz.shape[0], uvz.shape[1], 4, device=uvz.device
                    ),  # (B, 500, 4) - dummy features
                ],
                dim=-1,
            )  # (B, 500, 7)

            point_data = {
                "pos": uvz_padded[
                    :, :, :3
                ].contiguous(),  # (B, 500, 3) - still use xyz for positions
                "x": uvz_padded.transpose(
                    1, 2
                ).contiguous(),  # (B, 7, 500) - all 7 channels
            }
            # print('uvz shape:', uvz.shape) #vz shape: torch.Size([4, 500, 3])

            encoder_output = self.pointnext_encoder(point_data)

            # PointNeXt returns (position_list, feature_list) - NOTE: REVERSED!
            if isinstance(encoder_output, tuple):
                position_list, feature_list = encoder_output

                # Debug: Verify what we're getting
                # print("\n=== PointNeXt Encoder Output ===")
                # print(f"Position list length: {len(position_list)}") #6
                # print(f"Feature list length: {len(feature_list)}") #6

                # The FEATURE_LIST contains the actual high-dimensional features
                # The last element has the most abstract representation
                features = feature_list[-1]  #  (B, 1024, 31)

                # print(f"\nUsing last layer features: {features.shape}") #([4, 1024, 31])

            else:
                features = encoder_output

            # Ensure (B, N, C) format
            # Currently: (B, 1024, 31) -> Need: (B, 31, 1024)
            if features.dim() == 3:
                if features.shape[1] > features.shape[2]:
                    # (B, C, N) -> (B, N, C)
                    features = features.transpose(1, 2)

            # print(f"Features after transpose: {features.shape}") #([4, 31, 1024])

            # Project to d_model (already 1024, so this is likely identity)
            pointnext_embedding = self.encoder_proj(features)  # (B, 31, 1024)

            # print(f"After projection: {pointnext_embedding.shape}") #([4, 31, 1024])

            # Resample to exactly 256 points
            pointnext_embedding = self.resample_to_seq_length(
                pointnext_embedding, training=True  # ← Use random sampling
            )

            # print(f"Final pointnext_embedding shape: {pointnext_embedding.shape}") #([4, 256, 1024])

        else:
            pointnext_embedding = torch.zeros(
                batch_size, self.seq_length, 1024, device=self.device
            )
            raise NotImplementedError(
                "Training without PointNeXt encoder is not implemented."
            )

        # Add noise to occupancy grid
        noise = torch.randn_like(pointnext_embedding)
        noisy_pointnext_embedding = self.noise_scheduler.add_noise(
            pointnext_embedding, noise, timesteps
        )

        # Predict noise
        noise_pred = self.model(
            noisy_pointnext_embedding, timesteps, encoder_hidden_states=condition
        )

        # Compute loss
        diffusion_loss = nn.functional.mse_loss(noise_pred, noise)

        if self.geometric_loss_weight > 0 and self.use_pointnext:

            # Decode CLEAN latent to UVZ points
            predicted_uvz, confidence = self.pointcloud_decoder(pointnext_embedding)

            # Chamfer distance between predicted and ground truth
            recon_loss = chamfer_distance(predicted_uvz, uvz)

            total_loss = diffusion_loss + self.geometric_loss_weight * recon_loss
        else:
            total_loss = diffusion_loss

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.use_pointnext:
            torch.nn.utils.clip_grad_norm_(self.pointcloud_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.encoder_proj.parameters(), 1.0)
        optimizer.step()
        return {
            "total": total_loss.item(),
            "diffusion": diffusion_loss.item(),
            "recon": recon_loss.item() if self.geometric_loss_weight > 0 else 0.0,
        }

    @torch.no_grad()
    def sample(self, depth_image, clip_feature, num_inference_steps=50, seed=None):
        """
        return {pointnext_enmbedding:.., uvz:..}
        this function perform the "sample" operation of DDPM
        generate pointnext_enmbedding and uvz points from depth image and clip feature
        """
        if seed is not None:
            # Set seed for this sampling only
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.model.eval()

        with torch.no_grad():

            # Prepare condition
            condition = self.prepare_condition(depth_image, clip_feature)

            # Set timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps

            batch_size = depth_image.shape[0]
            # Start from pure noise
            pointnext_embedding = torch.randn(
                batch_size, self.seq_length, 1024, device=self.device
            )
        output_embeddings = {}
        for t in tqdm(timesteps, desc="Sampling timesteps"):
            # Expand timestep to batch size
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(
                pointnext_embedding, t_batch, encoder_hidden_states=condition
            )

            # Compute previous noisy sample x_t -> x_t-1
            pointnext_embedding = self.noise_scheduler.step(
                noise_pred, t, pointnext_embedding
            ).prev_sample
            predicted_uvz, confidence = self.autoencoder.pointcloud_decoder(
                pointnext_embedding
            )
            output_embeddings[t.item()] = {
                "pointnext_embedding": pointnext_embedding.clone(),
                "uvz": predicted_uvz.clone(),
                "confidence": confidence.clone(),
            }

        output = {"pointnext_embedding": pointnext_embedding}
        # Decode pointnext embedding to uvz points
        predicted_uvz, confidence = self.autoencoder.pointcloud_decoder(
            pointnext_embedding
        )
        output["uvz"] = predicted_uvz  # (B, 500, 3)
        output["confidence"] = confidence  # (B, 500, 1)

        return output, output_embeddings

    def save_checkpoint(self, path, optimizer=None, epoch=None):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "depth_tokenizer_state_dict": self.depth_tokenizer.state_dict(),
            "config": {
                "num_train_timesteps": self.noise_scheduler.config.num_train_timesteps,
            },
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint.get("epoch", None)
        print(f"Loaded checkpoint from {path} (epoch {epoch})")
        return epoch


def train_eval_ae_epoch(
    autoencoder, dataloader, optimizer,lr_scheduler, config, writer, global_step, train=True
):
    if train:
        autoencoder.train()
    else:
        autoencoder.eval()  
    pbar = tqdm(dataloader, leave=False, desc="Batches")
    epoch_loss = 0.0
    epoch_7d_cd = 0.0
    epoch_xyz_cd = 0.0
    num_samples = 0
    time_records ={}
    for batch in pbar:
        with torch.no_grad() if not train else torch.enable_grad():
            time0 = time.time()
            if train:
                optimizer.zero_grad()

            filtered_radar_data = batch["filtered_radar_data"].to(config["device"])
            batch_size = filtered_radar_data.shape[0] 
            # normalize the whole data

            man_dataset = dataloader
            while not isinstance(man_dataset, MANDataset):
                man_dataset = man_dataset.dataset

            data_mean = man_dataset.data_mean.to(config["device"])
            data_std = man_dataset.data_std.to(config["device"])
            # print("during training, using data mean and std from MANDataset:")
            # print(f"Data mean: {data_mean.cpu().numpy()}, Data std: {data_std.cpu().numpy()}")

            normalized_filtered_radar_data = (
                filtered_radar_data - data_mean
            ) / data_std
            time1 = time.time()
            predicted_normalized_filtered_radar_data, confidence, latent = autoencoder(
                normalized_filtered_radar_data
            )
            time2 = time.time()
            predicted_filtered_radar_data = (
                predicted_normalized_filtered_radar_data * data_std + data_mean
            )

            # print("min max of filtered_radar_data:", np.min(filtered_radar_data.cpu().numpy(), axis=(0,1)), np.max(filtered_radar_data.cpu().numpy(), axis=(0,1))) # min max of filtered_radar_data: [0. 0. 0.] [1980. 943. 250.]
            # print("min max of normalized_filtered_radar_data", np.min(normalized_filtered_radar_data.cpu().numpy(), axis=(0,1)), np.max(normalized_filtered_radar_data.cpu().numpy(), axis=(0,1))) # min max of normalized_filtered_radar_data: [-2.5 -2.5 -2.5 ...] [2.5 2.5 2.5 ...]
            # print("min max of predicted_normalized_filtered_radar_data", np.min(predicted_normalized_filtered_radar_data.detach().cpu().numpy(), axis=(0,1)), np.max(predicted_normalized_filtered_radar_data.detach().cpu().numpy(), axis=(0,1))) # min max of normalized_predicted_filtered_radar_data: [-2.5 -2.5 -2.5 ...] [2.5 2.5 2.5 ...]
            # print("min max of predicted_filtered_radar_data:", np.min(predicted_filtered_radar_data.detach().cpu().numpy(), axis=(0,1)), np.max(predicted_filtered_radar_data.detach().cpu().numpy(), axis=(0,1))) # min max of pred_xyz: [0. 0. 0.] [1980. 943. 250.]

            time3 = time.time()

            # print(
            #     "predicted_filtered_radar_data shape:", predicted_filtered_radar_data.shape
            # )#[1, 800, 7])
            # print("filtered_radar_data shape:", filtered_radar_data.shape)#[1, 800, 7])

            # filter point only with confidence > 0.5


            # RECONSTRUCTION LOSS ONLY
            if config["ae_loss_type"] == "mse":
                loss = nn.functional.mse_loss(
                    predicted_filtered_radar_data, filtered_radar_data
                )
            elif config["ae_loss_type"] == "good":
                # 1.	Normalize cost matrix
                # 2.	Tune ε 
                # 3.	Combine with Chamfer
                # 4.	Detach transport plan
                # 5.	KNN regularization
                # 6.	Singkhorn Clip gradients
                # 7.    Balanced OT
                
                # Use all 7D points for both Chamfer and Sinkhorn losses
                P = predicted_normalized_filtered_radar_data  # (B, N, 7)
                G = normalized_filtered_radar_data           # (B, M, 7)

                # Combine losses
                loss = 0
                cd_weight, ot_weight, knn_weight = config["ae_weight_good_loss"]
                if cd_weight > 0:

                    # Chamfer Distance (uses first 3 dims for geometry)
                    loss_cd, _ = pt3d_chamfer_distance(P[:, :, :3], G[:, :, :3], batch_reduction ="mean")

                    # ^ Chamfer on xyz only
                    loss += cd_weight * loss_cd
                
                if ot_weight > 0:

                    # Sinkhorn OT on all 7 dims
                    singkhorn = SamplesLoss("sinkhorn", p=2, blur=config["sk_eps"])  # ε regularization
                    # Detach transport plan if desired
                    loss_ot = singkhorn(P, G)
                    # print("loss_ot:", loss_ot   )#tensor([3.3298], device='cuda:0', grad_fn=<AddBackward0>)
                    # print("loss:", loss, ot_weight, loss_ot)
                    if config["ot_clip"]:
                        # Clip gradients for Sinkhorn loss
                        torch.nn.utils.clip_grad_norm_([loss_ot], max_norm=1.0)
                    if config["sk_detach"]:
                        loss_ot = loss_ot.detach()
                    loss += ot_weight * loss_ot[0]

                if knn_weight > 0:
                    # KNN regularization (example: mean distance to kNN in P)
                    edge_index = knn_graph(P[:, :, :3].reshape(-1, 3), k=5, batch=None)
                    knn_loss = ((P[:, :, :3].reshape(-1, 3)[edge_index[0]] - P[:, :, :3].reshape(-1, 3)[edge_index[1]])**2).sum(-1).mean()
                
                    loss += knn_weight * knn_loss

            elif config["ae_loss_type"] == "chamfer":
                loss = chamfer_distance(
                    predicted_filtered_radar_data[:, :, :3],
                    filtered_radar_data[:, :, :3],
                    config["device"],
                )
            elif config["ae_loss_type"] == "chamfer-attr":
                ae_weight_attr_loss = config["ae_weight_attr_loss"]
                attr_loss = chamfer_with_attr_loss(
                    predicted_filtered_radar_data,
                    filtered_radar_data,
                    config["device"],
                    w_xyz=ae_weight_attr_loss[0],
                    w_velocity=ae_weight_attr_loss[1],
                    w_rcs=ae_weight_attr_loss[2],
                )
                loss = attr_loss[0]
                loss_details = attr_loss[1]
                # {
                #     "xyz": xyz_cd.item(),
                #     "velocity": vel_loss.item(),
                #     "rcs": rcs_loss.item(),
                # }
                writer.add_scalar(
                    f"ae_pretrain/loss_details/xyz_cd",
                    loss_details["xyz"],
                    global_step,
                )
                writer.add_scalar(
                    f"ae_pretrain/loss_details/velocity_loss",
                    loss_details["velocity"],
                    global_step,
                )
                writer.add_scalar(
                    f"ae_pretrain/loss_details/rcs_loss",
                    loss_details["rcs"],
                    global_step,
                )
            elif config["ae_loss_type"] == "hungarian":
                ae_weight_attr_loss = config["ae_weight_attr_loss"]

                # attr_loss = chamfer_with_attr_loss(
                #     predicted_filtered_radar_data,
                #     filtered_radar_data,
                #     config["device"],
                #     w_xyz=ae_weight_attr_loss[0],
                #     w_velocity=ae_weight_attr_loss[1],
                #     w_rcs=ae_weight_attr_loss[2],
                # )
                hungarian_loss = hungarian_l2_with_attr_loss(
                    predicted_filtered_radar_data,
                    filtered_radar_data,
                    w_xyz=ae_weight_attr_loss[0],
                    w_velocity=ae_weight_attr_loss[1],
                    w_rcs=ae_weight_attr_loss[2],use_greedy=config["hungarian_use_greedy"]

                )
                # print("hungarian_loss:", hungarian_loss[1]["xyz"])
                # print("chamfer_loss:", attr_loss[1]["xyz"])
                loss = hungarian_loss[0]
                loss_details = hungarian_loss[1]
                # {
                #     "xyz": xyz_cd.item(),
                #     "velocity": vel_loss.item(),
                #     "rcs": rcs_loss.item(),
                # }
                writer.add_scalar(
                    f"ae_pretrain/loss_details/xyz_cd",
                    loss_details["xyz"],
                    global_step,
                )
                writer.add_scalar(
                    f"ae_pretrain/loss_details/velocity_loss",
                    loss_details["velocity"],
                    global_step,
                )
                writer.add_scalar(
                    f"ae_pretrain/loss_details/rcs_loss",
                    loss_details["rcs"],
                    global_step,
                )
            else:
                raise ValueError(f"Unknown ae_loss_type: {config['ae_loss_type']}")
            time4 = time.time()
            # Backprop
            if train:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if True:
                    # After loss.backward()
                    for name, param in autoencoder.named_parameters():
                        if param.grad is not None:
                            writer.add_scalar(f'grad_norm/{name}', param.grad.norm().item(), global_step)
            time5 = time.time()
            batch_loss = loss.item()
            num_samples += batch_size
            epoch_loss += batch_loss* batch_size #support last non-full batch

            batch_xyz_cd  = pt3d_chamfer_distance(predicted_filtered_radar_data[:, :, :3], filtered_radar_data[:, :, :3])[0].item()
            batch_7d_cd = pt3d_chamfer_distance(predicted_filtered_radar_data, filtered_radar_data)[0].item()

            epoch_7d_cd += batch_7d_cd* batch_size
            epoch_xyz_cd += batch_xyz_cd* batch_size
            
            pbar.set_description(f"CD" f"{batch_loss:.4f}")
            
            time_records_batch = {
                "data_prep_time": time1 - time0,
                "forward_time": time2 - time1,
                "denormalize_time": time3 - time2,
                "loss_time": time4 - time3,
                "backward_time": time5 - time4,
            }
            time_records = {k: time_records.get(k, 0) + time_records_batch[k]* batch_size for k in time_records_batch}

            global_step += 1
    time_records = {k: v / num_samples for k, v in time_records.items()}
    avg_loss = epoch_loss / num_samples
    avg_xyz_cd = epoch_xyz_cd / num_samples
    avg_7d_cd = epoch_7d_cd / num_samples

    # print(
    #     "epoch_loss",
    #     epoch_loss,
    #     "len(dataloader)",
    #     len(dataloader),
    #     "avg_loss",
    #     avg_loss,
    # )
    return avg_loss, global_step, time_records, avg_xyz_cd, avg_7d_cd


def load_ae_checkpoint(model, optimizer, lr_scheduler,config):
    if config["point_ae_checkpoint"] == "":
        print("No checkpoint path provided, training from scratch.")
        return 0
    checkpoint_path = config["point_ae_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])

    # check if config matches
    important_fields = [
        "scene_ids",
        "data_file",
        "num_points",
        "num_input_frames",
        "point_ae_model",
        "seed",
        "latent_dim",
        "latent_seq_length",
        "ae_decay",
        "ae_lr",
        "batch_size",
        "query_num_heads",
        "query_dropout",
        "decoder_num_layers",
        "decoder_num_heads",
        "decoder_dropout",
        "pointnext_config",
        "hungarian_use_greedy",
        "ae_loss_type",
        "ae_weight_attr_loss",
        "ddpm_clip_sample",
        "ae_lr_decay_step",
        "ae_lr_decay_gamma",
        "ae_weight_good_loss","sk_eps","sk_detach","ot_clip","ae_lr_clr_factor","ae_lr_clr_mode"
    ]
    matched = True
    for field in important_fields:
        # if its a list
        if field not in checkpoint["config"]:
            print(f"Field {field} not in checkpoint config.")
            matched = False
            continue
        if isinstance(checkpoint["config"][field], list):

            if len(checkpoint["config"][field]) != len(config[field]):
                print(field, "length >", checkpoint["config"][field], config[field])
                matched = False
            else:
                for a, b in zip(checkpoint["config"][field], config[field]):
                    if a != b:
                        print(field, ">", a, b)
                        matched = False
                        break

        # if its not a list
        else:
            if checkpoint["config"][field] != config[field]:
                matched = False
                print(field, ">", checkpoint["config"][field], config[field])

    if not matched:
        print(
            "Warning: Checkpoint config does not match current config. NOT loaded check point."
        )
        exit()

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
    return epoch


import hashlib


def _short_tag(s: str, max_len: int = 80) -> str:
    """Make a filesystem-safe short tag. Keeps prefix + adds stable hash."""
    s = str(s)
    if len(s) <= max_len:
        return s
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    keep = max_len - (1 + len(h))
    return f"{s[:keep]}_{h}"


def _short_last_segments(s: str, num_underscore_segments: int = 17) -> str:
    """e.g. full fname: pretrain_ae_target_e150000_sc0_mini_np800_fr2_modelpointnext-attention_sd42_latdim768_latseq64_decay1E-04_lr3E-05_bs1_qnh8_qdrop0E+00_dnl24_dnh4_ddrop0E+00_scannet_untitled_ltypechamfer-attr_wattr1E+00-1E-01-1E-02_bqn24_bqr64.0

    this function hash the last num_underscore_segments segments, while keep the rest the same
    """
    segments = s.split("_")
    if len(segments) <= num_underscore_segments:
        return s
    h = (
        hashlib.sha1(("_".join(segments[-num_underscore_segments:]))
        .encode("utf-8"))
        .hexdigest()[:10]
    )

    return f"{'_'.join(segments[:-num_underscore_segments])}_{h}"


def save_ae_checkpoint(model, optimizer, lr_scheduler, epoch, metrics, path, config):
    """Save autoencoder checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)


def plot_ae(
    dsname: str,
    ds: torch.utils.data.Dataset,
    autoencoder: PretrainedPointNeXtEncoderPointAE,
    config: dict,
    plot_dir: str,
    run_id: str,
    epoch: int,
):

    assert plot_dir is not None, "plot_dir must be provided for plotting"
    autoencoder.eval()
    cds = []
    paths = []
    man_dataset = ds
    while not isinstance(man_dataset, MANDataset):
        man_dataset = man_dataset.dataset

    data_mean = man_dataset.data_mean.to(config["device"])
    data_std = man_dataset.data_std.to(config["device"])
    for i in range(len(ds)):
        sample_data = ds[i]
        filtered_radar_data = (
            sample_data["filtered_radar_data"].to(config["device"]).unsqueeze(0)
        )
        normalized_filtered_radar_data = (
                filtered_radar_data - data_mean
            ) / data_std
        
        with torch.no_grad():
            normalized_predicted_filtered_radar_data, confidence, latent = autoencoder(
                normalized_filtered_radar_data
            )

            predicted_filtered_radar_data = (
                normalized_predicted_filtered_radar_data * data_std + data_mean
            )

            # cd_tensor = chamfer_distance(
            #     predicted_filtered_radar_data[:, :, :3],
            #     filtered_radar_data[:, :, :3],
            #     config["device"],
            # )
        gt_xyz = filtered_radar_data[0, :, :3].cpu().numpy()
        pred_xyz = predicted_filtered_radar_data[0, :, :3].cpu().numpy()

        # print(f"{cd_tensor.item():.2f}", end=", ")

        cd, save_path = ds.dataset.visualize_uvz_comparison(
            title=f"{dsname}_s{i}_"
            + sample_data["frame_token"][-5:]
            + f"_ae_e{epoch+1}_{run_id}",
            frame_token=f"{dsname}_s{i}_"
            + sample_data["frame_token"][-5:]
            + f"_ae_e{epoch+1}_{run_id}",
            original_uvz=gt_xyz,
            reconstructed_uvz=pred_xyz,
            save_dir=plot_dir,
            plotlims= {"u": (-50, 200), "v": (-100, 150), "z": (-25, 25)},
            marker_config={
                "original": {
                    "color": "blue",
                    "marker": "o",
                    "size": 10,
                    "alpha": 0.6,
                },
                "reconstructed": {
                    "color": "red",
                    "marker": "x",
                    "size": 10,
                    "alpha": 0.6,
                },
            },
            fig_size=(16, 9),  # width, height in inches
            device="cpu",
        )

        cd_wthattr = chamfer_with_attr_loss(
            predicted_filtered_radar_data,
            filtered_radar_data,
            config["device"],
            w_xyz=1.0,
            w_velocity=0.10,
            w_rcs=0.01,
        )
        # assert (
        #     np.abs(cd - cd_wthattr[1]["xyz"]) < 1e-3
        # ), f"Chamfer distance mismatch: {cd} vs {cd_wthattr[1]['xyz']}"
        cds.append(cd_wthattr)
        paths.append(save_path)
    return cds,paths


def pretrain_autoencoder(
    dataset,
    val_dataset,
    config,
    checkpoint_dir=None,
    run_id=None,
    tb_dir=None,
    plot_dir=None,
):
    """
    Pre-train PointNeXt from scratch as an autoencoder on UVZ point clouds.
    """
    worker_init_fn = set_seed(config["seed"])
    writer = SummaryWriter(f"{tb_dir}/{_short_last_segments(run_id, num_underscore_segments=29)}")
    writer.add_text("run_id", run_id)

    # Log hyperparameters
    config["train_size"] = len(dataset)
    config["node_name"] = os.uname().nodename
    config["gpu_name"] = torch.cuda.get_device_name(0)
    config["run_id"] = run_id
    torch.cuda.reset_peak_memory_stats()

    autoencoder = PretrainedPointNeXtEncoderPointAE(
        d_model=config["latent_dim"],
        output_points=config["num_points"],
        seq_length=config["latent_seq_length"],
        query_latent_pool_nhead=config["query_num_heads"],
        query_latent_pool_dropout=config["query_dropout"],
        device=config["device"],
        decoder_model=config["point_ae_model"][10:],
        num_decoder_layers=config["decoder_num_layers"],
        num_decoder_head=config["decoder_num_heads"],
        decoder_dropout=config["decoder_dropout"],
        pointnext_config=config["pointnext_config"],
        ball_query_nsample=config["ae_ball_query_nsample"],
        ball_query_radius=config["ae_ball_query_radius"],
        # 48 nsample, 8 m radius
        # pointnext_config="scannet/pointnext-s.yaml",
        output_dim=7,
    ).to(config["device"])

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),  # Much simpler!
        lr=config["ae_lr"],  # Higher LR for autoencoder
        weight_decay=config["ae_decay"],
    )

    #need to add LR scheduler
    if config["ae_lr_scheduler_type"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["ae_lr_decay_step"], gamma=config["ae_lr_decay_gamma"])
    elif config["ae_lr_scheduler_type"] == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["ae_lr"]/config['ae_lr_clr_factor'], max_lr=config["ae_lr"]*config['ae_lr_clr_factor'], step_size_up=config["ae_lr_decay_step"], mode=config["ae_lr_clr_mode"], cycle_momentum=False)
    else:
        # Default to no scheduler (constant LR)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    dataloader = makeDataloaders(
        dataset,
        config,
        is_train=True,
    )
    eval_dataloader = makeDataloaders(
        val_dataset,
        config,
        is_train=False,
    )
    start_epoch = load_ae_checkpoint(autoencoder, optimizer,lr_scheduler, config)
    num_epochs = config["ae_epochs"]
    print(f"Starting autoencoder pretraining from epoch {start_epoch + 1} to {num_epochs}...")
    global_step = 0
    # set start epoch, as start_epoch
    epoch_trange = trange(
        start_epoch, num_epochs, desc="AE Pretrain", unit="epoch", initial=start_epoch
    )
    saved_epoch = 0
    eval_loss = 0
    prev_paths = None
    for epoch in epoch_trange:

        avg_train_loss, global_step,time_records,avg_xyz_cd,avg_7d_cd = train_eval_ae_epoch(
            autoencoder, dataloader, optimizer, lr_scheduler, config, writer, global_step, train=True
        )

        epoch_trange.set_postfix(
            {"tr": f"{avg_train_loss:.4f}", "val": f"{eval_loss:.4f}"}
        )

        # ✅ LOG EPOCH METRICS
        writer.add_scalar("ae/loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("ae/lr", lr_scheduler.get_last_lr()[0], epoch + 1)
        writer.add_scalar("ae/avg_xyz_cd/train", avg_xyz_cd, epoch + 1)
        writer.add_scalar("ae/avg_7d_cd/train", avg_7d_cd, epoch + 1)

            
        if True:
            sum_time = sum(time_records.values())
            for k in time_records:
                writer.add_scalar(f"train/time/{k}", time_records[k]/sum_time, epoch + 1)
        if (epoch + 1) % config["eval_every"] == 0:
            eval_loss, _,time_recordsm,avg_xyz_cd,avg_7d_cd = train_eval_ae_epoch(
                autoencoder,
                eval_dataloader,
                optimizer,
                lr_scheduler,
                config,
                writer,
                global_step,
                train=False,
            )
            writer.add_scalar("ae/loss/val", eval_loss, epoch + 1)
            writer.add_scalar("ae/avg_xyz_cd/val", avg_xyz_cd, epoch + 1)
            writer.add_scalar("ae/avg_7d_cd/val", avg_7d_cd, epoch + 1)


            # time_records_batch = {
            #     "data_prep_time": time1 - time0,
            #     "forward_time": time2 - time1,
            #     "denormalize_time": time3 - time2,
            #     "loss_time": time4 - time3,
            #     "backward_time": time5 - time4,
            # }
            if True:
                sum_time = sum(time_records.values())
                for k in time_records:
                    writer.add_scalar(f"val/time/{k}", time_records[k]/sum_time, epoch + 1)

        if (epoch + 1) % config["gpu_log_every"] == 0:
            stats = query_gpu_stats(gpu_index=0)
            if stats is not None:
                writer.add_scalar("gpu/utilization_pct", stats["util_gpu"], epoch + 1)
                writer.add_scalar("gpu/power_w", stats["power_w"], epoch + 1)
                writer.add_scalar(
                    "gpu/memory_used_mib", stats["mem_used_mib"], epoch + 1
                )
                writer.add_scalar(
                    "gpu/memory_total_mib", stats["mem_total_mib"], epoch + 1
                )

        if (
            config["plot_every"] != 0 and (epoch + 1) % config["plot_every"] == 0
        ) or epoch == num_epochs - 1:
            if prev_paths is not None:
                for path in prev_paths:
                    if os.path.exists(path):
                        os.remove(path)
            cds,prev_tr_paths = plot_ae(
                "train",
                dataset,
                autoencoder,
                config,
                plot_dir,
                run_id,
                epoch,
            )
            cds,prev_val_paths = plot_ae(
                "val",
                val_dataset,
                autoencoder,
                config,
                plot_dir,
                run_id,
                epoch,
            )
            # prev_paths = prev_tr_paths + prev_val_paths

        if (epoch + 1) % config["save_every"] == 0 or epoch == num_epochs - 1:
            _short_last_segments(run_id, num_underscore_segments=29)
            checkpoint_path = os.path.join(checkpoint_dir, f"{_short_last_segments(run_id, num_underscore_segments=29)}_at{epoch+1}.pth")
            prev_path = os.path.join(checkpoint_dir, f"{_short_last_segments(run_id, num_underscore_segments=29)}_at{saved_epoch+1}.pth")
            if os.path.exists(prev_path):
                os.remove(prev_path)
            saved_epoch = epoch

            save_ae_checkpoint(
                autoencoder,
                optimizer,
                lr_scheduler,
                epoch + 1,
                {
                    "train_loss": avg_train_loss,
                },
                checkpoint_path,
                config,
            )

    tblogHparam(
        config,
        writer,
        {"ae_train_loss": avg_train_loss, "ae_val_loss": eval_loss},
    )
    writer.flush()
    writer.close()
    print("\n✅ Autoencoder pretraining complete!")
    return autoencoder


def create_uvz_comparison_plot(gt_uvz, gen_uvz, title):
    """
    Create a matplotlib figure comparing ground truth and generated UVZ point clouds.
    Args:
        gt_uvz: (N, 3) numpy array of ground truth UVZ points
        gen_uvz: (N, 3) numpy array of generated UVZ points
        title: Title for the plot
    Returns:

        fig: Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(
        121,
    )
    ax1.scatter(gt_uvz[:, 0], gt_uvz[:, 1], s=1)
    ax1.set_title("UV Plane (Z as color)")
    sc = ax1.scatter(gen_uvz[:, 0], gen_uvz[:, 1], s=1)
    ax1.legend(["Ground Truth UVZ", "Generated UVZ"])
    ax1.set_xlabel("U")
    ax1.set_ylabel("V")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        gen_uvz[:, 0], gen_uvz[:, 1], gen_uvz[:, 2], c="r", s=1, label="Generated UVZ"
    )
    ax2.scatter(
        gt_uvz[:, 0],
        gt_uvz[:, 1],
        gt_uvz[:, 2],
        c="b",
        s=1,
        alpha=0.1,
        label="Ground Truth UVZ",
    )
    ax2.set_title("3D Comparison")
    ax2.legend(["Generated UVZ", "Ground Truth UVZ"])
    ax2.set_xlabel("U")
    ax2.set_ylabel("V")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=30, azim=120)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


### Phase 2: Train DiT with Frozen Autoencoder
def train_dit_diffusion(
    dataset, config, autoencoder, checkpoint_dir=None, run_id=None, tb_dir=None
):
    """
    Train DiT diffusion model with FROZEN autoencoder.
    This learns to denoise in the latent space.
    """
    worker_init_fn = set_seed(config["seed"])
    writer = SummaryWriter(f"{tb_dir}/{run_id}/dit_training")

    # Log hyperparameters
    writer.add_text("config/dit_lr", str(config["dit_lr"]))
    writer.add_text("config/dit_epochs", str(config["dit_epochs"]))
    writer.add_text("config/batch_size", str(config["batch_size"]))
    writer.add_text("config/latent_dim", str(config["latent_dim"]))
    writer.add_text("config/latent_seq_length", str(config["latent_seq_length"]))
    writer.add_text("config/num_points", str(config["num_points"]))
    writer.add_text("config/dataset_size", str(len(dataset)))
    torch.cuda.reset_peak_memory_stats()

    ddpm = DitDDPM(
        autoencoder=autoencoder,
        num_train_timesteps=config["num_train_timesteps"],
        device=config["device"],
        # use_pointnext=config["use_pointnext"],
        # geometric_loss_weight=0,
        # image_size=config["scaled_image_size"],
        d_model=config["latent_dim"],
        seq_length=config["latent_seq_length"],
        # output_points=config["num_points"]
    )

    # Only train DiT + depth tokenizer
    trainable_params = []
    trainable_params.extend(ddpm.model.parameters())  # DiT transformer
    trainable_params.extend(ddpm.depth_tokenizer.parameters())  # Depth tokenizer

    print(f"DiT trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    writer.add_text(
        "model/trainable_params", f"{sum(p.numel() for p in trainable_params):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["dit_lr"], weight_decay=config["dit_decay"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        # collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=config["num_workers"],
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * config["dit_epochs"],
    )

    num_epochs = config["dit_epochs"]
    global_step = 0

    epoch_trange = trange(num_epochs, desc="DiT Training")
    for epoch in epoch_trange:
        ddpm.model.train()
        epoch_loss = 0
        batch_losses = []

        for batch in tqdm(dataloader, leave=False, desc="Batches"):
            rgb = batch["camera_front"].to(config["device"])
            depth = batch["depth_image"].to(config["device"])
            uvz = batch["uvz"].to(config["device"])
            clip_feature = batch["clip_feature"].to(config["device"])

            batch_size = rgb.shape[0]

            # Prepare condition
            condition = ddpm.prepare_condition(depth, clip_feature)

            # Sample timesteps
            timesteps = torch.randint(
                0,
                ddpm.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=config["device"],
            ).long()

            with torch.no_grad():
                latent = ddpm.autoencoder.encode(uvz, training=True)

            # DIFFUSION TRAINING (only this is trainable)
            noise = torch.randn_like(latent)
            noisy_latent = ddpm.noise_scheduler.add_noise(latent, noise, timesteps)

            noise_pred = ddpm.model(
                noisy_latent, timesteps, encoder_hidden_states=condition
            )

            loss = F.mse_loss(noise_pred, noise)

            # Backprop (only updates DiT + depth tokenizer)
            optimizer.zero_grad()
            loss.backward()

            # Track gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            lr_scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)

            # ✅ LOG BATCH METRICS TO TENSORBOARD
            writer.add_scalar("dit/batch_loss", batch_loss, global_step)
            writer.add_scalar("dit/grad_norm", grad_norm.item(), global_step)
            writer.add_scalar(
                "dit/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            # Log timestep and noise statistics
            if global_step % 10 == 0:
                writer.add_scalar(
                    "dit/timestep_mean", timesteps.float().mean().item(), global_step
                )
                writer.add_scalar(
                    "dit/timestep_std", timesteps.float().std().item(), global_step
                )
                writer.add_scalar("dit/noise_mean", noise.mean().item(), global_step)
                writer.add_scalar("dit/noise_std", noise.std().item(), global_step)
                writer.add_scalar(
                    "dit/noise_pred_mean", noise_pred.mean().item(), global_step
                )
                writer.add_scalar(
                    "dit/noise_pred_std", noise_pred.std().item(), global_step
                )

                # Log MSE at different timesteps
                for t_bin in [0, 250, 500, 750, 999]:
                    mask = (timesteps >= t_bin) & (timesteps < t_bin + 50)
                    if mask.any():
                        t_loss = F.mse_loss(noise_pred[mask], noise[mask]).item()
                        writer.add_scalar(f"dit/loss_at_t{t_bin}", t_loss, global_step)

            global_step += 1

        # Epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        std_loss = np.std(batch_losses)

        epoch_trange.set_postfix({"Diffusion Loss": f"{avg_loss:.4f}"})

        # ✅ LOG EPOCH METRICS
        writer.add_scalar("dit/epoch_loss_mean", avg_loss, epoch)
        writer.add_scalar("dit/epoch_loss_std", std_loss, epoch)

        # ✅ GENERATE AND VISUALIZE SAMPLES EVERY 20 EPOCHS
        if (epoch + 1) % 20 == 0:
            ddpm.model.eval()
            with torch.no_grad():
                # Take first sample from dataset
                sample_data = dataset[0]

                # Generate
                generated, outputs = ddpm.sample(
                    sample_data["depth_image"].unsqueeze(0).to(config["device"]),
                    sample_data["clip_feature"].unsqueeze(0).to(config["device"]),
                    num_inference_steps=50,
                    seed=config["seed"],
                )

                if "uvz" in generated:
                    original_uvz = sample_data["uvz"].to(config["device"])

                    # Visualize reconstruction
                    original_uvz[:, 0] = original_uvz[:, 0] * (
                        config["original_image_size"][0]
                        / config["scaled_image_size"][0]
                    )
                    original_uvz[:, 1] = original_uvz[:, 1] * (
                        config["original_image_size"][1]
                        / config["scaled_image_size"][1]
                    )
                    # Visualize
                    for t in outputs:
                        fig = create_uvz_comparison_plot(
                            original_uvz.cpu().numpy(),
                            generated["uvz"][0].cpu().numpy(),
                            title=f"Epoch {epoch+1} Timestep {t}",
                        )

                        writer.add_figure("dit/generated_sample", fig, epoch)
                        plt.close(fig)

                    # Log Chamfer distance
                    chamfer_dist = chamfer_distance(
                        generated["uvz"],
                        original_uvz.unsqueeze(0).to(config["device"]),
                    )
                    writer.add_scalar(
                        "dit/sample_chamfer_distance", chamfer_dist.item(), epoch
                    )

                    # Log confidence statistics
                    writer.add_scalar(
                        "dit/confidence_mean",
                        generated["confidence"].mean().item(),
                        epoch,
                    )
                    writer.add_scalar(
                        "dit/confidence_std",
                        generated["confidence"].std().item(),
                        epoch,
                    )

            ddpm.model.train()

        # ✅ LOG HISTOGRAMS EVERY 20 EPOCHS
        if (epoch + 1) % 20 == 0:
            for name, param in ddpm.model.named_parameters():
                writer.add_histogram(f"dit/model/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"dit/model/{name}.grad", param.grad, epoch)

            for name, param in ddpm.depth_tokenizer.named_parameters():
                writer.add_histogram(f"dit/depth_tokenizer/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(
                        f"dit/depth_tokenizer/{name}.grad", param.grad, epoch
                    )

    writer.close()
    print("\n✅ DiT training complete!")
    # Return anything that need to be saved
    return ddpm, lr_scheduler, optimizer, global_step, epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test DDPM for radar point cloud generation."
    )
    parser.add_argument(
        "--scene_ids",
        type=int,
        nargs="*",
        default=[],
        help="Scene IDs to use for training (e.g., 0 1 2 3 4).",
    )
    # num input frame
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=24000,
        help="Number of input frames to use.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="man-mini",
        help="Dataset file to use (e.g., 'man-mini').",
    )  # else   man-full
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--radar_channel",
        type=str,
        default="RADAR_RIGHT_FRONT",
        help="Radar channel to use.",
    )
    parser.add_argument(
        "--camera_channel",
        type=str,
        default="CAMERA_RIGHT_FRONT",
        help="Camera channel to use.",
    )
    parser.add_argument(
        "--original_image_size",
        type=int,
        nargs=2,
        default=[943, 1860],
        help="Original image size (height, width).",
    )
    parser.add_argument(
        "--scaled_image_size",
        type=int,
        nargs=2,
        default=[16, 32],
        help="Scaled image size (height, width).",
    )

    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate the model every N epochs.",
    )
    # infer steps
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDPM inference steps.",
    )
    # laten len and dim
    parser.add_argument(
        "--latent_seq_length",
        type=int,
        default=256,
        help="Latent sequence length for DiT.",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=1024, help="Latent dimension for DiT."
    )
    # ae altent normalizning std
    parser.add_argument(
        "--ae_latent_normalizing_std",
        type=float,
        default=1.5e-4,
        help="Standard deviation for latent normalization in autoencoder.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--ae_lr",
        type=float,
        default=1e-3,
        help="Learning rate for autoencoder pretraining.",
    )
    parser.add_argument(
        "--ae_decay",
        type=float,
        default=0.01,
        help="Decay rate for autoencoder learning rate.",
    )
    parser.add_argument(
        "--ae_epochs",
        type=int,
        default=50,
        help="Number of epochs for autoencoder pretraining.",
    )
    parser.add_argument(
        "--ae_ball_query_nsample",
        type=int,
        default=48,
        help="Number of samples for ball query in autoencoder.",
    )
    parser.add_argument(
        "--ae_ball_query_radius",
        type=float,
        default=8.0,
        help="Radius for ball query in autoencoder.",
    )
    parser.add_argument(
        "--dit_lr", type=float, default=1e-4, help="Learning rate for DiT training."
    )
    parser.add_argument(
        "--dit_decay",
        type=float,
        default=0.01,
        help="Decay rate for DiT learning rate.",
    )
    parser.add_argument(
        "--dit_epochs", type=int, default=100, help="Number of epochs for DiT training."
    )

    # parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--plot_every", type=int, default=100, help="Plot samples every N epochs."
    )
    parser.add_argument(
        "--save_every", type=int, default=1000, help="Save checkpoint every N epochs."
    )
    # gpu_log_every
    parser.add_argument(
        "--gpu_log_every",
        type=int,
        default=100,
        help="Log GPU memory usage every N epochs.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--point_ae_model",
        type=str,
        default="pointnext-attn",
        help="Type of point cloud autoencoder to use (e.g., 'pointnext-attn', 'attn-attn').",
    )
    parser.add_argument(
        "--point_ae_checkpoint",
        type=str,
        default="",
        help="Path to pretrained point cloud autoencoder checkpoint.",
    )
    parser.add_argument(
        "--dit_checkpoint",
        type=str,
        default="",
        help="Path to pretrained DiT checkpoint.",
    )
    parser.add_argument(
        "--geometric_loss_weight",
        type=float,
        default=0.0,
        help="Weight for geometric loss.",
    )
    parser.add_argument(
        "--num_points", type=int, default=500, help="Number of points for sampling."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--seed_model", type=int, default=42, help="Random seed for model initialization."
    )
    # parser.add_argument(
    #     "--num_encoder_layers",
    #     type=int,
    #     default=4,
    #     help="Number of encoder layers for transformer point AE.",
    # )
    # parser.add_argument(
    #     "--num_decoder_layers",
    #     type=int,
    #     default=4,
    #     help="Number of decoder layers for transformer point AE.",
    # )
    # n dit block
    parser.add_argument(
        "--num_transformer_blocks",
        type=int,
        default=12,
        help="Number of transformer blocks in DiT model.",
    )
    # n head
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=16,
        help="Number of attention heads in DiT model.",
    )

    parser.add_argument(
        "--max_voxel_grid_depth", type=float, default=250.0, help="Maximum depth value."
    )
    # mn max u and v pixel roi for training
    parser.add_argument(
        "--training_roi",
        type=int,
        nargs=4,
        default=[0, 943, 0, 1860],
        help="Region of interest for training (min_v/height, max_v, min_u/width, max_u).",
    )

    parser.add_argument(
        "--depth_voxel_grid_bins", type=int, default=8, help="Number of depth bins."
    )
    parser.add_argument(
        "--use_global_avg_pool",
        action="store_true",
        help="Whether to use global average pooling to process VAE latents.",
    )
    # zero_cond
    parser.add_argument(
        "--zero_conditioning",
        action="store_true",
        help="Whether to use zero conditioning for depth and CLIP features.",
    )
    # learn sigma
    parser.add_argument(
        "--learn_sigma",
        action="store_true",
        help="Whether the DiT model should learn the noise sigma.",
    )
    # patch size
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="Patch size for DiT model.",
    )
    # grid_binary_range "0-1" or "neg1-1"
    parser.add_argument(
        "--grid_binary_range",
        type=str,
        default="0-1",
        help="Range for grid binary encoding ('0-1' or 'neg1-1').",
    )
    # aux_occ_weight , aux_occ_scale
    parser.add_argument(
        "--aux_occ_weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary occupancy loss.",
    )
    parser.add_argument(
        "--aux_occ_scale",
        type=float,
        default=5.0,
        help="Scale for auxiliary occupancy loss.",
    )
    # query_num_heads
    parser.add_argument(
        "--query_num_heads",
        type=int,
        default=8,
        help="Number of attention heads for query latent pooling in point AE.",
    )
    # query_dropout
    parser.add_argument(
        "--query_dropout",
        type=float,
        default=0.0,
        help="Dropout rate for query latent pooling in point AE.",
    )
    # decoder_num_layers
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=6,
        help="Number of decoder layers in point AE.",
    )
    # decoder_num_heads
    parser.add_argument(
        "--decoder_num_heads",
        type=int,
        default=8,
        help="Number of attention heads in decoder of point AE.",
    )
    # decoder_dropout
    parser.add_argument(
        "--decoder_dropout",
        type=float,
        default=0.0,
        help="Dropout rate in decoder of point AE.",
    )
    # pointnext_config
    parser.add_argument(
        "--pointnext_config",
        type=str,
        default="scannet/untitled.yaml",
        help="Path to PointNeXt configuration file.",
    )
    # loss_type
    parser.add_argument(
        "--ae_loss_type",
        type=str,
        default="chamfer",
        help="Type of loss function for autoencoder ('chamfer' or 'chamfer-attr').",
    )
    # w attr loss, tripple, weight for position, velocity,and rcs
    parser.add_argument(
        "--ae_weight_attr_loss",
        type=float,
        nargs=3,
        default=[1.0, 0.1, 0.01],
        help="Weights for attribute loss components (position, velocity, RCS).",
    )
    parser.add_argument(
        "--ddpm_clip_sample",
        action="store_true",
        help="Whether to clip DDPM samples to valid range.",
    )
    # exp_name
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="tag, used for mainly filter tensorboard runs",
    )
    parser.add_argument(    
        "--hungarian_use_greedy",
        action="store_true",
        help="Whether to use greedy matching instead of Hungarian algorithm for point cloud matching in attribute loss.",
    )
    parser.add_argument(
        "--sk_eps",
        type=float,
        default=0.05,
        help="Epsilon value for Sinkhorn distance in attribute loss.",
    )   
    parser.add_argument(
        "--sk_detach",
        action="store_true",
        help="Whether to detach the cost matrix in Sinkhorn distance computation for attribute loss.",
    )
    parser.add_argument(
        "--ot_clip",
        action="store_true",
        help="Whether to clip the optimal transport cost in attribute loss to prevent exploding gradients.",
    )

    parser.add_argument("--ae_weight_good_loss", type=float,
        nargs=3,
        default=[1.0, 0.1, 0.01],
        help="Weight for good point loss in autoencoder: CD, OT, and KNN losses.")


    parser.add_argument(
        "--ae_lr_decay_step",
        type=int,
        default=100000,
        help="Step size for AE learning rate decay or if Cyclic LR, number of steps to reach max learning rate (i.e., half cycle length).",
    )
    parser.add_argument(
        "--ae_lr_decay_gamma",
        type=float,
        default=0.5,
        help="Gamma for AE learning rate decay.",
    )
    parser.add_argument(
        "--ae_lr_clr_factor",
        type=float,
        default=10.0,
        help="Factor to determine base learning rate for triangular cyclic learning rate scheduler (base_lr = ae_lr / ae_lr_clr_factor), max learning rate is ae_lr * ae_lr_factor.",
    )
    parser.add_argument(        "--ae_lr_scheduler_type",  
        type=str, 
        default="cyclic",  #step. clr,none
        help="Type of learning rate scheduler for autoencoder training (e.g., 'step', 'clr', 'none'). If 'step', uses StepLR with ae_lr_decay_step and ae_lr_decay_gamma. If 'clr', uses CyclicLR with base_lr = ae_lr / ae_lr_clr_factor and max_lr = ae_lr * ae_lr_clr_factor. If 'none', no learning rate scheduler is used."
    )
    # ae_lr_clr_mode
    parser.add_argument(
        "--ae_lr_clr_mode",
        type=str,
        default="triangular",
        help="Mode for CyclicLR if ae_lr_scheduler_type is 'clr' (e.g., 'triangular', 'triangular2', 'exp_range').",
    )   
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Type of learning rate scheduler ('cosine' or 'step' or 'constant') to use for DiT training.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=30,
        help="Step size for StepLR if lr_scheduler is 'step'.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5,
        help="Gamma for StepLR if lr_scheduler is 'step'.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: int - random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For DataLoader worker processes
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def get_pointnextdit_runid(config):

    scene_str = "-".join(map(str, sorted(config["scene_ids"])))

    return f"{config['exp_name']}_e{config['dit_epochs']}_sc{scene_str}_{config['data_file'].replace('man-','')}_np{config['num_points']}_fr{config['num_input_frames']}_sd{config['seed']}_latdim{config['latent_dim']}_latseq{config['latent_seq_length']}_decay{config['dit_decay']:.2E}_lr{config['dit_lr']:.2E}_bs{config['batch_size']}_hd{config['num_attention_heads']}_blk{config['num_transformer_blocks']}_latstd{config['ae_latent_normalizing_std']:.2E}_clip{int(config['ddpm_clip_sample'])}_pl{config['use_global_avg_pool']}_sig{config['learn_sigma']}_zc{config['zero_conditioning']}"


def get_pointnext_runid(config):

    scene_str = "-".join(map(str, sorted(config["scene_ids"])))
    good_loss_str = '-'.join([f'{w:.2E}' for w in config['ae_weight_good_loss']])

    return f"{config['exp_name']}_e{config['ae_epochs']}_sc{scene_str}_{config['data_file'].replace('man-','')}_np{config['num_points']}_fr{config['num_input_frames']}_model{config['point_ae_model'].replace('modelpointnext-','')}_sd{config['seed']}_latdim{config['latent_dim']}_latseq{config['latent_seq_length']}_decay{config['ae_decay']:.2E}_lr{config['ae_lr']:.2E}_bs{config['batch_size']}_qnh{config['query_num_heads']}_qdrop{config['query_dropout']:.2E}_dnl{config['decoder_num_layers']}_dnh{config['decoder_num_heads']}_ddrop{config['decoder_dropout']:.2E}_{config['pointnext_config'].replace('/', '_').replace('.yaml','')}_ltype{config['ae_loss_type']}_wattr{'-'.join([f'{w:.2E}' for w in config['ae_weight_attr_loss']])}_bqn{config['ae_ball_query_nsample']}_bqr{config['ae_ball_query_radius']}_greedy{int(config['hungarian_use_greedy'])}_clip{int(config['ddpm_clip_sample'])}_lrdec{config['ae_lr_decay_step']}_{config['ae_lr_decay_gamma']:.2E}_skeps{config['sk_eps']}_skdetach{int(config['sk_detach'])}_otclip{int(config['ot_clip'])}_goodloss{good_loss_str}_aeclr{config['ae_lr_clr_factor']:.2E}_aels{config['ae_lr_scheduler_type']}_clrmde{config['ae_lr_clr_mode']}"


def get_runid(config):

    scene_str = "-".join(map(str, sorted(config["scene_ids"])))
    roi_text = f"{config['training_roi'][0]}-{config['training_roi'][1]}_{config['training_roi'][2]}-{config['training_roi'][3]}"
    return f"{config['depth_voxel_grid_bins']}_{config['scaled_image_size'][0]}x{config['scaled_image_size'][1]}_e{config['dit_epochs']}_b{config['batch_size']}_sc{scene_str}_fr{config['num_input_frames']}_sd{config['seed']}_pl{config['use_global_avg_pool']}_sig{config['learn_sigma']}_zc{config['zero_conditioning']}_pat{config['patch_size']}_blk{config['num_transformer_blocks']}_hd{config['num_attention_heads']}_lat{config['latent_dim']}_{config['grid_binary_range']}_w{config['aux_occ_weight']:.2E}_scal{config['aux_occ_scale']:.1f}_roi{roi_text}_mxd{config['max_voxel_grid_depth']:.0f}"
    # return f"{config['depth_voxel_grid_bins']}bins_{config['scaled_image_size'][0]}x{config['scaled_image_size'][1]}_{config['dit_epochs']}epochs_batch{config['batch_size']}_scenes{scene_str}_numframes{config['num_input_frames']}_seed{config['seed']}_pool{config['use_global_avg_pool']}_sigma{config['learn_sigma']}_zeroCond{config['zero_conditioning']}_patch{config['patch_size']}_transBlocks{config['num_transformer_blocks']}_attnHeads{config['num_attention_heads']}_latentdim{config['latent_dim']}_grange{config['grid_binary_range']}_auxoccw{config['aux_occ_weight']:.2E}_auxoccs{config['aux_occ_scale']:.1f}_roi{roi_text}"


def query_gpu_stats(gpu_index: int = 0) -> Optional[Dict[str, float]]:
    """
    Returns dict with utilization.gpu (%), power.draw (W), memory.used (MiB), memory.total (MiB)
    using nvidia-smi. Returns None if nvidia-smi is unavailable.
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return None
        util_str, power_str, mem_used_str, mem_total_str = [
            x.strip() for x in out.split(",")
        ]
        return {
            "util_gpu": float(util_str),
            "power_w": float(power_str),
            "mem_used_mib": float(mem_used_str),
            "mem_total_mib": float(mem_total_str),
        }
    except Exception as e:
        print("Could not query GPU stats via nvidia-smi.", e)  # noqa: T201
        return None


def makeOptimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["dit_lr"],
        weight_decay=config["dit_decay"],
    )


def tblogHparam(
    config, writer: SummaryWriter, metrics: dict, run_name="", global_step=None
):
    # keep only simple types (tensorboard hparams plugin likes scalars/strings/bools)
    hparams = {}
    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            for i in range(len(v)):
                hparams[f"{k}_{i}"] = v[i]
        elif isinstance(v, (int, float, str, bool)):
            hparams[k] = v
        else:
            print(f"Warning: skipping hparam {k} with type {type(v)}")


    # metrics must be scalar numbers, prepend "hparam/" to avoid conflicts with other metrics
    hmetrics = {f"hparam/{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    # IMPORTANT: do this exactly once
    writer.add_hparams(hparams, hmetrics)
    print(f"Logged hyperparameters and metrics to TensorBoard:")
    print("Hyperparameters:")
    for k, v in hparams.items():
        print(f"  {k}: {v}")

    print("Metrics:")
    for k, v in hmetrics.items():
        print(f"  {k}: {v}")
    # make sure it hits disk
    writer.flush()


def makeDataset(
    config,
    plot_dir: str = None,
    get_occ_grid=True,
    get_camera=True,
    get_wan_vae=True,
):
    """
    return train and val dataloaders
    """
    dataset = MANDataset(
        scene_ids=config["scene_ids"],
        data_file=config["data_file"],
        device=config["device"],
        radar_channel=config["radar_channel"],
        camera_channel=config["camera_channel"],
        double_flip_images=False,
        coord_only=False,  # not providing vel and rcs
        visualize_uvz=False,  # plotting, slow
        scaled_image_size=config["scaled_image_size"],  # 512 1024 dead
        n_p=config["num_points"],
        max_depth=config["max_voxel_grid_depth"],
        roi=config["training_roi"],
        depth_bins=config["depth_voxel_grid_bins"],
        get_clip=False,
        get_depth=False,
        get_occ_grid=get_occ_grid,
        get_camera=get_camera,
        wan_vae=get_wan_vae,  # for vae-based training
        wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
        viz_dir=plot_dir,
        grid_binary_range=config["grid_binary_range"],  # "0-1" or "neg1-1"
        keep_frames=config["num_input_frames"],
    )
    assert (
        len(dataset) == config["num_input_frames"]
    ), f"Dataset length {len(dataset)} does not match num_input_frames {config['num_input_frames']}"
    return dataset


def splitDataset(dataset: Dataset, split: float = 0.8):

    val_split = int(split * len(dataset))
    train_dataset = Subset(dataset, range(0, val_split))
    val_dataset = Subset(dataset, range(val_split, len(dataset)))
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def makeDataloaders(
    dataset: Dataset,
    config,
    is_train: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=is_train,
        num_workers=config["num_workers"],
    )
