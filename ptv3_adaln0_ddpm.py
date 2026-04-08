"""
diffusion class for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]],
DDPM form DiT paper's repo
PTv3 as denoiser architecture, from official repo
"""

from logging import config
from random import random, seed
import sys, os, math
import sys, subprocess
from typing import Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import textwrap
from mono_adaln0_ddpm_2 import makeLR_Scheduler
from mono_adaln0_import import makeDiTSetModel

import geomloss
from copy import deepcopy

sys.path.insert(0, "/home/palakons/DiT")
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


sys.path.insert(0, "/home/palakons/Pointcept")
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)


import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Subset
import torch.nn.functional as F
from dit_ddpm_class import (
    parse_args,
    set_seed,
    query_gpu_stats,
    makeDataset,
    splitDataset,
    makeDataloaders,
    makeOptimizer,
    tblogHparam
)
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from pytorch3d.loss import chamfer_distance   as  pt3d_chamfer_distance



class SimplePointUNet(nn.Module):
    """
    A minimal UNet-style denoiser for (B, 3, N) point clouds.
    This is a 1D UNet (conv over N), with skip connections and no stochasticity.
    It should be able to overfit small data and fit points well.
    """
    def __init__(self, in_channels=3, base_channels=64, out_channels=3, num_layers=4, t_embed_dim=32):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = in_channels + t_embed_dim
        # Encoder
        for i in range(num_layers):
            self.encoders.append(nn.Conv1d(ch, base_channels * 2 ** i, kernel_size=3, padding=1))
            self.pools.append(nn.MaxPool1d(2))
            ch = base_channels * 2 ** i
        # Bottleneck
        self.bottleneck = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        # Decoder
        for i in reversed(range(num_layers)):
            self.ups.append(nn.ConvTranspose1d(ch, base_channels * 2 ** i, kernel_size=2, stride=2))
            self.decoders.append(nn.Conv1d(base_channels * 2 ** i * 2, base_channels * 2 ** i, kernel_size=3, padding=1))
            ch = base_channels * 2 ** i
        self.final = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x, t=None, condition=None, **kwargs):
        # x: (B, 3, N), t: (B,) or (B,1)
        B, C, N = x.shape
        if t is None:
            t = torch.zeros(B, device=x.device)
        t = t.view(B, 1).float() / 1000.0  # scale for stability
        t_emb = self.t_mlp(t)  # (B, t_embed_dim)
        # Expand t_emb to (B, t_embed_dim, N)
        t_emb_exp = t_emb.unsqueeze(-1).expand(-1, self.t_embed_dim, N)
        # Concatenate t embedding as extra channels
        out = torch.cat([x, t_emb_exp], dim=1)
        enc_feats = []
        # Encoder
        for enc, pool in zip(self.encoders, self.pools):
            out = F.relu(enc(out))
            enc_feats.append(out)
            out = pool(out)
        # Bottleneck
        out = F.relu(self.bottleneck(out))
        # Decoder
        for up, dec, enc_feat in zip(self.ups, self.decoders, reversed(enc_feats)):
            out = up(out)
            # Pad if needed (for odd N)
            if out.shape[-1] > enc_feat.shape[-1]:
                out = out[..., :enc_feat.shape[-1]]
            elif out.shape[-1] < enc_feat.shape[-1]:
                pad = enc_feat.shape[-1] - out.shape[-1]
                out = F.pad(out, (0, pad))
            out = torch.cat([out, enc_feat], dim=1)
            out = F.relu(dec(out))
        out = self.final(out)
        return out  # (B, 3, N)

class PTv3Dnsr(nn.Module):
    def __init__(
        self,
        n_in_channels=3,
        context_channels=256,
        out_channels=3,
        grid_size=0.02,
        shuffle_orders=True,
        serialized_inverse=True,
        n_stages=5, # Allow sweeping n_stages from 2 to 5
        seed=42,
        backbone_type="full",
    ):
        """
        PTv3 Denoiser for DDPM. Supports dynamic depth via n_stages.
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.n_in_channels = n_in_channels
        self.context_channels = context_channels
        self.grid_size = grid_size
        self.serialized_inverse = serialized_inverse

        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.SiLU(),
            nn.Linear(context_channels, context_channels),
        )

        # 2. Define standard 5-stage parameters
        full_stride = (2, 2, 2, 2)
        full_enc_depths = (2, 2, 2, 6, 2)
        full_enc_channels = (32, 64, 128, 256, 512)
        full_enc_num_head = (2, 4, 8, 16, 32)
        full_enc_patch_size = (48, 48, 48, 48, 48)
        
        full_dec_depths = (2, 2, 2, 2)
        full_dec_channels = (64, 64, 128, 256)
        full_dec_num_head = (4, 4, 8, 16)
        full_dec_patch_size = (48, 48, 48, 48)

        # 3. Cut off stages based on n_stages (must be at least 2)
        n_stages = max(2, min(n_stages, 5))
        stride = full_stride[:n_stages-1]
        enc_depths = full_enc_depths[:n_stages]
        enc_channels = full_enc_channels[:n_stages]
        enc_num_head = full_enc_num_head[:n_stages]
        enc_patch_size = full_enc_patch_size[:n_stages]
        
        dec_depths = full_dec_depths[:n_stages-1]
        dec_channels = full_dec_channels[:n_stages-1]
        dec_num_head = full_dec_num_head[:n_stages-1]
        dec_patch_size = full_dec_patch_size[:n_stages-1]

        # 4. PTv3 Backbone
        if backbone_type == "full":
            self.ptv3 = PointTransformerV3(
                in_channels=self.n_in_channels,
                enable_flash=True,
                pdnorm_ln=True, 
                pdnorm_adaptive=True,
                pdnorm_conditions=("DDPM",),
                shuffle_orders=shuffle_orders,
                enc_channels=enc_channels,
                enc_num_head=enc_num_head,
                enc_patch_size=enc_patch_size,
                dec_channels=dec_channels,
                dec_num_head=dec_num_head,
                dec_patch_size=dec_patch_size,
                qkv_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
                enable_rpe=False,
                stride=stride,
                enc_depths=enc_depths,
                dec_depths=dec_depths,
                drop_path=0.3,
                pdnorm_bn=False,
            )
        else: # backbone_type == "simple"
            self.ptv3 = PointTransformerV3(
                in_channels=self.n_in_channels,
                enable_flash=True,
                pdnorm_ln=True, 
                pdnorm_adaptive=True,
                pdnorm_conditions=("DDPM",),
                shuffle_orders=shuffle_orders,
                enc_channels=enc_channels,
                enc_num_head=enc_num_head,
                enc_patch_size=enc_patch_size,
                dec_channels=dec_channels,
                dec_num_head=dec_num_head,
                dec_patch_size=dec_patch_size,

                qkv_bias=True, # Enable qkv bias for better expressiveness, especially important when using fewer layers
                attn_drop=0.0, # Disable attention dropout for more stable training with normalized coords
                proj_drop=0.0, # Disable projection dropout for more stable training with normalized coords

                enable_rpe=False, # Disable RPE for more stable training with normalized coords
                stride=[1 for _ in stride], # Disable merging by setting stride to 1, which should help maintain point count and stability with normalized coords
                enc_depths=[1 for _ in enc_depths],
                dec_depths=[1 for _ in dec_depths],
                drop_path=0.0, # Disable drop path for more stable training with normalized coords
                pdnorm_bn=False, # Disable batch norm for more stable training with normalized coords
            )

        # 5. Output Head
        # dec_channels[0] is always the output dimension of the last decoder stage
        self.head = nn.Linear(dec_channels[0], out_channels*2)
        
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def get_time_embedding(self, timesteps):
        """Standard sinusoidal time embedding."""
        half_dim = self.context_channels // 2
        exponent = (
            -math.log(10000)
            * torch.arange(half_dim, device=timesteps.device)
            / half_dim
        )
        emb = timesteps[:, None].float() * exponent.exp()[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.time_mlp(emb)

    def forward(self, x, t, condition=None, **kwargs):
        """
        Forward pass for denoising.

        Args:
            x (torch.Tensor): Noisy point cloud of shape (B, C, N).
            t (torch.Tensor): Timesteps of shape (B,).
            condition (torch.Tensor, optional): Conditioning latent.
            **kwargs: Additional arguments from diffusion loop.

        Returns:
            torch.Tensor: Predicted noise and variance in shape (B, C_out, N).
        """
        B, C, N = x.shape
        
        # --- FIX: Ensure coordinates are not interleaved ---
        # Transpose (B, 3, N) -> (B, N, 3) so that flatten gives (x, y, z) rows
        x_pts = x.transpose(1, 2).contiguous()
        coord = x_pts.reshape(-1, 3)

        # Use provided condition if available, otherwise fallback to default
        ptv3_condition = "DDPM" 

        # 1. Prepare Features
        feat = torch.zeros(coord.shape[0], self.n_in_channels, device=x.device)
        feat[:, :3] = coord

        # 2. Prepare Batch indices and Offsets for Pointcept
        # Batch: [0, 0, ... 1, 1, ... B-1, B-1] (B*N elements)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        # Offset: Cumulative point counts [N, 2N, 3N, ... B*N]
        offset = torch.arange(1, B + 1, device=x.device, dtype=torch.long) * N
        # print("batch")
        # print(batch.detach().cpu().numpy())
        # print("offset")
        # print(offset.detach().cpu().numpy())

        # 3. Process Time/Conditioning
        t_emb = self.get_time_embedding(t)  # (B, context_channels)
        point_context = t_emb[batch]

        # 4. Construct data_dict
        data_dict = {
            "coord": coord,
            "feat": feat,
            "batch": batch,
            "offset": offset, # RESTORED
            "grid_size": self.grid_size, # Increased slightly for more stable voxelization with normalized coords
            "context": point_context,
            "condition": ptv3_condition,
        }

        # 5. Backbone Forward
        point_out = self.ptv3(data_dict)
        # print("[PTv3Dnsr] point_out keys after PTv3 forward: ", point_out.keys()) #
        #                ['coord', 'feat', 'batch', 'offset', 'grid_size', 'context', 'condition', 'order', 'grid_coord', 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 'sparse_shape', 'sparse_conv_feat', 'pad', 'unpad', 'cu_seqlens_key']
        
        #explain each key:
        # 'coord': (B*N, 3) tensor of point coordinates after processing
        # 'batch': (B*N,) tensor of batch indices for each point
        # 'feat': (B*N, C_out) tensor of point features after PTv3 processing
        # 'grid_size': scalar value of the grid size used for voxelization
        # 'context': (B*N, context_channels) tensor of time/condition embeddings aligned with points
        # 'condition': the conditioning string used for PTv3, in this case "DDPM"
        # 'offset': (B,) tensor of offsets for each batch, used for separating samples in sparse convs
        # 'order': (B*N,) tensor of point indices sorted by Morton order for efficient sparse convs
        # 'grid_coord': (B*N, 3) tensor of voxel grid coordinates for each point
        # 'serialized_depth': scalar value of the depth of serialization (number of times points were merged)
        # 'serialized_code': (B*N,) tensor of codes indicating which points were merged during serialization
        # 'serialized_order': (B*N,) tensor of point indices sorted by serialization order
        # 'serialized_inverse': list of (B*N,) tensors of inverse indices for each serialization order, used to unshuffle points back to original order
        # 'sparse_shape': (D, D, D) tuple of the shape of the sparse voxel grid
        # 'sparse_conv_feat': (B*N, C_out) tensor of features after sparse convolution, aligned with 'order'
        # 'pad': scalar value of how many points were padded to make the count divisible by the grid size
        # 'unpad': scalar value of how many points to unpad after processing
        # 'cu_seqlens_key': key for cumulative sequence lengths used in sparse convs, not directly relevant for our use case but part of PTv3's internal handling
        
        # 6. Unshuffle points if requested
        # print("[PTv3Dnsr] coor: ", point_out["coord"].detach().cpu().numpy().flatten())  # Print coordinates after PTv3 processing
        feat = point_out["feat"]
        # print("[PTv3Dnsr] feat shape after PTv3: ", feat.shape) # should be (B*N, C_out) #[2, 64]
        if self.serialized_inverse:
            # point_out["serialized_inverse"] is a list of inverse indices for each serialization order.
            # We use the first one [0] as it corresponds to the order used in the final layer.
            feat = feat[point_out["serialized_inverse"][0]]
            # print("[PTv3Dnsr] feat shape after optional unshuffle: ", feat.shape) # should still be (B*N, C_out)
            # print("[PTv3Dnsr] feat after optional unshuffle: ", feat.detach().cpu())
        # 7. Final Head
        # Using dict-style access for Pointcept DataDict compatibility
        out = self.head(feat)  # (B*N, out_channels*2)   
        # print("[PTv3Dnsr] out shape before reshaping: ", out.shape)  #[2, 6]
        # print("[PTv3Dnsr] out : ", out.detach().cpu()[:,:3].numpy().flatten())  # Print predicted noise for the first few points

        # Reshape to (B, N, 6) and transpose back to (B, 6, N) to align with vision style
        if out.shape[0] != B * N:
             # If points are merged, PointTransformerV3 returns fewer than B*N points.
             # We need grid_size to be small enough that points stay unique, 
             # but large enough for the 5x5x5 conv kernel to see neighbors.
             raise RuntimeError(f"Point count mismatch: {out.shape[0]} vs {B*N}. grid_size={self.grid_size} might be too large and merging points.")
        out = out.view(B, N, -1).transpose(1, 2).contiguous()  # (B, out_channels*2, N)
        # print("[PTv3Dnsr] out shape after reshaping: ", out.shape) #[1, 6, 2]
        return out




def visualize_xyz_pip(
    fname,
    original_xyz,
    reconstructed_xyz,
    title="",
    save_dir="/home/palakons/D-PCC/output/plots",
    plotlims= {"x": (0, 1980), "y": (0, 943), "z": (0, 250)},
    marker_config={
        "original": {"color": "blue", "marker": "o", "size": 10, "alpha": 0.6},
        "reconstructed": {"color": "red", "marker": "x", "size": 10, "alpha": 0.6},
    },
    fig_size=( 16,9), #width, height in inches
    device="cpu",
):
    """
    Visualize original vs reconstructed XYZ point clouds.

    Args:
        fname: str - filename for the saved plot
        original_xyz: (N, 3) tensor - [x, y, z]
        reconstructed_xyz: (M, 3) tensor - [x, y, z]
        save_dir: str - directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    def _draw_oob_arrows(ax, points, xlim, ylim, color):
        """Draw arrows at the edge of the plot for points that are out of bounds."""
        if points is None or len(points) == 0:
            return
        x, y = points[:, 0], points[:, 1]
        oob_mask = (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        if not np.any(oob_mask):
            return
        oob_points = points[oob_mask]
        if len(oob_points) > 20:
            indices = np.random.choice(len(oob_points), 20, replace=False)
            oob_points = oob_points[indices]
        rx, ry = xlim[1] - xlim[0], ylim[1] - ylim[0]
        for p in oob_points:
            px, py = p[0], p[1]
            bx, by = np.clip(px, xlim[0], xlim[1]), np.clip(py, ylim[0], ylim[1])
            dx, dy = px - bx, py - by
            dist = np.sqrt((dx/rx)**2 + (dy/ry)**2)
            if dist > 0:
                ax.arrow(bx, by, (dx/rx/dist)*rx*0.04, (dy/ry/dist)*ry*0.04, 
                         color=color, alpha=0.6, head_width=min(rx, ry)*0.015,
                         clip_on=False, length_includes_head=True)

    orig_np = (
        original_xyz.cpu().numpy()
        if torch.is_tensor(original_xyz)
        else original_xyz       
    )
    recon_np = (
        reconstructed_xyz.cpu().numpy()
        if torch.is_tensor(reconstructed_xyz)
        else reconstructed_xyz
    )

    fig = plt.figure(figsize=fig_size)

    # Original point cloud views
    ax1 = fig.add_subplot(2, 2, 1)

    # plot actual and predict in the same plot with different colors
    ax1.scatter(
        orig_np[:, 0],
        orig_np[:, 1],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax1.scatter(
        recon_np[:, 0],
        recon_np[:, 1],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )

    # Add PiP (inset) showing only GT
    def _add_gt_inset(ax, points_np, idx1, idx2, title_suffix):
        try:
            if len(points_np) == 0: return
            inset = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
            inset.scatter(
                points_np[:, idx1],
                points_np[:, idx2],
                color=marker_config["original"]["color"],
                s=marker_config["original"]["size"] / 2,
                alpha=marker_config["original"]["alpha"],
                marker=marker_config["original"]["marker"],
                rasterized=True,
            )
            # Zoom inset to GT only
            min_gt = points_np[:, [idx1, idx2]].min(axis=0)
            max_gt = points_np[:, [idx1, idx2]].max(axis=0)
            diff = max_gt - min_gt
            pad_gt = np.maximum(diff * 0.15, 0.1)
            inset.set_xlim(min_gt[0] - pad_gt[0], max_gt[0] + pad_gt[0])
            inset.set_ylim(min_gt[1] - pad_gt[1], max_gt[1] + pad_gt[1])
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(f"GT {title_suffix}", fontsize=8)
        except Exception:
            pass

    _add_gt_inset(ax1, orig_np, 0, 1, "XY")

    # CD Calculation
    cd_gt = torch.tensor(orig_np[None, :, :3], device=device, dtype=torch.float32)
    cd_recon = torch.tensor(recon_np[None, :, :3], device=device, dtype=torch.float32)
    cd, _ = pt3d_chamfer_distance(cd_gt, cd_recon)
    cd = cd.item()

    ax1.set_title(f"XY View CD:{cd:.2e}")
    ax1.set_aspect("equal", adjustable="box")
    if plotlims is not None:
        ax1.set_xlim(plotlims["x"])
        ax1.set_ylim(plotlims["y"])
        _draw_oob_arrows(ax1, orig_np[:, [0, 1]], plotlims["x"], plotlims["y"], marker_config["original"]["color"])
        _draw_oob_arrows(ax1, recon_np[:, [0, 1]], plotlims["x"], plotlims["y"], marker_config["reconstructed"]["color"])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(orig_np[:, 0], orig_np[:, 2], color=marker_config["original"]["color"], s=marker_config["original"]["size"], alpha=marker_config["original"]["alpha"], marker=marker_config["original"]["marker"], rasterized=True)
    ax2.scatter(recon_np[:, 0], recon_np[:, 2], color=marker_config["reconstructed"]["color"], s=marker_config["reconstructed"]["size"], alpha=marker_config["reconstructed"]["alpha"], marker=marker_config["reconstructed"]["marker"], rasterized=True)
    _add_gt_inset(ax2, orig_np, 0, 2, "XZ")
    ax2.set_title("XZ View")
    if plotlims is not None:
        ax2.set_xlim(plotlims["x"])
        ax2.set_ylim(plotlims["z"])
        _draw_oob_arrows(ax2, orig_np[:, [0, 2]], plotlims["x"], plotlims["z"], marker_config["original"]["color"])
        _draw_oob_arrows(ax2, recon_np[:, [0, 2]], plotlims["x"], plotlims["z"], marker_config["reconstructed"]["color"])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(orig_np[:, 2], orig_np[:, 1], color=marker_config["original"]["color"], s=marker_config["original"]["size"], alpha=marker_config["original"]["alpha"], marker=marker_config["original"]["marker"], rasterized=True)
    ax3.scatter(recon_np[:, 2], recon_np[:, 1], color=marker_config["reconstructed"]["color"], s=marker_config["reconstructed"]["size"], alpha=marker_config["reconstructed"]["alpha"], marker=marker_config["reconstructed"]["marker"], rasterized=True)
    _add_gt_inset(ax3, orig_np, 2, 1, "YZ")
    ax3.set_title("YZ View")
    if plotlims is not None:
        ax3.set_xlim(plotlims["z"])
        ax3.set_ylim(plotlims["y"])
        _draw_oob_arrows(ax3, orig_np[:, [2, 1]], plotlims["z"], plotlims["y"], marker_config["original"]["color"])
        _draw_oob_arrows(ax3, recon_np[:, [2, 1]], plotlims["z"], plotlims["y"], marker_config["reconstructed"]["color"])
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Y")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(orig_np[:, 0], orig_np[:, 1], orig_np[:, 2], color=marker_config["original"]["color"], label="Original", s=marker_config["original"]["size"], alpha=marker_config["original"]["alpha"], marker=marker_config["original"]["marker"], rasterized=True)
    ax4.scatter(recon_np[:, 0], recon_np[:, 1], recon_np[:, 2], color=marker_config["reconstructed"]["color"], label="Reconstructed", s=marker_config["reconstructed"]["size"], alpha=marker_config["reconstructed"]["alpha"], marker=marker_config["reconstructed"]["marker"], rasterized=True)
    ax4.legend(loc="upper right")
    ax4.set_title("3D View")
    if plotlims is not None:
        ax4.set_xlim(plotlims["x"])
        ax4.set_ylim(plotlims["y"])
        ax4.set_zlim(plotlims["z"])
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")

    if title:
        # make sure wrapt he line to 80 characters
        wrapped_title = "\n".join(textwrap.wrap(title, width=120))
        plt.suptitle(wrapped_title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(
        save_dir,
        f"{fname}.jpg",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cd , save_path
def plot_xy_timestep(
                            fname,
                            gt,
                            coords,
                            title,
                            save_dir,
                            figsize=(8, 6)
                        ):
    """
    Plot separated X-Y scatter, one plot per timestep, horizontally subplot, with different colors for each sample in the batch.
    Args:
    fname: str - filename for the saved plot
    gt: (B,N, 3) tensor - ground truth point cloud for reference, will be plotted in black in all subplots
    coords: list of coordinates for each timestep (timestep x B x N x 3)
    title: str - title for the plot
    save_dir: str - directory to save the plot
    """
    assert len(coords)==50, "Expected 50 timesteps in coords"
    fig, axs = plt.subplots(10,5, figsize=(figsize[0]*5, figsize[1]*10))
    axs_flat = axs.flatten()
    n_current = len(coords[0])  # number of samples in the current batch
    
    # Use a safe device for CD calculation
    calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert GT to tensor once
    gt_tensor = torch.tensor(gt, device=calc_device)

    for t in range(len(coords)):
        ax = axs_flat[t]
        #set title
        ax.set_title(f'T {len(coords)-1-t}')
        for i in range(n_current):
            sample_coords = coords[t][i]  # (N, 3)
            sample_tensor = torch.tensor(sample_coords[None, :, :3], device=calc_device)
            
            # Calculate CD for label
            cd_val, _ = pt3d_chamfer_distance(gt_tensor[i:i+1, :, :3], sample_tensor)
            
            ax.scatter(sample_coords[:, 0], sample_coords[:, 1], label=f'S{i}: {cd_val.item():.1f}', s=10, alpha=0.6)
            # plot gt in black
            ax.scatter(gt[i, :, 0], gt[i, :, 1], color='black', label='GT', s=10, alpha=0.4)
            ax.set_aspect('equal', adjustable='box')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.legend(fontsize='x-small')
    plt.suptitle(title, fontsize=16)
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path

def plot_cd_timestep_curve(
                            fname,
                            cds,
                            title,
                            save_dir,
                            figsize=(8, 6)
                        ):
    """
    Plot separated Chamfer Distance curves vs Timestep curve.
    Args:
    fname: str - filename for the saved plot
    cds: list of chamfer distances for each timestep (timestep x B)
    title: str - title for the plot
    save_dir: str - directory to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_current = len(cds[0])  # number of samples in the current batch
    for i in range(n_current):
        sample_cds = [cd[i] for cd in cds]
        # PTv3 library might provide them in forward/backward order, 
        # we plot sample_cds[::-1] to show high-noise to clean.
        plot_vals = sample_cds[::-1]
        line, = ax.plot(plot_vals, label=f'Sample {i}')
        # Annotate first and last points
        ax.annotate(f"{plot_vals[0]:.1f}", (0, plot_vals[0]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line.get_color())
        ax.annotate(f"{plot_vals[-1]:.1f}", (len(plot_vals)-1, plot_vals[-1]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line.get_color())
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Chamfer Distance')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path

def train_ptv3_ddpm(config, checkpoint_dir, tb_dir, plot_dir):
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    checkpoint = None
    writer = SummaryWriter(tb_dir)

    if config["dit_checkpoint"] != "":
        print(f"Loading checkpoint from {config['dit_checkpoint']}...")
        checkpoint = torch.load(config["dit_checkpoint"], map_location=config["device"], weights_only=False)

        #update config (with exceptions) to become the one in the checkpoint, so that we can resume training with the same config as before
        print("overriding config with checkpoint config...")
        exceptions = ["dit_checkpoint", "ae_checkpoint","dit_epochs", "ae_epochs","exp_name"]
        checkpoint_config = checkpoint.get("config", {})
        for key in checkpoint_config:
            if key not in exceptions and key in config:
                if config[key] != checkpoint_config[key]: # allow overriding epoch-related configs for resuming with different training length
                    print(f"Overriding config key {key}: {config[key]} -> {checkpoint_config[key]}")
                config[key] = checkpoint_config[key]
            else:
                print(f"Keeping current config key {key}: {config.get(key, 'N/A')} (not in checkpoint or in exceptions)")


    # model = PTv3().to(config["device"]) #TODO: make this a PTv3 model with the appropriate conditioning for the RGB image's WAN VAE Latent
    print("prams for PTv3Dnsr: ", config["ptv3_grid_size"],'ptv3_shuffle_orders', config["ptv3_shuffle_orders"], 'ptv3_serialized_inverse',config["ptv3_serialized_inverse"])
    print("num_inference_steps: ", config["num_inference_steps"])
    print("batch_size: ", config["batch_size"],"num_points: ", config["num_points"],"ditepochs: ", config["dit_epochs"])
    # Swap to SimplePointUNet for deterministic overfitting test
    if config["denoiser_model"] == "simple_unet":
        print("Using SimplePointUNet as denoiser!")
        model = SimplePointUNet(in_channels=3, base_channels=64, out_channels=3, num_layers=4).to(config["device"])
    else:
        model = PTv3Dnsr(
            n_in_channels=3,
            context_channels=256,
            out_channels=3,
            grid_size=config["ptv3_grid_size"],
            shuffle_orders=config["ptv3_shuffle_orders"],
            serialized_inverse=config["ptv3_serialized_inverse"],
            n_stages=config["ptv3_n_stages"],
            seed=config["seed_model"],
            backbone_type=config["ptv3_backbone"],
        ).to(config["device"])
    #print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params/1e6:.2f} million parameters.")
    writer.add_text("Model/Architecture", str(model), global_step)
    writer.add_scalar("Model/NumParams", num_params, global_step)
    diffusion = create_diffusion(
        timestep_respacing=str(config["num_inference_steps"]),  # default: 1000 steps, linear noise schedule
    )  # default: 1000 steps, linear noise schedule

    optimizer = makeOptimizer(model, config)
    lr_scheduler = makeLR_Scheduler(optimizer, config)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        global_step = checkpoint.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch} with best_loss {best_loss:.4f} and global_step {global_step}")



    train_dataset, val_dataset = splitDataset(
        makeDataset(
            config,
            get_occ_grid=False,
            get_camera=True,
            get_wan_vae=True,
        ),
        split=0.5,
    )
    data_mean = train_dataset.dataset.data_mean.to(config["device"])
    data_std = train_dataset.dataset.data_std.to(config["device"])
    # avoid divide-by-zero in normalization (protect against constant features)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)

    train_dataloader = makeDataloaders(
        train_dataset,
        config,
        is_train=True,
    )


    # Add model graph to TensorBoard
    print("Adding model graph to TensorBoard...")

    writer.add_text("Model", str(model), global_step)

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    num_epochs = config["dit_epochs"]
    epoch_bar = trange(
        start_epoch,
        num_epochs,
        desc="Epochs",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    )

    OT_loss_func = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    
    # Create fixed noise reference for debug modes 3 and 4
    fixed_noise_ref = torch.randn(1, 3, config["num_points"]).to(config["device"])

    for epoch in epoch_bar:
        train_losses = []
        for batch in tqdm(train_dataloader, desc="Train Batches", leave=False):
            pointcloud = batch["filtered_radar_data"].to(config["device"])[
                :, :, :3
            ]  # (B,N,3)
            B = pointcloud.shape[0]
            normalized_pointcloud = (pointcloud - data_mean[:3]) / data_std[:3]
            # Transpose to vision style (B, 3, N) for diffusion library
            min_val = torch.amin(normalized_pointcloud, dim=[0, 1])
            max_val = torch.amax(normalized_pointcloud, dim=[0, 1])
            writer.add_scalar("Data/norm_min_x", min_val[0].item(), global_step)
            writer.add_scalar("Data/norm_max_x", max_val[0].item(), global_step)
            writer.add_scalar("Data/norm_min_y", min_val[1].item(), global_step)
            writer.add_scalar("Data/norm_max_y", max_val[1].item(), global_step)
            writer.add_scalar("Data/norm_min_z", min_val[2].item(), global_step)
            writer.add_scalar("Data/norm_max_z", max_val[2].item(), global_step)
            normalized_pointcloud = normalized_pointcloud.transpose(1, 2).contiguous()

            # Ensure conditioning latent exists and has correct shape (B, 16, 2, 60, 104)
            wan_vae_latent = batch["wan_vae_latent"].to(config["device"])                        
            wan_vae_latent = wan_vae_latent * 0 
            
            # --- DEBUG MODE TIMESTEP SELECTION ---
            if config["debug_mode"] in [1, 2, 3, 5]:
                # Use 10% of total timesteps to avoid out-of-bounds errors (e.g. if num_inference_steps=50)
                t_val = max(0, diffusion.num_timesteps // 10)
                t = torch.full((B,), t_val, device=config["device"]).long()
            else:
                t = torch.randint(0, diffusion.num_timesteps, (B,), device=config["device"]).long()

            model_kwargs = dict()

            # --- DEBUG MODE NOISE SELECTION ---
            if config["debug_mode"] == 1:
                noise = torch.zeros_like(normalized_pointcloud)
            elif config["debug_mode"] == 2:
                noise = torch.full_like(normalized_pointcloud, 0.5)
            elif config["debug_mode"] in [3, 4]:
                noise = fixed_noise_ref.repeat(B, 1, 1)
            else:
                noise = torch.randn_like(normalized_pointcloud)

            # Center noise if requested to prevent global drift (Bias fix)
            if config["ptv3_zero_mean_noise"]:
                noise = noise - noise.mean(dim=2, keepdim=True)
            
            x_t = diffusion.q_sample(normalized_pointcloud, t, noise=noise)
            
            # 2. Get model prediction
            model_output = model(x_t, t, **model_kwargs)
            pred_epsilon = model_output[:, :3, :] # First 3 channels
            
            # 3. Standard MSE Loss on noise
            loss_mse = F.mse_loss(pred_epsilon, noise)
            
            # 4. GEOMETRIC RECONSTRUCTION LOSSES
            train_loss = loss_mse

            if config["lambda_cd"] > 0 or config["lambda_ot"] > 0 or (epoch + 1) % config["plot_every"] == 0:
                # Reconstruct x0 from noisy x_t and predicted noise
                # Shapes for Geometric Losses: (B, N, 3)
                pred_xstart = diffusion._predict_xstart_from_eps(x_t, t, pred_epsilon)
                P = pred_xstart.transpose(1, 2)
                G = normalized_pointcloud.transpose(1, 2)

                if (epoch + 1) % config["plot_every"] == 0:
                    cd, _ = visualize_xyz_pip(
                        fname=f"train_epoch{epoch+1}_step{global_step}_pip",
                        original_xyz=G[0].detach().cpu()*data_std[:3].cpu() + data_mean[:3].cpu(),
                        reconstructed_xyz=P[0].detach().cpu()*data_std[:3].cpu() + data_mean[:3].cpu(),
                        title=f"Epoch {epoch+1} Step {global_step} PIP Visualization (Mode {config['debug_mode']})",
                        save_dir=plot_dir,
                        plotlims=None,
                        device=config["device"],
                    )
                    writer.add_scalar("CD/train_cd_0", cd, global_step)
                
                if config["lambda_cd"] > 0:
                    loss_cd, _ = pt3d_chamfer_distance(P, G)
                    writer.add_scalar("Loss/train/cd", loss_cd.item(), global_step)
                    train_loss += config["lambda_cd"] * loss_cd
                
                if config["lambda_ot"] > 0:
                    loss_ot = OT_loss_func(P, G).mean()            
                    writer.add_scalar("Loss/train/ot", loss_ot.item(), global_step)
                    train_loss += config["lambda_ot"] * loss_ot

            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            
            # --- GRADIENT CLIPPING ---
            # Prevents sudden loss jumps by capping the maximum weight update
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar("Gradients/Norm", grad_norm.item(), global_step)
            
            # --- WEIGHT & PREDICTION TRACKING ---
            with torch.no_grad():
                param_norm = sum(p.norm().item() for p in model.parameters())
                writer.add_scalar("Weights/Total_Norm", param_norm, global_step)
                writer.add_scalar("Prediction/Mean_Abs", pred_epsilon.abs().mean().item(), global_step)
                writer.add_scalar("Prediction/Max_Abs", pred_epsilon.abs().max().item(), global_step)

            optimizer.step()
            
            # Log specific gradients every 100 steps to monitor health
            if global_step % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Only log head and first block to avoid huge log files
                        if "head" in name or "ptv3.enc.0" in name:
                            writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

            # Log loss and model outputs to tensorboard per batch
            global_step += 1
            writer.add_scalar("Loss/train", train_loss.item(), global_step)
            writer.add_scalar("Loss/train/mse", loss_mse.item(), global_step)
        # Log epoch-level metrics
        avg_epoch_loss = np.mean(train_losses)
        is_best = avg_epoch_loss < best_loss/2 # Use a more forgiving threshold for "best" since this is a noisy metric, we just want to track significant improvements
        if is_best:
            best_loss = avg_epoch_loss
        writer.add_scalar("Loss/train_epoch_avg", avg_epoch_loss, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        lr_scheduler.step()


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
        if (epoch + 1) % config["plot_every"] == 0 or is_best:
            if is_best:
                print(f"New best model at epoch {epoch+1} with avg loss {avg_epoch_loss:.4f}")
            
            # Log gradients to check for vanishing/exploding gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch + 1)

            # Generate samples and log Denoising Curve
            try:
                model.eval()
                with torch.no_grad():
                    # Progressive sampling to log the CD-timestep curve
                    noise = torch.randn(B, 3, config["num_points"]).to(config["device"])
                    samples_gen = diffusion.p_sample_loop_progressive(
                        model, 
                        (B, 3, config["num_points"]), 
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        device=config["device"],
                        progress=True,noise=noise
                        
                    )
                    
                    # Track steps and collect all CDs for the ridge plot
                    all_step_cds = []
                    all_pred_xyz = []  # To track XY evolution for the ridge plot
                    all_normalized_pred_x0 = [] # To track normalized XY evolution for the ridge plot
                    for step_idx, sample in enumerate(samples_gen):
                        # Log the predicted clean x0 at each step (sampled at intervals)
                        pred_x0 = sample["pred_xstart"]
                        normalized_pred_x0=pred_x0.transpose(1, 2)
                        # Denormalize and compare to GT
                        pred_x0_xyz = normalized_pred_x0 * data_std[:3] + data_mean[:3]

                        # Calculate CD for all samples in batch, no reduction
                        # Shapes: pred_x0_xyz (B, N, 3), pointcloud (B, N, 3)
                        step_cd, _ = pt3d_chamfer_distance(pred_x0_xyz, pointcloud, batch_reduction=None)
                        # Convert to CPU numpy array immediately for plotting compatibility
                        step_cd_np = step_cd.detach().cpu().numpy()
                        all_step_cds.append(step_cd_np)
                        all_pred_xyz.append(pred_x0_xyz.detach().cpu().numpy())
                        all_normalized_pred_x0.append(normalized_pred_x0.detach().cpu().numpy())

                        generated = sample["sample"] # This is the final result at the end of the loop

                    # Plot the CD Evolution curve
                    plot_cd_timestep_curve(
                        fname=f"epoch_{epoch+1}_cd-evolution",
                        cds=all_step_cds,
                        title=f"Epoch {epoch+1} CD-Timestep Evolution" + (" (Best)" if is_best else ""),
                        save_dir=plot_dir,figsize=(6,4  )
                    )
                    plot_xy_timestep(
                        fname=f"epoch_{epoch+1}_xy-timestep",
                        gt=pointcloud.detach().cpu().numpy(), # Plot GT of first sample for reference
                        coords=all_pred_xyz,
                        title=f"Epoch {epoch+1} XY-Timestep Evolution" + (" (Best)" if is_best else ""),
                        save_dir=plot_dir,figsize=(6,4  )
                    )
                    #normalized version
                    try:
                        plot_xy_timestep(
                            fname=f"epoch_{epoch+1}_xy-timestep-normalized",
                            gt=normalized_pointcloud.detach().cpu().numpy(), # Plot GT of first sample for reference
                            coords=all_normalized_pred_x0,
                            title=f"Epoch {epoch+1} Normalized XY-Timestep Evolution" + (" (Best)" if is_best else ""),
                            save_dir=plot_dir,figsize=(6,4  )
                        )
                    except Exception as e:
                        print(f"Error plotting normalized xy-timestep: {e}")

                    # Log distribution of predicted noise separately for X, Y, Z
                    pred_output = model(normalized_pointcloud, t, **model_kwargs)
                    writer.add_histogram("Distribution/Pred_Noise_X", pred_output[:, 0, :], epoch + 1)
                    writer.add_histogram("Distribution/Pred_Noise_Y", pred_output[:, 1, :], epoch + 1)
                    writer.add_histogram("Distribution/Pred_Noise_Z", pred_output[:, 2, :], epoch + 1)

                    # Transpose back to (B, N, 3) and denormalize
                    generated = generated.transpose(1, 2).contiguous()
                    generated = generated * data_std[:3] + data_mean[:3]
                    # Save plots for the first few samples in the batch
                    cds = []
                    for i in range(min(4, B)):
                        minax = torch.amin(pointcloud[i], dim=0).cpu().numpy()  # (3,)
                        maxax = torch.amax(pointcloud[i], dim=0).cpu().numpy()  # (3,)

                        minax_pred = torch.amin(generated[i], dim=0).cpu().numpy()  # (3,)
                        maxax_pred = torch.amax(generated[i], dim=0).cpu().numpy()  # (3,)

                        minax = np.minimum(minax, minax_pred)
                        maxax = np.maximum(maxax, maxax_pred)
                        rangeax = maxax - minax
                        # add 10% padding to the range
                        padding = rangeax * 0.1
                        minax = minax - padding
                        maxax = maxax + padding

                        cd, path = visualize_xyz_pip(
                            fname=f"epoch_{epoch+1}_sample_{i+1}",
                            original_xyz=pointcloud[i],
                            reconstructed_xyz=generated[i],
                            title=f"Epoch {epoch+1} Sample {i+1}" + (" Best" if is_best else ""),
                            save_dir=plot_dir,
                            device=config["device"],
                        plotlims= {"x": (minax[0], maxax[0]), "y":( minax[1],maxax[1]),"z":(minax[2],maxax[2])},
                        )
                        cds.append(cd)
                    avg_cd = np.mean(cds)
                    try:
                        writer.add_scalar("Sample/avg_cd", avg_cd, epoch)
                    except Exception as e:
                        print(f"Error logging avg_cd to TensorBoard: {e}")
                    writer.add_histogram("Distribution/Final_Sample_X", generated[:, :, 0], epoch + 1)
                    writer.add_histogram("Distribution/Final_Sample_Y", generated[:, :, 1], epoch + 1)
                    writer.add_histogram("Distribution/Final_Sample_Z", generated[:, :, 2], epoch + 1)
            except Exception as e:
                print(f"Error during sampling/visualization at epoch {epoch+1}: {e}")
            model.train() # Ensure model stays in train mode after sampling step

        epoch_bar.set_description(
            f"Loss: {np.mean(train_losses):.4f}"
        )

        # CHECKPOINTING
        if (epoch + 1) % config["save_every"] == 0 or is_best:
            save_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "config": config,
                "best_loss": best_loss
            }

            
            # Save 'latest' every save_every or every best improvement
            latest_path = os.path.join(checkpoint_dir, "ptv3_latest.pt")
            torch.save(save_dict, latest_path)
            
            # If it's the best loss so far, also save to 'best' file
            if is_best:
                best_path = os.path.join(checkpoint_dir, "ptv3_best.pt")
                torch.save(save_dict, best_path)

    tblogHparam(config, writer, {"final_avg_cd": avg_cd})
    writer.close()
    # torch save model, optimizer, lr_scheduler, and training config
    checkpoint_path = os.path.join(checkpoint_dir, f"final_ptv3.pt")
    torch.save(
        {
                "epoch": config["dit_epochs"],
                "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": config,
                "best_loss": best_loss
        },
        checkpoint_path,
    )
    print(f"Training completed. Final model saved at {checkpoint_path}.")   


def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    tb_key = f"ptv3_ddpm"
    working_dir = f"{config['exp_name']}"
    plot_dir = f"/data/palakons/{tb_key}/plots/{working_dir}"
    checkpoint_dir = f"/data/palakons/{tb_key}/checkpoints/{working_dir}"
    tb_log_dir = f"/home/palakons/logs/tb_log/{tb_key}/{working_dir}"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print("Starting training...")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    torch.manual_seed(config["seed"])
    # torch.cuda.set_device(config["device"])

    train_ptv3_ddpm(
        config,
        checkpoint_dir=checkpoint_dir,
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
