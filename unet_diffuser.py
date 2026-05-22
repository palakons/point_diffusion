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
import time

import geomloss
from copy import deepcopy

sys.path.insert(0, "/home/palakons/DiT")
from models import DiT_models

## Removed: from diffusion import create_diffusion
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
    tblogHparam,
)
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance


class SimplePointUNet(nn.Module):
    """
    A minimal UNet-style denoiser for (B, 3, N) point clouds.
    This is a 1D UNet (conv over N), with skip connections and no stochasticity.
    It should be able to overfit small data and fit points well.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        out_channels=3,
        num_layers=4,
        t_embed_dim=32,
    ):
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
            self.encoders.append(
                nn.Conv1d(ch, base_channels * 2**i, kernel_size=3, padding=1)
            )
            self.pools.append(nn.MaxPool1d(2))
            ch = base_channels * 2**i
        # Bottleneck
        self.bottleneck = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        # Decoder
        for i in reversed(range(num_layers)):
            self.ups.append(
                nn.ConvTranspose1d(ch, base_channels * 2**i, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Conv1d(
                    base_channels * 2**i * 2,
                    base_channels * 2**i,
                    kernel_size=3,
                    padding=1,
                )
            )
            ch = base_channels * 2**i
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
                out = out[..., : enc_feat.shape[-1]]
            elif out.shape[-1] < enc_feat.shape[-1]:
                pad = enc_feat.shape[-1] - out.shape[-1]
                out = F.pad(out, (0, pad))
            out = torch.cat([out, enc_feat], dim=1)
            out = F.relu(dec(out))
        out = self.final(out)
        return out  # (B, 3, N)


class PointNetLikeDenoiser(nn.Module):
    """
    Minimal per-point denoiser for (B, 3, N) point clouds, updated
    with a PointNet-like global max pooling path for structure awareness.
    """

    def __init__(self, hidden_channels=64, time_embed_dim=32, num_blocks=3):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.stem = nn.Conv1d(3 + time_embed_dim, hidden_channels, kernel_size=1)
        
        # PointNet-like global feature extractor
        self.global_mlp = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8 if hidden_channels >= 8 else 1, hidden_channels * 2 if i == 0 else hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels * 2 if i == 0 else hidden_channels, hidden_channels, kernel_size=1),
                    nn.GroupNorm(8 if hidden_channels >= 8 else 1, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                )
                for i in range(num_blocks)
            ]
        )
        self.skip_projs = nn.ModuleList(
            [nn.Conv1d(hidden_channels * 2 if i == 0 else hidden_channels, hidden_channels, kernel_size=1) for i in range(num_blocks)]
        )
        self.out = nn.Conv1d(hidden_channels, 3, kernel_size=1)

        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.constant_(self.stem.bias, 0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x, t=None, condition=None, **kwargs):
        # x: (B, 3, N), t: (B,)
        batch_size, _, num_points = x.shape
        if t is None:
            t = torch.zeros(batch_size, device=x.device)

        t = t.view(batch_size, 1).float() / 1000.0
        t_emb = self.t_mlp(t)
        t_rep = t_emb.unsqueeze(-1).expand(-1, self.time_embed_dim, num_points)

        h = torch.cat([x, t_rep], dim=1)
        h = self.stem(h)
        
        # ---- PointNet Pooling ----
        # Extract global shape descriptor via max pooling
        g = self.global_mlp(h)                            # (B, hidden, N)
        g = torch.max(g, dim=2, keepdim=True)[0]          # (B, hidden, 1)
        g_rep = g.expand(-1, -1, num_points)              # (B, hidden, N)
        
        # Inject global descriptor into local points
        h = torch.cat([h, g_rep], dim=1)                  # (B, hidden * 2, N)
        # --------------------------

        for block, skip_proj in zip(self.blocks, self.skip_projs):
            residual = skip_proj(h)
            h = block(h) + residual

        return self.out(F.silu(h))


class MLPDenoiser(nn.Module):
    """Per-point MLP denoiser: simple, fast baseline for DDPM/identity tests.

    API: forward(x, t, condition=None) -> (B, out_channels, N)
    Accepts x shape (B, 3, N), t shape (B,) (long or float). Time is sinusoidally encoded
    and passed through an MLP then concatenated to per-point features.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        context_channels=256,
        hidden_dim=256,
        num_layers=4,
        coord_projector_dim=0,
        dropout=0.0,
        scene_embed_dim=0,  # 0 = no scene conditioning, >0 = scene latent dim
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_channels = context_channels
        self.scene_embed_dim = scene_embed_dim

        # optional coord projector
        if coord_projector_dim and coord_projector_dim > 0:
            self.coord_projector = nn.Sequential(
                nn.Linear(in_channels, coord_projector_dim),
                nn.GELU(),
                nn.Linear(coord_projector_dim, coord_projector_dim),
            )
            feat_dim = coord_projector_dim
        else:
            self.coord_projector = None
            feat_dim = in_channels

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.GELU(),
            nn.Linear(context_channels, context_channels),
        )
        self.cond_emb = None

        # optional scene embedding processor
        if scene_embed_dim > 0:
            print(f"Initializing MLPDenoiser with scene conditioning. Scene embed dim: {scene_embed_dim}, context_channels: {context_channels}")
            # input torch.Size([B, 16, 2, 60, 104]) --> after flattening spatial dims --> (B, scene_embed_dim)
            self.scene_mlp = nn.Sequential(
                nn.Linear(scene_embed_dim, context_channels),
                nn.GELU(),
                nn.Linear(context_channels, context_channels),
            )
        else:
            self.scene_mlp = None

        # per-point MLP: input = feat_dim + context_channels + (scene_context if enabled)
        cond_channels = context_channels + (context_channels if scene_embed_dim > 0 else 0)
        layers = []
        cur = feat_dim + cond_channels
        for i in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_channels))
        self.point_mlp = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_time_embedding(self, timesteps: torch.Tensor):
        # Sinusoidal embedding -> time_mlp
        half = self.context_channels // 2
        exponents = -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
        emb = timesteps[:, None].float() * exponents.exp()[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] != self.context_channels:
            if emb.shape[-1] < self.context_channels:
                pad = self.context_channels - emb.shape[-1]
                emb = torch.cat([emb, emb.new_zeros(emb.shape[0], pad)], dim=-1)
            else:
                emb = emb[:, : self.context_channels]
        return self.time_mlp(emb)

    def forward(self, x, t, condition=None, **kwargs):
        # x: (B, 3, N), t: (B,), condition: (B, scene_embed_dim) optional scene latent
        B, C, N = x.shape
        device = x.device

        # per-point features
        x_pts = x.permute(0, 2, 1).contiguous().view(B * N, C)
        if self.coord_projector is not None:
            feat = self.coord_projector(x_pts)
        else:
            feat = x_pts

        # time embedding expanded per point
        t_emb = self.get_time_embedding(t.to(device))  # (B, context)
        t_expanded = t_emb.repeat_interleave(N, dim=0)  # (B*N, context)

        # scene conditioning (optional)
        if self.scene_mlp is not None and condition is not None:
            # condition: (B, scene_embed_dim), need to flatten dim 1 ...
            cond = condition.to(device).view(B, -1)  # flatten all non-batch dims
            s_emb = self.scene_mlp(cond)  # (B, context)
            s_expanded = s_emb.repeat_interleave(N, dim=0)  # (B*N, context)
            cond = torch.cat([t_expanded, s_expanded], dim=-1)
            # print(f"[MsLPDenoiser] Using scene conditioning. Scene embed shape: {condition.shape}, processed scene embed shape: {s_emb.shape}, expanded scene embed shape: {s_expanded.shape}, cond shape: {cond.shape}")
        else:
            cond = t_expanded

        inp = torch.cat([feat, cond], dim=-1)
        out = self.point_mlp(inp)  # (B*N, out_channels)
        out = out.view(B, N, self.out_channels).permute(0, 2, 1).contiguous()
        return out


def visualize_xyz_pip(
    fname,
    original_xyz,
    reconstructed_xyz,
    title="",
    save_dir="/home/palakons/D-PCC/output/plots",
    plotlims=None,
    grid=None,
    marker_config={
        "original": {"color": "blue", "marker": "o", "size": 10, "alpha": 0.6},
        "reconstructed": {"color": "red", "marker": "x", "size": 10, "alpha": 0.6},
    },
    fig_size=(16, 9),  # width, height in inches
    device="cpu",
):
    """
    Visualize original vs reconstructed XYZ point clouds.

    Args:
        fname: str - filename for the saved plot
        original_xyz: (N, 3) tensor - [x, y, z]
        reconstructed_xyz: (M, 3) tensor - [x, y, z]
        save_dir: str - directory to save the plot
        plotlims: dict or None - plot limits for x, y, z. If None, computed from GT + 10% buffer
        grid: float or None - if set, draw grid lines on 2D plots at this spacing (same units as xyz)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    orig_np = (
        original_xyz.cpu().numpy() if torch.is_tensor(original_xyz) else original_xyz
    )
    recon_np = (
        reconstructed_xyz.cpu().numpy()
        if torch.is_tensor(reconstructed_xyz)
        else reconstructed_xyz
    )
    
    # Compute plotlims from GT with 10% buffer if not provided
    if plotlims is None:
        min_coords = orig_np.min(axis=0)
        max_coords = orig_np.max(axis=0)
        ranges = max_coords - min_coords
        buffer = ranges * 0.1
        plotlims = {
            "x": (min_coords[0] - buffer[0], max_coords[0] + buffer[0]),
            "y": (min_coords[1] - buffer[1], max_coords[1] + buffer[1]),
            "z": (min_coords[2] - buffer[2], max_coords[2] + buffer[2]),
        }



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
            dist = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
            if dist > 0:
                try:
                    ax.arrow(
                        bx,
                        by,
                        (dx / rx / dist) * rx * 0.04,
                        (dy / ry / dist) * ry * 0.04,
                        color=color,
                        alpha=0.6,
                        head_width=min(rx, ry) * 0.015,
                        clip_on=False,
                        length_includes_head=True,
                    )
                except Exception as e:
                    print(f"{__name__}: Error drawing arrow for point {p}: {e}: dx={dx}, dy={dy}, rx={rx}, ry={ry}, dist={dist}")



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
            if len(points_np) == 0:
                return
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
        _draw_oob_arrows(
            ax1,
            orig_np[:, [0, 1]],
            plotlims["x"],
            plotlims["y"],
            marker_config["original"]["color"],
        )
        _draw_oob_arrows(
            ax1,
            recon_np[:, [0, 1]],
            plotlims["x"],
            plotlims["y"],
            marker_config["reconstructed"]["color"],
        )
        # Draw grid lines on 2D plots if requested
        if grid is not None:
            try:
                g = float(grid)
                if g > 0:
                    x0, x1 = plotlims["x"]
                    y0, y1 = plotlims["y"]
                    xt = np.arange(np.floor(x0 / g) * g, x1 + g, g)
                    yt = np.arange(np.floor(y0 / g) * g, y1 + g, g)
                    
                    ax1.set_xticks(xt)
                    ax1.set_yticks(yt)
                    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
            except Exception:
                pass
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(
        orig_np[:, 0],
        orig_np[:, 2],
        color=marker_config["original"]["color"],
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax2.scatter(
        recon_np[:, 0],
        recon_np[:, 2],
        color=marker_config["reconstructed"]["color"],
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    _add_gt_inset(ax2, orig_np, 0, 2, "XZ")
    ax2.set_title("XZ View")
    if plotlims is not None:
        ax2.set_xlim(plotlims["x"])
        ax2.set_ylim(plotlims["z"])
        _draw_oob_arrows(
            ax2,
            orig_np[:, [0, 2]],
            plotlims["x"],
            plotlims["z"],
            marker_config["original"]["color"],
        )
        _draw_oob_arrows(
            ax2,
            recon_np[:, [0, 2]],
            plotlims["x"],
            plotlims["z"],
            marker_config["reconstructed"]["color"],
        )
        if grid is not None:
            try:
                g = float(grid)
                if g > 0:
                    x0, x1 = plotlims["x"]
                    y0, y1 = plotlims["z"]
                    xt = np.arange(np.floor(x0 / g) * g, x1 + g, g)
                    yt = np.arange(np.floor(y0 / g) * g, y1 + g, g)
                    
                    ax2.set_xticks(xt)
                    ax2.set_yticks(yt)
                    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
            except Exception:
                pass
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(
        orig_np[:, 2],
        orig_np[:, 1],
        color=marker_config["original"]["color"],
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax3.scatter(
        recon_np[:, 2],
        recon_np[:, 1],
        color=marker_config["reconstructed"]["color"],
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    _add_gt_inset(ax3, orig_np, 2, 1, "YZ")
    ax3.set_title("YZ View")
    if plotlims is not None:
        ax3.set_xlim(plotlims["z"])
        ax3.set_ylim(plotlims["y"])
        _draw_oob_arrows(
            ax3,
            orig_np[:, [2, 1]],
            plotlims["z"],
            plotlims["y"],
            marker_config["original"]["color"],
        )
        _draw_oob_arrows(
            ax3,
            recon_np[:, [2, 1]],
            plotlims["z"],
            plotlims["y"],
            marker_config["reconstructed"]["color"],
        )
        if grid is not None:
            try:
                g = float(grid)
                if g > 0:
                    x0, x1 = plotlims["z"]
                    y0, y1 = plotlims["y"]
                    xt = np.arange(np.floor(x0 / g) * g, x1 + g, g)
                    yt = np.arange(np.floor(y0 / g) * g, y1 + g, g)
                    
                    ax3.set_xticks(xt)
                    ax3.set_yticks(yt)
                    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
            except Exception:
                pass
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Y")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(
        orig_np[:, 0],
        orig_np[:, 1],
        orig_np[:, 2],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax4.scatter(
        recon_np[:, 0],
        recon_np[:, 1],
        recon_np[:, 2],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
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

    return cd, save_path


def visualize_xyz_batch_grid(
    fname,
    original_xyz,
    reconstructed_xyz,
    title="",
    save_dir="/home/palakons/D-PCC/output/plots",
    grid=None,
    max_cols=4,
    fig_size=(4.5, 4.5),
    device="cpu",
):
    """Plot a whole batch as a grid of per-sample overlays and save one image."""
    os.makedirs(save_dir, exist_ok=True)

    orig = original_xyz.detach().cpu() if torch.is_tensor(original_xyz) else torch.tensor(original_xyz)
    recon = reconstructed_xyz.detach().cpu() if torch.is_tensor(reconstructed_xyz) else torch.tensor(reconstructed_xyz)

    if orig.dim() != 3 or recon.dim() != 3:
        raise ValueError("visualize_xyz_batch_grid expects batched point clouds with shape (B, 3, N) or (B, N, 3).")

    if orig.shape[1] != 3 and orig.shape[-1] == 3:
        orig = orig.transpose(1, 2).contiguous()
    if recon.shape[1] != 3 and recon.shape[-1] == 3:
        recon = recon.transpose(1, 2).contiguous()

    batch_size = orig.shape[0]
    n_cols = min(max_cols, batch_size)
    n_rows = int(math.ceil(batch_size / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows), squeeze=False)

    cds = []
    for idx in range(batch_size):
        ax = axs[idx // n_cols][idx % n_cols]
        gt_np = orig[idx].transpose(0, 1).numpy()
        recon_np = recon[idx].transpose(0, 1).numpy()
        cd_gt = torch.tensor(gt_np[None, :, :3], device=device, dtype=torch.float32)
        cd_recon = torch.tensor(recon_np[None, :, :3], device=device, dtype=torch.float32)
        cd, _ = pt3d_chamfer_distance(cd_gt, cd_recon)
        cd_val = cd.item()
        cds.append(cd_val)

        ax.scatter(gt_np[:, 0], gt_np[:, 1], s=8, alpha=0.35, color="blue", label="GT" if idx == 0 else None)
        ax.scatter(recon_np[:, 0], recon_np[:, 1], s=8, alpha=0.55, color="red", label="Recon" if idx == 0 else None)
        ax.set_title(f"S{idx} CD:{cd_val:.2e}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        if grid is not None:
            try:
                g = float(grid)
                if g > 0:
                    x0, x1 = np.min(gt_np[:, 0]), np.max(gt_np[:, 0])
                    y0, y1 = np.min(gt_np[:, 1]), np.max(gt_np[:, 1])
                    ax.set_xticks(np.arange(np.floor(x0 / g) * g, x1 + g, g))
                    ax.set_yticks(np.arange(np.floor(y0 / g) * g, y1 + g, g))
                    ax.grid(True, linestyle="--", linewidth=0.4, color="gray", alpha=0.5)
            except Exception:
                pass

    for idx in range(batch_size, n_rows * n_cols):
        axs[idx // n_cols][idx % n_cols].axis("off")

    if title:
        plt.suptitle("\n".join(textwrap.wrap(title, width=120)), fontsize=14)
    if batch_size > 0:
        axs[0][0].legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return float(np.mean(cds)) if cds else 0.0, save_path


def plot_xy_timestep(fname, gt, coords, title, save_dir, figsize=(8, 6)):
    """
    Plot separated X-Y scatter, one plot per timestep, horizontally subplot, with different colors for each sample in the batch.
    Args:
    fname: str - filename for the saved plot
    gt: (B,N, 3) tensor - ground truth point cloud for reference, will be plotted in black in all subplots
    coords: list of coordinates for each timestep (timestep x B x N x 3)
    title: str - title for the plot
    save_dir: str - directory to save the plot
    """
    n_rows = int(len(coords) ** 0.5)
    n_cols = int(math.ceil(len(coords) / n_rows))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False,
    )
    axs_flat = axs.flatten()
    n_current = len(coords[0])  # number of samples in the current batch

    # Use a safe device for CD calculation
    calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert GT to tensor once
    gt_tensor = torch.tensor(gt, device=calc_device)

    for t in range(len(coords)):
        ax = axs_flat[t]
        # set title
        ax.set_title(f"T {len(coords)-1-t}")
        for i in range(n_current):
            sample_coords = coords[t][i]  # (N, 3)
            sample_tensor = torch.tensor(sample_coords[None, :, :3], device=calc_device)

            # Calculate CD for label
            cd_val, _ = pt3d_chamfer_distance(
                gt_tensor[i : i + 1, :, :3], sample_tensor
            )

            ax.scatter(
                sample_coords[:, 0],
                sample_coords[:, 1],
                label=f"S{i}: {cd_val.item():.1f}",
                s=10,
                alpha=0.6,
                color="red",
            )
            # plot gt in blue
            ax.scatter(
                gt[i, :, 0], gt[i, :, 1], color="blue", label="GT", s=10, alpha=0.4
            )
            ax.set_aspect("equal", adjustable="box")
        if t == 0 or t == len(coords) - 1:
            ax.legend(fontsize="x-small")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_multi_xy_timestep(fname, gt, coords, title, save_dir, figsize=(8, 6)):
    """
    Plot separated X-Y scatter, one plot per timestep, horizontally subplot, with different colors for each sample in the batch.
    Args:
    fname: str - filename for the saved plot
    gt: (B,N, 3) tensor - ground truth point cloud for reference, will be plotted in black in all subplots
    coords: dict of coordinates for each timestep, with keys as sample indices and values as list of coordinates for each timestep (B x T x N x 3)
    title: str - title for the plot
    save_dir: str - directory to save the plot
    """
    assert isinstance(
        coords, dict
    ), "Coords should be a dict with sample indices as keys and list of coordinates for each timestep as values"
    # assert same number of timesteps for each sample as the number of steps in coord
    n_current = len(coords[list(coords.keys())[0]])

    for sample in coords:
        assert (
            len(coords[sample]) == n_current
        ), f"All samples should have the same number of timesteps as the number of steps in coord. Sample {sample} has {len(coords[sample])} timesteps, expected {n_current}"
    scatter_plot_markers = [
        "x",
        "^",
        "s",
        "D",
        "P",
        "*",
        "H",
        "v",
        "<",
        ">",
    ]  # Add more markers for more samples

    n_rows = int(n_current**0.5)
    n_cols = int(math.ceil(n_current / n_rows))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False,
    )
    axs_flat = axs.flatten()

    # Use a safe device for CD calculation
    calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert GT to tensor once
    gt_tensor = torch.tensor(gt, device=calc_device)

    for t in trange(n_current, leave=False):  # time step
        ax = axs_flat[t]
        # set title
        ax.set_title(f"T {n_current-1-t}")
        B, N, _ = coords[sample][t].shape
        for i in range(
            B
        ):  # sample in batch, B is 1 in our case, but we keep it general for future use
            for j, sample in enumerate(
                coords
            ):  # sample index in dict, we use j to select marker and color for each sample
                sample_coords = coords[sample][t][i]  # (N, 3)
                sample_tensor = torch.tensor(
                    sample_coords[None, :, :3], device=calc_device
                )
                cd_val, _ = pt3d_chamfer_distance(
                    gt_tensor[i : i + 1, :, :3], sample_tensor
                )

                ax.scatter(
                    sample_coords[:, 0],
                    sample_coords[:, 1],
                    label=f"S{i}/{sample}: {cd_val.item():.1f}",
                    s=10,
                    alpha=0.6,
                    color=plt.cm.tab10(j),
                    marker=scatter_plot_markers[j % len(scatter_plot_markers)],
                )
            # plot gt in blue
            ax.scatter(
                gt[i, :, 0], gt[i, :, 1], color="blue", label="GT", s=10, alpha=0.4
            )
            ax.set_aspect("equal", adjustable="box")
        if t == 0 or t == n_current - 1:
            # legen outside of plot area
            ax.legend(
                fontsize="x-small",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        # remove the frame but keep the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cd_timestep_curve(fname, cds, title, save_dir, figsize=(8, 6)):
    """
    Plot separated Chamfer Distance curves vs Timestep curve.
    Args:
    fname: str - filename for the saved plot
    cds: list of chamfer distances for each timestep (timestep x B)
    title: str - title for the plot
    save_dir: str - directory to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    # print(f" [plot_cd_timestep_curve] {len(cds)}, first CD: {cds[0]}")
    n_current = len(cds[0])  # number of samples in the current batch
    for i in range(n_current):
        sample_cds = [cd[i] for cd in cds]
        # PTv3 library might provide them in forward/backward order,
        # we plot sample_cds[::-1] to show high-noise to clean.
        plot_vals = sample_cds[::-1]
        (line,) = ax.plot(plot_vals, label=f"Sample {i}")
        # Annotate first and last points
        ax.annotate(
            f"{plot_vals[0]:.1f}",
            (0, plot_vals[0]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color=line.get_color(),
        )
        ax.annotate(
            f"{plot_vals[-1]:.1f}",
            (len(plot_vals) - 1, plot_vals[-1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color=line.get_color(),
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Chamfer Distance")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    save_path = os.path.join(save_dir, f"{fname}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


class PTv3Dnsr(nn.Module):
    def __init__(
        self,
        n_in_channels=3,
        context_channels=256,
        out_channels=3,
        grid_size=0.02,
        shuffle_orders=True,
        serialized_inverse=False,
        n_stages=5,  # Allow sweeping n_stages from 2 to 5
        seed=42,
        backbone_type="full",
        param_multiplier=1.0,
        project_coord_dim=0,  # Optional: project 3D coordinates to higher dim, e.g., 32, 64. If 0, use raw coords.
        time_conditioning_mode="pdnorm_only",  # How to inject time: "pdnorm_only", "feat_add", "hybrid", "feat_concat"
        use_cpe=True,
        use_head=True,  # If False, decoder outputs out_channels directly (no projection bottleneck)
        scene_embed_dim=0,  # Scene embedding dimension; if > 0, create cond_mlp to map scene condition to context
    ):
        """
        PTv3 Denoiser for DDPM. Supports dynamic depth via n_stages.
        
        Args:
            project_coord_dim: If set, project 3D coordinates to this dimension using a Linear layer.
                              Overrides n_in_channels for PTv3 input. If 0, use raw coords (n_in_channels=3).
            time_conditioning_mode: How to condition on time embeddings:
                - "pdnorm_only" (default): Time injected via PDNorm context/condition (backward compatible)
                - "feat_add": Project time to feat dimension and add to feat
                - "hybrid": Both PDNorm context AND feat_add
                - "feat_concat": Concatenate time to feat (non-backward-compatible, increases in_channels)
                - "no_time": Do not inject timestep anywhere (neither PDNorm nor features)
            use_head: If True (default), use final linear head projection for backward compatibility.
                     If False, decoder outputs out_channels directly (removes gradient bottleneck).
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.n_in_channels = n_in_channels
        self.context_channels = context_channels
        self.grid_size = grid_size
        self.use_cpe = use_cpe
        self.serialized_inverse = serialized_inverse
        self.project_coord_dim = project_coord_dim
        self.time_conditioning_mode = time_conditioning_mode
        self.use_head = use_head
        self.scene_embed_dim = scene_embed_dim
        
        assert time_conditioning_mode in ["pdnorm_only", "feat_add", "hybrid", "feat_concat", "no_time"], \
            f"Invalid time_conditioning_mode: {time_conditioning_mode}"
        
        # Optional coordinate projection layer
        if self.project_coord_dim != 0:
            self.coord_projector = nn.Linear(3, self.project_coord_dim)
            nn.init.xavier_uniform_(self.coord_projector.weight)
            nn.init.constant_(self.coord_projector.bias, 0)
            # Effective input channels to PTv3 is the projection dimension
            feat_dim = self.project_coord_dim
        else:
            self.coord_projector = None
            feat_dim = self.n_in_channels
        
        # Time-to-feat projection layers (for feat_add, hybrid, feat_concat modes)
        if self.time_conditioning_mode in ["feat_add", "hybrid"]:
            # Project time embedding to feat dimension and add
            self.time_to_feat_add = nn.Linear(self.context_channels, feat_dim)
            nn.init.xavier_uniform_(self.time_to_feat_add.weight)
            nn.init.constant_(self.time_to_feat_add.bias, 0)
        else:
            self.time_to_feat_add = None
        
        if self.time_conditioning_mode == "feat_concat":
            # Concatenate time embedding to feat
            ptv3_in_channels = feat_dim + self.context_channels
        else:
            ptv3_in_channels = feat_dim

        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            nn.SiLU(),
            nn.Linear(context_channels, context_channels),
        )
        
        # 1b. Scene Condition MLP (if scene_embed_dim > 0)
        if self.scene_embed_dim > 0:
            # Scene to PDNorm context pathway
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.scene_embed_dim, context_channels),
                nn.SiLU(),
                nn.Linear(context_channels, context_channels),
            )
            for layer in self.cond_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
            
            # Scene to feature pathway (dual-pathway conditioning)
            self.scene_feat_mlp = nn.Sequential(
                nn.Linear(self.scene_embed_dim, feat_dim),
                nn.SiLU(),
                nn.Linear(feat_dim, feat_dim),
            )
            for layer in self.scene_feat_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
            
            # Learnable strength for scene context mixing (allows adaptive weighting)
            self.scene_strength = nn.Parameter(torch.ones(1))
        else:
            self.cond_mlp = None
            self.scene_feat_mlp = None
            self.scene_strength = None
        def _scale_int_tuple(tpl, f, min_val=1):
            return tuple(max(min_val, int(x * f)) for x in tpl)

        # def _adjust_heads_for_channels(channels_tpl, heads_tpl, f):
        #     out = []
        #     for ch, h_base in zip(channels_tpl, heads_tpl):
        #         h = max(1, int(h_base * f))
        #         if h > ch:
        #             h = ch
        #         # find largest h' <= h that divides ch
        #         while h > 1 and (ch % h) != 0:
        #             print(f"Adjusting heads: {f} channels {ch} not divisible by heads {h}, reducing heads...")
        #             h -= 1
        #         out.append(max(1, h))
        #     return tuple(out)

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

        print(f"Original full parameters:\nEnc Depths: {full_enc_depths}\nEnc Channels: {full_enc_channels}\nEnc Heads: {full_enc_num_head}\nEnc Patch Size: {full_enc_patch_size}\nDec Depths: {full_dec_depths}\nDec Channels: {full_dec_channels}\nDec Heads: {full_dec_num_head}\nDec Patch Size: {full_dec_patch_size}")

        full_enc_depths = _scale_int_tuple(full_enc_depths, param_multiplier, min_val=1)
        full_dec_depths = _scale_int_tuple(full_dec_depths, param_multiplier, min_val=1)
        full_enc_num_head = _scale_int_tuple(full_enc_num_head, param_multiplier, min_val=1)
        full_dec_num_head = _scale_int_tuple(full_dec_num_head, param_multiplier, min_val=1)
        full_enc_patch_size = _scale_int_tuple(full_enc_patch_size, param_multiplier, min_val=1)
        full_dec_patch_size = _scale_int_tuple(full_dec_patch_size, param_multiplier, min_val=1)

        #ch = 16*head
        full_enc_channels = tuple(16 * h for h in full_enc_num_head)
        full_dec_channels = tuple(16 * h for h in full_dec_num_head)

        print(f"Multipled full parameters:\nEnc Depths: {full_enc_depths}\nEnc Channels: {full_enc_channels}\nEnc Heads: {full_enc_num_head}\nEnc Patch Size: {full_enc_patch_size}\nDec Depths: {full_dec_depths}\nDec Channels: {full_dec_channels}\nDec Heads: {full_dec_num_head}\nDec Patch Size: {full_dec_patch_size}")



        # 3. Cut off stages based on n_stages (must be at least 2)
        n_stages = max(2, min(n_stages, 5))
        stride = full_stride[: n_stages - 1]
        enc_depths = full_enc_depths[:n_stages]
        enc_channels = full_enc_channels[:n_stages]
        enc_num_head = full_enc_num_head[:n_stages]
        enc_patch_size = full_enc_patch_size[:n_stages]

        dec_depths = full_dec_depths[: n_stages - 1]
        dec_channels = list(full_dec_channels[: n_stages - 1])
        # If not using head, set last decoder channels to out_channels; we'll adjust
        # decoder heads below to ensure block constraints hold.
        if not use_head and dec_channels:
            dec_channels[-1] = out_channels
        dec_channels = tuple(dec_channels)
        dec_num_head = list(full_dec_num_head[: n_stages - 1])
        dec_patch_size = full_dec_patch_size[: n_stages - 1]

        # If using no-head mode, ensure dec_num_head[0] divides out_channels.
        # If not, reduce the number of heads (down to 1) so the Block assertion
        # `channels % num_heads == 0` will hold. Prefer the largest head count <=
        # original that divides out_channels.
        if not use_head and dec_channels and len(dec_num_head) > 0:
            orig_h0 = int(dec_num_head[0])
            new_h0 = None
            for h in range(orig_h0, 0, -1):
                if out_channels % h == 0:
                    new_h0 = h
                    break
            if new_h0 is None:
                new_h0 = 1
            if new_h0 != orig_h0:
                print(f"[PTv3Dnsr] Adjusting decoder head count: dec_num_head[0] {orig_h0} -> {new_h0} to match out_channels={out_channels}")
            dec_num_head[0] = new_h0
        dec_num_head = tuple(dec_num_head)

        # 4. PTv3 Backbone
        if backbone_type == "full":
            self.ptv3 = PointTransformerV3(
                in_channels=ptv3_in_channels,
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
                use_cpe=self.use_cpe,
            )
        elif backbone_type == "full-nodrop":
            self.ptv3 = PointTransformerV3(
                in_channels=ptv3_in_channels,
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
                drop_path=0.0,
                pdnorm_bn=False,
                use_cpe=self.use_cpe,
            )
        elif backbone_type == "simple":
            self.ptv3 = PointTransformerV3(
                in_channels=ptv3_in_channels,
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
                qkv_bias=True,  # Enable qkv bias for better expressiveness, especially important when using fewer layers
                attn_drop=0.0,  # Disable attention dropout for more stable training with normalized coords
                proj_drop=0.0,  # Disable projection dropout for more stable training with normalized coords
                enable_rpe=False,  # Disable RPE for more stable training with normalized coords
                stride=[
                    1 for _ in stride
                ],  # Disable merging by setting stride to 1, which should help maintain point count and stability with normalized coords
                enc_depths=[1 for _ in enc_depths],
                dec_depths=[1 for _ in dec_depths],
                drop_path=0.0,  # Disable drop path for more stable training with normalized coords
                pdnorm_bn=False,  # Disable batch norm for more stable training with normalized coords
                use_cpe=self.use_cpe,
            )
        elif backbone_type == "simpler":
            dec_channels = (8,)  # minimal channels for decoder
            self.ptv3 = PointTransformerV3(
                in_channels=3,  # or 6 if your data needs it
                enable_flash=False,
                pdnorm_ln=False,
                pdnorm_adaptive=False,
                pdnorm_conditions=(),
                shuffle_orders=False,
                enc_channels=(8,8,),  # minimal channels
                enc_num_head=(1,1,),  # 1 attention head
                enc_patch_size=(8,8,),  # small patch size
                dec_channels=dec_channels,  # minimal channels
                dec_num_head=(1,),  # 1 attention head
                dec_patch_size=(8,),  # small patch size
                qkv_bias=False,
                attn_drop=0.0,
                proj_drop=0.0,
                enable_rpe=False,
                stride=(2,),  # minimal stride
                enc_depths=(1,1,),  # 1 block per stage
                dec_depths=(1,),  # 1 block per decoder
                drop_path=0.0,
                pdnorm_bn=False,
                order=("z",),  # default: ("z", "z-trans"),
                mlp_ratio=2,  # default 4
                qk_scale=None,
                pre_norm=True,
                upcast_attention=False,
                upcast_softmax=False,
                enc_mode=False,
                pdnorm_decouple=False,  # deault True
                pdnorm_affine=False,  # default True
                use_cpe=self.use_cpe,
            )
        else:
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. Choose from 'full', 'simple', or 'simpler'."
            )

        # 5. Output Head
        if use_head:
            # Traditional: linear projection from last decoder channels to out_channels
            self.head = nn.Linear(dec_channels[0], out_channels)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.constant_(self.head.bias, 0)
        else:
            # No head: decoder outputs out_channels directly
            self.head = None

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
        device = next(self.parameters()).device
        x = x.to(device)
        t = t.to(device)

        B, C, N = x.shape

        # --- FIX: Ensure coordinates are not interleaved ---
        # Transpose (B, 3, N) -> (B, N, 3) so that flatten gives (x, y, z) rows
        x_pts = x.transpose(1, 2).contiguous()
        coord = x_pts[:,:, :3].reshape(-1, 3).to(device)

        # Use provided condition if available, otherwise fallback to default
        ptv3_condition = "DDPM"

        # 1. Prepare Features
        if self.coord_projector is not None:
            # Project 3D coordinates to higher dimension
            feat = self.coord_projector(coord)  # (B*N, 3) -> (B*N, project_coord_dim)
        else:
            # Use raw coordinates as features
            feat = torch.zeros(coord.shape[0], self.n_in_channels, device=device)
            feat[:, :3] = coord.clone()  # Use original coordinates as features; PTv3 will learn to ignore them if not needed

        # 2. Prepare Batch indices and Offsets for Pointcept
        # Batch: [0, 0, ... 1, 1, ... B-1, B-1] (B*N elements)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        # Offset: Cumulative point counts [N, 2N, 3N, ... B*N]
        offset = torch.arange(1, B + 1, device=device, dtype=torch.long) * N

        # 3. Process Time/Conditioning
        try:
            t_emb = self.get_time_embedding(t)  # (B, context_channels)
        except Exception as e:
            print(f"Error in time embedding: {e}")
            print("[PTv3Dnsr] t shape: ", t.shape)  # Should be (B,)
            print(f"t: {t}")
            print(f"x shape: {x.shape}, x device: {x.device}")
            raise e
        
        # Apply time-based feature modulation depending on conditioning mode
        if self.time_conditioning_mode == "feat_add" or self.time_conditioning_mode == "hybrid":
            # Add time embedding to features AND keep PDNorm context
            t_feat = self.time_to_feat_add(t_emb[batch]).to(device)  # (B*N, feat_dim)
            feat = feat + t_feat
        elif self.time_conditioning_mode == "feat_concat":
            # Concatenate time embedding to features
            t_feat = t_emb[batch].to(device)  # (B*N, context_channels)
            feat = torch.cat([feat, t_feat], dim=1)  # (B*N, feat_dim + context_channels)
        # else: pdnorm_only - no feat modification, time stays in context
        
        # Apply scene-based feature modulation (dual-pathway conditioning)
        if condition is not None and self.scene_feat_mlp is not None:
            scene_feat = self.scene_feat_mlp(condition.to(device))  # (B, feat_dim)
            scene_feat_expanded = scene_feat[batch].to(device)  # (B*N, feat_dim)
            feat = feat + self.scene_strength * scene_feat_expanded
        
        point_context = t_emb[batch].to(device)
        # print("[PTv3Dnsr] point_context shape: ", point_context.shape,"t_emb", t_emb.shape)  # Should be (B*N, context_channels)

        # 4. Construct data_dict
        # Only pass timestep context into PDNorm when requested by mode
        if self.time_conditioning_mode in ["pdnorm_only", "hybrid"]:
            pdnorm_context = point_context
        else:
            # Provide zeros of the correct shape so PDNorm receives no timestep signal
            pdnorm_context = torch.zeros_like(point_context, device=device)
        
        # 4b. Add scene conditioning to PDNorm context if available
        if condition is not None and self.cond_mlp is not None:
            # print(f"[PTv3Dnsr] conditioning to PDNorm context. condition shape: {condition.shape} pdnorm_context shape: {pdnorm_context.shape}")  # Debug print for condition shape
            # condition: (B, scene_embed_dim)
            cond_emb = self.cond_mlp(condition.to(device))  # (B, context_channels)
            # print(f"[PTv3Dnsr] cond_emb shape: {cond_emb.shape}")  # Debug print for cond_emb shape

            # Expand to per-point: (B*N, context_channels)
            cond_emb_expanded = cond_emb[batch].to(device)
            # print(f"[PTv3Dnsr] cond_emb_expanded shape: {cond_emb_expanded.shape}")  # Debug print for expanded condition shape

            # Add scene condition to PDNorm context with learnable strength for context mixing
            pdnorm_context = pdnorm_context + self.scene_strength * cond_emb_expanded
            # print(f"[PTv3Dnsr] pdnorm_context shape after adding condition: {pdnorm_context.shape}")  # Debug print for final context shape


        data_dict = {
            "coord": coord,
            "feat": feat,
            "batch": batch,
            "offset": offset,  # RESTORED
            "grid_size": self.grid_size,  # Increased slightly for more stable voxelization with normalized coords
            "context": pdnorm_context,
            # keep a copy of the original coordinates so we can validate ordering
            # preserve original coordinates under a private key so PTv3 internals can't clobber them
            # "_preserved_coord": coord.clone(),
            "condition": ptv3_condition,
        }
        org_coord = coord.clone()
        # Keep a copy of the original coordinates for debugging
        # print("[PTv3Dnsr] feat shape before PTv3: ", feat.shape)  # Should be (B*N, n_in_channels)
        # 5. Backbone Forward

        #input: [B*N,3]

        point_out = self.ptv3(data_dict)  
        #ouput: [B*N, C_Out] 
        #check against org coordinates to ensure they are not interleaved or shuffled in a way that breaks the (x,y,z) grouping per point
        # print(f"[PTv3Dnsr] same coord order after PTv3: {torch.allclose(org_coord, point_out['coord'])}, shape  {org_coord.shape}")  # Should be True if coordinates are preserved correctly
        # print("[PTv3Dnsr] coord after PTv3: ", point_out["coord"].detach().cpu().numpy()[:10])  # Print 
        # print("[PTv3Dnsr] coord before PTv3: ", coord.detach().cpu().numpy()[:10])  # Print original coordinates for comparison
        # print(f"serializd order: {point_out['serialized_order']}")  # Check if serialized order is the same as batch order, which would indicate no shuffling
        

        # print("[PTv3Dnsr] point_out keys after PTv3 forward: ", point_out.keys()) #
        #                ['coord', 'feat', 'batch', 'offset', 'grid_size', 'context', 'condition', 'order', 'grid_coord', 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 'sparse_shape', 'sparse_conv_feat', 'pad', 'unpad', 'cu_seqlens_key']

        # explain each key:
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
        if self.serialized_inverse :
            # point_out["serialized_inverse"] is a list of inverse indices for each serialization order.
            # We use the first one [0] as it corresponds to the order used in the final layer.
            feat = feat[point_out["serialized_inverse"][0]]
            # print("[PTv3Dnsr] feat shape after optional unshuffle: ", feat.shape) # should still be (B*N, C_out)
            # print("[PTv3Dnsr] feat after optional unshuffle: ", feat.detach().cpu())

        if self.use_head and self.head is not None:
            out = self.head(feat)  # Apply head projection
        else:
            out = feat  # Decoder already outputs out_channels
        # print("[PTv3Dnsr] out shape before reshaping: ", out.shape)  #[2, 6]
        # print("[PTv3Dnsr] out : ", out.detach().cpu()[:,:3].numpy().flatten())  # Print predicted noise for the first few points

        # Reshape to (B, N, 6) and transpose back to (B, 6, N) to align with vision style
        if out.shape[0] != B * N:
            # If points are merged, PointTransformerV3 returns fewer than B*N points.
            # We need grid_size to be small enough that points stay unique,
            # but large enough for the 5x5x5 conv kernel to see neighbors.
            raise RuntimeError(
                f"Point count mismatch: {out.shape[0]} vs {B*N}. grid_size={self.grid_size} might be too large and merging points."
            )
        out = out.view(B, N, -1).transpose(1, 2).contiguous()  # (B, out_channels*2, N)
        # print("[PTv3Dnsr] out shape after reshaping: ", out.shape) #[1, 6, 2]
        return out
    

def check_voxel_collisions(pts, grid_size):
    grid_coord = torch.div(
        pts - pts.min(dim=0).values,
        grid_size,
        rounding_mode="trunc"
    ).int()

    unique_voxels, counts = torch.unique(grid_coord, dim=0, return_counts=True)

    num_points = pts.shape[0]
    num_unique = unique_voxels.shape[0]
    num_collided_points = counts[counts > 1].sum().item()
    num_collision_voxels = (counts > 1).sum().item()
    max_points_in_one_voxel = counts.max().item()

    print(f"N points                : {num_points}")
    print(f"Unique voxels           : {num_unique}")
    print(f"Collision voxels        : {num_collision_voxels}")
    print(f"Points in collided bins : {num_collided_points}")
    print(f"Max points in one voxel : {max_points_in_one_voxel}")
    print(f"Voxel occupancy ratio   : {num_unique / num_points:.4f}")

    return {
        "grid_coord": grid_coord,
        "counts": counts,
        "num_points": num_points,
        "num_unique": num_unique,
        "collision_voxels": num_collision_voxels,
        "collided_points": num_collided_points,
        "max_points_in_one_voxel": max_points_in_one_voxel,
        "occupancy_ratio": num_unique / num_points,
    }

def get_noise_old(normalized_pointcloud_b3n, debug_mode=0, fixed_noise_ref=None):
    B = normalized_pointcloud_b3n.shape[0]
    # --- DEBUG MODE NOISE SELECTION ---
    if debug_mode == 1:
        noise = torch.zeros_like(normalized_pointcloud_b3n)
    elif debug_mode == 2:
        noise = torch.full_like(normalized_pointcloud_b3n, 0.5)
    elif debug_mode in [3, 4]:
        noise = fixed_noise_ref.repeat(B, 1, 1)
    else:
        noise = torch.randn_like(normalized_pointcloud_b3n)
    return noise

def get_noise(shape, debug_mode=0,fixed_noise_ref=None):
    if debug_mode == 0 or debug_mode == 5:
        return torch.randn(shape)
    elif debug_mode == 1:
        # Return zeros for no noise, to test if the model can learn identity mapping
        return torch.zeros(shape)
    elif debug_mode == 2:
        return torch.full(shape, 0.5)  # Constant noise for deterministic behavior, can be tuned to other values
    elif debug_mode == 3 or debug_mode == 4:
        return fixed_noise_ref.repeat(shape[0], 1, 1)
        # noise = fixed_noise_ref.expand(shape)
    else:
        raise ValueError(f"Unsupported debug_mode: {debug_mode}")

def train_ptv3_ddpm(config, checkpoint_dir, tb_dir, plot_dir):
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    checkpoint = None
    writer = SummaryWriter(tb_dir)

    if config["dit_checkpoint"] != "":
        print(f"Loading checkpoint from {config['dit_checkpoint']}...")
        checkpoint = torch.load(
            config["dit_checkpoint"], map_location=config["device"], weights_only=False
        )

        # update config (with exceptions) to become the one in the checkpoint, so that we can resume training with the same config as before
        print("overriding config with checkpoint config...")
        exceptions = [
            "dit_checkpoint",
            "ae_checkpoint",
            "dit_epochs",
            "ae_epochs",
            "exp_name",
        ]
        checkpoint_config = checkpoint.get("config", {})
        for key in checkpoint_config:
            if key not in exceptions and key in config:
                if (
                    config[key] != checkpoint_config[key]
                ):  # allow overriding epoch-related configs for resuming with different training length
                    print(
                        f"Overriding config key {key}: {config[key]} -> {checkpoint_config[key]}"
                    )
                config[key] = checkpoint_config[key]
            else:
                print(
                    f"Keeping current config key {key}: {config.get(key, 'N/A')} (not in checkpoint or in exceptions)"
                )

    # model = PTv3().to(config["device"]) #TODO: make this a PTv3 model with the appropriate conditioning for the RGB image's WAN VAE Latent
    print("num_inference_steps: ", config["num_inference_steps"])
    print(
        "batch_size: ",
        config["batch_size"],
        "num_points: ",
        config["num_points"],
        "ditepochs: ",
        config["dit_epochs"],
    )
    # Swap denoiser implementation without changing the default PTv3 path.
    if config["denoiser_model"] == "simple_unet":
        print("Using SimplePointUNet as denoiser!")
        model = SimplePointUNet(
            in_channels=3,
            base_channels=config['latent_dim'],
            out_channels=3,
            num_layers=config["ptv3_n_stages"],
        ).to(config["device"])
    elif config["denoiser_model"] == "minimal":
        print("Using MinimalPointDenoiser as denoiser!")
        model = PointNetLikeDenoiser(
            hidden_channels=config['latent_dim'],
            time_embed_dim=32,
            num_blocks=config["ptv3_n_stages"],
        ).to(config["device"])
    elif config["denoiser_model"] == "ptv3":
        print("Using PTv3Dnsr as denoiser with the following parameters:")
        print(
            "prams for PTv3Dnsr: ",
            config["ptv3_grid_size"],
            "ptv3_shuffle_orders",
            config["ptv3_shuffle_orders"],
            "ptv3_serialized_inverse",
            config["ptv3_serialized_inverse"],
        )
        model = PTv3Dnsr(
            n_in_channels=3,  #  We use raw coordinates as input features, so n_in_channels=3. PTv3 will learn to ignore them if not needed.
            context_channels=256,  # Standard size for time embedding, can be tuned
            out_channels=3,  # We predict noise only, not variance, so out_channels=3 instead of 6
            grid_size=config["ptv3_grid_size"],
            shuffle_orders=config["ptv3_shuffle_orders"],
            serialized_inverse=config["ptv3_serialized_inverse"],
            n_stages=config["ptv3_n_stages"],
            seed=config["seed"],
            backbone_type=config["ptv3_backbone"],
            param_multiplier=config["ptv3_param_multiplier"],
            time_conditioning_mode=config["ptv3_time_conditioning_mode"],
            project_coord_dim=config["ptv3_project_coord_dim"],
        ).to(config["device"])
    else:
        raise ValueError(
            f"Unsupported denoiser_model: {config['denoiser_model']}, choose from ['simple_unet', 'minimal', 'ptv3']"
        )
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params/1e6:.2f} million parameters.")
    writer.add_text("Model/Architecture", str(model), global_step)
    writer.add_scalar("Model/NumParams", num_params, global_step)

    # Use HuggingFace DDPMScheduler
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        clip_sample=config["ddpm_clip_sample"],
        prediction_type="epsilon",  # We predict noise (epsilon) directly, not x0 or variance
    )
    ddpm_scheduler.set_timesteps(config["num_inference_steps"])

    optimizer = makeOptimizer(model, config)
    lr_scheduler = makeLR_Scheduler(optimizer, config)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        global_step = checkpoint.get("global_step", 0)
        print(
            f"Resuming from epoch {start_epoch} with best_loss {best_loss:.4f} and global_step {global_step}"
        )

    train_dataset, val_dataset = splitDataset(
        makeDataset(
            config,
            get_occ_grid=False,
            get_camera=True,
            get_wan_vae=False,
        ),
        split=0.5,
    )
    data_mean = train_dataset.dataset.data_mean.to(config["device"])
    data_std = train_dataset.dataset.data_std.to(config["device"])

    train_dataloader = makeDataloaders(
        train_dataset,
        config,
        is_train=True,
    )

    if config["denoiser_model"] == "ptv3":
        #check for collision, assume one batch, one sample
        test_point = next(iter(train_dataloader))["filtered_radar_data"][0, :, :3].to(config["device"])
        print(f"shape of test_point: {test_point.shape}, device: {test_point.device}")

        stats = check_voxel_collisions(pts=test_point, grid_size=config["ptv3_grid_size"])
        print("Voxel collision stats: ", stats)
        if stats["occupancy_ratio"] < 0.9 or stats["occupancy_ratio"] > .99:
            print(
                f"Grid size {config['ptv3_grid_size']} gives low occupancy ratio {stats['occupancy_ratio']:.4f} ({1-stats['occupancy_ratio']:.4f} collisions ratio). Consider reducing grid size to avoid excessive point merging in PTv3, which can cause dimension mismatches and loss of detail."
            )
            while True:            # suggest a good grid size based on occupancy ratio and exit
                if stats["occupancy_ratio"] < 0.9:
                    config["ptv3_grid_size"] *= 0.8  # reduce grid size by 20%
                elif stats["occupancy_ratio"] > 0.99:
                    config["ptv3_grid_size"] *= 1.2  # increase grid size by 20%
                else:
                    break
                stats = check_voxel_collisions(pts=test_point, grid_size=config["ptv3_grid_size"])
                print(f"Trying grid size {config['ptv3_grid_size']:.4f} gives occupancy ratio {stats['occupancy_ratio']:.4f}")
            print(
                f"Final grid size {config['ptv3_grid_size']} gives occupancy ratio {stats['occupancy_ratio']:.4f} with {stats['collision_voxels']} collision voxels out of {stats['num_unique']} unique voxels. This should help maintain point uniqueness while still allowing for some merging to enable sparse convolution in PTv3."
            )   
            exit()


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
    state = torch.random.get_rng_state()

    print(state.shape, state.dtype)
    print("Random state before setting seed for fixed noise reference: ", state.cpu().numpy()[:10])  # Print first 10 values for verification
    print(f"seed set to {config['seed']} for fixed noise reference.")
    torch.random.manual_seed(config["seed"])  # Ensure reproducibility for fixed noise reference
    state = torch.random.get_rng_state()
    print("Random state after setting seed for fixed noise reference: ", state.cpu().numpy()[:10])  # Print first 10 values for verification

    fixed_noise_ref = torch.randn(1, 3, config["num_points"]).to(config["device"])
    assert config["batch_size"] == 1 or config["debug_mode"] not in [3, 4], "Debug modes 3 and 4 require batch_size of 1 for the fixed noise reference to be valid."

    print(f"fixed noise_ref, first 6 values: {fixed_noise_ref.flatten()[:6].detach().cpu().numpy()}")
    print(f"Random state after creating fixed noise reference (should be the same as after setting seed): ", torch.random.get_rng_state().cpu().numpy()[:10])  # Print first 10 values for verification

    for epoch in epoch_bar:
        train_losses = []
        batch_bar = tqdm(train_dataloader, desc="Train Batches", leave=False)
        time9 = time.time()
        time_record = {"wait":[],"data_prep": [], "noise_add": [], "model_forward": [], "loss_compute": [],"zero_grad":[],"backward":[],"clip_grads":[],"record_norms":[],"other":[]}
        for batch in batch_bar:
            time0 = time.time()
            pointcloud_bn3 = batch["filtered_radar_data"].to(config["device"])[
                :, :, :3
            ]  # (B,N,3)
            batch_bar.set_description(
                f"npoint: org{batch['npoints_original']}/ROI{batch['npoints_after_distance_filter']}/inview{batch['npoints_filtered']}"
            )  # Show point counts in progress bar

            B = pointcloud_bn3.shape[0]
            normalized_pointcloud_bn3 = (pointcloud_bn3 - data_mean[:3]) / data_std[:3]
            # Transpose to vision style (B, 3, N) for diffusion library
            min_val = torch.amin(normalized_pointcloud_bn3, dim=[0, 1])
            max_val = torch.amax(normalized_pointcloud_bn3, dim=[0, 1])
            writer.add_scalar("Data/norm_min_x", min_val[0].item(), global_step)
            writer.add_scalar("Data/norm_max_x", max_val[0].item(), global_step)
            writer.add_scalar("Data/norm_min_y", min_val[1].item(), global_step)
            writer.add_scalar("Data/norm_max_y", max_val[1].item(), global_step)
            writer.add_scalar("Data/norm_min_z", min_val[2].item(), global_step)
            writer.add_scalar("Data/norm_max_z", max_val[2].item(), global_step)
            normalized_pointcloud_b3n = normalized_pointcloud_bn3.transpose(
                1, 2
            ).contiguous()

            # # Ensure conditioning latent exists and has correct shape (B, 16, 2, 60, 104)
            # wan_vae_latent = batch["wan_vae_latent"].to(config["device"])
            # wan_vae_latent = wan_vae_latent * 0

            # --- DEBUG MODE TIMESTEP SELECTION ---

            if config["debug_mode"] in [1, 2, 3, 5]:
                t_val = config["ddpm_fixed_timestep"]
                assert 0 <= t_val < ddpm_scheduler.config.num_train_timesteps, f"ddpm_fixed_timestep must be in [0, {ddpm_scheduler.config.num_train_timesteps-1}]"
                t = torch.full((B,), t_val, device=config["device"]).long()
            else:
                t = torch.randint(
                    0,
                    ddpm_scheduler.config.num_train_timesteps,
                    (B,),
                    device=config["device"],
                ).long()

            model_kwargs = dict()


            noise = get_noise(normalized_pointcloud_b3n.shape,debug_mode=config["debug_mode"],fixed_noise_ref=fixed_noise_ref.to(config["device"])).to(config["device"])

            # Center noise if requested to prevent global drift (Bias fix)
            if config["ptv3_zero_mean_noise"]:
                noise = noise - noise.mean(dim=2, keepdim=True)
            time1 = time.time()

            # HuggingFace: add noise to x_0
            # print(f"device of normalized_pointcloud_b3n: {normalized_pointcloud_b3n.device}, noise device: {noise.device}, t device: {t.device}")
            x_t = ddpm_scheduler.add_noise(normalized_pointcloud_b3n, noise, t)
            time2 = time.time()

            # 2. Get model prediction
            model_output = model(x_t, t, **model_kwargs)
            pred_epsilon = model_output[:, :3, :]
            time3 = time.time()

            # 3. Standard MSE Loss on noise
            loss_mse = F.mse_loss(pred_epsilon, noise)
            time4 = time.time()

            # 4. GEOMETRIC RECONSTRUCTION LOSSES
            train_loss = loss_mse

            if (
                config["lambda_cd"] > 0
                or config["lambda_ot"] > 0
                or (epoch + 1) % config["plot_every"] == 0
            ):
                # Reconstruct x0 from noisy x_t and predicted noise using HuggingFace formula
                # x0 = (x_t - sqrt(1 - alpha_cumprod_t) * eps) / sqrt(alpha_cumprod_t)
                alphas_cumprod = ddpm_scheduler.alphas_cumprod.to(x_t.device)
                at = alphas_cumprod[t].reshape(-1, 1, 1)
                sqrt_at = at.sqrt()
                sqrt_one_minus_at = (1 - at).sqrt()
                pred_xstart_b3n = (x_t - sqrt_one_minus_at * pred_epsilon) / sqrt_at
                P_bn3 = pred_xstart_b3n.transpose(1, 2)
                G_bn3 = normalized_pointcloud_b3n.transpose(1, 2)

                if (epoch + 1) % config["plot_every"] == 0:
                    cd, _ = visualize_xyz_pip(
                        fname=f"train_epoch{epoch+1}_step{global_step}_pip",
                        original_xyz=G_bn3[0].detach().cpu() * data_std[:3].cpu()
                        + data_mean[:3].cpu(),
                        reconstructed_xyz=P_bn3[0].detach().cpu() * data_std[:3].cpu()
                        + data_mean[:3].cpu(),
                        title=f"Epoch {epoch+1} Step {global_step} timestep {t[0].item()} PIP Visualization (Mode {config['debug_mode']})",
                        save_dir=plot_dir,
                        plotlims=None,
                        device=config["device"],
                    )
                    writer.add_scalar("CD/train_cd_0", cd, global_step)

                if config["lambda_cd"] > 0:
                    loss_cd, _ = pt3d_chamfer_distance(P_bn3, G_bn3)
                    writer.add_scalar("Loss/train/cd", loss_cd.item(), global_step)
                    train_loss += config["lambda_cd"] * loss_cd

                if config["lambda_ot"] > 0:
                    loss_ot = OT_loss_func(P_bn3, G_bn3).mean()
                    writer.add_scalar("Loss/train/ot", loss_ot.item(), global_step)
                    train_loss += config["lambda_ot"] * loss_ot

            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            time5 = time.time()
            train_loss.backward()
            time6 = time.time()

            # --- GRADIENT CLIPPING ---
            # Prevents sudden loss jumps by capping the maximum weight update
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar("Gradients/Norm", grad_norm.item(), global_step)
            time7 = time.time()

            # --- WEIGHT & PREDICTION TRACKING ---
            with torch.no_grad():
                param_norm = sum(p.norm().item() for p in model.parameters())
                writer.add_scalar("Weights/Total_Norm", param_norm, global_step)
                writer.add_scalar(
                    "Prediction/Mean_Abs", pred_epsilon.abs().mean().item(), global_step
                )
                writer.add_scalar(
                    "Prediction/Max_Abs", pred_epsilon.abs().max().item(), global_step
                )

            time8 = time.time()
            optimizer.step()

            # Log loss and model outputs to tensorboard per batch
            global_step += 1
            writer.add_scalar("Loss/train", train_loss.item(), global_step)
            writer.add_scalar("Loss/train/mse", loss_mse.item(), global_step)
            time_record["wait"].append(time0 - time9)
            time9 = time.time()
            time_record["data_prep"].append(time1 - time0)
            time_record["noise_add"].append(time2 - time1)
            time_record["model_forward"].append(time3 - time2)
            time_record["loss_compute"].append(time4 - time3)
            time_record["zero_grad"].append(time5 - time4)
            time_record["backward"].append(time6 - time5)
            time_record["clip_grads"].append(time7 - time6)
            time_record["record_norms"].append(time8 - time7)
            time_record["other"].append(time9 - time8)
        # Log epoch-level metrics
        avg_epoch_loss = np.mean(train_losses)
        is_best = (
            avg_epoch_loss < best_loss / 2
        )  # Use a more forgiving threshold for "best" since this is a noisy metric, we just want to track significant improvements
        if is_best:
            best_loss = avg_epoch_loss
        writer.add_scalar("Loss/train_epoch_avg", avg_epoch_loss, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        lr_scheduler.step()

        if (epoch + 1) % config["gpu_log_every"] == 0:
            stats = query_gpu_stats(gpu_index=0)
            # print(f"Epoch {epoch+1} GPU Stats: {stats}")
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
                print(
                    f"New best model at epoch {epoch+1} with avg loss {avg_epoch_loss:.4e}"
                )

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Only log head and first block to avoid huge log files
                    if "head" in name or "ptv3.enc.0" in name:
                        try:
                            writer.add_histogram(
                                f"Gradients/{name}", param.grad, global_step
                            )
                        except Exception as e:
                            print(
                                f"Error logging gradient histogram for {name}: {e}",
                                param.grad,
                                param.grad.shape,
                            )

            # Generate samples and log Denoising Curve
            model.eval()
            with torch.no_grad():
                # Progressive sampling to log the CD-timestep curve
                # noise = torch.randn(B, 3, config["num_points"]).to(config["device"])
                noise = get_noise(
                    (B, 3, config["num_points"]),
                    debug_mode=config["debug_mode"],
                    fixed_noise_ref=fixed_noise_ref,
                ).to(config["device"])

                # HuggingFace sampling loop
                all_step_cds = []
                all_pred_x0_bn3 = []
                all_normalized_pred_x0_bn3 = []
                all_pred_xt1_bn3 = []
                all_normalized_pred_xt1_bn3 = []
                # Start from pure noise
                sample = noise
                ddpm_scheduler.set_timesteps(
                    config["num_inference_steps"], device=config["device"]
                )  # device here is important
                avg_cd = None
                for i, t in enumerate(ddpm_scheduler.timesteps):
                    t_batch = torch.full(
                        (B,),
                        t,
                        dtype=torch.long,
                        device=sample.device,
                    )
                    # Predict noise
                    model_output = model(sample, t_batch)
                    # Compute previous sample and predicted x0 using scheduler
                    # print("device sample device", sample.device, "model output device", model_output.device, "t_batch device", t_batch.device)
                    step_result = ddpm_scheduler.step(model_output, t_batch, sample)
                    pred_x0_b3n = step_result.pred_original_sample
                    prev_sample = step_result.prev_sample

                    normalized_pred_x0_bn3 = pred_x0_b3n.transpose(1, 2)
                    pred_x0_xyz_bn3 = (
                        normalized_pred_x0_bn3 * data_std[:3] + data_mean[:3]
                    )

                    step_cd, _ = pt3d_chamfer_distance(
                        pred_x0_xyz_bn3, pointcloud_bn3, batch_reduction=None
                    )
                    step_cd_np = step_cd.detach().cpu().numpy()
                    # print("timestep", t, "CD:", step_cd_np.shape)
                    all_step_cds.append(step_cd_np)
                    # print(f"len of all_step_cds: {len(all_step_cds)}, shape of last CD: {all_step_cds[-1].shape}")

                    all_normalized_pred_x0_bn3.append(
                        normalized_pred_x0_bn3.detach().cpu().numpy()
                    )
                    all_pred_x0_bn3.append(pred_x0_xyz_bn3.detach().cpu().numpy())

                    normalized_pred_xt1_bn3 = prev_sample.transpose(1, 2)
                    all_normalized_pred_xt1_bn3.append(
                        normalized_pred_xt1_bn3.detach().cpu().numpy()
                    )
                    pred_xt1_xyz_bn3 = (
                        normalized_pred_xt1_bn3 * data_std[:3] + data_mean[:3]
                    )
                    all_pred_xt1_bn3.append(pred_xt1_xyz_bn3.detach().cpu().numpy())

                    # #DEBUUG:
                    # alphas = ddpm_scheduler.alphas_cumprod.to(sample.device)
                    # at = alphas[t_batch].reshape(-1,1,1)          # shape match your data
                    # sqrt_at = at.sqrt()
                    # sqrt_one_minus_at = (1 - at).sqrt()
                    # manual_x0 = (sample - sqrt_one_minus_at * model_output) / sqrt_at

                    # diff = (manual_x0 - step_result.pred_original_sample).abs().max()
                    # print("max diff pred_x0:", diff.item())
                    # exit()

                    sample = prev_sample

                # Plot the CD Evolution curve
                plot_cd_timestep_curve(
                    fname=f"epoch_{epoch+1}_cd-evolution",
                    cds=all_step_cds,
                    title=f"Epoch {epoch+1} CD-Timestep Evolution"
                    + (" (Best)" if is_best else ""),
                    save_dir=plot_dir,
                    figsize=(6, 4),
                )
                # plot_xy_timestep(
                #     fname=f"epoch_{epoch+1}_xy-timestep",
                #     gt=pointcloud_bn3.detach().cpu().numpy(),
                #     coords=all_pred_x0_bn3,
                #     title=f"Epoch {epoch+1} XY-Timestep Evolution" + (" (Best)" if is_best else ""),
                #     save_dir=plot_dir, figsize=(3, 2)
                # )
                # plot_xy_timestep(
                #         fname=f"epoch_{epoch+1}_xy-timestep-normalized",
                #         gt=normalized_pointcloud_bn3.detach().cpu().numpy(),
                #         coords=all_normalized_pred_x0_bn3,
                #         title=f"Epoch {epoch+1} Normalized XY-Timestep Evolution" + (" (Best)" if is_best else ""),
                #         save_dir=plot_dir, figsize=(3, 2)
                #     )

                plot_multi_xy_timestep(
                    fname=f"epoch_{epoch+1}_xy-timestep-normalized-triadic",
                    gt=normalized_pointcloud_bn3.detach().cpu().numpy(),
                    coords={
                        "pred_x0": all_normalized_pred_x0_bn3,
                        "pred_xt1": all_normalized_pred_xt1_bn3,
                    },
                    title=f"Epoch {epoch+1} Normalized XY-Timestep Evolution"
                    + (" (Best)" if is_best else ""),
                    save_dir=plot_dir,
                    figsize=(3, 2),
                )

                # Log distribution of predicted noise separately for X, Y, Z
                pred_output = model(
                    normalized_pointcloud_b3n,
                    torch.tensor([t], device=config["device"]).long(),
                    **model_kwargs,
                )
                try:
                    writer.add_histogram(
                        "Distribution/Pred_Noise_X", pred_output[:, 0, :], epoch + 1
                    )
                    writer.add_histogram(
                        "Distribution/Pred_Noise_Y", pred_output[:, 1, :], epoch + 1
                    )
                    writer.add_histogram(
                        "Distribution/Pred_Noise_Z", pred_output[:, 2, :], epoch + 1
                    )
                except Exception as e:
                    print(
                        f"Error logging predicted noise histogram: {e}",
                        pred_output.shape,
                        "min",
                        min(pred_output[:, 0, :]),
                        "max",
                        max(pred_output[:, 0, :]),
                    )

                # Transpose back to (B, N, 3) and denormalize
                generated = sample.transpose(1, 2).contiguous()
                generated = generated * data_std[:3] + data_mean[:3]
                cds = []
                for i in range(min(4, B)):
                    minax = torch.amin(pointcloud_bn3[i], dim=0).cpu().numpy()
                    maxax = torch.amax(pointcloud_bn3[i], dim=0).cpu().numpy()
                    minax_pred = torch.amin(generated[i], dim=0).cpu().numpy()
                    maxax_pred = torch.amax(generated[i], dim=0).cpu().numpy()
                    minax = np.minimum(minax, minax_pred)
                    maxax = np.maximum(maxax, maxax_pred)
                    rangeax = maxax - minax
                    padding = rangeax * 0.1
                    minax = minax - padding
                    maxax = maxax + padding
                    cd, path = visualize_xyz_pip(
                        fname=f"epoch_{epoch+1}_sample_{i+1}",
                        original_xyz=pointcloud_bn3[i],
                        reconstructed_xyz=generated[i],
                        title=f"Epoch {epoch+1} Sample {i+1}"
                        + (" Best" if is_best else ""),
                        save_dir=plot_dir,
                        device=config["device"],
                        plotlims={
                            "x": (minax[0], maxax[0]),
                            "y": (minax[1], maxax[1]),
                            "z": (minax[2], maxax[2]),
                        },
                    )
                    cds.append(cd)
                avg_cd = np.mean(cds)

                writer.add_scalar("Sample/avg_cd", avg_cd, epoch)

                writer.add_histogram(
                    "Distribution/Final_Sample_X", generated[:, :, 0], epoch + 1
                )
                writer.add_histogram(
                    "Distribution/Final_Sample_Y", generated[:, :, 1], epoch + 1
                )
                writer.add_histogram(
                    "Distribution/Final_Sample_Z", generated[:, :, 2], epoch + 1
                )
            model.train()

        sum_meanm_time = sum(np.mean(v) for v in time_record.values())
        # print(f"time record for epoch {time_record}, total {sum_meanm_time:.2f} seconds")
        time_str = ";".join(f"{k[:1]}:{np.mean(v)/sum_meanm_time:.2f}" for k, v in time_record.items())
        for k, v in time_record.items():
            writer.add_scalar(f"Time/{k}", np.mean(v), epoch)
            
        epoch_bar.set_description(f"Loss: {np.mean(train_losses):.4e};time;{time_str}")

        # CHECKPOINTING
        if (epoch + 1) % config["save_every"] == 0 or is_best:
            save_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "config": config,
                "best_loss": best_loss,
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
            "best_loss": best_loss,
        },
        checkpoint_path,
    )
    print(f"Training completed. Final model saved at {checkpoint_path}.")


def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    tb_key = f"ptv3_ddpm2"
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
