import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os
from tqdm import tqdm, trange
import numpy as np



class FiLM(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, dim * 2),
        )
    def forward(self, h: torch.Tensor, context: torch.Tensor):
        """
        h:       [B, N, D]
        context: [B, C]
        """
        gamma, beta = self.to_gamma_beta(context).chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        return h * (1.0 + gamma) + beta

class FullSetTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.film1 = FiLM(dim, context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.film2 = FiLM(dim, context_dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout)
    def forward(self, h: torch.Tensor, context: torch.Tensor):
        """
        h:       [B, N, D]
        context: [B, C]
        """
        # Attention residual branch
        a = self.norm1(h)
        a = self.film1(a, context)
        a, _ = self.attn(a, a, a, need_weights=False)
        h = h + self.drop1(a)
        # MLP residual branch
        m = self.norm2(h)
        m = self.film2(m, context)
        m = self.mlp(m)
        h = h + self.drop2(m)
        return h

class FullSetTransformerDenoiser(nn.Module):
    """
    Global non-serialized set-attention DDPM denoiser.
    Input:
        x:         [B, N, 3]
        t:         [B]
        condition: [B, 16, 2, 60, 104] or already flattened [B, C]
    Output:
        eps_hat or x0_hat: [B, N, 3]
    """
    def __init__(
        self,
        dim: int = 256,
        time_dim: int = 256,
        wan_shape=(16, 2, 60, 104),
        wan_hidden: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        out_channels: int = 3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        wan_dim = math.prod(wan_shape)
        self.wan_mlp = nn.Sequential(
            nn.Linear(wan_dim, wan_hidden),
            nn.SiLU(),
            nn.Linear(wan_hidden, dim),
            nn.LayerNorm(dim),
        )
        self.xyz_proj = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                FullSetTransformerBlock(
                    dim=dim,
                    context_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_channels),
        )
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition=None, **kwargs):
        """
        x: [B, N, 3]
        """
        B, N, C = x.shape
        # assert C == 3, f"Expected x [B,N,3], got {x.shape}"
        # xyz as the point feature
        h = self.xyz_proj(x)
        # timestep context
        t_context = self.time_mlp(self.time_emb(t.to(x.device)))
        # WAN condition context
        if condition is not None:
            wan = condition.to(x.device).view(B, -1)
            wan_context = self.wan_mlp(wan)
            context = t_context + wan_context
        else:
            context = t_context
        for block in self.blocks:
            h = block(h, context)
        out = self.out(self.out_norm(h))
        return out


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
                    nn.GroupNorm(
                        8 if hidden_channels >= 8 else 1,
                        hidden_channels * 2 if i == 0 else hidden_channels,
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        hidden_channels * 2 if i == 0 else hidden_channels,
                        hidden_channels,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(8 if hidden_channels >= 8 else 1, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                )
                for i in range(num_blocks)
            ]
        )
        self.skip_projs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_channels * 2 if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                )
                for i in range(num_blocks)
            ]
        )
        self.out = nn.Conv1d(hidden_channels, 3, kernel_size=1)

        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.constant_(self.stem.bias, 0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x, t=None, condition=None, **kwargs):
        # x: (B, 3, N), t: (B,)
        x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
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
        g = self.global_mlp(h)  # (B, hidden, N)
        g = torch.max(g, dim=2, keepdim=True)[0]  # (B, hidden, 1)
        g_rep = g.expand(-1, -1, num_points)  # (B, hidden, N)

        # Inject global descriptor into local points
        h = torch.cat([h, g_rep], dim=1)  # (B, hidden * 2, N)
        # --------------------------

        for block, skip_proj in zip(self.blocks, self.skip_projs):
            residual = skip_proj(h)
            h = block(h) + residual

        return self.out(F.silu(h)).transpose(1, 2)  # (B, hidden, N) -> (B, N, 3)


class SimplePointUNet(nn.Module):
    """
    A minimal UNet-style denoiser for (B,N,3
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
        x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
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
        return out.transpose(1, 2)  # (B, out_channels, N) -> (B, N, out_channels)


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
                nn.Linear(3, coord_projector_dim),
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
            print(
                f"Initializing MLPDenoiser with scene conditioning. Scene embed dim: {scene_embed_dim}, context_channels: {context_channels}"
            )
            # input torch.Size([B, 16, 2, 60, 104]) --> after flattening spatial dims --> (B, scene_embed_dim)
            self.scene_mlp = nn.Sequential(
                nn.Linear(scene_embed_dim, context_channels),
                nn.GELU(),
                nn.Linear(context_channels, context_channels),
            )
        else:
            self.scene_mlp = None

        # per-point MLP: input = feat_dim + context_channels + (scene_context if enabled)
        cond_channels = context_channels + (
            context_channels if scene_embed_dim > 0 else 0
        )
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
        exponents = (
            -math.log(10000)
            * torch.arange(half, device=timesteps.device).float()
            / half
        )
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

        B, N, C = x.shape
        device = x.device

        # per-point features
        x_pts = x
        if self.coord_projector is not None:
            feat = self.coord_projector(x_pts)
        else:
            feat = x_pts

        # time embedding expanded per point
        t_emb = self.get_time_embedding(t.to(device))  # (B, context)
        # make B,N,context for concatenation
        t_expanded = t_emb[:, None, :].repeat(1, N, 1)  #

        # scene conditioning (optional)
        if self.scene_mlp is not None and condition is not None:
            # condition: (B, scene_embed_dim), need to flatten dim 1 ...
            cond = condition.to(device)
            s_emb = self.scene_mlp(cond)  # (B, context)
            s_expanded = s_emb.unsqueeze(1).repeat(1, N, 1)  # (B, N, context)
            cond = torch.cat([t_expanded, s_expanded], dim=-1)
            # print(f"[MsLPDenoiser] Using scene conditioning. Scene embed shape: {condition.shape}, processed scene embed shape: {s_emb.shape}, expanded scene embed shape: {s_expanded.shape}, cond shape: {cond.shape}")
        else:
            cond = t_expanded
        # print(f"[MLPDenoiser] Forward pass. x shape: {x.shape}, t shape: {t.shape}, feat shape: {feat.shape}, cond shape: {cond.shape},t_emb shape: {t_emb.shape}")
        inp = torch.cat([feat, cond], dim=-1)
        out = self.point_mlp(inp)  # (B*N, out_channels)
        out = out.view(B, N, self.out_channels)
        return out



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: [B], integer or float timestep
        return: [B, dim]
        """
        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )

        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class PointNetDenoiser(nn.Module):
    def __init__(
        self,
        num_points: int = 64,
        time_dim: int = 128,
        hidden_dim: int = 512,
        use_point_id: bool = True,
        point_id_dim: int = 64,
    ):
        super().__init__()

        self.num_points = num_points
        self.use_point_id = use_point_id

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        if use_point_id:
            self.point_id_emb = nn.Embedding(num_points, point_id_dim)
        else:
            point_id_dim = 0

        in_dim = 3 + time_dim + point_id_dim

        self.point_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + time_dim + point_id_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        x_t: [B, N, 3]
        t:   [B]
        return epsilon_pred: [B, N, 3]
        """
        B, N, _ = x_t.shape
        assert N == self.num_points, f"Expected N={self.num_points}, got N={N}"

        t_emb = self.time_emb(t)  # [B, time_dim]
        t_emb_point = t_emb[:, None, :].expand(B, N, -1)

        features = [x_t, t_emb_point]

        if self.use_point_id:
            ids = torch.arange(N, device=x_t.device)
            id_emb = self.point_id_emb(ids)  # [N, point_id_dim]
            id_emb = id_emb[None, :, :].expand(B, N, -1)
            features.append(id_emb)
        else:
            id_emb = None

        h = torch.cat(features, dim=-1)  # [B, N, in_dim]
        h = self.point_encoder(h)  # [B, N, hidden_dim]

        global_h = h.max(dim=1).values  # [B, hidden_dim]
        global_h = global_h[:, None, :].expand(B, N, -1)

        dec_features = [h, global_h, t_emb_point]

        if self.use_point_id:
            dec_features.append(id_emb)

        h_dec = torch.cat(dec_features, dim=-1)
        eps_pred = self.decoder(h_dec)

        return eps_pred


from diffusers import DDPMScheduler


def make_man_pc(
    num_points=64, n_scene=1, device="cpu", is_dense=False, data_file="man-mini"
):
    # B 128 128 pt 9.482Gi/15.992Gi
    import sys

    sys.path.append("/home/palakons/point_diffusion")
    from man_ddpm import MANDataset

    # return torch.stack([MANDataset(
    #     scene_ids=[i],
    #     data_file="man-mini",
    #     device=device,
    #     wan_vae=False,
    #     wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
    #     n_p=num_points,
    #     normalize_type="minmax",
    #     get_camera=False,
    #     keep_frames=n_scene,
    #     point_preset="original",x_range=[0,50], y_range=[-50, 50], z_range=[-2, 2],
    # )[0]['filtered_radar_data'] for i in range(n_scene)], dim=0).to(device) # [B, N, 3]
    if is_dense:
        ds = MANDataset(
            # scene_ids=list(range(450,598)),
            scene_ids=[],
            data_file=data_file,
            device=device,
            wan_vae=True,
            wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
            n_p=num_points,
            normalize_type="minmax",
            get_camera=False,
            keep_frames=n_scene,
            point_preset="original",
            x_range=[0, 50],
            y_range=[-50, 50],
            z_range=[-2, 2],
            wan_preprocess_dir="/data/palakons/man_wan_preprocessed",
            coord_only=False
        )

        # print(f"keys in man dataset item: {ds[0].keys()}")  # keys
        npoints_originals =[ds[i]['npoints_original'] for i in range(n_scene)]
        npoints_after_distance_filter = [ds[i]['npoints_after_distance_filter'] for i in range(n_scene)]
        npoints_filtereds = [ds[i]['npoints_filtered'] for i in range(n_scene)]
        # print(f"n point npoints_original {ds[0]['npoints_original']}, npoints_after_distance_filter: {ds[0]['npoints_after_distance_filter']}, npoints_filtered: {ds[0]['npoints_filtered']}")  #n point npoints_original 800, npoints_after_distance_filter: 185, npoints_filtered: 135
        #print min max mean
        print(f"npoints_originals: min {min(npoints_originals)}, max {max(npoints_originals)}, mean {sum(npoints_originals)/len(npoints_originals)}")
        print(f"npoints_after_distance_filter: min {min(npoints_after_distance_filter)}, max {max(npoints_after_distance_filter)}, mean {sum(npoints_after_distance_filter)/len(npoints_after_distance_filter)}")
        print(f"npoints_filtereds: min {min(npoints_filtereds)}, max {max(npoints_filtereds)}, mean {sum(npoints_filtereds)/len(npoints_filtereds)}")
        # npoints_originals: min 173, max 800, mean 644.32
        # npoints_after_distance_filter: min 81, max 273, mean 179.62
        # npoints_filtereds: min 58, max 221, mean 136.66
        # exit()
        x0sbn3 = torch.stack(
            [ds[i]["filtered_radar_data"] for i in range(n_scene)], dim=0
        ).to(
            device
        )  # [B, N, 3]
        wan_cond = torch.stack(
            [ds[i]["wan_vae_latent"] for i in range(n_scene)], dim=0
        ).to(
            device
        )  # [B, latent_dim]
        return x0sbn3[:,:,:3], wan_cond, ds, x0sbn3[:,:,3:6],x0sbn3[:,:,6:]

    else:
        ds = [
            MANDataset(
                scene_ids=[i],
                data_file=data_file,
                device=device,
                wan_vae=True,
                wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
                n_p=num_points,
                normalize_type="minmax",
                get_camera=False,
                keep_frames=1,
                point_preset="original",
                x_range=[0, 50],
                y_range=[-50, 50],
                z_range=[-2, 2],
                wan_preprocess_dir="/data/palakons/man_wan_preprocessed",
            coord_only=False
            )
            for i in range(n_scene)
        ]
        combined_ds = torch.utils.data.ConcatDataset(ds)
        x0sbn3 = torch.stack([data[0]["filtered_radar_data"] for data in ds], dim=0).to(
            device
        )  # [B, N, 3]
        wan_cond = torch.stack([data[0]["wan_vae_latent"] for data in ds], dim=0).to(
            device
        )  # [B, latent_dim]
        # print(f"shapes x0sbn3: {x0sbn3.shape}, wan_cond: {wan_cond.shape}") #shapes x0sbn3: torch.Size([B, 128, 3]), wan_cond: torch.Size([B, 16, 2, 60, 104])
        return x0sbn3[:,:,:3], wan_cond, combined_ds, x0sbn3[:,:,3:6],x0sbn3[:,:,6:]


def make_various_pc(num_points=64, device="cpu", n_shapes=7):
    theta = torch.linspace(0, math.pi / 2, num_points)
    x = torch.cos(theta)
    y = torch.sin(theta)
    z = torch.zeros_like(x)
    shape_wedge = torch.stack([x, y, z], dim=-1)  # wedge

    theta = torch.linspace(0, 4 * math.pi, num_points)
    z = torch.linspace(-1, 1, num_points)
    x = torch.cos(theta) * (z + 1)
    y = torch.sin(theta) * (z + 1)
    shape_spiral = torch.stack([x, y, z], dim=-1)  # spiral

    points_per_side = num_points // 6
    remainder = num_points % 6
    if remainder > 0:
        points_per_side += 1  # Distribute extra points to the first few sides

    sides = []
    for i in range(6):
        count = points_per_side
        if i == 0:  # Front
            x = torch.linspace(-1, 1, count)
            y = torch.linspace(-1, 1, count)
            z = torch.ones_like(x)
        elif i == 1:  # Back
            x = torch.linspace(-1, 1, count)
            y = torch.linspace(-1, 1, count)
            z = -torch.ones_like(x)
        elif i == 2:  # Left
            x = -torch.ones_like(x)
            y = torch.linspace(-1, 1, count)
            z = torch.linspace(-1, 1, count)
        elif i == 3:  # Right
            x = torch.ones_like(x)
            y = torch.linspace(-1, 1, count)
            z = torch.linspace(-1, 1, count)
        elif i == 4:  # Top
            x = torch.linspace(-1, 1, count)
            y = torch.ones_like(x)
            z = torch.linspace(-1, 1, count)
        else:  # Bottom
            x = torch.linspace(-1, 1, count)
            y = -torch.ones_like(x)
            z = torch.linspace(-1, 1, count)

        side_points = torch.stack([x, y, z], dim=-1)
        sides.append(side_points)

    shape_boxside = torch.cat(sides, dim=0)[:num_points]

    theta = torch.linspace(0, 2 * math.pi, num_points)
    x = 1.5 * torch.cos(theta)
    y = torch.sin(theta)
    z = torch.zeros_like(x)
    shape_oval = torch.stack([x, y, z], dim=-1)

    theta = torch.linspace(0, 2 * math.pi, num_points)
    phi = torch.linspace(0, math.pi, num_points)
    theta_grid, phi_grid = torch.meshgrid(theta, phi)
    r = 1 + 0.5 * torch.sin(3 * theta_grid) * torch.sin(2 * phi_grid)
    x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = r * torch.sin(phi_grid) * torch.sin(theta_grid)
    z = r * torch.cos(phi_grid)
    shape_metaball = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)[
        :num_points
    ]

    theta = torch.linspace(0, 2 * math.pi, num_points)
    r = 1 + 0.2 * torch.sin(5 * theta)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = 0.2 * torch.sin(3 * theta)
    shape_undulatingcircle = torch.stack([x, y, z], dim=-1)

    # B 128 128 pt 9.482Gi/15.992Gi
    import sys

    sys.path.append("/home/palakons/point_diffusion")
    from man_ddpm import MANDataset

    dataset = MANDataset(
        scene_ids=[0],
        data_file="man-mini",
        device=device,
        wan_vae=False,
        wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
        n_p=num_points,
        normalize_type="minmax",
        get_camera=False,
        keep_frames=1,
        point_preset="original",
        x_range=[0, 50],
        y_range=[-50, 50],
        z_range=[-2, 2],
    )
    data = dataset[0]
    shape_man = data["filtered_radar_data"]


    torch.manual_seed(42)
    random_shape = torch.rand(num_points, 3) * 2 - 1  # random shape in [-1,1]

    data = torch.stack(
        [
            shape_spiral,
            shape_undulatingcircle,
            shape_oval,
            shape_metaball,
            shape_wedge,
            random_shape,
            shape_boxside,
            shape_man,
        ],
        dim=0,
    ).to(device)
    print(f"old shape before adding extra features: {data.shape}") # should be [7, num_points, 3]
    #attached 2 2 tot he last dim, random data, to test conditioning on extra features
    torch.manual_seed(42) #this ensure the random features are the same across runs for consistency
    data = torch.cat([data, torch.rand_like(data[:,:,:2])], dim=-1)
    print(
        "Created various shapes point cloud with shape: ",
        data.shape,
        "bt will be used only first ",
        n_shapes,
        " shapes and ",
        num_points,
        " points per shape",
    )
    if False:  #will be normed outside
        # normalize each shape, subrtact mean, devide by max distance from mean
        data = data - data.mean(dim=[1], keepdim=True)
        # print("center : ", data.mean(dim=[1], keepdim=True)) # shape
        data = data / data.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return data[:n_shapes, :num_points, :]


def plot_pc(
    pc, gt, title="Point Cloud", fname="pc.png", azm=45, progress=None, elev=30
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pc = pc.cpu().numpy()
    gt = gt.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color="blue", label="Noisy", marker="o")
    ax.scatter(
        gt[:, 0], gt[:, 1], gt[:, 2], color="red", label="Ground Truth", marker="^"
    )
    ax.legend()
    # square aspect ratio
    ax.set_box_aspect([1, 1, 1])
    # set limits
    lim = 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # label
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azm)
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cax.barh(0, progress, color="dodgerblue")
        cax.set_xlim(0, 1)
        cax.axis("off")
    plt.savefig(fname)
    plt.close()


def plot_pc_batch(
    pc,
    gt,
    title="Point Cloud Batch",
    fname="pc_batch.png",
    azm=45,
    progress=None,
    elev=30,
    max_cols=4,
    batch_titles=None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    pc = pc.cpu().numpy()
    if gt is not None:
        gt = gt.cpu().numpy()
    batch_size = pc.shape[0]
    has_attr = pc.shape[-1] >= 5 and gt is not None and gt.shape[-1] >= 5
    n_cols = min(max_cols, batch_size)
    n_rows = int(math.ceil(batch_size / n_cols))
    # make sure 3d plot
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )

    for idx in range(batch_size):
        ax = axs[idx // n_cols, idx % n_cols]
        ax.scatter(
            pc[idx, :, 0],
            pc[idx, :, 1],
            pc[idx, :, 2],
            color="blue",
            label="Noisy",
            marker="o",
        )
        if gt is not None:
            ax.scatter(
                gt[idx, :, 0],
                gt[idx, :, 1],
                gt[idx, :, 2],
                color="red",
                label="Ground Truth",
                marker="^",
            )
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)

        cd_string = ""
        if gt is not None:
            if has_attr:

                # i have to compute the gt-pc correlation, pc_idx, wihci point in pc corresponds to which point in gt,, by find the NN between pc and gt, 
                # then from this paring, identify corresponding doppler and rcs values, compute error, plot histogram of error, and compute mse loss for doppler and rcs, and add to title

                pc_idx = np.argmin(
                    np.linalg.norm(
                        pc[idx, :, :3][:, None, :] - gt[idx, :, :3][None, :, :], axis=-1
                    ),
                    axis=1,
                )

                doppler_err = pc[idx, :, 3] - gt[idx, pc_idx, 3]
                rcs_err = pc[idx, :, 4] - gt[idx, pc_idx, 4]
                xyz_err = np.linalg.norm(pc[idx, :, :3] - gt[idx, pc_idx, :3], axis=-1)
                position_loss = np.mean(xyz_err**2)
                
                minmax_doppler = [-1, 1]
                minmax_rcs = [-1, 1]
                bins_doppler = np.linspace(minmax_doppler[0], minmax_doppler[1], 21, endpoint=True)
                bins_rcs = np.linspace(minmax_rcs[0], minmax_rcs[1], 21, endpoint=True)

                axins_l = inset_axes(ax, width="38%", height="20%", loc="lower left", borderpad=1)
                # min max [-.5, .5] for doppler,  
                axins_l.hist(doppler_err,  #bins=bins_doppler, 
                             alpha=0.75, color="tab:orange", log=True)
                axins_l.axvline(0.0, color="black", linewidth=1)
                axins_l.tick_params(axis='both', which='major', labelsize=7)
                #set ticklabel_format
                axins_l.ticklabel_format(axis='x', style='plain')
                # axins_l.set_xticks([])
                # axins_l.set_yticks([])
                axins_l.set_title(f"doppler L: {doppler_err.mean():.1e}", fontsize=7)
                # axins_l.set_title(f"doppler [{minmax_doppler[0]}, {minmax_doppler[1]}] L: {doppler_loss.item():.1e}", fontsize=7)

                axins_r = inset_axes(ax, width="38%", height="20%", loc="lower right", borderpad=1)
                #[-1, 1] for rcs,
                axins_r.hist(rcs_err, #bins=bins_rcs,
                              alpha=0.75, color="tab:green",  log=True)
                axins_r.axvline(0.0, color="black", linewidth=1)
                #axis ticks alebls should ahve size 7 font, and in normal number nor "r" format
                axins_r.tick_params(axis='both', which='major', labelsize=7)
                axins_r.ticklabel_format(axis='x', style='plain')
                # axins_r.set_xticks([])
                # axins_r.set_yticks([])
                # axins_r.set_title(f"rcs [{minmax_rcs[0]}, {minmax_rcs[1]}] L: {rcs_loss.item():.1e}", fontsize=7)
                axins_r.set_title(f"rcs L: {rcs_err.mean():.1e}", fontsize=7)

                #add top-left inset, plot 2d scatter of doppler vs rcs, with limits [-0.5, 0.5] for doppler and [-1, 1] for rcs, one set for pred, another for gt
                axins_tl = inset_axes(ax, width="38%", height="20%", loc="upper left", borderpad=1)
                axins_tl.scatter(pc[idx, :, 3], pc[idx, :, 4], alpha=0.75, color="tab:blue", label="Noisy", marker="o", s=10)
                axins_tl.scatter(gt[idx, :, 3], gt[idx, :, 4], alpha=0.75, color="tab:red", label="GT", marker="^", s=10)


                # axins_tl.set_xlim(minmax_doppler)
                # axins_tl.set_ylim(minmax_rcs)
                # axins_tl.set_xticks([])
                # axins_tl.set_yticks([])
                #set x y label
                axins_tl.set_xlabel("Doppler", fontsize=6)
                axins_tl.set_ylabel("RCS", fontsize=6)
                axins_tl.set_title("Doppler vs RCS", fontsize=7)
                # axins_tl.legend(fontsize=6)


                # rmse_doppler = np.sqrt(np.mean(doppler_err**2))
                # rmse_rcs = np.sqrt(np.mean(rcs_err**2))
                cd = pt3d_chamfer_distance(
                    torch.from_numpy(pc[idx : idx + 1, :, :3]), torch.from_numpy(gt[idx : idx + 1, :, :3])
                )[0]
                cd_string = f"CD:{cd.item():.1e} L:{position_loss.item():.1e}"  
            else:
                cd = pt3d_chamfer_distance(
                    torch.from_numpy(pc[idx : idx + 1]), torch.from_numpy(gt[idx : idx + 1])
                )[0]
                cd_string = f"CD: {cd.item():.1e}"
        if batch_titles and len(batch_titles) == batch_size:
            ax.set_title(f"{batch_titles[idx]} {cd_string}", fontsize=10)
        else:
            ax.set_title(cd_string, fontsize=10)
        ax.view_init(elev=elev, azim=azm)

    for idx in range(batch_size, n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis("off")

    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    fig.suptitle(wrapped_title, fontsize=14)

    if batch_size > 0:
        axs[0, 0].legend()
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        axs[0, 0].set_zlabel("Z")
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cax.barh(0, progress, color="dodgerblue")
        cax.set_xlim(0, 1)
        cax.axis("off")
    plt.savefig(fname)
    plt.close()


@torch.no_grad()
def p_sample_loop(
    model,
    shape,
    scheduler,
    num_inference_steps=None,
    device="cuda",
    condition=None,
    model_name="",
):
    prev_mode = model.training
    model.eval()
    if model_name == "PTv3Dnsr":
        B, C, N = shape
        x = torch.randn(shape, device=device)
    else:
        B, N, D = shape
        x = torch.randn(shape, device=device)

    steps = (
        num_inference_steps
        if num_inference_steps is not None
        else scheduler.config.num_train_timesteps
    )
    scheduler.set_timesteps(steps, device=device)
    for t_step in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        t_tensor = torch.full((B,), t_step.item(), device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor, condition=condition)
        x = scheduler.step(eps_pred, t_step, x).prev_sample

    if prev_mode:
        model.train()
    return x


def azm_easing(step, total_steps, style="cosine"):
    # Ease in and out from 0 to 360 degrees
    progress = step / total_steps
    if style == "linear":
        eased = progress
    elif style == "cosine":
        eased = 0.5 - 0.5 * math.cos(progress * math.pi)  # Cosine easing
    elif style == "quadratic":
        eased = (
            2 * progress**2 if progress < 0.5 else 1 - 2 * (1 - progress) ** 2
        )  # Quadratic easing
    elif style == "exponential":
        eased = (
            0.5 * (2 ** (10 * (progress - 1)))
            if progress < 0.5
            else 1 - 0.5 * (2 ** (-10 * progress))
        )  # Exponential easing
    else:
        raise ValueError(f"Unknown easing style: {style}")
    return eased * 360


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_path, config):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved at step {step}: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, epoch, device="cuda"):
    """Load training checkpoint and return the step to resume from.
    checkpoint_path: wildcard path to checkpoint file, e.g., "checkpoints/latest*.pt". The function will load the most recent checkpoint matching the pattern.
    """
    matched_files = [
        f
        for f in os.listdir(os.path.dirname(checkpoint_path))
        if f.startswith(os.path.basename(checkpoint_path).split("*")[0])
        and f.endswith(os.path.basename(checkpoint_path).split("*")[1])
    ]
    print(
        f"Looking for checkpoints in {os.path.dirname(checkpoint_path)}/{os.path.basename(checkpoint_path)}. Found: {matched_files}"
    )
    latest_step = -1
    checkpoint = None
    for match in matched_files:
        _checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), match)
        try:
            loaded_checkpoint = torch.load(_checkpoint_path, map_location=device)
            print(f"Found checkpoint file: {match}, at step {loaded_checkpoint['step']}")
            if loaded_checkpoint["step"] > latest_step:
                if epoch <loaded_checkpoint["step"]:
                    print(
                        f"Checkpoint step {loaded_checkpoint['step']} is greater than current epoch {epoch}. Skipping this checkpoint in file {match}."
                    )
                    continue
                latest_step = loaded_checkpoint["step"]
                checkpoint = loaded_checkpoint
        except Exception as e:
            print(f"Error loading checkpoint {_checkpoint_path}: {e}")
            continue
    if checkpoint is None:
        print(f"No valid checkpoint found in matched files: {matched_files}")
        return 0, {}

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    # scheduler.load_state_dict(checkpoint['scheduler_state'])
    step = checkpoint["step"]
    config = checkpoint.get("config", {})
    print(f"Checkpoint loaded from: {checkpoint_path} (resuming from step {step})")
    return step, config


def argparse():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a point cloud diffusion model on a single shape."
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of points in the point cloud"
    )
    parser.add_argument("--B", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--n_scene",
        type=int,
        default=1,
        help="Number of scenes to use from the dataset (for multi-scene datasets)",
    )
    parser.add_argument(
        "--T", type=int, default=1000, help="Number of diffusion steps during training"
    )
    parser.add_argument(
        "--T_infer",
        type=int,
        default=50,
        help="Number of diffusion steps during inference",
    )
    parser.add_argument(
        "--epoch", type=int, default=10000 * 3, help="Number of training epochs"
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for the output GIF"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=200,
        help="Number of frames to save during training for visualization",
    )
    parser.add_argument(
        "--cond_mode",
        type=str,
        default="pdnorm_only",
        help="Time conditioning mode for the model: 'pdnorm_only', 'feat_add', 'hybrid', 'feat_concat'",
    )
    parser.add_argument(
        "--shape_name",
        type=str,
        default="realman",
        help="Shape to train on: 'realman' or 'various'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Whether to run training or just inference test. Options: 'train', 'test'",
    )
    parser.add_argument(
        "--cond_method",
        type=str,
        default="scene_id",
        help="Conditioning method for multi-scene training: 'scene_id' (simple learnable embedding), 'wan' (use Wan's VAE latent)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="man-mini",
        help="Data file to use for MANDataset when shape_name is 'realman'. Options: 'man-mini', 'man-full', or path to custom data file compatible with MANDataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PTv3Dnsr",
        help="Model architecture to use: 'SetTxDnsr', 'PTv3Dnsr'",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save and load checkpoints",
    )
    parser.add_argument(
        "--train_rcs_doppler",
        action="store_true",
        help="Whether to train on RCS and Doppler features",
    )
    
    parser.add_argument(
        "--loss_weight_position",
        type=float,
        default=1.0,
        help="Loss weight for position feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_doppler",
        type=float,
        default=1.0,
        help="Loss weight for Doppler feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_rcs",
        type=float,
        default=1.0,
        help="Loss weight for RCS feature, only used if --train_rcs_doppler is set",
    )
    args = parser.parse_args()

    return args

def normalize_data(x, mean=None, max_half_range=None,save_filename_title=None):
    '''
    x: [B, N, D]
    mean: optional precomputed mean to use for centering, 
    max_half_range: optional precomputed max_half_range to use for scaling,
    '''
    is_train = mean is None 
    if mean is None:
        mean = x.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
    x_centered = x - mean
    if max_half_range is None:
        max_half_range = x_centered.abs().max()  # [B, 1, 1]
    x_normalized = x_centered / max_half_range

    if save_filename_title is not None:
        fname,title = save_filename_title
        #plot log historgam of the original data and the normalized data for each dimension
        import matplotlib.pyplot as plt
        #2 col for before and after normalization, and D rows for each dimension
        fig, axs = plt.subplots( x.shape[2], 2, figsize=(8, 4 * x.shape[2]))
        if x.shape[2] == 1:
            axs = axs[None, :]
        for d in range(x.shape[2]):
            n,bins,patch = axs[d, 0].hist(x[:,:,d].cpu().numpy().flatten(), bins=50, log=True, color="tab:blue", alpha=0.75)
            axs[d, 0].set_title(f"Original data - dim {d}")
            n,bins,patch = axs[d, 1].hist(x_normalized[:,:,d].cpu().numpy().flatten(), bins=50, log=True, color="tab:orange", alpha=0.75)
            # print(f"Dimension {d}: n {n}, bins {bins}")
            axs[d, 1].set_title(f"{title} - dim {d} {'train' if is_train else 'eval'}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    return x_normalized, mean, max_half_range

if __name__ == "__main__":
    args = argparse()
    # Example setup
    device = "cuda"
    shape_name = args.shape_name
    data_file = args.data_file
    T = args.T
    T_infer = args.T_infer
    epoch = args.epoch
    fps = args.fps
    n_frames = args.n_frames
    N = args.N
    B = args.B
    n_scene = args.n_scene 
    cond_mode = args.cond_mode
    cond_method = args.cond_method
    scene_embed_dim = (
        0 if args.cond_mode == "none" else (1 if cond_method == "scene_id" else 199680)
    )
    model_name =args.model_name
    inout_dim= 5 if args.train_rcs_doppler else 3
    print(
        f"Using conditioning mode: {cond_mode}, conditioning method: {cond_method}, scene_embed_dim: {scene_embed_dim}"
    )


    loss_weights = {"doppler": args.loss_weight_doppler if args.train_rcs_doppler else None, "rcs": args.loss_weight_rcs if args.train_rcs_doppler else None, "position": args.loss_weight_position}
    print(f"Using loss weights: {loss_weights}")

    if cond_mode == "wan" and not shape_name.startswith("realman"):
        raise ValueError(
            f"cond_mode 'wan' is only compatible with shape_name 'realman' since it relies on Wan's VAE latent. Got shape_name: {shape_name}"
        )

    if shape_name == "various":
        total_sc = int(1.25 * n_scene)
        assert total_sc <= 8, f"n_scene is too large for 'various' shape_name. "
        x0sbn3 = make_various_pc(
            num_points=N, device=device, n_shapes=total_sc
        )  # [B,N, 3]
        x0sbn3 = x0sbn3.cpu()  # Move to CPU for preprocessing to save GPU memory

        doppler = x0sbn3[:,:,3:4]  # Use the 4th dimension as dummy doppler for loss calculation
        rcs = x0sbn3[:,:,4:5]  # Use the 5th dimension as dummy rcs for loss calculation
        x0sbn3 = x0sbn3[:,:,:3]  # Use only the first 3 dimensions as point cloud data

        x0sbn3_eval = x0sbn3[n_scene : ]
        x0sbn3 = x0sbn3[:n_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation
        doppler_eval = doppler[n_scene : ]
        doppler = doppler[:n_scene]
        rcs_eval = rcs[n_scene : ]
        rcs = rcs[:n_scene]
    elif shape_name == "realman":
        total_sc = int(1.25 * n_scene)
        x0sbn3, wan_cond, dataset,doppler,rcs = make_man_pc(
            num_points=N, n_scene=total_sc, device=device, data_file=data_file
        )  # [B,N, 3]
        x0sbn3, wan_cond, dataset,doppler,rcs = x0sbn3.cpu(), wan_cond.cpu(), dataset, doppler.cpu(), rcs.cpu()  # Move to CPU for preprocessing to save GPU memory
        x0sbn3_eval = x0sbn3[n_scene : ]
        x0sbn3 = x0sbn3[:n_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation
        
        wan_cond_eval = wan_cond[n_scene : ]
        wan_cond = wan_cond[:n_scene]
        doppler_eval = doppler[n_scene : ]
        doppler = doppler[:n_scene]
        rcs_eval = rcs[n_scene : ]
        rcs = rcs[:n_scene]
    elif shape_name == "realman_dense":
        total_sc = int(1.25 * n_scene)
        x0sbn3, wan_cond, dataset,doppler,rcs = make_man_pc(
            num_points=N,
            n_scene=total_sc,
            is_dense=True,
            device=device,
            data_file=data_file,
        )  # [B,N, 3]
        x0sbn3, wan_cond, dataset,doppler,rcs = x0sbn3.cpu(), wan_cond.cpu(), dataset, doppler.cpu(), rcs.cpu()  # Move to CPU for preprocessing to save GPU memory
        x0sbn3_eval = x0sbn3[n_scene : ]
        x0sbn3 = x0sbn3[:n_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation
        wan_cond_eval = wan_cond[n_scene : ]
        wan_cond = wan_cond[:n_scene]
        doppler_eval = doppler[n_scene : ]
        doppler = doppler[:n_scene]
        rcs_eval = rcs[n_scene : ]
        rcs = rcs[:n_scene]
    else:
        raise ValueError(f"Unknown shape_name: {shape_name}")
    # print(f"Loaded point cloud shape: {x0sbn3.shape}, device: {x0sbn3.device}, dtype: {x0sbn3.dtype}")

    x0sbn3_norm,meanv,max_half_range = normalize_data(x0sbn3,save_filename_title=['/home/palakons/point_diffusion/output/sample/x0sbn3_normalization.png', "x0sbn3"])
    #norm each dim separatedly
    # meanx = x0sbn3.mean(dim=[0, 1], keepdim=True)
    # x0sbn3_norm = x0sbn3 - meanx
    # maxrange = x0sbn3_norm.abs().max()
    # x0sbn3_norm = x0sbn3_norm / maxrange


    x0sbn3_eval_norm,_,_ = normalize_data(x0sbn3_eval, mean=meanv, max_half_range=max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/x0sbn3_eval_normalization.png', "x0sbn3_eval"])

    # x0sbn3_eval_norm = x0sbn3_eval - meanx
    # x0sbn3_eval_norm = x0sbn3_eval_norm / maxrange

    # assert torch.allclose(x0sbn3_normx, x0sbn3_norm, atol=1e-5), "Normalization mismatch: x0sbn3_normx and x0sbn3_norm are not close enough"
    # assert torch.allclose(x0sbn3_eval_normx, x0sbn3_eval_norm, atol=1e-5), "Normalization mismatch: x0sbn3_eval_normx and x0sbn3_eval_norm are not close enough"


    # calculate norm 2 (rms) of doppler (vx vy vz -> v)
    doppler = torch.norm(doppler, p=2, dim=-1, keepdim=True)
    doppler_eval = torch.norm(doppler_eval, p=2, dim=-1, keepdim=True)
    # print(f"Original doppler shape: {doppler.shape}, rcs shape: {rcs.shape}") # Original doppler shape: torch.Size([B, 128, 1]), rcs shape: torch.Size([B, 128, 1])

    doppler_norm, doppler_mean, doppler_max_half_range = normalize_data(doppler, save_filename_title=['/home/palakons/point_diffusion/output/sample/doppler_normalization.png', "doppler"])
    doppler_eval_norm, _, _ = normalize_data(doppler_eval, mean=doppler_mean, max_half_range=doppler_max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/doppler_eval_normalization.png', "doppler_eval"])

    # meandoppler = doppler.mean(dim=[0, 1], keepdim=True)
    # doppler_norm = doppler - meandoppler
    # maxrange_doppler = doppler_norm.abs().max()
    # doppler_norm = doppler_norm / maxrange_doppler

    # doppler_eval_norm = doppler_eval - meandoppler
    # doppler_eval_norm = doppler_eval_norm / maxrange_doppler

    rcs_norm, rcs_mean, rcs_max_half_range = normalize_data(rcs, save_filename_title=['/home/palakons/point_diffusion/output/sample/rcs_normalization.png', "rcs"])
    rcs_eval_norm, _, _ = normalize_data(rcs_eval, mean=rcs_mean, max_half_range=rcs_max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/rcs_eval_normalization.png', "rcs_eval"])


    print(f"min max x0 after norm: {x0sbn3_norm.amin(dim=[0,1])}, {x0sbn3_norm.amax(dim=[0,1])}") # min max x0 after norm: tensor([-1., -1., -1.]), tensor([1., 1., 1.]
    print(f"min max doppler after norm: {doppler_norm.amin(dim=[0,1])}, {doppler_norm.amax(dim=[0,1])}") # min max doppler after norm: tensor([-1.]), tensor([1.])
    print(f"min max rcs after norm: {rcs_norm.amin(dim=[0,1])}, {rcs_norm.amax(dim=[0,1])}") # min max rcs after norm: tensor([-1.]), tensor([1.])

    # rcs_norm = rcs - rcs.mean(dim=[0, 1], keepdim=True)
    # rcs_norm = rcs_norm / rcs_norm.abs().max()

    if args.train_rcs_doppler:
        print(f"Training with RCS and Doppler. Original input shape: {x0sbn3.shape}, doppler shape: {doppler.shape}, rcs shape: {rcs.shape}") # [B, 128, 3], doppler shape: [B, 128, 3], rcs shape: [B, 128, 1]
        x0sbn3_norm = torch.cat([x0sbn3_norm, doppler_norm, rcs_norm], dim=-1)  # [B,N,5]
        print(f"Training with RCS and Doppler. New input shape: {x0sbn3_norm.shape}") # Training with RCS and Doppler. New input shape: torch.Size([B, 128, 5])

        x0sbn3_eval_norm = torch.cat([x0sbn3_eval_norm, doppler_eval_norm, rcs_eval_norm], dim=-1)  # [B,N,5]
        print(f"Evaluation with RCS and Doppler. New input shape: {x0sbn3_eval_norm.shape}") # Evaluation with RCS and Doppler. New input shape:


    # model = PointNetDenoiser(
    #     num_points=N,
    #     time_dim=128,
    #     hidden_dim=512,
    #     use_point_id=False,
    #     point_id_dim=64,
    # ).to(device)
    # model = MLPDenoiser(
    #     in_channels=3,
    #     out_channels=3,
    #     context_channels=128,
    #     hidden_dim=256,
    #     num_layers=4,
    #     coord_projector_dim=0,
    #     dropout=0.0,
    #     scene_embed_dim=0,
    # ).to(device)

    # model =SimplePointUNet(
    #     in_channels=3,
    #     base_channels=64,
    #     out_channels=3,
    #     num_layers=4,        t_embed_dim=32,
    # ).to(device)
    # model_name = "PointNetLikeDenoiser"
    # model = PointNetLikeDenoiser(
    #     hidden_channels=64,
    #     time_embed_dim=32,
    #     num_blocks=3,
    # ).to(device)
    import sys

    sys.path.append("/home/palakons/point_diffusion")
    from unet_diffuser import (
        MLPDenoiser,
        SimplePointUNet,
        PointNetLikeDenoiser,
        PTv3Dnsr,
    )

    from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance

    if model_name == "PTv3Dnsr":
        model = PTv3Dnsr(
            n_in_channels=inout_dim,
            context_channels=256,
            out_channels=inout_dim,
            grid_size=0.02,
            shuffle_orders=True,
            serialized_inverse=False,
            n_stages=5,  # Allow sweeping n_stages from 2 to 5
            seed=42,
            backbone_type="full",
            param_multiplier=1.0,
            project_coord_dim=0,  # Optional: project 3D coordinates to higher dim, e.g., 32, 64. If 0, use raw coords.
            time_conditioning_mode=(
                "pdnorm_only" if cond_mode == "none" else cond_mode
            ),  # How to inject time: "pdnorm_only", "feat_add", "hybrid", "feat_concat" (hybrid = PDNorm + feature addition)
            use_cpe=True,
            use_head=True,
            scene_embed_dim=scene_embed_dim,  # Enable scene conditioning with scalar ID
        ).to(device)

        dop_suffix = f"_dop{loss_weights['doppler']:.1e}" if loss_weights["doppler"] is not None and loss_weights["doppler"] !=1 else ""
        rcs_suffix = f"_rcs{loss_weights['rcs']:.1e}" if loss_weights["rcs"] is not None and loss_weights["rcs"] !=1 else ""
        position_suffix = f"_pos{loss_weights['position']:.1e}" if  loss_weights["position"] !=1 else ""

        run_id = (
            f"{model_name}_{shape_name}_{N}"
            f"{f'_dim{inout_dim}' if inout_dim == 5 else ''}"
            f"_e{epoch}_T{T}_Inf{T_infer}_b{B}_sc{args.n_scene}_mode{cond_mode}_cond{cond_method}"
            f"{position_suffix}{dop_suffix}{rcs_suffix}"
        )
        # run_id = f"{model_name}_{shape_name}_{N}{f'_dim{inout_dim}' if inout_dim ==5 else f''}_e{epoch}_T{T}_Inf{T_infer}_b{B}_sc{args.n_scene}_mode{cond_mode}_cond{cond_method}{'' if loss_weigths['doppler'] is None else f'_dop{loss_weigths["doppler"]:.1e}'}{'' if loss_weigths['rcs'] is None else f'_rcs{loss_weigths["rcs"]:.1e}'}"
    elif model_name == "SetTxDnsr":
        dim=64
        model = FullSetTransformerDenoiser(
            in_channels=inout_dim,
            dim =dim,
            time_dim=256,
            num_layers=5,
            num_heads=8,
            dropout=0.,
            out_channels=inout_dim,
            wan_shape=(16, 2, 60, 104) if cond_mode != "none" and cond_method != 'scene_id' else (1,),
        ).to(device)

        dop_suffix = f"_dop{loss_weights['doppler']:.1e}" if loss_weights["doppler"] is not None else ""
        rcs_suffix = f"_rcs{loss_weights['rcs']:.1e}" if loss_weights["rcs"] is not None else ""
        position_suffix = f"_pos{loss_weights['position']:.1e}" if  loss_weights["position"] !=1 else ""

        run_id = f"{model_name}_{shape_name}_{N}{f'_dim{inout_dim}' if inout_dim ==5 else f''}_e{epoch}_T{T}_Inf{T_infer}_b{B}_sc{args.n_scene}_cond{cond_method}_d{dim}{position_suffix}{dop_suffix}{rcs_suffix}"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
    )

    temp_dir = f"/home/palakons/point_diffusion/output/temp_{run_id}"
    checkpoint_dir = f"/home/palakons/point_diffusion/output/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_{run_id}.pt")
    os.makedirs(temp_dir, exist_ok=True)

    # Load checkpoint if exists
    start_step, config = load_checkpoint(
        model,
        optimizer,
        scheduler,
        re.sub(r"e\d+", "e*", checkpoint_path),
        epoch,
        device=device,
    )
    # start_step, config = 0,{}
    if config:
        assert (
            config.get("N", N) == N
        ), f"Checkpoint N {config.get('N')} does not match current N {N}"
        assert (
            config.get("T", T) == T
        ), f"Checkpoint T {config.get('T')} does not match current T {T}"
        assert (
            config.get("T_infer", T_infer) == T_infer
        ), f"Checkpoint T_infer {config.get('T_infer')} does not match current T_infer {T_infer}"
        assert (
            config.get("B", B) == B
        ), f"Checkpoint B {config.get('B')} does not match current B {B}"
        # assert config.get("cond_method", cond_method) == cond_method, f"Checkpoint cond_method {config.get('cond_method')} does not match current cond_method {cond_method}"

    print(f"mode: {args.mode}")

    if args.mode == "interpolate":
        assert (
            shape_name == "various"
        ), f"Interpolation test is designed for 'various' shape_name to show effect of conditioning. Got shape_name: {shape_name}"
        """
        interpolate, "various" 10 steps, from -1 to 1 condition, show effect on generated samples
        """
        print(
            f"Running interpolation test... model at step: {start_step}, config: {config}"
        )
        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        shapes = (1, inout_dim, N) if model_name == "PTv3Dnsr" else (1, N, inout_dim)
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        B = 1
        expanded_cond = torch.ones((B, 1), device=device)  # Same condition for both
        output = {}
        n_interpolate = 16
        with torch.no_grad():
            lambs = torch.linspace(-1, -0.333333, n_interpolate)
            for lamb in lambs:
                expanded_cond[0] = lamb
                print(
                    f"[Interpolation Test] Generating sample with condition: {expanded_cond.cpu().numpy().flatten()}"
                )
                pred_x = p_sample_loop(
                    model,
                    shapes,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=expanded_cond,
                    model_name=model_name,
                )
                if model_name == "PTv3Dnsr":
                    pred_x = pred_x.transpose(1, 2)
                output[lamb.item()] = pred_x

        pred = torch.stack([output[lamb.item()] for lamb in lambs], dim=0).squeeze(
            1
        )  # [n_interpolate, N, 3]
        print(f"Generated interpolated samples with shape: {pred.shape}")
        btitles = [f"cond:{lamb:.2f}" for lamb in lambs]
        plot_pc_batch(
            pred,
            None,
            title=f"Interpolation Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}",
            fname=f"{temp_dir}/../sample/interpolation_test.png",
            azm=45,
            elev=30,
            batch_titles=btitles,
        )

    elif args.mode == "eval":
        assert shape_name.startswith(
            "realman"
        ), f"Evaluation test is designed for 'realman' shape_name with multiple scenes to show effect of conditioning. Got shape_name: {shape_name}"
        assert (
            cond_method == "wan"
        ), f"Evaluation test is designed for 'wan' conditioning method to use Wan's VAE latent for conditioning. Got cond_method: {cond_method}"
        """get 1 more than n_scene samples, 
        sampel the model with each condition, plot will show CD, the frist n_scene samples should have low CD, the last one should have high CD if the model is learning the conditioning correctly
        """
        print(
            f"Running evaluation test... model at step: {start_step}, config: {config}"
        )
        model.eval()
        try:
            print(f"scene_strength {model.scene_strength.item():.4f}")
        except AttributeError:
            pass
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        expanded_cond_eval = wan_cond_eval.view(x0sbn3_eval_norm.shape[0], -1) / wan_cond.abs().max()  # divide by max abs value OF THE TRAINING SET
        expanded_cond_train = wan_cond.view(wan_cond.shape[0], -1) / wan_cond.abs().max()  # divide by max abs value OF THE TRAINING SET
        print(f"shapes wan_cond; {wan_cond.shape}") #torch.Size([320, 16, 2, 60, 104])
        

        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        with torch.no_grad():
            for name,expanded_cond,gt in zip(["Eval", "Train"],[expanded_cond_eval, expanded_cond_train],[x0sbn3_eval_norm, x0sbn3_norm]): # first eval condition, then training condition as control
                B = gt.shape[0]
                shapes = (B, inout_dim, N) if model_name == "PTv3Dnsr" else (B, N, inout_dim)
                print(
                f"[Evaluation Test] {name} Generating sample: shape: {shapes}, condition: {expanded_cond.shape} (normalized using training set max abs value {wan_cond.abs().max().item():.4f})"
            )
                pred_x = p_sample_loop(
                    model,
                    shapes,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=expanded_cond ,
                    model_name=model_name,
                )
                if model_name == "PTv3Dnsr":
                    pred_x = pred_x.transpose(1, 2)
                print(f"shapes pred_x: {pred_x.shape}, gt: {gt.shape}")
                cd = pt3d_chamfer_distance(
                    pred_x.cpu(), gt.cpu()
                )[0]  # Compute Chamfer Distance for each sample
                plot_pc_batch(
                    pred_x[:12],
                    gt=gt[:12],
                    title=f"{name} {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B} scene {gt.shape[0]} overall CD: {cd.item():.1e}",
                    fname=f"{temp_dir}/../sample/evaluation_{name}_{model_name}_{shape_name}.png",
                    azm=45,
                    elev=30,
                )
                print(
                    f"Evaluation test completed. Sample saved at: {temp_dir}/../sample/evaluation_{name}_{model_name}_{shape_name}.png"
                )

    elif args.mode == "test_cond_infer": #testing if 2 conditiong gives different output, 
        print(
            f"Running inference test... model at step: {start_step}, config: {config}"
        )

        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        B = 2
        cond = torch.ones((B, 1), device=device)  # Same condition for both
        for i in range(40):
            if i >= 20:
                cond[1] = (
                    -1.0
                )  # Change condition for the second sample to see the effect
            t = torch.randint(0, T, (1,), device=device).repeat(B)
            x_t = torch.randn((1, N, 3), device=device).repeat(B, 1, 1)
            diff_xt = (x_t[0] - x_t[1]).abs().mean()

            if model_name == "PTv3Dnsr":
                pred = model(x_t.transpose(1, 2), t, condition=cond).transpose(1, 2)
            else:
                pred = model(x_t, t, condition=cond)

            diff = (pred[0] - pred[1]).abs().mean()
            print(
                f"  Iter {i}: T={t.cpu().numpy().flatten()}, cond={cond.cpu().numpy().flatten()}: diff b4 condition: {diff_xt.item():.4f}, diff: {diff.item():.4f}"
            )

    elif args.mode == "sample":

        n_sampling = 12
        shapes = (n_sampling, inout_dim, N) if model_name == "PTv3Dnsr" else (n_sampling, N, inout_dim)
        picked_indices = torch.randint(0, n_scene, (n_sampling,), device=device)
        print(f"Picked scene indices for sampling: {picked_indices.shape}")

        scene_id_condition = picked_indices.float() / max(
            n_scene - 1, 1
        )  # [n_sampling], normalized to [0,1]
        scene_id_condition = scene_id_condition * 2 - 1
        expanded_cond = scene_id_condition.unsqueeze(-1)  # [n_sampling, 1]
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)
        with torch.no_grad():
            print(
                f"[Inference Test] Generating sample: shape: {shapes}, condition: {expanded_cond}"
            )
            pred_x = p_sample_loop(
                model,
                shapes,
                scheduler,
                num_inference_steps=T_infer,
                device=device,
                condition=expanded_cond,
                model_name=model_name,
            )
            if model_name == "PTv3Dnsr":
                pred_x = pred_x.transpose(1, 2)
            plot_pc_batch(
                pred_x,
                gt=x0sbn3_norm[picked_indices],
                title=f"Inference Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}",
                fname=f"{temp_dir}/../sample/inference_test.png",
                azm=45,
                elev=30,
                batch_titles=[
                    f"cond:{i:.2f}" for i in scene_id_condition.cpu().numpy()
                ],
            )
            print(
                f"Inference test completed. Sample saved at: {temp_dir}/../sample/inference_test.png"
            )
    elif args.mode == "permutation":
        print(
            f"Running permutation test... model at step: {start_step}, config: {config}"
        )
        '''
        cann the model 2 times, with the second time have the input to the model permuted in a different order, [B,N,3] (the "N" dimension is permuted)
        '''
        print(f"tesing {model_name} permutation invariance...")
        model.eval()
        B=1
        condition = torch.zeros_like(wan_cond[:B]) if cond_method == "wan" else torch.zeros((B, 1), device=device)
        #set seed
        torch.manual_seed(42)
        for i in range(10):
            with torch.no_grad():
                t = torch.randint(0, T, (1,), device=device).repeat(B)*0+T//2
                x_t = torch.randn((1, N, 3), device=device).repeat(B, 1, 1)
                perm = torch.randperm(x_t.shape[1], device=x_t.device)
                inv_perm = torch.argsort(perm)

                if model_name == "PTv3Dnsr":
                    x_t= x_t.transpose(1, 2) # [B, 3, N]
                    condition = condition.reshape(B, -1)  # [B, scene_embed_dim]
                    eps1 = model(x_t, t, condition).transpose(1, 2)
                    eps2_perm = model(x_t[:, :, perm], t, condition).transpose(1, 2)
                    eps2 = eps2_perm[:, inv_perm, :]    

                elif model_name == "SetTxDnsr":
                    eps1 = model(x_t, t, condition)
                    eps2_perm = model(x_t[:, perm, :], t, condition)
                    eps2 = eps2_perm[:, inv_perm, :]

                err = (eps1 - eps2).abs().mean()

                print(f"  Iter {i}: T={t.cpu().numpy().flatten()[0]}, err between original and permuted: {err.item():.4e}")
                # if err.item() <1e-5:
                #     print(f"  Permutation invariance test PASSED with error {err.item():.4e}, perm idx sample: {perm.cpu().numpy()}")
    elif args.mode == "train":

        cd = None
        # repeat time
        rep_time = (B // n_scene) + 1
        assert (
            x0sbn3_norm.shape[0] == n_scene
        ), f"Expected x0sbn3_norm shape[0] to match n_scene {n_scene}, got {x0sbn3_norm.shape}"
        print(
            f"Batch size: {B}, Number of scenes: {n_scene}, Replication factor for scenes: {rep_time} 0sbn3_norm.shape: {x0sbn3_norm.shape}"
        ) # torch.Size([40, 128, 5])          
        x0sbn3_norm_rep = x0sbn3_norm.repeat(rep_time, 1, 1)  # [rep_time, N, 3]
        print(
            f"Replicated x0sbn3_norm shape: {x0sbn3_norm_rep.shape}, expected: ({B}, {N}, 3) if not predicting doppler/rcs, ({B}, {N}, 5) if predicting doppler/rcs"
        )

        if args.cond_mode == "none":
            scene_condition = torch.zeros((n_scene, 1), device=device).float()
        else:
            if cond_method == "wan":
                scene_condition = wan_cond.view(n_scene, -1) / wan_cond.abs().max()

            elif cond_method == "scene_id":
                scene_condition = torch.arange(n_scene, device=device).float() / max(
                    n_scene - 1, 1
                )  # [n_scene], normalized to [0,1]
                scene_condition = (scene_condition * 2 - 1).unsqueeze(-1)

        print(f"Initial scene_condition shape: {scene_condition.shape}")
        scene_condition_rep = scene_condition.repeat(
            rep_time, 1
        )  # [rep_time,] repeat the scene_condition for each sample in the batch
        print(f"Replicated scene_id_condition shape: {scene_condition_rep.shape}")
        time0 = time.time()
        tt = tqdm(
            range(start_step, epoch),
            total=epoch,
            initial=start_step,
            desc="Training",
            unit="step",
        )
        for step in tt:
            time1 = time.time()
            model.train()
            time2 = time.time()
            # Randomly shuffle the order of scenes in the batch for each step
            idx = torch.randperm(x0sbn3_norm_rep.shape[0], device='cpu')
            x0 = x0sbn3_norm_rep[idx][:B].to(device)  # [B, N, 3]
            cond = scene_condition_rep[idx][
                :B
            ].to(device)  # [B, 1] scene conditioning (batch-wise)
            # print(f"shapes okay: {x0.shape}, {cond.shape}") #  torch.Size([128, 128, 5]), torch.Size([128, 1])  


            t = torch.randint(0, T, (B,), device=device)
            # print(f"shape of x0: {x0.shape}, t: {t.shape}, cond: {cond.shape}")
            noise = torch.randn_like(x0)
            time3 = time.time()
            x_t = scheduler.add_noise(x0, noise, t)

            time4 = time.time()

            if model_name == "PTv3Dnsr":
                pred = model(x_t.transpose(1, 2), t, condition=cond).transpose(1, 2)
            else:
                # Other models expect condition: (B, scene_embed_dim)
                pred = model(x_t, t, condition=cond)
            time5 = time.time()
            # loss = F.mse_loss(pred, noise)

            loss_position = F.mse_loss(pred[..., :3], noise[..., :3]) # CD? 
            if inout_dim > 3:
                loss_doppler = F.mse_loss(pred[..., 3:3+1], noise[..., 3:3+1])
                loss_rcs = F.mse_loss(pred[..., 3+1:], noise[..., 3+1:])
                loss = loss_position + loss_weights['doppler'] * loss_doppler + loss_weights['rcs'] * loss_rcs
            else:
                loss = loss_position



            time6 = time.time()
            optimizer.zero_grad()
            loss.backward()
            time7 = time.time()
            optimizer.step()

            time8 = time.time()
            # Save checkpoint every N steps
            if step % max(1, epoch // 100) == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step, checkpoint_path, vars(args)
                )

            if step % (epoch // n_frames) == 0:
                model.eval()
                try:
                    with torch.no_grad():
                        if n_scene < B:
                            shapes = (
                                (n_scene, inout_dim, N)
                                if model_name == "PTv3Dnsr"
                                else (n_scene, N, inout_dim)
                            )
                            expanded_cond = scene_condition  # [n_scene, 1]
                        else:
                            shapes = (
                                (B, inout_dim, N) if model_name == "PTv3Dnsr" else (B, N, inout_dim)
                            )
                            expanded_cond = scene_condition[:B]
                        if shapes[0] > 8:
                            shapes = (8, shapes[1], shapes[2])
                            expanded_cond = expanded_cond[:8]
                        # print(f"[Inference] Generating sample at step {step}: shape: {shapes}, condition: {expanded_cond}")
                        pred_x = p_sample_loop(
                            model,
                            shapes,
                            scheduler,
                            num_inference_steps=T_infer,
                            device=device,
                            condition=expanded_cond,
                            model_name=model_name,
                        )

                    if model_name == "PTv3Dnsr":
                        pred_x = pred_x.transpose(1, 2)

                    # plot_pc(pred_x[0], gt=x0[0], title=f"{shape_name} {model_name} step{step} noiseL:{loss.item():.1e} CD:{cd.item():.1e} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/denoised_{step:06d}.png",azm = azm_easing(step, epoch, style="cosine"), progress=step/epoch,elev=azm_easing(step, epoch, style="linear")/360*90)
                    plot_pc_batch(
                        pred_x,
                        gt=x0sbn3_norm[: shapes[0]],
                        title=f"step{step} noiseL:{loss.item():.1e} N:{N} T:{T} Inf:{T_infer} B:{B}",
                        fname=f"{temp_dir}/denoised_{step:06d}.png",
                        azm=azm_easing(step, epoch, style="cosine"),
                        progress=step / epoch,
                        elev=azm_easing(step, epoch, style="cosine") / 360 * 90,
                    )  # batch_titles=[f"cond:{i:.2f}" for i in scene_condition.cpu().numpy()])
                except Exception as e:
                    print(f"Error during inference/plotting at step {step}: {e}")

            time9 = time.time()
            time_dict = {
                "between_steps": time1 - time0,
                ".train()": time2 - time1,
                "prep_tensors": time3 - time2,
                "noise_add": time4 - time3,
                "model_forward": time5 - time4,
                "loss_compute": time6 - time5,
                "loss_backward": time7 - time6,
                "optimizer_step": time8 - time7,
                "plotting": time9 - time8,
            }
            total_time = time9 - time0
            tt.set_description(f"Loss: {loss.item():.1e}")
            # print(f"time%{[f'{k}:{v/total_time*100:.1f}%' for k,v in time_dict.items()]}")
            time0 = time.time()
        # Final checkpoint
        save_checkpoint(
            model, optimizer, scheduler, start_step + epoch, checkpoint_path, vars(args)
        )
        os.system(
            f"ls -v {temp_dir}/denoised_*.png | xargs cat | ffmpeg -y -framerate {fps} -f image2pipe -i - /home/palakons/point_diffusion/output/cond_repeat_{run_id}.gif"
        )
        os.system(f"rm {temp_dir}/denoised_*.png")
        os.system(f"rm -r {temp_dir}")
