import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math,os
from tqdm import tqdm, trange


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
        g = self.global_mlp(h)                            # (B, hidden, N)
        g = torch.max(g, dim=2, keepdim=True)[0]          # (B, hidden, 1)
        g_rep = g.expand(-1, -1, num_points)              # (B, hidden, N)
        
        # Inject global descriptor into local points
        h = torch.cat([h, g_rep], dim=1)                  # (B, hidden * 2, N)
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
        
        B, N,C = x.shape
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
        t_expanded = t_emb[:, None, :].repeat(1, N, 1) #

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
            -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
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

        t_emb = self.time_emb(t)                 # [B, time_dim]
        t_emb_point = t_emb[:, None, :].expand(B, N, -1)

        features = [x_t, t_emb_point]

        if self.use_point_id:
            ids = torch.arange(N, device=x_t.device)
            id_emb = self.point_id_emb(ids)      # [N, point_id_dim]
            id_emb = id_emb[None, :, :].expand(B, N, -1)
            features.append(id_emb)
        else:
            id_emb = None

        h = torch.cat(features, dim=-1)          # [B, N, in_dim]
        h = self.point_encoder(h)                # [B, N, hidden_dim]

        global_h = h.max(dim=1).values           # [B, hidden_dim]
        global_h = global_h[:, None, :].expand(B, N, -1)

        dec_features = [h, global_h, t_emb_point]

        if self.use_point_id:
            dec_features.append(id_emb)

        h_dec = torch.cat(dec_features, dim=-1)
        eps_pred = self.decoder(h_dec)

        return eps_pred

from diffusers import DDPMScheduler

def make_man_pc(num_points=64,n_scene=1,device="cpu",is_dense=False):
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
            scene_ids=[],
            data_file="man-mini",
            device=device,
            wan_vae=True,
            wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
            n_p=num_points,
            normalize_type="minmax",
            get_camera=False,
            keep_frames=n_scene,
                point_preset="original",x_range=[0,50], y_range=[-50, 50], z_range=[-2, 2],
                wan_preprocess_dir="/data/palakons/man_wan_preprocessed"
        )
        x0sbn3 = torch.stack([ds[i]['filtered_radar_data'] for i in range(n_scene)], dim=0).to(device) # [B, N, 3]
        wan_cond = torch.stack([ds[i]['wan_vae_latent'] for i in range(n_scene)], dim=0).to(device) # [B, latent_dim]
        return x0sbn3, wan_cond,ds
    
    else:
        ds = [MANDataset(
            scene_ids=[i],
            data_file="man-mini",
            device=device,
            wan_vae=True,
            wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
            n_p=num_points,
            normalize_type="minmax",
            get_camera=False,
            keep_frames=1,
            point_preset="original",x_range=[0,50], y_range=[-50, 50], z_range=[-2, 2],
            wan_preprocess_dir="/data/palakons/man_wan_preprocessed"
        ) for i in range(n_scene)]
        combined_ds = torch.utils.data.ConcatDataset(ds)
        x0sbn3 = torch.stack([data[0]['filtered_radar_data'] for data in ds], dim=0).to(device) # [B, N, 3]
        wan_cond = torch.stack([data[0]['wan_vae_latent'] for data in ds], dim=0).to(device) # [B, latent_dim] 

        # print(f"shapes x0sbn3: {x0sbn3.shape}, wan_cond: {wan_cond.shape}") #shapes x0sbn3: torch.Size([B, 128, 3]), wan_cond: torch.Size([B, 16, 2, 60, 104])
        return x0sbn3, wan_cond, combined_ds
def make_various_pc(num_points=64, device="cpu",n_shapes=7):
    theta = torch.linspace(0, math.pi / 2, num_points)
    x = torch.cos(theta)
    y = torch.sin(theta)
    z = torch.zeros_like(x)
    shape_wedge = torch.stack([x, y, z], dim=-1) #wedge

    theta = torch.linspace(0, 4 * math.pi, num_points)
    z = torch.linspace(-1, 1, num_points)
    x = torch.cos(theta) * (z + 1)
    y = torch.sin(theta) * (z + 1)
    shape_spiral = torch.stack([x, y, z], dim=-1) #spiral
    

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
    shape_metaball = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)[:num_points   ]
    
    
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
        point_preset="original",x_range=[0,50], y_range=[-50, 50], z_range=[-2, 2],
    )
    data = dataset[0]
    shape_man = data["filtered_radar_data"]

    data = torch.stack([shape_spiral, shape_undulatingcircle,  shape_oval, shape_metaball,shape_wedge, shape_boxside, shape_man], dim=0).to(device)
    
    print("Created various shapes point cloud with shape: ", data.shape,'bt will be used only first ',n_shapes,' shapes and ',num_points,' points per shape')
    #normalize each shape, subrtact mean, devide by max distance from mean
    data = data - data.mean(dim=[1], keepdim=True)
    # print("center : ", data.mean(dim=[1], keepdim=True)) # shape 
    data = data / data.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return data[:n_shapes, :num_points, :]

    
def plot_pc(pc, gt, title="Point Cloud",fname="pc.png",azm=45, progress=None, elev=30):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pc = pc.cpu().numpy()
    gt = gt.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='blue', label='Noisy',marker='o' )
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color='red', label='Ground Truth',marker='^' )
    ax.legend()
    #square aspect ratio
    ax.set_box_aspect([1, 1, 1])
    #set limits
    lim = 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    #label
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azm)
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # [left, bottom, width, height]
        cax.barh(0, progress, color='dodgerblue')
        cax.set_xlim(0, 1)
        cax.axis('off')
    plt.savefig(fname)
    plt.close()

def plot_pc_batch(pc, gt, title="Point Cloud Batch", fname="pc_batch.png", azm=45, progress=None, elev=30, max_cols=4,batch_titles=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pc = pc.cpu().numpy()
    if gt is not None:
        gt = gt.cpu().numpy()
    batch_size = pc.shape[0]
    n_cols = min(max_cols, batch_size)
    n_rows = int(math.ceil(batch_size / n_cols))
    #make sure 3d plot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False, subplot_kw={'projection': '3d'})

    for idx in range(batch_size):
        ax = axs[idx // n_cols, idx % n_cols]
        ax.scatter(pc[idx, :, 0], pc[idx, :, 1], pc[idx, :, 2], color='blue', label='Noisy', marker='o')
        if gt is not None:
            ax.scatter(gt[idx, :, 0], gt[idx, :, 1], gt[idx, :, 2], color='red', label='Ground Truth', marker='^')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.set_zlim(-1., 1.)

        if gt is not None:
            cd = pt3d_chamfer_distance(torch.from_numpy(pc[idx:idx+1]), torch.from_numpy(gt[idx:idx+1]))[0]
            if batch_titles and len(batch_titles) == batch_size:
                ax.set_title(f"{batch_titles[idx]} CD: {cd.item():.1e}", fontsize=10)
            else:
                ax.set_title(f"CD: {cd.item():.1e}", fontsize=10)
        ax.view_init(elev=elev, azim=azm)

    for idx in range(batch_size, n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis('off')

    import textwrap
    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    fig.suptitle(wrapped_title, fontsize=14)
    
    if batch_size > 0:
        axs[0, 0].legend()
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        axs[0, 0].set_zlabel('Z')
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # [left, bottom, width, height]
        cax.barh(0, progress, color='dodgerblue')
        cax.set_xlim(0, 1)
        cax.axis('off')
    plt.savefig(fname)
    plt.close()

@torch.no_grad()
def p_sample_loop(model, shape, scheduler, num_inference_steps=None, device="cuda", condition=None, model_name=""):
    prev_mode = model.training
    model.eval()
    if model_name == "PTv3Dnsr":
        B, C, N = shape
        x = torch.randn(shape, device=device)
    else:
        B, N, D = shape
        x = torch.randn(shape, device=device)
    
    steps = num_inference_steps if num_inference_steps is not None else scheduler.config.num_train_timesteps
    scheduler.set_timesteps(steps, device=device)
    for t_step in tqdm(scheduler.timesteps, desc="Sampling",leave=False)    :
        t_tensor = torch.full((B,), t_step.item(), device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor, condition=condition)
        x = scheduler.step(eps_pred, t_step, x).prev_sample
    
    if prev_mode:
        model.train()
    return x
def azm_easing(step, total_steps,style="cosine"):
    # Ease in and out from 0 to 360 degrees
    progress = step / total_steps
    if style=="linear":
        eased = progress
    elif style=="cosine":
        eased = 0.5 - 0.5 * math.cos(progress * math.pi)  # Cosine easing
    elif style=="quadratic":
        eased = 2 * progress**2 if progress < 0.5 else 1 - 2 * (1 - progress)**2  # Quadratic easing
    elif style=="exponential":
        eased = 0.5 * (2 ** (10 * (progress - 1))) if progress < 0.5 else 1 - 0.5 * (2 ** (-10 * progress))  # Exponential easing
    else:
        raise ValueError(f"Unknown easing style: {style}")
    return eased * 360

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_path,config):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved at step {step}: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cuda"):
    """Load training checkpoint and return the step to resume from.
    checkpoint_path: wildcard path to checkpoint file, e.g., "checkpoints/latest*.pt". The function will load the most recent checkpoint matching the pattern.
    """
    matched_files = [f for f in os.listdir(os.path.dirname(checkpoint_path)) if f.startswith(os.path.basename(checkpoint_path).split('*')[0]) and f.endswith(os.path.basename(checkpoint_path).split('*')[1])]
    print(f"Looking for checkpoints in {os.path.dirname(checkpoint_path)} matching {os.path.basename(checkpoint_path)}. Found: {matched_files}")
    latest_step=-1
    checkpoint = None
    for match in matched_files:
        _checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), match)
        loaded_checkpoint = torch.load(_checkpoint_path, map_location=device)
        print(f"Found checkpoint file: {match}, at step {loaded_checkpoint['step']}")
        if loaded_checkpoint['step'] > latest_step:
            latest_step = loaded_checkpoint['step']
            checkpoint = loaded_checkpoint
    if checkpoint is None:
        print(f"No valid checkpoint found in matched files: {matched_files}")
        return 0,{}

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    # scheduler.load_state_dict(checkpoint['scheduler_state'])
    step = checkpoint['step']
    config = checkpoint.get("config", {})
    print(f"Checkpoint loaded from: {checkpoint_path} (resuming from step {step})")
    return step,config
def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train a point cloud diffusion model on a single shape.")
    parser.add_argument("--N", type=int, default=128, help="Number of points in the point cloud")
    parser.add_argument("--B", type=int, default=128, help="Batch size for training")
    parser.add_argument("--n_scene", type=int, default=1, help="Number of scenes to use from the dataset (for multi-scene datasets)")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps during training")
    parser.add_argument("--T_infer", type=int, default=50, help="Number of diffusion steps during inference")
    parser.add_argument("--epoch", type=int, default=10000*3, help="Number of training epochs")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the output GIF")
    parser.add_argument("--n_frames", type=int, default=200, help="Number of frames to save during training for visualization")
    parser.add_argument("--cond_mode", type=str, default="pdnorm_only", help="Time conditioning mode for the model: 'pdnorm_only', 'feat_add', 'hybrid', 'feat_concat'")
    parser.add_argument("--shape_name", type=str, default="realman", help="Shape to train on: 'realman' or 'various'")
    parser.add_argument("--mode", type=str, default="train", help="Whether to run training or just inference test. Options: 'train', 'test'")
    parser.add_argument("--cond_method", type=str, default="scene_id", help="Conditioning method for multi-scene training: 'scene_id' (simple learnable embedding), 'wan' (use Wan's VAE latent)")
    args = parser.parse_args()
    
    return args
if __name__ == "__main__":
    args = argparse()
    # Example setup
    device = "cuda"
    shape_name = args.shape_name

    T = args.T
    T_infer = args.T_infer
    epoch = args.epoch
    fps = args.fps
    n_frames = args.n_frames
    N = args.N
    B = args.B
    n_scene = args.n_scene + (1  if args.mode == "eval" else 0)  # add 1 to n_scene during eval to have a separate unseen scene for testing
    cond_mode = args.cond_mode
    cond_method = args.cond_method  
    scene_embed_dim = 0 if args.cond_mode == "none" else (1 if cond_method == "scene_id" else 199680)

    if cond_mode  =='wan' and not shape_name.startswith("realman"):
        raise ValueError(f"cond_mode 'wan' is only compatible with shape_name 'realman' since it relies on Wan's VAE latent. Got shape_name: {shape_name}")

    if shape_name == "various":
        x0sbn3 = make_various_pc(num_points=N, device=device,n_shapes=n_scene)  # [B,N, 3]
    elif shape_name == "realman":
        x0sbn3,wan_cond,dataset = make_man_pc(num_points=N, n_scene=n_scene,device=device)  # [B,N, 3]
        x0sbn3 = x0sbn3.to(device)
        wan_cond = wan_cond.to(device)
    elif shape_name == "realman_dense":
        x0sbn3,wan_cond,dataset = make_man_pc(num_points=N, n_scene=n_scene,is_dense=True,device=device)  # [B,N, 3]
        x0sbn3 = x0sbn3.to(device)
        wan_cond = wan_cond.to(device)
    else:
        raise ValueError(f"Unknown shape_name: {shape_name}")
    # print(f"Loaded point cloud shape: {x0sbn3.shape}, device: {x0sbn3.device}, dtype: {x0sbn3.dtype}")
    x0sbn3_norm = x0sbn3 - x0sbn3.mean(dim=[0,1], keepdim=True)
    x0sbn3_norm = x0sbn3_norm / x0sbn3_norm.abs().max()

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
    from unet_diffuser import MLPDenoiser, SimplePointUNet, PointNetLikeDenoiser, PTv3Dnsr

    from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance
    model_name = "PTv3Dnsr"
    model = PTv3Dnsr(
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
        time_conditioning_mode="pdnorm_only" if cond_mode=="none" else cond_mode,  # How to inject time: "pdnorm_only", "feat_add", "hybrid", "feat_concat" (hybrid = PDNorm + feature addition)
        use_cpe=True,
        use_head=True,
        scene_embed_dim=scene_embed_dim,  # Enable scene conditioning with scalar ID
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = DDPMScheduler(
        num_train_timesteps=T, 
        beta_start=1e-4, 
        beta_end=0.02, 
        beta_schedule="linear", 
        clip_sample=False
    )
    
    run_id = f"{model_name}_{shape_name}_{N}_e{epoch}_T{T}_Inf{T_infer}_b{B}_sc{args.n_scene}_mode{cond_mode}_cond{cond_method}"
    temp_dir = f"/home/palakons/point_diffusion/output/temp_{run_id}"
    checkpoint_dir = f"/home/palakons/point_diffusion/output/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_{run_id}.pt")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load checkpoint if exists
    start_step, config = load_checkpoint(model, optimizer, scheduler, re.sub(r"e\d+", "e*", checkpoint_path), device=device)
    # start_step, config = 0,{}
    if config:
        assert config.get("N", N) == N, f"Checkpoint N {config.get('N')} does not match current N {N}"
        assert config.get("T", T) == T, f"Checkpoint T {config.get('T')} does not match current T {T}"
        assert config.get("T_infer", T_infer) == T_infer, f"Checkpoint T_infer {config.get('T_infer')} does not match current T_infer {T_infer}"
        assert config.get("B", B) == B, f"Checkpoint B {config.get('B')} does not match current B {B}"  
        # assert config.get("cond_method", cond_method) == cond_method, f"Checkpoint cond_method {config.get('cond_method')} does not match current cond_method {cond_method}"

    print(f"mode: {args.mode}")

    if args.mode == "interpolate":
        assert shape_name == "various", f"Interpolation test is designed for 'various' shape_name to show effect of conditioning. Got shape_name: {shape_name}"
        '''
        interpolate, "various" 10 steps, from -1 to 1 condition, show effect on generated samples
        '''
        print(f"Running interpolation test... model at step: {start_step}, config: {config}")
        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        shapes =  (1, 3, N) if model_name == "PTv3Dnsr" else (1, N, 3)
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        B=1
        expanded_cond = torch.ones((B, 1), device=device)  # Same condition for both
        output = {}
        n_interpolate=16
        with torch.no_grad():
            lambs = torch.linspace(-1, -.333333, n_interpolate)
            for lamb in lambs:
                expanded_cond[0] = lamb
                print(f"[Interpolation Test] Generating sample with condition: {expanded_cond.cpu().numpy().flatten()}")
                pred_x = p_sample_loop(model, shapes, scheduler, num_inference_steps=T_infer, device=device, condition=expanded_cond, model_name=model_name)
                if model_name == "PTv3Dnsr":
                    pred_x = pred_x.transpose(1, 2)
                output[lamb.item()] = pred_x
                

        pred = torch.stack([output[lamb.item()] for lamb in lambs], dim=0).squeeze(1)  # [n_interpolate, N, 3]
        print(f"Generated interpolated samples with shape: {pred.shape}")
        btitles = [f"cond:{lamb:.2f}" for lamb in lambs]
        plot_pc_batch(pred, None, title=f"Interpolation Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/../sample/interpolation_test.png",azm = 45, elev=30, batch_titles=btitles)

    elif args.mode == "eval":
        assert shape_name.startswith("realman"), f"Evaluation test is designed for 'realman' shape_name with multiple scenes to show effect of conditioning. Got shape_name: {shape_name}"
        assert cond_method  =="wan" , f"Evaluation test is designed for 'wan' conditioning method to use Wan's VAE latent for conditioning. Got cond_method: {cond_method}"
        '''get 1 more than n_scene samples, 
        sampel the model with each condition, plot will show CD, the frist n_scene samples should have low CD, the last one should have high CD if the model is learning the conditioning correctly
        '''
        print(f"Running evaluation test... model at step: {start_step}, config: {config}")
        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")
        B = n_scene  #already added +1 to n_scene if mode is eval, so B will be equal to n_scene here
        shapes =  (B, 3, N) if model_name == "PTv3Dnsr" else (B, N, 3)
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True) 


        expanded_cond = wan_cond.view(n_scene, -1)/ wan_cond.abs().max() # [B, 1]

        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)
        
        with torch.no_grad():
            print(f"[Evaluation Test] Generating sample: shape: {shapes}, condition: {expanded_cond}")
            pred_x = p_sample_loop(model, shapes, scheduler, num_inference_steps=T_infer, device=device, condition=expanded_cond, model_name=model_name)
            if model_name == "PTv3Dnsr":
                pred_x = pred_x.transpose(1, 2)
            plot_pc_batch(pred_x, gt=x0sbn3_norm, title=f"Evaluation Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/../sample/evaluation_test.png",azm = 45, elev=30)   
            print(f"Evaluation test completed. Sample saved at: {temp_dir}/../sample/evaluation_test.png")

    elif args.mode == "test_cond_infer":
        print(f"Running inference test... model at step: {start_step}, config: {config}")

        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        B=2
        cond = torch.ones((B, 1), device=device)  # Same condition for both
        for i in range(40):
            if i>=20:
                cond[1] = -1.0  # Change condition for the second sample to see the effect
            t = torch.randint(0, T, (1,), device=device).repeat(B)
            x_t = torch.randn((1, N, 3), device=device).repeat(B, 1, 1)
            diff_xt = (x_t[0] - x_t[1]).abs().mean()

            if model_name=="PTv3Dnsr":
                pred = model(x_t.transpose(1, 2), t, condition = cond ).transpose(1, 2)
            else:
                pred = model(x_t, t, condition=cond)

            diff = (pred[0] - pred[1]).abs().mean()
            print(f"  Iter {i}: T={t.cpu().numpy().flatten()}, cond={cond.cpu().numpy().flatten()}: diff b4 condition: {diff_xt.item():.4f}, diff: {diff.item():.4f}")
        
    elif args.mode == "sample":


        n_sampling = 12
        shapes =  (n_sampling, 3, N) if model_name == "PTv3Dnsr" else (n_sampling, N, 3)
        picked_indices =  torch.randint(0, n_scene, (n_sampling,), device=device)
        print(f"Picked scene indices for sampling: {picked_indices.shape}")

        scene_id_condition =picked_indices.float() / max(n_scene - 1, 1)  # [n_sampling], normalized to [0,1]
        scene_id_condition = scene_id_condition*2 -1
        expanded_cond = scene_id_condition.unsqueeze(-1)  # [n_sampling, 1]
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)
        with torch.no_grad():
            print(f"[Inference Test] Generating sample: shape: {shapes}, condition: {expanded_cond}")
            pred_x = p_sample_loop(model, shapes, scheduler, num_inference_steps=T_infer, device=device, condition=expanded_cond, model_name=model_name)
            if model_name == "PTv3Dnsr":
                pred_x = pred_x.transpose(1, 2)
            plot_pc_batch(pred_x, gt=x0sbn3_norm[picked_indices], title=f"Inference Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/../sample/inference_test.png",azm = 45, elev=30, batch_titles=[f"cond:{i:.2f}" for i in scene_id_condition.cpu().numpy()])   
            print(f"Inference test completed. Sample saved at: {temp_dir}/../sample/inference_test.png")

            
    elif args.mode == "train": 

        cd = None
        #repeat time
        rep_time = (B // n_scene)+1
        assert x0sbn3_norm.shape[0] == n_scene, f"Expected x0sbn3_norm shape[0] to match n_scene {n_scene}, got {x0sbn3_norm.shape}"
        print(f"Batch size: {B}, Number of scenes: {n_scene}, Replication factor for scenes: {rep_time} 0sbn3_norm.shape: {x0sbn3_norm.shape}")
        x0sbn3_norm_rep = x0sbn3_norm.repeat(rep_time, 1, 1)   # [rep_time, N, 3]
        print(f"Replicated x0sbn3_norm shape: {x0sbn3_norm_rep.shape}, expected: ({B}, {N}, 3)")

        if args.cond_mode=="none":
            scene_condition = torch.zeros((n_scene,1), device=device).float()  
        else:
            if cond_method == "wan":
                scene_condition = wan_cond.view(n_scene, -1)/ wan_cond.abs().max() 
                
            elif cond_method == "scene_id":
                scene_condition = torch.arange(n_scene, device=device).float() / max(n_scene - 1, 1)  # [n_scene], normalized to [0,1]
                scene_condition = (scene_condition*2 -1).unsqueeze(-1)

        print(f"Initial scene_condition shape: {scene_condition.shape}")
        scene_condition_rep = scene_condition.repeat(rep_time, 1)  # [rep_time,] repeat the scene_condition for each sample in the batch
        print(f"Replicated scene_id_condition shape: {scene_condition_rep.shape}")
        time0 = time.time()
        tt = tqdm(range(start_step, epoch), total=epoch, initial=start_step, desc="Training", unit="step")
        for step in tt:
            time1 = time.time()
            model.train()
            time2 = time.time()
            # Randomly shuffle the order of scenes in the batch for each step
            idx = torch.randperm(x0sbn3_norm_rep.shape[0], device=device)
            x0 = x0sbn3_norm_rep[idx][:B]  # [B, N, 3]
            cond = scene_condition_rep[idx][:B]  # [B, 1] scene conditioning (batch-wise)

            # dataset = torch.utils.data.TensorDataset(x0sbn3_norm_rep, scene_condition_rep)
            # dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)
            # x0, cond = next(iter(dataloader))  # Get the first batch of shuffled data

            # assert x0.shape == x0_old.shape, f"Expected x0 shape {x0_old.shape}, got {x0.shape}"
            # assert cond.shape == cond_old.shape, f"Expected cond shape {cond_old.shape}, got {cond.shape}"
            # print("shapes okay: ", x0.shape, cond.shape)#            shapes okay:  torch.Size([128, 96, 3]) torch.Size([128, 199680])
            # print("old shapes for reference: ", x0_old.shape, cond_old.shape) #old shapes for reference:  torch.Size([128, 96, 3]) torch.Size([128, 199680])


            # print(f"Step {step}: x0 shape: {x0.shape}, cond shape: {cond.shape}")
            # assert x0.shape == (B, N, 3), f"Expected x0 shape {(B, N, 3)}, got {x0.shape}"
            # assert cond.shape == (B, 1), f"Expected cond shape {(B, 1)}, got {cond.shape}"
            
            t = torch.randint(0, T, (B,), device=device)
            # print(f"shape of x0: {x0.shape}, t: {t.shape}, cond: {cond.shape}")
            noise = torch.randn_like(x0)
            time3 = time.time()
            x_t = scheduler.add_noise(x0, noise, t)

            time4 = time.time()

            if model_name=="PTv3Dnsr":
                pred = model(x_t.transpose(1, 2), t, condition = cond ).transpose(1, 2)
            else:
                # Other models expect condition: (B, scene_embed_dim)
                pred = model(x_t, t, condition=cond)
            time5 = time.time()
            loss = F.mse_loss(pred, noise)
            time6 = time.time()
            optimizer.zero_grad()
            loss.backward()
            time7 = time.time()
            optimizer.step()


            time8 = time.time()
            # Save checkpoint every N steps
            if step % max(1, epoch // 100) == 0:
                save_checkpoint(model, optimizer, scheduler, step, checkpoint_path,vars(args))

            if step % (epoch//n_frames) == 0:
                model.eval()
                try:
                    with torch.no_grad():
                        if n_scene<B:
                            shapes =  (n_scene, 3, N) if model_name == "PTv3Dnsr" else (n_scene, N, 3)
                            expanded_cond = scene_condition  # [n_scene, 1]
                        else:
                            shapes =  (B, 3, N) if model_name == "PTv3Dnsr" else (B, N, 3)
                            expanded_cond = scene_condition[:B] 
                        # print(f"[Inference] Generating sample at step {step}: shape: {shapes}, condition: {expanded_cond}")
                        pred_x = p_sample_loop(model, shapes, scheduler, num_inference_steps=T_infer, device=device, condition=expanded_cond, model_name=model_name)
                    
                    if model_name == "PTv3Dnsr":
                        pred_x = pred_x.transpose(1, 2)
                    
                    # plot_pc(pred_x[0], gt=x0[0], title=f"{shape_name} {model_name} step{step} noiseL:{loss.item():.1e} CD:{cd.item():.1e} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/denoised_{step:06d}.png",azm = azm_easing(step, epoch, style="cosine"), progress=step/epoch,elev=azm_easing(step, epoch, style="linear")/360*90)
                    plot_pc_batch(pred_x, gt=x0sbn3_norm, title=f"step{step} noiseL:{loss.item():.1e} N:{N} T:{T} Inf:{T_infer} B:{B}", fname=f"{temp_dir}/denoised_{step:06d}.png",azm = azm_easing(step, epoch, style="cosine"), progress=step/epoch,elev=azm_easing(step, epoch, style="cosine")/360*90, )#batch_titles=[f"cond:{i:.2f}" for i in scene_condition.cpu().numpy()])
                except Exception as e:
                    print(f"Error during inference/plotting at step {step}: {e}")
        
            time9 = time.time()
            time_dict={
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
        save_checkpoint(model, optimizer, scheduler, start_step + epoch, checkpoint_path,vars(args))
        os.system(f"ls -v {temp_dir}/denoised_*.png | xargs cat | ffmpeg -y -framerate {fps} -f image2pipe -i - /home/palakons/point_diffusion/output/cond_repeat_{run_id}.gif")
        os.system(f"rm {temp_dir}/denoised_*.png")
        os.system(f"rm -r {temp_dir}")