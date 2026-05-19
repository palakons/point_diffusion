import torch
import torch.nn as nn
import torch.nn.functional as F
import math,os
from tqdm import trange


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
def make_beta_schedule(T=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, alpha_bars):
    """
    x0: [B, N, 3]
    t:  [B]
    """
    noise = torch.randn_like(x0)

    a_bar = alpha_bars[t].view(-1, 1, 1)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    return x_t, noise

def make_pc(num_points=64,shape="wedge"):
    #a quarter orange wedge
    if shape=="wedge":
        theta = torch.linspace(0, math.pi / 2, num_points)
        x = torch.cos(theta)
        y = torch.sin(theta)
        z = torch.zeros_like(x)
        pc = torch.stack([x, y, z], dim=-1)
        return pc
    if shape=="spiral":
        theta = torch.linspace(0, 4 * math.pi, num_points)
        z = torch.linspace(-1, 1, num_points)
        x = torch.cos(theta) * (z + 1)
        y = torch.sin(theta) * (z + 1)
        pc = torch.stack([x, y, z], dim=-1)
        return pc
    if shape =="boxsides":
        points_per_side = num_points // 6
        remainder = num_points % 6

        sides = []
        for i in range(6):
            count = points_per_side + (1 if i < remainder else 0)
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

        pc = torch.cat(sides, dim=0)[:num_points]
        return pc
    if shape == "oval":
        theta = torch.linspace(0, 2 * math.pi, num_points)
        x = 1.5 * torch.cos(theta)
        y = torch.sin(theta)
        z = torch.zeros_like(x)
        pc = torch.stack([x, y, z], dim=-1)
        return pc
    if shape == "metaball":
        theta = torch.linspace(0, 2 * math.pi, num_points)
        phi = torch.linspace(0, math.pi, num_points)
        theta_grid, phi_grid = torch.meshgrid(theta, phi)
        r = 1 + 0.5 * torch.sin(3 * theta_grid) * torch.sin(2 * phi_grid)
        x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
        y = r * torch.sin(phi_grid) * torch.sin(theta_grid)
        z = r * torch.cos(phi_grid)
        pc = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        print(pc.shape)
        return pc[:num_points]
    if shape == "undulatingcircle":
        theta = torch.linspace(0, 2 * math.pi, num_points)
        r = 1 + 0.2 * torch.sin(5 * theta)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = 0.2 * torch.sin(3 * theta)
        pc = torch.stack([x, y, z], dim=-1)
        return pc
def plot_pc(pc, gt, title="Point Cloud",fname="pc.png",azm=45):
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
    ax.view_init(elev=30, azim=azm)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

@torch.no_grad()
def p_sample_loop(model, shape, T, betas, alphas, alpha_bars, device="cuda"):
    prev_mode = model.training
    model.eval()
    B, N, D = shape
    x = torch.randn(shape, device=device)
    
    for t_step in reversed(range(T)):
        t_tensor = torch.full((B,), t_step, device=device, dtype=torch.long)
        
        eps_pred = model(x, t_tensor)
        
        alpha_t = alphas[t_step]
        alpha_bar_t = alpha_bars[t_step]
        beta_t = betas[t_step]
        
        if t_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred)
        x = x + torch.sqrt(beta_t) * noise
    
    if prev_mode:
        model.train()
    return x
if __name__ == "__main__":

    # Example setup
    device = "cuda"
    N = 128
    T = 100
    shape_name = "undulatingcircle"
    epoch = 10000

    x0_single = make_pc(num_points=N, shape=shape_name).to(device)  # [N, 3]
    x0_single = x0_single - x0_single.mean(dim=0, keepdim=True)
    x0_single = x0_single / x0_single.abs().max()

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

    model = PointNetLikeDenoiser(
        hidden_channels=64,
        time_embed_dim=32,
        num_blocks=3,
    ).to(device)

    # model = PTv3Dnsr


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    _, _, alpha_bars = make_beta_schedule(T=T, device=device)
    tt=trange(epoch)
    for step in tt:
        model.train()
        B = 128

        x0 = x0_single[None, :, :].repeat(B, 1, 1)   # [B, N, 3]
        t = torch.randint(0, T, (B,), device=device)

        x_t, noise = q_sample(x0, t, alpha_bars)

        pred = model(x_t, t)

        loss = F.mse_loss(pred, noise)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % (epoch//200) == 0:
            tt.set_description(f"Loss: {loss.item():.4f}")
            model.eval()
            pred_x = p_sample_loop(model, (1, N, 3), T, *make_beta_schedule(T=T, device=device))
            
            plot_pc(pred_x[0], gt=x0[0], title=f"Denoised Point Cloud at step {step} loss: {loss.item():.4f} N: {N} shape: {shape_name}", fname=f"/home/palakons/point_diffusion/output/denoised_{step:06d}.png",azm = step/epoch*360)
    os.system(f"ls -v /home/palakons/point_diffusion/output/denoised_*.png | xargs cat | ffmpeg -y -framerate 10 -f image2pipe -i - /home/palakons/point_diffusion/output/training_nopid_PointNetLikeDenoiser_{shape_name}_{N}_{epoch}.gif")
    os.system("rm /home/palakons/point_diffusion/output/denoised_*.png") 