
import sys,math

from altair import condition
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        cond_type: str = "film",
    ):
        super().__init__()
        self.cond_type = cond_type
        self.norm1 = nn.LayerNorm(dim)
        if self.cond_type in ["film", "film-xattn"]:
            print(f"make FiLM block with context_dim {context_dim}")
            self.film1 = FiLM(dim, context_dim)

        if self.cond_type in ["xattn", "film-xattn"]:
            print(f"make xattn block with context_dim {context_dim} and num_heads {num_heads}")
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        if self.cond_type in ["film", "film-xattn"]:
            self.film2 = FiLM(dim, context_dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, context: torch.Tensor, cross_context: torch.Tensor = None):
        """
        h:       [B, N, D]
        context: [B, C]
        """
        # Attention residual branch
        a = self.norm1(h)
        if self.cond_type in ["film", "film-xattn"]:
            a = self.film1(a, context)
        a, _ = self.attn(a, a, a, need_weights=False)
        h = h + self.drop1(a)

        # Cross-attention residual branch
        if self.cond_type in ["xattn", "film-xattn"] and cross_context is not None:
            c = self.norm_cross(h)
            c, _ = self.cross_attn(c, cross_context, cross_context, need_weights=False) # Q from h, K,V from cross_context
            h = h + self.drop1(c)

        # MLP residual branch
        m = self.norm2(h)
        if self.cond_type in ["film", "film-xattn"]:
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
        cond_type: str = "film",
        mlp_ratio: int = 4,
        use_condition_pooling: bool = False,
        condition_pool_kernel: int = 4,
    ):
        super().__init__()

        valid_cond_types = {"film", "xattn", "film-xattn", "none"}
        assert cond_type in valid_cond_types, f"Unknown cond_type: {cond_type}"

        self.dim = dim
        self.cond_type = cond_type
        self.mlp_ratio = mlp_ratio

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wan_shape = tuple(wan_shape)
        self.use_condition_pooling = use_condition_pooling
        self.condition_pool_kernel = condition_pool_kernel
        self.wan_shape = wan_shape

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        if self.cond_type in ["film", "film-xattn"]:
            # Backward compatibility: flatten the entire WAN latent into a single vector
            # wan_dim = math.prod(wan_shape)
            if self.use_condition_pooling and len(wan_shape) == 4:
                c, f, h, w = wan_shape
                assert h % condition_pool_kernel == 0 and w % condition_pool_kernel == 0, "WAN spatial dimensions must be divisible by pool kernel"
                pooled_h = h // condition_pool_kernel
                pooled_w = w // condition_pool_kernel
                wan_dim = c * f * pooled_h * pooled_w
            else:
                wan_dim = math.prod(wan_shape)

            self.wan_mlp_film = nn.Sequential(
                nn.Linear(wan_dim, wan_hidden),
                nn.SiLU(),
                nn.Linear(wan_hidden, dim),
                nn.LayerNorm(dim),
            )
        if self.cond_type in ["xattn", "film-xattn"]:
            # For cross-attention: process tokens from the WAN latent space.
            wan_token_dim = wan_shape[0] * wan_shape[1] if len(wan_shape) >= 2 else math.prod(wan_shape)
            self.wan_mlp_cross = nn.Sequential(
                nn.Linear(wan_token_dim, wan_hidden),
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
                    cond_type=cond_type,
                    mlp_ratio=mlp_ratio
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
    def _pool_wan_condition(self, condition: torch.Tensor, B: int) -> torch.Tensor:
        """
        condition: [B, 16, 2, 60, 104]
        returns:   [B, 32, 15, 26] if pooling enabled with k=4
                [B, 32, 60, 104] if pooling disabled
        """
        if condition.ndim != 5:
            raise ValueError(
                f"Expected WAN condition [B, C, F, H, W], got {condition.shape}"
            )

        wan = condition.reshape(
            B,
            condition.shape[1] * condition.shape[2],
            condition.shape[3],
            condition.shape[4],
        )

        if self.use_condition_pooling:
            wan = F.avg_pool2d(
                wan,
                kernel_size=self.condition_pool_kernel,
                stride=self.condition_pool_kernel,
            )

        return wan
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
        # WAN condition context #shape [16, 2, 60, 104] -> flatten to [B, wan_dim]
        context = t_context
        cross_context = None
        #if all are zeros in condition, set to None
        if condition is not None:
            if self.cond_type in ["film", "film-xattn"]:
                # wan_film = condition.to(x.device).view(B, -1)
                # wan_context_film = self.wan_mlp_film(wan_film)
                # context = t_context + wan_context_film

                cond = condition.to(device=x.device, dtype=x.dtype)
                if self.use_condition_pooling:
                    wan = self._pool_wan_condition(cond, B)  # [B, 32, 15, 26]
                    wan_film = wan.reshape(B, -1)
                else:
                    wan_film = cond.reshape(B, -1)
                # print(f"[FullSetTransformerDenoiser] WAN condition shape: {condition.shape}, processed WAN shape for FiLM: {wan_film.shape} should be [B, 12480]") #[B, 32, 15, 26] 
                wan_context_film = self.wan_mlp_film(wan_film)
                # print(f"[FullSetTransformerDenoiser] FiLM context shape: {wan_context_film.shape} should be [B, dim]") #→ [B, 12480]
                context = t_context + wan_context_film
            if self.cond_type in ["xattn", "film-xattn"]:
                # wan_cross = condition.to(x.device) # [B, 16, 2, 60, 104]
    
                # if wan_cross.ndim == 5:
                #     # Reshape WAN latent into a sequence of tokens
                #     # [B, 16, 2, 60, 104] -> [B, 60*104, 16*2]
                #     num_tokens = wan_cross.shape[3] * wan_cross.shape[4]
                #     token_dim = wan_cross.shape[1] * wan_cross.shape[2]
                #     wan_tokens = wan_cross.permute(0, 3, 4, 1, 2).reshape(B, num_tokens, token_dim)
                # else:
                #     # Fallback for non-WAN 1D/2D conditions (e.g., scene_id)
                #     wan_tokens = wan_cross.view(B, 1, -1)
                #     raise ValueError(f"Expected WAN condition with shape [B, 16, 2, 60, 104], got {condition.shape}")
                # Project tokens to the model's dimension
                # cross_context = self.wan_mlp_cross(wan_tokens) # [B, num_tokens, dim]

                cond = condition.to(device=x.device, dtype=x.dtype)

                if cond.ndim == 5:
                    wan = self._pool_wan_condition(cond, B)  # [B, 32, H', W']
                    # print(f"[FullSetTransformerDenoiser] WAN condition shape: {condition.shape}, pooled WAN shape: {wan.shape} should be [B, 32, 15, 26]") #[B, 32, 15, 26]
                    wan_tokens = wan.permute(0, 2, 3, 1).reshape(B, -1, wan.shape[1])
                else:
                    raise ValueError(
                        f"Expected WAN condition with shape [B, 16, 2, 60, 104], got {condition.shape}"
                    )
                # print(f"[FullSetTransformerDenoiser] WAN condition shape: {condition.shape}, processed WAN shape for cross-attention: {wan_tokens.shape} should be [B, 390, 32]") # → [B, 390, 32] 
                cross_context = self.wan_mlp_cross(wan_tokens)
                # print(f"[FullSetTransformerDenoiser] cross-attention context shape: {cross_context.shape} should be [B, 390, dim]") #[B, 390, dim]

    
    

        for block in self.blocks:
            h = block(h, context, cross_context)
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

def make_model( device, args):
    scene_embed_dim = 0 if args.cond_mode == "none" else (1 if args.cond_method == "scene_id" else 199680)
    model_name = args.model_name
    inout_dim = 5 if args.train_rcs_doppler else 3
    cond_mode = args.cond_mode
    cond_method = args.cond_method  

    
    if model_name == "PTv3Dnsr":
        sys.path.append("/home/palakons/point_diffusion")
        from unet_diffuser import (
            # MLPDenoiser,
            # SimplePointUNet,
            # PointNetLikeDenoiser,
            PTv3Dnsr,
        )
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
            accept_bnc=True
        ).to(device)

    elif model_name == "SetTxDnsr":
        dim=args.set_tx_dim
        set_cond_type =args.set_cond_type
        model = FullSetTransformerDenoiser(
            in_channels=inout_dim,
            dim =dim,
            time_dim=256,
            num_layers=5,
            num_heads=8,
            dropout=0.,
            out_channels=inout_dim,
            wan_shape=(16, 2, 60, 104) if cond_mode != "none" and cond_method != 'scene_id' else (1,),
            cond_type=set_cond_type ,
            mlp_ratio=4,
            use_condition_pooling=args.use_condition_pooling,
            condition_pool_kernel=args.condition_pool_kernel,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model