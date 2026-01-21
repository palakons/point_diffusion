import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from man_ddpm import MANDataset, chamfer_distance
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
import argparse
import sys

sys.path.insert(0, "/home/palakons/PointNeXt")
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint
from torch.utils.tensorboard import SummaryWriter

# import F
from torch.nn import functional as F


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
            nn.Linear(latent_dim // 4, 3),
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
            uvz: (B, output_points, 3)
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
        uvz = self.coord_head(decoded_features)
        confidence = self.confidence_head(decoded_features)

        return uvz, confidence


class AttentionPointDecoder(nn.Module):
    """
    Decode latent embeddings to UVZ point cloud coordinates.
    Uses cross-attention mechanism similar to DETR/Point-E.
    """

    def __init__(self, latent_dim=1024, output_points=500, hidden_dim=512):
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
            embed_dim=latent_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feedforward network to refine features
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Output (u, v, z) coordinates
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
            uvz: (B, output_points, 3) - UVZ coordinates
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
        uvz = self.coord_head(refined)  # (B, output_points, 3)

        # Optional: predict confidence scores
        confidence = self.confidence_head(refined)  # (B, output_points, 1)

        return uvz, confidence


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
        device="cuda",
        pointnext_home="/home/palakons/PointNeXt",
        decoder_model="attention",  # "transformer" or "attention"
        num_decoder_layers=6,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_points = output_points
        self.seq_length = seq_length
        self.device = device

        # --- 1. Load and Freeze PointNeXt Encoder ---
        cfg = EasyConfig()
        cfg.load(
            f"{pointnext_home}/cfgs/shapenetpart/pointnext-s_c64.yaml", recursive=True
        )
        full_model = build_model_from_cfg(cfg.model)
        pretrained_path = f"{pointnext_home}/pretrained/shapenetpart/pointnext-s-c64/checkpoint/shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth"
        load_checkpoint(full_model, pretrained_path=pretrained_path)

        self.pointnext_encoder = full_model.encoder.to(device)
        for param in self.pointnext_encoder.parameters():
            param.requires_grad = False

        encoder_dim = cfg.model.encoder_args.width * (
            2 ** (len(cfg.model.encoder_args.blocks) - 1)
        )

        # --- 2. Define Trainable Components ---
        self.encoder_proj = nn.Linear(encoder_dim, d_model)
        if decoder_model == "transformer":
            self.pointcloud_decoder = TransformerPointDecoder(
                latent_dim=d_model,
                output_points=output_points,
                num_decoder_layers=num_decoder_layers,
            )
        elif decoder_model == "attention":
            self.pointcloud_decoder = AttentionPointDecoder(
                latent_dim=d_model, output_points=output_points
            )

    def encode(self, uvz, training=True):
        """Converts a UVZ point cloud to a latent representation."""
        # Pad UVZ for PointNeXt
        uvz_padded = torch.cat(
            [uvz, torch.zeros(uvz.shape[0], uvz.shape[1], 4, device=uvz.device)], dim=-1
        )

        point_data = {
            "pos": uvz_padded[:, :, :3].contiguous(),
            "x": uvz_padded.transpose(1, 2).contiguous(),
        }

        # Get features from frozen encoder
        with torch.no_grad():
            encoder_output = self.pointnext_encoder(point_data)
            features = (
                encoder_output[1][-1]
                if isinstance(encoder_output, tuple)
                else encoder_output
            )
            if features.shape[1] > features.shape[2]:
                features = features.transpose(1, 2)

        # Project and resample to the correct sequence length
        latent = self.encoder_proj(features)
        latent = self.resample_to_seq_length(latent, training=training)
        return latent

    def decode(self, latent):
        """Converts a latent representation back to a UVZ point cloud."""
        return self.pointcloud_decoder(latent)

    def forward(self, uvz):
        """Full autoencoder pass for reconstruction."""
        latent = self.encode(uvz, training=self.training)
        predicted_uvz, confidence = self.decode(latent)
        return predicted_uvz, confidence, latent

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


def pretrain_autoencoder(
    dataset, config, autoencoder, checkpoint_path=None, run_id=None, tb_dir=None
):
    """
    Pre-train PointNeXt encoder projection + decoder separately.
    This learns a good UVZ ↔ Latent mapping.
    """
    worker_init_fn = set_seed(config["seed"])
    writer = SummaryWriter(f"{tb_dir}/{run_id}/ae_pretraining")

    # Log hyperparameters
    writer.add_text("config/ae_lr", str(config["ae_lr"]))
    writer.add_text("config/ae_epochs", str(config["ae_epochs"]))
    writer.add_text("config/batch_size", str(config["batch_size"]))
    writer.add_text("config/latent_dim", str(config["latent_dim"]))
    writer.add_text("config/latent_seq_length", str(config["latent_seq_length"]))
    writer.add_text("config/num_points", str(config["num_points"]))
    writer.add_text("config/point_ae_model", str(config["point_ae_model"]))
    writer.add_text("config/dataset_size", str(len(dataset)))
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),  # Much simpler!
        lr=config["ae_lr"],  # Higher LR for autoencoder
        weight_decay=config["ae_decay"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        # collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=config["num_workers"],
    )

    num_epochs = config["ae_epochs"]  # Separate epoch count

    global_step = 0
    epoch_trange = trange(num_epochs, desc="Autoencoder Pretraining")
    for epoch in epoch_trange:
        epoch_loss = 0
        batch_losses = []

        for batch in tqdm(dataloader, leave=False, desc="Batches"):
            uvz = batch["uvz"].to(config["device"])

            predicted_uvz, confidence, latent = autoencoder(uvz)

            # RECONSTRUCTION LOSS ONLY
            loss = chamfer_distance(predicted_uvz, uvz)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            # Track gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)

            # ✅ LOG BATCH METRICS TO TENSORBOARD
            writer.add_scalar("ae/batch_loss", batch_loss, global_step)
            writer.add_scalar("ae/grad_norm", grad_norm.item(), global_step)
            writer.add_scalar(
                "ae/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            # Log latent statistics
            if global_step % 10 == 0:
                writer.add_scalar("ae/latent_mean", latent.mean().item(), global_step)
                writer.add_scalar("ae/latent_std", latent.std().item(), global_step)
                writer.add_scalar(
                    "ae/confidence_mean", confidence.mean().item(), global_step
                )

            global_step += 1

        # Epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        std_loss = np.std(batch_losses)

        epoch_trange.set_postfix({"Chamfer Loss": f"{avg_loss:.6f}"})

        # ✅ LOG EPOCH METRICS
        writer.add_scalar("ae/epoch_loss_mean", avg_loss, epoch)
        writer.add_scalar("ae/epoch_loss_std", std_loss, epoch)
        epoch_peak = torch.cuda.max_memory_allocated() / 1024**3
        writer.add_scalar("gpu/epoch_peak_gb", epoch_peak, epoch)

        # ✅ VISUALIZE RECONSTRUCTION EVERY 5 EPOCHS
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Take first sample from batch
                sample_uvz = uvz[:1]
                sample_latent = latent[:1]
                sample_pred = predicted_uvz[:1]  # Autoencoder prediction

                # Create visualization
                fig = create_uvz_comparison_plot(
                    sample_uvz[0].cpu().numpy(),
                    sample_pred[0].cpu().numpy(),
                    title=f"Epoch {epoch+1}",
                )

                # Log figure to TensorBoard
                writer.add_figure("ae/reconstruction", fig, epoch)
                plt.close(fig)

                # Log Chamfer distance for this sample
                sample_chamfer = chamfer_distance(sample_pred, sample_uvz).item()
                writer.add_scalar("ae/sample_chamfer", sample_chamfer, epoch)

        # ✅ LOG HISTOGRAMS EVERY 10 EPOCHS
        # if (epoch + 1) % 10 == 0:
        #     for name, param in autoencoder.encoder_proj.named_parameters():
        #         writer.add_histogram(f"ae/encoder_proj/{name}", param, epoch)
        #         if param.grad is not None:
        #             writer.add_histogram(
        #                 f"ae/encoder_proj/{name}.grad", param.grad, epoch
        #             )

        #     for name, param in autoencoder.pointcloud_decoder.named_parameters():
        #         writer.add_histogram(f"ae/decoder/{name}", param, epoch)
        #         if param.grad is not None:
        #             writer.add_histogram(f"ae/decoder/{name}.grad", param.grad, epoch)

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
        "--save_every", type=int, default=1000, help="Save checkpoint every N epochs."
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
        "--num_encoder_layers",
        type=int,
        default=4,
        help="Number of encoder layers for transformer point AE.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=4,
        help="Number of decoder layers for transformer point AE.",
    )
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


def get_runid(config):

    scene_str = "-".join(map(str, sorted(config["scene_ids"])))
    return f"{config['depth_voxel_grid_bins']}bins_{config['scaled_image_size'][0]}x{config['scaled_image_size'][1]}_{config['dit_epochs']}epochs_batch{config['batch_size']}_scenes{scene_str}_numframes{config['num_input_frames']}_seed{config['seed']}_pool{config['use_global_avg_pool']}_sigma{config['learn_sigma']}_zeroCond{config['zero_conditioning']}_patch{config['patch_size']}_transBlocks{config['num_transformer_blocks']}_attnHeads{config['num_attention_heads']}_latentdim{config['latent_dim']}_grange{config['grid_binary_range']}_auxoccw{config['aux_occ_weight']:.0E}_auxoccs{config['aux_occ_scale']:.1f}"
