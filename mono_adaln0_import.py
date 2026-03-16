import sys

sys.path.insert(0, "/home/palakons/DiT")
from models import TimestepEmbedder,VAEFeatureEmbedder, DiTBlock, modulate

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from man_ddpm import MANDataset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from torch.utils.data import Subset
import torch.nn.functional as F
from dit_ddpm_class import parse_args, set_seed, TransformerDenoiser, get_runid
from torch.utils.tensorboard import SummaryWriter
from math import prod


class SimpleDDPM(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, depth=4):
        super().__init__()
        layers = []
        assert depth>0, "Depth must be at least 1"
        for i in range(depth ):
            layers.append(nn.Linear(in_channels if i==0 else hidden_dim , hidden_dim if i<depth-1 else in_channels, bias=True))
            if i < depth - 1:
                layers.append(nn.ReLU())
        print("layers",layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # Optionally, add timestep embedding (not strictly necessary for basic DDPM)
        return self.net(x)
def train_basic_ddpm(model, dataloader, optimizer, scheduler, device, num_timesteps=1000, num_epochs=10):
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, beta_schedule="linear")
    writer = SummaryWriter()
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            pointcloud = batch.to(device)  # shape: (B, N, 3)
            B, N, C = pointcloud.shape
            optimizer.zero_grad()

            # Sample random timesteps
            timesteps = torch.randint(0, num_timesteps, (B,), device=device).long()

            # Sample noise
            noise = torch.randn_like(pointcloud)

            # Add noise to pointcloud
            noised_pointcloud = noise_scheduler.add_noise(pointcloud, noise, timesteps)

            # Model predicts noise
            noise_pred = model(noised_pointcloud, timesteps)

            # Loss: MSE between predicted noise and true noise
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("loss", loss.item(), epoch)
        print(f"Epoch {epoch}: Loss {loss.item()}")
# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,60) --> square (B,16,2,60,60)
# 'occupancy_grid' (B,Z,H,H)  --> (B,1,Z,H,H)


class FinalLayerSet(nn.Module):
    """
    The final layer of DiT for sets (no unpatchify).
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        print("FinalLayerSet input x:\n", x.detach().cpu().numpy() ) #torch.Size([1, 500, 1024])
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        print("FinalLayerSet shift, scale", shift.detach().cpu().numpy(), scale.detach().cpu().numpy())
        normed_x = self.norm_final(x)
        print("FinalLayerSet normed_x:\n", normed_x.detach().cpu().numpy()) #torch.Size([1, 500, 1024])
        x = modulate(normed_x, shift, scale)
        print("FinalLayerSet modulated x:\n", x.detach().cpu().numpy()) #torch.Size([1, 500, 1024])
        x = self.linear(x)
        print("FinalLayerSet output x:\n", x.detach().cpu().numpy()) #torch.Size([1, 500, 3])
        return x
class DiTSet(nn.Module):
    """
    DiT for unordered point sets.
    Input x: (B, N, C_in) e.g. C_in=3 for xyz noise
    Output:  (B, N, C_out) (predict epsilon for diffusion)
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        vae_feature_dim: int = 0,
        learn_sigma=False,
        use_token_pos_embed=False,  # keep False for strict permutation equivariance
        seed_model=42,
    ):
        super().__init__()
        if vae_feature_dim <= 0:
            raise ValueError("vae_feature_dim must be > 0")
        torch.manual_seed(seed_model)
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.use_token_pos_embed = use_token_pos_embed

        #put names in to enery layers
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.x_embedder.__name__ = "x_embedder"

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder.__name__ = "t_embedder"
        self.cond_embedder = VAEFeatureEmbedder(
            in_dim=vae_feature_dim,
            hidden_size=hidden_size,
            dropout_prob=class_dropout_prob,
        )
        self.cond_embedder.__name__ = "cond_embedder"

        self.blocks = nn.ModuleList(
        )
        for i in range(depth):
            self.blocks.add_module(f"block_{i}", DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio))

        self.final = FinalLayerSet(hidden_size, self.out_channels)
        self.final.__name__ = "final"


        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0
        )
    def forward(self, x, t, vae_feature=None,force_drop_ids=None,verbose=False):
        # 1. Embeddings
        h = self.x_embedder(x)
        t_emb = self.t_embedder(t).unsqueeze(1) # (B, 1, D)
        # print("t: ", t.detach().cpu().numpy())
        # print("->t_emb:\n", t_emb.detach().cpu().numpy())
        # print("noisy x:\n", x.detach().cpu().numpy())
        # print("->h after x_embedder:\n", h.detach().cpu().numpy())

        # 2. Combine (Skip vae_feature for the simplest test)
        h = h + t_emb

        # 3. Simple MLP pass (if blocks are 0, this does nothing)
        for block in self.blocks:
            h   = block(h, t_emb.squeeze(1))

        # 4. Final Projection (Ensure self.final.linear is NOT zero-initialized)
        out = self.final.linear(h)
        # print("out shape:", out.shape) #torch.Size([1, 500, 3])
        # print("->out:\n", out.detach().cpu().numpy())
        return out
    def forward_old(self, x, t, vae_feature, force_drop_ids=None,verbose=False):
            # print("x shape:", x.shape) #([1, 500, 3])

        h = self.x_embedder(x)  # (B, N, D), embed the last dimension (C) to hidden_size (D)
        # print("h after x_embedder shape:", h.shape) #torch.Size([1, 500, 1024])

        if self.use_token_pos_embed:
            # not recommended if you want strict order invariance
            if (self.token_pos is None) or (self.token_pos.shape[1] != h.shape[1]):
                self.token_pos = nn.Parameter(torch.zeros(1, h.shape[1], self.hidden_size, device=h.device))
                nn.init.normal_(self.token_pos, std=0.02)
            h = h + self.token_pos

        t_emb = self.t_embedder(t)  # (B, D)
        print("t: ", t.detach().cpu().numpy())
        print("=>t_emb:\n", t_emb.detach().cpu().numpy())
        print("noisy x:\n", x.detach().cpu().numpy())
        print("=>h after x_embedder:\n", h.detach().cpu().numpy())
        # print("t_emb shape:", t_emb.shape) #([1, 1024])
        c_emb = self.cond_embedder(vae_feature, self.training, force_drop_ids=force_drop_ids)  # (B, D)
        # print("c_emb shape:", c_emb.shape) #([1, 1024])
        # print("h shape:", h.shape) #torch.Size([1, 500, 1024])
        print("vae_feature shape:\n", vae_feature.shape, "norm:", vae_feature.norm().item())
        print("=>c_emb:\n", c_emb.detach().cpu().numpy())
        c = t_emb + c_emb
        
        for i,block in enumerate(self.blocks):
            h = block(h, c)
            print(f" .. h after block {i}:\n", h)
            # print(" .. h after block shape:", h.shape) #torch.Size([1, 500, 1024])

        # print("h after blocks shape:", h.shape) #torch.Size([1, 500, 1024])
        print("h after blocks:\n", h.detach().cpu().numpy())

        out = self.final(h, c)  # (B, N, C_out)
        # print("out shape:", out.shape) #torch.Size([1, 500, 3])
        print("=>out:\n", out.detach().cpu().numpy())
        if verbose:
            print()
        return out

def save_checkpoint(model: DiTSet, optimizer, epoch, loss, checkpoint_path, config,lr_scheduler):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: DiTSet, optimizer, config,lr_scheduler):
    if config["dit_checkpoint"] == "":
        return 0, 0
    checkpoint_path = config["dit_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    epoch = checkpoint["epoch"]

    print(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
    return epoch, best_epoch_loss


def get_crop(tensor: torch.Tensor, roi, original_size, dims):
    """
    Crop `tensor` along two dimensions given by `dims` using an ROI defined in the
    coordinate system of `original_size`.

    Args:
        tensor: torch.Tensor (N-D)
        roi: tuple (min_v, max_v, min_u, max_u) in pixels of original image space
        original_size: tuple (H, W) of the original image space used by roi
        dims: tuple/list of two ints -> which tensor dims correspond to (v/h, u/w).
              Example: for occupancy_grid (B,Z,H,W), dims=(-2, -1)
                       for wan_vae_latent   (B,16,2,H,W), dims=(-2, -1)

    Returns:
        Cropped tensor (a view when possible).
    """
    if tensor is None:
        return None
    if len(dims) != 2:
        raise ValueError(f"dims must have length 2, got {dims}")

    dv, du = int(dims[0]), int(dims[1])
    nd = tensor.ndim
    if dv < 0:
        dv += nd
    if du < 0:
        du += nd
    if dv == du or not (0 <= dv < nd) or not (0 <= du < nd):
        raise ValueError(f"Invalid dims {dims} for tensor.ndim={nd}")

    min_v, max_v, min_u, max_u = roi
    H, W = original_size

    tv = tensor.shape[dv]
    tu = tensor.shape[du]

    # map roi from original pixel space -> tensor index space
    h0 = int(min_v * tv / H)
    h1 = int(max_v * tv / H)
    w0 = int(min_u * tu / W)
    w1 = int(max_u * tu / W)

    # clamp to valid range
    h0 = max(0, min(tv, h0))
    h1 = max(0, min(tv, h1))
    w0 = max(0, min(tu, w0))
    w1 = max(0, min(tu, w1))

    # ensure non-empty / ordered (optional; you can also allow empty crops)
    if h1 < h0:
        h0, h1 = h1, h0
    if w1 < w0:
        w0, w1 = w1, w0

    sl = [slice(None)] * nd
    sl[dv] = slice(h0, h1)
    sl[du] = slice(w0, w1)
    return tensor[tuple(sl)]


def get_latent_and_grid_crop(wan_vae_latent, occupancy_grid):
    # wan_vae_latent: [B,16,2,60,104]
    # occupancy_grid: (B,Z,H,W)

    assert len(wan_vae_latent.shape) == 5
    assert len(occupancy_grid.shape) == 4
    assert (
        occupancy_grid.shape[2] <= occupancy_grid.shape[3]
    ), "occupancy grid height should be less than width"
    assert (
        wan_vae_latent.shape[3] <= wan_vae_latent.shape[4]
    ), "wan vae latent height should be less than width"
    # lazy centering crop
    vae_left = (wan_vae_latent.shape[4] - wan_vae_latent.shape[3]) // 2
    grid_left = (occupancy_grid.shape[3] - occupancy_grid.shape[2]) // 2

    wan_vae_latent = wan_vae_latent[
        :, :, :, :, vae_left : vae_left + wan_vae_latent.shape[3]
    ]
    occupancy_grid = occupancy_grid[
        :, :, :, grid_left : grid_left + occupancy_grid.shape[2]
    ]
    return wan_vae_latent, occupancy_grid


def train_eval_batch(
    model,
    dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    config,
    global_step: int,
    train=True,
    loss_fn=F.mse_loss,
    writer: SummaryWriter = None,
    data_mean=None,
    data_std=None,
):
    if train:
        model.train()
    else:
        model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        # beta_schedule="squaredcos_cap_v2",
        clip_sample=False,  # problem
        clip_sample_range=1,
        prediction_type="epsilon",
    )

    pbar = tqdm(dataloader, leave=False, desc="Train Batch" if train else "Eval Batch")
    sum_ddpm_loss = 0.0
    for i_batch, batch in enumerate(pbar):
        with torch.no_grad() if not train else torch.enable_grad():
            if train:
                optimizer.zero_grad()
            # print("Batch keys:", batch.keys()) #['filtered_radar_data', 'uvz', 'frame_token', 'npoints_original', 'npoints_filtered', 'wan_vae_latent', 'camera_front', 'scene_id', 'frame_index', 'occupancy_grid'
            pointcloud = batch["filtered_radar_data"].to(config["device"])  # (B,N,3)
            #is ti normalized? NO
            # print(f"shape {pointcloud.shape}") #[1, 500, 7])                                                       
            # print(f"min {pointcloud.min().item()}, max {pointcloud.max().item()}, mean {pointcloud.mean().item()}, std {pointcloud.std().item()}") #min -75.28197479248047, max 179.88827514648438, mean 14.010276794433594, std 37.37834167480469
            

            batch_size = pointcloud.shape[0]


            # DIFFUSION TRAINING (only this is trainable)
            normalized_pointcloud = ((pointcloud - data_mean) / data_std)[:,:,:3] #dont care the extra attribute for now
            # print(f"normalized pointcloud stats - min: {normalized_pointcloud.min().item()}, max: {normalized_pointcloud.max().item()}, mean: {normalized_pointcloud.mean().item()}, std: {normalized_pointcloud.std().item()}") #min: -4.259920120239258, max: 6.928348541259766, mean: 0.054124411195516586, std: 0.9864383935928345

            noise = torch.zeros_like(normalized_pointcloud) + torch.randn_like(normalized_pointcloud)  # (B,N,3)

            # Make noisy grid
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=config["device"],
            ).long()  # random timesteps for each sample in the batch [B,]
            noised_pointcloud = noise_scheduler.add_noise(
                normalized_pointcloud, noise, timesteps
            )  # (B,N,3)
            # print("actual noise :\n", noise.detach().cpu().numpy() ) #torch.Size([1, 800, 3])
            # print("actual pointcloud :\n", normalized_pointcloud.detach().cpu().numpy() ) #torch.Size([1, 800, 3])

            #if model is type DiTSet
            if isinstance(model, DiTSet):

                if config["zero_conditioning"]:
                    # print("zero conditioning!")
                    wan_vae_latent = torch.zeros(
                        (batch_size, 16, 2, 60, 104),
                        device=config["device"],
                    )  # [4, 16, 2, 60, 104]
                else:
                    wan_vae_latent = batch["wan_vae_latent"].to(
                        config["device"]
                    )  # [4, 16, 2, 60, 104]            

                # print("wan_vae_latent shape:", wan_vae_latent.shape) #torch.Size([1, 16, 2, 60, 104])
                
                if config["use_global_avg_pool"]:
                    condition = wan_vae_latent  # , because will be avg pooled in model
                else:
                    condition = wan_vae_latent.flatten(1)  #
                # print("condition shape:", condition.shape)  # ([1, 115200])

                # print("noised_pointcloud t c",noised_pointcloud.shape,timesteps.shape,condition.shape) #torch.Size([1, 800, 3]) torch.Size([1]) torch.Size([1, 199680]) 
                output = model(noised_pointcloud, timesteps, vae_feature=condition, verbose=True)  # (B,N,3) or (B,2*N,3) if learn_sigma

                if config["learn_sigma"]:
                    noise_pred = output[:, : output.shape[1] // 2, ...]  # ([1, 8, 8, 8])
                    sigma_pred = output[:, output.shape[1] // 2 :, ...]  # ([1, 8, 8, 8])
                else:
                    noise_pred = output  # ([1, 8, 8, 8])
                    sigma_pred = None
                # print("noise_pred shape:", noise_pred.shape)  # ([1, 8, 8, 8])
                # print("sigma_pred shape:", sigma_pred.shape)  # ([1, 8, 8, 8])
            #if model is type SimpleDDPM
            elif isinstance(model, SimpleDDPM):
                noise_pred = model(noised_pointcloud, timesteps)
            else:
                raise ValueError(f"Unknown model type {type(model)}")

            global_step += 1
            ddpm_loss = loss_fn(noise, noise_pred)
            # print(f"ddpm_loss: {ddpm_loss.item()}") #ddpm_loss: 0.6931
            sum_ddpm_loss += ddpm_loss.item()

            if train:
                ddpm_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if True:# log gradients histogram, layer by layer

                    for i, (name, param_named) in enumerate(model.named_parameters()):
                        # print(f"Parameter {i}: {name}, requires_grad={param_named.requires_grad}, grad is None: {param_named.grad is None}")
                        if param_named.grad is not None:
                            writer.add_histogram(f"gradients/{i}-{name}", param_named.grad.cpu().data.numpy(), global_step)
                            writer.add_scalar(f"gradients/{i}-{name}_mean", param_named.grad.cpu().data.mean().item(), global_step)
                            writer.add_scalar(f"gradients/{i}-{name}_std", param_named.grad.cpu().data.std().item(), global_step)
                            # print(f"Logged gradients for {name} at step {global_step}")

                        # also log parameters themselves
                        writer.add_histogram(f"parameters/{i}-{name}", param_named.cpu().data.numpy(), global_step)
                        writer.add_scalar(f"parameters/{i}-{name}_mean", param_named.cpu().data.mean().item(), global_step)
                        writer.add_scalar(f"parameters/{i}-{name}_std", param_named.cpu().data.std().item(), global_step)
                        # print(f"Logged parameters for {name} at step {global_step}")

            pbar.set_postfix(
                {
                    "loss": f"{ddpm_loss:.4f}"
                }
            )

    avg_ddpm_loss = sum_ddpm_loss / len(dataloader)
    return avg_ddpm_loss,  global_step


def makeDiTSetModel(config):

    # in_channels=3,
    # hidden_size=768,
    # depth=12,
    # num_heads=12,
    # mlp_ratio=4.0,
    # class_dropout_prob=0.1,
    # vae_feature_dim: int = 0,
    # learn_sigma=False,
    # use_token_pos_embed=False,
    return DiTSet(
        in_channels=3,
        hidden_size=config["latent_dim"],
        depth=config["num_transformer_blocks"],
        num_heads=config["num_attention_heads"],
        mlp_ratio=4.0,
        class_dropout_prob=0,
        vae_feature_dim=(16 * 2 * 60 * 104) if not config["use_global_avg_pool"] else 16,
        learn_sigma=config["learn_sigma"],
        use_token_pos_embed=False,
        seed_model=config["seed_model"],
    )