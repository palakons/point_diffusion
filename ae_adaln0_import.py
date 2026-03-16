import sys

sys.path.insert(0, "/home/palakons/DiT")
from models_org import AEDiT

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
from dit_ddpm_class import (
    parse_args,
    set_seed,
    TransformerDenoiser,
    get_runid,
    PretrainedPointNeXtEncoderPointAE,
    get_pointnextdit_runid,
)
from torch.utils.tensorboard import SummaryWriter
from math import prod


# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,60) --> square (B,16,2,60,60)
# 'occupancy_grid' (B,Z,H,H)  --> (B,1,Z,H,H)


def save_checkpoint(model: AEDiT, optimizer, epoch, loss, checkpoint_path, config):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch_loss": loss,
        "best_epoch": epoch,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def find_checkpoint_path(checkpoint_dir, config):
    runid = get_pointnextdit_runid(config)
    #  f"best_vae_voxel_ddpm_{run_id}_at{best_epoch}.pth"
    #  f"final_vae_voxel_ddpm_{run_id}.pth"
    #  f"vae_voxel_ddpm_{run_id}_at{saved_epoch}.pth"
    print("Looking for checkpoints in:", checkpoint_dir, "with runid:", runid)
    first_pattern = f"vae_pointae_ddpm_AEDiT_{runid}_at"
    second_pattern = f"final_vae_pointae_ddpm_AEDiT_{runid}.pth"
    third_pattern = f"vae_pointae_ddpm_AEDiT_{runid}_at"

    # find files in checkpoint dir matches first_pattern, using
    for pattern in [first_pattern, second_pattern, third_pattern]:
        checkpoint_files = os.listdir(f"{checkpoint_dir}")
        # print("Searching:", pattern)
        matched_files = [
            f for f in checkpoint_files if f.startswith(pattern) and f.endswith(".pth")
        ]
        if len(matched_files) == 0:
            print("No matching files found for pattern:", pattern)
            continue
        elif len(matched_files) == 1:
            checkpoint_path = os.path.join(checkpoint_dir, matched_files[0])
            print("Found checkpoint:", checkpoint_path)
            return checkpoint_path
        elif len(matched_files) > 1:
            # sort by the number
            max_id = -1
            for f in matched_files:
                num_str = f[len(pattern) : -4]  # remove pattern and .pth
                try:
                    num = int(num_str)
                    if num > max_id:
                        max_id = num
                        checkpoint_path = os.path.join(checkpoint_dir, f)
                except:
                    continue
            if max_id == -1:
                print("No valid checkpoint number found in files:", matched_files)
                continue
            print("Found checkpoint with max id:", checkpoint_path)
            return checkpoint_path

    return ""


def load_checkpoint(model: AEDiT, optimizer, config):
    if config["dit_checkpoint"] == "":
        return 0, 0
    checkpoint_path = config["dit_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])

    # check if config matches
    important_fields = [
        "batch_size",
        "camera_channel",
        "data_file",
        "dit_decay",
        "dit_lr",
        "latent_dim",
        "latent_seq_length",
        "learn_sigma",
        "num_attention_heads",
        "num_input_frames",
        "num_points",
        "num_train_timesteps",
        "num_transformer_blocks",
        "radar_channel",
        "scene_ids",
        "seed",
        "use_global_avg_pool",
        "zero_conditioning",
        "ae_latent_normalizing_std",
        "ddpm_clip_sample",
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
    epoch = checkpoint["epoch"]
    best_epoch_loss = checkpoint["best_epoch_loss"]
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


def train_eval_batch_pointae_ddpm(
    model: AEDiT,
    autoencoder: PretrainedPointNeXtEncoderPointAE,
    noise_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer,
    config,
    global_step: int,
    train=True,
    loss_fn=F.mse_loss,
    writer: SummaryWriter = None,
):
    autoencoder.eval()
    if train:
        model.train()
    else:
        model.eval()

    bce_weight = float(config["aux_occ_weight"])  # e.g. 0.01
    bce_logit_scale = float(config["aux_occ_scale"])  # e.g. 5 or 10

    pbar = tqdm(dataloader, leave=False, desc="Train Batch" if train else "Eval Batch")
    sum_ddpm_loss = 0.0
    for i_batch, batch in enumerate(pbar):
        if train:
            optimizer.zero_grad()
        print("batch keys:", batch.keys())
        # npoints_original
        print("npoints_original:", batch["npoints_original"])
        exit()
        # if both not equal,
        filtered_radar_data = batch["filtered_radar_data"].to(config["device"])
        with torch.no_grad():
            predicted_radar_7d, confidence, latent = autoencoder(filtered_radar_data)
            writer.add_scalar(
                "latent/mean",
                latent.mean().item(),
                global_step,
            )
            writer.add_scalar(
                "latent/std",
                latent.std().item(),
                global_step,
            )
            latent = latent / config["ae_latent_normalizing_std"]
            latent = latent.detach()
            # print("after normalizing latent std:", latent.std().item())

            writer.add_scalar(
                "latent/std-after-norm",
                latent.std().item(),
                global_step,
            )

        # print("latent shape:", latent.shape)  # e([1 ,64, 768])
        batch_size = filtered_radar_data.shape[0]
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

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=config["device"],
        ).long()  # random timesteps for each sample in the batch [B,]

        # DIFFUSION TRAINING (only this is trainable)
        noise = torch.randn_like(latent)
        noise_point_latent = noise_scheduler.add_noise(latent, noise, timesteps)
        # print("noisy grid shape:", noise_center_grid.shape)  # ([1, 8, 8, 8])

        condition = (
            wan_vae_latent
            if config["use_global_avg_pool"]
            else wan_vae_latent.flatten(1)
        )
        with torch.set_grad_enabled(train):

            output = model(noise_point_latent, timesteps, vae_feature=condition)
            if config["learn_sigma"]:
                # output: (B, T, 2*D) -> split last dim
                noise_pred, sigma_pred = output.chunk(2, dim=-1)
            else:
                noise_pred = output  # ([1, 8, 8, 8])
                sigma_pred = None
            # print("noise_pred shape:", noise_pred.shape)  # ([1, 8, 8, 8])
            # print("sigma_pred shape:", sigma_pred.shape)  # ([1, 8, 8, 8])

            if writer is not None and global_step is not None:
                if global_step % config["save_every"] == 0:
                    tag_prefix = "train" if train else "val"

                    # writer.add_scalar(
                    #     f"{tag_prefix}/noise_pred_std",
                    #     noise_pred.std().item(),
                    #     global_step,
                    # )

            if train:
                global_step += 1
            # print("noise shape:", noise.shape)
            # print("noise_pred shape:", noise_pred.shape)
            ddpm_loss = loss_fn(noise, noise_pred)
            total_loss = ddpm_loss

            if train:
                total_loss.backward()
                optimizer.step()

            # -------------------------------
            # Teacher-forced x0 reconstruction loss (normalized latent space)
            # -------------------------------
            with torch.no_grad():
                step_out = noise_scheduler.step(
                    noise_pred, timesteps, noise_point_latent
                )
                x0_hat = (
                    step_out.pred_original_sample
                )  # same space as `latent` (normalized)
                x0_loss = F.mse_loss(x0_hat, latent)

            if writer is not None and global_step is not None:
                tag_prefix = "train" if train else "val"
                writer.add_scalar(
                    f"{tag_prefix}/ddpm_eps_mse", ddpm_loss.item(), global_step
                )
                writer.add_scalar(f"{tag_prefix}/x0_mse", x0_loss.item(), global_step)

            sum_ddpm_loss += ddpm_loss.item()

            pbar.set_postfix({"loss": f"{ddpm_loss:.4f}"})

    avg_ddpm_loss = sum_ddpm_loss / len(dataloader)
    return avg_ddpm_loss, global_step


def loadAECheckpoint(config):
    """
    load from pre-trained AE checkpoint
    pat, config['point_ae_checkpoint']
    """

    # checkpoint = {
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "epoch": epoch,
    #     "metrics": metrics,
    #     "config": config,
    # }
    checkpoint_path = config["point_ae_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])
    assert (
        checkpoint["config"]["latent_dim"] == config["latent_dim"]
    ), "latent dim mismatch"
    assert (
        checkpoint["config"]["latent_seq_length"] == config["latent_seq_length"]
    ), "latent seq length mismatch"

    autoencoder = PretrainedPointNeXtEncoderPointAE(
        d_model=config["latent_dim"],
        output_points=config["num_points"],
        seq_length=config["latent_seq_length"],
        query_latent_pool_nhead=checkpoint["config"]["query_num_heads"],
        query_latent_pool_dropout=checkpoint["config"]["query_dropout"],
        device=config["device"],
        decoder_model=checkpoint["config"]["point_ae_model"][10:],
        num_decoder_layers=checkpoint["config"]["decoder_num_layers"],
        num_decoder_head=checkpoint["config"]["decoder_num_heads"],
        decoder_dropout=checkpoint["config"]["decoder_dropout"],
        pointnext_config=checkpoint["config"]["pointnext_config"],
        # 48 nsample, 8 m radius
        # pointnext_config="scannet/pointnext-s.yaml",
        output_dim=7,
    ).to(config["device"])
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    print(f"Loaded AE checkpoint from {checkpoint_path}")
    return autoencoder, checkpoint["config"]


def makeAEDiTModel(config):
    return AEDiT(
        n_tokens=64,
        hidden_size=config["latent_dim"],
        depth=config["num_transformer_blocks"],  # n AEDiT blocks
        num_heads=config["num_attention_heads"],  # AEDiT block param
        mlp_ratio=4.0,  # AEDiT block param
        vae_feature_dim=(
            16 * 2 * 60 * 104 if not config["use_global_avg_pool"] else 16
        ),  # 16 * 2 * 60 * 104 because 832, 480
        class_dropout_prob=0.1,
        learn_sigma=config["learn_sigma"],
    ).to(config["device"])


def makeDDPMScheduler(config):
    return DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        # beta_schedule="squaredcos_cap_v2",
        clip_sample=config["ddpm_clip_sample"],  # problem
        clip_sample_range=1,
        prediction_type="epsilon",
    )
