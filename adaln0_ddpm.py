"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

import sys

sys.path.insert(0, "/home/palakons/DiT")
from models import DiT

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
    query_gpu_stats,
)
from torch.utils.tensorboard import SummaryWriter
from adaln0_import import (
    save_checkpoint,
    load_checkpoint,
    train_eval_batch,
    makeDataset,
    splitDataset,
    makeDataloaders,
    makeOptimizer,
    tblogHparam,
    makeDiTModel,
)


# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,60) --> square (B,16,2,60,60)
# 'occupancy_grid' (B,Z,H,H)  --> (B,1,Z,H,H)


def train_vae_voxel_ddpm(
    dataset,
    val_dataset,
    config,
    checkpoint_dir,
    run_id=None,
    tb_dir=None,
):
    """
    Train DDPM model

    Args:
        dataset: MANDataset instance
        config: dict with training config
    """
    # Initialize model

    writer = SummaryWriter(f"{tb_dir}/{run_id}")

    # Log hyperparameters
    config["train_size"] = len(dataset)
    config["node_name"] = os.uname().nodename
    config["gpu_name"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()

    print("actual image size:", config["original_image_size"])
    print("scaled image size:", config["scaled_image_size"])
    print("max voxel grid depth:", config["max_voxel_grid_depth"])
    print("depth voxel grid bins:", config["depth_voxel_grid_bins"])
    print("==" * 10)
    print(
        f"resolution per depth bin (meters): {config['max_voxel_grid_depth'] / config['depth_voxel_grid_bins']:.4f} m, pixel resolution: y {config['original_image_size'][1] / config['scaled_image_size'][1]:.4f}, x {config['original_image_size'][0] / config['scaled_image_size'][0]:.4f}",
    )
    print("FOV 120deg H, 73deg V, One scaled pixel covers:")
    print(
        f"@125m h:{125 * np.tan(np.radians(120/2)) / config['scaled_image_size'][1]:.4f}, v:{125 * np.tan(np.radians(73 / 2)) / config['scaled_image_size'][0]:.4f} m",
    )
    print("==" * 10)

    ddpm = makeDiTModel(config)

    dataloader = makeDataloaders(
        dataset,
        config,
        is_train=True,
    )
    eval_dataloader = makeDataloaders(
        dataset,
        config,
        is_train=False,
    )

    # Setup optimizer
    optimizer = makeOptimizer(ddpm, config)

    print(
        "Total parameters:",
        sum(p.numel() for p in ddpm.parameters() if p.requires_grad),
    )
    config["model/total_params"] = sum(
        p.numel() for p in ddpm.parameters() if p.requires_grad
    )
    # Training loop
    start_epoch, _ = load_checkpoint(ddpm, optimizer, config)
    num_epochs = config["dit_epochs"]
    epoch_bar = trange(
        start_epoch,
        num_epochs,
        desc="Epochs",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    )
    print("from epoch:", start_epoch, "to", num_epochs)
    best_epoch_train_total_loss = float("inf")
    best_epoch = start_epoch + 1
    saved_epoch = start_epoch + 1
    global_step = 0
    for epoch in epoch_bar:

        avg_ddpm_train_loss, avg_bce_train_loss, avg_total_train_loss, global_step = (
            train_eval_batch(
                ddpm,
                dataloader,
                optimizer,
                config,
                global_step,
                train=True,
                writer=writer,
            )
        )
        writer.add_scalar("Loss/train/ddpm", avg_ddpm_train_loss, epoch + 1)
        if config["aux_occ_weight"] > 0:
            writer.add_scalar("Loss/train/bce", avg_bce_train_loss, epoch + 1)
            writer.add_scalar("Loss/train/total", avg_total_train_loss, epoch + 1)
        if (epoch + 1) % config["eval_every"] == 0:
            avg_val_ddpm_loss, avg_val_bce_loss, avg_val_total_loss, _ = (
                train_eval_batch(
                    ddpm,
                    eval_dataloader,
                    None,
                    config,
                    global_step,
                    train=False,
                    writer=writer,
                )
            )
            writer.add_scalar("Loss/val/ddpm", avg_val_ddpm_loss, epoch + 1)
            if config["aux_occ_weight"] > 0:
                writer.add_scalar("Loss/val/bce", avg_val_bce_loss, epoch + 1)
                writer.add_scalar("Loss/val/total", avg_val_total_loss, epoch + 1)

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

        if (epoch + 1) % config[
            "save_every"
        ] == 0 or avg_total_train_loss < best_epoch_train_total_loss:
            if (epoch + 1) % config["save_every"] == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"vae_voxel_ddpm_{run_id}_at{epoch+1}.pth"
                )
                prev_path = os.path.join(
                    checkpoint_dir, f"vae_voxel_ddpm_{run_id}_at{saved_epoch}.pth"
                )
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                saved_epoch = epoch + 1
            if avg_total_train_loss < best_epoch_train_total_loss:
                best_epoch_train_total_loss = avg_total_train_loss
                # Remove previous best checkpoint
                prev_path = os.path.join(
                    checkpoint_dir, f"best_vae_voxel_ddpm_{run_id}_at{best_epoch}.pth"
                )
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                best_epoch = epoch + 1
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"best_vae_voxel_ddpm_{run_id}_at{best_epoch}.pth"
                )

            save_checkpoint(
                ddpm,
                optimizer,
                epoch + 1,
                {
                    "ddpm_loss": avg_ddpm_train_loss,
                },
                checkpoint_path,
                config,
            )
        epoch_bar.set_description(
            f"l:{avg_ddpm_train_loss:.4f}/{avg_bce_train_loss:.4f}/{avg_total_train_loss:.4f}best:{best_epoch_train_total_loss:.4f}@{best_epoch}"
        )

    tblogHparam(config, writer, {"avg_ddpm_train_loss": avg_ddpm_train_loss})
    writer.close()
    checkpoint_path = os.path.join(checkpoint_dir, f"final_vae_voxel_ddpm_{run_id}.pth")
    save_checkpoint(
        ddpm,
        optimizer,
        num_epochs,
        {
            "ddpm_loss": avg_ddpm_train_loss,
        },
        checkpoint_path,
        config,
    )
    print(f"Training completed. Final model saved.", checkpoint_path)


def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    plot_dir = "/data/palakons/man_vaevoxelmetadit/plots"
    checkpoint_dir = "/data/palakons/man_vaevoxelmetadit/checkpoints"
    tb_log_dir = "/home/palakons/logs/tb_log/vaevoxelmetadit"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Load dataset
    dataset_point = makeDataset(
        config,
    )

    train_dataset, val_dataset = splitDataset(dataset_point, split=0.5)

    print("Starting training...")
    runid = f"MetaDiT_{get_runid(config)}"
    ddpm = train_vae_voxel_ddpm(
        train_dataset,
        val_dataset,
        config,
        checkpoint_dir=checkpoint_dir,
        run_id=runid,
        tb_dir=tb_log_dir,
    )


if __name__ == "__main__":
    main()
