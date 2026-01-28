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
from dit_ddpm_class import parse_args, set_seed, TransformerDenoiser, get_runid
from torch.utils.tensorboard import SummaryWriter


# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,60) --> square (B,16,2,60,60)
# 'occupancy_grid' (B,Z,H,H)  --> (B,1,Z,H,H)


def save_checkpoint(model: DiT, optimizer, epoch, loss, checkpoint_path, config):

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
    runid = get_runid(config)
    #  f"best_vae_voxel_ddpm_{run_id}_at{best_epoch}.pth"
    #  f"final_vae_voxel_ddpm_{run_id}.pth"
    #  f"vae_voxel_ddpm_{run_id}_at{saved_epoch}.pth"
    first_pattern = f"best_vae_voxel_ddpm_MetaDiT_{runid}_at"
    second_pattern = f"final_vae_voxel_ddpm_MetaDiT_{runid}.pth"
    third_pattern = f"vae_voxel_ddpm_MetaDiT_{runid}_at"

    # find files in checkpoint dir matches first_pattern, using
    for pattern in [first_pattern, second_pattern, third_pattern]:
        checkpoint_files = os.listdir(f"{checkpoint_dir}")
        print("Searching:", pattern)
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


def load_checkpoint(model: DiT, optimizer, config):
    if config["dit_checkpoint"] == "":
        return 0, 0
    checkpoint_path = config["dit_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])

    # check if config matches
    important_fields = [
        "aux_occ_scale",
        "aux_occ_weight",
        "batch_size",
        "camera_channel",
        "data_file",
        "depth_voxel_grid_bins",
        "dit_decay",
        "dit_lr",
        "grid_binary_range",
        "latent_dim",
        "latent_seq_length",
        "learn_sigma",
        "max_voxel_grid_depth",
        "num_attention_heads",
        "num_input_frames",
        "num_points",
        "num_train_timesteps",
        "num_transformer_blocks",
        "patch_size",
        "radar_channel",
        "scaled_image_size",
        "scene_ids",
        "seed",
        "use_global_avg_pool",
        "zero_conditioning",
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


def get_center_crop_latent_and_grid(wan_vae_latent, occupancy_grid):
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
    model: DiT,
    dataloader: DataLoader,
    optimizer,
    config,
    global_step: int,
    train=True,
    loss_fn=F.mse_loss,
    writer: SummaryWriter = None,
):
    if train:
        model.train()
    else:
        model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        # beta_schedule="squaredcos_cap_v2",
        clip_sample=True,  # problem
        clip_sample_range=1,
        prediction_type="epsilon",
    )
    bce_weight = float(config["aux_occ_weight"])  # e.g. 0.01
    bce_logit_scale = float(config["aux_occ_scale"])  # e.g. 5 or 10

    pbar = tqdm(dataloader, leave=False, desc="Train Batch" if train else "Eval Batch")
    sum_ddpm_loss = 0.0
    sum_bce_loss = 0.0
    sum_total_loss = 0.0
    for i_batch, batch in enumerate(pbar):
        with torch.no_grad() if not train else torch.enable_grad():
            if train:
                optimizer.zero_grad()
            else:
                if i_batch == 0:
                    # Sample the first item

                    # Process the first item as needed
                    # print(f"Eval Batch")
                    pass
                    # TODO: Handle eval batch processing as needed, implement .sample()

            occupancy_grid = batch["occupancy_grid"].to(config["device"])  # (B,Z,H,W)
            batch_size = occupancy_grid.shape[0]
            if config["zero_conditioning"]:
                wan_vae_latent = torch.zeros(
                    (batch_size, 16, 2, 60, 104),
                    device=config["device"],
                )  # [4, 16, 2, 60, 104]
            else:
                wan_vae_latent = batch["wan_vae_latent"].to(
                    config["device"]
                )  # [4, 16, 2, 60, 104]
            center_wan_vae_latent, center_occupancy_grid = (
                get_center_crop_latent_and_grid(wan_vae_latent, occupancy_grid)
            )

            # print(
            #     "center_wan_vae_latent shape:", center_wan_vae_latent.shape
            # )  # [4, 16, 2, 60, 60]
            # print(
            #     "center_occupancy_grid shape:", center_occupancy_grid.shape
            # )  # [ 4,  8,8,8]

            # Make noisy grid
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=config["device"],
            ).long()  # random timesteps for each sample in the batch [B,]

            # DIFFUSION TRAINING (only this is trainable)
            noise = torch.randn_like(center_occupancy_grid)
            noise_center_grid = noise_scheduler.add_noise(
                center_occupancy_grid, noise, timesteps
            )
            # print("noisy grid shape:", noise_center_grid.shape)  # ([1, 8, 8, 8])
            if config["use_global_avg_pool"]:
                condition = (
                    center_wan_vae_latent  # , because will be avg pooled in model
                )
            else:
                condition = center_wan_vae_latent.flatten(1)  #

            # print("condition shape:", condition.shape)  # ([1, 115200])

            output = model(noise_center_grid, timesteps, vae_feature=condition)
            if config["learn_sigma"]:
                noise_pred = output[:, : output.shape[1] // 2, ...]  # ([1, 8, 8, 8])
                sigma_pred = output[:, output.shape[1] // 2 :, ...]  # ([1, 8, 8, 8])
            else:
                noise_pred = output  # ([1, 8, 8, 8])
                sigma_pred = None
            # print("noise_pred shape:", noise_pred.shape)  # ([1, 8, 8, 8])
            # print("sigma_pred shape:", sigma_pred.shape)  # ([1, 8, 8, 8])

            if writer is not None and global_step is not None:
                if global_step % config["save_every"] == 0:
                    tag_prefix = "train" if train else "val"

                    # write hist of condition
                    writer.add_histogram(
                        f"{tag_prefix}/condition",
                        condition.detach().float().cpu(),
                        global_step,
                    )
                    # write  center_occupancy_grid
                    writer.add_histogram(
                        f"{tag_prefix}/center_occupancy_grid",
                        center_occupancy_grid.detach().float().cpu(),
                        global_step,
                    )
                    writer.add_histogram(
                        f"{tag_prefix}/noise_pred",
                        noise_pred.detach().float().cpu(),
                        global_step,
                    )
                    writer.add_histogram(
                        f"{tag_prefix}/x_t_noisy",
                        noise_center_grid.detach().float().cpu(),
                        global_step,
                    )
                    # optional quick scalars
                    writer.add_scalar(
                        f"{tag_prefix}/noise_pred_mean",
                        noise_pred.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f"{tag_prefix}/noise_pred_std",
                        noise_pred.std().item(),
                        global_step,
                    )

            global_step += 1
            ddpm_loss = loss_fn(noise, noise_pred)
            sum_ddpm_loss += ddpm_loss.item()
            total_loss = ddpm_loss
            bce_loss = 0

            if bce_weight > 0:
                # Estimate x0 from current x_t and eps_pred
                x0_pred = noise_scheduler.step(
                    noise_pred, timesteps, noise_center_grid
                ).pred_original_sample  # (B,Z,H,W)

                # BCE targets in {0,1} (since your x0 is {-1,+1})
                logits = bce_logit_scale * x0_pred
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    (
                        center_occupancy_grid
                        > (0 if config["grid_binary_range"] == "neg1-1" else 0.5)
                    ).float(),
                )

                total_loss = total_loss + bce_weight * bce_loss
                sum_bce_loss += bce_loss.item()
            sum_total_loss += total_loss.item()
            if train:
                total_loss.backward()
                optimizer.step()

            pbar.set_postfix({"loss": f"{ddpm_loss:.4f}/{bce_loss:.4f}"})

    avg_ddpm_loss = sum_ddpm_loss / len(dataloader)
    avg_bce_loss = sum_bce_loss / len(dataloader)
    avg_total_loss = sum_total_loss / len(dataloader)
    return avg_ddpm_loss, avg_bce_loss, avg_total_loss, global_step


def makeDiTModel(config):
    return DiT(
        input_size=config["scaled_image_size"][0],  # H
        patch_size=config["patch_size"],
        in_channels=config["depth_voxel_grid_bins"],  # depth bins per patch
        hidden_size=config["latent_dim"],
        depth=config["num_transformer_blocks"],  # n DiT blocks
        num_heads=config["num_attention_heads"],  # DiT block param
        mlp_ratio=4.0,  # DiT block param
        vae_feature_dim=(16 * 2 * 60 * 60 if not config["use_global_avg_pool"] else 16),
        class_dropout_prob=0.1,
        learn_sigma=config["learn_sigma"],
    ).to(config["device"])


def makeOptimizer(model: DiT, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["dit_lr"],
        weight_decay=config["dit_decay"],
    )


def tblogHparam(config, writer: SummaryWriter, metrics: dict):
    # log all config items as hparams
    filterd_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)):
            filterd_config[k] = v
    writer.add_hparams(filterd_config, {})

    filterd_config = {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, str, bool, torch.Tensor))
    }
    writer.add_hparams(filterd_config, metrics)


def makeDataset(
    config,
    plot_dir: str = None,
):
    """
    return train and val dataloaders
    """
    dataset = MANDataset(
        scene_ids=config["scene_ids"],
        data_file=config["data_file"],
        device=config["device"],
        radar_channel=config["radar_channel"],
        camera_channel=config["camera_channel"],
        double_flip_images=False,
        coord_only=False,
        visualize_uvz=False,  # plotting, slow
        scaled_image_size=config["scaled_image_size"],  # 512 1024 dead
        n_p=config["num_points"],
        point_only=False,  # False for dit training
        max_depth=config["max_voxel_grid_depth"],
        depth_bins=config["depth_voxel_grid_bins"],
        wan_vae=True,  # for vae-based training
        wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
        viz_dir=plot_dir,
        grid_binary_range=config["grid_binary_range"],  # "0-1" or "neg1-1"
        keep_frames=config["num_input_frames"],
    )
    assert (
        len(dataset) == config["num_input_frames"]
    ), f"Dataset length {len(dataset)} does not match num_input_frames {config['num_input_frames']}"
    return dataset


def splitDataset(dataset: Dataset, split: float = 0.8):

    val_split = int(split * len(dataset))
    train_dataset = Subset(dataset, range(0, val_split))
    val_dataset = Subset(dataset, range(val_split, len(dataset)))
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def makeDataloaders(
    dataset: Dataset,
    config,
    is_train: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=is_train,
        num_workers=config["num_workers"],
    )
