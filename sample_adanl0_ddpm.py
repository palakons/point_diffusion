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
    chamfer_distance,
    get_runid,
)
from torch.utils.tensorboard import SummaryWriter
from adaln0_import import (
    save_checkpoint,
    load_checkpoint,
    get_center_crop_latent_and_grid,
    train_eval_batch,
    find_checkpoint_path,
    makeDiTModel,
    makeOptimizer,
    makeDataset,
    splitDataset,
    makeDataloaders,
)

# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,104)


@torch.no_grad()
def sample_adaln0(
    center_wan_vae_latent: torch.Tensor,
    model: DiT,
    config,
):
    """
    DDPM sampler for voxel occupancy grids conditioned on WAN VAE latent.

    Args:
        wan_vae_latent: (B,16,2,60,60)
        model: DiT model

    Returns:
        occupancy: (B,Z,H,W)
    """
    g = torch.Generator(device=config["device"])
    g.manual_seed(config["seed"])

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        # beta_schedule="squaredcos_cap_v2",
        clip_sample=True,  # problem
        clip_sample_range=1,
        prediction_type="epsilon",
    )

    center_wan_vae_latent = center_wan_vae_latent.to(config["device"])
    B = center_wan_vae_latent.shape[0]

    Z, H = (config["depth_voxel_grid_bins"], config["scaled_image_size"][0])

    # start from pure noise in voxel space
    x = torch.randn((B, Z, H, H), device=config["device"], generator=g)

    if config["zero_conditioning"]:
        center_wan_vae_latent = torch.zeros(
            (B, 16, 2, 60, 60),
            device=config["device"],
        )  # [4, 16, 2, 60, 10604]

    if config["use_global_avg_pool"]:
        cond = center_wan_vae_latent
    else:
        cond = center_wan_vae_latent.flatten(1)  # (B,16*2*60*60)

    # eval mode for deterministic dropout etc.
    model.eval()

    xs = {}
    noise_scheduler.set_timesteps(
        config["num_inference_steps"], device=config["device"]
    )
    print("train timesteps:", noise_scheduler.num_train_timesteps)
    print("len infer tiemesteps:", len(noise_scheduler.timesteps))
    assert (
        len(noise_scheduler.timesteps) == config["num_inference_steps"]
    ), f"Mismatch in inference steps {len(noise_scheduler.timesteps)} vs {config['num_inference_steps']}"
    for t in tqdm(noise_scheduler.timesteps, desc="Sampling", leave=False):
        t_batch = torch.full((B,), int(t), device=config["device"], dtype=torch.long)

        output = model(x, t_batch, cond)  # (B,Z,H,H)
        if config["learn_sigma"]:
            eps, log_sigma = torch.chunk(output, 2, dim=1)
            # combine eps and log_sigma into a single tensor for noise scheduler
        else:
            eps = output

        # one DDPM step
        x = noise_scheduler.step(eps, int(t), x).prev_sample
        xs.update({int(t): x.detach().cpu()})

    return xs


def sample_vae_voxel_ddpm(
    dataset,
    config,
    run_id=None,
    tb_dir=None,
    plot_dir=None,
):
    """
    Sample DDPM model, all data in dataset

    Args:
        dataset: MANDataset dataset
        config: dict with training config
        run_id: unique run identifier
        tb_dir: tensorboard directory
        plot_dir: directory to save sampled results
    """

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
    ddpm.eval()

    dataloader = makeDataloaders(
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
    _, _ = load_checkpoint(ddpm, optimizer, config)

    for batch in tqdm(dataloader, desc="Sampling DDPM", leave=True):  # stays
        bs = batch["wan_vae_latent"].shape[0]
        frame_tokens = batch["frame_token"]
        center_wan_vae_latent, center_actual_occupancy_grid = (
            get_center_crop_latent_and_grid(
                batch["wan_vae_latent"].to(config["device"]),
                batch["occupancy_grid"].to(config["device"]),
            )
        )
        batch_size = center_wan_vae_latent.shape[0]
        with torch.no_grad():

            # print("center_wan_vae_latent shape:", center_wan_vae_latent.shape)  # [2, 16, 2, 60, 60]
            sampled_center_occupancy_grids = sample_adaln0(
                center_wan_vae_latent,
                model=ddpm,
                config=config,
            )
            # print(
            #     "shape of sampled occupancy grid:", sampled_occupancy_grid.shape
            # )  # [2,  128, 128, 256]
            for time_step in tqdm(sampled_center_occupancy_grids, leave=False):
                for i_batch in trange(
                    bs, desc="Each iteM in batch", leave=False
                ):  # leave=True (default behavior): The progress bar stays
                    print("Time step:", time_step)
                    reconstructed_uvz = (
                        dataloader.dataset.dataset.dataset.occupancy_grid_to_uvz(
                            sampled_center_occupancy_grids[time_step][i_batch, :, :, :],
                            original_image_size=config["original_image_size"],
                            max_depth=config["max_voxel_grid_depth"],
                            threshold=(
                                0.0 if config["grid_binary_range"] == "neg1-1" else 0.5
                            ),
                        )
                    )
                    actual_uvz = (
                        dataloader.dataset.dataset.dataset.occupancy_grid_to_uvz(
                            center_actual_occupancy_grid[i_batch, :, :, :],
                            original_image_size=config["original_image_size"],
                            max_depth=config["max_voxel_grid_depth"],
                            threshold=(
                                0.0 if config["grid_binary_range"] == "neg1-1" else 0.5
                            ),
                        )
                    )
                    # print(
                    #     "min max of reconstructed uvz each dim:",
                    #     reconstructed_uvz.amin(dim=0),
                    #     reconstructed_uvz.amax(dim=0),
                    # )  # min max of reconstructed uvz: 0.0 1852.734375
                    # print(
                    #     "min max of actual uvz:",
                    #     actual_uvz.amin(dim=0),
                    #     actual_uvz.amax(dim=0),
                    # )  # min max of actual uvz: 4.455341339111328 176.2727508544922
                    # print(
                    #     f'Chamfer Distance: {chamfer_distance(actual_uvz.unsqueeze(0).to(config["device"]),reconstructed_uvz.unsqueeze(0).to(config["device"]),).item():.4f}'
                    # )

                    dataloader.dataset.dataset.dataset.visualize_uvz_comparison(
                        frame_token=frame_tokens[i_batch]
                        + f"_grid_{run_id}_step_{time_step:03d}",
                        title=frame_tokens[i_batch]
                        + f"_grid_{run_id}_step_{time_step:03d}",
                        original_uvz=actual_uvz,
                        reconstructed_uvz=reconstructed_uvz,
                        save_dir=plot_dir,
                        marker_config={
                            "original": {
                                "color": "blue",
                                "marker": "o",
                                "size": 10,
                                "alpha": 0.6,
                            },
                            "reconstructed": {
                                "color": "red",
                                "marker": "x",
                                "size": 10,
                                "alpha": 0.6,
                            },
                        },
                        fig_size=(16, 9),  # width, height in inches
                    )
                    # print("saved uvz comparison plot.", plot_dir)


def main():
    args = parse_args()
    config = vars(args)
    set_seed(config["seed"])

    torch.set_grad_enabled(False)

    plot_dir = "/data/palakons/man_vaevoxelmetadit/plots"
    checkpoint_dir = "/data/palakons/man_vaevoxelmetadit/checkpoints"
    tb_log_dir = "/home/palakons/logs/tb_log/vaevoxelmetadit"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"Setting random seed: {args.seed}")
    if config["dit_checkpoint"] == "":
        config["dit_checkpoint"] = find_checkpoint_path(checkpoint_dir, config)
        if config["dit_checkpoint"] == "":
            raise ValueError("Checkpoint path could not be found automatically.")
        print("Auto-discover checkpoint:", config["dit_checkpoint"])

    # Load dataset
    dataset_point = makeDataset(
        config,
        plot_dir=plot_dir,
    )

    train_dataset, val_dataset = splitDataset(dataset_point, split=0.5)

    print("Starting sampling...")
    runid = get_runid(config)

    sample_vae_voxel_ddpm(
        train_dataset,
        config,
        run_id=runid + "_train",
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )
    sample_vae_voxel_ddpm(
        val_dataset,
        config,
        run_id=runid + "_val",
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
