"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

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
import os, time
import argparse
from torch.utils.data import Subset
import torch.nn.functional as F
from dit_ddpm_class import (
    PretrainedPointNeXtEncoderPointAE,
    parse_args,
    set_seed,
    TransformerDenoiser,
    get_pointnextdit_runid,
    get_pointnext_runid,
    get_runid,
    makeDataset,
    splitDataset,
    makeDataloaders,
    makeOptimizer,
    plot_ae,
    chamfer_with_attr_loss,
)
from torch.utils.tensorboard import SummaryWriter
from ae_adaln0_import import (
    save_checkpoint,
    load_checkpoint,
    loadAECheckpoint,
    makeAEDiTModel,
    makeDDPMScheduler,
    find_checkpoint_path,
)

# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,104)


@torch.no_grad()
def ae_sample_adaln0(
    wan_vae_latent: torch.Tensor,
    model: AEDiT,
    autoencoder: PretrainedPointNeXtEncoderPointAE,
    noise_scheduler: DDPMScheduler,
    config,
):
    """
    DDPM sampler for point AE conditioned on WAN VAE latent.

    Args:
        wan_vae_latent: (B,16,2,60,60)
        model: AEDiT model
        autoencoder: pretrained autoencoder model
        noise_scheduler: DDPM noise scheduler
        config: dict with training config

    Returns:
        tokens: dict of sampled occupancy grids at each timestep
    """
    g = torch.Generator(device=config["device"])
    g.manual_seed(config["seed"])

    wan_vae_latent = wan_vae_latent.to(config["device"])
    B = wan_vae_latent.shape[0]

    if config["zero_conditioning"]:
        wan_vae_latent = torch.zeros_like(wan_vae_latent)

    cond = (
        wan_vae_latent if config["use_global_avg_pool"] else wan_vae_latent.flatten(1)
    )

    N, T = config["latent_seq_length"], config["latent_dim"]
    x = torch.randn((B, N, T), device=config["device"], generator=g)
    print("Initial noise x shape:", x.shape)

    # eval mode for deterministic dropout etc.
    model.eval()
    autoencoder.eval()

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

        output = model(x, t_batch, vae_feature=cond)  # (B, N, T) or (B, N, 2T)
        eps = torch.chunk(output, 2, dim=-1)[0] if config["learn_sigma"] else output

        # one DDPM step
        step_out = noise_scheduler.step(eps, int(t), x)
        x = step_out.prev_sample  # advanced to prev sample
        xs[int(t)] = step_out.pred_original_sample.detach().cpu()  # x0 estimate

    return xs


def calculate_ae_ddpm_7d_loss(
    gt,
    unnorm_dit_predicted_latent,
    predicted_attributes,
    autoencoder,
    device,
    w_xyz=1.0,
    w_velocity=0.10,
    w_rcs=0.01,
):
    """
    Calculate AE-DDPM loss in 7D space
            v-------------------1------------------------v
    GT attributes -(encode)-> latent -(decode)-> predicted attirbutes
            ^                   ^
            +--------------3----|------------+
                    v-----2-----+            v
    vae -(DDDM)-> latent -(decode)-> predicted attributes


    - loss 1: AE attribute reconstruction loss in 7D space
    - loss 2: DDPM latent loss in latent space
    - loss 3: DDPM attribute loss in 7D space
    Args:
        gt: ground truth radar points with 7 attributes [B, N, 7]
        dit_predicted_latent: predicted latent from DDPM [B, N, latent_dim]
        predicted_attributes: predicted radar points with 7 attributes from DDPM latent [B, N, 7]
        autoencoder: pretrained autoencoder model
        w_xyz: weight for xyz in chamfer loss
        w_velocity: weight for velocity in chamfer loss
        w_rcs: weight for rcs in chamfer loss
    """
    ae_predicted_radar_7d, confidence, gt_point_latent = autoencoder(gt)
    loss1 = chamfer_with_attr_loss(
        ae_predicted_radar_7d,
        gt,
        device,
        w_xyz=w_xyz,
        w_velocity=w_velocity,
        w_rcs=w_rcs,
    )
    print("shape of unnorm_dit_predicted_latent:", unnorm_dit_predicted_latent.shape)
    # count nans in unnorm_dit_predicted_latent
    print(
        "nans in unnorm_dit_predicted_latent:",
        torch.isnan(unnorm_dit_predicted_latent).sum().item(),
        "which is out of total ",
        unnorm_dit_predicted_latent.numel(),
        "Or fraction:",
        torch.isnan(unnorm_dit_predicted_latent).sum().item()
        / unnorm_dit_predicted_latent.numel(),
    )
    print("shape of gt_point_latent:", gt_point_latent.shape)
    loss2 = F.mse_loss(unnorm_dit_predicted_latent, gt_point_latent)
    # print mean/std of dit_predicted_latent and gt_point_latent
    print(
        "dit_predicted_latent mean/std:",
        torch.mean(unnorm_dit_predicted_latent).item(),
        torch.std(unnorm_dit_predicted_latent).item(),
    )
    print(
        "gt_point_latent mean/std:",
        torch.mean(gt_point_latent).item(),
        torch.std(gt_point_latent).item(),
    )
    loss3 = chamfer_with_attr_loss(
        predicted_attributes,
        gt,
        device,
        w_xyz=w_xyz,
        w_velocity=w_velocity,
        w_rcs=w_rcs,
    )
    return loss1, loss2, loss3


def sample_pointae_ddpm(
    ddpm,
    autoencoder,
    noise_scheduler,
    optimizer,
    dataset,
    config,
    run_id=None,
    tb_dir=None,
    plot_dir=None,
):
    """
    Sample DDPM model, all data in dataset

    Args:
        ddpm: trained DDPM model
        autoencoder: pretrained autoencoder model
        noise_scheduler: DDPM noise scheduler
        optimizer: optimizer for DDPM model
        dataset: MANDataset dataset
        config: dict with training config
        run_id: unique run identifier
        tb_dir: tensorboard directory
        plot_dir: directory to save sampled results
    """
    dataloader = makeDataloaders(
        dataset,
        config,
        is_train=False,
    )

    for batch in tqdm(dataloader, desc="Sampling DDPM", leave=True):  # stays
        bs = batch["wan_vae_latent"].shape[0]
        frame_tokens = batch["frame_token"]
        actual_occupancy_grid = batch["occupancy_grid"].to(config["device"])
        wan_vae_latent = batch["wan_vae_latent"].to(config["device"])
        filtered_radar_data = batch["filtered_radar_data"].to(config["device"])

        batch_size = wan_vae_latent.shape[0]
        with torch.no_grad():

            # print("center_wan_vae_latent shape:", center_wan_vae_latent.shape)  # [2, 16, 2, 60, 60]
            sampled_latent = ae_sample_adaln0(
                wan_vae_latent=wan_vae_latent,
                model=ddpm,
                autoencoder=autoencoder,
                noise_scheduler=noise_scheduler,
                config=config,
            )
            # print(
            #     "sampled_latent[0].shape:", sampled_latent[0].shape
            # )  # (B,N,T) 16 64 748
            # print("wan_vae_latent shape:", wan_vae_latent.shape)  # [B, 16, 2, 60, 104]
            # print("frame_tokens:", frame_tokens)["..", "..."]

            # print("std of first sampled latent:", torch.std(sampled_latent[0])) #tensor(0.5169)
            tstq = tqdm(
                sorted(sampled_latent.keys())[:1],
                leave=False,
                desc="Time steps",
            )
            for time_step in tstq:  # this step takes time
                time_0 = time.time()
                time_1 = time.time()
                trr = trange(bs * 0 + 1, desc="Each in batch", leave=False)
                for (
                    i_batch
                ) in trr:  # leave=True (default behavior): The progress bar stays
                    print("Time step:", time_step)
                    time_2 = time.time()
                    time_3 = time.time()
                    latent_x0_norm = sampled_latent[time_step].to(config["device"])
                    print(
                        "mena and std of latent for decoding:",
                        torch.mean(latent_x0_norm),
                        torch.std(latent_x0_norm),
                    )
                    latent_for_decoding = (
                        latent_x0_norm * config["ae_latent_normalizing_std"]
                    )
                    print(
                        "after un-scaling, mean and std of latent for decoding:",
                        torch.mean(latent_for_decoding),
                        torch.std(latent_for_decoding),
                    )
                    predicted_radar_7d, predicted_confidence = autoencoder.decode(
                        latent_for_decoding
                    )
                    actual_radar_7d = batch["filtered_radar_data"][i_batch, :, :].to(
                        config["device"]
                    )  # (N,7)
                    time_4 = time.time()

                    time_5 = time.time()
                    dataloader.dataset.dataset.visualize_uvz_comparison(
                        frame_token=frame_tokens[i_batch][-3:]
                        + f"_grid_{run_id}_step_{time_step:03d}",
                        title=frame_tokens[i_batch]
                        + f"_grid_{run_id}_step_{time_step:03d}",
                        original_uvz=actual_radar_7d[:, :3],
                        reconstructed_uvz=predicted_radar_7d[i_batch, :, :3],
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
                        plotlims=None,
                        # {
                        #     "u": (0, config["original_image_size"][1]),
                        #     "v": (0, config["original_image_size"][0]),
                        #     "z": (0, 250),
                        # },
                        device="cpu",  # too alrge for gpu
                    )
                    time_6 = time.time()
                    trr.set_description(
                        f"recon{time_3 - time_2:.2f}/actual{time_4 - time_3:.2f}/viz{time_6 - time_5:.2f}/pad{time_1 - time_0:.2f}"
                    )
                    loss1, loss2, loss3 = calculate_ae_ddpm_7d_loss(
                        gt=actual_radar_7d.unsqueeze(0),
                        unnorm_dit_predicted_latent=latent_for_decoding[
                            i_batch, :, :
                        ].unsqueeze(0),
                        predicted_attributes=predicted_radar_7d[
                            i_batch, :, :
                        ].unsqueeze(0),
                        autoencoder=autoencoder,
                        device=config["device"],
                        w_xyz=1.0,
                        w_velocity=0.10,
                        w_rcs=0.01,
                    )
                    print("loss1 (AE attribute recon):", loss1[1])
                    print("loss2 (DDPM MSE latent loss):", loss2.item())
                    print("loss3 (DDPM attribute loss):", loss3[1])

                    if True:
                        # check ddpm eps loss, should close to 1e-3
                        ae_predicted_radar_7d, confidence, x0 = autoencoder(
                            filtered_radar_data
                        )
                        x0_norm = (x0 / config["ae_latent_normalizing_std"]).to(
                            config["device"]
                        )

                        cond = (
                            wan_vae_latent.flatten(1)
                            if not config["use_global_avg_pool"]
                            else wan_vae_latent
                        )

                        mses = {
                            "ddpm_eps_loss": [],
                            "x0_recon_loss": [],
                            "xyz_recon_loss": [],
                        }
                        for jj in range(100):

                            eps = torch.randn_like(x0_norm)

                            # sample random timesteps like training
                            t = torch.randint(
                                0,
                                noise_scheduler.config.num_train_timesteps,
                                (batch_size,),
                                device=config["device"],
                            ).long()
                            x_t = noise_scheduler.add_noise(x0_norm, eps, t)

                            eps_hat = ddpm(x_t, t, vae_feature=cond)

                            mse_loss = F.mse_loss(eps, eps_hat)
                            alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                                x_t.device
                            )  # shape: (num_train_timesteps,)
                            a = alphas_cumprod[t].view(
                                -1, 1, 1
                            )  # broadcast to (B,1,1) for (B,N,T)

                            x0_hat_norm = (
                                x_t - torch.sqrt(1.0 - a) * eps_hat
                            ) / torch.sqrt(a)

                            predicted_attributes = autoencoder.decode(
                                x0_hat_norm * config["ae_latent_normalizing_std"]
                            )[0]
                            xyz_loss = chamfer_with_attr_loss(
                                predicted_attributes,
                                filtered_radar_data,
                                config["device"],
                                w_xyz=1.0,
                                w_velocity=0.1,
                                w_rcs=0.01,
                            )[1]["xyz"]

                            # print(
                            #     "eps prediction MSE:",
                            #     mse_loss.item(),
                            #     "close to 1e-3 indicates good training?",
                            #     mse_loss.item() < 9e-3,
                            #     "x0 reconstruction error (normed):",
                            #     F.mse_loss(x0_norm, x0_hat_norm).item(),
                            # )
                            mses["ddpm_eps_loss"].append(mse_loss.item())
                            mses["x0_recon_loss"].append(
                                F.mse_loss(x0_norm, x0_hat_norm).item()
                            )
                            mses["xyz_recon_loss"].append(xyz_loss)

                        # plot ddpm eps loss and x0 recon loss
                        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
                        ax[0].plot(
                            mses["ddpm_eps_loss"], label="DDPM eps loss", color="blue"
                        )
                        ax[0].set_title(
                            f"DDPM Eps Loss over 100 samples, avg: {np.mean(mses['ddpm_eps_loss']):.6g}"
                        )
                        ax[0].set_xlabel("Sample Index")
                        ax[0].set_ylabel("MSE Loss")
                        ax[0].legend()
                        ax[1].plot(
                            mses["x0_recon_loss"], label="x0 recon loss", color="orange"
                        )
                        ax[1].set_title(
                            f"x0 Reconstruction Loss over 100 samples, avg: {np.mean(mses['x0_recon_loss']):.6g}"
                        )
                        ax[1].set_xlabel("Sample Index")
                        ax[1].set_ylabel("MSE Loss")
                        ax[1].legend()
                        ax[2].plot(
                            mses["xyz_recon_loss"],
                            label="xyz recon loss",
                            color="green",
                        )
                        ax[2].set_title(
                            f"Chamfer Distance of xyz Reconstruction over 100 samples, avg: {np.mean(mses['xyz_recon_loss']):.6g}"
                        )
                        ax[2].set_xlabel("Sample Index")
                        ax[2].set_ylabel("Chamfer Distance")
                        ax[2].legend()

                        plt.tight_layout()
                        loss_plot_path = os.path.join(
                            plot_dir, f"{run_id}_ddpm_losses.png"
                        )
                        plt.savefig(loss_plot_path)
                        plt.close()

                        # ------- interpolate between x_hat_norm  (from ddpm )and x0_norm (real), calculate chamfer xyz
                        alpha_data = {}
                        print(
                            "shapes for interpolation:",
                            x0_norm.shape,
                            latent_for_decoding.shape,
                        )
                        for alpha in np.linspace(0, 1, 5):
                            x_interp_norm = (
                                alpha * x0_norm + (1 - alpha) * latent_x0_norm
                            )
                            predicted_attributes = autoencoder.decode(
                                x_interp_norm * config["ae_latent_normalizing_std"]
                            )[0]
                            xyz_loss = chamfer_with_attr_loss(
                                predicted_attributes,
                                filtered_radar_data,
                                config["device"],
                                w_xyz=1.0,
                                w_velocity=0.1,
                                w_rcs=0.01,
                            )[1]["xyz"]
                            alpha_data[alpha] = xyz_loss

                            # plot and save comparison plot
                            print(
                                "shapes for uvz plot:",
                                actual_radar_7d.shape,
                                predicted_attributes[i_batch, :, :3].shape,
                            )
                            mean_interp, std_interp = torch.mean(
                                x_interp_norm * config["ae_latent_normalizing_std"]
                            ), torch.std(
                                x_interp_norm * config["ae_latent_normalizing_std"]
                            )
                            cd, _path = (
                                dataloader.dataset.dataset.visualize_uvz_comparison(
                                    frame_token=frame_tokens[i_batch][-3:]
                                    + f"_grid_{run_id}_step_{time_step:03d}_alpha_{alpha:.2f}",
                                    title=frame_tokens[i_batch]
                                    + f"_grid_{run_id}_step_{time_step:03d}_alpha_{alpha:.2f} ",
                                    original_uvz=actual_radar_7d[:, :3],
                                    reconstructed_uvz=predicted_attributes[
                                        i_batch, :, :3
                                    ],
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
                                    plotlims=None,
                                    # {
                                    #     "u": (0, config["original_image_size"][1]),
                                    #     "v": (0, config["original_image_size"][0]),
                                    #     "z": (0, 250),
                                    # },
                                    device="cpu",  # too alrge for gpu
                                )
                            )

                            alpha_data[alpha] = cd

                        # plot interpolation chamfer
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(
                            list(alpha_data.keys()),
                            list(alpha_data.values()),
                            marker="o",
                        )
                        ax.set_title("Chamfer Distance vs Interpolation Alpha")
                        ax.set_xlabel("Alpha (0: x0_hat, 1: x0)")
                        ax.set_ylabel("Chamfer Distance")
                        plt.tight_layout()
                        interp_plot_path = os.path.join(
                            plot_dir, f"{run_id}_ddpm_interpolation_chamfer.png"
                        )
                        plt.savefig(interp_plot_path)
                        plt.close()

                time_7 = time.time()
                tstq.set_description(
                    f"pad{time_1 - time_0:.2f}/infer{time_7 - time_1:.2f}:recon{time_3 - time_2:.2f}/actual{time_4 - time_3:.2f}/viz{time_6 - time_5:.2f}"  # pad0.00/infer2.46:recon0.00/actual0.00/viz2.46
                )  # time for all in batch

                # print("saved uvz comparison plot.", plot_dir)


def main():
    args = parse_args()
    config = vars(args)
    set_seed(config["seed"])

    torch.set_grad_enabled(False)

    exp_set = "man_pointnextmetaditddpm"
    plot_dir = f"/data/palakons/{exp_set}/plots"
    checkpoint_dir = f"/data/palakons/{exp_set}/checkpoints"
    tb_log_dir = f"/home/palakons/logs/tb_log/{exp_set}"
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
    runid = f"AEDiT_sampling_{get_pointnextdit_runid(config)}"

    ddpm = makeAEDiTModel(config)
    ddpm.eval()
    autoencoder, ae_config = loadAECheckpoint(config)
    autoencoder.eval()
    noise_scheduler = makeDDPMScheduler(config)
    optimizer = makeOptimizer(ddpm, config)

    print(
        "Total parameters:",
        sum(p.numel() for p in ddpm.parameters() if p.requires_grad),
    )
    _, _ = load_checkpoint(ddpm, optimizer, config)

    ae_run_id = f"pretrain_ae_{get_pointnext_runid(ae_config)}"
    train_cds = plot_ae(
        "train",
        train_dataset,
        autoencoder,
        config,
        plot_dir,
        ae_run_id,
        ae_config["ae_epochs"],
    )
    val_cds = plot_ae(
        "val",
        val_dataset,
        autoencoder,
        config,
        plot_dir,
        ae_run_id,
        ae_config["ae_epochs"],
    )

    # total, {
    #     "xyz": xyz_cd.item(),
    #     "velocity": vel_loss.item(),
    #     "rcs": rcs_loss.item(),
    # }
    # make 3 plot in 3 rows for xyz, velocity, rcs losses

    labels = ["Chamfer Distance", "Velocity MSE", "RCS MSE"]
    train_values = [
        [a[1]["xyz"] for a in train_cds],
        [a[1]["velocity"] for a in train_cds],
        [a[1]["rcs"] for a in train_cds],
    ]
    val_values = [
        [a[1]["xyz"] for a in val_cds],
        [a[1]["velocity"] for a in val_cds],
        [a[1]["rcs"] for a in val_cds],
    ]
    # make subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    for i in range(3):
        axs[i].plot(train_values[i], label="Train", color="blue")
        axs[i].plot(val_values[i], label="Val", color="orange")
        axs[i].set_title(labels[i])
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        # tight layout
    plt.tight_layout()

    cd_plot_path = os.path.join(plot_dir, f"{ae_run_id}_chamfer_distance.png")
    plt.savefig(cd_plot_path)
    plt.close()

    sample_pointae_ddpm(
        ddpm,
        autoencoder,
        noise_scheduler,
        optimizer,
        train_dataset,
        config,
        run_id=runid + "_train",
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )
    sample_pointae_ddpm(
        ddpm,
        autoencoder,
        noise_scheduler,
        optimizer,
        val_dataset,
        config,
        run_id=runid + "_val",
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
