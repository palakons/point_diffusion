"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

import sys
import textwrap


import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch3d.loss import chamfer_distance   as  pt3d_chamfer_distance
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
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
    get_runid,
    query_gpu_stats,
    makeDataset,
    splitDataset,
    makeDataloaders,
    makeOptimizer,
    tblogHparam,
)
from torch.utils.tensorboard import SummaryWriter
from mono_adaln0_import import (
    save_checkpoint,
    load_checkpoint,
    train_eval_batch,
    makeDiTSetModel,DiTSet,SimpleDDPM
)


def visualize_xyz_comparison(
    frame_token,
    original_xyz,
    reconstructed_xyz,
    title="",
    save_dir="/home/palakons/D-PCC/output/plots",
    plotlims= {"x": (0, 1980), "y": (0, 943), "z": (0, 250)},
    marker_config={
        "original": {"color": "blue", "marker": "o", "size": 10, "alpha": 0.6},
        "reconstructed": {"color": "red", "marker": "x", "size": 10, "alpha": 0.6},
    },
    fig_size=( 16,9), #width, height in inches
    device="cpu",
):
    """
    Visualize original vs reconstructed XYZ point clouds.

    Args:
        frame_token: str - unique identifier for the data sample
        original_xyz: (N, 3) tensor - [x, y, z]
        reconstructed_xyz: (M, 3) tensor - [x, y, z]
        save_dir: str - directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    orig_np = (
        original_xyz.cpu().numpy()
        if torch.is_tensor(original_xyz)
        else original_xyz       
    )
    recon_np = (
        reconstructed_xyz.cpu().numpy()
        if torch.is_tensor(reconstructed_xyz)
        else reconstructed_xyz
    )

    fig = plt.figure(figsize=fig_size)

    # Original point cloud views
    ax1 = fig.add_subplot(2, 2, 1)

    # plot actualy and predict in the same plot with different colors
    ax1.scatter(
        orig_np[:, 0],
        orig_np[:, 1],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax1.scatter(
        recon_np[:, 0],
        recon_np[:, 1],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    ax1.legend(loc="upper right")  # is slow
    
    cd_allpt= pt3d_chamfer_distance(torch.tensor(orig_np[None, :, :], device=device), torch.tensor(recon_np[None, :, :], device=device))[0].item()
    cd_3pt= pt3d_chamfer_distance(torch.tensor(orig_np[None, :, :3], device=device), torch.tensor(recon_np[None, :, :3], device=device))[0].item()
    ax1.set_title(
        f"XYZ View (n=ori{len(orig_np)}, recon{len(recon_np)}) CD all dims:{cd_allpt:.2f} CD xyz:{cd_3pt:.2f}"
    )
    ax1.set_aspect("equal", adjustable="box")
    if plotlims is not None:
        ax1.set_xlim(plotlims["x"])
        ax1.set_ylim(plotlims["y"])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(
        orig_np[:, 0],
        orig_np[:, 2],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax2.scatter(
        recon_np[:, 0],
        recon_np[:, 2],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    ax2.legend(loc="upper right")
    ax2.set_title("XZ View")
    if plotlims is not None:
        ax2.set_xlim(plotlims["x"])
        ax2.set_ylim(plotlims["z"])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(
        orig_np[:, 2],
        orig_np[:, 1],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax3.scatter(
        recon_np[:, 2],
        recon_np[:, 1],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    ax3.legend(loc="upper right")
    ax3.set_title("YZ View")
    if plotlims is not None:
        ax3.set_ylim(plotlims["y"])
        ax3.set_xlim(plotlims["z"])
    ax3.set_ylabel("Y")
    ax3.set_xlabel("Z")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(
        orig_np[:, 0],
        orig_np[:, 2],
        orig_np[:, 1],
        color=marker_config["original"]["color"],
        label="Original",
        s=marker_config["original"]["size"],
        alpha=marker_config["original"]["alpha"],
        marker=marker_config["original"]["marker"],
        rasterized=True,
    )
    ax4.scatter(
        recon_np[:, 0],
        recon_np[:, 2],
        recon_np[:, 1],
        color=marker_config["reconstructed"]["color"],
        label="Reconstructed",
        s=marker_config["reconstructed"]["size"],
        alpha=marker_config["reconstructed"]["alpha"],
        marker=marker_config["reconstructed"]["marker"],
        rasterized=True,
    )
    ax4.legend(loc="upper right")

    ax4.set_title(f"3D View")
    if plotlims is not None:
        ax4.set_xlim(plotlims["x"])
        ax4.set_zlim(plotlims["y"])
        ax4.set_ylim(plotlims["z"])
    ax4.set_xlabel("X")
    ax4.set_zlabel("Y")
    ax4.set_ylabel("Z")

    if title:
        # make sure wrapt he line to 80 characters
        wrapped_title = "\n".join(textwrap.wrap(title, width=120))
        plt.suptitle(wrapped_title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(
        save_dir,
        f"{frame_token}.jpg",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cd_allpt, save_path
def sample_mono_ddpm(wan_vae_latent, ddpm, noise_scheduler: DDPMScheduler, config,seed=42):
    #wan_vae_latent: (B,16,2,60,104)

    ddpm.eval()
    with torch.no_grad():
            
        g = torch.Generator(device=config["device"])
        g.manual_seed(seed)

        wan_vae_latent = wan_vae_latent  if not config["zero_conditioning"] else torch.zeros_like(wan_vae_latent)
        wan_vae_latent = wan_vae_latent.to(config["device"])
        if not config["use_global_avg_pool"]:
            wan_vae_latent =wan_vae_latent.flatten(1) #(B,16*2*60*104)

        B,N,T = wan_vae_latent.shape[0], config["num_points"], 3
        x = torch.randn(B, N, T, device=config["device"], generator=g)  #(B,N,3)
        # print(f"Sampling DDPM with wan_vae_latent shape {wan_vae_latent.shape} and initial noise shape {x.shape}") #torch.Size([1, 199680]) and initial noise shape torch.Size([1, 800, 3]) 

        x0s = {}
        xts = {}
        noise_scheduler.set_timesteps(config["num_inference_steps"],device=config["device"])
        timesteps = noise_scheduler.timesteps.flip(0)
        for t in tqdm(
            timesteps,
            desc="Sampling DDPM",
            total=len(timesteps),
            leave=False,
        ):
            t_batch = torch.full((B,), t, device=config["device"], dtype=torch.long)
            # print("x t c",x.shape,t_batch.shape,wan_vae_latent.shape)
            if isinstance(ddpm, DiTSet):
                output = ddpm(x, t_batch, wan_vae_latent)
            elif isinstance(ddpm, SimpleDDPM):
                output = ddpm(x, t_batch    )
            else:
                raise ValueError("Unknown model type in sampling")

            eps = torch.chunk(output, 2, dim=-1)[0] if config["learn_sigma"] else output

            step_out = noise_scheduler.step(eps, t, x)
            x = step_out.prev_sample
            x0s[t.item()] = step_out.pred_original_sample.detach().cpu()
            xts[t.item()] = x.detach().cpu()
    return x0s, xts
def makeLR_Scheduler(optimizer, config):
    if config["lr_scheduler"]=='constant': #constant LR
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif config["lr_scheduler"]=='step': #StepLR
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {config['lr_scheduler']}. Supported types: 'constant', 'step', 'cosine'.")

        
def train_mono_ddpm( config, checkpoint_dir,  tb_dir, plot_dir
):
    train_dataset, val_dataset = splitDataset(makeDataset(
        config,
    ), split=0.5)
    data_mean = train_dataset.dataset.data_mean.to(config["device"])
    data_std = train_dataset.dataset.data_std.to(config["device"])

    train_dataloader = makeDataloaders(
        train_dataset,
        config,
        is_train=True,
    )
    val_dataloader = makeDataloaders(
        val_dataset,
        config,
        is_train=False,
    )
    #add time to exp name
    writer = SummaryWriter(tb_dir)
    # ddpm = makeDiTSetModel(config).to(config["device"])
    ddpm = SimpleDDPM(in_channels=3, hidden_dim=config["latent_dim"], depth=config["num_transformer_blocks"]).to(config["device"])
    optimizer = makeOptimizer(ddpm, config)
    lr_scheduler = makeLR_Scheduler(optimizer, config)

    if True:
        #use torchviz top plot to visualize the model graph and save it to tb
        if isinstance(ddpm, DiTSet):
            writer.add_graph(ddpm, (torch.randn(1, config["num_points"], 3, device=config["device"]), torch.tensor([0], device=config["device"]), torch.randn(1, 16*2*60*104, device=config["device"])))
        elif isinstance(ddpm, SimpleDDPM):
            writer.add_graph(ddpm, (torch.randn(1, config["num_points"], 3, device=config["device"]), torch.tensor([0], device=config["device"])))
        else:
            print("Unknown model type for graph visualization")


    config["node_name"] = os.uname().nodename
    config["gpu_name"] = torch.cuda.get_device_name(0)

    start_epoch, _ = load_checkpoint(ddpm, optimizer, config, lr_scheduler)   
    
    num_epochs = config["dit_epochs"]
    epoch_bar = trange(
        start_epoch,
        num_epochs,
        desc="Epochs",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    )

    best_epoch_train_ddpm_loss = float("inf")
    best_epoch = start_epoch + 1
    saved_epoch = start_epoch + 1
    global_step = 0
    for epoch in epoch_bar:

        avg_ddpm_train_loss,  global_step = (
            train_eval_batch(
                ddpm,
                train_dataloader,
                optimizer,
                lr_scheduler,
                config,
                global_step,
                train=True,
                writer=writer,
                data_mean   = data_mean,
                data_std    = data_std,
            )
        )
        writer.add_scalar("Loss/train/ddpm", avg_ddpm_train_loss, epoch )   

        if (epoch + 1) % config["eval_every"] == 0:
            avg_val_ddpm_loss,  _ = (
                train_eval_batch(
                    ddpm,
                    val_dataloader,
                    None,
                    None,
                    config,
                    global_step,
                    train=False,
                    writer=writer,
                data_mean   = data_mean,
                data_std    = data_std,

                )
            )
            writer.add_scalar("Loss/val/ddpm", avg_val_ddpm_loss, epoch )
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch )

        if (epoch +1) % config["plot_every"]==0:

            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_train_timesteps"],
                beta_schedule="linear",
                # beta_schedule="squaredcos_cap_v2",
                clip_sample=False,  # problem
                clip_sample_range=1,
                prediction_type="epsilon",
    )
            for i, batch in enumerate(train_dataloader):
                pointcloud = batch["filtered_radar_data"].to(config["device"])[:,:,:3]  # (B,N,3)                

                batch_size = pointcloud.shape[0]
                if config["zero_conditioning"]:
                    wan_vae_latent = torch.zeros(
                        (batch_size, 16, 2, 60, 104),
                        device=config["device"],
                    )  # [4, 16, 2, 60, 104]
                else:
                    wan_vae_latent = batch["wan_vae_latent"].to(
                        config["device"]
                    )  # [4, 16, 2, 60, 104]            

                if config["use_global_avg_pool"]:
                    condition = wan_vae_latent  # , because will be avg pooled in model
                else:
                    condition = wan_vae_latent.flatten(1)

                x0s, xts= sample_mono_ddpm(wan_vae_latent, ddpm, 
                noise_scheduler, config,seed=42)

                # Plotting
                x0normed = x0s[0]  # Original sample at t=0
                x0 = x0normed * data_std.cpu().numpy()[:3] + data_mean.cpu().numpy()[:3]  # Denormalize
                # print(f"Plotting DDPM samples at epoch {epoch} with original pointcloud shape {pointcloud.shape} and x0 shape {x0.shape}")
                # print("min max mean std of original pointcloud:", pointcloud.min().item(), pointcloud.max().item(), pointcloud.mean().item(), pointcloud.std().item())
                # print("min max mean std of x0:", x0.min(), x0.max(), x0.mean(), x0.std())
                x0 = x0.to(config["device"])

                # print("shapea dn dev x0", x0.shape, x0.device, type(x0))
                # print("pointcloud",pointcloud.shape, pointcloud.device,type(pointcloud))
                # shapea dn dev x0 torch.Size([1, 800, 3]) cuda:0 <class 'torch.Tensor'>
                # pointcloud torch.Size([1, 800, 3]) cuda:0 <class 'torch.Tensor'>
                cds = pt3d_chamfer_distance(
                        x0, pointcloud, batch_reduction=None)[0].item()
                writer.add_scalar("CD/train/avg", np.mean(cds), epoch )

                #find min among the  last dim, dim=2
                minax = torch.amin(pointcloud, dim=[0,1]).cpu().numpy()  #(B,3)
                maxax = torch.amax(pointcloud, dim=[0,1]).cpu().numpy()  #(B,3)

                minax_pred = torch.amin(x0, dim=[0,1]).cpu().numpy()  #(B,3)
                maxax_pred = torch.amax(x0, dim=[0,1]).cpu().numpy()  #(B,3)

                minax = np.minimum(minax, minax_pred)*0.9
                maxax = np.maximum(maxax, maxax_pred)*1.1

                # print("pointcloud min max per batch:", minax, maxax) # [  0.       -75.281975 -14.384188] [179.88828  135.78143   24.116446]
                for i_batch in range(batch_size):
                    cd, save_path = visualize_xyz_comparison(
                    frame_token=f"train_epoch_{epoch}_{i_batch}",
                    original_xyz=pointcloud[i_batch].cpu().numpy(),  
                    reconstructed_xyz=x0[i_batch].detach().cpu().numpy(), 
                    title=f"train_{config['exp_name']}_epoch_{epoch}_{i_batch}",
                    save_dir=plot_dir,
                    plotlims= {"x": (minax[0], maxax[0]), "y":( minax[1],maxax[1]),"z":(minax[2],maxax[2])},
                    device=config["device"],
                    fig_size=( 16,9), #width, height in inches
                )                
        if (epoch + 1) % config["gpu_log_every"] == 0:
            stats = query_gpu_stats(gpu_index=0)
            if stats is not None:
                writer.add_scalar("gpu/utilization_pct", stats["util_gpu"], epoch )
                writer.add_scalar("gpu/power_w", stats["power_w"], epoch )
                writer.add_scalar(
                    "gpu/memory_used_mib", stats["mem_used_mib"], epoch 
                )
                writer.add_scalar(
                    "gpu/memory_total_mib", stats["mem_total_mib"], epoch 
                )

        if (epoch + 1) % config[
            "save_every"
        ] == 0 or avg_ddpm_train_loss < best_epoch_train_ddpm_loss:
            if (epoch + 1) % config["save_every"] == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"epoch_{epoch:06d}.pt"
                )
                prev_path = os.path.join(
                    checkpoint_dir,  f"epoch_{saved_epoch:06d}.pt"
                )
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                saved_epoch = epoch 
            if avg_ddpm_train_loss < best_epoch_train_ddpm_loss:
                best_epoch_train_ddpm_loss = avg_ddpm_train_loss
                # Remove previous best checkpoint
                prev_path = os.path.join(
                    checkpoint_dir,  f"best_{best_epoch:06d}.pt"
                )
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                best_epoch = epoch 
                checkpoint_path = os.path.join(
                    checkpoint_dir,  f"best_{epoch:06d}.pt"
                )

            save_checkpoint(
                ddpm,
                optimizer,
                epoch ,
                {
                    "ddpm_loss": avg_ddpm_train_loss,
                },
                checkpoint_path,
                config,
                lr_scheduler,
            )

        epoch_bar.set_description(
            f"l:{avg_ddpm_train_loss:.4f}@{best_epoch}"
        )
    tblogHparam(config, writer, {"avg_ddpm_train_loss": avg_ddpm_train_loss})

    writer.close()

    save_checkpoint(
        ddpm,
        optimizer,
        epoch,
        avg_ddpm_train_loss,
        f"{checkpoint_dir}/final_checkpoint.pt",
        config,
        lr_scheduler,
    )

def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    tb_key = f"mono_ddpm"
    working_dir = f"{config['exp_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plot_dir = f"/data/palakons/{tb_key}/plots/{working_dir}"
    checkpoint_dir = f"/data/palakons/{tb_key}/checkpoints/{working_dir}"
    tb_log_dir = f"/home/palakons/logs/tb_log/{tb_key}/{working_dir}"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)


    print("Starting training...")
    ddpm = train_mono_ddpm(
        config,
        checkpoint_dir=checkpoint_dir,
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
