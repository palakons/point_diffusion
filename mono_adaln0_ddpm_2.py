"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

import sys
import textwrap


import torch,math
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
    # ax1.legend(loc="upper right")  # is slow
    
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
    # ax2.legend(loc="upper right")
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
    # ax3.legend(loc="upper right")
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
    get_occ_grid=False,
    get_camera=True,
    get_wan_vae=True,
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
    # ddpm = SimpleDDPM(in_channels=3, hidden_dim=config["latent_dim"], depth=config["num_transformer_blocks"]).to(config["device"])

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="linear",
        # beta_schedule="squaredcos_cap_v2",
        clip_sample=False,  # problem
        clip_sample_range=1,
        prediction_type="epsilon",
    )

    denoiser_model = SimpleDenoiser(in_channels=3, hidden_dim=config["latent_dim"], depth=config["num_transformer_blocks"]).to(config["device"])
    # denoiser_model = PointNetUNet(in_channels=3, hidden=64).to(config["device"])
    ddpm = SimpleDDPM(noise_scheduler)

    optimizer = makeOptimizer(denoiser_model, config)
    lr_scheduler = makeLR_Scheduler(optimizer, config)

    if True:
            
        writer.add_graph(denoiser_model, (torch.randn(1, config["num_points"], 3).to(config["device"]), torch.tensor([0], device=config["device"])))


    config["node_name"] = os.uname().nodename
    config["gpu_name"] = torch.cuda.get_device_name(0)

    # start_epoch, _ = load_checkpoint(ddpm, optimizer, config, lr_scheduler)   
    start_epoch = 0
    
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

    x = torch.randn(config["batch_size"], config["num_points"], 3, device= config["device"])
    pointcloud = (x * data_std[:3] + data_mean[:3]).to(config["device"])

    min_loss = float('inf')
    min_loss_step = -1
    min_cd = float('inf')
    min_cd_epoch = -1
    cd = float('inf')
    for epoch in epoch_bar:
        train_losses = []
        
        for batch in tqdm(train_dataloader, desc="Train Batches", leave=False):

            # pointcloud = batch["filtered_radar_data"].to(config["device"])  # (B,N,3)


            # # DIFFUSION TRAINING (only this is trainable)
            # x = ((pointcloud - data_mean) / data_std)[:,:,:3] #dont care the extra attribute for now
            B = x.shape[0]
                

            t = torch.randint(0, noise_scheduler.num_train_timesteps, (B,), device=config["device"]).long()

            loss = ddpm.p_losses(denoiser_model, x, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss_step = global_step
                min_loss = loss.item()
            epoch_bar.set_description(f"L:{loss.item():.2e},minL:{min_loss:.2e}@{min_loss_step}, CD:{cd:.2e},minCD:{min_cd:.2e}@{min_cd_epoch}")

            writer.add_scalar("Loss/train/ddpm", loss.item(), global_step )   
            train_losses.extend([loss.item()] * B)  # assume same loss for all in batch for logging purposes
            global_step += 1

        if False and (epoch + 1) % config["eval_every"] == 0:
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

        if (1+epoch) % config["plot_every"]==0:

            with torch.no_grad():
                sample = ddpm.sample(denoiser_model, shape=(B, config["num_points"], 3), num_inference_steps= config["num_inference_steps"] , device=config["device"])
                cd = pt3d_chamfer_distance((sample * data_std[:3] + data_mean[:3])
                , ( x * data_std[:3] + data_mean[:3] ))[0].item()
                if cd < min_cd:
                    min_cd = cd
                    min_cd_epoch = epoch

                x0 = sample * data_std[:3] + data_mean[:3]  
                
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


                for i_batch in range(B):
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
        ] == 0 or np.mean(train_losses) < best_epoch_train_ddpm_loss:
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
            if np.mean(train_losses) < best_epoch_train_ddpm_loss:
                best_epoch_train_ddpm_loss = np.mean(train_losses)
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
                denoiser_model,
                optimizer,
                epoch ,
                {
                    "ddpm_loss": np.mean(train_losses),
                },
                checkpoint_path,
                config,
                lr_scheduler,
            )

        epoch_bar.set_description(
            f"l:{np.mean(train_losses):.4f}@{best_epoch}"
        )
    tblogHparam(config, writer, {"avg_ddpm_train_loss": np.mean(train_losses)})

    writer.close()

    save_checkpoint(
        denoiser_model,
        optimizer,
        epoch,
        np.mean(train_losses),
        f"{checkpoint_dir}/final_checkpoint.pt",
        config,
        lr_scheduler,
    )


def sinusoidal_embedding(t, dim):
    # t: (B,) int/float tensor
    device = t.device
    t = t.float().unsqueeze(1)  # (B,1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / half)
    args = t * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


class PointNetUNet(nn.Module):
    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        # encoder (shared MLP)
        self.enc1 = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.enc2 = nn.Sequential(nn.Linear(hidden*2, hidden*2), nn.ReLU(), nn.Linear(hidden*2, hidden*2))
        # bottleneck
        # bottleneck (g1 has size hidden, g2 has size hidden*2 -> total hidden*3)
        self.bottleneck = nn.Sequential(nn.Linear(hidden*3, hidden*4), nn.ReLU(), nn.Linear(hidden*4, hidden*2))
        # decoder (shared MLPs)
        self.dec2 = nn.Sequential(nn.Linear(hidden*2 + hidden*2, hidden*2), nn.ReLU(), nn.Linear(hidden*2, hidden))
        self.dec1 = nn.Sequential(nn.Linear(hidden + hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_channels))

    def forward(self, x, t=None):
        # x: (B, N, C)
        B, N, C = x.shape
        x_flat = x.view(B*N, C)
        f1 = self.enc1(x_flat).view(B, N, -1)             # (B,N,h)
        g1 = f1.max(dim=1).values                         # (B,h)
        g1b = g1.unsqueeze(1).expand(-1, N, -1)           # (B,N,h)
        f1cat = torch.cat([f1, g1b], dim=-1)              # (B,N,2h)

        f1cat_flat = f1cat.view(B*N, -1)
        f2 = self.enc2(f1cat_flat).view(B, N, -1)         # (B,N,2h)
        g2 = f2.max(dim=1).values                         # (B,2h)

        bott_in = torch.cat([g1, g2], dim=-1)            # (B,3h) approx
        bott = self.bottleneck(bott_in)                  # (B,2h)
        bottb = bott.unsqueeze(1).expand(-1, N, -1)      # (B,N,2h)

        dec2_in = torch.cat([f2, bottb], dim=-1).view(B*N, -1)
        d2 = self.dec2(dec2_in).view(B, N, -1)           # (B,N,h)

        dec1_in = torch.cat([d2, g1b], dim=-1).view(B*N, -1)
        out = self.dec1(dec1_in).view(B, N, C)           # (B,N,C)
        return out

class SimpleDenoiser(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, depth=2, time_emb_dim=128, time_hidden=64):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        self.time_hidden = time_hidden

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_hidden),
            nn.ReLU()
        )

        input_dim = in_channels + time_hidden
        layers = []
        for i in range(depth):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, in_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # x: (B, N, C), t: (B,)
        B, N, C = x.shape
        time_emb = sinusoidal_embedding(t, self.time_emb_dim)           # (B, time_emb_dim)
        time_feat = self.time_mlp(time_emb)                            # (B, time_hidden)
        time_feat = time_feat.unsqueeze(1).expand(-1, N, -1)           # (B, N, time_hidden)
        x_in = torch.cat([x, time_feat], dim=-1)                       # (B, N, C + time_hidden)
        x_flat = x_in.view(B * N, -1)
        out = self.net(x_flat)
        return out.view(B, N, C)

class SimpleDDPM:
    def __init__(self, noise_scheduler: DDPMScheduler = None):
        self.noise_scheduler = noise_scheduler


    def p_losses(self, model, x_start, t,  noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        pred = model(x_noisy, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, shape, num_inference_steps=50, device=None, generator=None):
        ns = self.noise_scheduler
        ns.set_timesteps(num_inference_steps)
        sample = torch.randn(shape, device=device, generator=generator)
        timesteps = ns.timesteps.flip(0)
        for t in timesteps:
            t_int = int(t.item())
            t_batch = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            model_output = model(sample, t_batch)
            step = ns.step(model_output, t_int, sample)
            sample = step.prev_sample
        return sample

def train(epochs=10, B=4, N=32, C=3, ddpm_steps=1000, lr=1e-4, model = 'mlp',sample_steps=50,device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = SimpleDenoiser(in_channels=C, hidden_dim=64, depth=2).to(device) if model == 'mlp' else PointNetUNet(in_channels=C, hidden=64).to(device)
    ddpm = SimpleDDPM(noise_scheduler)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=ddpm_steps, beta_schedule="linear")

    tt = trange(epochs, desc="train")
    x = torch.randn(B, N, C, device=device)
    min_loss = float('inf')
    min_cd = float('inf')
    cd = float('inf')
    for epoch in tt:
        # dummy batch
        t = torch.randint(0, ddpm_steps, (B,), device=device).long()
        loss = ddpm.p_losses(model, x, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
        if epoch % 100 == 0:
            with torch.no_grad():
                sample = ddpm.sample(model, shape=(B, N, C), num_inference_steps=sample_steps, device=device)
                cd = pt3d_chamfer_distance(sample, x)[0].item()
                if cd < min_cd:
                    min_cd = cd
        tt.set_description(f"L:{loss.item():.2e},minL:{min_loss:.2e}, CD:{cd:.2e},minCD:{min_cd:.2e}")

    return ddpm, model, noise_scheduler, loss.item(), min_loss, cd, min_cd

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
    train_mono_ddpm(
        config,
        checkpoint_dir=checkpoint_dir,
        tb_dir=tb_log_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
