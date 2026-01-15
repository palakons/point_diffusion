"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

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
from dit_ddpm_class import parse_args, set_seed, TransformerDenoiser
from torch.utils.tensorboard import SummaryWriter


# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])

# wan_vae_latent: (B,16,2,60,104)


class InContextConditioningTransformerDenoiser(nn.Module):
    """
    Custom transformer-based denoiser model.
    the x input is of shape (B, <seq length>, in_channels)
    no condition input
    """

    def __init__(
        self,
        in_channels,
        num_channels=256,
        num_heads=8,
        num_layers=12,
        max_seq_len=256,
    ):
        super().__init__()

        self.in_channels = in_channels
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

        # Learnable positional embeddings - THIS WAS MISSING!
        self.pos_embed_x = nn.Parameter(
            torch.randn(1, max_seq_len, num_channels) * 0.02
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

    def forward(
        self,
        x,
        timesteps,
    ):
        """
        x: (B, <seq length>, in_channels)
        timesteps: (B,) - not used in this simple implementation
        """
        batch_size, seq_len, _ = x.shape

        # Encode timesteps
        t_emb = self.timestep_embedding(
            timesteps, self.num_channels
        )  # (B, num_channels)
        t_emb = self.time_embed(t_emb)  # (B, num_channels)

        # Project inputs
        x = self.input_proj(x)  # (B, <seq length>, num_channels)

        # Add positional embeddings
        x = x + self.pos_embed_x[:, :seq_len, :]

        # Add timestep embedding to input (broadcast across sequence)
        x = x + t_emb.unsqueeze(1)  # (B, seq_len, num_channels)

        # Pass through transformer
        x_transformed = self.transformer(x)  # (B, seq_len, num_channels)

        x_transformed = self.norm(x_transformed)
        output = self.output_proj(x_transformed)  # (B, seq_len, in_channels)

        return output


class VaeDitDDPM:
    """
    this DiT model will take in the wan_vae_latent as condition and output the occupancy grid, using cross-attention blocks to condition on the wan_vae_latent

    """

    def __init__(
        self,
        num_train_timesteps=1000,
        image_size=(16, 32),
        device="cuda",
        d_model: int = 1024,
        seq_length=256,
        depth_bins: int = 10,
        C_r=1,  # radar occupancy grid channels
        patch=(2, 8, 8),
    ):
        self.device = device
        self.image_size = image_size
        self.seq_length = seq_length
        self.model = InContextConditioningTransformerDenoiser(
            in_channels=d_model,  # PointNeXt embedding dim
            num_channels=256,  # Example: number of transformer channels
            num_heads=8,  # Example: number of attention heads
            num_layers=12,  # Example: number of transformer layers
            max_seq_len=self.seq_length,
        ).to(device)

        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,  # problem
            clip_sample_range=1,
            prediction_type="epsilon",
        )

        self.d_model = d_model
        self.depth_bins = depth_bins

        self.patch = patch  # (pZ, pH, pW) choose small first
        self.C_r = C_r

        self.radar_patch = nn.Conv3d(
            in_channels=self.C_r,
            out_channels=self.d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=True,
        ).to(device)
        self.cond_proj = nn.Linear(16, d_model, bias=True).to(device)

        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(
            device
        )  # halves 60x104 → 30x52
        self.radar_unpatch = nn.ConvTranspose3d(
            in_channels=self.d_model,
            out_channels=self.C_r,
            kernel_size=self.patch,
            stride=self.patch,
            bias=True,
        ).to(device)

    # @torch.no_grad()
    # def sample(self, depth_image, clip_feature, num_inference_steps=50, seed=None):
    #     """
    #     return {pointnext_enmbedding:.., uvz:..}
    #     this function perform the "sample" operation of DDPM
    #     generate pointnext_enmbedding and uvz points from depth image and clip feature
    #     """
    #     if seed is not None:
    #         # Set seed for this sampling only
    #         torch.manual_seed(seed)
    #         if torch.cuda.is_available():
    #             torch.cuda.manual_seed(seed)

    #     self.model.eval()

    #     with torch.no_grad():

    #         # Prepare condition
    #         condition = self.prepare_condition(depth_image, clip_feature)

    #         # Set timesteps
    #         self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
    #         timesteps = self.noise_scheduler.timesteps

    #         batch_size = depth_image.shape[0]
    #         # Start from pure noise
    #         pointnext_embedding = torch.randn(
    #             batch_size, self.seq_length, 1024, device=self.device
    #         )
    #     output_embeddings = {}
    #     for t in tqdm(timesteps, desc="Sampling timesteps"):
    #         # Expand timestep to batch size
    #         t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

    #         # Predict noise
    #         noise_pred = self.model(
    #             pointnext_embedding, t_batch, encoder_hidden_states=condition
    #         )

    #         # Compute previous noisy sample x_t -> x_t-1
    #         pointnext_embedding = self.noise_scheduler.step(
    #             noise_pred, t, pointnext_embedding
    #         ).prev_sample
    #         predicted_uvz, confidence = self.autoencoder.pointcloud_decoder(
    #             pointnext_embedding
    #         )
    #         output_embeddings[t.item()] = {
    #             "pointnext_embedding": pointnext_embedding.clone(),
    #             "uvz": predicted_uvz.clone(),
    #             "confidence": confidence.clone(),
    #         }

    #     output = {"pointnext_embedding": pointnext_embedding}
    #     # Decode pointnext embedding to uvz points
    #     predicted_uvz, confidence = self.autoencoder.pointcloud_decoder(
    #         pointnext_embedding
    #     )
    #     output["uvz"] = predicted_uvz  # (B, 500, 3)
    #     output["confidence"] = confidence  # (B, 500, 1)

    #     return output, output_embeddings


def save_checkpoint(ddpm: VaeDitDDPM, optimizer, epoch, loss, checkpoint_path, config):

    checkpoint = {
        "model_state_dict": {
            "model": ddpm.model.state_dict(),
            "radar_patch": ddpm.radar_patch.state_dict(),
            "radar_unpatch": ddpm.radar_unpatch.state_dict(),
            "pool": ddpm.pool.state_dict(),
            "cond_proj": ddpm.cond_proj.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch_loss": loss,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def train_eval_batch(
    ddpm: VaeDitDDPM, dataloader: DataLoader, optimizer, config, train=True
):
    if train:
        ddpm.model.train()
        ddpm.radar_patch.train()
        ddpm.radar_unpatch.train()
        ddpm.pool.train()
        ddpm.cond_proj.train()
    else:
        ddpm.model.eval()
        ddpm.radar_patch.eval()
        ddpm.radar_unpatch.eval()
        ddpm.pool.eval()
        ddpm.cond_proj.eval()
    pbar = tqdm(dataloader, leave=False, desc="Train Batch" if train else "Eval Batch")
    sum_ddpm_loss = 0.0
    for i_batch, batch in enumerate(pbar):
        with torch.no_grad() if not train else torch.enable_grad():
            if train:
                optimizer.zero_grad()
            else:
                if i_batch == 0:
                    # Sample the first item

                    # Process the first item as needed
                    print(f"Eval Batch")
                    # TODO: Handle eval batch processing as needed, implement .sample()

            wan_vae_latent = batch["wan_vae_latent"].to(config["device"])
            occupancy_grid = batch["occupancy_grid"].to(config["device"]).unsqueeze(1)
            batch_size = wan_vae_latent.shape[0]

            # Make noisy grid
            timesteps = torch.randint(
                0,
                ddpm.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=config["device"],
            ).long()

            # DIFFUSION TRAINING (only this is trainable)
            noise = torch.randn_like(occupancy_grid)
            noise_grid = ddpm.noise_scheduler.add_noise(
                occupancy_grid, noise, timesteps
            )

            # print("wan_vae_latent shape:", wan_vae_latent.shape)  #[4, 16, 2, 60, 104]
            # print("occupancy_grid shape:", occupancy_grid.shape)  #[4, 1, 8, 23, 64]
            # print("noisy_grid shape:", noisy_grid.shape)  #
            patched_noisy_grid_5d = ddpm.radar_patch(noise_grid)  # (B, C_r, Z', H', W')
            Zp, Hp, Wp = patched_noisy_grid_5d.shape[-3:]
            # print("patched_noisy_grid 5d shape:", patched_noisy_grid_5d.shape) #[4, 1024, 4, 2, 8]
            patched_noisy_grid = patched_noisy_grid_5d.flatten(2).transpose(
                1, 2
            )  # (B, N_radar_token, d_model)
            # print("patched_noisy_grid shape:", patched_noisy_grid.shape) #[4, 64, 1024]
            grid_latent_seq_length = patched_noisy_grid.shape[1]
            grid_seq_length = patched_noisy_grid.shape[1]

            condition = ddpm.pool(wan_vae_latent)  #  #no noise
            condition = condition.flatten(2).transpose(1, 2)  # (B, N_cond_token, 16)
            condition = ddpm.cond_proj(condition)  # (B, N_cond_token, d_model)

            condition_seq_length = condition.shape[1]

            # print("condition shape:", condition.shape)  # [4, N_cond_token, d_model] [4, 3120, 1024]
            tokens = torch.cat(
                [condition, patched_noisy_grid], dim=1
            )  # (B, N_cond+N_radar, d_model)
            # print("tokens shape:", tokens.shape)  # [4, total_tokens, d_model] [4, 3184, 1024]

            noise_pred = ddpm.model(tokens, timesteps)
            # print("outputs shape:", noise_pred.shape)  # [4, total_tokens, d_model]

            x_grid = (
                noise_pred[:, -grid_seq_length:, :]
                .transpose(1, 2)
                .view(batch_size, config["latent_dim"], Zp, Hp, Wp)
            )
            # print("x_grid shape:", x_grid.shape)  #
            pred_occupancy_noise_grid = ddpm.radar_unpatch(
                x_grid
            )  # # (B, C_r, Z, H, W)
            # print(
            #     "pred_occupancy_noise_grid shape:", pred_occupancy_noise_grid.shape
            # )  # [4, C_r, Z, H, W]

            ddpm_loss = F.mse_loss(noise, pred_occupancy_noise_grid)
            # loss = F.mse_loss(pred_occupancy_grid, occupancy_grid)

            if train:
                ddpm_loss.backward()
                optimizer.step()

            sum_ddpm_loss += ddpm_loss.item()

            pbar.set_postfix({"loss": f"{ddpm_loss:.4f}"})

    avg_ddpm_loss = sum_ddpm_loss / len(dataloader)
    return avg_ddpm_loss


def train_vae_voxel_ddpm(
    dataset, val_dataset, config, checkpoint_dir, run_id=None, tb_dir=None
):
    """
    Train DDPM model

    Args:
        dataset: MANDataset instance
        config: dict with training config
    """
    # Initialize model

    writer = SummaryWriter(f"{tb_dir}/{run_id}")

    patch = (2, 8, 8)
    latent_seq_length = 3120 + (config["depth_voxel_grid_bins"] // patch[0]) * (
        config["scaled_image_size"][0] // patch[1]
    ) * (config["scaled_image_size"][1] // patch[2])
    print(f"Latent sequence length: {latent_seq_length}")
    # Log hyperparameters
    writer.add_text("config/lr", str(config["dit_lr"]))
    writer.add_text("config/epochs", str(config["dit_epochs"]))
    writer.add_text("config/batch_size", str(config["batch_size"]))
    writer.add_text("config/latent_dim", str(config["latent_dim"]))
    writer.add_text(
        "config/latent_seq_length",
        str(latent_seq_length),  # hardcoded 3120 for wan_vae_latent tokens
    )
    writer.add_text("config/num_points", str(config["num_points"]))
    writer.add_text("config/dataset_size", str(len(dataset)))
    torch.cuda.reset_peak_memory_stats()

    print("actual image size:", config["original_image_size"])
    print("scaled image size:", config["scaled_image_size"])
    print("max voxel grid depth:", config["max_voxel_grid_depth"])
    print("depth voxel grid bins:", config["depth_voxel_grid_bins"])
    print("=-" * 10)
    print(
        f"resolution per depth bin (meters): {config['max_voxel_grid_depth'] / config['depth_voxel_grid_bins']:.4f} m, pixel resolution: y {config['original_image_size'][1] / config['scaled_image_size'][1]:.4f}, x {config['original_image_size'][0] / config['scaled_image_size'][0]:.4f}",
    )

    ddpm = VaeDitDDPM(
        num_train_timesteps=config["num_train_timesteps"],
        image_size=config["scaled_image_size"],  # (height, width).
        device=config["device"],
        d_model=config["latent_dim"],  # model dimension
        # seq_length=config["latent_seq_length"],  # sequence length
        seq_length=3120
        + (config["depth_voxel_grid_bins"] // patch[0])
        * (config["scaled_image_size"][0] // patch[1])
        * (config["scaled_image_size"][1] // patch[2]),
        depth_bins=config["depth_voxel_grid_bins"],
        C_r=1,  # radar occupancy grid channels
        patch=patch,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        # collate_fn=collate_fn,
        num_workers=config["num_workers"],
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        # collate_fn=collate_fn,
        num_workers=config["num_workers"],
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(ddpm.model.parameters())
        + list(ddpm.radar_patch.parameters())
        + list(ddpm.radar_unpatch.parameters())
        + list(ddpm.pool.parameters())
        + list(ddpm.cond_proj.parameters()),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    print("model parameters:")
    for module_name, module in [
        ("model", ddpm.model),
        ("radar_patch", ddpm.radar_patch),
        ("radar_unpatch", ddpm.radar_unpatch),
        ("pool", ddpm.pool),
        ("cond_proj", ddpm.cond_proj),
    ]:
        if len(list(module.parameters())) > 0:
            print(f"{module_name}: {sum(p.numel() for p in module.parameters())}")
    print(
        "Total parameters:",
        sum(
            p.numel()
            for p in list(ddpm.model.parameters())
            + list(ddpm.radar_patch.parameters())
            + list(ddpm.radar_unpatch.parameters())
            + list(ddpm.pool.parameters())
            + list(ddpm.cond_proj.parameters())
        ),
    )
    # Training loop
    num_epochs = config["dit_epochs"]
    epoch_bar = trange(num_epochs, desc="Epochs")
    best_epoch_train_ddpm_loss = float("inf")
    for epoch in epoch_bar:
        ddpm.model.train()
        ddpm.radar_patch.train()
        ddpm.radar_unpatch.train()
        ddpm.pool.train()
        ddpm.cond_proj.train()
        pbar = tqdm(dataloader, leave=False, desc="Train Batch")
        avg_ddpm_train_loss = train_eval_batch(
            ddpm, dataloader, optimizer, config, train=True
        )
        writer.add_scalar("Loss/train/ddpm", avg_ddpm_train_loss, epoch + 1)
        if (epoch + 1) % config["eval_every"] == 0:
            avg_val_ddpm_loss = train_eval_batch(
                ddpm, eval_dataloader, None, config, train=False
            )

            writer.add_scalar("Loss/val/ddpm", avg_val_ddpm_loss, epoch + 1)

        if avg_ddpm_train_loss < best_epoch_train_ddpm_loss:
            best_epoch_train_ddpm_loss = avg_ddpm_train_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_vae_voxel_ddpm_{run_id}.pth"
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
            print(
                f"Saved best model checkpoint to {checkpoint_path}, avg_loss: {avg_ddpm_train_loss:.4f}, epoch: {epoch+1}"
            )

    writer.close()
    save_checkpoint(
        ddpm,
        optimizer,
        num_epochs,
        {
            "ddpm_loss": avg_ddpm_train_loss,
        },
        os.path.join(checkpoint_dir, f"final_vae_voxel_ddpm_{run_id}.pth"),
        config,
    )
    print(f"Training completed. Final model saved.")


def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    sample_result_dir = "/data/palakons/man_vaevoxeldit/plots"
    checkpoint_dir = "/data/palakons/man_vaevoxeldit/checkpoints"
    tb_log_dir = "/home/palakons/logs/tb_log/vaevoxeldit"
    os.makedirs(sample_result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Load dataset
    dataset_point = MANDataset(
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
        viz_dir=sample_result_dir,
    )

    dataset_point = Subset(
        dataset_point, range(min(config["num_input_frames"], len(dataset_point)))
    )

    val_split = int(0.5 * len(dataset_point))
    # TODO: Adjust the dataset splits if necessary,.8 for real work
    train_dataset = Subset(dataset_point, range(0, val_split))
    val_dataset = Subset(dataset_point, range(val_split, len(dataset_point)))
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Train
    print("Starting training...")

    scene_str = "-".join(map(str, sorted(config["scene_ids"])))
    runid = f"{config['depth_voxel_grid_bins']}bins_{config['scaled_image_size'][0]}x{config['scaled_image_size'][1]}_{config['dit_epochs']}epochs_batch{config['batch_size']}_scenes{scene_str}_numrfames{config['num_input_frames']}_seed{config['seed']}"
    ddpm = train_vae_voxel_ddpm(
        train_dataset,
        val_dataset,
        config,
        checkpoint_dir=checkpoint_dir,
        run_id=runid,
        tb_dir=tb_log_dir,
    )

    # Test sampling
    print("\nTesting sampling...")
    sample_data = val_dataset[0]
    depth_image = sample_data["depth_image"].to(config["device"])
    camera_front = sample_data["camera_front"].to(config["device"])
    points_uvz = sample_data["uvz"].to(config["device"])
    actual_occupancy_grid = sample_data["occupancy_grid"].to(config["device"])

    # Generate occupancy grid
    # generated_occupancy_grid = ddpm.sample(
    #     camera_front,
    #     depth_image,
    #     num_inference_steps=config["num_inference_steps"],
    #     seed=42,
    # )

    # # Visualize reconstruction

    # original_uvz = points_uvz
    # original_uvz[:, 0] = original_uvz[:, 0] * (
    #     config["original_image_size"][0] / config["scaled_image_size"][0]
    # )
    # original_uvz[:, 1] = original_uvz[:, 1] * (
    #     config["original_image_size"][1] / config["scaled_image_size"][1]
    # )
    # dataset.visuaplotvz_comparison(
    #     frame_token=sample_data["frame_token"] + f"_{runid}",
    #     original_uvz=dataset.occupancy_grid_to_uvz(
    #         actual_occupancy_grid,
    #         original_image_size=config["original_image_size"],
    #         max_depth=config["max_voxel_grid_depth"],
    #         threshold=0.5,
    #     ),
    #     reconstructed_uvz=dataset.occupancy_grid_to_uvz(
    #         generated_occupancy_grid,
    #         original_image_size=config["original_image_size"],
    #         max_depth=config["max_voxel_grid_depth"],
    #         threshold=0.5,
    #     ),
    #     save_dir=sample_result_dir,
    # )

    # # Confusion matrix, actual zero and predicted zero
    # supposed_to_be_negative = actual_occupancy_grid <= 0

    # z_z = (generaplotcupancy_grid[supposed_to_be_negative] <= 0).sum().item()
    # z_nz = (generated_occupancy_grid[supposed_to_be_negative] > 0).sum().item()
    # nz_z = (generated_occupancy_grid[~supposed_to_be_negative] <= 0).sum().item()
    # nz_nz = (generated_occupancy_grid[~supposed_to_be_negative] > 0).sum().item()
    # total = z_z + z_nz + nz_z + nz_nz
    # print(f"Confusion Matrix (number/percent):")
    # print(f"                Predicted Zero   Predicted Non-Zero")
    # print(
    #     f"Actual Zero        {z_z}/{(z_z/total)*100:.2f}%               {z_nz}/{(z_nz/total)*100:.2f}%"
    # )
    # print(
    #     f"Actual Non-Zero   {nz_z}/{(nz_z/total)*100:.2f}%               {nz_nz}/{(nz_nz/total)*100:.2f}%"
    # )

    # with open(f"{sample_result_dir}/confusion_matrix_results_{runid}.txt", "w") as f:
    #     f.write(f"Confusion Matrix (number/percent):\n")
    #     f.write(f"                Predicted Zero   Predicted Non-Zero\n")
    #     f.write(
    #         f"Actual Zero        {z_z}/{(z_z/total)*100:.2f}%               {z_nz}/{(z_nz/total)*100:.2f}%\n"
    #     )
    #     f.write(
    #         f"Actual Non-Zero   {nz_z}/{(nz_z/total)*100:.2f}%               {nz_nz}/{(nz_nz/total)*100:.2f}%\n"
    #     )
    #     # also write config
    #     f.write("Configuration:\n")
    #     for key, value in config.items():
    #         f.write(f"{key}: {value}\n")

    #     f.write("\n")

    #     f.write("Additional Metrics:\n")
    #     f.write(f"Total Samples: {total}\n")
    #     f.write(f"Zero Predictions: {z_z + z_nz}\n")
    #     f.write(f"Non-Zero Predictions: {nz_z + nz_nz}\n")

    #     # f1 score, precision, recall
    #     precision = nz_nz / (nz_nz + z_nz) if (nz_nz + z_nz) > 0 else 0
    #     recall = nz_nz / (nz_nz + nz_z) if (nz_nz + nz_z) > 0 else 0
    #     f1_score = (
    #         2 * (precision * recall) / (precision + recall)
    #         if (precision + recall) > 0
    #         else 0
    #     )
    #     f.write(f"Precision: {precision:.4f}\n")
    #     f.write(f"Recall: {recall:.4f}\n")
    #     f.write(f"F1 Score: {f1_score:.4f}\n")

    # # save ddpm model
    # model_save_path = f"{sample_result_dir}/ddpm_model_{runid}.pt"
    # torch.save(
    #     {"model_state_dict": ddpm.model.state_dict(), "config": config},
    #     model_save_path,
    # )
    # print(f"Saved DDPM model to {model_save_path}")


if __name__ == "__main__":
    main()
