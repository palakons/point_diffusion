import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from man_ddpm import MANDataset
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

from dit_ddpm_class import (
    parse_args,
    set_seed,
    PretrainedPointNeXtEncoderPointAE,
    TransformerPointAE,
    TransformerAttentionPointAE,
    train_dit_diffusion,
)


def main():
    args = parse_args()

    # Training config
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])

    sample_result_dir = "/data/palakons/man_ds_sample_dit/generated_uvz"
    checkpoint_dir = "/data/palakons/man_ds_sample_dit/checkpoints"
    tb_log_dir = "/home/palakons/logs/tb_log/dit_ddpm"
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
        visualize_uvz=False,
        scaled_image_size=config["scaled_image_size"],
        n_p=config["num_points"],
        point_only=False,  # False for dit training
    )
    print(f"Dataset loaded with {len(dataset_point)} samples.")

    runid = f"dit-epochs{config['dit_epochs']}_batch{config['batch_size']}_aemodel{config['point_ae_model']}_pointlen{len(dataset_point)}_pointperframe{config['num_points']}_radarch{config['radar_channel']}_camerach{config['camera_channel']}_timesteps{config['num_train_timesteps']}_latdim{config['latent_dim']}_latseq{config['latent_seq_length']}"
    print(f"Run ID: {runid}")

    print("\n" + "=" * 60)
    print("PHASE 1: AUTOENCODER LOADING")
    print("=" * 60)

    print(f"'{config['point_ae_model']}'")

    ae_checkpoint_file = config["point_ae_checkpoint"]
    ae_checkpoint = torch.load(ae_checkpoint_file, map_location=config["device"])
    print(f"Loaded point AE checkpoint from {ae_checkpoint_file}")
    # Build the autoencoder
    autoencoder = None
    if config["point_ae_model"] == "pointnext-attn":
        autoencoder = PretrainedPointNeXtEncoderPointAE(
            d_model=config["latent_dim"],
            seq_length=config["latent_seq_length"],
            output_points=config["num_points"],
            device=config["device"],
            decoder_model="attention",
        ).to(config["device"])

        # state = {
        #     "encoder_proj": autoencoder.encoder_proj.state_dict(),
        #     "decoder": autoencoder.pointcloud_decoder.state_dict(),
        # }
        autoencoder.encoder_proj.load_state_dict(ae_checkpoint["encoder_proj"])
        autoencoder.pointcloud_decoder.load_state_dict(ae_checkpoint["decoder"])
    elif config["point_ae_model"] == "pointnext-transformer":
        autoencoder = PretrainedPointNeXtEncoderPointAE(
            d_model=config["latent_dim"],
            seq_length=config["latent_seq_length"],
            output_points=config["num_points"],
            device=config["device"],
            decoder_model="transformer",
            num_decoder_layers=config["num_decoder_layers"],
        ).to(config["device"])

        # state = {
        #     "encoder_proj": autoencoder.encoder_proj.state_dict(),
        #     "decoder": autoencoder.pointcloud_decoder.state_dict(),
        # }
        autoencoder.encoder_proj.load_state_dict(ae_checkpoint["encoder_proj"])
        autoencoder.pointcloud_decoder.load_state_dict(ae_checkpoint["decoder"])

    elif config["point_ae_model"] == "transformer":
        autoencoder = TransformerPointAE(
            d_model=config["latent_dim"],
            seq_length=config["latent_seq_length"],
            output_points=config["num_points"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            device=config["device"],
        ).to(config["device"])

        # state = autoencoder.state_dict()
        autoencoder.load_state_dict(ae_checkpoint)
    elif config["point_ae_model"] == "transformer-attn":
        autoencoder = TransformerAttentionPointAE(
            seq_length=config["latent_seq_length"],
            output_points=config["num_points"],
            num_encoder_layers=config["num_encoder_layers"],
            device=config["device"],
        ).to(config["device"])

        # state = autoencoder.state_dict()
        autoencoder.load_state_dict(ae_checkpoint)
    else:
        raise ValueError(f"Unknown point_ae_model: {config['point_ae_model']}")

    print(f"Autoencoder model '{config['point_ae_model']}' loaded.")

    # PHASE 2: Train DiT diffusion
    print("\n" + "=" * 60)
    print("PHASE 2: DIT DIFFUSION TRAINING")
    print("=" * 60)

    dit_checkpoint_file = config["dit_checkpoint"]
    dit_checkpoint = torch.load(dit_checkpoint_file, map_location=config["device"])
    print(f"Loaded DiT checkpoint from {dit_checkpoint_file}")

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
    # Load DDPM model
    assert config["dit_checkpoint"], "DiT checkpoint missing 'model_state_dict'"
    model_save_path = config["dit_checkpoint"]
    dit_checkpoint = torch.load(dit_checkpoint_file, map_location=config["device"])
    # {
    #     "model_state_dict": self.model.state_dict(),
    #     "depth_tokenizer_state_dict": self.depth_tokenizer.state_dict(),
    #     "config": {
    #         "num_train_timesteps": self.noise_scheduler.config.num_train_timesteps,
    #     },
    # }
    ddpm.model.load_state_dict(dit_checkpoint["model_state_dict"])
    ddpm.depth_tokenizer.load_state_dict(dit_checkpoint["depth_tokenizer_state_dict"])
    print(f"Loaded DiTDDPM model from {model_save_path}")

    # Test sampling
    print("\nTesting sampling...")
    sample_data = dataset_point[0]
    set_seed(config["seed"])  # For reproducibility
    predicted_uvz, outputs = ddpm.sample(
        sample_data["depth_image"].unsqueeze(0).to(config["device"]),
        sample_data["clip_feature"].unsqueeze(0).to(config["device"]),
        num_inference_steps=50,
        seed=config["seed"],
    )
    # {
    #     "pointnext_embedding": pointnext_embedding.clone(),
    #     "uvz": predicted_uvz.clone(),
    #     "confidence": confidence.clone(),
    # }

    # print("keys in predicted_uvz:", predicted_uvz.keys()) #(['pointnext_embedding', 'uvz', 'confidence'])

    if "uvz" in predicted_uvz:
        original_uvz = sample_data["uvz"].to(config["device"])

        # Visualize reconstruction
        original_uvz[:, 0] = original_uvz[:, 0] * (
            config["original_image_size"][0] / config["scaled_image_size"][0]
        )
        original_uvz[:, 1] = original_uvz[:, 1] * (
            config["original_image_size"][1] / config["scaled_image_size"][1]
        )

        for t in outputs:
            dataset_point.visualize_uvz_comparison(
                title=runid + "_dit_sample",
                frame_token=sample_data["frame_token"][-5:] + f"_{runid}_sample_at{t}",
                original_uvz=original_uvz.cpu().numpy(),
                reconstructed_uvz=outputs[t]["uvz"].squeeze(0).cpu().numpy(),
                save_dir=sample_result_dir,
            )


if __name__ == "__main__":
    main()
