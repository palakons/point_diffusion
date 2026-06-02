#!/usr/bin/env python3
"""
Interactive DDPM training launcher with model selection.

Supports multiple denoiser architectures for point cloud diffusion.
Leans towards simplicity and clarity over maximum expressiveness.

Usage:
    # See all available models
    python train_ddpm_man.py --list-models
    
    # Quick test (5 min)
    python train_ddpm_man.py --task synthetic --model mlp --iters 100 --batch_size 8
    
    # Balanced MAN training (recommended)
    python train_ddpm_man.py --task man --model unet --iters 1000 --batch_size 4 --use_scene_cond
    
    # MAN legacy (backward compatible, defaults to MAN task)
    python train_ddpm_man.py --data_file man-mini --iters 500 --num_points 512

Available Denoiser Models:
    mlp             - Per-point MLP: fast, low memory, good baseline
    unet            - 1D UNet: encoder-decoder with max pooling & skip connections
    minimal         - Minimal: lightweight per-point with residual blocks
    ptv3            - PTv3 sparse transformer: slow but powerful (expert-level)
"""

import sys
import os

# Dynamic path detection - works from any installation location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

TESTS_DIR = os.path.join(BASE_DIR, "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# Lazy imports - only imported when actually running
import argparse
from pathlib import Path


# ============================================================================
# Model Registry: Lean & Clear (Using lazy class resolution)
# ============================================================================

DENOISER_REGISTRY = {
    'mlp': {
        'class_name': 'MLPDenoiser',
        'module': 'unet_diffuser',
        'desc': 'Per-point MLP: fast, low-memory baseline. Good for testing.',
        'quick_args': {'context_channels': 256, 'hidden_dim': 256, 'num_layers': 4},
    },
    'unet': {
        'class_name': 'SimplePointUNet',
        'module': 'unet_diffuser',
        'desc': '1D UNet with skip connections. Balanced performance.',
        'quick_args': {'base_channels': 64, 'num_layers': 4, 't_embed_dim': 32},
    },
    'pointnet': {
        'class_name': 'PointNetLikeDenoiser',
        'module': 'unet_diffuser',
        'desc': 'PointNet-like: compact residual blocks. Fast inference.',
        'quick_args': {'hidden_channels': 64, 'time_embed_dim': 32, 'num_blocks': 3},
    },
    'ptv3': {
        'class_name': 'PTv3Dnsr',
        'module': 'unet_diffuser',
        'desc': 'PTv3 Sparse Transformer. Powerful but memory-heavy.',
        'quick_args': {'context_channels': 256, 'grid_size': 0.02, 'n_stages': 5},
    },
}


def print_model_guide():
    """Print concise guide to available models."""
    print("\n" + "="*80)
    print("DENOISER MODEL GUIDE")
    print("="*80)
    for name, info in DENOISER_REGISTRY.items():
        print(f"\n{name.upper()}")
        print(f"  {info['desc']}")
    print("\n" + "="*80 + "\n")


def create_denoiser(model_name, scene_embed_dim=0, **model_kwargs):
    """Factory to instantiate denoiser by name."""
    # Lazy import here
    from unet_diffuser import MLPDenoiser, SimplePointUNet, PointNetLikeDenoiser, PTv3Dnsr
    
    if model_name not in DENOISER_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(DENOISER_REGISTRY.keys())}")
    
    registry_entry = DENOISER_REGISTRY[model_name]
    class_name = registry_entry['class_name']
    
    # Resolve class by name
    class_map = {
        'MLPDenoiser': MLPDenoiser,
        'SimplePointUNet': SimplePointUNet,
        'PointNetLikeDenoiser': PointNetLikeDenoiser,
        'PTv3Dnsr': PTv3Dnsr,
    }
    ModelClass = class_map[class_name]
    quick_args = registry_entry['quick_args'].copy()
    
    # Override with user kwargs
    quick_args.update(model_kwargs)
    
    # Special handling for MLPDenoiser scene conditioning
    if model_name == 'mlp':
        quick_args['scene_embed_dim'] = scene_embed_dim
    
    return ModelClass(**quick_args)


# def run_synthetic_test(args):
#     """Run identity test on synthetic point clouds."""
#     # Import heavy dependencies here
#     import torch
#     from torch.utils.data import DataLoader
#     from identity_test import train_denoiser, make_synthetic_dataset, SyntheticPointDataset
    
#     print("\n" + "="*80)
#     print(f"SYNTHETIC IDENTITY TEST: {args.model.upper()} Denoiser")
#     print("="*80)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}\n")
    
#     # Create model
#     model = create_denoiser(
#         args.model,
#         scene_embed_dim=0,
#         **{k: v for k, v in vars(args).items() if k.startswith(f'{args.model}_')}
#     )
#     print(f"Model: {args.model}")
#     print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Create synthetic data
#     num_samples = args.num_samples or (args.batch_size * 10)
#     data = make_synthetic_dataset(num_samples, args.num_points, device=device, seed=args.seed)
    
#     num_train = int(num_samples * (1 - args.val_split))
#     train_data, val_data = data[:num_train], data[num_train:]
    
#     train_dl = DataLoader(SyntheticPointDataset(train_data), batch_size=args.batch_size, shuffle=True)
#     val_dl = DataLoader(SyntheticPointDataset(val_data), batch_size=args.batch_size, shuffle=False)
    
#     # Setup logging
#     tb_dir = f"/home/palakons/logs/tb_log/identity_test_{args.model}/{args.exp_name}"
#     plot_dir = f"/data/palakons/identity_test_{args.model}/{args.exp_name}"
    
#     # Train
#     print(f"\nTraining for {args.iters} iterations...")
#     train_denoiser(
#         model=model,
#         train_dataloader=train_dl,
#         val_dataloader=val_dl,
#         device=device,
#         model_name=args.model,
#         runname=args.exp_name,
#         num_points=args.num_points,
#         iters=args.iters,
#         lr=args.lr,
#         print_every=args.print_every,
#         tb_log_dir=tb_dir,
#         plot_dir=plot_dir,
#         grid=args.grid,
#         noise_mode=args.noise_mode,
#         gaussian_noise_std=args.gaussian_noise_std,
#         scheduler_mode=args.scheduler_mode,
#         num_train_timesteps=args.num_train_timesteps,
#         ddpm_beta_schedule=args.ddpm_beta_schedule,
#         target_mode=args.target_mode,
#         sample_every=args.sample_every,
#         sample_inference_steps=args.sample_inference_steps,
#     )
    
#     print(f"\nTensorBoard: tensorboard --logdir {tb_dir}")


def run_man_training(args):
    """Run scene-conditioned DDPM on MAN dataset."""
    # Import heavy dependencies here
    import torch
    from torch.utils.data import DataLoader
    from identity_test import train_denoiser
    from man_ddpm import MANDataset
    
    print("\n" + "="*80)
    print(f"MAN DATASET SCENE-CONDITIONED TEST: {args.model.upper()} Denoiser")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load MAN dataset
    print(f"Loading MANDataset ({args.data_file})...")
    dataset = MANDataset(
        scene_ids=args.scene_ids or [],
        data_file=args.data_file,
        device=device,
        wan_vae=True,
        wan_vae_checkpoint=args.wan_vae_path,
        n_p=args.num_points,
        normalize_type="minmax",
        get_camera=False,
        keep_frames=args.keep_frames,
        point_preset="original"
    )
    print(f"Loaded {len(dataset)} frames")
    
    # Auto-detect scene embedding dim
    first_sample = dataset[0]
    scene_embed_dim = int(torch.tensor(first_sample["wan_vae_latent"].shape).prod()) if "wan_vae_latent" in first_sample else 0
    print(f"Scene embedding dimension: {scene_embed_dim}\n")
    
    # Create model
    model = create_denoiser(
        args.model,
        scene_embed_dim=scene_embed_dim,
        **{k.replace(f'{args.model}_', ''): v for k, v in vars(args).items() if k.startswith(f'{args.model}_')}
    )
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Split data
    train_size = int(len(dataset) * (1 - args.val_split))
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    val_data = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Setup logging
    key_dir = f'ddpm_cond_scene'
    tb_dir = f"/home/palakons/logs/tb_log/{key_dir}/{args.exp_name}"
    plot_dir = f"/data/palakons/{key_dir}/{args.exp_name}/plots"
    
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Train
    print(f"\nTraining for {args.iters} iterations...")
    train_denoiser(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        device=device,
        model_name=args.model,
        runname=args.exp_name,
        num_points=args.num_points,
        iters=args.iters,
        lr=args.lr,
        tb_log_dir=tb_dir,
        plot_dir=plot_dir,
        noise_mode=args.noise_mode,
        gaussian_noise_std=args.gaussian_noise_std,
        scheduler_mode=args.scheduler_mode,
        num_train_timesteps=args.num_train_timesteps,
        ddpm_beta_schedule=args.ddpm_beta_schedule,
        target_mode=args.target_mode,
        use_scene_conditioning=args.use_scene_cond,
        log_train_visuals=True,
        sample_every=args.sample_every,
        sample_inference_steps=args.sample_inference_steps,
        duplications=args.duplications,
    )
    
    print(f"\nTensorBoard: tensorboard --logdir {tb_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset & task
    parser.add_argument(
        "--task",
        type=str,
        default="man",
        choices=["synthetic", "man"],
        help="Task: 'synthetic' (identity test) or 'man' (real dataset). Default: man",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=list(DENOISER_REGISTRY.keys()),
        help=f"Denoiser architecture. Available: {', '.join(DENOISER_REGISTRY.keys())}",
    )
    parser.add_argument("--list-models", action="store_true", help="Print model guide and exit")
    
    # Data
    parser.add_argument("--num_points", type=int, default=512, help="Points per cloud")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="Synthetic: num samples (default: batch_size*10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # MAN dataset only
    parser.add_argument("--data_file", type=str, default="man-mini", choices=["man-mini", "man-full"])
    parser.add_argument("--keep_frames", type=int, default=0, help="Limit frames (0=no limit)")
    parser.add_argument("--scene_ids", type=int, nargs="*", default=[], help="Scene IDs (empty=all)")
    parser.add_argument("--use_scene_cond", action="store_true", help="Enable scene conditioning (MLP only)")
    parser.add_argument("--wan_vae_path", type=str, default="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth")
    
    # Training
    parser.add_argument("--iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    
    # DDPM
    parser.add_argument("--noise_mode", type=str, default="gaussian", choices=["fixed", "gaussian"])
    parser.add_argument("--gaussian_noise_std", type=float, default=1.0)
    parser.add_argument("--scheduler_mode", type=str, default="ddpm", choices=["fixed", "ddpm"])
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear", 
                       choices=["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"])
    parser.add_argument("--target_mode", type=str, default="clean", choices=["clean", "noise"])
    
    # Sampling
    parser.add_argument("--sample_every", type=int, default=20, help="Sample every N iters")
    parser.add_argument("--sample_inference_steps", type=int, default=50)
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="exp_ddpm", help="Experiment name")
    parser.add_argument("--grid", type=float, default=None, help="Grid spacing for plots")
    parser.add_argument("--pointnet_hidden_channels", type=int, default=64, help="PointNet hidden channels")
    parser.add_argument("--pointnet_time_embed_dim", type=int, default=32, help="PointNet time embedding dimension")
    parser.add_argument("--pointnet_num_blocks", type=int, default=3, help="PointNet number of residual blocks")
    parser.add_argument("--duplications", type=int, default=1, help="Number of times to duplicate each point cloud in the dataset (for data augmentation)")
    parser.add_argument("--ptv3_grid_size", type=float, default=0.02, help="PTv3 grid size for sparse attention")

    args = parser.parse_args()
    
    if args.list_models:
        print_model_guide()
        return
    
    # Validate task
    if args.task == "synthetic":
        run_synthetic_test(args)
    elif args.task == "man":
        run_man_training(args)


if __name__ == "__main__":
    main()
