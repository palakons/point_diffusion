"""
Identity Test for point denoisers
- Trains a model to map x -> x (self-reconstruction) on synthetic point clouds
- Default: `minimal` model (lightweight). Attempts `ptv3` if available; skips if imports fail.
- Generates progress bars with tqdm
- Saves intermediate visualization of GT vs predicted points every sample_every iterations

Usage:
    python tests/identity_test.py --model minimal --iters 200 --batch_size 4 --num_points 512 --sample_every 20
    python tests/identity_test.py --model ptv3 --iters 50 --batch_size 2 --num_points 256 --sample_every 10

Outputs:
    - Saved plots: /data/palakons/test/identity_test_minimal_iter20.jpg, iter40.jpg, ...
    - Saved plots: /data/palakons/test/identity_test_ptv3_iter10.jpg, iter20.jpg, ...
"""

from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance
import os
import sys
import time
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusers import DDPMScheduler

import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

BASE_DIR = os.path.abspath("/home/palakons/point_diffusion")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import denoisers and utilities
from unet_diffuser import PTv3Dnsr, visualize_xyz_pip, visualize_xyz_batch_grid, plot_cd_timestep_curve, MLPDenoiser


def make_synthetic_dataset(num_samples, num_points, device="cpu", seed=0):
    rng = np.random.RandomState(seed)
    # points in [-1, 1]
    data = rng.uniform(-1.0, 1.0, size=(num_samples, 3, num_points)).astype(np.float32)
    return torch.tensor(data, device=device)


class SyntheticPointDataset(torch.utils.data.Dataset):
    """Simple Dataset that returns a dict with key 'points'.

    Each item is a tensor of shape (3, N).
    DataLoader will collate batches into {'points': tensor(B,3,N)}.
    """

    def __init__(self, data_tensor):
        # Expect a tensor of shape (num_samples, 3, num_points)
        self.data = data_tensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"points": self.data[idx]}


def make_fixed_noise(num_points,
    n_noise_levels=1,
    noise_min=0.0,
    noise_max=1.0,
):
    """generate  (,num_points) noise tensor with values between noise_min and noise_max, inclusive, with n_noise_levels distinct values, then expand to (3,num_points) by repeating across the 3 channel dimension."""

    noise_values = np.linspace(noise_min, noise_max, n_noise_levels, dtype=np.float32, endpoint=True)

    # Create a repeating pattern of noise values across the num_points dimension
    noise_pattern = np.tile(noise_values, (num_points // n_noise_levels + 1))[:num_points]

    # Expand to (3, num_points) by repeating across the channel dimension
    noise_tensor = np.tile(noise_pattern, (3, 1))

    return torch.tensor(noise_tensor, dtype=torch.float32)


def _prefix_levels(name, max_levels=5):
    """Return hierarchical parameter prefixes up to max_levels tokens.

    Example:
    ptv3.enc.enc0.block0.attn.qkv.weight ->
    [ptv3, ptv3.enc, ptv3.enc.enc0, ptv3.enc.enc0.block0, ptv3.enc.enc0.block0.attn]
    """
    parts = name.split('.')
    limit = min(len(parts), max_levels)
    return ['.'.join(parts[:level]) for level in range(1, limit + 1)]


def _gradient_component(name):
    """Map a parameter name to a coarse component bucket.

    The buckets are chosen to answer the main question directly:
    attention vs MLP vs CPE vs sparseconv/stem paths.
    """
    name_lower = name.lower()

    if name_lower.startswith("head"):
        return "head"
    if ".attn." in name_lower:
        return "attn"
    if ".mlp." in name_lower:
        return "mlp"
    if ".cpe." in name_lower:
        return "cpe"
    if "stem" in name_lower or "embedding" in name_lower:
        return "sparseconv_stem"
    if ".down." in name_lower or ".up." in name_lower or "proj_skip" in name_lower:
        return "sparseconv_io"
    if ".proj." in name_lower:
        return "proj"
    if ".norm." in name_lower or name_lower.endswith(".norm"):
        return "norm"
    if "modulation" in name_lower:
        return "modulation"
    if "time_mlp" in name_lower:
        return "time_mlp"
    if name_lower.startswith("ptv3"):
        return "ptv3_other"
    return "other"


def log_gradient_diagnostics(model, writer, it,verbose =False):
    """Log train-time gradient diagnostics to TensorBoard.

    Call this only after `loss.backward()` and before the next `zero_grad()`.
    """
    level_grads = {}
    component_grads = {}
    all_grads = []
    component_sum_abs = {}
    component_numel = {}
    component_l2_sq = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_mag = param.grad.abs().mean().item()
        all_grads.append(grad_mag)

        component = _gradient_component(name)
        component_grads.setdefault(component, []).append(grad_mag)

        grad_abs = param.grad.abs()
        component_sum_abs[component] = component_sum_abs.get(component, 0.0) + grad_abs.sum().item()
        component_numel[component] = component_numel.get(component, 0) + param.grad.numel()
        grad_l2 = param.grad.norm().item()
        component_l2_sq[component] = component_l2_sq.get(component, 0.0) + grad_l2 * grad_l2

        for level, prefix in enumerate(_prefix_levels(name, max_levels=5), start=1):
            level_grads.setdefault((level, prefix), []).append(grad_mag)
    if verbose:
        print(f"----Gradient categories at iter {it}:")
    if all_grads:
        global_mean = float(np.mean(all_grads))
        writer.add_scalar("grad/global_mean", global_mean, it)
        writer.add_histogram("grad/global_hist", np.asarray(all_grads), it)
        if verbose:
            print(f"Global mean grad: {global_mean:6e}")

    for (level, prefix), grads in sorted(level_grads.items()):
        safe_prefix = prefix.replace('/', '_')
        mean_grad = float(np.mean(grads))
        writer.add_scalar(f"grad/lvl{level}/{safe_prefix}", mean_grad, it)
        try:
            writer.add_histogram(f"grad_hist/lvl{level}/{safe_prefix}", np.asarray(grads), it)
        except Exception:
            pass
        if verbose:
            print(f"Level {level}: {prefix}, mean grad: {mean_grad:6e}")

    for component, grads in sorted(component_grads.items()):
        mean_grad = float(np.mean(grads))
        writer.add_scalar(f"grad/component/{component}", mean_grad, it)
        try:
            writer.add_histogram(f"grad_hist/component/{component}", np.asarray(grads), it)
        except Exception:
            pass
        if verbose:
            print(f"Component: {component}, mean grad: {mean_grad:6e}")

    output_mean = float(np.mean(component_grads.get("head", []))) if component_grads.get("head") else None
    backbone_components = ["attn", "mlp", "cpe", "sparseconv_stem", "sparseconv_io", "proj", "norm", "modulation", "time_mlp", "ptv3_other"]
    backbone_values = [float(np.mean(component_grads[c])) for c in backbone_components if component_grads.get(c)]
    backbone_mean = float(np.mean(backbone_values)) if backbone_values else None

    if output_mean is not None:
        writer.add_scalar("grad/output_mean", output_mean, it)
        if verbose:
            print(f"Output-related categories: ['head'], mean grad: {output_mean:6e}")

    if backbone_mean is not None:
        writer.add_scalar("grad/backbone_mean", backbone_mean, it)
        if verbose:
            print(f"Backbone-related mean grad: {backbone_mean:6e}")

    if output_mean is not None and backbone_mean is not None:
        ratio = output_mean / (backbone_mean + 1e-8)
        writer.add_scalar("grad/amplification_ratio", ratio, it)
        if verbose:
            print(f"Amplification ratio: {ratio:6e}")

    # Robust alternatives to per-parameter averaging:
    # 1) element-weighted mean(|grad|) ratio
    # 2) L2 norm ratio
    output_sum_abs = component_sum_abs.get("head", 0.0)
    output_numel = component_numel.get("head", 0)
    output_l2_sq = component_l2_sq.get("head", 0.0)

    backbone_sum_abs = 0.0
    backbone_numel = 0
    backbone_l2_sq = 0.0
    for c in backbone_components:
        backbone_sum_abs += component_sum_abs.get(c, 0.0)
        backbone_numel += component_numel.get(c, 0)
        backbone_l2_sq += component_l2_sq.get(c, 0.0)

    if output_numel > 0 and backbone_numel > 0:
        output_elem_mean = output_sum_abs / max(1, output_numel)
        backbone_elem_mean = backbone_sum_abs / max(1, backbone_numel)
        ratio_weighted = output_elem_mean / (backbone_elem_mean + 1e-12)
        writer.add_scalar("grad/amplification_ratio_weighted", ratio_weighted, it)
        writer.add_scalar("grad/output_elem_mean", output_elem_mean, it)
        writer.add_scalar("grad/backbone_elem_mean", backbone_elem_mean, it)
        if verbose:
            print(f"Amplification ratio weighted: {ratio_weighted:6e}")

    if output_l2_sq > 0.0 and backbone_l2_sq > 0.0:
        output_l2 = float(np.sqrt(output_l2_sq))
        backbone_l2 = float(np.sqrt(backbone_l2_sq))
        ratio_l2 = output_l2 / (backbone_l2 + 1e-12)
        writer.add_scalar("grad/amplification_ratio_l2", ratio_l2, it)
        writer.add_scalar("grad/output_l2", output_l2, it)
        writer.add_scalar("grad/backbone_l2", backbone_l2, it)
        if verbose:
            print(f"Amplification ratio L2: {ratio_l2:6e}")


def log_prediction_statistics(writer, it, pred_bn3, target, split="val", verbose=False):
    """Log prediction statistics under `torch.no_grad()`.

    This is safe for validation because it only inspects outputs, not gradients.
    Logs: mean, std, histograms for predictions and targets.
    """
    # Flatten for statistics
    pred_flat = pred_bn3.flatten()
    target_flat = target.flatten()
    
    pred_mean = pred_flat.mean().item()
    pred_std = pred_flat.std().item()
    target_mean = target_flat.mean().item()
    target_std = target_flat.std().item()
    
    # Spatial variance (per channel, averaged)

    
    # Log scalars: mean and std
    writer.add_scalar(f"{split}/pred_mean", pred_mean, it)
    writer.add_scalar(f"{split}/pred_std", pred_std, it)
    writer.add_scalar(f"{split}/target_mean", target_mean, it)
    writer.add_scalar(f"{split}/target_std", target_std, it)
    
    # Log spatial variance (existing)

    
    # Log histograms
    writer.add_histogram(f"{split}/pred_histogram", pred_flat, it, bins=50)
    writer.add_histogram(f"{split}/target_histogram", target_flat, it, bins=50)
    
    if verbose:
        print(f"{split.capitalize()} Pred: mean={pred_mean:7.4f}, std={pred_std:7.4f}; Target: mean={target_mean:7.4f}, std={target_std:7.4f}")



def log_plot_image(writer, tag, image_path, it):
    """Load a saved plot and push it to TensorBoard as an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image).transpose(2, 0, 1)
        writer.add_image(tag, image_array, it)
    except Exception as exc:
        print(f"Failed to log TensorBoard image {tag} from {image_path}: {exc}")


def _prepare_scene_condition(condition):
    """Convert WAN VAE conditioning to a per-sample 2D embedding if needed."""
    if condition is None:
        return None
    # if condition.dim() > 2:
    #     condition = condition.mean(dim=tuple(range(2, condition.dim())))
    return condition


def sample_ddpm_batch(
    model,
    ddpm_scheduler,
    xb,
    device,
    condition=None,
    target_mode="noise",
    num_inference_steps=None,
    return_trace=False,
):
    """Generate a sampled point cloud batch with the trained denoiser."""
    if ddpm_scheduler is None:
        raise ValueError("sample_ddpm_batch requires a DDPMScheduler instance.")

    if target_mode not in {"noise", "clean"}:
        raise ValueError(f"Unsupported target_mode={target_mode} for sampling.")

    condition = _prepare_scene_condition(condition)
    sample_steps = (
        num_inference_steps
        if num_inference_steps is not None
        else ddpm_scheduler.config.num_train_timesteps
    )

    was_training = model.training
    model.eval()
    ddpm_scheduler.set_timesteps(sample_steps, device=device)

    sample = torch.randn_like(xb)
    cd_trace = []
    step_trace = []

    with torch.no_grad():
        for timestep in ddpm_scheduler.timesteps:
            timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            t_batch = torch.full((xb.shape[0],), timestep_value, device=device, dtype=torch.long)
            if condition is not None:
                model_out = model(sample, t_batch, condition=condition)
            else:
                model_out = model(sample, t_batch)

            if target_mode == "clean":
                alpha_prod = ddpm_scheduler.alphas_cumprod.to(device)[timestep_value]
                model_out = (
                    sample - torch.sqrt(alpha_prod) * model_out
                ) / torch.sqrt(1.0 - alpha_prod)

            sample = ddpm_scheduler.step(model_out, timestep_value, sample).prev_sample

            if return_trace:
                step_trace.append(timestep_value)
                sample_cd, _ = pt3d_chamfer_distance(
                    sample.transpose(1, 2),
                    xb.transpose(1, 2),
                    batch_reduction=None,
                )
                cd_trace.append(sample_cd.detach().cpu().numpy())


    if was_training:
        model.train()

    if return_trace:
        return sample, cd_trace, step_trace
    return sample


def _process_batch(
    batch,
    model,
    beta,
    noise,
    target_mode,
    device,
    noise_mode="fixed",
    gaussian_noise_std=1.0,
    scheduler_mode="fixed",
    ddpm_scheduler=None,
    loss_fn=None,
    use_scene_conditioning=False,
    lambda_cd=0.0,
    duplications=1,
):
    """Run corruption, model forward, normalization, and optional loss for a batch.

    Returns a dict with keys: xb, xt, noise, t, raw_out, target, loss, recon, scene_cond (if used)
    
    Args:
        use_scene_conditioning: if True, extract 'wan_vae_latent' from batch as scene conditioning
        duplications: number of times to duplicate each point cloud in the batch
    """
    # Accept different batch formats: dict {'points': ...}, tuple/list, or raw tensor
    # print(f"keys {batch.keys()}") #keys dict_keys(['filtered_radar_data', 'uvz', 'frame_token', 'npoints_original', 'npoints_after_distance_filter', 'npoints_filtered', 'wan_vae_latent', 'scene_id', 'frame_index'])

    if isinstance(batch, dict) and "points" in batch:
        xb = batch["points"].to(device)
    elif isinstance(batch, (tuple, list)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
        xb = batch[0].to(device)
    elif isinstance(batch, torch.Tensor):
        xb = batch.to(device)
    elif isinstance(batch, dict) and "filtered_radar_data" in batch:
        xb = batch["filtered_radar_data"].to(device).transpose(1, 2).contiguous()
    else:
        raise ValueError(
            "Unsupported batch format. Expected dict with 'points' key, tuple/list of tensors, or raw tensor."
        )
    
    if duplications > 1:
        xb = xb.repeat_interleave(duplications, dim=0)

    if xb.dim() == 3 and xb.shape[-1] == 3 and xb.shape[1] != 3:
        xb = xb.transpose(1, 2).contiguous()


    B = xb.shape[0]
    
    # Extract scene conditioning if available
    scene_cond = None
    if use_scene_conditioning:
        if isinstance(batch, dict) and "wan_vae_latent" in batch:
            scene_cond = _prepare_scene_condition(batch["wan_vae_latent"].to(device))
        else:
            raise ValueError(
                "use_scene_conditioning=True but batch does not contain 'wan_vae_latent' key."
            )

    if noise_mode == "fixed":
        if noise is None:
            raise ValueError("noise_mode='fixed' requires a fixed noise tensor.")
        if noise.dim() == 2:
            expanded_noise = noise[None, :, :].to(device).expand(B, -1, -1)
        elif noise.dim() == 3:
            if noise.shape[0] != B:
                raise ValueError(
                    f"Batch noise has shape {noise.shape}, expected batch dimension {B}."
                )
            expanded_noise = noise.to(device)
        else:
            raise ValueError(
                f"Unsupported fixed noise shape {noise.shape}; expected (3,N) or (B,3,N)."
            )
    elif noise_mode == "gaussian":
        expanded_noise = torch.randn(B, 3, xb.shape[-1], device=device) * gaussian_noise_std
    else:
        raise ValueError(
            f"Unsupported noise_mode={noise_mode}. Choose from ['fixed', 'gaussian']."
        )

    if scheduler_mode == "ddpm":
        if ddpm_scheduler is None:
            raise ValueError("scheduler_mode='ddpm' requires a DDPMScheduler instance.")
        t = torch.randint(
            0,
            ddpm_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
            dtype=torch.long,
        )
        alpha_bar = ddpm_scheduler.alphas_cumprod.to(device)[t].view(B, 1, 1)
        xt_bn3 = torch.sqrt(alpha_bar) * xb + torch.sqrt(1.0 - alpha_bar) * expanded_noise
        model_t = t
    elif scheduler_mode == "fixed":
        xt_bn3 = xb * (1 - beta) + expanded_noise * beta
        model_t = torch.zeros(B, device=device, dtype=torch.long)
        alpha_bar = None
    else:
        raise ValueError(
            f"Unsupported scheduler_mode={scheduler_mode}. Choose from ['fixed', 'ddpm']."
        )

    # Forward with optional scene conditioning
    if scene_cond is not None:
        # print(f"shape of xt_bn3: {xt_bn3.shape}, model_t: {model_t.shape}, scene_cond: {scene_cond.shape}") #shape of xt_bn3: torch.Size([8, 3, 512]), model_t: torch.Size([8]), scene_cond: torch.Size([8, 16, 2, 60, 104])


        pred_bn3 = model(xt_bn3, model_t, condition=scene_cond)
    else:
        pred_bn3 = model(xt_bn3, model_t)

    target = xb if target_mode == "clean" else expanded_noise  # predict noise

    main_loss = loss_fn(pred_bn3, target)
    cd_loss_mean = None    
    total_loss = main_loss

    if lambda_cd > 0.0:
        cd_loss, _ = pt3d_chamfer_distance(
            pred_bn3.transpose(1, 2),  # (B, N, 3)
            target.transpose(1, 2),    # (B, N, 3)
        )
        cd_loss_mean = cd_loss.mean()
        total_loss = total_loss + lambda_cd * cd_loss_mean

    if target_mode == "noise":
        if scheduler_mode == "ddpm":
            recon = (xt_bn3 - torch.sqrt(1.0 - alpha_bar) * pred_bn3) / torch.sqrt(alpha_bar)
        else:
            recon = (xt_bn3 - pred_bn3 * beta) / (1 - beta)
    else:
        recon = pred_bn3  # predict clean points

    result = {
        "xb": xb,
        "noise": expanded_noise,
        "beta": beta,
        "t": model_t,
        "xt_bn3": xt_bn3,
        "target": target,
        "pred_bn3": pred_bn3,
        "recon": recon,
        "loss": total_loss,
        "main_loss": main_loss,
        "cd_loss": cd_loss_mean,
    }
    
    if scene_cond is not None:
        result["scene_cond"] = scene_cond
    
    return result

def logmetrics(writer, it, result_dict, prefix,model,model_name,target_mode,plot_dir,grid,device,batch_id,ddpm_scheduler,sample_inference_steps):
    '''
    will plot/log to tensorboard:
    - GT vs recon for the 1st sample in the batch (using visualize_xyz_pip)
    - GT vs recon for the whole batch in a grid (using visualize_xyz_batch_grid)
    - GT vs sampled for the whole batch in a grid (using visualize_xyz_batch_grid)
    - CD vs reverse step curve for the sampled batch (using plot_cd_timestep_curve)
    - GT vs sampled for the 1st sample in the batch (using visualize_xyz_pip) 
    '''

    batch_cd, batch_plot_path = visualize_xyz_batch_grid(
        fname=f"{model_name}_iter{it+1}_{prefix}_batch",
        original_xyz=result_dict["xb"],
        reconstructed_xyz=result_dict["recon"],
        title=f"{model_name}: {prefix} Batch - Iter {it+1} timestep {result_dict['t'].float().mean().item():.1f}",
        save_dir=plot_dir,
        grid=grid,
        device=device,
    )
    log_plot_image(writer, f"{prefix}/batch_recon-0", batch_plot_path, it)



    for ibb in range(result_dict["xt_bn3"].shape[0]):
        gt_np = result_dict["xb"][ibb].cpu().numpy().T # n,3
        recon_np = result_dict["recon"][ibb].detach().cpu().numpy().T # n,3
        
        if ibb == 0:
            cd, save_path = visualize_xyz_pip( #plot recon vs xb
                fname=f"{model_name}_iter{it+1}_{prefix}_xt_batch{batch_id}_sample{ibb}",
                original_xyz=gt_np,
                reconstructed_xyz=recon_np,
                title=f"{model_name}: {prefix} Input - Iter {it+1} xt timestep {result_dict['t'][0].item():.1f}",
                save_dir=plot_dir,
                plotlims=None,
                grid=grid,
                device=device,
            )
            log_plot_image(writer, f"{prefix}/recon-0-0", save_path, it)
            break

    sampled_bn3, sampled_cd_trace, sampled_step_trace = sample_ddpm_batch(
        model,
        ddpm_scheduler,
        result_dict["xb"],
        device,
        condition=result_dict.get("scene_cond"),
        target_mode=target_mode,
        num_inference_steps=sample_inference_steps,
        return_trace=True,
    )

    sample_batch_cd, sample_batch_plot_path = visualize_xyz_batch_grid(
        fname=f"{model_name}_iter{it+1}_{prefix}_sampled_batch",
        original_xyz=result_dict["xb"],
        reconstructed_xyz=sampled_bn3,
        title=f"{model_name}: Sampled {prefix} Batch - Iter {it+1}",
        save_dir=plot_dir,
        grid=grid,
        device=device,
    )
    log_plot_image(writer, f"{prefix}/batch_sampled-0", sample_batch_plot_path, it)
    sampled_cd_avg = sample_batch_cd
    writer.add_scalar(f"{prefix}/sample_cd-0", sampled_cd_avg, it)
    sampled_cd_plot_path = plot_cd_timestep_curve(
        fname=f"{model_name}_iter{it+1}_{prefix}_sampled_cd",
        cds=sampled_cd_trace,
        title=f"{model_name}: Sampled CD vs Reverse Step - Iter {it+1}",
        save_dir=plot_dir,
    )
    log_plot_image(writer, f"{prefix}/sample_cd_curve-0", sampled_cd_plot_path, it)

    sampled_plot_path = visualize_xyz_pip(
        fname=f"{model_name}_iter{it+1}_{prefix}_sampled_batch{batch_id}_sample0",
        original_xyz=result_dict["xb"][0].detach().cpu().numpy().T,
        reconstructed_xyz=sampled_bn3[0].detach().cpu().numpy().T,
        title=f"{model_name}: {prefix} Sampled - Iter {it+1} avg timestep {result_dict['t'][0].item():.1f}",
        save_dir=plot_dir,
        plotlims=None,
        grid=grid,
        device=device,
    )[1]
    log_plot_image(writer, f"{prefix}/sample-0-0", sampled_plot_path, it)


def train_denoiser(
    model,
    train_dataloader,
    val_dataloader,
    device,
    model_name,
    runname,
    num_points,
    iters=200,
    lr=1e-3,
    sample_every=20,
    tb_log_dir="logs",
    plot_dir="/data/palakons/test",
    grid=None,
    n_noise_levels=1,
    target_mode="clean",
    beta=0.5,
    noise_range=(0.0, 1.0),
    noise_mode="fixed",
    gaussian_noise_std=1.0,
    scheduler_mode="fixed",
    num_train_timesteps=1000,
    ddpm_beta_schedule="linear",
    head_lr_factor=0.1,
    lambda_cd=0.0,
    log_train_visuals=False,
    sample_inference_steps=50,
    use_scene_conditioning=False,
    duplications=1,
):
    """Universal training loop for point cloud denoisers with validation.
    
    Args:
        use_scene_conditioning: if True, expect batches with 'wan_vae_latent' key for scene conditioning
    """
    model = model.to(device)
    model.train()
    if True:
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("head"):
                head_params.append(p)
            else:
                backbone_params.append(p)

        opt = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": lr},
                {"params": head_params, "lr": lr * head_lr_factor},
            ]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    val_losses = []
    it = 0
    pbar = trange(iters, desc="Training", leave=True)

    # Setup directories
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    writer = SummaryWriter(tb_log_dir)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar("model/total_params", total_params, 0)
    writer.add_scalar("model/trainable_params", trainable_params, 0)
    print(f"Model {model_name} has {total_params} total parameters, {trainable_params} trainable parameters.")

    writer.add_scalar("training/num_points", num_points, 0)
    writer.add_scalar("training/target_mode", 0 if target_mode == "noise" else 1, 0)  # 0 for noise, 1 for clean    


    if noise_mode == "fixed":
        noise = make_fixed_noise(
            num_points=num_points,
            n_noise_levels=n_noise_levels,
            noise_min=noise_range[0],
            noise_max=noise_range[1],
        )
        print(f"noise_mode=fixed, noise={noise}")
    elif noise_mode == "gaussian":
        noise = None
        print(f"noise_mode=gaussian, gaussian_noise_std={gaussian_noise_std}")
    else:
        raise ValueError(
            f"Unsupported noise_mode={noise_mode}. Choose from ['fixed', 'gaussian']."
        )

    if scheduler_mode == "ddpm":
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=ddpm_beta_schedule,
            prediction_type="epsilon",
            clip_sample=False,
        )
        print(
            f"scheduler_mode=ddpm, num_train_timesteps={num_train_timesteps}, beta_schedule={ddpm_beta_schedule}"
        )
    elif scheduler_mode == "fixed":
        ddpm_scheduler = None
        print(f"scheduler_mode=fixed, beta={beta}")
    else:
        raise ValueError(
            f"Unsupported scheduler_mode={scheduler_mode}. Choose from ['fixed', 'ddpm'].")


    while it < iters:
        for ibtrain,batch in enumerate(train_dataloader):
            opt.zero_grad()
            res = _process_batch(
                batch,
                model,
                beta,
                noise,
                target_mode,
                device,
                noise_mode=noise_mode,
                gaussian_noise_std=gaussian_noise_std,
                scheduler_mode=scheduler_mode,
                ddpm_scheduler=ddpm_scheduler,
                loss_fn=loss_fn,
                use_scene_conditioning=use_scene_conditioning,
                lambda_cd=lambda_cd,
                duplications=duplications,
            )

            total_loss = res["loss"]
            
            total_loss.backward()
            log_gradient_diagnostics(model, writer, it,verbose=(it+1) % sample_every == 0)
            opt.step()
            losses.append(total_loss.item())
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.6f}",
                    "avg": f"{np.mean(losses[-sample_every:]):.6f}",
                }
            )
            
            writer.add_scalar("train/main_loss", res["main_loss"].item(), it)
            if res["cd_loss"] is not None:
                writer.add_scalar("train/cd_loss", res["cd_loss"].item(), it) 
            writer.add_scalar("train/total_loss", total_loss.item(), it)

            if (it + 1) % sample_every == 0:
                train_cd, _ = pt3d_chamfer_distance(
                    res["recon"].transpose(1, 2),
                    res["xb"].transpose(1, 2),
                    batch_reduction=None,
                )
                train_cd_avg = train_cd.mean().item()
                writer.add_scalar("train/cd_recon", train_cd_avg, it)
                writer.add_scalar("train/avg_timestep", res["t"].float().mean().item(), it)
                
                # Log prediction statistics for training (histograms and mean/std)
                log_prediction_statistics(writer, it, res["pred_bn3"], res["target"], split="train", verbose=False)

                if log_train_visuals:
                    # train_plot_path = visualize_xyz_pip(
                    #     fname=f"{model_name}_iter{it+1}_xt_train",
                    #     original_xyz=res["xb"][0].detach().cpu().numpy().T,
                    #     reconstructed_xyz=res["recon"][0].detach().cpu().numpy().T,
                    #     title=f"{model_name}: Train Input - Iter {it+1} xt timestep {res['t'][0].item()}",
                    #     save_dir=plot_dir,
                    #     plotlims=None,
                    #     grid=grid,
                    #     device=device,
                    # )[1]
                    # log_plot_image(writer, "train/recon-0", train_plot_path, it)

                    logmetrics(writer, it, res, 'train',model,model_name,target_mode,plot_dir,grid,device,ibtrain,ddpm_scheduler,sample_inference_steps)

                model.eval()
                with torch.no_grad():
                    # Validation on WHOLE validation set
                    val_loss_sum = 0.0
                    val_count = 0
                    val_cd_sum = 0.0
                    val_cd_count = 0
                    for ib,xb_val in enumerate(val_dataloader):
                        res_val = _process_batch(
                            xb_val,
                            model,
                            beta,
                            noise,
                            target_mode,
                            device,
                            noise_mode=noise_mode,
                            gaussian_noise_std=gaussian_noise_std,
                            scheduler_mode=scheduler_mode,
                            ddpm_scheduler=ddpm_scheduler,
                            loss_fn=loss_fn,
                            use_scene_conditioning=use_scene_conditioning,
                        )
                        out_val = res_val["pred_bn3"]
                        val_loss = res_val["loss"]
                        B_val = out_val.shape[0]

                        val_loss_sum += val_loss.item() * B_val
                        val_count += B_val

                        val_cd, _ = pt3d_chamfer_distance(
                            res_val["recon"].transpose(1, 2),
                            res_val["xb"].transpose(1, 2),
                            batch_reduction=None,
                        )
                        val_cd_sum += val_cd.sum().item()
                        val_cd_count += val_cd.numel()

                        if ib == 0: #sample only the first batch for visualization
                            log_prediction_statistics(writer, it, res_val["pred_bn3"], res_val["target"], split="val",verbose=True)

                            logmetrics(writer, it, res_val, 'val',model,model_name,target_mode,plot_dir,grid,device,ib,ddpm_scheduler,sample_inference_steps)

                            # batch_cd, batch_plot_path = visualize_xyz_batch_grid(
                            #     fname=f"{model_name}_iter{it+1}_val_batch",
                            #     original_xyz=res_val["xb"],
                            #     reconstructed_xyz=res_val["recon"],
                            #     title=f"{model_name}: Val Batch - Iter {it+1} timestep {res_val['t'].float().mean().item():.1f}",
                            #     save_dir=plot_dir,
                            #     grid=grid,
                            #     device=device,
                            # )
                            # log_plot_image(writer, "val/batch_recon-0", batch_plot_path, it)

                            # # Save visualizations using the existing plotting function (validation views)
                            # for ibb in range(res_val["xt_bn3"].shape[0]):
                            #     gt_np = res_val["xb"][ibb].cpu().numpy().T # n,3
                            #     recon_np = res_val["recon"][ibb].cpu().numpy().T # n,3
                                
                            #     if ibb == 0:
                            #         cd, save_path = visualize_xyz_pip( #plot recon vs xb
                            #             fname=f"{model_name}_iter{it+1}_xt_val_{ib}_{ibb}",
                            #             original_xyz=gt_np,
                            #             reconstructed_xyz=recon_np,
                            #             title=f"{model_name}: Val Input - Iter {it+1} xt timestep {res_val['t'].float().mean().item():.1f}",
                            #             save_dir=plot_dir,
                            #             plotlims=None,
                            #             grid=grid,
                            #             device=device,
                            #         )
                            #         log_plot_image(writer, "val/recon-0-0", save_path, it)

                            # if ( ddpm_scheduler is not None
                            # ):
                            #     sampled_bn3, sampled_cd_trace, sampled_step_trace = sample_ddpm_batch(
                            #         model,
                            #         ddpm_scheduler,
                            #         res_val["xb"],
                            #         device,
                            #         condition=res_val.get("scene_cond"),
                            #         target_mode=target_mode,
                            #         num_inference_steps=sample_inference_steps,
                            #         return_trace=True,
                            #     )
                            #     sampled_cd, _ = pt3d_chamfer_distance(
                            #         sampled_bn3.transpose(1, 2),
                            #         res_val["xb"].transpose(1, 2),
                            #         batch_reduction=None,
                            #     )
                            #     sampled_cd_avg = sampled_cd.mean().item()
                            #     writer.add_scalar("val/sample_cd-0", sampled_cd_avg, it)
                            #     sampled_cd_plot_path = plot_cd_timestep_curve(
                            #         fname=f"{model_name}_iter{it+1}_sampled_cd",
                            #         cds=sampled_cd_trace,
                            #         title=f"{model_name}: Sampled CD vs Reverse Step - Iter {it+1}",
                            #         save_dir=plot_dir,
                            #     )
                            #     log_plot_image(writer, "val/sample_cd_curve-0", sampled_cd_plot_path, it)

                            #     sampled_plot_path = visualize_xyz_pip(
                            #         fname=f"{model_name}_iter{it+1}_sampled_val",
                            #         original_xyz=res_val["xb"][0].detach().cpu().numpy().T,
                            #         reconstructed_xyz=sampled_bn3[0].detach().cpu().numpy().T,
                            #         title=f"{model_name}: Val Sampled - Iter {it+1} avg timestep {res_val['t'].float().mean().item():.1f}",
                            #         save_dir=plot_dir,
                            #         plotlims=None,
                            #         grid=grid,
                            #         device=device,
                            #     )[1]
                            #     log_plot_image(writer, "val/sample-0-0", sampled_plot_path, it)

                    val_loss_avg = val_loss_sum / val_count if val_count > 0 else 0.0
                    val_losses.append(val_loss_avg)
                    # Log metrics to TensorBoard
                    writer.add_scalar("val/loss", val_loss_avg, it)
                    writer.add_scalar("val/cd_recon", val_cd_sum / val_cd_count if val_cd_count > 0 else 0.0, it)

                model.train()

            it += 1
            if it >= iters:
                break
    pbar.close()
    writer.close()
    return losses, val_losses


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create timestamp suffix for this run
    runname = f"{args.exp_name}"

    # Setup log and plot directories
    tb_base_dir = f"/home/palakons/logs/tb_log/ptv3_test/{runname}"
    plot_base_dir = f"/data/palakons/ptv3_test/{runname}"
    # plot_checkpoint_dir = f"/data/palakons/ptv3_test/{runname}"

    num_samples = args.num_scenes if args.num_scenes is not None else args.batch_size * 10
    data = make_synthetic_dataset(
        num_samples, args.num_points, device=device, seed=args.seed
    )

    # Split into train and validation sets
    num_train = int(num_samples * (1 - args.val_split))
    num_val = num_samples - num_train
    train_data = data[:num_train]
    val_data = data[num_train:]

    train_dataset = SyntheticPointDataset(train_data)
    val_dataset = SyntheticPointDataset(val_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # No test set: only train and validation used for visualization

    print(f"Device: {device}")
    print(
        f"Train samples: {num_train}, Val samples: {num_val}, batch: {args.batch_size}, points: {args.num_points}"
    )
    print(f"Experiment: {runname}")
    print(f"TensorBoard logs: {tb_base_dir}")
    print(f"Plot dir: {plot_base_dir}")

    # Model selection
    if args.model == "ptv3":
        model = PTv3Dnsr(
            n_in_channels=3,
            context_channels=args.context_channels,
            out_channels=3,
            grid_size=args.grid_size,
            use_cpe=args.ptv3_use_cpe,
            n_stages=2,
            project_coord_dim=0,
            use_head=args.use_head,
        )
        print("Using PTv3Dnsr (PTv3)")
        print(f"PTv3 grid_size: {args.grid_size}, use_cpe: {args.ptv3_use_cpe}")
    elif args.model == "mlp":
        model = MLPDenoiser(
            in_channels=3,
            out_channels=3,
            context_channels=args.context_channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.blocks,
            coord_projector_dim=args.coord_projector_dim,
            scene_embed_dim=args.scene_embed_dim,
        )
        print("Using MLPDenoiser (per-point MLP)")
    else:
        # fallback: minimal point denoiser if present
        try:
            model = MinimalPointDenoiser(hidden_channels=args.hidden_dim, time_embed_dim=args.tdim, num_blocks=args.blocks)
            print("Using MinimalPointDenoiser (fallback)")
        except Exception:
            model = PTv3Dnsr(
                n_in_channels=3,
                context_channels=args.context_channels,
                out_channels=3,
                grid_size=args.grid_size,
                use_cpe=args.ptv3_use_cpe,
                n_stages=2,
                project_coord_dim=0,
                use_head=args.use_head,
            )
            print("Fallback to PTv3Dnsr (ptv3)")

    train_losses, val_losses = train_denoiser(
        model,
        train_dataloader,
        val_dataloader,
        device,
        "ptv3",
        runname,
        num_points=args.num_points,
        iters=args.iters,
        lr=args.lr,
        sample_every=args.sample_every,
        tb_log_dir=tb_base_dir,
        plot_dir=plot_base_dir,
        grid=args.grid_size,
        n_noise_levels=args.n_noise_levels,
        target_mode=args.target_mode,
        beta = args.beta,
        noise_range=args.noise_range,
        noise_mode=args.noise_mode,
        gaussian_noise_std=args.gaussian_noise_std,
        scheduler_mode=args.scheduler_mode,
        num_train_timesteps=args.num_train_timesteps,
        ddpm_beta_schedule=args.ddpm_beta_schedule,
        head_lr_factor=args.head_lr_factor,
        lambda_cd=args.lambda_cd,
        log_train_visuals=args.log_train_visuals,
    )
    if len(train_losses) > 0:
        print(
            f"Final train loss (ptv3): {train_losses[-1]:.6f}, Final val loss: {(val_losses[-1] if val_losses else np.nan):.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="Experiment name for run identification",
    )
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (0.0-1.0)",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Explicit number of synthetic scenes to generate. Use 2 with --val_split 0.5 for a 1-train/1-val split.",
    )
    parser.add_argument("--context_channels", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--tdim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.5, help="Noise mixing factor for the fixed-noise experiment.")
    parser.add_argument(
        "--n_noise_levels",
        type=int,
        default=1,
        help="Number of noise levels for fixed-noise mode (ignored when --noise_mode gaussian).",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="fixed",
        choices=["fixed", "gaussian"],
        help="Noise source: fixed template (backward-compatible) or random Gaussian noise.",
    )
    parser.add_argument(
        "--gaussian_noise_std",
        type=float,
        default=1.0,
        help="Std for random Gaussian noise when --noise_mode gaussian.",
    )
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="fixed",
        choices=["fixed", "ddpm"],
        help="Corruption schedule: fixed beta mixing (default) or DDPM random timestep schedule.",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of training timesteps for DDPM scheduler (used when --scheduler_mode ddpm).",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"],
        help="Diffusers DDPM beta schedule type (used when --scheduler_mode ddpm).",
    )
    parser.add_argument(
        "--noise_range",
        type=float, nargs=2,
        default=[0.0, 1.0],
        help="Min and max noise values for the fixed-noise experiment, inclusive. Should be two floats: --noise_range 0.0 1.0",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="clean",
        choices=["clean", "noise"],
        help="Whether the model predicts the clean points or the corruption tensor.",
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.2, help="Grid size for PTv3Dnsr"
    )
    parser.add_argument(
        "--ptv3_use_cpe",
        action="store_true",
        default=False,
        help="Enable CPE in PTv3 (use to test with CPE).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ptv3",
        choices=["ptv3", "mlp", "minimal"],
        help="Model to use: 'ptv3' (default), 'mlp' (MLP per-point denoiser), or 'minimal' (existing minimal denoiser).",
    )
    parser.add_argument(
        "--head_lr_factor",
        type=float,
        default=1,
        help="Learning rate factor for head parameters relative to backbone. Default 1 means same LR; <1 means smaller LR for head.",
    )
    parser.add_argument(
        "--lambda_cd",
        type=float,
        default=0.0,
        help="Weight for Chamfer Distance loss. When > 0, adds CD loss to MSE: total_loss = mse + lambda_cd * cd. Recommended: 0.01-0.05.",
    )
    parser.add_argument(
        "--log_train_visuals",
        action="store_true",
        default=False,
        help="Also write a train reconstruction image to TensorBoard at each print interval.",
    )
    parser.add_argument(
        "--no_head",
        action="store_false",
        dest="use_head",
        help="Remove head projection; decoder outputs out_channels (3) directly. Eliminates gradient bottleneck.",
    )
    parser.add_argument(
        "--coord_projector_dim",
        type=int,
        default=0,
        help="If > 0, adds a learnable linear projection of input coordinates to this dimension, concatenated to the model input. Can provide a bottleneck for testing gradient flow through coordinate conditioning.", 
    )
    args = parser.parse_args()
    if args.noise_mode == "fixed":
        assert args.n_noise_levels <= args.num_points, "n_noise_levels must be <= num_points"    
    # Enforce single-level for random-t (DDPM scheduler): multi-level is ill-posed without per-point conditioning
    if args.scheduler_mode == "ddpm" and args.n_noise_levels > 1:
        print(f"[WARNING] scheduler_mode='ddpm' with n_noise_levels={args.n_noise_levels} is ill-posed (noise-level identity not geometrically observable).")
        print(f"[INFO] Auto-correcting: n_noise_levels -> 1 for DDPM scheduler.")
        args.n_noise_levels = 1
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run(args)
