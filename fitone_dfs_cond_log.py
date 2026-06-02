import re
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os
from tqdm import tqdm, trange
import numpy as np
from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance
from pytorch3d.ops import knn_points

from logging_utils import ExperimentLogger
from models import FullSetTransformerDenoiser, make_model
from diffusers import DDPMScheduler
from plot_utils import plot_pc_batch, azm_easing, calculate_pointset_stat
from io_dataset import (
    make_various_pc,
    make_man_pc,
    normalize_data,
    load_checkpoint,
    save_checkpoint,
    duplicate_batch,
    make_dataset,
    save_point_sample,
)
def debug_batch(x, pred=None, target=None, loss=None, name=""):
    print(f"\n--- DEBUG {name} ---")
    print("x:", tuple(x.shape), x.dtype, x.device, "grad?", x.requires_grad)
    if pred is not None:
        print("pred:", tuple(pred.shape), pred.dtype, pred.device, "grad?", pred.requires_grad)
        print("pred finite:", torch.isfinite(pred).all().item())
    if target is not None:
        print("target:", tuple(target.shape), target.dtype, target.device, "grad?", target.requires_grad)
        print("target finite:", torch.isfinite(target).all().item())
    if loss is not None:
        print("loss:", loss, "finite:", torch.isfinite(loss).item())
        print("loss item:", loss.item())

    print("------------------\n")

def check_model(model):
    print(f"Model: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    print(f"Total gradient norm: {math.sqrt(total):.4f}")
def check_tensor(name, x):
    print(
        name,
        "shape=", tuple(x.shape),
        "type=", type(x),
        "dtype=", x.dtype,
        "device=", x.device,
        "requires_grad=", x.requires_grad,
        "min=", x.min().item() if x.numel() else None,
        "max=", x.max().item() if x.numel() else None,
        "nan=", torch.isnan(x).any().item() if x.is_floating_point() else False,
    )
def chamfer_xyz_with_matched_attrs(
    pred,
    gt,
    bidirectional_attr=False,
):
    """
    pred: [B, N, 5] = x,y,z,doppler,rcs
    gt:   [B, M, 5]
    Computes:
    - Chamfer Distance using xyz only
    - Doppler/RCS loss using xyz nearest-neighbor matching
    Returns:
        total_attr_loss, loss_dict
    """
    pred_xyz = pred[..., :3]
    gt_xyz = gt[..., :3]
    # Forward NN: each pred point -> nearest GT point
    # dists: [B, N, 1], idx: [B, N, 1]
    fwd = knn_points(pred_xyz, gt_xyz, K=1, return_nn=False)
    fwd_dists = fwd.dists[..., 0]  # [B, N]
    fwd_idx = fwd.idx[..., 0]  # [B, N]
    # Backward NN: each GT point -> nearest pred point
    bwd = knn_points(gt_xyz, pred_xyz, K=1, return_nn=False)
    bwd_dists = bwd.dists[..., 0]  # [B, M]
    bwd_idx = bwd.idx[..., 0]  # [B, M]
    # Chamfer xyz, symmetric
    cd_xyz = fwd_dists.mean() + bwd_dists.mean()
    # Gather GT attributes matched to each predicted point
    B, N, D = pred.shape
    batch_idx = torch.arange(B, device=pred.device)[:, None]
    gt_matched_to_pred = gt[batch_idx, fwd_idx]  # [B, N, 5]
    doppler_fwd = F.mse_loss(
        pred[..., 3:4],
        gt_matched_to_pred[..., 3:4],
    )
    rcs_fwd = F.mse_loss(
        pred[..., 4:5],
        gt_matched_to_pred[..., 4:5],
    )
    if bidirectional_attr:
        # Also compare each GT point to nearest predicted point
        pred_matched_to_gt = pred[batch_idx, bwd_idx]  # [B, M, 5]
        doppler_bwd = F.mse_loss(
            pred_matched_to_gt[..., 3:4],
            gt[..., 3:4],
        )
        rcs_bwd = F.mse_loss(
            pred_matched_to_gt[..., 4:5],
            gt[..., 4:5],
        )
        doppler_loss = 0.5 * (doppler_fwd + doppler_bwd)
        rcs_loss = 0.5 * (rcs_fwd + rcs_bwd)
    else:
        doppler_loss = doppler_fwd
        rcs_loss = rcs_fwd
    loss_dict = {
        "cd_xyz": cd_xyz,
        "doppler_attr_loss": doppler_loss,
        "rcs_attr_loss": rcs_loss,
    }
    return loss_dict


def reconstruct_x0(pred, x_t, t, scheduler, prediction_type):
    if prediction_type == "sample":
        return pred
    alpha_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1)
    return (x_t - torch.sqrt(1 - alpha_bar) * pred) / torch.sqrt(alpha_bar)


def make_run_id(args):
    cond_spec = f"_{args.cond_method}" if args.cond_method != "none" else ""

    dop_rcs_loss_weight = (
        f"_{args.loss_weight_position:1.3f}-{args.loss_weight_doppler:1.3f}-{args.loss_weight_rcs:1.3f}"
        if args.train_rcs_doppler
        else ""
    )

    model_spec = f"_dim{64}" if args.model_name == "SetTxDnsr" else ""

    shape_spec = f"_{args.data_file}" if args.shape_name.startswith("realman") else ""

    cd_spec = f"_cdmd{args.cd_mode}" if args.lambda_cd > 0 else ""

    # --prediction_type epsilon|sample
    # --sampler ddpm|ddim

    return f"{args.model_name}{model_spec}_{args.shape_name}{shape_spec}_train_sc{args.n_scene}_N{args.N}_B{args.B}_T{args.T}-{args.T_infer}_{args.prediction_type}-{args.sampler}_it{args.ddpm_iteration}_{args.cond_mode}{cond_spec}_weight{dop_rcs_loss_weight}_lmse{args.lambda_mse:1.3f}_lcd{args.lambda_cd:1.3f}{cd_spec}"


@torch.no_grad()
def p_sample_loop(
    model,
    shape,
    scheduler,
    num_inference_steps=None,
    device="cuda",
    condition=None,
    seed=42,
):
    # print(f"recorded model mode")
    prev_mode = model.training
    # print(f"switching model to eval mode for sampling")
    model.eval()
    # print(f"rand")
    B = shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(shape, device=device, generator=generator)

    steps = (
        num_inference_steps
        if num_inference_steps is not None
        else scheduler.config.num_train_timesteps
    )
    # print(f"set step")
    scheduler.set_timesteps(steps, device=device)  # num_inference_steps
    if condition is not None:
        condition = condition.to(device)
    # print(f"loop")
    try:
        for t_step in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
            # print(f"make tensor")
            t_tensor = torch.full((B,), t_step.item(), device=device, dtype=torch.long)
            # print(f"model pred")
            model_output = model(x, t_tensor, condition=condition)
            # print(f"step")
            x = scheduler.step(model_output, t_step, x).prev_sample
    finally:
        model.train(prev_mode)
    return x


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a point cloud diffusion model on a single shape."
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of points in the point cloud"
    )
    parser.add_argument("--B", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--n_scene",
        type=int,
        default=1,
        help="Number of scenes to use from the dataset (for multi-scene datasets)",
    )
    parser.add_argument(
        "--T", type=int, default=1000, help="Number of diffusion steps during training"
    )
    parser.add_argument(
        "--T_infer",
        type=int,
        default=50,
        help="Number of diffusion steps during inference",
    )
    parser.add_argument(
        "--ddpm_iteration",
        type=int,
        default=10000 * 3,
        help="Number of training iteration",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for the output GIF"
    )
    parser.add_argument(
        "--cond_mode",
        type=str,
        default="pdnorm_only",
        help="Time conditioning mode for the model: 'pdnorm_only', 'feat_add', 'hybrid', 'feat_concat'",
    )
    parser.add_argument(
        "--shape_name",
        type=str,
        default="realman",
        help="Shape to train on: 'realman' or 'various'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Whether to run training or just inference test. Options: 'train', 'test'",
    )
    parser.add_argument(
        "--cond_method",
        type=str,
        default="scene_id",
        help="Conditioning method for multi-scene training: 'scene_id' (simple learnable embedding), 'wan' (use Wan's VAE latent)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="man-mini",
        help="Data file to use for MANDataset when shape_name is 'realman'. Options: 'man-mini', 'man-full', or path to custom data file compatible with MANDataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PTv3Dnsr",
        help="Model architecture to use: 'SetTxDnsr', 'PTv3Dnsr'",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save and load checkpoints",
    )
    parser.add_argument(
        "--train_rcs_doppler",
        action="store_true",
        help="Whether to train on RCS and Doppler features",
    )

    parser.add_argument(
        "--loss_weight_position",
        type=float,
        default=1.0,
        help="Loss weight for position feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_doppler",
        type=float,
        default=1.0,
        help="Loss weight for Doppler feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_rcs",
        type=float,
        default=1.0,
        help="Loss weight for RCS feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--num_train_log",
        type=int,
        default=100,
        help="Logging frequency (in steps) during training",
    )
    parser.add_argument(
        "--num_checkpoints_save",
        type=int,
        default=1000,
        help="Checkpoint saving frequency (in steps) during training",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=1000,
        help="Number of evaluation points to use during training",
    )
    parser.add_argument(
        "--lambda_cd",
        type=float,
        default=0.0,
        help="Weight for the CD loss term",
    )
    parser.add_argument(
        "--cd_mode",
        type=str,
        default="xyz_attr",
        help="Chamfer Distance mode: 'xyz_attr' (use chamfer with attribute loss), 'cd5d' (treat doppler and rcs as extra dimensions in chamfer)",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=1.0,
        help="Weight for the MSE loss term",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The type of prediction the model makes: 'epsilon' (predict noise), 'sample' (predict x0 directly)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        help="The sampling method to use during inference: 'ddpm' (standard DDPM sampling), 'ddim' (DDIM sampling)",
    )
    args = parser.parse_args()

    return args


def train_eval_step(
    model,
    optimizer,
    scheduler,
    x0sbn3_norm_rep,
    scene_condition_rep,
    T,
    B,
    device,
    loss_weights,
    inout_dim,
    is_train=True,
    lambda_mse=1.0,
    lambda_cd=0.0,
    cd_mode = "xyz_attr",
    prediction_type="epsilon",
):
    num_samples = x0sbn3_norm_rep.shape[0]

    if is_train:
        model.train()
        # idx = torch.randperm(x0sbn3_norm_rep.shape[0], device="cpu")
        idx = torch.randint(0, num_samples, (B,), device="cpu")
    else:
        model.eval()
        idx = torch.arange(
        #     x0sbn3_norm_rep.shape[0], device="cpu"
        # )  # use the same order for evaluation for consistency
        idx = torch.arange(min(B, num_samples), device="cpu")

    # x0 = x0sbn3_norm_rep[idx][:B].to(device)  # [B, N, 3]
    # cond = scene_condition_rep[idx][:B].to(device)

    x0_cpu = x0sbn3_norm_rep[idx]
    cond_cpu = scene_condition_rep[idx]
    x0 = x0_cpu.to(device, non_blocking=True)
    cond = cond_cpu.to(device, non_blocking=True)

    t = torch.randint(0, T, (B,), device=device)
    noise = torch.randn_like(x0)

    x_t = scheduler.add_noise(x0, noise, t)
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = x0
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    with torch.set_grad_enabled(is_train):
        pred = model(x_t, t, condition=cond)

    assert pred.shape == target.shape
    assert pred.device == target.device
    
    loss_dict = {
        "pred_mean": pred.mean().item(),
        "pred_std": pred.std().item(),
        "target_mean": target.mean().item(),
        "target_std": target.std().item(),
    }
    loss = torch.zeros((), device=device)

    if lambda_mse > 0:  # include MSE loss

        loss_mse_position = F.mse_loss(pred[..., :3], target[..., :3])  

        # loss_mse_position = F.mse_loss(pred[..., :3], noise[..., :3]) 
        loss_dict.update({"mse_3d_loss": loss_mse_position.item()})

        if inout_dim > 3:
            loss_mse_doppler = F.mse_loss(pred[..., 3 : 3 + 1], target[..., 3 : 3 + 1])
            loss_mse_rcs = F.mse_loss(pred[..., 3 + 1 :], target[..., 3 + 1 :])

            # loss_mse_doppler = F.mse_loss(pred[..., 3:3+1], noise[..., 3:3+1])
            # loss_mse_rcs = F.mse_loss(pred[..., 3+1:], noise[..., 3+1:])
            loss_mse = (
                loss_weights["position"]  *loss_mse_position
                + loss_weights["doppler"] * loss_mse_doppler
                + loss_weights["rcs"] * loss_mse_rcs
            )
            loss_dict["mse_doppler_loss"] = loss_mse_doppler.item()
            loss_dict["mse_rcs_loss"] = loss_mse_rcs.item()
        else:
            loss_mse = loss_weights["position"]  *loss_mse_position
        loss += lambda_mse * loss_mse

    if lambda_cd > 0.0:  # include CD loss

        with torch.set_grad_enabled(is_train):
            scheduler.set_timesteps(
                T, device=device
            )  # set timesteps to max T for get_x0_from_noise

            alpha_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1)
            x0_hat_o = (x_t - torch.sqrt(1 - alpha_bar) * pred) / torch.sqrt(alpha_bar)
            x0_hat = reconstruct_x0(pred, x_t, t, scheduler, prediction_type)
            if prediction_type == "epsilon":
                assert torch.allclose(
                    x0_hat, x0_hat_o, atol=1e-5
                ), f"x0_hat from reconstruct_x0 and x0_hat from direct calculation do not match. max diff: {(x0_hat - x0_hat_o).abs().max().item():.4e}"

        assert x0_hat.is_cuda, f"x0_hat device {x0_hat.device} is not on CUDA"
        assert x0.is_cuda, f"x0 device {x0.device} is not on CUDA"
        # assert requires grad is true
        assert (
            x0_hat.requires_grad == is_train
        ), f"requires_grad mismatch: x0_hat requires_grad {x0_hat.requires_grad}, expected {is_train}"
        assert (
            pred.requires_grad == is_train
        ), f"requires_grad mismatch: pred requires_grad {pred.requires_grad}, expected {is_train}"

        if cd_mode == "xyz_attr":
            cd_loss_dict = chamfer_xyz_with_matched_attrs(
                x0_hat,
                x0,
                bidirectional_attr=False,
            )

            loss_dict["cd_3d_loss"] = cd_loss_dict["cd_xyz"].item()
            loss_dict["cd_doppler_loss"] = cd_loss_dict["doppler_attr_loss"].item()
            loss_dict["cd_rcs_loss"] = cd_loss_dict["rcs_attr_loss"].item()
            loss += ( loss_weights["position"]  * cd_loss_dict["cd_xyz"]
                + loss_weights["doppler"]  * cd_loss_dict["doppler_attr_loss"]
                + loss_weights["rcs"] * cd_loss_dict["rcs_attr_loss"]
            )* lambda_cd
        elif cd_mode == "cd5d":
            loss_cd5d = pt3d_chamfer_distance(x0_hat, x0)[0]  # Chamfer Distance between predicted x0 and true x0
            # loss_cd_3d = pt3d_chamfer_distance(x0_hat[..., :3], x0[..., :3])[0]  # Chamfer Distance for position only, ignoring doppler/rcs
            loss_dict["cd_5d_loss"] = loss_cd5d.item()
            loss += lambda_cd * loss_cd5d
    assert torch.isfinite(loss).all()

    loss_dict["total_loss"] = loss.item()
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, loss_dict


if __name__ == "__main__":
    args = parse_args()
    # Example setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shape_name = args.shape_name
    data_file = args.data_file
    T = args.T
    T_infer = args.T_infer
    ddpm_iteration = args.ddpm_iteration
    fps = args.fps
    N = args.N
    B = args.B
    n_scene = args.n_scene
    cond_mode = args.cond_mode
    cond_method = args.cond_method
    model_name = args.model_name
    loss_weights = {
        "doppler": args.loss_weight_doppler if args.train_rcs_doppler else None,
        "rcs": args.loss_weight_rcs if args.train_rcs_doppler else None,
        "position": args.loss_weight_position,
    }

    (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (
        x0sbn3_eval_norm,
        cond_eval_norm,
        doppler_eval_norm,
        rcs_eval_norm,
    ) = make_dataset(
        shape_name=shape_name,
        n_train_scene=n_scene,
        N=N,
        device=device,
        data_file=data_file,
        cond_method=cond_method,
        cond_mode=cond_mode 
    )
    print(
        f"shapes after dataset creation: x0sbn3_norm {x0sbn3_train_norm.shape}, cond {cond_train_norm.shape if cond_train_norm is not None else None}, doppler_norm {doppler_train_norm.shape if doppler_train_norm is not None else None}, rcs_norm {rcs_train_norm.shape if rcs_train_norm is not None else None}"
    )

    if args.train_rcs_doppler:
        x0sbn3_train_norm = torch.cat(
            [x0sbn3_train_norm, doppler_train_norm, rcs_train_norm], dim=-1
        )  # [B,N,5]
        x0sbn3_eval_norm = torch.cat(
            [x0sbn3_eval_norm, doppler_eval_norm, rcs_eval_norm], dim=-1
        )  # [B,N,5]
        inout_dim = 5
    else:
        inout_dim = 3

    model = make_model(device, args)
    run_id = make_run_id(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if args.sampler == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=T,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type=args.prediction_type,  # "epsilon" or "sample"
        )
    elif args.sampler == "ddim":
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler(
            num_train_timesteps=T,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            steps_offset=1,  # to match the noise schedule of DDPMScheduler
            prediction_type=args.prediction_type,  # "epsilon" or "sample"
        )
    else:
        raise ValueError(f"Unsupported sampler: {args.sampler}")

    system_key = "ddpm_cond_5"
    data_dir = f"/data/palakons/{system_key}/{run_id}"
    tb_dir = f"/home/palakons/logs/tb_log/{system_key}/{run_id}"
    temp_dir = f"{data_dir}/temp"
    samples_dir = f"{data_dir}/samples"
    checkpoint_dir = f"/data/palakons/{system_key}/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_{run_id}.pt")
    # creat dir, nested if not exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Load checkpoint if exists
    start_step, config = load_checkpoint(
        model,
        optimizer,
        scheduler,
        re.sub(r"it\d+", "it*", checkpoint_path),
        ddpm_iteration,
        device=device,
    )
    # start_step, config = 0,{}
    if config:
        assert (
            config.get("N", N) == N
        ), f"Checkpoint N {config.get('N')} does not match current N {N}"
        assert (
            config.get("T", T) == T
        ), f"Checkpoint T {config.get('T')} does not match current T {T}"
        assert (
            config.get("T_infer", T_infer) == T_infer
        ), f"Checkpoint T_infer {config.get('T_infer')} does not match current T_infer {T_infer}"
        assert (
            config.get("B", B) == B
        ), f"Checkpoint B {config.get('B')} does not match current B {B}"
        # assert config.get("cond_method", cond_method) == cond_method, f"Checkpoint cond_method {config.get('cond_method')} does not match current cond_method {cond_method}"

    print(f"mode: {args.mode}")

    if args.mode == "interpolate":
        assert (
            shape_name == "various"
        ), f"Interpolation test is designed for 'various' shape_name to show effect of conditioning. Got shape_name: {shape_name}"
        """
        interpolate, "various" 10 steps, from -1 to 1 condition, show effect on generated samples
        """
        print(
            f"Running interpolation test... model at step: {start_step}, config: {config}"
        )
        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        shapes = (1, N, inout_dim)
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        B = 1
        expanded_cond = torch.ones((B, 1), device=device)  # Same condition for both
        output = {}
        n_interpolate = 16
        with torch.no_grad():
            lambs = torch.linspace(-1, -0.333333, n_interpolate)
            for lamb in lambs:
                expanded_cond[0] = lamb
                print(
                    f"[Interpolation Test] Generating sample with condition: {expanded_cond.cpu().numpy().flatten()}"
                )
                pred_x = p_sample_loop(
                    model,
                    shapes,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=expanded_cond,
                )
                output[lamb.item()] = pred_x

        pred = torch.stack([output[lamb.item()] for lamb in lambs], dim=0).squeeze(
            1
        )  # [n_interpolate, N, 3]
        print(f"Generated interpolated samples with shape: {pred.shape}")
        btitles = [f"cond:{lamb:.2f}" for lamb in lambs]
        plot_pc_batch(
            pred,
            None,
            title=f"Interpolation Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}",
            fname=f"{temp_dir}/../sample/interpolation_test.png",
            azm=45,
            elev=30,
            batch_titles=btitles,
        )

    elif args.mode == "eval":
        assert shape_name.startswith(
            "realman"
        ), f"Evaluation test is designed for 'realman' shape_name with multiple scenes to show effect of conditioning. Got shape_name: {shape_name}"
        assert (
            cond_method == "wan"
        ), f"Evaluation test is designed for 'wan' conditioning method to use Wan's VAE latent for conditioning. Got cond_method: {cond_method}"
        """get 1 more than n_scene samples, 
        sampel the model with each condition, plot will show CD, the frist n_scene samples should have low CD, the last one should have high CD if the model is learning the conditioning correctly
        """
        print(
            f"Running evaluation test... model at step: {start_step}, config: {config}"
        )
        model.eval()
        try:
            print(f"scene_strength {model.scene_strength.item():.4f}")
        except AttributeError:
            pass
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        expanded_cond_eval = (
            wan_cond_eval.view(x0sbn3_eval_norm.shape[0], -1) / wan_cond.abs().max()
        )  # divide by max abs value OF THE TRAINING SET
        expanded_cond_train = (
            wan_cond.view(wan_cond.shape[0], -1) / wan_cond.abs().max()
        )  # divide by max abs value OF THE TRAINING SET
        print(f"shapes wan_cond; {wan_cond.shape}")  # torch.Size([320, 16, 2, 60, 104])

        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)

        with torch.no_grad():
            for name, expanded_cond, gt in zip(
                ["Eval", "Train"],
                [expanded_cond_eval, expanded_cond_train],
                [x0sbn3_eval_norm, x0sbn3_train_norm],
            ):  # first eval condition, then training condition as control
                B = gt.shape[0]
                shapes = (B, N, inout_dim)
                print(
                    f"[Evaluation Test] {name} Generating sample: shape: {shapes}, condition: {expanded_cond.shape} (normalized using training set max abs value {wan_cond.abs().max().item():.4f})"
                )
                pred_x = p_sample_loop(
                    model,
                    shapes,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=expanded_cond,
                )
                # print(f"shapes pred_x: {pred_x.shape}, gt: {gt.shape}")
                cd = pt3d_chamfer_distance(pred_x.cpu(), gt.cpu())[
                    0
                ]  # Compute Chamfer Distance for each sample
                plot_pc_batch(
                    pred_x[:12],
                    gt=gt[:12],
                    title=f"{name} {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B} scene {gt.shape[0]} overall CD: {cd.item():.1e}",
                    fname=f"{temp_dir}/../sample/evaluation_{name}_{model_name}_{shape_name}.png",
                    azm=45,
                    elev=30,
                )
                print(
                    f"Evaluation test completed. Sample saved at: {temp_dir}/../sample/evaluation_{name}_{model_name}_{shape_name}.png"
                )

    elif (
        args.mode == "test_cond_infer"
    ):  # testing if 2 conditiong gives different output,
        print(
            f"Running inference test... model at step: {start_step}, config: {config}"
        )

        model.eval()
        print(f"scene_strength {model.scene_strength.item():.4f}")

        B = 2
        cond_train_norm = torch.ones((B, 1), device=device)  # Same condition for both
        for i in range(40):
            if i >= 20:
                cond_train_norm[1] = (
                    -1.0
                )  # Change condition for the second sample to see the effect
            t = torch.randint(0, T, (1,), device=device).repeat(B)
            x_t = torch.randn((1, N, 3), device=device).repeat(B, 1, 1)
            diff_xt = (x_t[0] - x_t[1]).abs().mean()

            pred = model(x_t, t, condition=cond_train_norm)

            diff = (pred[0] - pred[1]).abs().mean()
            print(
                f"  Iter {i}: T={t.cpu().numpy().flatten()}, cond={cond_train_norm.cpu().numpy().flatten()}: diff b4 condition: {diff_xt.item():.4f}, diff: {diff.item():.4f}"
            )

    elif args.mode == "sample":

        n_sampling = 12
        shapes = (n_sampling, N, inout_dim)
        picked_indices = torch.randint(0, n_scene, (n_sampling,), device=device)
        print(f"Picked scene indices for sampling: {picked_indices.shape}")

        scene_id_condition = picked_indices.float() / max(
            n_scene - 1, 1
        )  # [n_sampling], normalized to [0,1]
        scene_id_condition = scene_id_condition * 2 - 1
        expanded_cond = scene_id_condition.unsqueeze(-1)  # [n_sampling, 1]
        os.makedirs(f"{temp_dir}/../sample", exist_ok=True)
        with torch.no_grad():
            print(
                f"[Inference Test] Generating sample: shape: {shapes}, condition: {expanded_cond}"
            )
            pred_x = p_sample_loop(
                model,
                shapes,
                scheduler,
                num_inference_steps=T_infer,
                device=device,
                condition=expanded_cond,
            )
            plot_pc_batch(
                pred_x,
                gt=x0sbn3_train_norm[picked_indices],
                title=f"Inference Test {model_name} {shape_name} N:{N} T:{T} Inf:{T_infer} B:{B}",
                fname=f"{temp_dir}/../sample/inference_test.png",
                azm=45,
                elev=30,
                batch_titles=[
                    f"cond:{i:.2f}" for i in scene_id_condition.cpu().numpy()
                ],
            )
            print(
                f"Inference test completed. Sample saved at: {temp_dir}/../sample/inference_test.png"
            )
    elif args.mode == "permutation":
        print(
            f"Running permutation test... model at step: {start_step}, config: {config}"
        )
        """
        cann the model 2 times, with the second time have the input to the model permuted in a different order, [B,N,3] (the "N" dimension is permuted)
        """
        print(f"tesing {model_name} permutation invariance...")
        model.eval()
        B = 1
        condition = (
            torch.zeros_like(wan_cond[:B])
            if cond_method == "wan"
            else torch.zeros((B, 1), device=device)
        )
        # set seed
        torch.manual_seed(42)
        for i in range(10):
            with torch.no_grad():
                t = torch.randint(0, T, (1,), device=device).repeat(B) * 0 + T // 2
                x_t = torch.randn((1, N, 3), device=device).repeat(B, 1, 1)
                perm = torch.randperm(x_t.shape[1], device=x_t.device)
                inv_perm = torch.argsort(perm)

                eps1 = model(x_t, t, condition)
                eps2_perm = model(x_t[:, perm, :], t, condition)
                eps2 = eps2_perm[:, inv_perm, :]

                err = (eps1 - eps2).abs().mean()

                print(
                    f"  Iter {i}: T={t.cpu().numpy().flatten()[0]}, err between original and permuted: {err.item():.4e}"
                )
                # if err.item() <1e-5:
                #     print(f"  Permutation invariance test PASSED with error {err.item():.4e}, perm idx sample: {perm.cpu().numpy()}")
    elif args.mode == "train":

        check_tensor( "x0sbn3_train_norm",x0sbn3_train_norm)
        # x0sbn3_train_norm shape= (40, 128, 5) type= <class 'torch.Tensor'> dtype= torch.float32 device= cpu requires_grad= False min= -0.8094146251678467 max= 1.0 nan= False


        x0sbn3_train_norm_rep = duplicate_batch(x0sbn3_train_norm, B)  # [B, N, 3]
        x0sbn3_eval_norm_rep = duplicate_batch(x0sbn3_eval_norm, B)  # [B, N, 3]

        cond_train_norm_rep = (
            duplicate_batch(cond_train_norm, B) if cond_train_norm is not None else None
        )  # [B, cond_dim]
        cond_eval_norm_rep = (
            duplicate_batch(cond_eval_norm, B) if cond_eval_norm is not None else None
        )  # [B, cond_dim]

        print(
            f"shapes after replication: x0sbn3_train_norm_rep {x0sbn3_train_norm_rep.shape}, cond_train_rep {cond_train_norm_rep.shape if cond_train_norm_rep is not None else None}, x0sbn3_eval_norm_rep {x0sbn3_eval_norm_rep.shape}, cond_eval_rep {cond_eval_norm_rep.shape if cond_eval_norm_rep is not None else None}"
        )

        x0sbn3_train_norm_rep = x0sbn3_train_norm_rep.contiguous().pin_memory()
        x0sbn3_eval_norm_rep = x0sbn3_eval_norm_rep.contiguous().pin_memory()
        if cond_train_norm_rep is not None:
            cond_train_norm_rep = cond_train_norm_rep.contiguous().pin_memory()
        if cond_eval_norm_rep is not None:
            cond_eval_norm_rep = cond_eval_norm_rep.contiguous().pin_memory()

        tt = tqdm(
            range(start_step, ddpm_iteration),
            total=ddpm_iteration,
            initial=start_step,
            desc="Training",
            unit="step",
        )
        total_work_step = ddpm_iteration - start_step
        log_train_every = max(1, total_work_step // args.num_train_log)
        save_checkpoint_every = max(1, total_work_step // args.num_checkpoints_save)
        eval_every = max(1, total_work_step // args.num_eval)
        print(
            f"From {start_step} to {ddpm_iteration}: Logging, evaluation, and checkpoint saving frequencies (in steps): {log_train_every}/{eval_every}/{save_checkpoint_every} steps respectively."
        )

        logger = ExperimentLogger(tb_dir, data_dir, vars(args))
        for step in tt:

            loss, loss_dict = train_eval_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                x0sbn3_norm_rep=x0sbn3_train_norm_rep,
                scene_condition_rep=cond_train_norm_rep,
                T=T,
                B=B,
                device=device,
                loss_weights=loss_weights,
                inout_dim=inout_dim,
                is_train=True,
                lambda_cd=args.lambda_cd,
                cd_mode=args.cd_mode,
                lambda_mse=args.lambda_mse,
                prediction_type=args.prediction_type,
            )

            if step % log_train_every == 0:
                # add lr,grad norm, eps,pred/gt mean/norm
                loss_dict.update(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1e5
                        ),
                    }
                )
                logger.log_train(step, loss_dict, log_group=False)

            if step > start_step and  step % save_checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step, checkpoint_path, vars(args)
                )

                model.eval()
                if True:
                    dir_name = f"{samples_dir}/step_{step:06d}"
                    os.makedirs(dir_name, exist_ok=True)
                    progressbar = tqdm(
                        total=2 * 3 * 2,
                        desc=f"Sampling and plotting at step {step}",
                        leave=False,
                    )

                    for set_name, set_cond, gt in zip(
                        ["eval", "train"],
                        [cond_eval_norm, cond_train_norm],
                        [x0sbn3_eval_norm, x0sbn3_train_norm],
                    ):
                        sample_B = min(
                            8, set_cond.shape[0]
                        )  # number of samples to generate for visualization, at most 8 to avoid long sampling time and overcrowded plots
                        assert (
                            set_cond.shape[0] >= sample_B
                        ), f"Need at least {sample_B} samples in the dataset for visualization, but got {set_cond.shape[0]}"
                        expanded_cond = set_cond[:sample_B]
                        shapes = (sample_B, N, inout_dim)

                        random_perm = torch.randperm(sample_B)
                        while torch.equal(random_perm, torch.arange(sample_B)) and sample_B > 1:
                            random_perm = torch.randperm(sample_B)
                        conditions = [
                            ("correct_cond", expanded_cond),
                            ("zero_cond", torch.zeros_like(expanded_cond)),
                            ("shuffled_cond", expanded_cond[random_perm]),
                        ]

                        for seed in [42, 43]:
                            for c_name, c_value in conditions:
                                npz_fname = (
                                    f"{dir_name}/{set_name}_{c_name}_sd{seed}.npz"
                                )
                                with torch.no_grad():
                                    progressbar.set_description(
                                        f"Sampling {set_name} {c_name} sd{seed}"
                                    )
                                    progressbar.update(1)
                                    pred_x = p_sample_loop(
                                        model,
                                        shapes,
                                        scheduler,
                                        num_inference_steps=T_infer,
                                        device=device,
                                        condition=c_value,
                                        seed=seed,  # use step as seed to get different sample each time
                                    )
                                # assert same device
                                # assert pred_x.device == x0sbn3_norm.device, f"Device mismatch: pred_x on {pred_x.device}, x0sbn3_norm on {x0sbn3_norm.device}"
                                # print(f"shapes for stat calculation: pred_x {pred_x.shape}, gt {x0sbn3_norm[: shapes[0]].shape}, condition {c_value.shape}  ")
                                point_stat_output = calculate_pointset_stat(
                                    pred_x.cpu(), gt[: shapes[0]].cpu(), c_name, seed
                                )

                                if (
                                    set_name == "eval"
                                ):  # log for seed, condition type, and eval/train set
                                    logger.log_val(step, point_stat_output, log_group=False)
                                elif set_name == "train":
                                    logger.log_train(step, point_stat_output, log_group=False)
                                else:
                                    raise ValueError(f"Unknown set_name: {set_name}")
                                save_point_sample(
                                    npz_fname,
                                    pred_x.cpu(),
                                    gt=gt[: shapes[0]].cpu(),
                                    condition=c_value.cpu(),
                                    meta={
                                        **point_stat_output,
                                        **{
                                            "seed": seed,
                                            "condition_type": c_name,
                                            "set_name": set_name,
                                        },
                                    },
                                )

                                batch_titles = [
                                    f"cond:{c:.2f}"
                                    for c in c_value[:sample_B]
                                    .view(sample_B, -1)
                                    .mean(dim=1)
                                ]
                                # print(f"batch_titles: {batch_titles}")
                                plot_pc_batch(
                                    pred_x,
                                    gt=gt[: shapes[0]],
                                    title=f"step{step} {set_name} {c_name} sd{seed} N:{N} T:{T} Inf:{T_infer} B:{B}",
                                    fname=f"{temp_dir}/denoised_{set_name}_{c_name}_{seed}_{step:06d}.png",
                                    azm=azm_easing(
                                        step, ddpm_iteration, style="cosine"
                                    ),
                                    progress=step / ddpm_iteration,
                                    elev=azm_easing(
                                        step, ddpm_iteration, style="cosine"
                                    )
                                    / 360
                                    * 90,
                                    batch_titles=batch_titles,
                                )
                    progressbar.close()
                # except Exception as e:
                #     print(f"Error during inference/plotting at step {step}: {e}")
                model.train()

            if step % eval_every == 0:

                _, val_dict = train_eval_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    x0sbn3_norm_rep=x0sbn3_eval_norm_rep,
                    scene_condition_rep=cond_eval_norm_rep,
                    T=T,
                    B=B,
                    device=device,
                    loss_weights=loss_weights,
                    inout_dim=inout_dim,
                    is_train=False,
                    lambda_cd=args.lambda_cd,
                    cd_mode=args.cd_mode,
                    lambda_mse=args.lambda_mse,
                    prediction_type=args.prediction_type,
                )
                logger.log_val(step, val_dict, log_group=False)

            tt.set_description(f"Loss: {loss.item():.1e}")

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            ddpm_iteration,
            checkpoint_path,
            vars(args),
        )
        for set_name in ["eval", "train"]:
            for seed in [42, 43]:
                for c_name in ["correct_cond", "zero_cond", "shuffled_cond"]:
                    os.system(
                        f"ls -v {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png | xargs cat | ffmpeg -y -framerate {fps} -f image2pipe -i - {temp_dir}/../{set_name}_{c_name}_{seed}_{run_id}.gif"
                    )
                    os.system(
                        f"rm {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png"
                    )
        os.system(f"rm -r {temp_dir}")
