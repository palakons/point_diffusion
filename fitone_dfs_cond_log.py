import json
import re
import time
import sys,csv
import pickle,random

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
    save_point_sample,make_proper_man_dataset
)
def sample_or_retrieve_in_batches(
    model, scheduler, gt_all, cond_all, cond_train_norm, x0sbn3_train_norm,
    c_name, seed, N, inout_dim, T_infer, device, batch_size=32,shuffle_perm=None ,
):
    preds = []
    conds_used = []

    total = gt_all.shape[0]

    for s in trange(0, total, batch_size,leave=False, desc=f"Sampling batch with {c_name}"):
        e = min(s + batch_size, total)
        gt_b = gt_all[s:e]
        cond_b = cond_all[s:e]
        b = e - s
        shapes_b = (b, N, inout_dim)

        if c_name == "correct_cond":
            c_value_use = cond_b.to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model, shapes_b, scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed,
                )

        elif c_name == "zero_cond":
            c_value_use = torch.zeros_like(cond_b).to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model, shapes_b, scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed,
                )

        elif c_name == "shuffled_cond":

            assert shuffle_perm is not None

            c_value_use = cond_all[shuffle_perm[s:e]].to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model, shapes_b, scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed,
                )

        elif c_name == "nn_retrieval":
            q = F.normalize(cond_b.reshape(b, -1).float().cpu(), dim=-1, eps=1e-8)
            k = F.normalize(
                cond_train_norm.reshape(cond_train_norm.shape[0], -1).float().cpu(),
                dim=-1,
                eps=1e-8,
            )
            nn_idx = (q @ k.T).argmax(dim=1)

            c_value_use = cond_train_norm[nn_idx].to(device)
            pred_b = x0sbn3_train_norm[nn_idx].to(device)

        else:
            raise ValueError(f"Unknown condition type: {c_name}")

        preds.append(pred_b.detach().cpu())
        conds_used.append(c_value_use.detach().cpu() if c_value_use is not None else None)

    pred_all = torch.cat(preds, dim=0)
    cond_used_all = torch.cat(conds_used, dim=0) if conds_used[0] is not None else None

    return pred_all, cond_used_all
def none_if_all_zero(x):
    if x is None:
        return None
    return None if torch.all(x == 0) else x
def append_eval_row(csv_path, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
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

    print("-=--------------=-\n")

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
    # return x0 and scale factor for epsilon to x0 conversion if applicable
    if prediction_type == "sample":
        return pred, None
    alpha_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1)
    return (x_t - torch.sqrt(1 - alpha_bar) * pred) / torch.sqrt(alpha_bar) , torch.sqrt((1 - alpha_bar)/alpha_bar)


def make_run_id(args):
    cond_spec = f"_{args.cond_method}" if args.cond_method != "none" else ""

    dop_rcs_loss_weight = (
        f"_{args.loss_weight_position:1.3f}-{args.loss_weight_doppler:1.3f}-{args.loss_weight_rcs:1.3f}"
        if args.train_rcs_doppler
        else ""
    )

    model_spec = f"_dim{args.set_tx_dim}" if args.model_name == "SetTxDnsr" else ""
    if args.model_name == "SetTxDnsr":
        if args.set_cond_type != "film":
            model_spec += f"_{args.set_cond_type}"
        if args.use_condition_pooling:
            model_spec += f"_pool_k{args.condition_pool_kernel}"
        if args.use_wan_pos_emb:
            model_spec += "_wanpe"
        

    shape_spec = f"_{args.data_file}" if args.shape_name in ['realman', 'man_proper_split', 'realman_dense'] else ""
    if args.man_one_distribution:
        shape_spec += "_1dist1eval"

    cd_spec = f"_cdmd{args.cd_mode}" if args.lambda_cd > 0 else ""


    stridge_str = f"_str{args.wan_frame_stride}_edge{args.wan_edge_policy}" if args.wan_frame_mode != "repeat" else ""
    wan_id = f"_fr{args.wan_frames}_mode{args.wan_frame_mode}{stridge_str}" if args.cond_method == "wan" else ""

    scale_eps2x0_str = "_scaleeps2x0" if args.scale_eps2x0_conversion else ""
        
    return f"{args.model_name}{model_spec}_{args.shape_name}{shape_spec}_train_sc{args.n_scene}_N{args.N}_B{args.B}_T{args.T}-{args.T_infer}_{args.prediction_type}-{args.sampler}{scale_eps2x0_str}_it{args.ddpm_iteration}_{args.cond_mode}{cond_spec}{wan_id}_weight{dop_rcs_loss_weight}_lmse{args.lambda_mse:1.3f}_lcd{args.lambda_cd:1.3f}{cd_spec}_sd{args.seed}"


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
    except Exception as e:
        print(f"Exception in p_sample_loop: {e}")
        print(f"shaps at exception: x {x.shape}, t_tensor {t_tensor.shape}, condition {condition.shape if condition is not None else None}")
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
        help="Shape to train on: 'realman' or 'various', 'realman_dense', 'man_proper_split'"
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
    parser.add_argument(
        "--wan_frames", 
        type=int,
        default=5,
        help="Number of frames to use for Wan's VAE latent conditioning"
    )   
    parser.add_argument(
        "--wan_frame_mode",
        type=str,
        default="repeat",
        help="Mode for handling Wan's VAE latent frames: 'repeat/center/past/future'"
    )
    parser.add_argument(
        "--wan_frame_stride",
        type=int,
        default=1,
        help="Stride for selecting frames for Wan's VAE latent conditioning"
    )
    parser.add_argument(
        "--wan_edge_policy",
        type=str,
        default="skip",
        help="Policy for handling edge frames in Wan's VAE latent conditioning: 'skip/pad'"
    )
    parser.add_argument(
        "--set_cond_type",
        type=str,
        default="film",
        choices=["film", "xattn", "film-xattn"],
        help="Conditioning type for SetTxDnsr: 'film', 'xattn', or 'film-xattn'",
    )
    parser.add_argument(
        "--scale_eps2x0_conversion",
        action="store_true",
        help="During epsilon to x0 conversion, scale the predicted epsilon by the standard deviation of the noise added at each timestep, as suggested in some implementations to improve stability. This is only applied when prediction_type is 'epsilon' and the CD loss is used."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default_exp_name",
        help="Optional experiment name to use in logging. If not provided, a name will be generated based on the other arguments.",
    )
    parser.add_argument(
        "--set_tx_dim",
        type=int,
        default=64,
        help="Dimension of the Set Transformer features in the SetTxDnsr model",
    )
    parser.add_argument(
        "--use_condition_pooling",
        action="store_true",
        help="Pool WAN condition spatially before FiLM/XAttn conditioning.",
    )

    parser.add_argument(
        "--condition_pool_kernel",
        type=int,
        default=4,
        help="Spatial pooling kernel/stride for WAN condition.",
    )
    parser.add_argument(
        "--use_wan_pos_emb",
        action="store_true",
        help="Add learned 2D positional embeddings to WAN tokens for cross-attention.",
    )
    parser.add_argument(
        "--man_one_distribution",
        action="store_true",
        help="Use a single distribution for all scenes in the MAN dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val_scene_id",
        type=int,
        default=-1,
        help="Scene ID to use for validation",
    )

    args = parser.parse_args()


    return args

def gather_man_ds(args, checkpoint_dir):
    x0sbn3_norm_all, cond_all, doppler_all, rcs_all = [], [], [], []
    frame_ids_all = {"train": {'token':[],"scene_id":[],"frame_index":[],"data_file":[]}}
    data_files = ['man-mini',"man-full"] if args.data_file == 'both' else [args.data_file]
    missing_files = {}
    for data_file in data_files:
        print(f"Processing data file: {data_file}")
        sc_ids = list(range(10 if data_file == 'man-mini' else 597)) 
        for sc_id in sc_ids:
            cache_fname = f"man_{data_file}_{sc_id}_{args.cond_method}_{args.N}_{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}.pkl"
            cache_path = os.path.join(checkpoint_dir, cache_fname)
            if not os.path.exists(cache_path):
                if data_file not in missing_files:
                    missing_files[data_file] = []
                missing_files[data_file].append(sc_id)
            # else:
            #     with open(cache_path, "rb") as f:   
            #         (x0sbn3_norm, cond_norm, doppler_norm, rcs_norm),frame_ids= pickle.load(f)
            #     print(f"Sc {sc_id} len frame_ids['train']['token'] {len(frame_ids['train']['token'])} x0sbn3_norm shape {x0sbn3_norm.shape} cond_norm shape {cond_norm.shape if cond_norm is not None else None} doppler_norm shape {doppler_norm.shape if doppler_norm is not None else None} rcs_norm shape {rcs_norm.shape if rcs_norm is not None else None}")
    print(f"Missing files: {missing_files}")
    if sum(len(v) for v in missing_files.values()) > 0:
        print(f"Error: {sum(len(v) for v in missing_files.values())} cache files are missing. Please run the preprocessing script to generate the missing cache files before training.")
        exit(1)
            


    for data_file in data_files:
        sc_ids = list(range(10 if data_file == 'man-mini' else 597)) 
        for sc_id in sc_ids:
            cache_fname = f"man_{data_file}_{sc_id}_{args.cond_method}_{args.N}_{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}.pkl"
            cache_path = os.path.join(checkpoint_dir, cache_fname)
            # assert  os.path.exists(cache_path), f"Cache file {cache_fname} not found, need to run python /palakons/point_diffusion/preprocess_man.py --cond_method wan --wan_frames 5 --wan_frame_mode center --wan_frame_stride 1 --wan_edge_policy skip --N 128  --data_file man-mini --num_scenes 100 --from_scene_id 0"
            with open(cache_path, "rb") as f:   
                (x0sbn3_norm, cond_norm, doppler_norm, rcs_norm),frame_ids= pickle.load(f)
                x0sbn3_norm_all.append(x0sbn3_norm)
                cond_all.append(cond_norm)
                doppler_all.append(doppler_norm)
                rcs_all.append(rcs_norm)

                frame_ids_all["train"]["token"].extend(frame_ids["train"]["token"])
                frame_ids_all["train"]["scene_id"].extend(frame_ids["train"]["scene_id"])
                frame_ids_all["train"]["frame_index"].extend(frame_ids["train"]["frame_index"])
                frame_ids_all["train"]["data_file"].extend([data_file] * len(frame_ids["train"]["token"]))
    x0sbn3_norm_all = torch.cat(x0sbn3_norm_all, dim=0)
    cond_all = torch.cat(cond_all, dim=0) if cond_all[0] is not None else None
    doppler_all = torch.cat(doppler_all, dim=0) if doppler_all[0] is not None else None
    rcs_all = torch.cat(rcs_all, dim=0) if rcs_all[0] is not None else None

    assert x0sbn3_norm_all.shape[0] == len(frame_ids_all["train"]["token"]) == len(frame_ids_all["train"]["scene_id"]) == len(frame_ids_all["train"]["frame_index"]), f"Mismatch in number of samples and frame IDs: {x0sbn3_norm_all.shape[0]} vs {len(frame_ids_all['train']['token'])}"
    print(f"Loaded {x0sbn3_norm_all.shape} samples from {len(data_files)} data files.")
    return x0sbn3_norm_all, cond_all, doppler_all, rcs_all,frame_ids_all

            

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
    scale_eps2x0_conversion=False,
):
    num_samples = x0sbn3_norm_rep.shape[0]

    if is_train:
        model.train()
        # idx = torch.randperm(x0sbn3_norm_rep.shape[0], device="cpu")
        idx = torch.randint(0, num_samples, (B,), device="cpu")
    else:
        model.eval()
        # idx = torch.arange(
        #     x0sbn3_norm_rep.shape[0], device="cpu"
        # )  # use the same order for evaluation for consistency
        idx = torch.arange(min(B, num_samples), device="cpu")

    # x0 = x0sbn3_norm_rep[idx][:B].to(device)  # [B, N, 3]
    # cond = scene_condition_rep[idx][:B].to(device)

    x0_cpu = x0sbn3_norm_rep[idx]
    x0 = x0_cpu.to(device, non_blocking=True)
    assert not torch.isnan(x0).any() and not torch.isinf(x0).any(), f"NaN or Inf detected in x0 after moving to device: {x0}"

    if scene_condition_rep is not None:
        cond_cpu = scene_condition_rep[idx]
        cond = cond_cpu.to(device, non_blocking=True)
    else:
        cond = None

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
            x0_hat,conversion_scale = reconstruct_x0(pred, x_t, t, scheduler, prediction_type)
            if False:
                from matplotlib import pyplot as plt
                print(f"t: {t}, conversion_scale {conversion_scale.shape}: {conversion_scale.view(-1) if conversion_scale is not None else None}") #conversion_scale torch.Size([1024, 1, 1])
                #plot scatter scale vs t to /home/palakons/point_diffusion/output/sample
                fig = plt.figure()
                plt.scatter(t.cpu(), conversion_scale.view(-1).cpu())
                plt.xlabel("t")
                plt.ylabel("conversion_scale")
                #log y
                plt.yscale("log")
                plt.title("Scatter plot of conversion scale vs t")
                plt.savefig("/home/palakons/point_diffusion/output/sample/conversion_scale_vs_t.png")
                plt.close()
                exit()
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
            cd_loss = ( loss_weights["position"]  * cd_loss_dict["cd_xyz"]
                + loss_weights["doppler"]  * cd_loss_dict["doppler_attr_loss"]
                + loss_weights["rcs"] * cd_loss_dict["rcs_attr_loss"]
            )* lambda_cd 
            loss += cd_loss 
        elif cd_mode == "cd5d":
            if scale_eps2x0_conversion and prediction_type == "epsilon":

                loss_cd5d_batch = pt3d_chamfer_distance(x0_hat, x0, point_reduction="mean", batch_reduction=None)[0]
                
                # print(f"shape of loss_cd5d_batch before scaling {loss_cd5d_batch.shape}, loss_cd5d_batch value {loss_cd5d_batch}")
                # print(f"shape of conversion_scale {conversion_scale.shape}, conversion_scale value {conversion_scale.view(-1)}")
                loss_cd5d_batch /= 1e-8 + conversion_scale.view(-1)  # scale the CD loss by the conversion scale to account for the magnitude difference between epsilon and x0, as suggested in some implementations for stability
                # print(f"shape of loss_cd5d_batch after scaling {loss_cd5d_batch.shape}, loss_cd5d_batch value {loss_cd5d_batch}")

                loss_dict["cd_5d_loss"] = loss_cd5d_batch.mean().item()
                cd_loss = lambda_cd * loss_cd5d_batch.mean()

                # exit()
            else:

                loss_cd5d = pt3d_chamfer_distance(x0_hat, x0)[0]  
                loss_dict["cd_5d_loss"] = loss_cd5d.item()
                cd_loss = lambda_cd * loss_cd5d

            loss += cd_loss
    if not torch.isfinite(loss).all():
        print(f"Non-finite loss detected! loss: {loss}, loss_dict: {loss_dict}")

    loss_dict["total_loss"] = loss.item()
    if is_train:
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        loss_dict["grad_norm"] = grad_norm.item()
        


        optimizer.step()
    return loss, loss_dict


if __name__ == "__main__":
    args = parse_args()
    # Example setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = make_run_id(args)
    system_key = "ddpm_cond_5"
    data_dir = f"/data/palakons/{system_key}/{args.exp_name}"
    tb_dir = f"/home/palakons/logs/tb_log/{system_key}/{args.exp_name}"
    temp_dir = f"{data_dir}/temp"
    samples_dir = f"{data_dir}/samples"
    checkpoint_dir = f"/data/palakons/{system_key}/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_{run_id}.pt")
    exists = {'tb_dir': os.path.exists(tb_dir), 'data_dir': os.path.exists(data_dir),"checkpoint_file": os.path.exists(checkpoint_path)}
    print(f"Directories and checkpoint existence: {exists}")
    assert not exists['tb_dir'] , f"TensorBoard log directory {tb_dir} already exists. Please choose a different experiment name or remove the existing logs to avoid overwriting."
    # creat dir, nested if not exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)


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
    frame_ids = None

    x0sbn3_norm, cond_norm, doppler_norm, rcs_norm,frame_ids_all = gather_man_ds(args, checkpoint_dir)
    all_frame_ids=frame_ids_all["train"] 
    print(f"Gathered MAN dataset: {x0sbn3_norm.shape} samples, cond shape: {cond_norm.shape if cond_norm is not None else None}, doppler shape: {doppler_norm.shape if doppler_norm is not None else None}, rcs shape: {rcs_norm.shape if rcs_norm is not None else None}")
    print(f"all_frame_ids: {all_frame_ids}")


    if args.shape_name == "man_heldout_split":
        data_file = 'both'
        # assert args.val_scene_id >= 0, f"Need to specify a validation scene ID with --val_scene_id when using the 'man_heldout_split' shape_name. Please provide a non-negative integer for --val_scene_id."
        seed = args.seed

        # cache_fname = f"man_one_dist_frame_ids_man-mini_288_{cond_method}_{cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}_seed{42}.pkl"
        # cache_path = os.path.join(checkpoint_dir, cache_fname)
        # assert  os.path.exists(cache_path), f"Cache file {cache_fname} not found"

        # with open(cache_path, "rb") as f:   
        #     (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (
        #         x0sbn3_eval_norm,
        #         cond_eval_norm,
        #         doppler_eval_norm,
        #         rcs_eval_norm,
        #     ),(x0sbn3_test_norm, cond_test_norm, doppler_test_norm, rcs_test_norm),frame_ids = pickle.load(f)
            
        # x0sbn3_norm = torch.cat([x0sbn3_train_norm, x0sbn3_eval_norm, x0sbn3_test_norm  ], dim=0)
        # cond_norm = torch.cat([cond_train_norm, cond_eval_norm, cond_test_norm], dim=0) if cond_train_norm is not None else None
        # doppler_norm = torch.cat([doppler_train_norm, doppler_eval_norm, doppler_test_norm], dim=0) if doppler_train_norm is not None else None
        # rcs_norm = torch.cat([rcs_train_norm, rcs_eval_norm, rcs_test_norm], dim=0) if rcs_train_norm is not None else None
        # all_frame_ids = {"token": frame_ids["train"]["token"] + frame_ids["eval"]["token"] + frame_ids["test"]["token"], "scene_id": frame_ids["train"]["scene_id"] + frame_ids["eval"]["scene_id"] + frame_ids["test"]["scene_id"], "frame_index": frame_ids["train"]["frame_index"] + frame_ids["eval"]["frame_index"] + frame_ids["test"]["frame_index"] }
        assert x0sbn3_norm.shape[0] == len(all_frame_ids["token"]) == len(all_frame_ids["scene_id"]) == len(all_frame_ids["frame_index"]), f"Number of samples in x0sbn3_norm {x0sbn3_norm.shape[0]} does not match number of frame ids {len(all_frame_ids['token'])}"

        allids = list(range(x0sbn3_norm.shape[0]))
        val_scene_id = args.val_scene_id
        val_frame_idx = [i for i, sid in enumerate(all_frame_ids["scene_id"]) if sid == val_scene_id]
        train_allids = [i for i in allids if i not in val_frame_idx]
        
        random.seed(seed)
        random.shuffle(train_allids)

        real_train_idx = train_allids[:n_scene]
        assert len(real_train_idx) == n_scene, f"Number of training scenes selected {len(real_train_idx)} does not match requested {n_scene}"

        print(f"ALL. frame_ids: {all_frame_ids}, shape of x0sbn3_norm: {x0sbn3_norm.shape}, shape of cond_norm: {cond_norm.shape if cond_norm is not None else None}, shape of doppler_norm: {doppler_norm.shape if doppler_norm is not None else None}, shape of rcs_norm: {rcs_norm.shape if rcs_norm is not None else None}")

        (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm) = x0sbn3_norm[ real_train_idx], cond_norm[real_train_idx] if cond_norm is not None else None, doppler_norm[real_train_idx] if doppler_norm is not None else None, rcs_norm[real_train_idx] if rcs_norm is not None else None

        (x0sbn3_eval_norm, cond_eval_norm, doppler_eval_norm, rcs_eval_norm) = x0sbn3_norm[ val_frame_idx], cond_norm[val_frame_idx] if cond_norm is not None else None, doppler_norm[val_frame_idx] if doppler_norm is not None else None, rcs_norm[val_frame_idx] if rcs_norm is not None else None

        frame_ids = {"train": {"token":[all_frame_ids["token"][ i] for i in real_train_idx],"scene_id":[all_frame_ids["scene_id"][ i] for i in real_train_idx],"frame_index":[all_frame_ids["frame_index"][ i] for i in real_train_idx]},
                    "eval": {"token":[all_frame_ids["token"][ i] for i in val_frame_idx],"scene_id":[all_frame_ids["scene_id"][ i] for i in val_frame_idx],"frame_index":[all_frame_ids["frame_index"][ i] for i in val_frame_idx]} }
        print("------")
        print(f"After splitting cached dataset: Training scenes: {x0sbn3_train_norm.shape[0]}, Evaluation scenes: {x0sbn3_eval_norm.shape[0]}, frame_ids: {frame_ids}")

    elif args.shape_name == "man_proper_split_real" and args.man_one_distribution:
        seed = args.seed
        data_file = 'both'
        # cache_fname = f"man_one_dist_frame_ids_man-mini_288_{cond_method}_{cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}_seed{42}.pkl"
        # cache_path = os.path.join(checkpoint_dir, cache_fname)
        if True or os.path.exists(cache_path):
            # print(f"Loading MAN dataset from cache: {cache_path}")
            # with open(cache_path, "rb") as f:   
            #     (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (
            #         x0sbn3_eval_norm,
            #         cond_eval_norm,
            #         doppler_eval_norm,
            #         rcs_eval_norm,
            #     ),(x0sbn3_test_norm, cond_test_norm, doppler_test_norm, rcs_test_norm),frame_ids = pickle.load(f)
            # #combine all to one big chunk, alsi frame ids
            # # frame_ids = {"train": {'token':[train_ds[2][i]['frame_token'][:5] for i in range(train_ds[0].shape[0])],"scene_id":[train_ds[2][i]['scene_id'] for i in range(train_ds[0].shape[0])],"frame_index":[train_ds[2][i]['frame_index'] for i in range(train_ds[0].shape[0])]},"eval": {'token':[eval_ds[2][i]['frame_token'][:5] for i in range(eval_ds[0].shape[0])],"scene_id":[eval_ds[2][i]['scene_id'] for i in range(eval_ds[0].shape[0])],"frame_index":[eval_ds[2][i]['frame_index'] for i in range(eval_ds[0].shape[0])]},"test": {'token':[test_ds[2][i]['frame_token'][:5] for i in range(test_ds[0].shape[0])],"scene_id":[test_ds[2][i]['scene_id'] for i in range(test_ds[0].shape[0])],"frame_index":[test_ds[2][i]['frame_index'] for i in range(test_ds[0].shape[0])]} }
    
            # x0sbn3_norm = torch.cat([x0sbn3_train_norm, x0sbn3_eval_norm, x0sbn3_test_norm  ], dim=0)
            # cond_norm = torch.cat([cond_train_norm, cond_eval_norm, cond_test_norm], dim=0) if cond_train_norm is not None else None
            # doppler_norm = torch.cat([doppler_train_norm, doppler_eval_norm, doppler_test_norm], dim=0) if doppler_train_norm is not None else None
            # rcs_norm = torch.cat([rcs_train_norm, rcs_eval_norm, rcs_test_norm], dim=0) if rcs_train_norm is not None else None
            # all_frame_ids = {"token": frame_ids["train"]["token"] + frame_ids["eval"]["token"] + frame_ids["test"]["token"], "scene_id": frame_ids["train"]["scene_id"] + frame_ids["eval"]["scene_id"] + frame_ids["test"]["scene_id"], "frame_index": frame_ids["train"]["frame_index"] + frame_ids["eval"]["frame_index"] + frame_ids["test"]["frame_index"] }
            # assert x0sbn3_norm.shape[0] == len(all_frame_ids["token"]) == len(all_frame_ids["scene_id"]) == len(all_frame_ids["frame_index"]), f"Number of samples in x0sbn3_norm {x0sbn3_norm.shape[0]} does not match number of frame ids {len(all_frame_ids['token'])}"

            allids = list(range(x0sbn3_norm.shape[0]))
            random.seed(seed)
            random.shuffle(allids)


            print(f"MAN dataset loaded from cache. frame_ids: {frame_ids}, shape of x0sbn3_norm: {x0sbn3_norm.shape}, shape of cond_norm: {cond_norm.shape if cond_norm is not None else None}, shape of doppler_norm: {doppler_norm.shape if doppler_norm is not None else None}, shape of rcs_norm: {rcs_norm.shape if rcs_norm is not None else None}")
            
            real_train_size = n_scene
            assert real_train_size <= x0sbn3_norm.shape[0], f"Requested number of training scenes {n_scene} exceeds total available scenes {x0sbn3_norm.shape[0]} in the dataset."
            if False:
                real_eval_size = max(2, int(0.1 * n_scene / 0.8))  # keep the same eval ratio as in the original split
            else:
                real_eval_size = 36
            assert real_eval_size + real_train_size <= x0sbn3_norm.shape[0], f"Requested number of training scenes {n_scene} and evaluation scenes {real_eval_size} exceeds total available scenes {x0sbn3_norm.shape[0]} in the dataset."

            (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm) = x0sbn3_norm[ allids[:real_train_size]], cond_norm[allids[:real_train_size]] if cond_norm is not None else None, doppler_norm[allids[:real_train_size]] if doppler_norm is not None else None, rcs_norm[allids[:real_train_size]] if rcs_norm is not None else None

            (x0sbn3_eval_norm, cond_eval_norm, doppler_eval_norm, rcs_eval_norm) = x0sbn3_norm[ allids[-real_eval_size:]], cond_norm[allids[-real_eval_size:]] if cond_norm is not None else None, doppler_norm[allids[-real_eval_size:]] if doppler_norm is not None else None, rcs_norm[allids[-real_eval_size:]] if rcs_norm is not None else None

            frame_ids = {"train": {"token":[all_frame_ids["token"][ i] for i in allids[:real_train_size]],"scene_id":[all_frame_ids["scene_id"][ i] for i in allids[:real_train_size]],"frame_index":[all_frame_ids["frame_index"][ i] for i in allids[:real_train_size]]},
                        "eval": {"token":[all_frame_ids["token"][ i] for i in allids[-real_eval_size:]],"scene_id":[all_frame_ids["scene_id"][ i] for i in allids[-real_eval_size:]],"frame_index":[all_frame_ids["frame_index"][ i] for i in allids[-real_eval_size:]]} }
            print(f"After splitting cached dataset: Training scenes: {x0sbn3_train_norm.shape[0]}, Evaluation scenes: {x0sbn3_eval_norm.shape[0]}, frame_ids: {frame_ids}")
             
        # elif args.shape_name == "man_proper_split":
        #     raise NotImplementedError(f"Lets only use the cached dataset for the specific case of one distribution, 288 scenes, and the man-mini data file for now to avoid accidentally overwriting existing cache. Requested n_scene: {n_scene}, data_file: {data_file}, cond_method: {cond_method}, cond_mode: {cond_mode}, wan_frames: {args.wan_frames}, wan_frame_mode: {args.wan_frame_mode}, wan_frame_stride: {args.wan_frame_stride}, wan_edge_policy: {args.wan_edge_policy}")

        #     (x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (
        #         x0sbn3_eval_norm,
        #         cond_eval_norm,
        #         doppler_eval_norm,
        #         rcs_eval_norm,
        #     ),test_ds,frame_ids = make_proper_man_dataset( N, cond_mode, cond_method, n_train_frames=n_scene, device=device, data_file=data_file,wan_spec={"wan_frames": args.wan_frames, "wan_frame_mode": args.wan_frame_mode, "wan_frame_stride": args.wan_frame_stride, "wan_edge_policy": args.wan_edge_policy}, split_ratio={"train":0.8, "eval":0.1, "test":0.1},split_seed=seed, scene_split_method="first", n_eval_frames=max(2, int(n_scene/8)), n_test_frames=max(2, int(n_scene/8)),one_distribution=args.man_one_distribution
        #     )        
        #     if args.man_one_distribution and n_scene == 288 and x0sbn3_train_norm.shape[0] == 288 and data_file == "man-mini":
        #         if not os.path.exists(cache_path):
        #             with open(cache_path, "wb") as f:
        #                 pickle.dump([(x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (
        #                     x0sbn3_eval_norm,
        #                     cond_eval_norm,
        #                     doppler_eval_norm,
        #                     rcs_eval_norm,
        #                 ),test_ds,frame_ids], f)
        #             print(f"MAN dataset cached at {cache_path} for future runs")
        #         else:
        #             print(f"MAN dataset already cached at {cache_path}")
        #             raise RuntimeError("This should not happen due to the earlier existence check, but just in case to avoid overwriting existing cache.")
    else:

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
            cond_mode=cond_mode,
            wan_spec={"wan_frames": args.wan_frames, "wan_frame_mode": args.wan_frame_mode, "wan_frame_stride": args.wan_frame_stride, "wan_edge_policy": args.wan_edge_policy}
        )
    print(f"Requested {n_scene} training scenes, got {x0sbn3_train_norm.shape[0]} training scenes after dataset creation")
    print(f"Dataset created. Training scenes: {x0sbn3_train_norm.shape[0]}, Evaluation scenes: {x0sbn3_eval_norm.shape[0]}")
    # assert x0sbn3_train_norm.shape[0] >= n_scene, f"Not enough training scenes in the dataset. Requested: {n_scene}, available: {x0sbn3_train_norm.shape[0]}"
    n_scene = x0sbn3_train_norm.shape[0]
    print(
        f"shapes after dataset creation: x0sbn3_norm {x0sbn3_train_norm.shape}, cond {cond_train_norm.shape if cond_train_norm is not None else None}, doppler_norm {doppler_train_norm.shape if doppler_train_norm is not None else None}, rcs_norm {rcs_train_norm.shape if rcs_train_norm is not None else None}"
    )
    print(f"shaep of x0sbn3_train_norm, cond_train_norm, doppler_train_norm, rcs_train_norm), (x0sbn3_eval_norm,            cond_eval_norm,            doppler_eval_norm,            rcs_eval_norm        ): {x0sbn3_train_norm.shape}, {cond_train_norm.shape if cond_train_norm is not None else None}, {doppler_train_norm.shape if doppler_train_norm is not None else None}, {rcs_train_norm.shape if rcs_train_norm is not None else None}), ({x0sbn3_eval_norm.shape}, {cond_eval_norm.shape if cond_eval_norm is not None else None}, {doppler_eval_norm.shape if doppler_eval_norm is not None else None}, {rcs_eval_norm.shape if rcs_eval_norm is not None else None})")



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
        # print(f"shape type dtype dev of cond_train_norm before replication: {cond_train_norm.shape if cond_train_norm is not None else None} {type(cond_train_norm) if cond_train_norm is not None else None} {cond_train_norm.dtype if cond_train_norm is not None else None} {cond_train_norm.device if cond_train_norm is not None else None}") #cuda
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

        summary_csv = os.path.join(data_dir+"/..", "condition_eval_summary.csv")
        logger.log_text("total_parameters", f"{sum(p.numel() for p in model.parameters())}",0)
        
        param_groups = {}
        for name, param in model.named_parameters():
            group_name = name.split(".")[0]
            if group_name not in param_groups:
                param_groups[group_name] = 0
            param_groups[group_name] += param.numel()
        for group_name, num_params in param_groups.items():
            logger.log_text(f"num_parameters_{group_name}", f"{num_params}", 0)
        logger.log_text("run_id", run_id, 0)
        logger.log_text("system_key", system_key, 0)
        logger.log_text("node_name", os.uname().nodename, 0)
        if frame_ids is not None:
            print(f"Saving frame_ids to logger config: {frame_ids}")
            logger.log_text("frame_ids",json.dumps(frame_ids, indent=4),0)
            
        print(f"Starting training loop from step {start_step} to {ddpm_iteration} skape x0sbn3_train_norm_rep: {x0sbn3_train_norm_rep.shape}, cond_train_norm_rep: {cond_train_norm_rep.shape if cond_train_norm_rep is not None else None}")
        exit()
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
                scale_eps2x0_conversion=args.scale_eps2x0_conversion,
            )

            if step % log_train_every == 0:
                # add lr,grad norm, eps,pred/gt mean/norm
                loss_dict.update(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        # "grad_norm": torch.nn.utils.clip_grad_norm_(
                        #     model.parameters(), max_norm=1e5
                        # ),
                    }
                )
                logger.log_train(step, loss_dict, log_group=False)
            is_final_sample_eval = (step + 1 >= ddpm_iteration)
            if step > start_step and  (step % save_checkpoint_every == 0 or is_final_sample_eval) :
                save_checkpoint(
                    model, optimizer, scheduler, step, checkpoint_path, vars(args)
                )

                model.eval()
                if True:
                    dir_name = f"{samples_dir}/step_{step:06d}"
                    os.makedirs(dir_name, exist_ok=True)

                    for set_name, set_cond, gt in tqdm(zip(
                        ["eval", "train"],
                        [cond_eval_norm, cond_train_norm],
                        [x0sbn3_eval_norm, x0sbn3_train_norm],
                    ), desc=f"Sampling and plotting at step {step}",leave=False):
                        assert gt.shape[0] == set_cond.shape[0], f"Number of samples in gt {gt.shape[0]} and condition {set_cond.shape[0]} must match for set {set_name}"
                        # sample_B = min(
                        #     8, set_cond.shape[0], gt.shape[0]
                        # ) 
                        # # sample_B = gt.shape[0]
                        # assert (
                        #     set_cond.shape[0] >= sample_B
                        # ), f"Need at least {sample_B} samples in the dataset for visualization, but got {set_cond.shape[0]}"
                        # expanded_cond = set_cond[:sample_B]
                        # extended_gt = gt[:sample_B]                        
                        # shapes = (sample_B, N, inout_dim)

                        # random_perm = torch.randperm(sample_B)
                        # while torch.equal(random_perm, torch.arange(sample_B)) and sample_B > 1:
                        #     random_perm = torch.randperm(sample_B)
                        # conditions = [
                        #     ("correct_cond", expanded_cond),
                        #     # ("zero_cond", None),
                        #     ("zero_cond", torch.zeros_like(expanded_cond)),
                        #     ("shuffled_cond", expanded_cond[random_perm]),
                        #     ("nn_retrieval", None),
                        # ]

                        for seed in tqdm([42, 43] if is_final_sample_eval else [42], desc=f"Seeds for {set_name}", leave=False):
                            for c_name in tqdm(["correct_cond", "zero_cond", "shuffled_cond", "nn_retrieval"], desc=f"Conditions for {set_name}", leave=False):
                            # for c_name, c_value in tqdm(conditions, desc=f"Conditions for {set_name}", leave=False):
                                if set_name == "train" and c_name == "nn_retrieval":
                                    continue
                                if c_name == "nn_retrieval" and seed != 42:
                                    continue
                                full_B = gt.shape[0] if is_final_sample_eval else min(8, gt.shape[0])
                                full_cond = set_cond[:full_B]
                                full_gt = gt[:full_B]

                                shuffle_perm = None
                                if c_name == "shuffled_cond":
                                    shuffle_perm = torch.randperm(full_B)
                                    while torch.equal(shuffle_perm, torch.arange(full_B)) and full_B > 1:
                                        shuffle_perm = torch.randperm(full_B)

                                pred_all, cond_used_all = sample_or_retrieve_in_batches(

                                    model=model,
                                    scheduler=scheduler,
                                    gt_all=full_gt,
                                    cond_all=full_cond,
                                    cond_train_norm=cond_train_norm,
                                    x0sbn3_train_norm=x0sbn3_train_norm,
                                    c_name=c_name,
                                    seed=seed,
                                    N=N,
                                    inout_dim=inout_dim,
                                    T_infer=T_infer,
                                    device=device,
                                    batch_size=32,   # tune this
                                    shuffle_perm=shuffle_perm,
                                )
                                try:
                                    point_stat_output = calculate_pointset_stat(
                                        pred_all.cpu(), full_gt.cpu(), c_name, seed
                                    )
                                except Exception as e:
                                    print(f"Error calculating point set statistics at step {step} for {set_name} with condition {c_name} and seed {seed}: {e}")
                                    point_stat_output = {"cd": float("nan"), "fidelity": float("nan"), "diversity": float("nan")}
                                
                                npz_fname = (
                                    f"{dir_name}/{set_name}_{c_name}_sd{seed}.npz"
                                )

                                # pred_x = None
                                # if c_name == "nn_retrieval":
                                #     # nn_idx = torch.cdist(expanded_cond.view(sample_B, -1).float(), cond_train_norm.view(cond_train_norm.shape[0], -1).float()).argmin(dim=1)

                                #     q = F.normalize(expanded_cond.reshape(sample_B, -1).float(), dim=-1)
                                #     k = F.normalize(cond_train_norm.reshape(cond_train_norm.shape[0], -1).float(), dim=-1)
                                #     nn_idx = (q @ k.T).argmax(dim=1)

                                #     c_value_use = cond_train_norm[nn_idx].to(device)
                                #     pred_x = x0sbn3_train_norm[nn_idx].to(device)
                                # else:
                                #     c_value_use = c_value.to(device) if c_value is not None else None
                                #     with torch.no_grad():
                                        
                                #         pred_x = p_sample_loop(
                                #             model,
                                #             shapes,
                                #             scheduler,
                                #             num_inference_steps=T_infer,
                                #             device=device,
                                #             condition=c_value_use,
                                #             seed=seed,  # use step as seed to get different sample each time
                                #         )
                                # # assert same device
                                # # assert pred_x.device == x0sbn3_norm.device, f"Device mismatch: pred_x on {pred_x.device}, x0sbn3_norm on {x0sbn3_norm.device}"
                                # # print(f"shapes for stat calculation: pred_x {pred_x.shape}, gt {x0sbn3_norm[: shapes[0]].shape}, condition {c_value.shape}  ")
                                # try:
                                #     point_stat_output = calculate_pointset_stat(
                                #         pred_x.cpu(), extended_gt.cpu(), c_name, seed
                                #     )
                                # except Exception as e:  
                                #     print(f"Error calculating point set statistics at step {step} for {set_name} with condition {c_name} and seed {seed}: {e}")
                                #     point_stat_output = {"cd": float("nan"), "fidelity": float("nan"), "diversity": float("nan")}

                                summary_row = {
                                    "run_id": run_id,
                                    "step": step,
                                    "n_scene": args.n_scene,
                                    "set_name": set_name,              # train/eval
                                    "condition_type": c_name,          # correct/zero/shuffled/nn
                                    "seed": seed,
                                    "model_name": args.model_name,
                                    "cond_type": getattr(args, "set_cond_type", None),
                                    "prediction_type": args.prediction_type,
                                    "cd_mode": args.cd_mode,
                                    "lambda_cd": args.lambda_cd,
                                    "lambda_mse": args.lambda_mse,
                                    "use_condition_pooling": getattr(args, "use_condition_pooling", False),
                                    "condition_pool_kernel": getattr(args, "condition_pool_kernel", None),
                                    "set_tx_dim": getattr(args, "set_tx_dim", None),
                                }
                                for k, v in point_stat_output.items():
                                    if isinstance(v, torch.Tensor):
                                        v = v.detach().cpu().item() if v.numel() == 1 else str(v.detach().cpu().tolist())
                                    summary_row[k] = v
                                append_eval_row(summary_csv, summary_row)

                                if (
                                    set_name == "eval"
                                ):  # log for seed, condition type, and eval/train set
                                    logger.log_val(step, point_stat_output, log_group=False)
                                elif set_name == "train":
                                    logger.log_train(step, point_stat_output, log_group=False)
                                else:
                                    raise ValueError(f"Unknown set_name: {set_name}")
                                try:
                                    plot_B = min(8, pred_all.shape[0]) 
                                    pred_x = pred_all[:plot_B]
                                    extended_gt = full_gt[:plot_B]
                                    c_value_use = cond_used_all[:plot_B] if cond_used_all is not None else None

                                    save_point_sample(
                                        npz_fname,
                                        pred_x.cpu(),
                                        gt=extended_gt.cpu(),
                                        condition=c_value_use.cpu() if c_value_use is not None else None,
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
                                        for c in c_value_use[:plot_B]
                                        .view(plot_B, -1)
                                        .mean(dim=1)
                                    ] if c_value_use is not None else [f"N/A" for _ in range(plot_B)]
                                    # print(f"batch_titles: {batch_titles}")
                                    plot_pc_batch(
                                        pred_x,
                                        gt=extended_gt,
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
                                except Exception as e:
                                    print(f"Error saving point sample at step {step} for {set_name} with condition {c_name} and seed {seed}: {e}")
                                    continue
                                
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
                    scale_eps2x0_conversion=args.scale_eps2x0_conversion,
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
                for c_name in ["correct_cond", "zero_cond", "shuffled_cond", "nn_retrieval"]:

                    if set_name == "train" and c_name == "nn_retrieval":
                        continue
                    if c_name == "nn_retrieval" and seed != 42:
                        continue
                    os.system(
                        f"ls -v {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png | xargs cat | ffmpeg -y -framerate {fps} -f image2pipe -i - {temp_dir}/../{set_name}_{c_name}_{seed}.gif"
                    )
                    #scale=640:-1
                    os.system(
                        f"ls -v {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png | xargs cat | ffmpeg -y -f image2pipe -vcodec png -i - -vf \"fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3\" {temp_dir}/../{set_name}_{c_name}_{seed}_scaled.gif"
                    )
                    os.system(
                        f"rm {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png"
                    )
        os.system(f"rm -r {temp_dir}")
