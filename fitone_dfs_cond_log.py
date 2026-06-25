import json
import re
import time
import sys,csv
import pickle,random
from datetime import datetime

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
    # sample_batch,
    make_dataset,
    save_point_sample,make_proper_man_dataset
)
def find_nn_cond_exact_chunked(cond_b, cond_all, train_idx_pool=None, train_chunk=32):
    """
    cond_b:          [B, ...] query conditions
    cond_all:        full condition tensor, not necessarily train-only
    train_idx_pool:  global indices allowed for NN retrieval
    returns:         global indices into cond_all / x0_all
    """
    assert cond_b is not None
    assert cond_all is not None

    if train_idx_pool is None:
        train_idx_pool = torch.arange(cond_all.shape[0], dtype=torch.long)
    else:
        train_idx_pool = train_idx_pool.cpu().long()

    q = F.normalize(
        cond_b.reshape(cond_b.shape[0], -1).float().cpu(),
        dim=-1,
        eps=1e-8,
    )

    best_sim = torch.full((q.shape[0],), -float("inf"))
    best_idx = torch.zeros(q.shape[0], dtype=torch.long)

    for a in trange(
        0,
        train_idx_pool.numel(),
        train_chunk,
        desc="Finding NN cond in train pool",
        leave=False,
    ):
        z = min(a + train_chunk, train_idx_pool.numel())
        idx_chunk = train_idx_pool[a:z]

        k = F.normalize(
            cond_all[idx_chunk].reshape(z - a, -1).float().cpu(),
            dim=-1,
            eps=1e-8,
        )

        sim = q @ k.T
        vals, local_idx = sim.max(dim=1)

        mask = vals > best_sim
        best_sim[mask] = vals[mask]
        best_idx[mask] = idx_chunk[local_idx[mask]]

        del k, sim, vals, local_idx

    return best_idx
    
def sample_or_retrieve_in_batches(
    model,
    scheduler,
    gt_all,
    cond_all,
    cond_train_norm,
    x0sbn3_train_norm,
    c_name,
    seed,
    N,
    inout_dim,
    T_infer,
    device,
    batch_size=32,
    shuffle_perm=None,
    train_idx_pool=None,
):
    preds = []
    conds_used = []

    total = gt_all.shape[0]

    for s in trange(0, total, batch_size, leave=False, desc=f"Sampling batch with {c_name}"):
        e = min(s + batch_size, total)
        cond_b = cond_all[s:e] if cond_all is not None else None
        b = e - s
        shapes_b = (b, N, inout_dim)

        if c_name == "none":
            c_value_use = None

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model,
                    shapes_b,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed + s,
                )

        elif c_name == "correct_cond":
            assert cond_b is not None
            c_value_use = cond_b.to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model,
                    shapes_b,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed + s,
                )

        elif c_name == "zero_cond":
            assert cond_b is not None
            c_value_use = torch.zeros_like(cond_b).to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model,
                    shapes_b,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed + s,
                )

        elif c_name == "shuffled_cond":
            assert cond_all is not None
            assert shuffle_perm is not None

            c_value_use = cond_all[shuffle_perm[s:e]].to(device)

            with torch.no_grad():
                pred_b = p_sample_loop(
                    model,
                    shapes_b,
                    scheduler,
                    num_inference_steps=T_infer,
                    device=device,
                    condition=c_value_use,
                    seed=seed + s,
                )

        elif c_name == "nn_retrieval":
            assert cond_b is not None
            assert cond_train_norm is not None
            assert x0sbn3_train_norm is not None

            nn_idx = find_nn_cond_exact_chunked(
                cond_b=cond_b,
                cond_all=cond_train_norm,
                train_idx_pool=train_idx_pool,
                train_chunk=32,
            )

            c_value_use = cond_train_norm[nn_idx].to(device)
            pred_b = x0sbn3_train_norm[nn_idx].to(device)

        else:
            raise ValueError(f"Unknown condition type: {c_name}")

        preds.append(pred_b.detach().cpu())

        # Keep condition only for first 8 samples for plotting/saving.
        # Do not store full WAN conditions for all eval frames.
        if c_value_use is not None and s < 8:
            keep = min(8 - s, b)
            conds_used.append(c_value_use[:keep].detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    cond_used_all = torch.cat(conds_used, dim=0) if len(conds_used) > 0 else None

    return pred_all, cond_used_all

def none_if_all_zero(x):
    if x is None:
        return None
    return None if torch.all(x == 0) else x

def append_per_scene_eval_rows(
    csv_path,
    pred_all,
    gt_all,
    selected_idx,
    all_frame_ids,
    run_id,
    exp_name,
    step,
    set_name,
    condition_type,
    seed,
    args,
):
    """
    Per-scene metrics from already generated pred_all.

    pred_all:      [B, N, D], CPU or GPU
    gt_all:        [B, N, D], CPU or GPU
    selected_idx:  global indices into all_frame_ids, length B

    Group key:
        (data_file, scene_id)

    This intentionally merges left/right side inside one scene.
    """
    selected_idx = selected_idx.cpu().long().tolist()

    groups = {}
    for local_j, global_i in enumerate(selected_idx):
        key = (
            str(all_frame_ids["data_file"][global_i]),
            int(all_frame_ids["scene_id"][global_i]),
        )
        groups.setdefault(key, []).append(local_j)

    for (data_file, scene_id), local_js in groups.items():
        local_t = torch.as_tensor(local_js, dtype=torch.long)

        pred_scene = pred_all[local_t].cpu()
        gt_scene = gt_all[local_t].cpu()

        global_js = [selected_idx[j] for j in local_js]
        sensor_sides = sorted(set(str(all_frame_ids["sensor_side"][i]) for i in global_js))
        frame_indices = [int(all_frame_ids["frame_index"][i]) for i in global_js]

        try:
            stat = calculate_pointset_stat(
                pred_scene,
                gt_scene,
            )
        except Exception as e:
            print(
                f"Per-scene metric error: set={set_name}, cond={condition_type}, "
                f"seed={seed}, data_file={data_file}, scene_id={scene_id}: {e}"
            )
            stat = {
                "cd": float("nan"),
                "fidelity": float("nan"),
                "diversity": float("nan"),
            }

        row = {
            "date_time": datetime.now().isoformat(),
            "run_id": run_id,
            "exp_name": exp_name,
            "step": step,
            "set_name": set_name,
            "data_file": data_file,
            "scene_id": scene_id,
            "sensor_sides": ",".join(sensor_sides),
            "n_frames": len(local_js),
            "frame_index_min": min(frame_indices) if len(frame_indices) > 0 else None,
            "frame_index_max": max(frame_indices) if len(frame_indices) > 0 else None,
            "condition_type": condition_type,
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

        for k, v in stat.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item() if v.numel() == 1 else str(v.detach().cpu().tolist())
            row[k] = v

        append_eval_row(csv_path, row)
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

def _safe(s):
    s = str(s)
    s = s.replace("man-", "")
    s = s.replace("man_", "")
    s = s.replace("heldout_split", "held")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s


def _k(num):
    if num >= 1000 and num % 1000 == 0:
        return f"{num // 1000}k"
    return str(num)


def _safe(s):
    s = str(s)
    s = s.replace("man-", "")
    s = s.replace("man_", "")
    s = s.replace("heldout_split", "held")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s


def _run_cfg_hash(args, n=8):
    """
    Hash only config that defines model/data/training identity.
    Exclude runtime-control fields so checkpoint resume still works.
    """
    ignore = {
        "ddpm_iteration",
        "mode",
        "fps",
        "num_train_log",
        "num_checkpoints_save",
        "num_eval",
        "exp_name",
        "checkpoint_dir",
    }

    d = {
        k: v
        for k, v in vars(args).items()
        if k not in ignore
    }

    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:n]


def _scene_spec_tag(spec, n_keys):
    """
    If explicit scene set is given:
        man-mini:0,1+man-full:10,11 -> mi0.1_fu10.11

    If empty:
        random set -> r60
    """
    scene_set = parse_scene_set_spec(spec)

    if len(scene_set) == 0:
        return f"r{n_keys}"

    return scene_set_tag(scene_set)


def _float_tag(x):
    return f"{x:g}"


def shorten_run_id(out_id,  keep_len=200):

    """
    Keep visible prefix, append hash of the full original string.
    Final length <= max_len.
    """
    if len(out_id) <= keep_len:
        return out_id
    hash_suffix = hashlib.md5(out_id[keep_len:].encode()).hexdigest()[:8]
    # "_h" + 8 chars = 10 chars
    suffix = f"_h{hash_suffix}"

    return out_id[:keep_len] + suffix

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

    shape_spec =""
    if args.shape_name.startswith("man_"):
        shape_spec = f"_{args.data_file}_side{args.sensor_side}"
        split_str = f"_split{args.split_seed}" if args.split_seed != 42 else ""
        
        if args.shape_name in ['man_heldout_split', 'man_proper_split_real']:
            shape_spec += f"{split_str}"
        if args.man_one_distribution:
            shape_spec += "_1dist1eval"
        if args.shape_name == 'man_heldout_split':
            eval_set = parse_scene_set_spec(args.eval_scene_set)
            test_set = parse_scene_set_spec(args.test_scene_set)
            eval_tag = scene_set_tag(eval_set)
            test_tag = scene_set_tag(test_set)
            #add n_eval_scene_keys, n_test_scene_keys
            shape_spec += f"_held_e{args.n_eval_scene_keys}-{eval_tag}_t{args.n_test_scene_keys}-{test_tag}"
        shape_spec += f"_minpts{args.min_frames_per_side}"
        

    cd_spec = f"_cdmd{args.cd_mode}" if args.lambda_cd > 0 else ""


    stridge_str = f"_str{args.wan_frame_stride}_edge{args.wan_edge_policy}" if args.wan_frame_mode != "repeat" else ""
    wan_id = f"_fr{args.wan_frames}_mode{args.wan_frame_mode}{stridge_str}" if args.cond_method == "wan" else ""

    scale_eps2x0_str = "_scaleeps2x0" if args.scale_eps2x0_conversion else ""
        
    out_id =  f"{args.model_name}{model_spec}_it{args.ddpm_iteration}_{args.shape_name}{shape_spec}_train_sc{args.n_scene}_N{args.N}_B{args.B}_T{args.T}-{args.T_infer}_{args.prediction_type}-{args.sampler}{scale_eps2x0_str}_{args.cond_mode}{cond_spec}{wan_id}_weight{dop_rcs_loss_weight}_lmse{args.lambda_mse:1.3f}_lcd{args.lambda_cd:1.3f}{cd_spec}_sd{args.seed}"

    



    return shorten_run_id(out_id, keep_len=200)


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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument(
        "--n_scene",
        type=int,
        default=-1,
        help="Number of FRAMES to use from the dataset (for multi-scene datasets)",
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
        choices=["film", "xattn", "film-xattn","none"],
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
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    # parser.add_argument(
    #     "--val_scene_id",
    #     type=int,
    #     default=-1,
    #     help="Scene ID to use for validation",
    # )
    parser.add_argument(
        "--sensor_side",
        type=str,
        default="left",
        help="Sensor side to use for training: 'both', 'left' or 'right'",
    )
    parser.add_argument(
        "--eval_scene_set",
        type=str,
        default="",
        help="Scene set for eval, e.g. 'man-mini:0,1+man-full:10,11'",
    )
    parser.add_argument(
        "--test_scene_set",
        type=str,
        default="",
        help="Scene set for test, e.g. 'man-mini:2,3+man-full:12,13'",
    )
    parser.add_argument(
        "--min_frames_per_side",
        type=int,
        default=20,
        help="Drop a (data_file, scene_id) if any loaded sensor side has fewer than this many frames.",
    )
    #add to run_id
    parser.add_argument(

        "--n_eval_scene_keys",

        type=int,

        default=60,

        help="Number of random eval (data_file, scene_id) keys if --eval_scene_set is empty.",

    )

    parser.add_argument(

        "--n_test_scene_keys",

        type=int,

        default=60,

        help="Number of random test (data_file, scene_id) keys if --test_scene_set is empty.",

    )
    args = parser.parse_args()


    return args

def auto_fill_scene_sets(
    valid_scene_keys,
    eval_scene_set,
    test_scene_set,
    n_eval_scene_keys,
    n_test_scene_keys,
    seed,
):
    """
    Randomly fill missing eval/test scene sets from valid_scene_keys.

    valid_scene_keys: set of (data_file, scene_id)
    """
    eval_scene_set = set(eval_scene_set)
    test_scene_set = set(test_scene_set)

    overlap = eval_scene_set & test_scene_set
    assert len(overlap) == 0, f"eval/test overlap before auto-fill: {overlap}"

    remaining = sorted(list(valid_scene_keys - eval_scene_set - test_scene_set))

    rng = random.Random(seed)
    rng.shuffle(remaining)

    p = 0

    if len(eval_scene_set) == 0:
        assert len(remaining) >= n_eval_scene_keys, (
            f"Need {n_eval_scene_keys} eval scene keys, only {len(remaining)} available"
        )
        eval_scene_set = set(remaining[p:p + n_eval_scene_keys])
        p += n_eval_scene_keys

    if len(test_scene_set) == 0:
        assert len(remaining) - p >= n_test_scene_keys, (
            f"Need {n_test_scene_keys} test scene keys, only {len(remaining) - p} available"
        )
        test_scene_set = set(remaining[p:p + n_test_scene_keys])
        p += n_test_scene_keys

    overlap = eval_scene_set & test_scene_set
    assert len(overlap) == 0, f"eval/test overlap after auto-fill: {overlap}"

    return eval_scene_set, test_scene_set

def filter_valid_scene_keys(all_frame_ids, min_frames_per_side=20):
    """
    Keep only (data_file, scene_id) where every present sensor_side
    has at least min_frames_per_side frames.

    If left is below threshold, both left/right for that scene are removed.
    """
    counts = {}

    n = len(all_frame_ids["scene_id"])

    for i in range(n):
        key = (
            str(all_frame_ids["data_file"][i]),
            int(all_frame_ids["scene_id"][i]),
        )
        side = str(all_frame_ids["sensor_side"][i])

        if key not in counts:
            counts[key] = {}

        counts[key][side] = counts[key].get(side, 0) + 1

    valid_keys = set()
    bad_keys = {}

    for key, side_counts in counts.items():
        min_count = min(side_counts.values())

        if min_count >= min_frames_per_side:
            valid_keys.add(key)
        else:
            bad_keys[key] = side_counts
    print(f"Valid scene keys (with at least {min_frames_per_side} frames per present sensor side): {valid_keys}"
          f"\nBad scene keys and their sensor side counts: {bad_keys}")
    return valid_keys, bad_keys

def parse_scene_set_spec(spec):
    """
    Parse:
        'man-mini:0,1;man-full:10,11'
    Returns:
        set of (data_file, scene_id)
    """
    if spec is None or spec.strip() == "":
        return set()
    out = set()
    for group in spec.split("+"):
        group = group.strip()
        if not group:
            continue
        assert ":" in group, f"Bad scene set group '{group}', expected data_file:id,id"
        data_file, ids_str = group.split(":", 1)
        data_file = data_file.strip()
        sids =[]
        for sid in ids_str.split(","):
            sid = sid.strip()
            sids.append(int(sid))
            if sid:
                out.add((data_file, int(sid)))
    return out 
import hashlib
def _compress_ids(ids, max_items=6):
    ids = sorted(set(int(x) for x in ids))
    if len(ids) == 0:
        return "none"
    if len(ids) <= max_items:
        return ".".join(str(x) for x in ids)
    h = hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()[:6]
    return f"{ids[0]}..{ids[-1]}n{len(ids)}h{h}"

def scene_set_tag(scene_set):

    """
    scene_set: set of (data_file, scene_id)
    Example:
        {("man-mini",0),("man-mini",1),("man-full",10),("man-full",11)}
        -> 'mi0.1_fu10.11'
    """
    if scene_set is None or len(scene_set) == 0:
        return "none"
    abbr = {
        "man-mini": "mi",
        "man-full": "fu",
    }
    parts = []
    for data_file in ["man-mini", "man-full"]:
        ids = [sid for df, sid in scene_set if df == data_file]
        if len(ids) > 0:
            parts.append(f"{abbr[data_file]}{_compress_ids(ids)}")
    return "_".join(parts) if len(parts) > 0 else "none"
def frame_key(all_frame_ids, i):
    # Split key deliberately excludes sensor_side.
    return (
        str(all_frame_ids["data_file"][i]),
        int(all_frame_ids["scene_id"][i]),
    )

def take_frame_ids(all_frame_ids, idxs):
    keys = ["token", "scene_id", "frame_index", "sensor_side", "data_file"]
    return {
        k: [all_frame_ids[k][int(i)] for i in idxs]
        for k in keys
        if k in all_frame_ids
    }
def gather_man_ds(args, checkpoint_dir):
    if args.cond_method in [ "wan", "scene_id"]:
        cond_method = args.cond_method
        cond_string = f"{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}"
    elif args.cond_method == "none": #get "wan", then set to None
        cond_method = "wan"
        cond_string = f"{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}"

        cond_string = f"pdnorm_only_5_center_1_skip"
    
    x0sbn3_norm_all, cond_all, doppler_all, rcs_all = [], [], [], []
    frame_ids_all = {"train": {'token':[],"scene_id":[],"frame_index":[],"data_file":[],"sensor_side":[]}}
    data_files = ['man-mini',"man-full"] if args.data_file == 'both' else [args.data_file]
    missing_files = {}
    for data_file in data_files:
        print(f"Processing data file: {data_file}")
        sc_ids = list(range(10 if data_file == 'man-mini' else 597)) 
        for sc_id in sc_ids:
            sensor_sides = ["left", "right"] if args.sensor_side == "both" else [args.sensor_side]
            for sensor_side in sensor_sides:
                side_str = "" if sensor_side == "left" else f"_{sensor_side}"
                cache_fname = f"man_{data_file}_{sc_id}{side_str}_{cond_method}_{args.N}_{cond_string}.pkl"
                cache_path = os.path.join(checkpoint_dir, cache_fname)
                if not os.path.exists(cache_path):
                    if data_file not in missing_files:
                        missing_files[data_file] = []
                    print(f"Missing cache file: {cache_path}")
                    missing_files[data_file].append(f"{sc_id}_{sensor_side}")
                # else:
                #     with open(cache_path, "rb") as f:   
                #         (x0sbn3_norm, cond_norm, doppler_norm, rcs_norm),frame_ids= pickle.load(f)
                #     print(f"Sc {sc_id} len frame_ids['train']['token'] {len(frame_ids['train']['token'])} x0sbn3_norm shape {x0sbn3_norm.shape} cond_norm shape {cond_norm.shape if cond_norm is not None else None} doppler_norm shape {doppler_norm.shape if doppler_norm is not None else None} rcs_norm shape {rcs_norm.shape if rcs_norm is not None else None}")

    print(f"Missing files: {missing_files}")
    if sum(len(v) for v in missing_files.values()) > 0:
        print(f"Error: {sum(len(v) for v in missing_files.values())} cache files are missing. Please run the preprocessing script to generate the missing cache files before training.")
        exit(1)
    print(f"No missing file.")
            

    for data_file in tqdm(data_files):
        sc_ids = list(range(10 if data_file == 'man-mini' else 597)) 
        for sc_id in tqdm(sc_ids, desc=f"Loading data for {data_file}", leave=False):
            sensor_sides = ["left", "right"] if args.sensor_side == "both" else [args.sensor_side]
            for sensor_side in sensor_sides:
                side_str = "" if sensor_side == "left" else f"_{sensor_side}"
                cache_fname = f"man_{data_file}_{sc_id}{side_str}_{cond_method}_{args.N}_{cond_string}.pkl"
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
                    frame_ids_all["train"]["sensor_side"].extend([sensor_side] * len(frame_ids["train"]["token"]))
                    frame_ids_all["train"]["data_file"].extend([data_file] * len(frame_ids["train"]["token"]))
                    if  x0sbn3_norm.shape[0] < 36:
                        print(f"{x0sbn3_norm.shape[0]} samples loaded for {sc_id}-{sensor_side}-{data_file}")
    x0sbn3_norm_all = torch.cat(x0sbn3_norm_all, dim=0)
    cond_all = torch.cat(cond_all, dim=0) if cond_all[0] is not None else None
    if args.cond_method == "none":
        cond_all = None
    doppler_all = torch.cat(doppler_all, dim=0) if doppler_all[0] is not None else None
    rcs_all = torch.cat(rcs_all, dim=0) if rcs_all[0] is not None else None

    assert x0sbn3_norm_all.shape[0] == len(frame_ids_all["train"]["token"]) == len(frame_ids_all["train"]["scene_id"]) == len(frame_ids_all["train"]["frame_index"]), f"Mismatch in number of samples and frame IDs: {x0sbn3_norm_all.shape[0]} vs {len(frame_ids_all['train']['token'])}"
    print(f"Loaded {x0sbn3_norm_all.shape} samples from {len(data_files)} data files.")
    return x0sbn3_norm_all, cond_all, doppler_all, rcs_all,frame_ids_all

            

def train_eval_step(
    model,
    optimizer,
    scheduler,
    x0sbn3_norm_all,
    scene_condition_all,
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
    eval_idx_fixed=None,
    idx_pool=None,
    
):
    num_samples = x0sbn3_norm_all.shape[0]

    if idx_pool is None:
        idx_pool = torch.arange(num_samples, device="cpu")
    else:
        idx_pool = idx_pool.cpu()
    if is_train:
        model.train()
        num_pool = idx_pool.numel()
        if num_pool >= B:
            local_idx = torch.randperm(num_pool, device="cpu")[:B]
        else:
            local_idx = torch.randint(0, num_pool, (B,), device="cpu")
        idx = idx_pool[local_idx]
    else:
        model.eval()
        assert eval_idx_fixed is not None, "eval_idx_fixed must be provided for eval"
        idx = eval_idx_fixed.cpu()

    # x0 = x0sbn3_norm_rep[idx][:B].to(device)  # [B, N, 3]
    # cond = scene_condition_rep[idx][:B].to(device)

    x0_cpu = x0sbn3_norm_all[idx]
    x0 = x0_cpu.to(device, non_blocking=True)
    assert not torch.isnan(x0).any() and not torch.isinf(x0).any(), f"NaN or Inf detected in x0 after moving to device: {x0}"

    if scene_condition_all is not None:
        cond_cpu = scene_condition_all[idx]
        cond = cond_cpu.to(device, non_blocking=True)
    else:
        cond = None

    t = torch.randint(0, T, (x0.shape[0],), device=device)
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
        "idx_hash":float( torch.sum(idx.cpu().long() * torch.arange(1, idx.numel() + 1)).item() % 1_000_000),
        "idx_unique": float(torch.unique(idx).numel()),
        "idx_min": float(idx.min().item()),
        "idx_max": float(idx.max().item())
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
        elif cd_mode in ["cd5d", "weighted"]:
            x0_hat_cd = x0_hat
            x0_cd = x0
            if cd_mode == "weighted":
                assert inout_dim == 5, f"weighted_5d expects 5D points, got inout_dim={inout_dim}"
                w = torch.tensor(
                    [
                        loss_weights["position"],
                        loss_weights["position"],
                        loss_weights["position"],
                        loss_weights["doppler"],
                        loss_weights["rcs"],
                    ],
                    device=x0_hat.device,
                    dtype=x0_hat.dtype,
                ).view(1, 1, 5)
                x0_hat_cd = x0_hat * w
                x0_cd = x0 * w
                loss_dict["cd_w_position"] = float(loss_weights["position"])
                loss_dict["cd_w_doppler"] = float(loss_weights["doppler"])
                loss_dict["cd_w_rcs"] = float(loss_weights["rcs"])
            if scale_eps2x0_conversion and prediction_type == "epsilon":
                loss_cd5d_batch = pt3d_chamfer_distance(
                    x0_hat_cd,
                    x0_cd,
                    point_reduction="mean",
                    batch_reduction=None,
                )[0]
                loss_cd5d_batch = loss_cd5d_batch / (1e-8 + conversion_scale.view(-1))
                loss_dict["cd_5d_loss"] = loss_cd5d_batch.mean().item()
                cd_loss = lambda_cd * loss_cd5d_batch.mean()
            else:
                loss_cd5d = pt3d_chamfer_distance(x0_hat_cd, x0_cd)[0]
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
    print(f"checkpoint_path: {checkpoint_path}")
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
    
    print(f"MAN dataset loaded from cache. frame_ids (first 8 items): ")
    for k in all_frame_ids.keys():
        print(f"{k}: {all_frame_ids[k][:8]}")


    if args.shape_name == "man_heldout_split":
        split_seed = args.split_seed 

        assert x0sbn3_norm.shape[0] == len(all_frame_ids["token"])== len(all_frame_ids["scene_id"]) == len(all_frame_ids["frame_index"]) == len(all_frame_ids["data_file"]) == len(all_frame_ids["sensor_side"]), f"Mismatch in number of samples and frame IDs: {x0sbn3_norm.shape[0]} vs {len(all_frame_ids['token'])}"
        
        allids = list(range(x0sbn3_norm.shape[0]))

        valid_scene_keys, bad_scene_keys = filter_valid_scene_keys(
            all_frame_ids,
            min_frames_per_side=args.min_frames_per_side,
        )

        print(
            f"Scene quality filter: kept {len(valid_scene_keys)} scene keys, "
            f"dropped {len(bad_scene_keys)} scene keys with min_frames_per_side={args.min_frames_per_side}"
        )

        if len(bad_scene_keys) > 0:
            print("First bad scene keys:")
            for k, v in list(bad_scene_keys.items())[:20]:
                print(f"  {k}: {v}")

        

        eval_scene_set = parse_scene_set_spec(args.eval_scene_set)
        test_scene_set = parse_scene_set_spec(args.test_scene_set)
        eval_scene_set, test_scene_set = auto_fill_scene_sets(
            valid_scene_keys=valid_scene_keys,
            eval_scene_set=eval_scene_set,
            test_scene_set=test_scene_set,
            n_eval_scene_keys=args.n_eval_scene_keys,
            n_test_scene_keys=args.n_test_scene_keys,
            seed=split_seed,
        )

        overlap = eval_scene_set & test_scene_set
        assert len(overlap) == 0, f"eval_scene_set and test_scene_set overlap: {overlap}"

        train_indices,eval_indices,test_indices = [], [], []

        for i in allids:
            k = frame_key(all_frame_ids, i)

            # Drop all frames from bad scene keys.
            # This removes both sides if either side is below threshold.
            if k not in valid_scene_keys:
                continue
            if k in eval_scene_set:
                eval_indices.append(i)
            elif k in test_scene_set:
                test_indices.append(i)
            else:
                train_indices.append(i)

        dropped_eval = eval_scene_set - valid_scene_keys
        dropped_test = test_scene_set - valid_scene_keys

        assert len(dropped_eval) == 0, (
            f"Some eval_scene_set keys were dropped by min_frames_per_side={args.min_frames_per_side}: "
            f"{sorted(list(dropped_eval))}"
        )

        assert len(dropped_test) == 0, (
            f"Some test_scene_set keys were dropped by min_frames_per_side={args.min_frames_per_side}: "
            f"{sorted(list(dropped_test))}"
        )

        assert len(eval_indices) > 0, f"No eval frames found for eval_scene_set={eval_scene_set}"

        if len(test_scene_set) > 0:

            assert len(test_indices) > 0, f"No test frames found for test_scene_set={test_scene_set}"

        random.seed(split_seed)

        random.shuffle(train_indices)



        if n_scene > 0:
            assert n_scene <= len(train_indices), f"Requested  {n_scene}, change n_scene to {len(train_indices)} or change the eval/test scene sets to free up more training scenes." 
            train_indices = train_indices[:n_scene]



        train_idx_pool = torch.as_tensor(train_indices, dtype=torch.long)

        eval_idx_pool = torch.as_tensor(eval_indices, dtype=torch.long)

        test_idx_pool = torch.as_tensor(test_indices, dtype=torch.long) if len(test_indices) > 0 else None


        # Keep references to full base tensors. No train/eval/test tensor copies.

        x0sbn3_train_norm = x0sbn3_norm

        cond_train_norm = cond_norm

        doppler_train_norm = doppler_norm

        rcs_train_norm = rcs_norm

        x0sbn3_eval_norm = x0sbn3_norm

        cond_eval_norm = cond_norm

        doppler_eval_norm = doppler_norm

        rcs_eval_norm = rcs_norm




        # frame_ids = {"train": {"token":[all_frame_ids["token"][ int(i)] for i in train_idx_pool],"scene_id":[all_frame_ids["scene_id"][ int(i)] for i in train_idx_pool],"frame_index":[all_frame_ids["frame_index"][ int(i)] for i in train_idx_pool], "sensor_side":[all_frame_ids["sensor_side"][ int(i)] for i in train_idx_pool], "data_file":[all_frame_ids["data_file"][ int(i)] for i in train_idx_pool]}, "eval": {"token":[all_frame_ids["token"][ int(i)] for i in eval_idx_pool],"scene_id":[all_frame_ids["scene_id"][ int(i)] for i in eval_idx_pool],"frame_index":[all_frame_ids["frame_index"][ int(i)] for i in eval_idx_pool], "sensor_side":[all_frame_ids["sensor_side"][ int(i)] for i in eval_idx_pool], "data_file":[all_frame_ids["data_file"][ int(i)] for i in eval_idx_pool]} , "test": {"token":[all_frame_ids["token"][ int(i)] for i in test_idx_pool],"scene_id":[all_frame_ids["scene_id"][ int(i)] for i in test_idx_pool],"frame_index":[all_frame_ids["frame_index"][ int(i)] for i in test_idx_pool], "sensor_side":[all_frame_ids["sensor_side"][ int(i)] for i in test_idx_pool], "data_file":[all_frame_ids["data_file"][ int(i)] for i in test_idx_pool]} if test_idx_pool is not None else None}

        frame_ids = {
            "split_method": "man_heldout_split",
            "split_key": "data_file,scene_id",
            "split_seed": split_seed,
            "eval_scene_set": sorted(list(eval_scene_set)),
            "test_scene_set": sorted(list(test_scene_set)),
            "train": take_frame_ids(all_frame_ids, train_idx_pool),
            "eval": take_frame_ids(all_frame_ids, eval_idx_pool),
            "test": take_frame_ids(all_frame_ids, test_idx_pool) if test_idx_pool is not None else None,
        }
        print("-#-----")
        print(
            f"After split: full={x0sbn3_norm.shape[0]}, "
            f"train_frames={train_idx_pool.numel()}, "
            f"eval_frames={eval_idx_pool.numel()}, "
            f"test_frames={0 if test_idx_pool is None else test_idx_pool.numel()}"
        )
        print(f"eval_scene_set={sorted(list(eval_scene_set))}")
        print(f"test_scene_set={sorted(list(test_scene_set))}")

    elif args.shape_name == "man_proper_split_real" and args.man_one_distribution:
        split_seed = args.split_seed
        data_file = 'both'
        # cache_fname = f"man_one_dist_frame_ids_man-mini_288_{cond_method}_{cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}_seed{42}.pkl"
        # cache_path = os.path.join(checkpoint_dir, cache_fname)
        if True or os.path.exists(cache_path):
            
            allids = list(range(x0sbn3_norm.shape[0]))
            random.seed(split_seed)
            random.shuffle(allids)

            print(f"shape of x0sbn3_norm: {x0sbn3_norm.shape}, shape of cond_norm: {cond_norm.shape if cond_norm is not None else None}, shape of doppler_norm: {doppler_norm.shape if doppler_norm is not None else None}, shape of rcs_norm: {rcs_norm.shape if rcs_norm is not None else None}")
            
            real_train_size = n_scene
            assert real_train_size <= x0sbn3_norm.shape[0], f"Requested number of training scenes {n_scene} exceeds total available scenes {x0sbn3_norm.shape[0]} in the dataset."
            if False:
                real_eval_size = max(2, int(0.1 * n_scene / 0.8))  # keep the same eval ratio as in the original split
            else:
                real_eval_size = 36
            assert real_eval_size + real_train_size <= x0sbn3_norm.shape[0], f"Requested number of training scenes {n_scene} and evaluation scenes {real_eval_size} exceeds total available scenes {x0sbn3_norm.shape[0]} in the dataset."

            # Define the slice index ranges from your shuffled allids list
            train_indices = allids[:real_train_size]
            eval_indices = allids[-real_eval_size:]

            train_idx_pool = torch.as_tensor(train_indices, dtype=torch.long)
            eval_idx_pool = torch.as_tensor(eval_indices, dtype=torch.long)

            # Keep references to the full base tensors. No large copy.
            x0sbn3_train_norm = x0sbn3_norm
            cond_train_norm = cond_norm
            doppler_train_norm = doppler_norm
            rcs_train_norm = rcs_norm

            x0sbn3_eval_norm = x0sbn3_norm
            cond_eval_norm = cond_norm
            doppler_eval_norm = doppler_norm
            rcs_eval_norm = rcs_norm

            frame_ids = {"train": {"token":[all_frame_ids["token"][ i] for i in train_indices],"scene_id":[all_frame_ids["scene_id"][ i] for i in train_indices],"frame_index":[all_frame_ids["frame_index"][ i] for i in train_indices], "sensor_side":[all_frame_ids["sensor_side"][ i] for i in train_indices], "data_file":[all_frame_ids["data_file"][ i] for i in train_indices]},
                        "eval": {"token":[all_frame_ids["token"][ i] for i in eval_indices],"scene_id":[all_frame_ids["scene_id"][ i] for i in eval_indices],"frame_index":[all_frame_ids["frame_index"][ i] for i in eval_indices], "sensor_side":[all_frame_ids["sensor_side"][ i] for i in eval_indices], "data_file":[all_frame_ids["data_file"][ i] for i in eval_indices]} }
            print(f"After splitting cached dataset: Training scenes: {x0sbn3_train_norm.shape[0]}, Evaluation scenes: {x0sbn3_eval_norm.shape[0]}, frame_ids (first 8): ")
            print(f"train: token:{frame_ids['train']['token'][:8]}, scene_id: {frame_ids['train']['scene_id'][:8]}, frame_index: {frame_ids['train']['frame_index'][:8]};")
            print(f"eval: token:{frame_ids['eval']['token'][:8]}, scene_id: {frame_ids['eval']['scene_id'][:8]}, frame_index: {frame_ids['eval']['frame_index'][:8]}")

    print(
        f"Dataset created. Full samples: {x0sbn3_train_norm.shape[0]}, "
        f"Train indexed frames: {train_idx_pool.numel()}, "
        f"Eval indexed frames: {eval_idx_pool.numel()}"
    )


    # assert x0sbn3_train_norm.shape[0] >= n_scene, f"Not enough training scenes in the dataset. Requested: {n_scene}, available: {x0sbn3_train_norm.shape[0]}"
    n_scene = train_idx_pool.numel()
    print(
        f"shapes after dataset creation: x0sbn3_norm {x0sbn3_train_norm.shape}, cond {cond_train_norm.shape if cond_train_norm is not None else None}, doppler_norm {doppler_train_norm.shape if doppler_train_norm is not None else None}, rcs_norm {rcs_train_norm.shape if rcs_train_norm is not None else None}"
    )
    


    if args.train_rcs_doppler:
        x0sbn3_all_5d = torch.cat(
            [x0sbn3_train_norm, doppler_train_norm, rcs_train_norm],
            dim=-1,
        )
        x0sbn3_train_norm = x0sbn3_all_5d
        x0sbn3_eval_norm = x0sbn3_all_5d
        inout_dim = 5


        del doppler_train_norm, doppler_eval_norm, rcs_train_norm, rcs_eval_norm

        del doppler_norm, rcs_norm
    else:
        inout_dim = 3

    model = make_model(device, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

    if args.mode == "eval":
        raise NotImplementedError("Evaluation mode is not implemented yet. Please use 'train' or 'interpolate' modes.")
    elif args.mode == "train":
        seed = args.seed

        check_tensor( "x0sbn3_train_norm",x0sbn3_train_norm)
        # x0sbn3_train_norm shape= (40, 128, 5) type= <class 'torch.Tensor'> dtype= torch.float32 device= cpu requires_grad= False min= -0.8094146251678467 max= 1.0 nan= False

        print(f"Whole data shape x0sbn3_train_norm: {x0sbn3_train_norm.shape}, x0sbn3_eval_norm: {x0sbn3_eval_norm.shape}, cond_train_norm: {cond_train_norm.shape if cond_train_norm is not None else None}, cond_eval_norm: {cond_eval_norm.shape if cond_eval_norm is not None else None}")

        print(f"indices length: train_idx_pool {train_idx_pool.numel()}, eval_idx_pool {eval_idx_pool.numel()} test_idx_pool {0 if test_idx_pool is None else test_idx_pool.numel()}")

        x0sbn3_train_norm = x0sbn3_train_norm.contiguous()
        x0sbn3_eval_norm = x0sbn3_eval_norm.contiguous()
        print(f"made x0sbn3_train_norm and x0sbn3_eval_norm contiguous, make date sits in one piece in memory for faster random access during training and evaluation.")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
        per_scene_csv = os.path.join(data_dir + "/..", "condition_eval_per_scene.csv")
        logger.log_text("total_parameters", f"{sum(p.numel() for p in model.parameters())}",0)
        logger.log_text("checkpoint_path", checkpoint_path, 0)
        
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
            print(f"Saving frame_ids to logger config: (first 8 frames)")
            print(f"train: token:{frame_ids['train']['token'][:8]}, scene_id: {frame_ids['train']['scene_id'][:8]}, frame_index: {frame_ids['train']['frame_index'][:8]};")
            print(f"eval:  token:{frame_ids['eval']['token'][:8]}, scene_id: {frame_ids['eval']['scene_id'][:8]}, frame_index: {frame_ids['eval']['frame_index'][:8]}")
            # logger.log_text("frame_ids",json.dumps(frame_ids, indent=4),0)
            print("Saving frame_ids to JSON file, not TensorBoard text.")

            frame_ids_path = os.path.join(data_dir, "frame_ids.json")
            with open(frame_ids_path, "w") as f:
                json.dump(frame_ids, f, indent=2, default=str)
            frame_ids_hash = hashlib.md5(
                json.dumps(frame_ids, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            frame_ids_summary = {
                "frame_ids_path": frame_ids_path,
                "frame_ids_hash": frame_ids_hash,
                "split_method": frame_ids.get("split_method"),
                "split_key": frame_ids.get("split_key"),
                "split_seed": frame_ids.get("split_seed"),
                "n_train": len(frame_ids["train"]["token"]) if frame_ids.get("train") else 0,
                "n_eval": len(frame_ids["eval"]["token"]) if frame_ids.get("eval") else 0,
                "n_test": len(frame_ids["test"]["token"]) if frame_ids.get("test") else 0,
                "eval_scene_set": frame_ids.get("eval_scene_set"),
                "test_scene_set": frame_ids.get("test_scene_set"),
                "train_first8": {
                    "scene_id": frame_ids["train"]["scene_id"][:8],
                    "frame_index": frame_ids["train"]["frame_index"][:8],
                    "sensor_side": frame_ids["train"]["sensor_side"][:8],
                    "data_file": frame_ids["train"]["data_file"][:8],
                },
                "eval_first8": {
                    "scene_id": frame_ids["eval"]["scene_id"][:8],
                    "frame_index": frame_ids["eval"]["frame_index"][:8],
                    "sensor_side": frame_ids["eval"]["sensor_side"][:8],
                    "data_file": frame_ids["eval"]["data_file"][:8],
                },
            }
            logger.log_text(
                "frame_ids_summary",
                json.dumps(frame_ids_summary, indent=2, default=str),
                0,
            )
            
            
        # print(f"Starting training loop from step {start_step} to {ddpm_iteration} skape x0sbn3_train_norm_rep: {x0sbn3_train_norm_rep.shape}, cond_train_norm_rep: {cond_train_norm_rep.shape if cond_train_norm_rep is not None else None}")
        g_eval = torch.Generator().manual_seed(seed + 999)

        eval_local = torch.randperm(eval_idx_pool.numel(), generator=g_eval)[:min(B, eval_idx_pool.numel())]
        eval_idx_fixed = eval_idx_pool[eval_local]

        for step in tt:
            is_final_sample_eval = step + 1 >= ddpm_iteration
            
            loss, loss_dict = train_eval_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                x0sbn3_norm_all=x0sbn3_train_norm,
                scene_condition_all=cond_train_norm,
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
                eval_idx_fixed=None,
                idx_pool=train_idx_pool
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
                        set_idx_pool = eval_idx_pool if set_name == "eval" else train_idx_pool
                        if set_cond is not None and gt is not None:
                            assert gt.shape[0] == set_cond.shape[0], f"Number of samples in gt {gt.shape[0]} and condition {set_cond.shape[0]} must match for set {set_name}"
                        

                        for sample_seed in tqdm([42, 43] if is_final_sample_eval else [42], desc=f"Seeds for {set_name}", leave=False):
                            for c_name in tqdm(["correct_cond", "zero_cond", "shuffled_cond", "nn_retrieval"] if set_cond is not None else ["none"], desc=f"Conditions for {set_name}", leave=False):
                                # print(f"-=-===--==--- {set_name}/{sample_seed}/{c_name}/{step}...")
                                if set_name == "train" and c_name == "nn_retrieval":
                                    # print(f"skip")
                                    continue
                                if c_name == "nn_retrieval" and sample_seed != 42:
                                    # print(f"skip")
                                    continue
                                # full_B = gt.shape[0] if is_final_sample_eval else min(8, gt.shape[0])
                                if is_final_sample_eval and set_name == "eval":
                                    selected_idx = set_idx_pool
                                else:
                                    selected_idx = set_idx_pool[:min(8, set_idx_pool.numel())]

                                full_B = selected_idx.numel()
                                full_gt = gt[selected_idx]
                                full_cond = set_cond[selected_idx] if set_cond is not None else None
                                # print(f"{set_name}/shape of cond: {set_cond.shape if set_cond is not None else None}, shape of gt: {gt.shape}, full_B: {full_B}")


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
                                    seed=sample_seed,
                                    N=N,
                                    inout_dim=inout_dim,
                                    T_infer=T_infer,
                                    device=device,
                                    batch_size=32,   # tune this
                                    shuffle_perm=shuffle_perm,
                                    train_idx_pool=train_idx_pool,   # REQUIRED
                                )
                                try:
                                    point_stat_output = calculate_pointset_stat(
                                        pred_all.cpu(), full_gt.cpu(), prefix=f"{c_name}_{sample_seed}_"
                                    )
                                except Exception as e:
                                    print(f"Error calculating point set statistics at step {step} for {set_name} with condition {c_name} and seed {sample_seed}: {e}")
                                    point_stat_output = {"cd": float("nan"), "fidelity": float("nan"), "diversity": float("nan")}
                                
                                npz_fname = (
                                    f"{dir_name}/{set_name}_{c_name}_sd{sample_seed}.npz"
                                )


                                summary_row = {
                                    "date_time": datetime.now().isoformat(),
                                    "run_id": run_id,
                                    "exp_name": args.exp_name,
                                    "step": step,
                                    "n_scene": args.n_scene,
                                    "set_name": set_name,              # train/eval
                                    "condition_type": c_name,          # correct/zero/shuffled/nn
                                    "seed": sample_seed,
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
                                if is_final_sample_eval:
                                    print(f"final step {step}/{set_name}/{c_name}/seed{sample_seed}: {summary_row}")
                                    append_eval_row(summary_csv, summary_row)
                                    if set_name in ["eval", "test"]:
                                        append_per_scene_eval_rows(
                                            csv_path=per_scene_csv,
                                            pred_all=pred_all,
                                            gt_all=full_gt,
                                            selected_idx=selected_idx,
                                            all_frame_ids=all_frame_ids,
                                            run_id=run_id,
                                            exp_name=args.exp_name,
                                            step=step,
                                            set_name=set_name,
                                            condition_type=c_name,
                                            seed=sample_seed,
                                            args=args,
                                        )

                                if (
                                    set_name == "eval"
                                ):  # log for seed, condition type, and eval/train set
                                    logger.log_val(step, point_stat_output, log_group=False)
                                elif set_name == "train":
                                    logger.log_train(step, point_stat_output, log_group=False)
                                else:
                                    raise ValueError(f"Unknown set_name: {set_name}")
                                try:
                                    if is_final_sample_eval:
                                        plot_B =  pred_all.shape[0]
                                    else:
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
                                                "seed": sample_seed,
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
                                    batch_id_titles = [
                                        f"sc:{all_frame_ids['scene_id'][int(i)]}, frame:{all_frame_ids['frame_index'][int(i)]}"
                                        for i in selected_idx[:plot_B]
                                    ]
                                    batch_titles =   [f"{t}- {bt}" for t, bt in zip(batch_titles, batch_id_titles)]
                                    plot_pc_batch(
                                        pred_x,
                                        gt=extended_gt,
                                        title=f"step{step} {set_name} {c_name} sd{sample_seed} N:{N} T:{T} Inf:{T_infer} B:{B}",
                                        fname=f"{temp_dir}/denoised_{set_name}_{c_name}_{sample_seed}_{step:06d}.png",
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
                                    print(f"Error saving point sample at step {step} for {set_name} with condition {c_name} and seed {sample_seed}: {e}")
                                    continue
                                
                                # kill full_B, pred_all, cond_used_all, full_gt, full_cond, shuffle_perm
                                
                # except Exception as e:
                #     print(f"Error during inference/plotting at step {step}: {e}")
                model.train()

            if step % eval_every == 0:

                _, val_dict = train_eval_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    x0sbn3_norm_all=x0sbn3_eval_norm,
                    scene_condition_all=cond_eval_norm,
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
                    eval_idx_fixed=eval_idx_fixed
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
        logger.close()
        for set_name in ["eval", "train"]:
            for seed in [42, 43]:# only seed 42 get animation!
                for c_name in ["correct_cond", "zero_cond", "shuffled_cond", "nn_retrieval"]:

                    if set_name == "train" and c_name == "nn_retrieval":
                        continue
                    if c_name == "nn_retrieval" and seed != 42:
                        continue
                    if seed ==42:
                        os.system(
                            f"ls -v {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png | xargs cat | ffmpeg -y -framerate {fps} -f image2pipe -i - {temp_dir}/../{set_name}_{c_name}_{seed}.gif"
                        )
                        #scale=640:-1
                        os.system(
                            f"ls -v {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png | xargs cat | ffmpeg -y -f image2pipe -vcodec png -i - -vf \"fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3\" {temp_dir}/../{set_name}_{c_name}_{seed}_scaled.gif"
                        )
                    else:#just copy over
                        os.system(
                            f"cp {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png {temp_dir}/../"
                        )
                    os.system(
                        f"rm {temp_dir}/denoised_{set_name}_{c_name}_{seed}_*.png"
                    )
        os.system(f"rm -r {temp_dir}")
