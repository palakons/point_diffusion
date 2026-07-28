from glob import glob
import json
import re
import time
import sys,csv
import pickle,random
from datetime import datetime
import hashlib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os
from tqdm import tqdm, trange
import numpy as np
from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance
from pytorch3d.ops import knn_points
import matplotlib.pyplot as plt
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
from fitone_dfs_cond_log import find_nn_cond_exact_chunked,sample_or_retrieve_in_batches,none_if_all_zero,append_per_scene_eval_rows,append_eval_row,debug_batch,check_model,check_tensor,chamfer_xyz_with_matched_attrs,p_sample_loop,shorten_run_id,make_run_id,auto_fill_scene_sets,filter_valid_scene_keys,parse_scene_set_spec,_compress_ids,scene_set_tag,reconstruct_x0,frame_key,take_frame_ids,gather_man_ds,train_eval_step,eval_multi_batch,TimeRecorder,make_frame_meta_np,append_per_frame_eval_rows,LazyNpyArray,compute_norm_stats_from_train,NormalizedX0Array,NormalizedCondArray
from truckscenes import TruckScenes

from types import ModuleType

# Mock open3d to bypass the headless GLIBC_2.38 driver clash
class MockOpen3D(ModuleType):
    def __getattr__(self, name):
        return MockOpen3D(name)

sys.modules['open3d'] = MockOpen3D('open3d')








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
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="Learning rate schedule.",

    )
    parser.add_argument(
        "--lr_eta_min_ratio",
        type=float,
        default=0.1,
        help="Factor multiplying LR to get min LR for cosine schedule. Only used if --lr_schedule is 'cosine'."
    )
    parser.add_argument(
        "--cond_ram_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for RAM storage of WAN condition. Only used if --cond_method is 'wan'.",
    )
    parser.add_argument(
        "--fresh_run"
,        action="store_true",
        help="If set, do not resume from existing checkpoints, start fresh.",
    )
    parser.add_argument(
        "--clip_until_step",
        type=int,
        default=0,
        help="If > 0, clip gradients to this value for the first N steps. to clip all gradients, set to 0. If < 0, do not clip gradients.",
    )
    parser.add_argument(
        "--lazy_npy",
        action="store_true",
        help="If set, do not precompute WAN condition for all frames, compute on-the-fly during training. This saves RAM but may slow down training.",
    )
    parser.add_argument(
        "--norm_per_scene",
        action="store_true",
        help="If set, normalize WAN condition per scene instead of globally.",
    )


    args = parser.parse_args()


    return args

def unnormalize_data(x, norm_stats):


    x = x.clone()
    '''
        x: [n,3 or 5]
        norm stat: {
                        "x0sbn3": {
                            "mean": [
                            29.298988342285156,
                            4.784998893737793,
                            0.002921066712588072
                            ],
                            "max_half_range": 45.21278762817383
                        },
                        "doppler": {
                            "mean": [
                            15.066837310791016
                            ],
                            "max_half_range": 100.08779907226562
                        },
                        "rcs": {
                            "mean": [
                            -10.122147560119629
                            ],
                            "max_half_range": 56.12214660644531
                        }
                    }
    '''
    # print(f"shape of x: {x.shape} minmax, {x.min()}, {x.max()}")

    # shape of x: torch.Size([128, 5]) minmax, -0.5435303449630737, 0.8864037394523621
    x[:,:3] = x[:,:3] * norm_stats["x0sbn3"]["max_half_range"] + np.array(norm_stats["x0sbn3"]["mean"])[None,:]

    if x.shape[-1] == 5:
        x[:,3:4] = x[:,3:4] * norm_stats["doppler"]["max_half_range"] + np.array(norm_stats["doppler"]["mean"])[None,:]
        x[:,4:5] = x[:,4:5] * norm_stats["rcs"]["max_half_range"] + np.array(norm_stats["rcs"]["mean"])[None,:]
    # print(f"shape of x: {x.shape} minmax, {x.min()}, {x.max()}")shape of x: torch.Size([128, 5]) minmax, -15.458029747009277, 47.90575408935547
    return x
    

def plot_combo(image_rgb_path,pred,gt,save_path,title):

    #plot img, ont he left, gt/pred top view ont he right, with gt in blue, pred in red
    fig,axs =  plt.subplots(2,1,figsize=(6,6))
    #plot image
    img = plt.imread(image_rgb_path)
    axs[0].imshow(img)
    axs[0].set_title(title)
    axs[0].axis('off')
    #plot gt/pred top view  , y is horizontal, x is vertical
    axs[1].scatter(gt[:,1], gt[:,0], c='blue', s=1, label='GT')
    # x_range=[0, 50],

    # y_range=[-50, 50],
    axs[1].scatter(pred[:,1], pred[:,0], c='red', s=1, label='Pred')
    axs[1].set_title(f"Top View: GT (blue) vs Pred (red)")
    axs[1].set_xlabel("Y (m)")
    axs[1].set_ylabel("X (m)")
    #set equal aspect ratio
    # axs[1].set_aspect('equal', adjustable='box')

    #reset range [-1,1] to *_lim_meters
    # axs[1].set_xlim(-50, 50)
    # axs[1].set_ylim(00, 50)

    #make sure aspect is equal
    axs[1].set_aspect('equal')
    


    axs[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved inference figure to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    # Example setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id,full_run_id = make_run_id(args)
    system_key = "ddpm_cond_slow"
    data_dir = f"/data/palakons/{system_key}/{args.exp_name}"
    tb_dir = f"/home/palakons/logs/tb_log/{system_key}/{args.exp_name}"
    temp_dir = f"{data_dir}/temp"
    samples_dir = f"{data_dir}/samples"
    inference_dir = f"{data_dir}/inference"
    checkpoint_dir = f"/data/palakons/{system_key}/checkpoints/"
    cache_dir = f"/data/palakons/{system_key}/cache_unnorm/"
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_{run_id}.pt")
    exists = {'tb_dir': os.path.exists(tb_dir), 'data_dir': os.path.exists(data_dir),"checkpoint_file": os.path.exists(checkpoint_path)}
    print(f"Directories and checkpoint existence: {exists}")
    print(f"checkpoint_path: {checkpoint_path}")
    #            re.sub(r"it\d+", "it*", checkpoint_path)
    ceckpoint_path_pattern = re.sub(r"it\d+", "it*", checkpoint_path)
    print(f"checkpoint_path pattern: {ceckpoint_path_pattern}")
    print(f"exists wild card checkpoint: {glob(ceckpoint_path_pattern)}")
    
    # creat dir, nested if not exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)


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


    model = make_model(device, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.ddpm_iteration,
            eta_min=args.lr * args.lr_eta_min_ratio,
        )
    elif args.lr_schedule == "constant":
        lr_scheduler = None
    else:
        raise ValueError(f"Unsupported lr_schedule: {args.lr_schedule}")

    if args.sampler == "ddpm":
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=T,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type=args.prediction_type,  # "epsilon" or "sample"
        )
    elif args.sampler == "ddim":
        from diffusers import DDIMScheduler

        ddpm_scheduler = DDIMScheduler(
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
    if  args.fresh_run:
        start_step, config = 0,{}
    else:
        start_step, config = load_checkpoint(
            model,
            optimizer,
            re.sub(r"it\d+", "it*", checkpoint_path),
            ddpm_iteration,
            device=device,
            lr_scheduler=lr_scheduler,
        )
    assert start_step > 0, f"Checkpoint loading failed. Please check the checkpoint path: {checkpoint_path}"
    
    if True: #load data

        if args.cond_method in [ "wan", "scene_id"]:
            cond_method = args.cond_method
            cond_string = f"{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}"
        elif args.cond_method == "none": #get "wan", then set to None
            cond_method = "wan"
            cond_string = f"pdnorm_only_5_center_1_skip"
            
        data_key = f"{args.data_file}_side{args.sensor_side}_{cond_method}_{args.N}_{cond_string}"
        print(f"data_key: {data_key}")
        gather_cahce_dir = os.path.join(cache_dir, data_key )
        os.makedirs(gather_cahce_dir, exist_ok=True)

        whole_ds_cache_fname= {k: f"man_{k}.npy" for k in ["x0sbn5_all", "cond_all"]}
        whole_ds_cache_fname.update({"frame_ids_all": f"man_frame_ids_all.json"})


        if not all(os.path.exists(os.path.join(gather_cahce_dir, fname)) for fname in whole_ds_cache_fname.values()): #prepare for lazy loading through NPY's MemMap
            print(f"some cache files are missing, gathering MAN dataset from individual scene cache files. This may take a while...")
            raise ValueError(f"Some cache files are missing in {gather_cahce_dir}. Please run the data gathering script to prepare the dataset before training.")
        else:
            
            print(f"Loading MAN dataset from cache: {whole_ds_cache_fname}")

            time_recorder = TimeRecorder(insert_order=True, cuda_sync=True)
            if args.lazy_npy:
                x0sbn5_all, cond_all= [ LazyNpyArray(os.path.join(gather_cahce_dir, whole_ds_cache_fname[k])) for k in ["x0sbn5_all", "cond_all"]]
            else:#load all to RAM
                x0sbn5_all, cond_all = [torch.as_tensor(np.load(os.path.join(gather_cahce_dir, whole_ds_cache_fname[k]), allow_pickle=False)) for k in ["x0sbn5_all", "cond_all"]]
            with open(os.path.join(gather_cahce_dir, whole_ds_cache_fname["frame_ids_all"]), "r") as f:
                frame_ids_all = json.load(f)         


            all_frame_ids=frame_ids_all
            time_recorder.record("load_from_cache")
            print(f"Loaded MAN dataset from cache in {time_recorder.get_records()}  with {'LazyNpyArray' if args.lazy_npy else 'non-Lazy'}")
            
            print(f"Gathered MAN dataset: {x0sbn5_all.shape} samples") 
            # print(f"cond : {cond_all}")
            print(f"cond shape: {cond_all.shape if cond_all is not None else None}") 
            print(f"frame_ids: {len(all_frame_ids['token'])} tokens") 
            print(f"{len(all_frame_ids['scene_id'])} scene_ids") 
            print(f"{len(all_frame_ids['frame_index'])} frame_indices")
            print(f"{len(all_frame_ids['sensor_side'])} sensor_sides") 
            print(f"{len(all_frame_ids['data_file'])} data_files key {data_key}")
            # Gathered MAN dataset: (43373, 128, 3) samples, cond shape: (43373, 16, 2, 60, 104), doppler shape: (43373, 128, 1), rcs shape: (43373, 128, 1) frame_ids: 43373 tokens, 43373 scene_ids, 43373 frame_indices, 43373 sensor_sides, 43373 data_files key both_wan_128_pdnorm_only_5_center_1_skip


        
        print(f"Gathered MAN dataset: {x0sbn5_all.shape} samples, cond shape: {cond_all.shape if cond_all is not None else None}")


        #make sure all token is 32 cahr long
        five_ch_token={}
        for i, token in enumerate(all_frame_ids['token']):
            if len(token) != 32:
                if all_frame_ids['data_file'][i] not in five_ch_token:
                    five_ch_token[all_frame_ids['data_file'][i]] = {}
                if all_frame_ids['sensor_side'][i] not in five_ch_token[all_frame_ids['data_file'][i]]:
                    five_ch_token[all_frame_ids['data_file'][i]][all_frame_ids['sensor_side'][i]] = set()
                five_ch_token[all_frame_ids['data_file'][i]][all_frame_ids['sensor_side'][i]].add(all_frame_ids['scene_id'][i])
        print(f" 5-character tokens in eval set: {five_ch_token}") 

        if len(five_ch_token) > 0:
            print(f"Warning: Found {len(five_ch_token)} 5-character tokens in the dataset. {five_ch_token}")
            raise ValueError(f"Found {len(five_ch_token)} 5-character tokens in the dataset. {five_ch_token}. Please check the dataset and ensure all tokens are 32 characters long.")
        
        
        print(f"MAN dataset loaded from cache. frame_ids (first 8 items): ")
        for k in all_frame_ids.keys():
            print(f"{k}: {all_frame_ids[k][:8]}")


        if args.shape_name == "man_heldout_split":
            split_seed = args.split_seed 

            assert x0sbn5_all.shape[0] == len(all_frame_ids["token"])== len(all_frame_ids["scene_id"]) == len(all_frame_ids["frame_index"]) == len(all_frame_ids["data_file"]) == len(all_frame_ids["sensor_side"]), f"Mismatch in number of samples and frame IDs: {x0sbn5_all.shape[0]} vs {len(all_frame_ids['token'])}"
            
            allids = list(range(x0sbn5_all.shape[0]))

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
                f"After split: x0sbn5_all={x0sbn5_all.shape[0]}, "
                f"train_frames={train_idx_pool.numel()}, "
                f"eval_frames={eval_idx_pool.numel()}, "
                f"test_frames={0 if test_idx_pool is None else test_idx_pool.numel()}"
            )
            print(f"eval_scene_set={sorted(list(eval_scene_set))}")
            print(f"test_scene_set={sorted(list(test_scene_set))}")

            norm_stats = compute_norm_stats_from_train(
                x0sbn5_src=x0sbn5_all,
                cond_src=cond_all,
                train_idx_pool=train_idx_pool,
                chunk_size=512,
            )
            

            print(f"Data normalization statistics from train set: {norm_stats}")

            norm_stats_path = os.path.join(data_dir, "normalization_stats.json")

            with open(norm_stats_path, "w") as f:

                json.dump(norm_stats, f, indent=2)

            x0sbn5_norm = NormalizedX0Array(x0sbn5_all, norm_stats)

            if cond_all is not None:

                cond_norm = NormalizedCondArray(cond_all, norm_stats["cond_absmax"])

            else:

                cond_norm = None
        else:
            raise NotImplementedError(f"Shape name {args.shape_name} is not implemented yet. Please use 'man_heldout_split' for MAN dataset.")


        print(
            f"Dataset created. Full samples: {x0sbn5_all.shape[0]}, "
            f"Train indexed frames: {train_idx_pool.numel()}, "
            f"Eval indexed frames: {eval_idx_pool.numel()}"
        )

        transverse_lim_meters = (
            norm_stats["x0sbn3"]["mean"][1] - norm_stats["x0sbn3"]["max_half_range"],
            norm_stats["x0sbn3"]["mean"][1] + norm_stats["x0sbn3"]["max_half_range"],
        )
        longitudinal_lim_meters = (
            norm_stats["x0sbn3"]["mean"][0] - norm_stats["x0sbn3"]["max_half_range"],
            norm_stats["x0sbn3"]["mean"][0] + norm_stats["x0sbn3"]["max_half_range"],
        )
        print(f"transverse_lim_meters: {transverse_lim_meters}")
        print(f"longitudinal_lim_meters: {longitudinal_lim_meters}")


        # assert x0sbn3_train_norm.shape[0] >= n_scene, f"Not enough training scenes in the dataset. Requested: {n_scene}, available: {x0sbn3_train_norm.shape[0]}"
        n_scene = train_idx_pool.numel()
        print(
            f"shapes after dataset creation: x0sbn5_all {x0sbn5_all.shape}, cond {cond_all.shape if cond_all is not None else None}"
        )
        
        inout_dim = 5  if args.train_rcs_doppler else 3
    
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
        print(f"Starting evaluation mode.")

        trucksc_all={'man-mini': {"data_root": "/data/palakons/new_dataset/MAN/mini/man-truckscenes", "version": "v1.0-mini","sc_class": TruckScenes("v1.0-mini", "/data/palakons/new_dataset/MAN/mini/man-truckscenes", False)},
                'man-full': {"data_root": "/data/palakons/new_dataset/MAN/man-truckscenes", "version": "v1.0-trainval","sc_class": TruckScenes("v1.0-trainval", "/data/palakons/new_dataset/MAN/man-truckscenes", False)}}

        # c_name = "correct_cond" 
        sampling_seed = 42
        
        # cond_use = ['correct_cond','none','zero_cond','shuffled_cond','nn_retrieval']
        cond_use = ['correct_cond','nn_retrieval','zero_cond','shuffled_cond',]

        for c_name in cond_use:
            per_frame_cds = []
            sampled_batch_cd_fname = f"sampled_stat_{c_name}_sd{sampling_seed}.csv"
            sampled_batch_cd_path = os.path.join(inference_dir, sampled_batch_cd_fname)

            sampled_batch_cache_fname = f"sampled_batch_cache_{c_name}_sd{sampling_seed}.pkl"
            sampled_batch_cache_path = os.path.join(inference_dir, sampled_batch_cache_fname)
            if os.path.exists(sampled_batch_cache_path):
                print(f"Loading sampled batch cache from {sampled_batch_cache_path}")
                with open(sampled_batch_cache_path, "rb") as f:
                    sampled_batch_cache = pickle.load(f)
                pred_all = sampled_batch_cache["pred_all"]
                cond_used_all = sampled_batch_cache["cond_used_all"]
            else:            
                gt_eval_norm = x0sbn5_norm[eval_idx_pool][:, :,:inout_dim]
                cond_eval_norm = cond_norm[eval_idx_pool] if cond_norm is not None else None


                shuffle_perm = None
                if c_name == "shuffled_cond":
                    full_B = gt_eval_norm.shape[0]
                    g_shuffle = torch.Generator(device="cpu").manual_seed(sampling_seed + 12345)
                    shuffle_perm = torch.randperm(full_B, generator=g_shuffle)
                    while torch.equal(shuffle_perm, torch.arange(full_B)) and full_B > 1:
                        shuffle_perm = torch.randperm(full_B, generator=g_shuffle)

                pred_all, cond_used_all = sample_or_retrieve_in_batches(
                    model=model,
                    scheduler=ddpm_scheduler,
                    gt_all= gt_eval_norm,
                    cond_all= cond_eval_norm if cond_eval_norm is not None else None,
                    
                    
                    cond_train_norm=cond_norm,
                    x0sbn3_train_norm=x0sbn5_norm,
                    
                    c_name=c_name,
                    seed=sampling_seed,
                    N=N,
                    inout_dim=inout_dim,
                    T_infer=T_infer,
                    device=device,
                    batch_size=512,   # tune this
                    shuffle_perm=shuffle_perm,
                    train_idx_pool=train_idx_pool,   # REQUIRED
                )

                with open(sampled_batch_cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "pred_all": pred_all,
                            "cond_used_all": cond_used_all,
                        },
                        f,
                    )
            print(f"pred_all, cond_used_all shapes: {pred_all.shape}, {cond_used_all.shape if cond_used_all is not None else None}, eval_idx_pool numel: {eval_idx_pool.numel()}")
                
            
            for frame_token, scene_id, frame_index, sensor_side, data_file,pred,gt in tqdm(zip(frame_ids['eval']['token'], frame_ids['eval']['scene_id'], frame_ids['eval']['frame_index'], frame_ids['eval']['sensor_side'], frame_ids['eval']['data_file'], pred_all, x0sbn5_norm[eval_idx_pool]), desc="Processing frames", total=len(frame_ids['eval']['token'])):
                pred_unnormed = unnormalize_data(pred.clone(), norm_stats)
                gt_unnormed = unnormalize_data(gt.clone(), norm_stats)

                if c_name == 'correct_cond':
                    # print(f"Token: {frame_token}, Scene ID: {scene_id}, Frame Index: {frame_index}, Sensor Side: {sensor_side}, Data File: {data_file}, Pred Shape: {pred.shape}, GT Shape: {gt.shape}")
                    trucksc = trucksc_all[data_file]["sc_class"]
                    frame = trucksc.get("sample", frame_token)
                    # print(f"Loaded sample for token {frame_token}: Sample keys: {list(frame.keys())}")
                    cam = trucksc.get("sample_data", frame["data"][f'CAMERA_{sensor_side.upper()}_FRONT'])
                    # print(f"Loaded camera data for sensor side {sensor_side}: Camera keys: {list(cam.keys())}")

                    image_rgb_path = os.path.join(trucksc_all[data_file]["data_root"], cam["filename"])
                
                    if not os.path.exists(image_rgb_path):
                        print(f"Image file does not exist: {image_rgb_path}")
                        raise FileNotFoundError(f"Image file does not exist: {image_rgb_path}")
                    else:
                        
                        print(f"Image file path: {image_rgb_path}, token: {frame_token}, scene_id: {scene_id}, frame_index: {frame_index}, sensor_side: {sensor_side}, data_file: {data_file}")
                        #plot image, gt, pred
                        save_fname =  f"combo_{data_file}_{sensor_side}_sc-{scene_id}_fr-{frame_index}.png"
                        save_path = os.path.join(inference_dir, save_fname)
                        title=f"RGB Image: {data_file}, {sensor_side}, scene {scene_id}, frame {frame_index}"

                        plot_combo(image_rgb_path,pred,gt,save_path,title)
                        # plot_combo(image_rgb_path,pred_unnormed,gt_unnormed,save_path,title)
                        exit()
                        

                pointset_error_stat = calculate_pointset_stat(pred_unnormed.unsqueeze(0), gt_unnormed.unsqueeze(0))

                per_frame_cds.append({'data_file': data_file, 'sensor_side': sensor_side, 'scene_id': scene_id, 'frame_index': frame_index, 'token': frame_token,"condition_use":c_name } | pointset_error_stat)
                    
                #save csv
                sampled_batch_cd_df = pd.DataFrame(per_frame_cds)
                sampled_batch_cd_df.to_csv(sampled_batch_cd_path, index=False)
                

        print(f"Saved per-frame Chamfer distances to {sampled_batch_cd_path}")

        #aggregate per-scene Chamfer distances
        per_scene_cds = []
        for (data_file, sensor_side, scene_id, condition_use), group in sampled_batch_cd_df.groupby(['data_file', 'sensor_side', 'scene_id', 'condition_use']):
            mean_cd = group['xyz_cd'].mean()
            std_cd = group['xyz_cd'].std()
            per_scene_cds.append({'data_file': data_file, 'sensor_side': sensor_side, 'scene_id': scene_id, 'condition_use': condition_use, 'mean_cd': mean_cd, 'std_cd': std_cd})
        best_sc = min(per_scene_cds, key=lambda x: x['mean_cd'])
        print(f"Best scene: {best_sc}")
        worst_sc = max(per_scene_cds, key=lambda x: x['mean_cd'])
        print(f"Worst scene: {worst_sc}")

                
    elif args.mode == "train":
        raise NotImplementedError("Training mode is not implemented yet. Please use 'eval' or 'interpolate' modes.")
