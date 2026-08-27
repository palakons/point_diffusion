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
from fitone_dfs_cond_log import find_nn_cond_exact_chunked,sample_or_retrieve_in_batches,none_if_all_zero,append_per_scene_eval_rows,append_eval_row,debug_batch,check_model,check_tensor,chamfer_xyz_with_matched_attrs,p_sample_loop,shorten_run_id,make_run_id,auto_fill_scene_sets,filter_valid_scene_keys,parse_scene_set_spec,_compress_ids,scene_set_tag,reconstruct_x0,frame_key,take_frame_ids,gather_man_ds,train_eval_step,eval_multi_batch,TimeRecorder,make_frame_meta_np,append_per_frame_eval_rows,LazyNpyArray,compute_norm_stats_from_train,NormalizedX0Array,NormalizedCondArray,parse_args
from truckscenes import TruckScenes
from scipy.ndimage import gaussian_filter
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







    
def unnormalize_data(x, norm_stats):
    x = x.clone()

    xyz_mean = torch.as_tensor(
        norm_stats["x0sbn3"]["mean"],
        dtype=x.dtype,
        device=x.device,
    )

    x[:, :3] = (
        x[:, :3] * norm_stats["x0sbn3"]["max_half_range"]
        + xyz_mean
    )

    if x.shape[-1] == 5:
        doppler_mean = torch.as_tensor(
            norm_stats["doppler"]["mean"],
            dtype=x.dtype,
            device=x.device,
        )

        rcs_mean = torch.as_tensor(
            norm_stats["rcs"]["mean"],
            dtype=x.dtype,
            device=x.device,
        )

        x[:, 3:4] = (
            x[:, 3:4] * norm_stats["doppler"]["max_half_range"]
            + doppler_mean
        )

        x[:, 4:5] = (
            x[:, 4:5] * norm_stats["rcs"]["max_half_range"]
            + rcs_mean
        )

    return x
def plot_combo(
    image_rgb_path,
    pred,
    gt,
    save_path,
    title,
    split_bottom=False,
    pred_multi=None,
    sigma_m=1.5
):
    img = plt.imread(image_rgb_path)

    # Convert if needed
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred)

    if torch.is_tensor(gt):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt)

    # CD_xyz [m^2]
    pred_t = torch.as_tensor(pred_np[:, :3], dtype=torch.float32)[None]
    gt_t   = torch.as_tensor(gt_np[:, :3], dtype=torch.float32)[None]

    cd_xyz = pt3d_chamfer_distance(pred_t, gt_t)[0].item()

    if not split_bottom:
        # ---------------- original overlay ----------------
        fig, axs = plt.subplots(
            2, 1,
            figsize=(6, 5),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )

        axs[0].imshow(img)
        axs[0].set_title(title, pad=2)
        axs[0].axis("off")

        axs[1].scatter(gt_np[:, 0], gt_np[:, 2],
                       c="blue", s=2, label="GT")
        axs[1].scatter(pred_np[:, 0], pred_np[:, 2],
                       c="red", s=2, label="Pred")

        axs[1].set_title(
            rf"GT vs Pred   CD$_{{xyz}}$={cd_xyz:.2f} m$^2$",
            pad=2,
        )
        axs[1].set_xlabel("X (m)")
        axs[1].set_ylabel("Z (m)")
        axs[1].set_xlim(-50, 50)
        axs[1].set_ylim(0, 50)
        axs[1].set_aspect("equal")
        axs[1].legend(
            loc="upper right",
            fontsize=7,
            markerscale=2,
            frameon=False,
        )

        plt.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.07,
            top=0.96,
            hspace=0.08,
        )

    else:
        fig = plt.figure(figsize=(7, 5.0))

        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[ 1.0,.75],
            hspace=0.02,
            wspace=0.0,
        )

        ax_img = fig.add_subplot(gs[0, :])
        ax_gt = fig.add_subplot(gs[1, 0])
        ax_pred = fig.add_subplot(gs[1, 1], sharey=ax_gt)

        ax_img.imshow(img)
        ax_img.set_title(title, pad=1)
        ax_img.axis("off")
        ax_img.set_anchor("S")

        # GT / Pred
        for ax in (ax_gt, ax_pred):
            ax.set_xlim(-30,30)
            ax.set_ylim(0, 50)
            ax.set_aspect("equal")
            # ax.set_xlabel("X (m)", labelpad=1)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)        

        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax_gt.scatter(gt_np[:, 0], gt_np[:, 2], c="blue", s=1, label="GT")
        ax_gt.set_xlabel("GT - X (m)", labelpad=1)
        # ax_pred.scatter(pred_np[:, 0], pred_np[:, 2], c="red", s=1, label="Pred")
        # print(f"pred_multi: {len(pred_multi) if pred_multi is not None else 0} samples")
        if pred_multi is None or len(pred_multi) <= 1:
            ax_pred.scatter(
                pred_np[:, 0],
                pred_np[:, 2],
                c="red",
                s=1,label="Pred"
            )

        else:
            # [n_multi * N, 3]
            multi_np = np.concatenate([
                p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
                for p in pred_multi
            ], axis=0)
            # 2D density in camera X-Z plane
            x_range = (-30, 30)
            z_range = (0, 50)
            x_bins = 120
            z_bins = 100

            H, xedges, zedges = np.histogram2d(
                multi_np[:, 0],
                multi_np[:, 2],
                bins=(x_bins, z_bins),
                range=[x_range, z_range],
            )

            # Gaussian bandwidth defined in physical units [m]
            # sigma_m = 1.0

            dx = (x_range[1] - x_range[0]) / x_bins   # 0.5 m/bin
            dz = (z_range[1] - z_range[0]) / z_bins   # 0.5 m/bin

            sigma_bins = (
                sigma_m / dx,
                sigma_m / dz,
            )

            H = gaussian_filter(H, sigma=sigma_bins)

            ax_pred.imshow(
                H.T,
                origin="lower",
                extent=[*x_range, *z_range],
                aspect="equal",
                cmap="Blues",
                interpolation="bilinear",
            )
            # ax_pred.scatter(
            #     pred_np[:, 0],
            #     pred_np[:, 2],
            #     c="red",
            #     s=0.5,
            #     alpha=0.3,
            # )
            



        # ax_pred.set_xlabel("Pred - X (m)", labelpad=1)
        ax_pred.set_xlabel(
            f"Pred ({len(pred_multi) if pred_multi is not None else 1} samples) - X (m)",
            labelpad=1,
        )

        # ax_gt.set_title(f"GT", pad=1)
        # ax_pred.set_title(f"Pred   CD$_{{xyz}}$={cd_xyz:.2f} m$^2$", pad=1)

        ax_gt.set_anchor("E")
        ax_pred.set_anchor("W")

        ax_gt.set_ylabel(rf"Z (m); CD$_{{xyz}}^{{(1)}}$={cd_xyz:.2f} m$^2$", labelpad=1)
        ax_pred.tick_params(axis="y", left=False, labelleft=False)

        #make axis font size, smaller
        for ax in (ax_img, ax_gt, ax_pred):
            ax.tick_params(axis="both", which="major", labelsize=7, length=2, width=0.5)
            ax.tick_params(axis="both", which="minor", labelsize=7, length=1, width=0.5)
            #also axis label font size
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(7)




    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def plot_combo_old(image_rgb_path,pred,gt,save_path,title):

    #plot img, ont he left, gt/pred top view ont he right, with gt in blue, pred in red
    fig,axs =  plt.subplots(2,1,figsize=(6,6))
    #plot image
    img = plt.imread(image_rgb_path)
    axs[0].imshow(img)
    axs[0].set_title(title)
    axs[0].axis('off')
    #plot gt/pred top view  , x is horizontal, z is vertical
    axs[1].scatter(gt[:,0], gt[:,2], c='blue', s=1, label='GT')
    # x_range=[-50, 50],

    # z_range=[0, 50],
    axs[1].scatter(pred[:,0], pred[:,2], c='red', s=1, label='Pred')
    axs[1].set_title(f"Top View: GT (blue) vs Pred (red)")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Z (m)")
    #set equal aspect ratio
    # axs[1].set_aspect('equal', adjustable='box')

    #reset range [-1,1] to *_lim_meters
    axs[1].set_xlim(-50, 50)
    axs[1].set_ylim(00, 50)

    #make sure aspect is equal
    axs[1].set_aspect('equal')
    


    axs[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    # print(f"Saved inference figure to {save_path}")

def select_medoid_sample(pred_multi):
    """
    pred_multi: list of K tensors [N,3]
    """
    K = len(pred_multi)

    if K == 1:
        return pred_multi[0], 0

    scores = []

    for i in range(K):
        cds = []

        for j in range(K):
            if i == j:
                continue

            cd = pt3d_chamfer_distance(
                pred_multi[i][None, :, :3],
                pred_multi[j][None, :, :3],
            )[0]

            cds.append(cd.item())

        scores.append(np.mean(cds))

    idx = int(np.argmin(scores))

    return pred_multi[idx], idx

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
    inference_dir = f"{data_dir}/inference_sc{args.eval_scene_id}-multi{args.n_multi}"
    checkpoint_dir = f"/data/palakons/{system_key}/checkpoints/"
    # cache_dir = f"/data/palakons/{system_key}/cache/"
    cache_dir = f"/data/palakons/{system_key}/cache{'_cameraxyz' if args.coord_frame == 'camera' else ''}_unnorm/"
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
            
        data_key = f"{args.data_file}_side{args.sensor_side}{'_cameraframe' if args.coord_frame == 'camera' else ''}_{cond_method}_{args.N}_{cond_string}"
        print(f"data_key: {data_key}")
        gather_cache_dir = os.path.join(cache_dir, data_key )
        os.makedirs(gather_cache_dir, exist_ok=True)

        whole_ds_cache_fname= {k: f"man_{k}.npy" for k in ["x0sbn5_all", "cond_all"]}
        whole_ds_cache_fname.update({"frame_ids_all": f"man_frame_ids_all.json"})


        if not all(os.path.exists(os.path.join(gather_cache_dir, fname)) for fname in whole_ds_cache_fname.values()): #prepare for lazy loading through NPY's MemMap
            print(f"some cache files are missing, gathering MAN dataset from individual scene cache files. This may take a while...")
            raise ValueError(f"Some cache files are missing in {gather_cache_dir}. Please run the data gathering script to prepare the dataset before training.")
        else:
            
            print(f"Loading MAN dataset from cache: {whole_ds_cache_fname}")

            time_recorder = TimeRecorder(insert_order=True, cuda_sync=True)
            if args.lazy_npy:
                x0sbn5_all, cond_all= [ LazyNpyArray(os.path.join(gather_cache_dir, whole_ds_cache_fname[k])) for k in ["x0sbn5_all", "cond_all"]]
            else:#load all to RAM
                x0sbn5_all, cond_all = [torch.as_tensor(np.load(os.path.join(gather_cache_dir, whole_ds_cache_fname[k]), allow_pickle=False)) for k in ["x0sbn5_all", "cond_all"]]
            with open(os.path.join(gather_cache_dir, whole_ds_cache_fname["frame_ids_all"]), "r") as f:
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

            if args.eval_scene_id >= 0:
                eval_idx_pool = torch.as_tensor(
                    [
                        i for i in eval_idx_pool.tolist()
                        if all_frame_ids["scene_id"][i] == args.eval_scene_id
                    ],
                    dtype=torch.long,
                )

                assert eval_idx_pool.numel() > 0, \
                    f"No eval frames found for scene_id={args.eval_scene_id}"

            print(
                f"Filtered eval to scene {args.eval_scene_id}: "
                f"{eval_idx_pool.numel()} frames"
            )
            
            
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
                chunk_size=4096,
            )#this takes 3 mins
            

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
        print(f"Starting evaluation mode. loading TruckScenes dataset for evaluation...")
        time0= time.time()
        trucksc_all={
            # 'man-mini': {"data_root": "/data/palakons/new_dataset/MAN/mini/man-truckscenes", "version": "v1.0-mini","sc_class": TruckScenes("v1.0-mini", "/data/palakons/new_dataset/MAN/mini/man-truckscenes", False)},
                'man-full': {"data_root": "/data/palakons/new_dataset/MAN/man-truckscenes", "version": "v1.0-trainval","sc_class": TruckScenes("v1.0-trainval", "/data/palakons/new_dataset/MAN/man-truckscenes", False)}}
        print(f"Loaded TruckScenes dataset for evaluation in {time.time()-time0:.2f} seconds.")

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

            # ---------------------------------------------------------
            # Multiple stochastic samples for uncertainty visualization
            # Keep pred_all as pass 0 for all existing quantitative metrics.
            # ---------------------------------------------------------
            pred_multi_all = [pred_all]

            if c_name == "correct_cond" and args.n_multi > 1:

                print(f"Generating {args.n_multi} stochastic passes...")

                gt_eval_norm = x0sbn5_norm[eval_idx_pool][:, :, :inout_dim]
                cond_eval_norm = cond_norm[eval_idx_pool] if cond_norm is not None else None

                for multi_i in trange(1, args.n_multi, desc="Pred multiple stochastic passes"):

                    pred_i, _ = sample_or_retrieve_in_batches(
                        model=model,
                        scheduler=ddpm_scheduler,
                        gt_all=gt_eval_norm,
                        cond_all=cond_eval_norm,

                        cond_train_norm=cond_norm,
                        x0sbn3_train_norm=x0sbn5_norm,

                        c_name=c_name,
                        seed=sampling_seed + multi_i,

                        N=N,
                        inout_dim=inout_dim,
                        T_infer=T_infer,
                        device=device,
                        batch_size=1024,
                        shuffle_perm=None,
                        train_idx_pool=train_idx_pool,
                    )

                    pred_multi_all.append(pred_i)
            print(f"pred_all, cond_used_all shapes: {pred_all.shape}, {cond_used_all.shape if cond_used_all is not None else None}, eval_idx_pool numel: {eval_idx_pool.numel()}")
                
            pred_export = {}
            for eval_i, (
                frame_token,
                scene_id,
                frame_index,
                sensor_side,
                data_file,
                pred,
                gt,
            ) in enumerate(tqdm(
                zip(
                    frame_ids['eval']['token'],
                    frame_ids['eval']['scene_id'],
                    frame_ids['eval']['frame_index'],
                    frame_ids['eval']['sensor_side'],
                    frame_ids['eval']['data_file'],
                    pred_all,
                    x0sbn5_norm[eval_idx_pool],
                ),
                desc="Processing frames",
                total=len(frame_ids['eval']['token']),
            )):
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

                        pred_multi_unnormed = [
                            unnormalize_data(pred_pass[eval_i].clone(), norm_stats)
                            for pred_pass in pred_multi_all
                        ]
                        pred_medoid, medoid_idx = select_medoid_sample(
                            pred_multi_unnormed
                        )

                        medoid_stats = calculate_pointset_stat(
                            pred_medoid.unsqueeze(0),
                            gt_unnormed.unsqueeze(0),
                        )

                        medoid_cd = medoid_stats["xyz_cd"]
                        medoid_f2 = medoid_stats["fscore_2m"]

                        multi_stats = [
                            calculate_pointset_stat(
                                p.unsqueeze(0),
                                gt_unnormed.unsqueeze(0),
                            )
                            for p in pred_multi_unnormed
                        ]

                        multi_cd_mean = np.mean([s["xyz_cd"] for s in multi_stats])
                        multi_cd_std = np.std([s["xyz_cd"] for s in multi_stats])

                        multi_f2_mean = np.mean([s["fscore_2m"] for s in multi_stats])
                        multi_f2_std = np.std([s["fscore_2m"] for s in multi_stats])
                        
                        # print(f"Image file path: {image_rgb_path}, token: {frame_token}, scene_id: {scene_id}, frame_index: {frame_index}, sensor_side: {sensor_side}, data_file: {data_file}")
                        #plot image, gt, pred
                        save_fname =  f"combo_{data_file}_{sensor_side}_sc-{scene_id}_fr-{frame_index}.png"
                        save_path = os.path.join(inference_dir, save_fname)
                        title=f"RGB Image: {data_file}, {sensor_side}, scene {scene_id}, frame {frame_index}"

                        # plot_combo(image_rgb_path,pred,gt,save_path,title)
                        plot_combo(image_rgb_path,pred_unnormed,gt_unnormed,save_path,title, split_bottom=True,pred_multi=pred_multi_unnormed,sigma_m=args.plot_sigma_m)
                        # plot medoid   
                        plot_combo(image_rgb_path,pred_medoid,gt_unnormed,save_path.replace('.png', '_medoid.png'),title, split_bottom=True,pred_multi=None,sigma_m=args.plot_sigma_m)
                        # plot_combo(image_rgb_path,pred_unnormed,gt_unnormed,save_path.replace('.png', '_notsplit.png'),title, split_bottom=False)

                        # print(f"shape of pred_unnorm {pred_unnormed.shape}, gt_unnorm {gt_unnormed.shape}, pred_medoid {pred_medoid.shape}, medoid_idx {medoid_idx}") #shape of pred_unnorm torch.Size([128, 3]), gt_unnorm torch.Size([128, 5]), pred_medoid torch.Size([128, 3]), medoid_idx 9

                        pred_export[frame_token] = {
                            "pred_xyz_camera": pred_medoid[:, :3].cpu().numpy(),
                            "sensor_side": sensor_side,
                            "scene_id": scene_id,
                            "frame_index": frame_index,
                        }
                     

                pointset_error_stat = calculate_pointset_stat(pred_unnormed.unsqueeze(0), gt_unnormed.unsqueeze(0))
                pointset_error_stat.update({
                    "multi_xyz_cd_mean": multi_cd_mean,
                    "multi_xyz_cd_std": multi_cd_std,
                    "multi_fscore_2m_mean": multi_f2_mean,
                    "multi_fscore_2m_std": multi_f2_std,
                    "medoid_xyz_cd": medoid_cd,
                    "medoid_fscore_2m": medoid_f2,
                })

                per_frame_cds.append({'data_file': data_file, 'sensor_side': sensor_side, 'scene_id': scene_id, 'frame_index': frame_index, 'token': frame_token,"condition_use":c_name } | pointset_error_stat)
                    
                #save csv
                sampled_batch_cd_df = pd.DataFrame(per_frame_cds)
                sampled_batch_cd_df.to_csv(sampled_batch_cd_path, index=False)

            with open(
                os.path.join(inference_dir, f"rerun_predictions_{c_name}.pkl"),
                "wb",
            ) as f:
                pickle.dump(pred_export, f)   

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
