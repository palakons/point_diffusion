import re
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os
from tqdm import tqdm, trange
import numpy as np

import numpy as np
from pathlib import Path


def save_point_sample(path, pred, gt=None, condition=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"pred": pred}
    if gt is not None:
        data["gt"] = gt
    if condition is not None:
        data["condition"] = condition
    if meta is not None:
        data["meta"] = np.array(str(meta))

    np.savez_compressed(path, **data)

def make_man_pc(
    num_points=64, n_scene=1, device="cpu", is_dense=False, data_file="man-mini"
):
    # B 128 128 pt 9.482Gi/15.992Gi
    import sys

    sys.path.append("/home/palakons/point_diffusion")
    from man_ddpm import MANDataset
    if is_dense:
        ds = MANDataset(
            # scene_ids=list(range(450,598)),
            scene_ids=[],
            data_file=data_file,
            device=device,
            wan_vae=True,
            wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
            n_p=num_points,
            normalize_type="minmax",
            get_camera=False,
            keep_frames=n_scene,
            point_preset="original",
            x_range=[0, 50],
            y_range=[-50, 50],
            z_range=[-2, 2],
            wan_preprocess_dir="/data/palakons/man_wan_preprocessed",
            coord_only=False
        )

        # print(f"keys in man dataset item: {ds[0].keys()}")  # keys
        npoints_originals =[ds[i]['npoints_original'] for i in range(n_scene)]
        npoints_after_distance_filter = [ds[i]['npoints_after_distance_filter'] for i in range(n_scene)]
        npoints_filtereds = [ds[i]['npoints_filtered'] for i in range(n_scene)]
        # print(f"n point npoints_original {ds[0]['npoints_original']}, npoints_after_distance_filter: {ds[0]['npoints_after_distance_filter']}, npoints_filtered: {ds[0]['npoints_filtered']}")  #n point npoints_original 800, npoints_after_distance_filter: 185, npoints_filtered: 135
        #print min max mean
        print(f"npoints_originals: min {min(npoints_originals)}, max {max(npoints_originals)}, mean {sum(npoints_originals)/len(npoints_originals)}")
        print(f"npoints_after_distance_filter: min {min(npoints_after_distance_filter)}, max {max(npoints_after_distance_filter)}, mean {sum(npoints_after_distance_filter)/len(npoints_after_distance_filter)}")
        print(f"npoints_filtereds: min {min(npoints_filtereds)}, max {max(npoints_filtereds)}, mean {sum(npoints_filtereds)/len(npoints_filtereds)}")
        # npoints_originals: min 173, max 800, mean 644.32
        # npoints_after_distance_filter: min 81, max 273, mean 179.62
        # npoints_filtereds: min 58, max 221, mean 136.66
        # exit()
        x0sbn3 = torch.stack(
            [ds[i]["filtered_radar_data"] for i in range(n_scene)], dim=0
        ).to(
            device
        )  # [B, N, 3]
        wan_cond = torch.stack(
            [ds[i]["wan_vae_latent"] for i in range(n_scene)], dim=0
        ).to(
            device
        )  # [B, latent_dim]
        return x0sbn3[:,:,:3], wan_cond, ds, x0sbn3[:,:,3:6],x0sbn3[:,:,6:]

    else:
        ds = [
            MANDataset(
                scene_ids=[i],
                data_file=data_file,
                device=device,
                wan_vae=True,
                wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
                n_p=num_points,
                normalize_type="minmax",
                get_camera=False,
                keep_frames=1,
                point_preset="original",
                x_range=[0, 50],
                y_range=[-50, 50],
                z_range=[-2, 2],
                wan_preprocess_dir="/data/palakons/man_wan_preprocessed",
            coord_only=False
            )
            for i in range(n_scene)
        ]
        combined_ds = torch.utils.data.ConcatDataset(ds)
        x0sbn3 = torch.stack([data[0]["filtered_radar_data"] for data in ds], dim=0).to(
            device
        )  # [B, N, 3]
        wan_cond = torch.stack([data[0]["wan_vae_latent"] for data in ds], dim=0).to(
            device
        )  # [B, latent_dim]
        # print(f"shapes x0sbn3: {x0sbn3.shape}, wan_cond: {wan_cond.shape}") #shapes x0sbn3: torch.Size([B, 128, 3]), wan_cond: torch.Size([B, 16, 2, 60, 104])
        return x0sbn3[:,:,:3], wan_cond, combined_ds, x0sbn3[:,:,3:6],x0sbn3[:,:,6:]


def make_various_pc(num_points=64, device="cpu", n_shapes=7):
    theta = torch.linspace(0, math.pi / 2, num_points)
    x = torch.cos(theta)
    y = torch.sin(theta)
    z = torch.zeros_like(x)
    shape_wedge = torch.stack([x, y, z], dim=-1)  # wedge

    theta = torch.linspace(0, 4 * math.pi, num_points)
    z = torch.linspace(-1, 1, num_points)
    x = torch.cos(theta) * (z + 1)
    y = torch.sin(theta) * (z + 1)
    shape_spiral = torch.stack([x, y, z], dim=-1)  # spiral

    points_per_side = num_points // 6
    remainder = num_points % 6
    if remainder > 0:
        points_per_side += 1  # Distribute extra points to the first few sides

    sides = []
    for i in range(6):
        count = points_per_side
        if i == 0:  # Front
            x = torch.linspace(-1, 1, count)
            y = torch.linspace(-1, 1, count)
            z = torch.ones_like(x)
        elif i == 1:  # Back
            x = torch.linspace(-1, 1, count)
            y = torch.linspace(-1, 1, count)
            z = -torch.ones_like(x)
        elif i == 2:  # Left
            x = -torch.ones_like(x)
            y = torch.linspace(-1, 1, count)
            z = torch.linspace(-1, 1, count)
        elif i == 3:  # Right
            x = torch.ones_like(x)
            y = torch.linspace(-1, 1, count)
            z = torch.linspace(-1, 1, count)
        elif i == 4:  # Top
            x = torch.linspace(-1, 1, count)
            y = torch.ones_like(x)
            z = torch.linspace(-1, 1, count)
        else:  # Bottom
            x = torch.linspace(-1, 1, count)
            y = -torch.ones_like(x)
            z = torch.linspace(-1, 1, count)

        side_points = torch.stack([x, y, z], dim=-1)
        sides.append(side_points)

    shape_boxside = torch.cat(sides, dim=0)[:num_points]

    theta = torch.linspace(0, 2 * math.pi, num_points)
    x = 1.5 * torch.cos(theta)
    y = torch.sin(theta)
    z = torch.zeros_like(x)
    shape_oval = torch.stack([x, y, z], dim=-1)

    theta = torch.linspace(0, 2 * math.pi, num_points)
    phi = torch.linspace(0, math.pi, num_points)
    theta_grid, phi_grid = torch.meshgrid(theta, phi)
    r = 1 + 0.5 * torch.sin(3 * theta_grid) * torch.sin(2 * phi_grid)
    x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = r * torch.sin(phi_grid) * torch.sin(theta_grid)
    z = r * torch.cos(phi_grid)
    shape_metaball = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)[
        :num_points
    ]

    theta = torch.linspace(0, 2 * math.pi, num_points)
    r = 1 + 0.2 * torch.sin(5 * theta)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = 0.2 * torch.sin(3 * theta)
    shape_undulatingcircle = torch.stack([x, y, z], dim=-1)

    # B 128 128 pt 9.482Gi/15.992Gi
    import sys

    sys.path.append("/home/palakons/point_diffusion")
    from man_ddpm import MANDataset

    dataset = MANDataset(
        scene_ids=[0],
        data_file="man-mini",
        device=device,
        wan_vae=False,
        wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
        n_p=num_points,
        normalize_type="minmax",
        get_camera=False,
        keep_frames=1,
        point_preset="original",
        x_range=[0, 50],
        y_range=[-50, 50],
        z_range=[-2, 2],
    )
    data = dataset[0]
    shape_man = data["filtered_radar_data"]


    torch.manual_seed(42)
    random_shape = torch.rand(num_points, 3) * 2 - 1  # random shape in [-1,1]

    data = torch.stack(
        [
            shape_spiral,
            shape_undulatingcircle,
            shape_oval,
            shape_metaball,
            shape_wedge,
            random_shape,
            shape_boxside,
            shape_man,
        ],
        dim=0,
    ).to(device)
    print(f"old shape before adding extra features: {data.shape}") # should be [7, num_points, 3]
    #attached 2 2 tot he last dim, random data, to test conditioning on extra features
    torch.manual_seed(42) #this ensure the random features are the same across runs for consistency
    data = torch.cat([data, torch.rand_like(data[:,:,:2])], dim=-1)
    print(
        "Created various shapes point cloud with shape: ",
        data.shape,
        "bt will be used only first ",
        n_shapes,
        " shapes and ",
        num_points,
        " points per shape",
    )
    if False:  #will be normed outside
        # normalize each shape, subrtact mean, devide by max distance from mean
        data = data - data.mean(dim=[1], keepdim=True)
        # print("center : ", data.mean(dim=[1], keepdim=True)) # shape
        data = data / data.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    return data[:n_shapes, :num_points, :]

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_path, config):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved at step {step}: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, epoch, device="cuda"):
    """Load training checkpoint and return the step to resume from.
    checkpoint_path: wildcard path to checkpoint file, e.g., "checkpoints/latest*.pt". The function will load the most recent checkpoint matching the pattern.
    """
    assert "*" in checkpoint_path, "checkpoint_path must contain a wildcard '*' to match checkpoint files, e.g., 'checkpoints/latest*.pt'"
    matched_files = [
        f
        for f in os.listdir(os.path.dirname(checkpoint_path))
        if f.startswith(os.path.basename(checkpoint_path).split("*")[0])
        and f.endswith(os.path.basename(checkpoint_path).split("*")[1])
    ]
    print(
        f"Looking for checkpoints in {os.path.dirname(checkpoint_path)}/{os.path.basename(checkpoint_path)}. Found: {matched_files}"
    )
    latest_step = -1
    checkpoint = None
    for match in matched_files:
        _checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), match)
        try:
            loaded_checkpoint = torch.load(_checkpoint_path, map_location=device)
            print(f"Found checkpoint file: {match}, at step {loaded_checkpoint['step']}")
            if loaded_checkpoint["step"] > latest_step:
                if epoch <loaded_checkpoint["step"]:
                    print(
                        f"Checkpoint step {loaded_checkpoint['step']} is greater than current epoch {epoch}. Skipping this checkpoint in file {match}."
                    )
                    continue
                latest_step = loaded_checkpoint["step"]
                checkpoint = loaded_checkpoint
        except Exception as e:
            print(f"Error loading checkpoint {_checkpoint_path}: {e}")
            continue
    if checkpoint is None:
        print(f"No valid checkpoint found in matched files: {matched_files}")
        return 0, {}

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    # scheduler.load_state_dict(checkpoint['scheduler_state'])
    step = checkpoint["step"]
    config = checkpoint.get("config", {})
    print(f"Checkpoint loaded from: {checkpoint_path} (resuming from step {step})")
    return step, config

def normalize_data(x, mean=None, max_half_range=None,save_filename_title=None):
    '''
    x: [B, N, D]
    mean: optional precomputed mean to use for centering, 
    max_half_range: optional precomputed max_half_range to use for scaling,
    '''
    is_train = mean is None 
    if mean is None:
        mean = x.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
    x_centered = x - mean
    if max_half_range is None:
        max_half_range = x_centered.abs().max()  # [B, 1, 1]
    x_normalized = x_centered / max_half_range

    if save_filename_title is not None:
        fname,title = save_filename_title
        #plot log historgam of the original data and the normalized data for each dimension
        import matplotlib.pyplot as plt
        #2 col for before and after normalization, and D rows for each dimension
        fig, axs = plt.subplots( x.shape[2], 2, figsize=(8, 4 * x.shape[2]))
        if x.shape[2] == 1:
            axs = axs[None, :]
        for d in range(x.shape[2]):
            n,bins,patch = axs[d, 0].hist(x[:,:,d].cpu().numpy().flatten(), bins=50, log=True, color="tab:blue", alpha=0.75)
            axs[d, 0].set_title(f"Original data - dim {d}")
            n,bins,patch = axs[d, 1].hist(x_normalized[:,:,d].cpu().numpy().flatten(), bins=50, log=True, color="tab:orange", alpha=0.75)
            # print(f"Dimension {d}: n {n}, bins {bins}")
            axs[d, 1].set_title(f"{title} - dim {d} {'train' if is_train else 'eval'}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    return x_normalized, mean, max_half_range
def duplicate_batch(x,target_batch_size):
    # x: [B, ...]
    B= x.shape[0]
    repeat_factor = math.ceil(target_batch_size / B)
    
    x_repeated = x.repeat(repeat_factor, *[1]*(len(x.shape)-1))  # Repeat along batch dimension

    x_repeated = x_repeated[:target_batch_size]  # Repeat and trim to target batch size
    assert x_repeated.shape[0] == target_batch_size, f"Expected batch size {target_batch_size}, but got {x_repeated.shape[0]}"
    return x_repeated
def make_dataset(shape_name, n_train_scene, N, cond_mode, cond_method, device='cpu', data_file=None):
    total_sc = int(1.25 * n_train_scene)

    if cond_mode =='none':
        cond = torch.zeros((total_sc, 1), device=device).float()
    else:
        if cond_method == 'scene_id':
            cond = torch.arange(total_sc, device=device).float() / max(
                        total_sc - 1, 1
                    )  # [n_scene], normalized to [0,1]
            cond = (cond * 2 - 1) # normalize to [-1,1], [batch_size, cond_dim]

    n_eval_scene = total_sc - n_train_scene
    if shape_name == "various":
        assert cond_mode is None or cond_method == 'scene_id', f"cond_mode {cond_mode} is not compatible with shape_name 'various' since it relies on scene_id conditioning. Please set cond_mode to 'scene_id' or None."
        assert total_sc <= 8, f"n_scene is too large for 'various' shape_name. "
        x0sbn3 = make_various_pc(
            num_points=N, device=device, n_shapes=total_sc
        )  # [B,N, 3]
        x0sbn3 = x0sbn3.cpu()  # Move to CPU for preprocessing to save GPU memory

        doppler = x0sbn3[:,:,3:4]  # Use the 4th dimension as dummy doppler for loss calculation
        rcs = x0sbn3[:,:,4:5]  # Use the 5th dimension as dummy rcs for loss calculation
        x0sbn3 = x0sbn3[:,:,:3]  # Use only the first 3 dimensions as point cloud data
        if cond_method == 'wan' and cond_mode != 'none':
            raise ValueError(f"cond_method 'wan' is not compatible with shape_name 'various' since it relies on Wan's VAE latent which is not available for 'various' shapes. Please set cond_method to 'scene_id' or choose a different shape_name that supports 'wan' conditioning.")

        x0sbn3_eval = x0sbn3[n_train_scene : n_train_scene+n_eval_scene]  # Reserve the last part for evaluation
        x0sbn3 = x0sbn3[:n_train_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation
        doppler_eval = doppler[n_train_scene : n_train_scene+n_eval_scene]
        doppler = doppler[:n_train_scene]
        rcs_eval = rcs[n_train_scene : n_train_scene+n_eval_scene]
        rcs = rcs[:n_train_scene]
        cond_eval = cond[n_train_scene : n_train_scene+n_eval_scene]
        cond = cond[:n_train_scene]

    elif shape_name == "realman":
        x0sbn3, wan_cond, dataset,doppler,rcs = make_man_pc(
            num_points=N, n_scene=total_sc, device=device, data_file=data_file
        )  # [B,N, 3]
        x0sbn3, wan_cond, dataset,doppler,rcs = x0sbn3.cpu(), wan_cond.cpu(), dataset, doppler.cpu(), rcs.cpu()  # Move to CPU for preprocessing to save GPU memory
        x0sbn3_eval = x0sbn3[n_train_scene : n_train_scene+n_eval_scene]  # Reserve the last part for evaluation
        x0sbn3 = x0sbn3[:n_train_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation
        
        if cond_method == 'wan' and cond_mode != 'none':
            cond = wan_cond / wan_cond.abs().max()  # Normalize wan_cond to [-1,1] for conditioning
            # assert cond is not the same for each sampel
            rand_idx = torch.randperm(cond.shape[0])[0]
            assert not torch.allclose(cond[0], cond[rand_idx]), f"Condition values are the same across samples, which may cause the model to ignore the condition. Please check the cond_method and cond_mode settings. Randomly selected sample idx for checking condition uniqueness: {rand_idx}"

        cond_eval = cond[n_train_scene : n_train_scene+n_eval_scene] 
        cond = cond[:n_train_scene]


        doppler = torch.norm(doppler, p=2, dim=-1, keepdim=True)

        doppler_eval = doppler[n_train_scene : n_train_scene+n_eval_scene]
        doppler = doppler[:n_train_scene]
        rcs_eval = rcs[n_train_scene : n_train_scene+n_eval_scene]
        rcs = rcs[:n_train_scene]
    elif shape_name == "realman_dense":
        x0sbn3, wan_cond, dataset,doppler,rcs = make_man_pc(
            num_points=N,
            n_scene=total_sc,
            is_dense=True,
            device=device,
            data_file=data_file,
        )  # [B,N, 3]
        x0sbn3, wan_cond, dataset,doppler,rcs = x0sbn3.cpu(), wan_cond.cpu(), dataset, doppler.cpu(), rcs.cpu()  # Move to CPU for preprocessing to save GPU memory

        if cond_method == 'wan' and cond_mode != 'none':
            cond = wan_cond / wan_cond.abs().max()  # Normalize wan_cond to [-1,1] for conditioning
            rand_idx = torch.randperm(cond.shape[0])[0]
            assert not torch.allclose(cond[0], cond[rand_idx]), f"Condition values are the same across samples, which may cause the model to ignore the condition. Please check the cond_method and cond_mode settings. Randomly selected sample idx for checking condition uniqueness: {rand_idx}"

        cond_eval = cond[n_train_scene : n_train_scene+n_eval_scene] 
        cond = cond[:n_train_scene]

        x0sbn3_eval = x0sbn3[n_train_scene : n_train_scene+n_eval_scene]  # Reserve the last part for evaluation
        x0sbn3 = x0sbn3[:n_train_scene]  # Use only the first n_scene shapes for training, reserve the rest for evaluation

        doppler = torch.norm(doppler, p=2, dim=-1, keepdim=True)
        doppler_eval = doppler[n_train_scene :  n_train_scene+n_eval_scene]
        doppler = doppler[:n_train_scene]
        rcs_eval = rcs[n_train_scene : n_train_scene+n_eval_scene]
        rcs = rcs[:n_train_scene]
    else:
        raise ValueError(f"Unknown shape_name: {shape_name}")
    assert doppler.shape[-1] == 1, f"Expected doppler to have shape [B, N, 1], but got {doppler.shape}"
    #normlize
    
    
    x0sbn3_norm,meanv,max_half_range = normalize_data(x0sbn3,save_filename_title=['/home/palakons/point_diffusion/output/sample/x0sbn3_normalization.png', "x0sbn3"])
    x0sbn3_eval_norm,_,_ = normalize_data(x0sbn3_eval, mean=meanv, max_half_range=max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/x0sbn3_eval_normalization.png', "x0sbn3_eval"])

    doppler_norm, doppler_mean, doppler_max_half_range = normalize_data(doppler, save_filename_title=['/home/palakons/point_diffusion/output/sample/doppler_normalization.png', "doppler"])
    doppler_eval_norm, _, _ = normalize_data(doppler_eval, mean=doppler_mean, max_half_range=doppler_max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/doppler_eval_normalization.png', "doppler_eval"])
    rcs_norm, rcs_mean, rcs_max_half_range = normalize_data(rcs, save_filename_title=['/home/palakons/point_diffusion/output/sample/rcs_normalization.png', "rcs"])
    rcs_eval_norm, _, _ = normalize_data(rcs_eval, mean=rcs_mean, max_half_range=rcs_max_half_range, save_filename_title=['/home/palakons/point_diffusion/output/sample/rcs_eval_normalization.png', "rcs_eval"])

    

    return [(x0sbn3_norm, cond, doppler_norm, rcs_norm), (x0sbn3_eval_norm, cond_eval, doppler_eval_norm, rcs_eval_norm)]
def save_point_sample(path, pred, gt=None, condition=None, meta=None):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"pred": pred}

    if gt is not None:

        data["gt"] = gt

    if condition is not None:

        data["condition"] = condition

    if meta is not None:

        data["meta"] = np.array(str(meta))

    np.savez_compressed(path, **data)