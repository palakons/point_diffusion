"""
Simple DDPM for Radar Point Cloud Generation
Conditional on RGB image's WAN VAE Latent[B, 16, 2, 60, 104]], outputs pixel-depth occupancy grid [B, W, H, D]
"""

import sys

sys.path.insert(0, "/home/palakons/DiT")
from models import DiT

import pandas as pd
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
from dit_ddpm_class import parse_args, set_seed, TransformerDenoiser, get_runid
from torch.utils.tensorboard import SummaryWriter
from adaln0_import import (
    save_checkpoint,
    load_checkpoint,
    train_eval_batch,
    makeDataset,
    splitDataset,
    makeDataloaders,
    makeOptimizer,
    tblogHparam,
    makeDiTModel,
)
import seaborn as sns


# print(f"Batch keys: {batch.keys()}")
# dict_keys(['depth_image', 'filtered_radar_data', 'uvz', 'camera_front', 'frame_token', 'npoints_original', 'npoints_filtered', 'clip_feature', 'scene_id', 'frame_index', 'occupancy_grid',"wan_vae_latent"])


# python /home/palakons/from_scratch/man_stat.py


sns.set_theme(style="whitegrid")


def plot_per_frame(per_frame: list[dict], plot_dir: str, n_bins=16):
    df = pd.DataFrame(per_frame)
    if df.empty:
        print("per_frame is empty; nothing to plot.")
        return

    # split
    df["num_middle_points"] = pd.to_numeric(df["num_middle_points"], errors="coerce")
    df = df.dropna(subset=["num_middle_points", "data-file"])

    fig, axes = plt.subplots(
        nrows=len(df["data-file"].unique()),
        ncols=1,
        figsize=(10, 8),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    host_data = {}
    for i, data_file in enumerate(df["data-file"].unique()):
        sub = df[df["data-file"] == data_file]
        if sub.empty:
            print(f"No data for {data_file}; skipping.")
            continue

        # shared bin edges so x-axes truly align
        x_all = sub["num_middle_points"].to_numpy()
        bins = np.linspace(float(x_all.min()), float(x_all.max()), n_bins + 1)

        # top: man-mini
        sns.histplot(
            data=sub,
            x="num_middle_points",
            bins=bins,
            element="step",
            fill=False,
            stat="count",
            ax=axes[i],
            color="tab:blue",
        )
        host_data[data_file] = {}
        counts, bin_edges = np.histogram(x_all, bins=bins)
        host_data[data_file]["counts"] = counts
        host_data[data_file]["bin_edges"] = bin_edges

        axes[i].set_title(f"{data_file}: num_middle_points per frame")
        if i == len(df["data-file"].unique()) - 1:
            axes[i].set_xlabel("Number of Middle Square Points per Frame")
        axes[i].set_ylabel("Count")

    outpath = os.path.join(plot_dir, "num_middle_points_per_frame_split.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # save host_data
    hist_outpath = os.path.join(plot_dir, f"num_middle_points_per_frame_hist_data.npy")
    np.save(hist_outpath, host_data)


def plot_per_point(per_point: list[dict], plot_dir: str):
    """
    Creates 3 plots (u, v, z). Each plot overlays man-mini and man-full:
      x = position
      y = min_dist
    Also includes marginal histograms via seaborn.jointplot.
    """
    df = pd.DataFrame(per_point)
    if df.empty:
        print("per_point is empty; nothing to plot.")
        return

    # Ensure numeric
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["min_dist"] = pd.to_numeric(df["min_dist"], errors="coerce")
    df = df.dropna(subset=["position", "min_dist", "data-file", "axis", "scene_id"])

    axis_order = [
        "axis-u",
        "axis-v",
        "axis-z",
    ]
    hist_data = {}
    for ax_name in tqdm(axis_order):
        hist_data[ax_name] = {}
        sub = df[df["axis"] == ax_name]
        if sub.empty:
            print(f"No data for {ax_name}; skipping.")
            continue

        # Joint plot: scatter + marginal histograms
        g = sns.jointplot(
            data=sub,
            x="position",
            y="min_dist",
            hue="data-file",  # man-mini vs man-full
            kind="scatter",
            height=8,
            s=4,  # point size
            alpha=0.25,  # transparency
        )
        g.ax_marg_x.cla()
        sns.histplot(
            data=sub,
            x="position",
            hue="data-file",
            bins=200,
            common_norm=False,
            element="step",
            fill=False,
            ax=g.ax_marg_x,
            legend=False,
        )
        # get bins counts using np.histogram
        hist_data[ax_name]["position"] = {}
        for data_file in sub["data-file"].unique():
            sub_file = sub[sub["data-file"] == data_file]
            counts, bin_edges = np.histogram(sub_file["position"].to_numpy(), bins=200)
            hist_data[ax_name]["position"][data_file] = {
                "counts": counts,
                "bin_edges": bin_edges,
            }

        # Y marginal: use LOG-SPACED bins (this fixes the visual correctness on log axis)
        y = sub["min_dist"].to_numpy()
        ymin, ymax = float(y.min()) + 1e-8, float(y.max())
        # print("ymin, ymax:", ymin, ymax)
        n_log_bins = 300  # 200-600 is typical
        y_bins = np.logspace(np.log10(ymin), np.log10(ymax), n_log_bins)
        # print("y_bins:", y_bins)

        g.ax_marg_y.cla()
        sns.histplot(
            data=sub,
            y="min_dist",
            hue="data-file",
            bins=y_bins,  # <-- key change
            common_norm=False,
            element="step",
            fill=False,
            ax=g.ax_marg_y,
            legend=False,
        )
        hist_data[ax_name]["min_dist"] = {}
        for data_file in sub["data-file"].unique():
            sub_file = sub[sub["data-file"] == data_file]
            counts, bin_edges = np.histogram(
                sub_file["min_dist"].to_numpy(), bins=y_bins
            )
            hist_data[ax_name]["min_dist"][data_file] = {
                "counts": counts,
                "bin_edges": bin_edges,
            }

        g.ax_joint.set_yscale("log")
        g.ax_marg_y.set_yscale("log")

        g.fig.suptitle(f"{ax_name}: position vs min_dist", y=1.02)
        g.set_axis_labels("position", "min_dist")

        outpath = os.path.join(plot_dir, f"min_dist_joint_{ax_name}.png")
        g.fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
    # save hist_data
    hist_outpath = os.path.join(plot_dir, f"min_dist_hist_data.npy")
    np.save(hist_outpath, hist_data)


def min_dist(tensor_1d: torch.Tensor, infd_dict) -> torch.Tensor:
    """Compute the minimum distance, of each point it their nearest neighbor in the tensor itself."""
    if tensor_1d.numel() == 0:
        return []
    # print("tensor_1d:", tensor_1d.shape)
    sorted_tensor, _ = torch.sort(tensor_1d)
    # print("sorted_tensor:", sorted_tensor)
    diffs = sorted_tensor[1:] - sorted_tensor[:-1]
    # add the first diff
    # print("diffs:", diffs.shape)
    diffs = torch.cat((diffs[:1], diffs))
    # print("diffs:", diffs)
    # return zip element (tensor_1d[i], diffs[i])
    zipped = [
        {**{"position": p, "min_dist": d}, **infd_dict}
        for p, d in zip(sorted_tensor.tolist(), diffs.tolist())
    ]
    # print("zipped:", zipped)
    return zipped


def main():
    args = parse_args()
    config = vars(args)
    print(f"Setting random seed: {args.seed}")
    set_seed(config["seed"])
    is_perpoint = True

    plot_dir = "/data/palakons/man_vaevoxelmetadit/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Load dataset
    per_frame = []
    per_point = []
    output_dir = "/home/palakons/from_scratch/"
    for data_file in ["man-mini", "man-full"]:
        config["data_file"] = data_file
        config["scene_ids"] = []
        config["num_input_frames"] = 40000
        dataset = MANDataset(
            scene_ids=config["scene_ids"],
            data_file=config["data_file"],
            device=config["device"],
            radar_channel=config["radar_channel"],
            camera_channel=config["camera_channel"],
            double_flip_images=False,
            coord_only=False,
            visualize_uvz=False,  # plotting, slow
            scaled_image_size=config["scaled_image_size"],  # 512 1024 dead
            n_p=0,  # use all points -----
            point_only=True,  # False for dit training, no depth, camera, CLIP, no grid ----
            max_depth=config["max_voxel_grid_depth"],
            depth_bins=config["depth_voxel_grid_bins"],
            wan_vae=True,  # for vae-based training
            wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
            viz_dir=plot_dir,
            grid_binary_range=config["grid_binary_range"],  # "0-1" or "neg1-1"
            keep_frames=config["num_input_frames"],
        )
        print(f"{data_file} Total dataset size: {len(dataset)}")
        for data in tqdm(dataset):
            # print(
            #     f"Data keys: {data.keys()}"
            # )  # ['filtered_radar_data', 'uvz', 'frame_token', 'npoints_original', 'npoints_filtered', 'scene_id', 'frame_index']
            # print("Sample uvz shape:", data["uvz"].shape)  # ([541, 3])
            # print("uvz min:", data["uvz"].min(dim=0))
            # print("uvz max:", data["uvz"].max(dim=0))
            # print("npoints_original:", data["npoints_original"])
            # print("npoints_filtered:", data["npoints_filtered"])
            # Sample uvz shape: torch.Size([541, 3])
            # uvz min: torch.return_types.min(
            # values=tensor([  1.4373, 389.5913,   5.8943]),
            # indices=tensor([217, 179,   0]))
            # uvz max: torch.return_types.max(
            # values=tensor([1353.3177,  800.7029,  187.6039]),
            # indices=tensor([179,   4, 540]))
            # npoints_original: 727
            # npoints_filtered: 541

            pad_left = (
                config["original_image_size"][1] - config["original_image_size"][0]
            ) // 2
            pad_right = (
                config["original_image_size"][1]
                - pad_left
                - config["original_image_size"][0]
            )
            # filter middle square, wher  pad_left < u < padleft +original_image_size[0]
            filter_u = (data["uvz"][:, 0] >= pad_left) & (
                data["uvz"][:, 0] < pad_left + config["original_image_size"][0]
            )
            middle_uvz = data["uvz"][filter_u]
            # print(
            #     "Middle square uvz points:",
            #     middle_uvz.shape,
            #     "scne_id:",
            #     data["scene_id"],
            #     "frame_index:",
            #     data["frame_index"],
            # )
            per_frame.append(
                {
                    "data-file": config["data_file"],
                    "scene_id": data["scene_id"],
                    "frame_index": data["frame_index"],
                    "num_middle_points": middle_uvz.shape[0],
                }
            )
            if is_perpoint:
                axisname = ["u", "v", "z"]
                for axis in range(3):
                    zipped = min_dist(
                        middle_uvz[:, axis],
                        {
                            "data-file": config["data_file"],
                            "scene_id": data["scene_id"],
                            "frame_index": data["frame_index"],
                            "axis": f"axis-{axisname[axis]}",
                        },
                    )
                    per_point.extend(zipped)

            # Middle square uvz points: torch.Size([405, 3])
            # scne_id: 0
            # frame_index: 0
    if False:  # too big
        pd.DataFrame(per_frame).to_csv(
            os.path.join(output_dir, f"man_stat_frame.csv"),
            index=False,
        )  # 24303 rows
        pd.DataFrame(per_point).to_csv(
            os.path.join(output_dir, f"man_stat_point.csv"),
            index=False,
        )  # 26995261 rows

    if is_perpoint:
        plot_per_point(per_point, plot_dir)

    plot_per_frame(per_frame, plot_dir)


if __name__ == "__main__":
    main()
