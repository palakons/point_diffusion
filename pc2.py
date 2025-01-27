from math import ceil, sqrt
import pandas as pd
import argparse
import open3d as o3d
import glob
from datetime import datetime
from pathlib import Path
from torch import Tensor
from typing import Optional, Union
import inspect

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2, registry)
from pytorch3d.implicitron.tools.config import expand_args_fields
from omegaconf import DictConfig
from config.structured import CO3DConfig, DataloaderConfig, ProjectConfig

from dataset.exclude_sequence import EXCLUDE_SEQUENCE, LOW_QUALITY_SEQUENCE
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
from geomloss import SamplesLoss  # Install via pip install geomloss
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm, trange
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter 
from model.point_cloud_model import PointCloudModel
from model.projection_model import PointCloudProjectionModel
from pytorch3d.structures import Pointclouds
import os
from pytorch3d.implicitron.dataset.data_loader_map_provider import \
    SequenceDataLoaderMapProvider
from torch.utils.data import SequentialSampler
from pytorch3d.implicitron.dataset.dataset_map_provider import (    DatasetMap)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



# Sinusoidal time embeddings


def get_sinusoidal_time_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class PointCloudLoss(nn.Module):
    def __init__(self, npoints: int, emd_weight: float = 0.5):
        super().__init__()
        if not 0 <= emd_weight <= 1:
            raise ValueError("emd_weight must be between 0 and 1")
        self.npoints = npoints
        self.emd_weight = emd_weight
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y_true.size() == y_pred.size()
        assert y_true.size(2) == 3
        assert len(y_true.size()) == 3
        # y_true = y_true.view(-1, self.npoints, 3)
        # y_pred = y_pred.view(-1, self.npoints, 3)
        chamfer = chamfer_distance(y_true, y_pred)[0]
        emd = self.sinkhorn_loss(y_true, y_pred)
        return (1 - self.emd_weight) * chamfer + self.emd_weight * emd


# Training function
def train_one_epoch(dataloader, model, optimizer, scheduler, args, criterion, device):
    model.train()
    losses = []

    pcpm=PointCloudProjectionModel( image_size=224,
image_feature_model='vit_small_patch16_224_msn'  ,
use_local_colors=True,
use_local_features=True,
use_global_features=False,
use_mask=True,
use_distance_transform=True,
predict_shape=True,
predict_color=False,
color_channels=3,
colors_mean=.5,
colors_std=.5,
scale_factor=1,).to(device)

    
    # image_size: int,
    # image_feature_model: str,
    # use_local_colors: bool = True,
    # use_local_features: bool = True,
    # use_global_features: bool = False,
    # use_mask: bool = True,
    # use_distance_transform: bool = True,
    # predict_shape: bool = True,
    # predict_color: bool = False,
    # process_color: bool = False,
    # image_color_channels: int = 3,  # for the input image, not the points
    # color_channels: int = 3,  # for the points, not the input image
    # colors_mean: float = 0.5,
    # colors_std: float = 0.5,
    # scale_factor: float = 1.0,
    # # Rasterization settings
    # raster_point_radius: float = 0.0075,  # point size
    # raster_points_per_pixel: int = 1,  # a single point per pixel, for now
    # bin_size: int = 0,

    for batch in dataloader:
        batch = batch.to(device)
        pc=batch.sequence_point_cloud
        camera=batch.camera
        image_rgb=batch.image_rgb
        mask=batch.fg_probability

        x_0 = pcpm.point_cloud_to_tensor(pc, normalize=True, scale=True)
        # print("x_0 type",type(x_0)) #<class 'pytorch3d.implicitron.dataset.frame_data.FrameData'>
        # print("x_0 shape",x_0.shape) #x_0 shape torch.Size([8, 16384, 3])

        B, N, D = x_0.shape
        noise = torch.randn_like(x_0)
        # print("noise shape",noise.shape) #noise shape torch.Size([4, 16384, 3])

        # print("B", B, "N", N, "D", D) #B 8 N 16384 D 3 
        
        # exit()
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=device,dtype=torch.long)
        
        # print("x", x.shape, "timesteps", timesteps.shape)
        x_t = scheduler.add_noise(x_0, noise, timesteps) #noisy_x


        # print("noisy_x", noisy_x.shape, "timesteps", timesteps.shape)
        # Conditioning
        #print location of each variable, cuda or cpu
        # print("x_t", x_t.device, "camera", camera.device, "image_rgb", image_rgb.device, "mask", mask.device, "timesteps", timesteps.device) #x_t cuda:0 camera cuda:0 image_rgb cuda:0 mask cuda:0 timesteps cuda:0 
        # print("train")
        # print("x_t.shape", x_t.shape)
        # print("image_rgb.shape", image_rgb.shape) 
        # print("mask.shape", mask.shape)
        # print('camera', len(camera))

        # x_t.shape torch.Size([4, 16384, 3])
        # image_rgb.shape torch.Size([4, 3, 224, 224])
        # mask.shape torch.Size([4, 1, 224, 224])

        x_t_input = pcpm.get_input_with_conditioning(x_t, camera=camera, 
            image_rgb=image_rgb, mask=mask, t=timesteps)
        # print("x_t_input", x_t_input.shape) #x_t_input torch.Size([8, 16384, 392])
        optimizer.zero_grad()
        # print("x_t_input", x_t_input.shape) #x_t_input torch.Size([8, 16384, 392])   
        noise_pred = model(x_t_input, timesteps)
        # print("output", output.shape)  # output torch.Size([1, 2, 3])

        if not noise_pred.shape == noise.shape:
            # raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
            raise ValueError(f'{noise_pred.shape} and {noise.shape} not equal')
        # print("noise_pred", noise_pred.shape, "noise", noise.shape) #noise_pred torch.Size([8, 16384, 3]) noise torch.Size([8, 16384, 3])
        # noise = noise[:, :, :3]
        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def plot_multi_gt(gts,  args,input_pc_file_list,  fname ):
    if fname is None:
        dir_name = f"logs/outputs/" +   args.run_name.replace("/", "_")
        #mkdir
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fname = f"{dir_name}/gts_M{args.M}.png"
    gts = gts.cpu().numpy()
    min_gt = gts.min(axis=0).min(axis=0)
    max_gt = gts.max(axis=0).max(axis=0)
    mean_gt = gts.mean(axis=0).mean(axis=0)
    range_gt = max_gt-min_gt

    # print("shpae", gts.shape)
    # print("min_gt", min_gt.shape, "max_gt", max_gt.shape, "mean_gt", mean_gt.shape, "range_gt", range_gt.shape)
    # exit()

    row_width = int(ceil(sqrt(len(gts))))
    n_row = int(ceil(len(gts) / row_width ))
    fig = plt.figure(figsize=(10*row_width, 10*n_row))
    # print("n_row", n_row, "row_width", row_width)
    if input_pc_file_list is None:
        input_pc_file_list = [f"gt_{i}" for i in range(len(gts))]
    for i, (gt,input_fname) in enumerate(zip(gts,input_pc_file_list)):   
        ax = fig.add_subplot(n_row, row_width, i+1, projection='3d')
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='g', marker='o', label='gt')
        ax.set_title(f"gt {i}: {input_fname}")
        factor_window = .5
        ax.set_xlim(mean_gt[0]-factor_window*range_gt[0],
                    mean_gt[0]+factor_window*range_gt[0])
        ax.set_ylim(mean_gt[1]-factor_window*range_gt[1],
                    mean_gt[1]+factor_window*range_gt[1])
        ax.set_zlim(mean_gt[2]-factor_window*range_gt[2],
                    mean_gt[2]+factor_window*range_gt[2])
    plt.tight_layout()
    plt.savefig(fname)
    print(f"saved at {fname}")



def plot_sample_multi_gt(gts, xts, x0s, steps, args, epoch, fname,input_pc_file_list):
    '''
    gts: ground truth point cloud, multi M (not unnormalized)
    xts: list of xt (not unnormalized)
    x0s: list of x0 pridcted from each step (not unnormalized)
    '''
    if input_pc_file_list is None:
        input_pc_file_list = [f"gt_{i}" for i in range(len(gts))]

    plot_multi_gt(gts,  args, input_pc_file_list, fname=None)
    if fname is None:
        dir_name = f"logs/outputs/" +   args.run_name.replace("/", "_")
        #mkdir
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)        
        fname = f"{dir_name}/sample_ep{epoch:09d}_M{args.M}.png"

    gts = gts.cpu().numpy()
    # find minimum from axes 0 and 1
    min_gt = gts.min(axis=0).min(axis=0)
    max_gt = gts.max(axis=0).max(axis=0)
    mean_gt = gts.mean(axis=0).mean(axis=0)
    range_gt = max_gt-min_gt

    min_all = min_gt.copy()
    max_all = max_gt.copy()

    fig = plt.figure(figsize=(30, 10))
    for i, (xt, x0, step) in enumerate(zip(xts, x0s, steps)):
        xt = xt[0].cpu().numpy()
        x0 = x0[0].cpu().numpy()
        step = step.cpu().numpy()

        ax = fig.add_subplot(2, len(xts), i+1, projection='3d')

        ax.scatter(xt[:, 0], xt[:, 1], xt[:, 2], c='r', marker='o', label='xt')
        ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], c='b', marker='o', label='x0')
        # ax.scatter(gts[:, 0], gts[:, 1], gts[:, 2], c='g', marker='o', label='gt')
        ax.set_title(f"step {step}")
        ax.legend()
        factor_window = 1.5
        ax.set_xlim(mean_gt[0]-factor_window*range_gt[0],
                    mean_gt[0]+factor_window*range_gt[0])
        ax.set_ylim(mean_gt[1]-factor_window*range_gt[1],
                    mean_gt[1]+factor_window*range_gt[1])
        ax.set_zlim(mean_gt[2]-factor_window*range_gt[2],
                    mean_gt[2]+factor_window*range_gt[2])
        # print(min_all,x0.min(axis=0),xt.min(axis=0))
        min_all = np.minimum(min_all, x0.min(axis=0), xt.min(axis=0))
        max_all = np.maximum(max_all, x0.max(axis=0), xt.max(axis=0))

    mean_all = (min_all+max_all)/2
    range_all = np.maximum(abs(min_all-mean_all), abs(max_all-mean_all))
    min_min = mean_all-range_all
    max_max = mean_all+range_all

    for i, (xt, x0, step) in enumerate(zip(xts, x0s, steps)):
        xt = xt[0].cpu().numpy()
        x0 = x0[0].cpu().numpy()
        step = step.cpu().numpy()
        ax2 = fig.add_subplot(2, len(xts), i+1+len(xts), projection='3d')
        ax2.scatter(xt[:, 0], xt[:, 1], xt[:, 2],
                    c='r', marker='o', label='xt')
        ax2.scatter(x0[:, 0], x0[:, 1], x0[:, 2],
                    c='b', marker='o', label='x0')
        # ax2.scatter(gts[:, 0], gts[:, 1], gts[:, 2],
        #             c='g', marker='o', label='gt')
        ax2.set_title(f"step {step}")
        ax2.legend()
        ax2.set_xlim(min_min[0], max_max[0])
        ax2.set_ylim(min_min[1], max_max[1])
        ax2.set_zlim(min_min[2], max_max[2])

    # plt.title(f"Evolution of point cloud at epoch {epoch}")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close() 

    
    #plot the last step, xts[-1], which would equal to x0s[-1]
    xt = xts[-1].cpu().numpy()[0]
    x0 = x0s[-1].cpu().numpy()[0]

    #find gt with least chamfer distance to xt[-1]
    min_cd = 1e10
    min_gt = None
    min_gt_idx = None
    for i,gt in enumerate(gts):
        gtt =torch.tensor(gt).unsqueeze(0)
        xtt = torch.tensor(xt).unsqueeze(0)
        # print("gtt",gtt.shape, "xtt",xtt.shape)
        cd, _ = chamfer_distance(gtt, xtt)
        if cd<min_cd:
            min_cd = cd
            min_gt = gt
            min_gt_idx = i
            
    fname_min_gt = input_pc_file_list[min_gt_idx].split("/")[-1]

    step = steps[-1].cpu().numpy()

    # print("shapes", xt.shape, x0.shape, min_gt.shape) #shapes (1, 100, 3) (1, 100, 3) (100, 3)


    #---plot combined
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xt[:, 0], xt[:, 1], xt[:, 2], c='r', marker='o', label=f'xt step {step}')
    #plot min_gt
    ax.scatter(min_gt[:, 0], min_gt[:, 1], min_gt[:, 2], c='g', marker='o', label=f'gt_{min_gt_idx}: {fname_min_gt}')
    ax.set_title(f"xt: step {step}")
    ax.legend()
    factor_window = 1.
    ax.set_xlim(mean_gt[0]-factor_window*range_gt[0],
                mean_gt[0]+factor_window*range_gt[0])
    ax.set_ylim(mean_gt[1]-factor_window*range_gt[1],
                mean_gt[1]+factor_window*range_gt[1])
    ax.set_zlim(mean_gt[2]-factor_window*range_gt[2],
                mean_gt[2]+factor_window*range_gt[2])
    plt.tight_layout()
    plt.savefig(fname.replace(".png","_last_step.png"))
    plt.close()

    #---plot each
    for label, pcs in zip(["xt", "x0", "gt"], [xt, x0, min_gt]):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c='g' if label == "gt" else 'r', marker=',')
        ax.set_title(f"{label} step {step}"+( f" {fname_min_gt}" if label=="gt" else ""))
        ax.set_xlim(mean_gt[0]-factor_window*range_gt[0],
                    mean_gt[0]+factor_window*range_gt[0])
        ax.set_ylim(mean_gt[1]-factor_window*range_gt[1],
                    mean_gt[1]+factor_window*range_gt[1])
        ax.set_zlim(mean_gt[2]-factor_window*range_gt[2],
                    mean_gt[2]+factor_window*range_gt[2])
        plt.tight_layout()  
        plt.savefig(fname.replace(".png",f"_{label}.png"))
        plt.close()

    #---plot distance by color
    color_map_name = "gist_rainbow" #https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # plot gt, with color based on distance to xt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    diff=dist=matrix = min_gt[None,:,:] - xt[:,None,:] # (100, 100, 3)
    print("diff", diff.shape)
    #norm2 distance
    dist = np.linalg.norm(diff, axis=-1)
    print("dist", dist.shape)
    #least distance for each gt
    min_dist = np.min(dist, axis=0)
    print("min_dist", min_dist.shape)
    plot = ax.scatter(min_gt[:, 0], min_gt[:, 1], min_gt[:, 2], c=min_dist, cmap=color_map_name, marker='o')
    fig.colorbar(plot, ax=ax)
    ax.set_title(f"gt_{min_gt_idx}: {fname_min_gt}")
    ax.set_xlim(mean_gt[0]-factor_window*range_gt[0],
                mean_gt[0]+factor_window*range_gt[0])
    ax.set_ylim(mean_gt[1]-factor_window*range_gt[1],

                mean_gt[1]+factor_window*range_gt[1])
    ax.set_zlim(mean_gt[2]-factor_window*range_gt[2],
                mean_gt[2]+factor_window*range_gt[2])
    plt.tight_layout()
    plt.savefig(fname.replace(".png","_gt_color_dist.png"))
    plt.close()
        




def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    args,
    device="cpu",
    start_epoch=0,
    criterion=nn.MSELoss(), writer=None
):
    pcpm=PointCloudProjectionModel( image_size=224,
        image_feature_model='vit_small_patch16_224_msn'  ,
        use_local_colors=True,
        use_local_features=True,
        use_global_features=False,
        use_mask=True,
        use_distance_transform=True,
        predict_shape=True,
        # predict_color=False,
        color_channels=3,
        colors_mean=.5,
        colors_std=.5,
        scale_factor=1,).to(device)
    model.train()
    tqdm_range = trange(start_epoch, args.epochs, desc="Epoch")
    checkpoint_fname = f"/data/palakons/checkpoint/cp_dm_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
    for epoch in tqdm_range:
        losses = train_one_epoch(
            dataloader, model, optimizer, scheduler,  args, criterion, device
        )
        tqdm_range.set_description(f"loss: {sum(losses)/len(losses):.4f}")
        if not args.no_tensorboard:
            writer.add_scalar("Loss/epoch", sum(losses) / len(losses), epoch)

        if not args.no_checkpoint and epoch % args.checkpoint_freq == 0:
            temp_epochs = args.epochs
            args.epochs = epoch
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            # save at /data/palakons/checkpoint/cp_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth
            torch.save(checkpoint, checkpoint_fname)
            args.epochs = temp_epochs
            print("checkpoint saved at", checkpoint_fname)
        if not args.no_tensorboard and (epoch+1) % args.visualize_freq == 0:

            batch = next(iter(dataloader))
            batch = batch.to(device)
            pc=batch.sequence_point_cloud
            camera=batch.camera
            image_rgb=batch.image_rgb
            mask=batch.fg_probability

            # print("image_rgb",image_rgb.shape)
            # print("mask",mask.shape)
            # image_rgb torch.Size([4, 3, 224, 224])
            # mask torch.Size([4, 1, 224, 224])

            sampled_point, xts, x0s, steps = sample(
                model,
                scheduler, args,camera=camera[0], image_rgb=image_rgb[:1], mask=mask[:1],
                num_inference_steps=50,
                evolution_freq=args.evolution_freq,
                device=device,
            )

            ## plot_sample_conditioned_gt()
            # data_mean = dataloader.dataset.mean.to(device)
            # data_factor = dataloader.dataset.factor.to(device)
            # all samples
            # gt_pcs = dataloader.dataset.data.to(device)

            gt_pcs = [pcpm.point_cloud_to_tensor(batch.sequence_point_cloud, normalize=True, scale=True) for batch in dataloader.dataset]
            # for i,gt_pc in enumerate(gt_pcs):
            #     print("gt_pc",i,gt_pc.shape)
            #keep only the sample with 16384 points
            gt_pcs = [gt_pc for gt_pc in gt_pcs if gt_pc.shape[1]==16384]
            #stack all gt_pcs 
            gt_pcs = torch.cat(gt_pcs, dim=0).to(device)

            # print("gt_pcs",gt_pcs.shape) #gt_pcs torch.Size([141, 16384, 3])
            # exit()

            plot_sample_multi_gt(gt_pcs, xts, x0s, steps,
                                 args, epoch,None,None)
            # gt_pcs = gt_pcs * data_factor + data_mean
            # sampled_point = sampled_point * data_factor + data_mean

            # cd_loss, _ = chamfer_distance(gt_pcs, sampled_point)
            cd_losses = []
            for i,gt_pc  in enumerate(gt_pcs):
                # print("gt_pc",i,gt_pc.shape)
                # print("sampled_point",sampled_point.shape)
                # gt_pc 0 torch.Size([16384, 3])
                # sampled_point torch.Size([4, 16384, 3])
                cd_loss_i, _ = chamfer_distance(
                    gt_pc.unsqueeze(0), sampled_point)
                cd_losses.append(cd_loss_i.item())

            writer.add_scalar("CD_min/epoch", min(cd_losses), epoch)
            writer.add_scalar("CD_max/epoch", max(cd_losses), epoch)
            writer.add_scalar(
                "CD_mean/epoch", sum(cd_losses) / len(cd_losses), epoch)

            # sampled_point = sampled_point.cpu().numpy()
            # gt_pcs = gt_pcs.cpu().numpy()

            # log_sample_to_tb(
            #     sampled_point[0, :, :], gt_pcs[0,
            #                                   :, :], f"for_visual", 50, epoch, writer
            # )  # support M=1 only
    if not args.no_checkpoint or True:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args,
        }
        # save at /data/palakons/checkpoint/cp_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth
        torch.save(checkpoint, checkpoint_fname)
        print("checkpoint saved at", checkpoint_fname)

    # print(f"Epoch {epoch+1} completed")


# Sampling function
@ torch.no_grad()
def sample(
    model,
    scheduler, args,camera=None, image_rgb=None, mask=None,color_channels=None,predict_color=False,
    num_inference_steps=None,
    evolution_freq=None,
    device="cpu",
):
    assert camera is not None and image_rgb is not None and mask is not None, "camera, image_rgb, mask must be provided"
    model.eval()
    num_inference_steps = num_inference_steps or scheduler.config.num_train_timesteps
    scheduler.set_timesteps(num_inference_steps)

    N = args.N
    B = 1 if image_rgb is None else image_rgb.shape[0]
    D = 3 + (color_channels if predict_color else 0)

    x_t = torch.randn(B, N, D, device=device)

    # print(" torch.randn x_t", x_t.shape) # torch.Size([1, 16384, 3])
    # print("image size", image_rgb.shape)
    # torch.randn x_t torch.Size([1, 16384, 3])
    # image size torch.Size([1, 3, 224, 224])


    accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {"offset": 1} if accepts_offset else {}
    scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    pcpm=PointCloudProjectionModel( image_size=224,
        image_feature_model='vit_small_patch16_224_msn'  ,
        use_local_colors=True,
        use_local_features=True,
        use_global_features=False,
        use_mask=True,
        use_distance_transform=True,
        predict_shape=True,
        # predict_color=False,
        color_channels=3,
        colors_mean=.5,
        colors_std=.5,
        scale_factor=1,).to(device)
    # extra_step_kwargs = {"eta": eta} if accepts_eta else {}
    extra_step_kwargs ={}
    xs = []
    x0t = []
    steps = []
    for i, t in enumerate(tqdm(scheduler.timesteps.to(device), desc="Sampling", leave=False)):
        # print("sampling i",i,"t",t)
        # print("x_t.shape", x_t.shape)
        # print("image_rgb.shape", image_rgb.shape) 
        # print("mask.shape", mask.shape)  
        # print("type camera",type(camera)) #type camera <class 'pytorch3d.renderer.cameras.PerspectiveCameras'>
        
        # x_t.shape torch.Size([1, 16384, 3])
        # image_rgb.shape torch.Size([1, 3, 224, 224])
        # mask.shape torch.Size([1, 1, 224, 224])

        # Conditioning
        x_t_input = pcpm.get_input_with_conditioning(x_t, camera=camera,
            image_rgb=image_rgb, mask=mask, t=torch.tensor([t]))
            
        # timesteps = torch.full(
        #     (x.size(0),), t, device=device, dtype=torch.long)
    
        noise_pred = model(x_t_input, t.reshape(1).expand(B))
        # print("output", output.shape)  # output torch.Size([1, 2, 3])
        # expanding outpu dim 2 (last dim) from 3 to 4 with zero (one mroe channel)

        x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs)


        # Convert output back into a point cloud, undoing normalization and scaling
        output_prev = pcpm.tensor_to_point_cloud(x_t.prev_sample, denormalize=True, unscale=True).points_padded().to(device)
        output_original_sample = pcpm.tensor_to_point_cloud(x_t.pred_original_sample, denormalize=True, unscale=True).points_padded().to(device)
        # print("type output_prev",type(output_prev)) #<class 'pytorch3d.structures.pointclouds.Pointclouds'>
        # print("type output_original_sample",type(output_original_sample)) #output_original_sample <class 'pytorch3d.structures.pointclouds.Pointclouds'>
        # print("output_prev", output_prev.shape, "output_original_sample", output_original_sample.shape) #output_prev torch.Size([4, 16384, 3]) output_original_sample torch.Size([4, 16384, 3])

        x_t = x_t.prev_sample

        if (
            evolution_freq is not None and i % evolution_freq == 0
        ) or i == num_inference_steps - 1:

            xs.append(output_prev)
            steps.append(t)
            x0t.append(output_original_sample)
    if num_inference_steps == 1:
        xs.append(output_prev)
        steps.append(0)
        x0t.append(output_original_sample)
    return output_prev, xs, x0t, steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for point clouds"
    )
    parser.add_argument(
        "--epochs", type=int, default=20*20, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int,
                        default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--N", type=int, default=16384, help="Number of points in each point cloud"
    )
    parser.add_argument( 
        "--M", type=int, default=1, help="Number of point cloud scenes to load"
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of training time steps",
    )
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="Beta schedule"
    )
    parser.add_argument("--no_tensorboard",  # action="store_true",
                        default=False,
                        help="Disable tensorboard logging")
    parser.add_argument(
        "--visualize_freq", type=int, default=8, help="Visualize frequency"
    )
    parser.add_argument(
        "--n_hidden_layers", type=int, default=1, help="Number of hidden layers"
    )
    parser.add_argument("--hidden_dim", type=int,
                        default=64, help="Hidden dimension")
    parser.add_argument("--loss_type", type=str,
                        default="mse", help="Loss function")
    parser.add_argument("--model", type=str,
                        default="pc2", help="Model type")
    parser.add_argument(
        "--checkpoint_freq", type=int, default=100000, help="Checkpoint frequency"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_checkpoint", default=True,
                        # action="store_true",
                        help="No checkpoint")
    parser.add_argument(
        "--evolution_freq", type=int, default=10, help="Evolution frequency"
    )
    # parser.add_argument(
    #     "--extra_channels", type=int, default=0, help="Extra channels in PVCNN >=0"
    # )
    parser.add_argument(
        "--point_path", type=str, default=None, help="Path to point cloud"  # either txt or ply
    )
    parser.add_argument("--tb_log_dir", type=str, default="./logs",
                        help="Path to store tensorboard logs")
    parser.add_argument("--run_name", type=str, default="first-27-jan", help="Run name")
    # normilzation method, std or min-max
    parser.add_argument("--norm_method", type=str,
                        default="std", help="Normalization method")
    
    # use_local_colors: bool = True
    # use_local_features: bool = True
    # use_mask: bool = True
    
    

    args = parser.parse_args()

    # if args.no_tensorboard is a string
    if type(args.no_tensorboard) == str:
        if args.no_tensorboard.lower() not in ["true", "false"]:
            print("no_tensorboard must be either True or False")

        args.no_tensorboard = (args.no_tensorboard.lower() == "true")
    # smae for no_checkpoint
    if type(args.no_checkpoint) == str:
        if args.no_checkpoint.lower() not in ["true", "false"]:
            print("no_checkpoint must be either True or False")

        args.no_checkpoint = (args.no_checkpoint.lower() == "true")
    return args


def match_args(args1, args2):
    # if number of args1 != number of args2:
    # make copy of args1, args2
    arga = vars(args1).copy()
    argb = vars(args2).copy()
    # remove checkpoint_freq,"epochs", "no_tensorboard" from both, if exist
    for key in ["checkpoint_freq", "epochs", "no_tensorboard", "no_checkpoint", "tb_log_dir", "run_name", "visualize_freq"]:
        if key in arga:
            del arga[key]
        if key in argb:
            del argb[key]

    # check if they are the same
    if len((arga)) != len((argb)):
        # print("number of args not the same")
        return False
    # check one by one if they are the same, except for epochs
    for key, value in arga.items():
        if value != (argb)[key]:
            # print(f"{key} not the same", value, "vs", (argb)[key])
            print(".", end="")
            return False
    print()
    return True

    # if checkpoint["args"].num_train_timesteps == args.num_train_timesteps and checkpoint["args"].beta_schedule == args.beta_schedule and checkpoint["args"].N == args.N and checkpoint["args"].M == args.M and checkpoint["args"].lr == args.lr and checkpoint["args"].batch_size == args.batch_size

    # for key, value in vars(args1).items():
    #     if value != vars(args2)[key]:
    #         return False
    # return True

# PVCNN-Based
class PVCNNDiffusionModel3D(nn.Module):
    def __init__(
        # pvcnnplusplus, pvcnn, simple
        self, data_dim, point_cloud_model_embed_dim=64, point_cloud_model="pvcnn",
            dropout=0.1,
            width_multiplier=1,
            voxel_resolution_multiplier=1,
    ):
        super().__init__()

        self.in_channels = data_dim  # 3
        self.out_channels = 3
        self.scale_factor = 1.0
        self.dropout = dropout
        self.width_multiplier = width_multiplier
        self.voxel_resolution_multiplier = voxel_resolution_multiplier

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dropout=self.dropout,
            width_multiplier=self.width_multiplier,
            voxel_resolution_multiplier=self.voxel_resolution_multiplier,
        )



    def forward(self, x, t):
        # (B, N, 3) (B,) #x torch.Size([1, 100, 3]) t torch.Size([1])
        noise_pred = self.point_cloud_model(x, t)

        return noise_pred

# # PVCNN-Based
# class ConditionalPVCNNDiffusionModel3D(nn.Module):
#     def __init__(
#         # pvcnnplusplus, pvcnn, simple
#         self, data_dim, point_cloud_model_embed_dim=64, point_cloud_model="pvcnn",
#             dropout=0.1,
#             width_multiplier=1,
#             voxel_resolution_multiplier=1,
#     ):
#         super().__init__()

#         self.in_channels = data_dim  # 3
#         self.out_channels = 3
#         self.scale_factor = 1.0
#         self.dropout = dropout
#         self.width_multiplier = width_multiplier
#         self.voxel_resolution_multiplier = voxel_resolution_multiplier

#         # Create point cloud model for processing point cloud at each diffusion step
#         self.point_cloud_model = PointCloudModel(
#             model_type=point_cloud_model,
#             embed_dim=point_cloud_model_embed_dim,
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             dropout=self.dropout,
#             width_multiplier=self.width_multiplier,
#             voxel_resolution_multiplier=self.voxel_resolution_multiplier,
#         )

#     def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
#         """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
#         points = pc * (self.scale_factor if scale else 1)
#         return points

#     def forward(self, x, t):
#         # (B, N, 3) (B,) #x torch.Size([1, 100, 3]) t torch.Size([1])
#         noise_pred = self.point_cloud_model(x, t)

#         return noise_pred




# class ConditionalModel(PointCloudProjectionModel):
#     def __init__(
#         self,
#         point_cloud_model: str,
#         point_cloud_model_embed_dim: int,
#         **kwargs,  # projection arguments
#     ):
#         super().__init__(**kwargs)
#         self.point_cloud_model = PointCloudModel(
#             model_type=point_cloud_model,
#             embed_dim=point_cloud_model_embed_dim,
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#         )
#     def forward(self, batch, **kwargs):
#         pc=batch.sequence_point_cloud, 
#         camera=batch.camera,
#         image_rgb=batch.image_rgb,
#         mask=batch.fg_probability,
        
#         # Normalize colors and convert to tensor
#         x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
#         B, N, D = x_0.shape

#         # Sample random noise
#         noise = torch.randn_like(x_0)

#         # Sample random timesteps for each point_cloud
#         timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), 
#             device=self.device, dtype=torch.long)
#         # timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), 
#         #     device=self.device, dtype=torch.long)

#         # Add noise to points
#         x_t = self.scheduler.add_noise(x_0, noise, timestep)

#         # Conditioning
#         x_t_input = self.get_input_with_conditioning(x_t, camera=camera, 
#             image_rgb=image_rgb, mask=mask, t=timestep)

#         # Forward
#         noise_pred = self.point_cloud_model(x_t_input, timestep)
        
#         # Check
#         if not noise_pred.shape == noise.shape:
#             # raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
#             raise ValueError(f'{noise_pred.shape} and {noise.shape} not equal')
        
#         return noise_pred
 
def get_pc2dataset(cfg,M):
    dataset_cfg: CO3DConfig = cfg.dataset
    # category:'car'
    # max_points: int = 16_384
    # image_size: int = 224
    # mask_images: bool = '${model.use_mask}'
    # use_mask: bool = True
    # restrict_model_ids: Optional[List] = None
    # subset_name: str = '80-20' 
    # root: str = os.getenv('ASTYX_DATASET_ROOT')

    dataloader_cfg: DataloaderConfig = cfg.dataloader
    
    # Exclude bad and low-quality sequences
    exclude_sequence = []
    exclude_sequence.extend(EXCLUDE_SEQUENCE.get(dataset_cfg.category, []))
    exclude_sequence.extend(LOW_QUALITY_SEQUENCE.get(dataset_cfg.category, []))
    
    # Whether to load pointclouds
    kwargs = dict(
        remove_empty_masks=True,
        n_frames_per_sequence=1,
        load_point_clouds=True,
        max_points=dataset_cfg.max_points,
        image_height=dataset_cfg.image_size,
        image_width=dataset_cfg.image_size,
        mask_images=dataset_cfg.mask_images,
        exclude_sequence=exclude_sequence,
        pick_sequence=() if dataset_cfg.restrict_model_ids is None else dataset_cfg.restrict_model_ids,
    )

    # Get dataset mapper
    dataset_map_provider_type = registry.get(JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2")
    expand_args_fields(dataset_map_provider_type)
    dataset_map_provider = dataset_map_provider_type(
        category=dataset_cfg.category,
        subset_name=dataset_cfg.subset_name,
        dataset_root=dataset_cfg.root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig(kwargs),
    )

    # Get datasets
    datasets = dataset_map_provider.get_dataset_map()

    #print length of train, val, test
    print("len(train)", len(datasets["train"]))
    print("len(val)", len(datasets["val"]))
    print("len(test)", len(datasets["test"]))
    print("M", M)
    # len(train) 144
    # len(val) 46
    # len(test) 144
    # M 1

    #pick only M first item in train,val,test

    # PATCH BUG WITH POINT CLOUD LOCATIONS!
    for dataset in (datasets["train"], datasets["val"]):
        for key, ann in dataset.seq_annots.items():
            correct_point_cloud_path = Path(dataset.dataset_root) / Path(*Path(ann.point_cloud.path).parts[-3:])
            assert correct_point_cloud_path.is_file(), correct_point_cloud_path
            ann.point_cloud.path = str(correct_point_cloud_path)

    # Get dataloader mapper
    data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
    expand_args_fields(data_loader_map_provider_type)
    data_loader_map_provider = data_loader_map_provider_type(
        batch_size=dataloader_cfg.batch_size,
        num_workers=dataloader_cfg.num_workers,
    )

    # QUICK HACK: Patch the train dataset because it is not used but it throws an error
    if (len(datasets['train']) == 0 and len(datasets[dataset_cfg.eval_split]) > 0 and 
            dataset_cfg.restrict_model_ids is not None and cfg.run.job == 'sample'):
        datasets = DatasetMap(train=datasets[dataset_cfg.eval_split], val=datasets[dataset_cfg.eval_split], 
                                test=datasets[dataset_cfg.eval_split])
        print('Note: You used restrict_model_ids and there were no ids in the train set.')

    # Get dataloaders
    dataloaders = data_loader_map_provider.get_data_loader_map(datasets)
    dataloader_train = dataloaders['train']
    dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

    # Replace validation dataloader sampler with SequentialSampler
    dataloader_val.batch_sampler.sampler = SequentialSampler(dataloader_val.batch_sampler.sampler.data_source)

    # Modify for accelerate
    dataloader_train.batch_sampler.drop_last = True
    dataloader_val.batch_sampler.drop_last = False

    return dataloader_train, dataloader_val, dataloader_vis
def get_model(args, device="cpu"):

    if args.model == "mlp3d":
        raise ValueError("model mlp3d not supported, please use pc2")
        data_dim = args.N * 3

        return SimpleDiffusionModel3D(
            data_dim=data_dim,
            time_embedding_dim=16,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.n_hidden_layers,
        ).to(device)
    elif args.model == "pvcnn" or args.model == "pc2":
        # raise ValueError("model pvcnn not supported, please use pc2")
        pcpm =PointCloudProjectionModel ( image_size=224, image_feature_model='vit_small_patch16_224_msn'   )

        data_dim = pcpm.in_channels  
        print("data_dim", data_dim)
        return PVCNNDiffusionModel3D(data_dim=data_dim,          point_cloud_model_embed_dim=args.hidden_dim, point_cloud_model="pvcnn",      dropout=0.1,            width_multiplier=1,            voxel_resolution_multiplier=1,).to(device)
    else:
        raise ValueError("model not supported, choose either mlp3d, pvcnn, or pc2")


def get_checkpoint_fname(args, CHECKPOINT_DIR):
    # check checkpoint
    files = glob.glob(f"{CHECKPOINT_DIR}/cp_dm_*.pth")
    max_epoch = 0
    current_cp_fname = None
    for fname in files:
        try:
            checkpoint = torch.load(fname)
            if match_args(checkpoint["args"], args):
                if (
                    checkpoint["args"].epochs <= args.epochs
                    and checkpoint["args"].epochs > max_epoch
                ):
                    max_epoch = checkpoint["args"].epochs
                    current_cp_fname = fname
                    # print("current_cp", current_cp_fname)
        except:
            print("error", fname)
            continue
    return current_cp_fname


def get_loss(args):
    if args.loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    elif args.loss_type == "chamfer":
        return PointCloudLoss(npoints=args.N, emd_weight=0)
    elif args.loss_type == "emd":
        return PointCloudLoss(npoints=args.N, emd_weight=1)
    else:
        raise ValueError("loss not supported")


# def log_sample_to_tb(x, gt_pc, key, evo, epoch, writer):
#     sampled_tensor = torch.tensor(x, dtype=torch.float)
#     gt_pc_tensor = torch.tensor(gt_pc, dtype=torch.float)

#     all_tensor = torch.cat([sampled_tensor, gt_pc_tensor], dim=0)

#     color_sampled = torch.tensor(
#         [[255, 0, 0] for _ in range(sampled_tensor.shape[0])])  # color: red
#     color_gt = torch.tensor(
#         [[0, 255, 0] for _ in range(gt_pc_tensor.shape[0])])  # color: green

#     all_color = torch.cat([color_sampled, color_gt], dim=0)
#     # print("shape", all_tensor.shape, all_color.shape)
#     # add dimension to tensor to dim 0
#     all_tensor = all_tensor.unsqueeze(0)
#     all_color = all_color.unsqueeze(0)
#     writer.add_mesh(f"PointCloud_{key}_{evo}", vertices=all_tensor, colors=all_color,
#                     global_step=epoch)


import hydra
@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):

    args = parse_args()
    if args.model != "pvcnn":
        args.extra_channels = 0
        print("extra_channels set to 0 for non-pvcnn model")
    else:
        assert args.extra_channels >= 0, "extra_channels must be >=0 for pvcnn"
    print(args)
    set_seed(args.seed)


    if not args.no_tensorboard:
        log_dir = args.tb_log_dir + \
            f"/{args.run_name if  args.run_name else datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        writer = SummaryWriter(
            log_dir=log_dir)
        print("tensorboard log at", log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    
    pcpm=PointCloudProjectionModel( image_size=224,
        image_feature_model='vit_small_patch16_224_msn'  ,
        use_local_colors=True,
        use_local_features=True,
        use_global_features=False,
        use_mask=True,
        use_distance_transform=True,
        predict_shape=True,
        # predict_color=False,
        color_channels=3,
        colors_mean=.5,
        colors_std=.5,
        scale_factor=1,).to(device)
    # dataset = PointCloudDataset(args)
    # dataset = PC2Dataset(args)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if False: #test plot_multi_gt
        gt_pcs = dataloader.dataset.data.to(device)
        input_pc_file_list = dataloader.dataset.use_files_list
        plot_multi_gt(gt_pcs,  args, input_pc_file_list, fname=None)
        exit()

    dataloader_train, dataloader_val, dataloader_vis = get_pc2dataset(cfg,args.M)

    model = get_model(args, device=device)

    if not args.no_tensorboard:
        writer.add_scalar("model_params", sum(p.numel()
                          for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # linear, scaled_linear, or squaredcos_cap_v2.
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule,
        prediction_type="epsilon",  # or "sample" or "v_prediction"
        clip_sample=False
    )

    CHECKPOINT_DIR = "/data/palakons/checkpoint"
    start_epoch = 0

    if not args.no_checkpoint:
        current_cp_fname = get_checkpoint_fname(args, CHECKPOINT_DIR)
        if current_cp_fname is not None:
            current_cp = torch.load(current_cp_fname)
            print("loading checkpoint", current_cp_fname)
            model.load_state_dict(current_cp["model"])
            optimizer.load_state_dict(current_cp["optimizer"])
            start_epoch = current_cp["args"].epochs
            print("start_epoch", start_epoch)
            # if not args.no_tensorboard:
            #     raise ValueError("not implemented")

        else:
            print("no checkpoint found")

    criterion = get_loss(args)
    # Train the model
    train(
        model,
        # dataloader,
        dataloader_train,
        optimizer,
        scheduler,
        args,
        device=device,
        start_epoch=start_epoch,
        criterion=criterion, writer=writer
    )

    losses = train_one_epoch(
        dataloader_train, model, optimizer, scheduler, args, criterion=criterion, device=device
    )
    metric_dict = {"Loss": sum(losses) / len(losses)}
    if not args.no_tensorboard:
        writer.add_scalar("Loss/epoch", sum(losses) /
                          len(losses), args.epochs - 1)

    # samples, _ = sample(model, scheduler, sample_shape=( 1000, data_dim), device=device)

    # Sample from the model
    num_sample_points = 1

    batch = next(iter(dataloader_train))
    batch = batch.to(device)
    pc=batch.sequence_point_cloud
    camera=batch.camera
    image_rgb=batch.image_rgb
    mask=batch.fg_probability
    
    samples = {}
    for i in [1, 5, 10, 50, 100]:
        samples[f"step{i}"] = sample(
            model,
            scheduler, args,camera=camera[0], image_rgb=image_rgb[:1], mask=mask[:1],
            num_inference_steps=i,
            evolution_freq=args.evolution_freq,
            device=device,
        )

    if not args.no_tensorboard:
        # make the plot that will be logged to tb
        plt.figure(figsize=(10, 10))

        for key, value in samples.items():
            # print(samples[key][0].shape)
            # print("dataset.std",dataset.std.shape)
            samples_updated = samples[key][0] 
            # samples_updated = samples[key][0] *                 dataset.factor.to(device) + dataset.mean.to(device)
            # print("samples_updated", key, samples_updated.shape) #samples_updated step1 torch.Size([1, 1, 3])
            # #shape  dataset[:]
            # print("dataset[:]", dataset[:].shape) #dataset[:] torch.Size([2, 1, 3])
            
            gt_pcs = [pcpm.point_cloud_to_tensor(batch.sequence_point_cloud, normalize=True, scale=True) for batch in dataloader_train.dataset]
            gt_pcs = [gt_pc for gt_pc in gt_pcs if gt_pc.shape[1]==16384]
            gt_pcs = torch.cat(gt_pcs, dim=0).to(device)
            cd_losses = []
            for i,gt_pc  in enumerate(gt_pcs):
                # print("gt_pc",i,gt_pc.shape)
                # print("samples_updated",samples_updated.shape)
                # gt_pc 0 torch.Size([1, 16384, 3])
                # samples_updated torch.Size([1, 16384, 3])
                loss, _ = chamfer_distance(
                    gt_pc.unsqueeze(0).to(device) ,
                    samples_updated.to(device),
                )
                cd_losses.append(loss.item())

            # log minumum loss

            print(
                key, "\t", f"CD: { min(cd_losses):.2f} at {cd_losses.index(min(cd_losses))}")
            if not args.no_tensorboard:
                writer.add_scalar(f"CD_{key}", min(cd_losses), args.epochs)

                # add cd to metric_dict
                metric_dict = {**{f"CD_{key}": min(cd_losses)}, **metric_dict}

            error = []
            assert len(samples[key][1]) > 1, "need more than 1 sample to plot"
            for x in samples[key][1]:
                # x = x * dataset.factor.to(device) + dataset.mean.to(device)

                cd_losses = []
                for i,gt_pc  in enumerate(gt_pcs):
                    loss, _ = chamfer_distance(
                        gt_pc.unsqueeze(0).to(device) ,
                        x.to(device),
                    )
                    # print(f"{loss.item():.2f}", end=", ")
                    cd_losses.append(loss.item())
                error.append(min(cd_losses))
            # ax = plt.plot(error, label=key)
            plt.plot(
                [i / (len(error) - 1) for i in range(len(error))],
                error,
                label=key,
                marker=None,
            )
            # print()
        plt.legend()
        plt.title(
            f"Diffusion model: {args.epochs} epochs, {args.num_train_timesteps} timesteps, {args.beta_schedule} schedule",
        )
        plt.xlabel("Evolution steps ratio")
        plt.ylabel("Chamfer distance")
        # ylog
        plt.yscale("log")

        writer.add_figure(f"Evolution", plt.gcf(), args.epochs)
        plt.close()

    print("done evo plots")
    # plot evolution

    # key_to_plot = "step50"

    # # get GT  from dataloader
    # gt_pc = next(iter(dataloader)).to(device)  # one sample
    # # gt_pc = gt_pc * dataset.factor.to(device) + dataset.mean.to(device)
    # # print("gt_pc", gt_pc.shape) #gt_pc torch.Size([1, N*3])
    # gt_pc = gt_pc.reshape(-1, 3)
    # gt_pc = gt_pc.cpu().numpy()

    # key = key_to_plot
    # value = samples[key]

    # temp = torch.stack(value[1], dim=0)
    # # print("temp", key, temp.shape)
    # samples_updated = temp.to(
    #     device) * dataset.factor.to(device) + dataset.mean.to(device)
    # # print("samples_updated", key, samples_updated.shape)
    # # print("samples_updated.reshape(-1,3).mean(dim=0)", samples_updated.reshape(-1,3).mean(dim=0))
    # # mean_coord_val = samples_updated.reshape(-1, 3).mean(dim=0).cpu()
    # # factor_coord_val = samples_updated.reshape(-1, 3).factor(dim=0).cpu()
    # # move to cpu
    # # min_coord_val = min_coord_val.cpu()
    # # max_coord_val = max_coord_val.cpu()
    # # print(key, "\tmin max, ", mean_coord_val, std_coord_val)
    # samples_updated = samples_updated.cpu().numpy()

    # if not args.no_tensorboard:
    #     for i, x in tqdm(enumerate(samples_updated)):
    #         # print("x", x.shape)
    #         # x_shape = x.reshape(x.shape[0], -1, 3)
    #         log_sample_to_tb(
    #             x[0, :, :],
    #             gt_pc,
    #             key,
    #             i * args.evolution_freq -
    #             (1 if i == len(samples_updated) - 1 else 0),
    #             args.epochs, writer
    #         )

    if not args.no_tensorboard:
        hparam_dict = vars(args)
        writer.add_hparams(hparam_dict, metric_dict)
        writer.close()


if __name__ == "__main__":
    main()
