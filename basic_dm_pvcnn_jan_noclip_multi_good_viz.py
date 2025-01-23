from math import ceil, sqrt
import pandas as pd
import argparse
import open3d as o3d
import glob
from datetime import datetime

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
from pytorch3d.structures import Pointclouds
import os


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PointCloudDataset(Dataset):  # normalized per axis, not per sample
    def __init__(self, args
                 ):
        super().__init__()
        num_points = args.N
        num_scene = args.M
        point_path = args.point_path
        norm_method = args.norm_method

        if point_path is not None:
            # assert if point_path is not a file, but a dir

            point_path = point_path.split(",")
            # get all files in point_path

            # list files in the dir
            files_list = []
            for each_path in point_path:

                assert os.path.isdir(
                    each_path), f"point_path {each_path} must be a directory"

                files_name = os.listdir(each_path)
                # if ends with txt or ply,
                for file_name in files_name:
                    if file_name.endswith(".txt") or file_name.endswith(".ply"):
                        files_list.append(os.path.join(each_path, file_name))

            assert len(files_list) >= num_scene

            # if num_scene > 1:
            #     raise ValueError("num_scene > 1 not supported, yet")

            # load point cloud from files_list
            # /data/palakons/dataset/astyx/scene/0/000000.txt
            # load point cloud from files_list, as a text file, separate by space, use pandas
            #
            # X Y Z V_r Mag
            # 0.108118820531769647 2.30565393293341492 -0.279097884893417358 0 48
            # 1.18417095921774496 2.25506417530992165 -0.122170589864253998 0 48

            # creat blank tensor dim (num_scene, num_points, 3)
            self.data = torch.zeros(num_scene, num_points, 3)

            # get "num_scene" indices from range(len(files_list))
            self.use_files_list = random.sample(files_list, num_scene)

            # skip the first line
            # assert it is either txt or  ply files
            for i, point_file_path in enumerate(self.use_files_list):
                print(f"loading {point_file_path}")

                assert point_file_path.endswith(".txt") or point_file_path.endswith(
                    ".ply"), "only support txt or ply files"

                if point_file_path.endswith(".txt"):

                    df = pd.read_csv(point_file_path, sep=" ",
                                     skip_blank_lines=True)
                    df = df.iloc[:, :3]
                    # convert to tensor
                    temp_data = torch.tensor(df.values)
                elif point_file_path.endswith(".ply"):
                    # load ply file
                    # raise ValueError("only support txt files")
                    pcd = o3d.io.read_point_cloud(point_file_path)

                    # Print basic information about the point cloud
                    # print(pcd)

                    # Print the first 10 points
                    # print("First 10 points:")
                    # print(np.asarray(pcd.points)[:10])
                    # [[-0.30167043 -0.90215492 -0.19377652]
                    # [-0.29331112 -0.89266336 -0.19701475]
                    # [-0.30492485 -0.89159894 -0.1947982 ]
                    # [-0.31571221 -0.89394379 -0.1983307 ]
                    # [-0.31758904 -0.8850534  -0.22456694]
                    # [-0.3248471  -0.8797946  -0.24294975]
                    # [-0.28513706 -0.88192683 -0.20044002]
                    # [-0.29690993 -0.88751572 -0.19525513]
                    # [-0.30618262 -0.88741416 -0.19390407]
                    # [-0.33165812 -0.87728107 -0.26005024]]
                    # read ply file
                    #
                    temp_data = torch.tensor(np.asarray(pcd.points))
                else:
                    raise ValueError("only support txt or ply files")

                # if number of row < num_points, pick random row to fill
                if temp_data.size(0) < num_points:
                    # pick random row
                    random_row = torch.randint(0, temp_data.size(
                        0), (num_points - temp_data.size(0),))
                    # fill the rest with random row
                    temp_data = torch.cat([temp_data, temp_data[random_row]])
                elif temp_data.size(0) > num_points:
                    # pick random row
                    random_row = torch.randint(
                        0, temp_data.size(0), (num_points,))
                    temp_data = temp_data[random_row]

                self.data[i, :, :] = temp_data

        # Generate random point clouds
        else:
            self.data = torch.randn(num_scene, num_points, 3) * \
                torch.tensor([10, 100, 5])
            print("random data")
            self.use_files_list = [f'random_{i}' for i in range(num_scene)]
        # print("data", self.data)
        self.mean = self.data.mean(dim=(0, 1))
        if norm_method == "std":
            self.factor = (
                self.data.std(dim=(0, 1))
                if num_scene * num_points > 1
                else torch.ones_like(self.mean)
            )
        elif norm_method == "min-max":
            print("min-max", torch.min(torch.min(self.data, dim=0)
                  [0], dim=0)[0], torch.max(torch.max(self.data, dim=0)[0], dim=0)[0])
            self.factor = (
                torch.max(torch.max(self.data, dim=0)[0], dim=0)[
                    0] - torch.min(torch.min(self.data, dim=0)[0], dim=0)[0]
                if num_scene * num_points > 1
                else torch.ones_like(self.mean)
            )

        print("mean", self.mean, "factor", self.factor)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return pre-normalized data
        return (self.data[idx, :, :] - self.mean) / self.factor


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
    for batch in dataloader:
        x = batch.to(device)
        # print("x", x.shape)  # x torch.Size([1, 2, 3])
        # expand dim 2 (last dim) from 3 to (3+ args.extra_channels ) with zero (one mroe channel)
        x = torch.cat(
            [x, torch.zeros((x.shape[0], x.shape[1], args.extra_channels)).to(device)], dim=-1)
        # print("x", x.shape)  # x torch.Size([1, 2, 4])

        # exit()
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (x.size(0),), device=device
        ).long()
        noise = torch.randn_like(x)
        # print("x", x.shape, "timesteps", timesteps.shape)
        noisy_x = scheduler.add_noise(x, noise, timesteps)

        # print("noisy_x", noisy_x.shape, "timesteps", timesteps.shape)

        optimizer.zero_grad()
        output = model(noisy_x, timesteps)
        # print("output", output.shape)  # output torch.Size([1, 2, 3])
        noise = noise[:, :, :3]
        loss = criterion(output, noise)
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

    row_width = int(ceil(sqrt(len(gts))))
    n_row = int(ceil(len(gts) / row_width ))
    fig = plt.figure(figsize=(10*row_width, 10*n_row))
    # print("n_row", n_row, "row_width", row_width)
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
            sampled_point, xts, x0s, steps = sample(
                model,
                scheduler, args,
                sample_shape=(1, args.N, 3),
                num_inference_steps=50,
                evolution_freq=args.evolution_freq,
                device=device,
            )
            data_mean = dataloader.dataset.mean.to(device)
            data_factor = dataloader.dataset.factor.to(device)
            # all samples
            gt_pcs = dataloader.dataset.data.to(device)

            plot_sample_multi_gt(gt_pcs, xts, x0s, steps,
                                 args, epoch,None,dataloader.dataset.use_files_list)
            gt_pcs = gt_pcs * data_factor + data_mean
            sampled_point = sampled_point * data_factor + data_mean

            # cd_loss, _ = chamfer_distance(gt_pcs, sampled_point)
            cd_losses = []
            for i, gt_pc in enumerate(gt_pcs):
                cd_loss_i, _ = chamfer_distance(
                    gt_pc.unsqueeze(0), sampled_point)
                cd_losses.append(cd_loss_i.item())

            writer.add_scalar("CD_min/epoch", min(cd_losses), epoch)
            writer.add_scalar("CD_max/epoch", max(cd_losses), epoch)
            writer.add_scalar(
                "CD_mean/epoch", sum(cd_losses) / len(cd_losses), epoch)

            sampled_point = sampled_point.cpu().numpy()
            gt_pcs = gt_pcs.cpu().numpy()

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
    scheduler, args,
    sample_shape,
    num_inference_steps=None,
    evolution_freq=None,
    device="cpu",
):
    model.eval()
    num_inference_steps = num_inference_steps or scheduler.config.num_train_timesteps
    scheduler.set_timesteps(num_inference_steps)

    # print("scheduler.timesteps", scheduler.timesteps)

    x = torch.randn(sample_shape, device=device)
    xs = []
    x0t = []
    steps = []
    for i, t in enumerate(tqdm(scheduler.timesteps.to(device), desc="Sampling", leave=False)):
        timesteps = torch.full(
            (x.size(0),), t, device=device, dtype=torch.long)
        # x = x.reshape(x.size(0), -1)
        # print("x", x.shape)
        # exit()

        x_xtra_chn = torch.cat(
            [x, torch.zeros((x.shape[0], x.shape[1], args.extra_channels)).to(device)], dim=-1)
        # print("x", x.shape, "timesteps", timesteps.shape) #x torch.Size([1, 100, 3]) timesteps torch.Size([1])
        output = model(x_xtra_chn, timesteps)
        # print("output", output.shape)  # output torch.Size([1, 2, 3])
        # expanding outpu dim 2 (last dim) from 3 to 4 with zero (one mroe channel)

        x_step_putput = scheduler.step(output, t, x)
        # print("x", x)
        x = x_step_putput["prev_sample"]  # or "pred_original_sample"
        # print("sample_shape", sample_shape)
        # x=x.reshape(sample_shape)
        # print("x", x.shape)
        if (
            evolution_freq is not None and i % evolution_freq == 0
        ) or i == num_inference_steps - 1:

            xs.append(x)
            steps.append(t)
            x0t.append(x_step_putput["pred_original_sample"])
    if num_inference_steps == 1:
        xs.append(x)
        steps.append(0)
        x0t.append(x_step_putput["pred_original_sample"])
    return x, xs, x0t, steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for point clouds"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int,
                        default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--N", type=int, default=100, help="Number of points in each point cloud"
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
        "--visualize_freq", type=int, default=1000, help="Visualize frequency"
    )
    parser.add_argument(
        "--n_hidden_layers", type=int, default=4, help="Number of hidden layers"
    )
    parser.add_argument("--hidden_dim", type=int,
                        default=256, help="Hidden dimension")
    parser.add_argument("--loss_type", type=str,
                        default="mse", help="Loss function")
    parser.add_argument("--model", type=str,
                        default="mlp3d", help="Model type")
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
    parser.add_argument(
        "--extra_channels", type=int, default=0, help="Extra channels in PVCNN >=0"
    )
    parser.add_argument(
        "--point_path", type=str, default=None, help="Path to point cloud"  # either txt or ply
    )
    parser.add_argument("--tb_log_dir", type=str, default="./logs",
                        help="Path to store tensorboard logs")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    # normilzation method, std or min-max
    parser.add_argument("--norm_method", type=str,
                        default="std", help="Normalization method")

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

    def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
        """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
        points = pc * (self.scale_factor if scale else 1)
        return points

    def forward(self, x, t):
        # (B, N, 3) (B,) #x torch.Size([1, 100, 3]) t torch.Size([1])
        x_t = self.point_cloud_model(x, t)

        return x_t


def get_model(args, device="cpu"):

    if args.model == "mlp3d":
        raise ValueError("model mlp3d not supported")
        data_dim = args.N * 3

        return SimpleDiffusionModel3D(
            data_dim=data_dim,
            time_embedding_dim=16,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.n_hidden_layers,
        ).to(device)
    elif args.model == "pvcnn":
        data_dim = args.extra_channels + 3
        return PVCNNDiffusionModel3D(data_dim=data_dim,          point_cloud_model_embed_dim=args.hidden_dim, point_cloud_model="pvcnn",      dropout=0.1,            width_multiplier=1,            voxel_resolution_multiplier=1,).to(device)
    else:
        raise ValueError("model not supported, choose either mlp3d or pvcnn")


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


def log_sample_to_tb(x, gt_pc, key, evo, epoch, writer):
    sampled_tensor = torch.tensor(x, dtype=torch.float)
    gt_pc_tensor = torch.tensor(gt_pc, dtype=torch.float)

    all_tensor = torch.cat([sampled_tensor, gt_pc_tensor], dim=0)

    color_sampled = torch.tensor(
        [[255, 0, 0] for _ in range(sampled_tensor.shape[0])])  # color: red
    color_gt = torch.tensor(
        [[0, 255, 0] for _ in range(gt_pc_tensor.shape[0])])  # color: green

    all_color = torch.cat([color_sampled, color_gt], dim=0)
    # print("shape", all_tensor.shape, all_color.shape)
    # add dimension to tensor to dim 0
    all_tensor = all_tensor.unsqueeze(0)
    all_color = all_color.unsqueeze(0)
    writer.add_mesh(f"PointCloud_{key}_{evo}", vertices=all_tensor, colors=all_color,
                    global_step=epoch)


def main():

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
    dataset = PointCloudDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if False: #test plot_multi_gt
        gt_pcs = dataloader.dataset.data.to(device)
        input_pc_file_list = dataloader.dataset.use_files_list
        plot_multi_gt(gt_pcs,  args, input_pc_file_list, fname=None)
        exit()

    # dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

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
        dataloader,
        optimizer,
        scheduler,
        args,
        device=device,
        start_epoch=start_epoch,
        criterion=criterion, writer=writer
    )

    losses = train_one_epoch(
        dataloader, model, optimizer, scheduler, args, criterion=criterion, device=device
    )
    metric_dict = {"Loss": sum(losses) / len(losses)}
    if not args.no_tensorboard:
        writer.add_scalar("Loss/epoch", sum(losses) /
                          len(losses), args.epochs - 1)

    # samples, _ = sample(model, scheduler, sample_shape=( 1000, data_dim), device=device)

    # Sample from the model
    num_sample_points = 1
    samples = {}
    for i in [1, 5, 10, 50, 100]:
        samples[f"step{i}"] = sample(
            model,
            scheduler, args,
            sample_shape=(num_sample_points, args.N, 3),
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
            samples_updated = samples[key][0] * \
                dataset.factor.to(device) + dataset.mean.to(device)
            # print("samples_updated", key, samples_updated.shape) #samples_updated step1 torch.Size([1, 1, 3])
            # #shape  dataset[:]
            # print("dataset[:]", dataset[:].shape) #dataset[:] torch.Size([2, 1, 3])
            cd_losses = []
            for i in range(len(dataset)):
                loss, _ = chamfer_distance(
                    dataset[i].unsqueeze(0).to(device) *
                    dataset.factor.to(device)
                    + dataset.mean.to(device),
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
                x = x * dataset.factor.to(device) + dataset.mean.to(device)

                cd_losses = []
                for i in range(len(dataset)):
                    loss, _ = chamfer_distance(
                        dataset[i].unsqueeze(0).to(
                            device) * dataset.factor.to(device)
                        + dataset.mean.to(device),
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

    key_to_plot = "step50"

    # get GT  from dataloader
    gt_pc = next(iter(dataloader)).to(device)  # one sample
    gt_pc = gt_pc * dataset.factor.to(device) + dataset.mean.to(device)
    # print("gt_pc", gt_pc.shape) #gt_pc torch.Size([1, N*3])
    gt_pc = gt_pc.reshape(-1, 3)
    gt_pc = gt_pc.cpu().numpy()

    key = key_to_plot
    value = samples[key]

    temp = torch.stack(value[1], dim=0)
    # print("temp", key, temp.shape)
    samples_updated = temp.to(
        device) * dataset.factor.to(device) + dataset.mean.to(device)
    # print("samples_updated", key, samples_updated.shape)
    # print("samples_updated.reshape(-1,3).mean(dim=0)", samples_updated.reshape(-1,3).mean(dim=0))
    # mean_coord_val = samples_updated.reshape(-1, 3).mean(dim=0).cpu()
    # factor_coord_val = samples_updated.reshape(-1, 3).factor(dim=0).cpu()
    # move to cpu
    # min_coord_val = min_coord_val.cpu()
    # max_coord_val = max_coord_val.cpu()
    # print(key, "\tmin max, ", mean_coord_val, std_coord_val)
    samples_updated = samples_updated.cpu().numpy()

    if not args.no_tensorboard:
        for i, x in tqdm(enumerate(samples_updated)):
            # print("x", x.shape)
            # x_shape = x.reshape(x.shape[0], -1, 3)
            log_sample_to_tb(
                x[0, :, :],
                gt_pc,
                key,
                i * args.evolution_freq -
                (1 if i == len(samples_updated) - 1 else 0),
                args.epochs, writer
            )

    if not args.no_tensorboard:
        hparam_dict = vars(args)
        writer.add_hparams(hparam_dict, metric_dict)
        writer.close()


if __name__ == "__main__":
    main()
