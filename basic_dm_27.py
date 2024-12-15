import argparse
import glob
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
from geomloss import SamplesLoss  # Install via pip install geomloss
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm, trange
import plotly.graph_objects as go
import wandb
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PointCloudDataset(Dataset):  # normalized per axis, not per sample
    def __init__(self, num_scene: int = 1000, num_points: int = 10):
        super().__init__()
        self.data = torch.randn(num_scene, num_points, 3) * torch.tensor([10, 100, 5])
        # print("data", self.data)
        self.mean = self.data.mean(dim=(0, 1))
        self.std = (
            self.data.std(dim=(0, 1))
            if num_scene * num_points > 1
            else torch.ones_like(self.mean)
        )

        print("mean", self.mean, "std", self.std)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return pre-normalized data
        return (self.data[idx, :, :] - self.mean) / self.std


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


# Simple diffusion model with time embeddings concatenated to the data
class SimpleDiffusionModel3D(nn.Module):
    def __init__(
        self, data_dim, time_embedding_dim=16, hidden_dim=128, num_hidden_layers=6
    ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_hidden_layers):
            self.net.add_module("layer", nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module("relu", nn.ReLU())
        self.net.add_module("layer2", nn.Linear(hidden_dim, data_dim))

    def forward(self, x, t):
        # Get time embeddings
        t_emb = get_sinusoidal_time_embedding(t, self.time_embedding_dim)
        # Concatenate x and t_emb
        x = x.reshape(x.size(0), -1)
        # print("x", x.shape, "t_emb", t_emb.shape)  #x torch.Size([1, 9]) t_emb torch.Size([1, 16])
        x_t = torch.cat([x, t_emb], dim=-1)
        # Pass through network
        x_t = self.net(x_t)
        return x_t.reshape(x.size(0), -1, 3)


# CNN-based model


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
def train_one_epoch(dataloader, model, optimizer, scheduler, criterion, device):
    model.train()
    losses = []
    for batch in dataloader:
        x = batch.to(device)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (x.size(0),), device=device
        ).long()
        noise = torch.randn_like(x)
        noisy_x = scheduler.add_noise(x, noise, timesteps)

        optimizer.zero_grad()
        loss = criterion(model(noisy_x, timesteps), noise)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    args,
    device="cpu",
    start_epoch=0,
    criterion=nn.MSELoss(),
):
    model.train()
    tqdm_range = trange(start_epoch, args.epochs, desc="Epoch")
    checkpoint_fname = f"/data/palakons/checkpoint/cp_dm_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
    for epoch in tqdm_range:
        losses = train_one_epoch(
            dataloader, model, optimizer, scheduler, criterion, device
        )
        tqdm_range.set_description(f"loss: {sum(losses)/len(losses):.4f}")
        if not args.no_wandb:
            wandb.log({"loss": sum(losses) / len(losses), "epoch": epoch})
        if not args.no_checkpoint and epoch % args.checkpoint_freq == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            # save at /data/palakons/checkpoint/cp_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth
            torch.save(checkpoint, checkpoint_fname)
            print("checkpoint saved at", checkpoint_fname)
        if not args.no_wandb and epoch % args.visualize_freq == 0:
            sampled_point, _ = sample(
                model,
                scheduler,
                sample_shape=(1, args.N, 3),
                num_inference_steps=50,
                evolution_freq=args.evolution_freq,
                device=device,
            )
            data_mean = dataloader.dataset.mean.to(device)
            data_std = dataloader.dataset.std.to(device)
            # get GT  from dataloader
            gt_pc = next(iter(dataloader)).to(device)  # one sample
            gt_pc = gt_pc * data_std + data_mean
            sampled_point = sampled_point * data_std + data_mean

            cd_loss, _ = chamfer_distance(gt_pc, sampled_point)
            wandb.log({"cd": cd_loss, "epoch": epoch})

            sampled_point = sampled_point.cpu().numpy()
            gt_pc = gt_pc.cpu().numpy()

            log_sample_to_wandb(
                sampled_point[0, :, :], gt_pc[0, :, :], f"for_visual", 50, epoch
            )  # support M=1 only
    if not args.no_checkpoint:
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
@torch.no_grad()
def sample(
    model,
    scheduler,
    sample_shape,
    num_inference_steps=None,
    evolution_freq=None,
    device="cpu",
):
    model.eval()
    num_inference_steps = num_inference_steps or scheduler.config.num_train_timesteps
    scheduler.set_timesteps(num_inference_steps)

    x = torch.randn(sample_shape, device=device)
    xs = []
    for i, t in enumerate(tqdm(scheduler.timesteps.to(device), desc="Sampling")):
        timesteps = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        # x = x.reshape(x.size(0), -1)
        x = scheduler.step(model(x, timesteps), t, x)["prev_sample"]
        # print("sample_shape", sample_shape)
        # x=x.reshape(sample_shape)
        # print("x", x.shape)
        if (
            evolution_freq is not None and i % evolution_freq == 0
        ) or i == num_inference_steps - 1:
            xs.append(x)
    if num_inference_steps ==1:
        xs.append(x)
    return x, xs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for point clouds"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--N", type=int, default=1, help="Number of points in each point cloud"
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
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--visualize_freq", type=int, default=10, help="Visualize frequency"
    )
    parser.add_argument(
        "--n_hidden_layers", type=int, default=6, help="Number of hidden layers"
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--loss_type", type=str, default="mse", help="Loss function")
    parser.add_argument("--model", type=str, default="mlp3d", help="Model type")
    parser.add_argument(
        "--checkpoint_freq", type=int, default=100000, help="Checkpoint frequency"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_checkpoint", action="store_true", help="No checkpoint")
    parser.add_argument(
        "--evolution_freq", type=int, default=10, help="Evolution frequency"
    )
    return parser.parse_args()


def match_args(args1, args2):
    # if number of args1 != number of args2:
    # make copy of args1, args2
    arga = vars(args1).copy()
    argb = vars(args2).copy()
    # remove checkpoint_freq,"epochs", "no_wandb" from both, if exist
    for key in ["checkpoint_freq", "epochs", "no_wandb", "no_checkpoint"]:
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


def get_model(args, device="cpu"):
    data_dim = args.N * 3
    if args.model == "mlp3d":
        return SimpleDiffusionModel3D(
            data_dim=data_dim,
            time_embedding_dim=16,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.n_hidden_layers,
        ).to(device)
    else:
        raise ValueError("model not supported")


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


def log_sample_to_wandb(x, gt_pc, key, evo, epoch):
    # x = x.cpu().numpy()
    # if gt_pc is not in cpu, move to cpu
    # gt_pc = gt_pc.cpu().numpy()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x[:, 0],
                y=x[:, 1],
                z=x[:, 2],
                mode="markers",
                name="sampled",
            ),
            go.Scatter3d(
                x=gt_pc[:, 0],
                y=gt_pc[:, 1],
                z=gt_pc[:, 2],
                mode="markers",
                name="gt",
            ),
        ]
    )

    # add legend
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), title=key
    )
    wandb.log({f"evolution_{key}_{evo}": wandb.Plotly(fig), "epoch": epoch})


args = parse_args()
print(args)
set_seed(args.seed)


if not args.no_wandb:
    wandb.init(project="point_cloud_diffusion", config=vars(args))
# Set up device, dataset, dataloader, model, optimizer, and scheduler


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
dataset = PointCloudDataset(num_scene=args.M, num_points=args.N)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = get_model(args, device=device)

if not args.no_wandb:
    wandb.log({"model_params": sum(p.numel() for p in model.parameters())})

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# linear, scaled_linear, or squaredcos_cap_v2.
scheduler = DDPMScheduler(
    num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule
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
        if not args.no_wandb:
            wandb.run.summary.update(
                {"start_epoch": start_epoch, "checkpoint": current_cp_fname}
            )
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
    criterion=criterion,
)

losses = train_one_epoch(
    dataloader, model, optimizer, scheduler, criterion=criterion, device=device
)
if not args.no_wandb:
    wandb.log({"loss": sum(losses) / len(losses), "epoch": args.epochs - 1})

# samples, _ = sample(model, scheduler, sample_shape=( 1000, data_dim), device=device)

# Sample from the model
num_sample_points = 1
samples = {}
for i in [1, 5, 10, 50, 100]:
    samples[f"step{i}"] = sample(
        model,
        scheduler,
        sample_shape=(num_sample_points, args.N, 3),
        num_inference_steps=i,
        evolution_freq=args.evolution_freq,
        device=device,
    )


# make the plot that will be logged to wandb
plt.figure(figsize=(10, 10))

for key, value in samples.items():
    # print(samples[key][0].shape)
    # print("dataset.std",dataset.std.shape)
    samples_updated = samples[key][0] * dataset.std.to(device) + dataset.mean.to(device)
    # print("samples_updated", key, samples_updated.shape) #samples_updated step1 torch.Size([1, 1, 3])
    # #shape  dataset[:]
    # print("dataset[:]", dataset[:].shape) #dataset[:] torch.Size([2, 1, 3])
    cd_losses = []
    for i in range(len(dataset)):
        loss, _ = chamfer_distance(
            dataset[i].unsqueeze(0).to(device) * dataset.std.to(device)
            + dataset.mean.to(device),
            samples_updated.to(device),
        )
        cd_losses.append(loss.item())

    # log minumum loss

    print(key, "\t", f"CD: { min(cd_losses):.2f} at {cd_losses.index(min(cd_losses))}")

    if not args.no_wandb:
        wandb.log({f"CD_{key}": min(cd_losses), "epoch": args.epochs})

    error = []
    assert len(samples[key][1]) > 1, "need more than 1 sample to plot"
    for x in samples[key][1]:
        x = x * dataset.std.to(device) + dataset.mean.to(device)

        cd_losses = []
        for i in range(len(dataset)):
            loss, _ = chamfer_distance(
                dataset[i].unsqueeze(0).to(device) * dataset.std.to(device)
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
# store_dir = wandb.run.dir if not args.no_wandb else "/home/palakons/from_scratch"
# plt.savefig(
#     f"{store_dir}/diffusion_model_{args.epochs}_{args.num_train_timesteps}_{args.beta_schedule}.png"
# )
if not args.no_wandb:
    wandb.log(
        {
            "plot": wandb.Image(plt),
            "epoch": args.epochs,
        }
    )

# plot evolution

key_to_plot = "step50"

# get GT  from dataloader
gt_pc = next(iter(dataloader)).to(device)  # one sample
gt_pc = gt_pc * dataset.std.to(device) + dataset.mean.to(device)
# print("gt_pc", gt_pc.shape) #gt_pc torch.Size([1, N*3])
gt_pc = gt_pc.reshape(-1, 3)
gt_pc = gt_pc.cpu().numpy()

key = key_to_plot
value = samples[key]


temp = torch.stack(value[1], dim=0)
# print("temp", key, temp.shape)
samples_updated = temp.to(device) * dataset.std.to(device) + dataset.mean.to(device)
# print("samples_updated", key, samples_updated.shape)
# print("samples_updated.reshape(-1,3).mean(dim=0)", samples_updated.reshape(-1,3).mean(dim=0))
mean_coord_val = samples_updated.reshape(-1, 3).mean(dim=0).cpu()
std_coord_val = samples_updated.reshape(-1, 3).std(dim=0).cpu()
# move to cpu
# min_coord_val = min_coord_val.cpu()
# max_coord_val = max_coord_val.cpu()
# print(key, "\tmin max, ", mean_coord_val, std_coord_val)
samples_updated = samples_updated.cpu().numpy()

if not args.no_wandb:
    for i, x in enumerate(samples_updated):
        # print("x", x.shape)
        # x_shape = x.reshape(x.shape[0], -1, 3)
        log_sample_to_wandb(
            x[0, :, :],
            gt_pc,
            key,
            i * args.evolution_freq - (1 if i == len(samples_updated) - 1 else 0),
            args.epochs,
        )

if not args.no_wandb:
    wandb.finish()
