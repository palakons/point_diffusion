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
            point_path = point_path.split(",")
            print("loading point cloud from", point_path[0])
            # assert point_path is  a list
            assert len(point_path) >= num_scene

            if num_scene > 1:
                raise ValueError("num_scene > 1 not supported, yet")

            # load point cloud from point_path
            # /data/palakons/dataset/astyx/scene/0/000000.txt
            # load point cloud from point_path, as a text file, separate by space, use pandas
            #
            # X Y Z V_r Mag
            # 0.108118820531769647 2.30565393293341492 -0.279097884893417358 0 48
            # 1.18417095921774496 2.25506417530992165 -0.122170589864253998 0 48

            # creat blank tensor dim (num_scene, num_points, 3)
            self.data = torch.zeros(num_scene, num_points, 3)
            # skip the first line
            # assert it is either txt or  ply files
            assert point_path[0].endswith(".txt") or point_path[0].endswith(
                ".ply"), "only support txt or ply files"
            if point_path[0].endswith(".txt"):

                df = pd.read_csv(point_path[0], sep=" ",
                                 skip_blank_lines=True)
                df = df.iloc[:, :3]
                # convert to tensor
                temp_data = torch.tensor(df.values)
            elif point_path[0].endswith(".ply"):
                # load ply file
                # raise ValueError("only support txt files")
                pcd = o3d.io.read_point_cloud(point_path[0])

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
                random_row = torch.randint(0, temp_data.size(0), (num_points,))
                temp_data = temp_data[random_row]

            self.data[0, :, :] = temp_data

        # Generate random point clouds
        else:
            self.data = torch.randn(num_scene, num_points, 3) * \
                torch.tensor([10, 100, 5])
        # print("data", self.data)
        self.mean = self.data.mean(dim=(0, 1))
        if norm_method == "std":
            self.factor = (
                self.data.std(dim=(0, 1))
                if num_scene * num_points > 1
                else torch.ones_like(self.mean)
            )
        elif norm_method == "min-max":
            print("min-max",torch.min(torch.min(self.data,dim=0)[0],dim=0)[0], torch.max(torch.max(self.data,dim=0)[0],dim=0)[0])
            self.factor = (
                torch.max(torch.max(self.data,dim=0)[0],dim=0)[0] -torch.min(torch.min(self.data,dim=0)[0],dim=0)[0]
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
            dataloader, model, optimizer, scheduler,  args, criterion, device
        )
        tqdm_range.set_description(f"loss: {sum(losses)/len(losses):.4f}")
        if not args.no_tensorboard:
            writer.add_scalar("Loss/epoch", sum(losses) / len(losses), epoch)

        if not args.no_checkpoint and epoch % args.checkpoint_freq == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            # save at /data/palakons/checkpoint/cp_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth
            torch.save(checkpoint, checkpoint_fname)
            print("checkpoint saved at", checkpoint_fname)
        if not args.no_tensorboard and epoch % args.visualize_freq == 0:
            sampled_point, _ = sample(
                model,
                scheduler, args,
                sample_shape=(1, args.N, 3),
                num_inference_steps=50,
                evolution_freq=args.evolution_freq,
                device=device,
            )
            data_mean = dataloader.dataset.mean.to(device)
            data_factor = dataloader.dataset.factor.to(device)
            # get GT  from dataloader
            gt_pc = next(iter(dataloader)).to(device)  # one sample
            gt_pc = gt_pc * data_factor + data_mean
            sampled_point = sampled_point * data_factor + data_mean

            cd_loss, _ = chamfer_distance(gt_pc, sampled_point)
            
            writer.add_scalar("CD/epoch", cd_loss, epoch)

            sampled_point = sampled_point.cpu().numpy()
            gt_pc = gt_pc.cpu().numpy()

            log_sample_to_tb(
                sampled_point[0, :, :], gt_pc[0,
                                              :, :], f"for_visual", 50, epoch
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

    x = torch.randn(sample_shape, device=device)
    xs = []
    for i, t in enumerate(tqdm(scheduler.timesteps.to(device), desc="Sampling")):
        timesteps = torch.full(
            (x.size(0),), t, device=device, dtype=torch.long)
        # x = x.reshape(x.size(0), -1)
        # print("x", x.shape)
        # exit()

        x_4 = torch.cat(
            [x, torch.zeros((x.shape[0], x.shape[1], args.extra_channels)).to(device)], dim=-1)
        # print("x", x.shape, "timesteps", timesteps.shape)
        output = model(x_4, timesteps)
        # print("output", output.shape)  # output torch.Size([1, 2, 3])
        # expanding outpu dim 2 (last dim) from 3 to 4 with zero (one mroe channel)

        x = scheduler.step(output, t, x)["prev_sample"]
        # print("sample_shape", sample_shape)
        # x=x.reshape(sample_shape)
        # print("x", x.shape)
        if (
            evolution_freq is not None and i % evolution_freq == 0
        ) or i == num_inference_steps - 1:
            xs.append(x)
    if num_inference_steps == 1:
        xs.append(x)
    return x, xs


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
    parser.add_argument("--no_tensorboard", #action="store_true", 
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
    # parser.add_argument(
    #     "--extra_channels", type=int, default=0, help="Extra channels in PVCNN >=0"
    # )
    parser.add_argument(
        "--point_path", type=str, default=None, help="Path to point cloud" #either txt or ply
    )
    parser.add_argument("--tb_log_dir", type=str, default="./logs",
                        help="Path to store tensorboard logs")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    #normilzation method, std or min-max
    parser.add_argument("--norm_method", type=str, default="std", help="Normalization method")
    return parser.parse_args()


def match_args(args1, args2):
    # if number of args1 != number of args2:
    # make copy of args1, args2
    arga = vars(args1).copy()
    argb = vars(args2).copy()
    # remove checkpoint_freq,"epochs", "no_tensorboard" from both, if exist
    for key in ["checkpoint_freq", "epochs", "no_tensorboard", "no_checkpoint"]:
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

    if args.model == "mlp3d":
        data_dim = args.N * 3
        return SimpleDiffusionModel3D(
            data_dim=data_dim,
            time_embedding_dim=16,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.n_hidden_layers,
        ).to(device)
    # elif args.model == "pvcnn":
    #     data_dim = args.extra_channels + 3
    #     return PVCNNDiffusionModel3D(data_dim=data_dim,          point_cloud_model_embed_dim=args.hidden_dim, point_cloud_model="pvcnn",      dropout=0.1,            width_multiplier=1,            voxel_resolution_multiplier=1,).to(device)
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


def log_sample_to_tb(x, gt_pc, key, evo, epoch):
    sampled_tensor = torch.tensor(x,dtype=torch.float)
    gt_pc_tensor = torch.tensor(gt_pc,dtype=torch.float)

    all_tensor = torch.cat([sampled_tensor, gt_pc_tensor], dim=0)

    color_sampled = torch.tensor([[255, 0, 0] for _ in range(sampled_tensor.shape[0])]) #color: red
    color_gt = torch.tensor([[0, 255, 0] for _ in range(gt_pc_tensor.shape[0])]) #color: green

    all_color = torch.cat([color_sampled, color_gt], dim=0)
    # print("shape", all_tensor.shape, all_color.shape)
    #add dimension to tensor to dim 0
    all_tensor = all_tensor.unsqueeze(0)
    all_color = all_color.unsqueeze(0)
    writer.add_mesh(f"PointCloud_{key}_{evo}", vertices=all_tensor, colors=all_color, 
                    global_step=epoch)
    



args = parse_args()
if args.model != "pvcnn":
    args.extra_channels = 0
    print("extra_channels set to 0 for non-pvcnn model")
else:
    assert args.extra_channels >= 0, "extra_channels must be >=0 for pvcnn"
print(args)
set_seed(args.seed)


if not args.no_tensorboard:
    writer = SummaryWriter(log_dir=args.tb_log_dir+f"/{args.run_name if  args.run_name else datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

        


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
dataset = PointCloudDataset(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

model = get_model(args, device=device)

if not args.no_tensorboard:
    writer.add_scalar("model_params", sum(p.numel() for p in model.parameters()))

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
        if not args.no_tensorboard:
            raise ValueError("not implemented")

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
    dataloader, model, optimizer, scheduler, args, criterion=criterion, device=device
)
metric_dict = {"Loss": sum(losses) / len(losses)}
if not args.no_tensorboard:
    writer.add_scalar("Loss/epoch", sum(losses) / len(losses), args.epochs - 1)

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
                dataset[i].unsqueeze(0).to(device) * dataset.factor.to(device)
                + dataset.mean.to(device),
                samples_updated.to(device),
            )
            cd_losses.append(loss.item())

        # log minumum loss

        print(
            key, "\t", f"CD: { min(cd_losses):.2f} at {cd_losses.index(min(cd_losses))}")
        if not args.no_tensorboard:
            writer.add_scalar(f"CD_{key}", min(cd_losses), args.epochs)
            
            #add cd to metric_dict
            metric_dict = {**{f"CD_{key}": min(cd_losses)},**metric_dict}

        error = []
        assert len(samples[key][1]) > 1, "need more than 1 sample to plot"
        for x in samples[key][1]:
            x = x * dataset.factor.to(device) + dataset.mean.to(device)

            cd_losses = []
            for i in range(len(dataset)):
                loss, _ = chamfer_distance(
                    dataset[i].unsqueeze(0).to(device) * dataset.factor.to(device)
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
    
    writer.add_figure( f"Evolution",plt.gcf(), args.epochs)
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
            args.epochs,
        )

if not args.no_tensorboard:
    hparam_dict = vars(args)
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()
