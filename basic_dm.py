import os
from matplotlib import pyplot as plt
import glob
import argparse
import inspect
from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
from tqdm import trange, tqdm
from datetime import datetime
import time
from geomloss import SamplesLoss  # Install via pip install geomloss

# Dataset of n-dimensional point clouds


class PointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=10):
        super().__init__()
        self.data = torch.randn(num_samples, num_points * 3) * torch.tensor(
            [10, 100, 5] * num_points
        )  # Random point clouds
        # self.data = torch.tensor([[6.,4.,2.]*num_points]* num_samples)

        # normalize data, extract mean and std
        self.mean = self.data.mean(dim=0)
        self.std = (
            self.data.std(
                dim=0) if num_samples > 1 else torch.ones_like(self.mean)
        )
        print("mean", self.mean, "std", self.std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return the normalized data
        return (self.data[idx] - self.mean) / self.std


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
class SimpleDiffusionModel(nn.Module):
    def __init__(self, data_dim, time_embedding_dim=16, hidden_dim=128, num_hidden_layers=6):
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
        x_t = torch.cat([x, t_emb], dim=-1)
        # Pass through network
        return self.net(x_t)
    
#CNN-based model
class SimpleDiffusionModelCNN(nn.Module):
    def __init__(self,data_dim, time_embedding_dim=16, hidden_dim=128, num_hidden_layers=6,cnn_dim=64,cnn_channels=1):
        super().__init__()
        #TODO: implement CNN-based model
# https://discuss.pytorch.org/t/simple-implemetation-of-chamfer-distance-in-pytorch-for-2d-point-cloud-data/143083/2
from pytorch3d.loss import chamfer_distance
class PointCloudLoss(nn.Module):
    def __init__(self, npoints,emd_weight =.5 ):
        super().__init__()
        self.npoints = npoints
        self.emd_weight = emd_weight 
        assert self.emd_weight >= 0 and self.emd_weight <= 1, "emd_weight in PointCloudLoss() should be between 0 and 1"

    # def earth_mover_distance(self, y_true, y_pred): #work for ordered, sequential data (e.g., histograms or signals)
    #     return torch.mean(torch.square(
    #         torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1).mean()
    
    def earth_mover_distance2(self, y_true, y_pred):
        dist = torch.cdist(y_true, y_pred, p=2)
        assignment = torch.argmin(dist, dim=-1)
        matched_pred = torch.gather(y_pred, dim=1, index=assignment.unsqueeze(-1).expand(-1, -1, y_pred.shape[-1]))
        return torch.mean((y_true - matched_pred) ** 2)

    def earth_mover_distance_sinkhorn(self,y_true, y_pred):
        # SamplesLoss("sinkhorn", p=2) computes an approximation of EMD
        loss = SamplesLoss("sinkhorn", p=2)
        return loss(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        y_true = y_true.view(-1, self.npoints, 3)
        y_pred = y_pred.view(-1, self.npoints, 3)
        # print("y_true", y_true.shape, "y_pred", y_pred.shape)
        # print("npoints", self.npoints)
        # print("ndim", y_true.ndim, y_pred.ndim)
        # print("type y_true", type(y_true), "type y_pred", type(y_pred))
        # if y_true.ndim != 3 and self.npoints is not None:
        #     # self.batch = y_true.shape[0] // self.npoints
        #     # y_true = y_true.view(self.batch, self.npoints, 3)
        #     y_true = y_true.view(-1, self.npoints, 3)

        # if y_pred.ndim != 3 and self.npoints is not None:
        #     # self.batch = y_true.shape[0] // self.npoints
        #     # y_pred = y_pred.view(self.batch, self.npoints, 3)
        #     y_pred = y_pred.view(-1, self.npoints, 3)
        
        # # print("y_true", y_true.shape, "y_pred", y_pred.shape)

        cd = torch.Tensor(chamfer_distance(y_true, y_pred)[0])
        # emd = self.earth_mover_distance(y_true, y_pred)
        # emd2 = self.earth_mover_distance2(y_true, y_pred)
        emd_sk = self.earth_mover_distance_sinkhorn(y_true, y_pred)

        # print("cd", cd, "emd2", emd2, "emd_sk", emd_sk)

        # print("cd", cd, "emd", emd)
        # print("type cd", type(cd), "type emd", type(emd))
        # y_true torch.Size([1, 6]) y_pred torch.Size([1, 6])
        # npoints 2
        # ndim 2 2
        # type y_true <class 'torch.Tensor'> type y_pred <class 'torch.Tensor'>
        # y_true torch.Size([1, 2, 3]) y_pred torch.Size([1, 2, 3])
        # cd (tensor(3.2807, device='cuda:0', grad_fn=<AddBackward0>), None) emd tensor(0.7304, device='cuda:0', grad_fn=<MeanBackward0>)
        # type cd <class 'tuple'> type emd <class 'torch.Tensor'>
        return (1- self.emd_weight) * cd + self.emd_weight * emd_sk


# Training function
def train_one_epoch(dataloader, model, optimizer, scheduler, criterion, device):
    
    losses = []
    for x in dataloader:
        x = x.to(device)
        # Sample random timesteps
        batch_size = x.size(0)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        # Get noise
        noise = torch.randn_like(x)
        # Get noisy x
        noisy_x = scheduler.add_noise(x, noise, timesteps)
        # Predict the noise residual
        noise_pred = model(noisy_x, timesteps)
        # Compute loss
        loss = criterion(noise_pred, noise)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_epochs=10,
    is_wandb=False,
    device="cpu",
    start_epoch=0,
    criterion=nn.MSELoss(),
):
    model.train()
    tr = trange(start_epoch, num_epochs)
    for epoch in tr:
        losses  =train_one_epoch(dataloader, model, optimizer, scheduler, criterion, device)
        tr.set_description(f"loss: {sum(losses)/len(losses):.4f}")
        if is_wandb:
            wandb.log({"loss": sum(losses) / len(losses), "epoch": epoch})
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
    # Set timesteps
    if num_inference_steps is None:
        num_inference_steps = scheduler.config.num_train_timesteps
    # print(num_inference_steps)
    scheduler.set_timesteps(num_inference_steps)
    # print(scheduler.timesteps)

    # Start from pure noise
    x = torch.randn(sample_shape).to(device)
    xs = []
    progress_bar = tqdm(scheduler.timesteps.to(
        device), desc=f"Sampling ({x.shape})")
    for i, t in enumerate(progress_bar):
        timesteps = torch.full(
            (x.size(0),), t, device=device, dtype=torch.long)
        # Predict noise
        noise_pred = model(x, timesteps)
        # Compute previous noisy sample x_t -> x_{t-1}
        # x = scheduler.step(noise_pred, t, x)['prev_sample']
        x = scheduler.step(noise_pred, t, x)["prev_sample"]
        if evolution_freq is not None and t % evolution_freq == 0:
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
                        default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--N", type=int, default=1, help="Number of points in each point cloud"
    )
    parser.add_argument(
        "--M", type=int, default=1, help="Number of point cloud scenes to load"
    )
    # num_train_timesteps
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of training time steps",
    )
    # beta_schedule: options linear, scaled_linear, or squaredcos_cap_v2
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="Beta schedule"
    )
    # not wandb
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    # visulize every 10
    parser.add_argument("--visualize_freq", type=int,
                        default=10, help="Visualize frequency")
    # n_lay + hidden_dim
    parser.add_argument("--n_hidden_layers", type=int,
                        default=6, help="Number of hidden layers")
    parser.add_argument("--hidden_dim", type=int,
                        default=128, help="Hidden dimension")
    #loss function mse, chamfer, emd
    parser.add_argument("--loss_type", type=str, default="mse", help="Loss function")
    #model mlp, cnn
    parser.add_argument("--model", type=str, default="mlp", help="Model type")
    return parser.parse_args()


def match_args(args1, args2):
    # if number of args1 != number of args2:
    if len(vars(args1)) != len(vars(args2)):
        # print("number of args not the same")
        return False
    # check one by one if they are the same, except for epochs
    for key, value in vars(args1).items():
        if key in ["epochs", "no_wandb"]:
            continue
        if value != vars(args2)[key]:
            print(f"{key} not the same", value, "vs", vars(args2)[key])
            return False
    return True

    # if checkpoint["args"].num_train_timesteps == args.num_train_timesteps and checkpoint["args"].beta_schedule == args.beta_schedule and checkpoint["args"].N == args.N and checkpoint["args"].M == args.M and checkpoint["args"].lr == args.lr and checkpoint["args"].batch_size == args.batch_size

    # for key, value in vars(args1).items():
    #     if value != vars(args2)[key]:
    #         return False
    # return True


args = parse_args()


if not args.no_wandb:
    import wandb

    wandb.init(project="point_cloud_diffusion", config=vars(args))
# Set up device, dataset, dataloader, model, optimizer, and scheduler


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
dataset = PointCloudDataset(num_samples=args.M, num_points=args.N)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

data_dim = dataset[0].shape[0]
if args.model == "mlp":
    model = SimpleDiffusionModel(data_dim=data_dim, hidden_dim=args.hidden_dim,
                             num_hidden_layers=args.n_hidden_layers).to(device)
elif args.model == "cnn":
    model = SimpleDiffusionModelCNN(data_dim=data_dim, hidden_dim=args.hidden_dim,
                             num_hidden_layers=args.n_hidden_layers).to(device)
else:
    raise ValueError("model not supported")

if not args.no_wandb:
    wandb.log({"model_params": sum(p.numel() for p in model.parameters())})
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# linear, scaled_linear, or squaredcos_cap_v2.
scheduler = DDPMScheduler(
    num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule
)

start_epoch = 0

# check checkpoint
# list files with while card /data/palakons/checkpoint/cp_dm_*.pth using glob
files = glob.glob("/data/palakons/checkpoint/cp_dm_*.pth")
# Find the checkpoint wih furthest epoch <= args.epochs, with matchign all other params
max_epoch = 0
current_cp_fname = None
for fname in files:
    checkpoint = torch.load(fname)
    if match_args(checkpoint["args"], args):
        if (
            checkpoint["args"].epochs <= args.epochs
            and checkpoint["args"].epochs > max_epoch
        ):
            max_epoch = checkpoint["args"].epochs
            current_cp_fname = fname
            # print("current_cp", current_cp_fname)
if current_cp_fname is not None:
    current_cp = torch.load(current_cp_fname)
    print("loading checkpoint")
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

if args.loss_type == "mse":
    criterion = nn.MSELoss()
elif args.loss_type == "chamfer":
    criterion = PointCloudLoss(npoints=args.N, emd_weight=0)
elif args.loss_type == "emd":
    criterion = PointCloudLoss(npoints=args.N, emd_weight=1)
else:
    raise ValueError("loss not supported")

# Train the model
train(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_epochs=args.epochs,
    is_wandb=not args.no_wandb,
    device=device,
    start_epoch=start_epoch,
    criterion=criterion,
)
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "args": args,
}
# save at /data/palakons/checkpoint/cp_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth
checkpoint_fname = f"/data/palakons/checkpoint/cp_dm_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
torch.save(checkpoint, checkpoint_fname)
print("checkpoint saved at", checkpoint_fname)


losses  =train_one_epoch(dataloader, model, optimizer, scheduler, criterion=criterion, 
    device=device)
if not args.no_wandb:
    wandb.log({"loss": sum(losses) / len(losses), "epoch": args.epochs-1})

# samples, _ = sample(model, scheduler, sample_shape=( 1000, data_dim), device=device)

# Sample from the model
samples = {
    "step1": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=1,
        evolution_freq=1,
        device=device,
    ),
    "step5": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=5,
        evolution_freq=1,
        device=device,
    ),
    "step10": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=10,
        evolution_freq=1,
        device=device,
    ),
    "step50": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=50,
        evolution_freq=1,
        device=device,
    ),
    "step100": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=100,
        evolution_freq=1,
        device=device,
    ),
    "step200": sample(
        model,
        scheduler,
        sample_shape=(1000, data_dim),
        num_inference_steps=200,
        evolution_freq=1,
        device=device,
    ),
    #    "step500": sample(model, scheduler, sample_shape=(1000, data_dim), num_inference_steps=500, evolution_freq=1, device=device),
    #    "step1000": sample(model, scheduler, sample_shape=(1000, data_dim), num_inference_steps=1000, evolution_freq=1, device=device)
}


fig = plt.figure(figsize=(10, 5))
for key, value in samples.items():
    samples_updated = samples[key][0] * \
        dataset.std.to(device) + dataset.mean.to(device)
    loss, _ = chamfer_distance(
        dataset[:].unsqueeze(0).to(device) * dataset.std.to(device)
        + dataset.mean.to(device),
        samples_updated.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device),
    )
    print(key, "\t", f"CD: {loss.item():.2f}")
    if not args.no_wandb:
        wandb.log({f"CD_{key}": loss.item(), "epoch": args.epochs})
    error = []
    for x in samples[key][1]:
        x = x * dataset.std.to(device) + dataset.mean.to(device)
        loss, _ = chamfer_distance(
            dataset[:].unsqueeze(0).to(device) * dataset.std.to(device)
            + dataset.mean.to(device),
            x.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device),
        )
        # print(f"{loss.item():.2f}", end=", ")
        loss = loss.item()
        error.append(loss)
    # ax = plt.plot(error, label=key)
    plt.plot(
        [i / (len(error) - (1 if len(error) > 1 else 0))
         for i in range(len(error))],
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
store_dir = wandb.run.dir if not args.no_wandb else "/home/palakons/from_scratch"
plt.savefig(
    f"{store_dir}/diffusion_model_{args.epochs}_{args.num_train_timesteps}_{args.beta_schedule}.png"
)


temp_dir = (
    wandb.run.dir if not args.no_wandb else "/home/palakons/from_scratch"
) + f"/evolutions"  # + f"/temp_{time.time()}"
mkdir_cmd = f"mkdir -p {temp_dir}"
os.system(mkdir_cmd)
for key, value in samples.items():
    try:
        temp = torch.stack(value[1], dim=0)
        # print("temp", key, temp.shape)
        samples_updated = temp.to(device) * dataset.std.to(
            device
        ) + dataset.mean.to(device)
        # print("samples_updated", key, samples_updated.shape)
    except:
        print("error, maybe too larger", key)
        continue
    # print("samples_updated.reshape(-1,3).mean(dim=0)", samples_updated.reshape(-1,3).mean(dim=0))
    mean_coord_val = samples_updated.reshape(-1, 3).mean(dim=0).cpu()
    std_coord_val = samples_updated.reshape(-1, 3).std(dim=0).cpu()
    # move to cpu
    # min_coord_val = min_coord_val.cpu()
    # max_coord_val = max_coord_val.cpu()
    # print(key, "\tmin max, ", mean_coord_val, std_coord_val)
    for i, x in enumerate(samples_updated):
        if i % args.visualize_freq != 0:
            # print("skip", i)
            continue
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        # print("x",x.shape)
        x = x.cpu().numpy()
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], label=f"{i}")
        ax.set_title(
            f"{i} {args.epochs} epochs, {args.num_train_timesteps} timesteps, {args.beta_schedule} schedule")
        # equal
        ax.set_aspect('equal')
        # set min max
        # sigma = 3
        # ax.set_xlim(mean_coord_val[0]-sigma*std_coord_val[0], mean_coord_val[0]+sigma*std_coord_val[0])
        # ax.set_ylim(mean_coord_val[1]-sigma*std_coord_val[1], mean_coord_val[1]+sigma*std_coord_val[1])
        # ax.set_zlim(mean_coord_val[2]-sigma*std_coord_val[2], mean_coord_val[2]+sigma*std_coord_val[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(f"{temp_dir}/{key}_{i:06}.png")
        # print(f"Save img at {temp_dir}/{key}_{i:06}.png")
        plt.close()
    # use ffmpeg to create video from image sorted by file name intemp_dir, 3Mbps video bitrate
    # ffmpeg_cmd = f"ffmpeg -r 3 -i {temp_dir}/{key}_%06d.png  -pix_fmt yuv420p  -b:v 3M {temp_dir}/../{key}.mp4"
    # print(ffmpeg_cmd)
    # os.system(ffmpeg_cmd)
    # print("video saved at", f"{temp_dir}/../{key}.mp4")
    # remove images

    # rm_cmd = f"rm {temp_dir}/{key}_*.png"
    # os.system(rm_cmd)
# # remove the empty dir
# rm_cmd = f"rmdir {temp_dir}"
# os.system(rm_cmd)
if not args.no_wandb:
    wandb.finish()
