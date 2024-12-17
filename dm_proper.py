import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from timm.utils import ModelEmaV3  # pip install timm
from simple_dm import PointCloudDiffusionModel, SimpleDiffusion, DDPM_Scheduler
import random


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
# Dataset for loading point clouds from text files


class PointCloudDataset(Dataset):
    def __init__(self, data_dir, N, M=1):
        self.files = [os.path.join(
            data_dir, f"{i}/{i:06}.txt") for i in range(min(M, 546))]
        self.N = N
        self.mu, self.sigma = None, None  # Normalization parameters
        self.data = []
        for fname in self.files:
            new_data = np.loadtxt(fname, usecols=(0, 1, 2), skiprows=2)
            # if the data is too long, remove the extra points
            if len(new_data) > self.N:
                # print how many points are removed
                print("remove", len(new_data) - self.N)
                new_data = new_data[:self.N]
            else:  # randomly sample the data rows to make it N
                print("add", self.N - len(new_data))
                new_data = np.vstack(
                    [new_data, new_data[np.random.choice(len(new_data), self.N - len(new_data))]])
            assert len(new_data) == self.N, "Data length must be N"
            self.data.append(new_data)

        self.all_points = np.vstack(self.data)
        self.mu, self.sigma = self.compute_normalization()
        self.data = [(d - self.mu) / self.sigma for d in self.data]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def compute_normalization(self):
        self.mu = np.mean(self.all_points, axis=0)
        self.sigma = np.std(self.all_points, axis=0)
        return self.mu, self.sigma


# Diffusion Training and Sampling Functions
def train(model, dataloader, optimizer, scheduler, model_ema, epochs, criterian, seed, device, is_log_wandb, args):
    start_epoch = args.start_epoch
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        # step = checkpoint['step']
        # train_state.best_val = checkpoint['best_val']
        model_ema.load_state_dict(checkpoint['model_ema'])
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    model.train()
    tr = trange(start_epoch, epochs)
    for epoch in tr:
        losses = []
        for batch in dataloader:
            batch_size = batch.size(0)
            x0 = batch.to(device).view(batch.size(0), -1)
            e = torch.randn_like(x0, requires_grad=False)
            # print device of x0 and e
            t = torch.randint(0, scheduler.num_time_steps,
                              (batch_size,), device=device)
            a = scheduler.alpha[t].view(batch_size, 1).cuda()
            # print("cheduler.alpha[t]",scheduler.alpha[t])
            # print("a",a.shape)
            x = (torch.sqrt(a)*x0) + (torch.sqrt(1-a)*e)
            # calculate loss using built-in loss function

            predicted_noise = model(x, t)
            optimizer.zero_grad()
            loss = criterian(predicted_noise, e)  # predict e, reduction='mean'
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            model_ema.update(model)
        tr.set_description(f"Loss: {np.mean(losses):.4f}")
        if is_log_wandb:
            # assume equal weight for each batch
            wandb.log({"loss":  np.mean(losses), "epoch": epoch})
            if epoch % (epochs // 10) == 0:

                x = (torch.sqrt(a[0,])*x0[0,]) + \
                    (torch.sqrt(1-a[0,])*predicted_noise[0,])

                sampled_point_cloud = x.detach().cpu().numpy().reshape(-1, 3)
                gt = batch[0].detach().cpu().numpy().reshape(-1, 3)
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Scatter3d(x=sampled_point_cloud[:, 0], y=sampled_point_cloud[:, 1],
                                z=sampled_point_cloud[:, 2], mode='markers', name=f"epoch={epoch} loss={loss.item()}")])
                fig.add_trace(go.Scatter3d(
                    x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode='markers', name="GT"))
                wandb.log({"sampled_point_cloud": fig})

                checkpoint_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    # 'step': train_state.step,
                    # 'best_val': train_state.best_val,
                    'model_ema': model_ema.state_dict() if model_ema else {},
                    'args': args
                }
                checkpoint_path = 'checkpoint-latest.pth'
                wandb_dir = wandb.run.dir
                torch.save(checkpoint_dict, os.path.join(
                    wandb_dir, checkpoint_path))


def sample(model, N, steps=100):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, N * 3)  # Start with noise
        # print device of x and model
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        # input = input.to(self.weight.device)
        for t in trange(steps):
            x = model(x, t)
        return x.view(N, 3)

# Argument Parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for point clouds")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing text files")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of points in each point cloud")
    parser.add_argument("--M", type=int, default=1,
                        help="Number of point cloud scenes to load")
    parser.add_argument("--config", type=str,
                        help="Path to config file (optional)")
    parser.add_argument("--no_wandb", action='store_true', help="Log to wandb")
    parser.add_argument("--n_hidden_layers", type=int,
                        default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_dim", type=int,
                        default=128, help="Hidden layer dimension")
    parser.add_argument("--checkpoint", type=str,
                        default=None, help="Path to checkpoint file")
    parser.add_argument("--model", type=str,
                        default="SimpleDiffusion", help="Model class")
    parser.add_argument("--num_time_steps", type=int,
                        default=100, help="Number of time steps")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="Exponential moving average decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start_epoch", type=int,
                        default=0, help="Start epoch")

    parser.add_argument("--beta_start", type=float,
                        default=1e-5, help="Beta start")
    parser.add_argument("--beta_end", type=float,
                        default=8e-3, help="Beta end")
    parser.add_argument("--beta_schedule", type=str,
                        default='linear', help="Beta schedule")

    return parser.parse_args()

# Main Training Function


def main():
    args = parse_args()
    print(args)
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    # Initialize wandb, if enabled
    if not args.no_wandb:
        # make sure the wandb dir is in /data/palakons/wandb_scratch
        wandb.init(project="point_cloud_diffusion", config=vars(
            args), dir="/data/palakons/wandb_scratch")

    # Load Dataset and Normalization
    dataset = PointCloudDataset(args.data_dir, args.N, args.M)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Optimizer, and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "SimpleDiffusion":
        model = SimpleDiffusion(
            args.N, args.n_hidden_layers, args.num_time_steps, args.hidden_dim).to(device)
    elif args.model == "PointCloudDiffusionModel":
        model = PointCloudDiffusionModel(args.beta_start, args.beta_end, args.beta_schedule,
                                         args.N, args.hidden_dim, args.n_hidden_layers, require_time_embedding=True, scheduler="ddpm", criteria=nn.MSELoss(reduction="mean")).to(device)
    if not args.no_wandb:
        wandb.config.update({"mu": dataset.mu, "sigma": dataset.sigma, "num_params": sum(
            p.numel() for p in model.parameters())})

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = DDPM_Scheduler(num_time_steps=args.num_time_steps)
    scheduler.alpha = scheduler.alpha.to(device)
    model_ema = ModelEmaV3(model, decay=args.ema_decay)
    criterian = nn.MSELoss(reduction='mean')

    # Train Model
    train(model, dataloader, optimizer, scheduler, model_ema, args.epochs,
          criterian, args.seed, device, not args.no_wandb, args)

    # Sample New Point Cloud

    # sampled_point_cloud = sample(model, args.N)

    # #sacle back to original scale
    # sampled_point_cloud = sampled_point_cloud.cpu().numpy() * dataset.sigma + dataset.mu
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
