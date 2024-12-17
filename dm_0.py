import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import torch
from simple_dm import DDPM_Scheduler, SimpleDiffusion
from timm.utils import ModelEmaV3  # pip install timm
from pc_dataset import PointCloudDataset


def train(model, dataloader, optimizer, epochs, device, is_log_wandb, args):
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        # train_state.step = checkpoint['step']
        # train_state.best_val = checkpoint['best_val']
        # model_ema.load_state_dict(checkpoint['model_ema'])

    model.train()
    tr = trange(start_epoch, epochs)
    for epoch in tr:
        losses = []
        for batch in dataloader:

            batch = batch.to(device).view(batch.size(0), -1)
            optimizer.zero_grad()
            # calculate loss using built-in loss function
            model_output = model(batch, [0]*batch.size(0))
            loss = nn.MSELoss()(model_output, batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        tr.set_description_str(f"Loss = {np.mean(losses)}")
        if is_log_wandb:
            wandb.log({"loss":  np.mean(losses), "epoch": epoch})
            # print(f"Epoch {epoch}: Loss = {np.mean(losses)}")
            # if every epochs//10, log the sampled point cloud
            if epoch % (epochs // 10) == 0:
                sampled_point_cloud = model_output[0].detach(
                ).cpu().numpy().reshape(-1, 3)
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
                    # 'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    # 'step': train_state.step,
                    # 'best_val': train_state.best_val,
                    # 'model_ema': model_ema.state_dict() if model_ema else {},
                    'args': args
                }
                checkpoint_path = 'checkpoint-latest.pth'
                wandb_dir = wandb.run.dir
                torch.save(checkpoint_dict, os.join(
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
            x = model(x, [t])
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
    parser.add_argument("--wandb", action='store_false', help="Log to wandb")
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
    parser.add_argument("--ema_decay", type=float,
                        default=0.999, help="Exponential moving average decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# Main Training Function


def main():
    args = parse_args()
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    # Initialize wandb, if enabled
    if args.wandb:
        # make sure the wandb dir is in /data/palakons/wandb_scratch
        wandb.init(project="point_cloud_diffusion", config=vars(
            args), dir="/data/palakons/wandb_scratch")

    # Load Dataset and Normalization
    dataset = PointCloudDataset(args.data_dir, args.N, args.M)
    if args.wandb:
        wandb.config.update({"mu": dataset.mu, "sigma": dataset.sigma, "num_params": sum(
            p.numel() for p in model.parameters())})
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Optimizer, and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "SimpleDiffusion":
        # model = SimpleDiffusion(
        #     args.N, args.n_hidden_layers, args.hidden_dim).to(device)
        model = SimpleDiffusion(
            args.N,  args.hidden_dim, args.num_time_steps,  args.n_hidden_layers).to(device)
    # elif args.model == "SimpleDiffusion":
    #     model = PVCNNDiffusion(args.N, resolution, diffusion_steps).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = DDPM_Scheduler(num_time_steps=args.num_time_steps)
    scheduler.alpha = scheduler.alpha.to(device)
    model_ema = ModelEmaV3(model, decay=args.ema_decay)
    criterian = nn.MSELoss(reduction='mean')

    # Train Model
    train(model, dataloader, optimizer, args.epochs, device, args.wandb, args)

    # train(model, dataloader, optimizer, scheduler, model_ema, args.epochs,
    #       criterian, args.seed, device, not args.wandb, args)

    # Sample New Point Cloud

    sampled_point_cloud = sample(model, args.N)

    # sacle back to original scale
    sampled_point_cloud = sampled_point_cloud.cpu().numpy() * \
        dataset.sigma + dataset.mu
    if args.wandb:
        # plot 3d and log to wandb
        # import plotly.graph_objects as go

        # #assert one input file
        # #log the GT point cloud in the same fig
        # gt_point_cloud = dataset[np.random.randint(len(dataset))].numpy()
        # gt_point_cloud = gt_point_cloud * dataset.sigma + dataset.mu
        # loss_value = nn.MSELoss()(torch.tensor(sampled_point_cloud), torch.tensor(gt_point_cloud)).item()
        # fig = go.Figure(data=[go.Scatter3d(x=sampled_point_cloud[:,0], y=sampled_point_cloud[:,1], z=sampled_point_cloud[:,2], mode='markers', name=f"sampled: loss={loss_value}")])
        # fig.add_trace(go.Scatter3d(x=gt_point_cloud[:,0], y=gt_point_cloud[:,1], z=gt_point_cloud[:,2], mode='markers',name="GT"))
        # wandb.log({"sampled_point_cloud": fig})

        wandb.finish()


if __name__ == "__main__":
    main()
