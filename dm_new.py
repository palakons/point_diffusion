import argparse
from datetime import datetime
import os
import random
from pathlib import Path

import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pc_dataset import PointCloudDataset
from simple_dm import DDPM_Scheduler, PointCloudDiffusionModel, SimpleDiffusion
from timm.utils import ModelEmaV3  # pip install timm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import wandb


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Dataset for loading point clouds from text files


# Diffusion Training and Sampling Functions
def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    model_ema,
    epochs,
    criterian,
    seed,
    device,
    is_log_wandb,
    args,
):
    start_epoch = args.start_epoch
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        # step = checkpoint['step']
        # train_state.best_val = checkpoint['best_val']
        model_ema.load_state_dict(checkpoint["model_ema"])
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)

    tr = trange(start_epoch, epochs)
    for epoch in tr:
        losses = []
        for batch in dataloader:
            model.train()

            loss = model(batch, mode="train")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            losses.append(loss_value)

        tr.set_description(f"Epoch Loss: {np.mean(losses):.4f}")

        if model_ema and epoch % args.ema_update_freq == 0:
            model_ema.update(model)
        if epoch % args.visualize_freq == 0:

            output, all_outputs = visualize(
                args,
                model,
                dataloader,
                num_batches=1, 
                ever_k=args.num_time_steps_to_visualize,
                # output_dir="/data/palakons/dataset/vis",
                output_dir="vis",
            )

            if is_log_wandb:
                sampled_point_cloud = (
                    output.points_padded().detach().cpu().numpy().reshape(-1, 3)
                )

                gt = batch[0].detach().cpu().numpy().reshape(-1, 3)

                # print("sampled_point_cloud.shape", sampled_point_cloud.shape)
                # print("gt.shape", gt.shape)

                criteria = nn.MSELoss(reduction="mean")
                newloss = criteria(output.points_padded().reshape(-1,3).to("cpu") , batch[0].reshape(-1,3).to("cpu")).item()
                # print("newloss", newloss)
                wandb.log({"sampling_loss": newloss, "epoch": epoch})
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=sampled_point_cloud[:, 0],
                            y=sampled_point_cloud[:, 1],
                            z=sampled_point_cloud[:, 2],
                            mode="markers",
                            name=f"epoch={epoch} train loss={loss.item():.4f} sample loss={newloss:.4f}",
                        )
                    ]
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="markers", name="GT"
                    )
                )
                wandb.log({"Predicted": fig, "epoch": epoch})

                # new fig, with all outputs as subfig (multiple point clouds in one plot)
                for i, o in enumerate(all_outputs):
                    fig = go.Figure()
                    o_pc = o.points_padded().detach().cpu().numpy().reshape(-1, 3)
                    fig.add_trace(
                        go.Scatter3d(
                            x=o_pc[:, 0],
                            y=o_pc[:, 1],
                            z=o_pc[:, 2],
                            mode="markers",
                            # name=f"epoch={epoch} loss={loss.item()}",
                        )
                    )
                    # fig.add_trace(
                    #     go.Scatter3d(
                    #         x=gt[:, 0],
                    #         y=gt[:, 1],
                    #         z=gt[:, 2],
                    #         mode="markers",
                    #         name="GT",
                    #     )
                    # )

                    wandb.log({f"Evo {i:03}": fig, "epoch": epoch})

        if is_log_wandb:
            # assume equal weight for each batch
            wandb.log(
                {
                    "train_loss": np.mean(losses),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if epoch % args.checkpoint_freq == 0:
                checkpoint_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    # 'step': train_state.step,
                    # 'best_val': train_state.best_val,
                    "model_ema": model_ema.state_dict() if model_ema else {},
                    "args": args,
                }
                torch.save(
                    checkpoint_dict,
                    os.path.join(wandb.run.dir, "checkpoint-latest.pth"),
                )


@torch.no_grad()
def visualize(
    args,
    model,
    dataloader_vis,
    num_batches,
    ever_k=10,
    output_dir: str = "vis",
    is_wandb_log=True,
):
    model.eval()
    # add current time to output_dir
    output_dir += f"/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir: Path = Path(output_dir)
    (output_dir / "pointclouds").mkdir(exist_ok=True, parents=True)
    (output_dir / "evolutions").mkdir(exist_ok=True, parents=True)

    output, all_outputs = model(
        None,
        mode="sample",
        return_sample_every_n_steps=ever_k,
        num_inference_steps=args.num_time_steps,
        disable_tqdm=True,
    )  # always outptu 1 sameple
    # print("len(all_outputs)", len(all_outputs))
    # Save the output point cloud as csv
    output_pc = output.points_padded().detach().cpu().numpy()
    np.savetxt(
        output_dir / "pointclouds" / f"output_final.csv",
        output_pc.reshape(-1, 3),
        delimiter=",",
    )

    # Save the evolution of the point cloud as csv
    for i, o in enumerate(all_outputs):
        o_pc = o.points_padded().detach().cpu().numpy()
        np.savetxt(
            output_dir / "evolutions" / f"output_x_{i*ever_k:06}.csv",
            o_pc.reshape(-1, 3),
            delimiter=",",
        )
    return output, all_outputs


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
        description="Train a diffusion model for point clouds"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/palakons/dataset/astyx_blank/scene/",
        help="Directory containing text files",
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--N", type=int, default=1000, help="Number of points in each point cloud"
    )
    parser.add_argument(
        "--M", type=int, default=1, help="Number of point cloud scenes to load"
    )
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--no_wandb", action="store_true", help="Log to wandb")
    parser.add_argument(
        "--n_hidden_layers", type=int, default=1, help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--model", type=str, default="PointCloudDiffusionModel", help="Model class"
    )
    parser.add_argument(
        "--num_time_steps", type=int, default=100, help="Number of time steps"
    )
    parser.add_argument( "--num_time_steps_to_visualize", type=int, default=10, help="Number of time steps to visualize")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="Exponential moving average decay",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")

    parser.add_argument("--beta_start", type=float, default=1e-5, help="Beta start")
    parser.add_argument("--beta_end", type=float, default=8e-3, help="Beta end")
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="Beta schedule"
    )
    parser.add_argument(
        "--ema_update_freq",
        type=int,
        default=20,
        help="Exponential moving average update frequency",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=20, help="Check point frequency"
    )
    parser.add_argument(
        "--visualize_freq", type=int, default=20, help="Visualize frequency"
    )
    return parser.parse_args()


# Main Training Function


def main():
    args = parse_args()
    print(args)
    if args.config:
        import json

        with open(args.config, "r") as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    # Initialize wandb, if enabled
    if not args.no_wandb:
        # make sure the wandb dir is in /data/palakons/wandb_scratch
        wandb.init(
            project="point_cloud_diffusion",
            config=vars(args),
            dir="/data/palakons/wandb_scratch",
        )

    # Load Dataset and Normalization
    dataset = PointCloudDataset(args.data_dir, args.N, args.M)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Optimizer, and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "SimpleDiffusion":
        model = SimpleDiffusion(
            args.N, args.n_hidden_layers, args.num_time_steps, args.hidden_dim
        ).to(device)
    elif args.model == "PointCloudDiffusionModel":
        model = PointCloudDiffusionModel(
            args.beta_start,
            args.beta_end,
            args.beta_schedule,
            args.N,
            args.hidden_dim,
            args.n_hidden_layers,
            require_time_embedding=True,
            scheduler="ddpm",
            criteria=nn.MSELoss(reduction="mean"),
        ).to(device)
    if not args.no_wandb:
        wandb.config.update(
            {
                "mu": dataset.mu,
                "sigma": dataset.sigma,
                "num_params": sum(p.numel() for p in model.parameters()),
            }
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = DDPM_Scheduler(num_time_steps=args.num_time_steps)
    scheduler.alpha = scheduler.alpha.to(device)
    model_ema = ModelEmaV3(model, decay=args.ema_decay)
    criterian = nn.MSELoss(reduction="mean")

    # Train Model
    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        model_ema,
        args.epochs,
        criterian,
        args.seed,
        device,
        not args.no_wandb,
        args,
    )

    # Sample New Point Cloud

    # sampled_point_cloud = sample(model, args.N)

    # #sacle back to original scale
    # sampled_point_cloud = sampled_point_cloud.cpu().numpy() * dataset.sigma + dataset.mu
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
