import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from datasets import PointCloudDataset
from pvcnn import PVCNNDiffusion

# Diffusion Training and Sampling Functions
def train(model, dataloader, optimizer, epochs, device, is_log_wandb,args): 
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
    for epoch in trange(start_epoch, epochs):
        losses = []
        for batch in dataloader:

            batch = batch.to(device).view(batch.size(0), -1)
            optimizer.zero_grad()
            #calculate loss using built-in loss function
            model_output = model(batch, 0)
            loss = nn.MSELoss()(model_output, batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if is_log_wandb:
            wandb.log({"loss":  np.mean(losses), "epoch": epoch})
            # print(f"Epoch {epoch}: Loss = {np.mean(losses)}")
            #if every epochs//10, log the sampled point cloud
            if epoch % (epochs // 10) == 0:
                sampled_point_cloud = model_output[0].detach().cpu().numpy().reshape(-1, 3)
                gt = batch[0].detach().cpu().numpy().reshape(-1, 3)
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Scatter3d(x=sampled_point_cloud[:,0], y=sampled_point_cloud[:,1], z=sampled_point_cloud[:,2], mode='markers', name=f"epoch={epoch} loss={loss.item()}")])
                fig.add_trace(go.Scatter3d(x=gt[:,0], y=gt[:,1], z=gt[:,2], mode='markers',name="GT"))
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
                torch.save(checkpoint_dict, os.join(wandb_dir, checkpoint_path))

def sample(model, N, steps=100):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, N * 3)  # Start with noise
        #print device of x and model
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        # input = input.to(self.weight.device)
        for t in trange(steps):
            x = model(x, t)
        return x.view(N, 3)

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model for point clouds")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing text files")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--N", type=int, default=1000, help="Number of points in each point cloud")
    parser.add_argument("--M", type=int, default=1, help="Number of point cloud scenes to load")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--wandb", action='store_false', help="Log to wandb" )
    parser.add_argument("--n_hidden_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--model", type=str, default="SimpleDiffusion", help="Model class")
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
        #make sure the wandb dir is in /data/palakons/wandb_scratch
        wandb.init(project="point_cloud_diffusion", config=vars(args), dir="/data/palakons/wandb_scratch")

    # Load Dataset and Normalization
    dataset = PointCloudDataset(args.data_dir, args.N, args.M)
    wandb.config.update({"mu": dataset.mu, "sigma": dataset.sigma,"num_params": sum(p.numel() for p in model.parameters())})
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Optimizer, and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "PVCNNDiffusion":
        model = PVCNNDiffusion(args.N, resolution, diffusion_steps).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    # Train Model
    train(model, dataloader, optimizer, args.epochs, device, args.wandb,args)

    # Sample New Point Cloud

    sampled_point_cloud = sample(model, args.N)
    
    #sacle back to original scale
    sampled_point_cloud = sampled_point_cloud.cpu().numpy() * dataset.sigma + dataset.mu
    if args.wandb:
        
        wandb.finish()

if __name__ == "__main__":
    main()
