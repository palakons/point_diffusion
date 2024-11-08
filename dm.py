import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

# Dataset for loading point clouds from text files
class PointCloudDataset(Dataset):
    def __init__(self, data_dir, N,M=1):
        self.files = [os.path.join(data_dir, f"{i}/{i:06}.txt")  for i in range(min(M,546))]
        self.N = N
        self.mu, self.sigma = None, None  # Normalization parameters
        self.data =[]
        for fname in self.files:
            new_data = np.loadtxt(fname, usecols=(0, 1, 2), skiprows=2)
            #if the data is too long, remove the extra points
            if len(new_data) > self.N:
                #print how many points are removed
                print("remove",len(new_data) - self.N)
                new_data = new_data[:self.N]
            else: #randomly sample the data rows to make it N
                print("add",self.N - len(new_data))
                new_data = np.vstack([new_data, new_data[np.random.choice(len(new_data), self.N - len(new_data))]])
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

# Simple diffusion model
class SimpleDiffusion(nn.Module):
    def __init__(self, N):
        super(SimpleDiffusion, self).__init__()
        self.fc1 = nn.Linear(N * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, N * 3)

    def forward(self, x, t):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Diffusion Training and Sampling Functions
def train(model, dataloader, optimizer, epochs, device, is_log_wandb): 
    model.train()
    for epoch in trange(epochs):
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
            print(f"Epoch {epoch}: Loss = {np.mean(losses)}")
            #if every epochs//10, log the sampled point cloud
            if epoch % (epochs // 10) == 0:
                sampled_point_cloud = model_output[0].detach().cpu().numpy().reshape(-1, 3)
                gt = batch[0].detach().cpu().numpy().reshape(-1, 3)
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Scatter3d(x=sampled_point_cloud[:,0], y=sampled_point_cloud[:,1], z=sampled_point_cloud[:,2], mode='markers', name=f"epoch={epoch} loss={loss.item()}")])
                fig.add_trace(go.Scatter3d(x=gt[:,0], y=gt[:,1], z=gt[:,2], mode='markers',name="GT"))
                wandb.log({"sampled_point_cloud": fig})

def sample(model, N, steps=100):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, N * 3)  # Start with noise
        #print device of x and model
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        # input = input.to(self.weight.device)
        for t in range(steps):
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, Optimizer, and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDiffusion(args.N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train Model
    train(model, dataloader, optimizer, args.epochs, device, args.wandb)

    # Sample New Point Cloud

    sampled_point_cloud = sample(model, args.N)
    
    #sacle back to original scale
    sampled_point_cloud = sampled_point_cloud.cpu().numpy() * dataset.sigma + dataset.mu
    if args.wandb:
        #plot 3d and log to wandb
        import plotly.graph_objects as go

        #assert one input file
        #log the GT point cloud in the same fig
        gt_point_cloud = dataset[np.random.randint(len(dataset))].numpy()
        gt_point_cloud = gt_point_cloud * dataset.sigma + dataset.mu
        loss_value = nn.MSELoss()(torch.tensor(sampled_point_cloud), torch.tensor(gt_point_cloud)).item()
        fig = go.Figure(data=[go.Scatter3d(x=sampled_point_cloud[:,0], y=sampled_point_cloud[:,1], z=sampled_point_cloud[:,2], mode='markers', name=f"sampled: loss={loss_value}")])
        fig.add_trace(go.Scatter3d(x=gt_point_cloud[:,0], y=gt_point_cloud[:,1], z=gt_point_cloud[:,2], mode='markers',name="GT"))
        wandb.log({"sampled_point_cloud": fig})
        
        wandb.finish()

if __name__ == "__main__":
    main()
