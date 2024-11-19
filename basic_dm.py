import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
from tqdm import trange

# Dataset of n-dimensional point clouds
class PointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=10):
        super().__init__()
        self.data = torch.randn(num_samples, num_points*3) * torch.tensor([10,100,5]) # Random point clouds
        # self.data = torch.tensor([[6.,4.,2.]*num_points]* num_samples)

        #normalize data, extract mean and std
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0) if num_samples > 1 else torch.ones_like(self.mean)
        print("mean",self.mean,"std",self.std)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # return the normalized data
        return self.data[idx] / self.std - self.mean

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
    def __init__(self, data_dim, time_embedding_dim=16):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_embedding_dim, 128),
            nn.ReLU(),
        )
        for _ in range(6):
            self.net.add_module("layer", nn.Linear(128, 128))
            self.net.add_module("relu", nn.ReLU())
        self.net.add_module("layer2", nn.Linear(128, data_dim))
    
    def forward(self, x, t):
        # Get time embeddings
        t_emb = get_sinusoidal_time_embedding(t, self.time_embedding_dim)
        # Concatenate x and t_emb
        x_t = torch.cat([x, t_emb], dim=-1)
        # Pass through network
        return self.net(x_t)

# Training function
def train(model, dataloader, optimizer, scheduler, num_epochs=10, device='cpu'):
    model.train()
    tr=trange(num_epochs)
    for epoch in tr:
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
            loss = nn.functional.mse_loss(noise_pred, noise)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        tr.set_description(f"loss: {sum(losses)/len(losses):.4f}")   
        # print(f"Epoch {epoch+1} completed")

# Sampling function
@torch.no_grad()
def sample(model, scheduler, sample_shape, device='cpu'):
    model.eval()
    num_timesteps = scheduler.config.num_train_timesteps
    # Start from pure noise
    x = torch.randn(sample_shape).to(device)
    for t in reversed(range(num_timesteps)):
        timesteps = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        # Predict noise
        noise_pred = model(x, timesteps)
        # Compute previous noisy sample x_t -> x_{t-1}
        x = scheduler.step(noise_pred, t, x)['prev_sample']
    return x

# Set up device, dataset, dataloader, model, optimizer, and scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device",device)

dataset = PointCloudDataset( num_samples=1, num_points=1)
print("data",dataset[:])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

data_dim = dataset[0].shape[0]

model = SimpleDiffusionModel(data_dim=data_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Train the model
train(model, dataloader, optimizer, scheduler, num_epochs=100000, device=device)

# Sample from the model
samples = sample(model, scheduler, sample_shape=(1000, data_dim), device=device)
# revert the normalization
# print(samples.mean(dim=0))
samples = samples * dataset.std.to(device) + dataset.mean.to(device) 
print(samples.mean(dim=0))
# calculate teh loss, appropriate to unordered point clouds,  maybe chamfer distance

from pytorch3d.loss import chamfer_distance


loss, _ = chamfer_distance(dataset[:].unsqueeze(0).to(device)* dataset.std.to(device) + dataset.mean.to(device) , samples.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device))  
print("Chamfer Distance:", loss.item())


