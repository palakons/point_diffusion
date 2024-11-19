import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from tqdm import tqdm


class PointCloudDiffusionModel(nn.Module):

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        n_pc_points: int,
        hidden_dim: int,
        n_hidden_layers: int,
        require_time_embedding: bool = True,
        scheduler: Optional[str] = "ddpm",
        criteria: nn.Module = nn.MSELoss(),
        device: str = "cuda",
    ):
        super(PointCloudDiffusionModel, self).__init__()

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        scheduler_kwargs.update(
            dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule)
        )
        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
        }
        # this can be changed for inference
        self.scheduler = self.schedulers_map[scheduler]

        # Create point cloud model for processing point cloud at each diffusion step
        # the model is
        self.point_cloud_block_model = SimpleDiffusionBlock(
            n_pc_points,
            n_hidden_layers,
            self.scheduler.config.num_train_timesteps,  # default at 1k
            hidden_dim,
        )
        self.point_cloud_block_model.to(device)
        self.device = device
        self.require_time_embedding = require_time_embedding
        self.criteria = criteria
        self.n_pc_points = n_pc_points

    def forward_train(
        self,
        pc: Pointclouds,
    ):

        B, N, D = pc.shape

        # Sample random noise
        pc = pc.to(self.device)
        pc = pc.reshape(B, -1)

        noise = torch.randn_like(pc, device=self.device)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
            dtype=torch.long,
        )
        # timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
        #     device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(pc, noise, timestep)

        # # Conditioning
        # x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
        #                                              image_rgb=image_rgb, mask=mask, t=timestep)

        # Forward
        noise_pred = self.point_cloud_block_model(x_t, timestep)

        # Check
        assert (
            noise_pred.shape == noise.shape
        ), f"Expected {noise.shape} but got {noise_pred.shape}"

        # Loss
        loss = self.criteria(noise_pred, noise)

        return loss

    @torch.no_grad()
    def forward_sample(
        self,
        num_points: int,
        num_inference_steps: int,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        scheduler = self.scheduler

        # Get the size of the noise
        N = num_points

        # Sample noise
        x_t = torch.randn(1, N * 3, device=self.device)

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = return_sample_every_n_steps > 0
        progress_bar = tqdm(
            scheduler.timesteps.to(self.device),
            desc=f"Sampling ({x_t.shape})",
            disable=disable_tqdm,
        )
        for i, t in enumerate(progress_bar):

            # # Conditioning
            # x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
            #                                              image_rgb=image_rgb, mask=mask, t=t)

            # Forward
            # noise_pred = self.point_cloud_model(
            #     x_t_input, t.reshape(1).expand(B))
            input_xt = torch.cat([x_t], dim=1).to(self.device)
            input_t = torch.Tensor([t]).long().to(self.device)

            noise_pred = self.point_cloud_block_model(input_xt, input_t)
            # print("noise_pred",noise_pred.shape) noise_pred torch.Size([1, 3000])
            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            # print(return_all_outputs ,
            #     i % return_sample_every_n_steps == 0, i == len(scheduler.timesteps) - 1)
            if return_all_outputs and (
                i % return_sample_every_n_steps == 0
                or i == len(scheduler.timesteps) - 1
            ):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling

        points = x_t.reshape(-1, self.n_pc_points, 3)
        output = self.tensor_to_point_cloud(points)
        all_outputs = [o.reshape(-1, self.n_pc_points, 3) for o in all_outputs]
        if return_all_outputs:
            # ( sample_steps, N, D)
            all_outputs = torch.stack(all_outputs, dim=0)
            # print("all_outputs", all_outputs.shape) #ll_outputs torch.Size([11, 1, 1000, 3])   
            all_outputs = [
                self.tensor_to_point_cloud(o)
                for o in all_outputs
            ]
        # print("len(all_outputs)", len(all_outputs)) #len(all_outputs) 11
        return (output, all_outputs) if return_all_outputs else output

    def tensor_to_point_cloud(self, x: Tensor):
        return Pointclouds(points=x)

    def forward(self, batch: FrameData, mode: str = "train", **kwargs):
        """A wrapper around the forward method for training and inference"""
        if isinstance(
            batch, dict
        ):  # fixes a bug with multiprocessing where batch becomes a dict
            # it really makes no sense, I do not understand it
            batch = FrameData(**batch)
        if mode == "train":
            return self.forward_train(pc=batch, **kwargs)
        elif mode == "sample":
            return self.forward_sample(num_points=self.n_pc_points,  **kwargs)
        else:
            raise NotImplementedError()


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings
        # print("Embeddings shape", self.embeddings.shape)

    def forward(self, x, t):
        self.embeddings = self.embeddings.to(x.device)
        embeds = self.embeddings[t].to(x.device)
        # print("Embeds shape", embeds.shape)
        return embeds[:, :]


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
        self.num_time_steps = num_time_steps

    def forward(self, t):
        return self.beta[t], self.alpha[t]


class BetaDerivatives:
    def __init__(self, betas, dtype=torch.float32):
        """Take in betas and pre-compute the dependent values to use in forward/ backward pass.

        Values are precomputed for all timesteps so that they can be used as and
        when required.
        """
        self.np_betas = betas
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.concat(
            [torch.Tensor([1.0]), self.alphas_cumprod[:-1]], axis=0
        )

        # calculations required for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

    def _gather(self, a, t):
        """
        Utility function to extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1] for broadcasting.
        """
        return torch.reshape(torch.gather(a, t), [-1, 1])


class DiffusionForward(BetaDerivatives):
    """
    Forward pass of the diffusion model.
    """

    def __init__(self, betas):
        super().__init__(betas)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward pass - sample of diffused data at time t.
        """
        if noise is None:
            noise = tf.random.normal(shape=x_start.shape)
        p1 = self._gather(self.sqrt_alphas_cumprod, t) * x_start
        p2 = self._gather(self.sqrt_one_minus_alphas_cumprod, t) * noise
        return p1 + p2


def get_beta_schedule(schedule_type, beta_start, beta_end, num_diffusion_timesteps):
    """
    Generate a beta schedule for the diffusion process.

    Args:
        schedule_type (str): The type of schedule ("linear", "quadratic", "cosine").
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        num_diffusion_timesteps (int): Number of timesteps in the diffusion process.

    Returns:
        Tensor: A 1D tensor of shape (num_diffusion_timesteps,) containing the beta values.
    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_type == "quadratic":
        return (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps) ** 2
        )
    elif schedule_type == "cosine":
        timesteps = torch.linspace(
            0, num_diffusion_timesteps, num_diffusion_timesteps + 1
        )
        alphas = (
            torch.cos(
                (timesteps / num_diffusion_timesteps + 0.008) / 1.008 * torch.pi / 2
            )
            ** 2
        )
        alphas = alphas / alphas[0]  # Normalize to start at 1
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clip(betas, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Get sinusoidal embeddings for timesteps.

    Args:
        timesteps (Tensor): 1D Tensor of shape (batch_size,) containing timesteps.
        embedding_dim (int): Dimension of the embedding.

    Returns:
        Tensor: A tensor of shape (batch_size, embedding_dim) containing the timestep embeddings.
    """
    assert (
        len(timesteps.shape) == 1
    ), "Timesteps should be a 1D tensor of shape (batch_size,)"
    half_dim = embedding_dim // 2
    timesteps = timesteps.float()

    # Compute frequencies for the sinusoidal embeddings
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    ).to(timesteps.device)
    # Create sinusoidal embeddings
    sinusoidal_embedding = torch.outer(
        timesteps, frequencies
    )  # Shape: (batch_size, half_dim)
    sinusoidal_embedding = torch.cat(
        [torch.sin(sinusoidal_embedding), torch.cos(sinusoidal_embedding)], dim=-1
    )

    # If embedding_dim is odd, pad the result
    if embedding_dim % 2 != 0:
        sinusoidal_embedding = F.pad(sinusoidal_embedding, (0, 1))

    return sinusoidal_embedding


def data_generator_forward(x, gdb):
    tstep = torch.uniform(
        shape=(x.shape[0],), minval=0, maxval=num_diffusion_timesteps, dtype=torch.int32
    )
    noise = torch.normal(shape=x.shape, dtype=x.dtype)
    noisy_out = gdb.q_sample(x, tstep, noise)
    return ((noisy_out, get_timestep_embedding(tstep, 128)), noise)


class BareDiffusion(nn.Module):
    def __init__(self, N, n_hidden_layers=1, hidden_dim=128):
        super(SimpleDiffusion, self).__init__()
        self.fc1 = nn.Linear(N * 3, hidden_dim)
        # self.fc2 = nn.Linear(128, 128)
        self.hidden = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, N * 3)

    def forward(self, x, t):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.fc3(x)


class SimpleDiffusionBlock(nn.Module):
    def __init__(
        self,
        N,
        hidden_dim,
        num_time_steps,
        n_hidden_layers=1,
        require_time_embedding=True,
    ):
        super(SimpleDiffusionBlock, self).__init__()
        self.fc1 = nn.Linear(N * 3, hidden_dim)
        self.hidden = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, N * 3)
        self.timestep_embedding = SinusoidalEmbeddings(num_time_steps, embed_dim=N * 3)
        self.require_time_embedding = require_time_embedding

    def forward(self, x, t):  # 1.0133

        if self.require_time_embedding:
            x = x.reshape(x.shape[0], -1)
            t_embedding = self.timestep_embedding(x, t)
            x = x + t_embedding
        # Forward pass through the network
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.fc3(x)
        return x


class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        N,
        hidden_dim,
        num_time_steps,
        n_hidden_layers=1,
        require_time_embedding=True,
    ):
        super(SimpleDiffusion, self).__init__()
        self.fc1 = nn.Linear(N * 3, hidden_dim)
        self.hidden = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, N * 3)
        self.timestep_embedding = SinusoidalEmbeddings(num_time_steps, embed_dim=N * 3)
        self.require_time_embedding = require_time_embedding

    def forward_2(self, x, t):  # 1.0143

        x = F.relu(self.fc1(x))
        # for layer in self.hidden:
        #     x = F.relu(layer(x))
        x = self.fc3(x)
        return x

    def forward(self, x, t):  # 1.0133

        if self.require_time_embedding:
            t_embedding = self.timestep_embedding(x, t)
            # print("x,t",x.size(), t_embedding.size())
            x = x + t_embedding

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    pass
