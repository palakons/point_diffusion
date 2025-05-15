import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pytorch3d.loss import chamfer_distance
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

import open3d as o3d

from truckscenes import TruckScenes


class SinusoidalTimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, time_embed_dim=64, model_layers=[]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, 128),
            nn.ReLU()
        )
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim + 128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim)
        # )

        self.net = nn.Sequential()
        for i, dim in enumerate(model_layers):
            if i == 0:
                self.net.append(nn.Linear(input_dim + 128, dim))
            else:
                self.net.append(nn.Linear(model_layers[i - 1], dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(model_layers[-1], output_dim))

    def forward(self, x_t, t):
        # print("x_t",x_t.shape)#"x_t torch.Size([1, 384])"
        t_embed = self.time_mlp(t)  # (B, 128)
        # print("t_embed",t_embed.shape) #t_embed torch.Size([1, 128])
        x = torch.cat([x_t, t_embed], dim=-1)
        return self.net(x)


def get_man_data(N, radar_ch):
    data_root = "/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/mini/man-truckscenes"
    trucksc = TruckScenes(
        'v1.0-mini', data_root, True)

    first_frame_token = trucksc.scene[0]['first_sample_token']
    frame = trucksc.get('sample', first_frame_token)
    radar = trucksc.get('sample_data', frame['data'][radar_ch])

    radar_pcd_path = os.path.join(data_root, radar['filename'])
    cloud = o3d.io.read_point_cloud(radar_pcd_path)
    radar_data = torch.tensor(cloud.points)
    # print("radar_data", radar_data.shape) #([800, 3])
    return radar_data[:N, :].unsqueeze(0)  # (1, N, 3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D = 1, 128, 3
    T = 100
    seed_value = 42
    method = "2_5_flatten_x0"
    # mlp_layers = [256, 128]
    mlp_layers = [2048, 1024]
    time_embed_dim = 64
    lr = 1e-4
    epochs = 3_000_001
    vis_freq = 1000
    radar_ch = "RADAR_LEFT_FRONT"
    base_dir = "/ist-nas/users/palakonk/singularity_logs/"
    run_name = f"{method}_{N:04d}_MAN_1m_model{'-'.join(str(layer) for layer in mlp_layers)}"
    log_dir = f"{base_dir}/tb_log/mlp_man/{run_name}"

    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(seed_value)
    # x_0 = torch.randn(B, N, D).to(device)
    x_0_org = get_man_data(N, radar_ch=radar_ch).to(device).float()
    print("x_0", x_0_org)
    x_0 = x_0_org.view(x_0_org.shape[0], -1)  # flatten
    data_mean, data_std = x_0.mean(dim=[0]), x_0.std(dim=[0])
    if x_0_org.shape[0] == 1:
        data_std = torch.ones_like(data_std)
    # data_mean, data_std = x_0.mean(dim=[0, 1]), x_0.std(dim=[0, 1])
    print("data_mean", data_mean)
    print("data_std", data_std)
    x_0 = (x_0 - data_mean) / data_std
    # data_mean_after, data_std_after = x_0.mean(dim=[0]), x_0.std(dim=[0])
    # print("data_mean", data_mean_after)
    # print("data_std", data_std_after)
    # exit()

    # model = SimpleDenoiser(input_dim=N * D, output_dim=N * D,
    #                        time_embed_dim=64, model_layers=[256, 128]).to(device)
    model = SimpleDenoiser(input_dim=N * D, output_dim=N * D,
                           time_embed_dim=time_embed_dim, model_layers=mlp_layers).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=T,
                              prediction_type="sample",  # or "epsilon" or "v_prediction"
                              clip_sample=False,  # important for point clouds
                              )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tt = trange(epochs)
    cd_loss = 0
    for i_epoch, epoch in enumerate(tt):
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t = scheduler.add_noise(x_0, noise, t)
        pred_x0 = model(x_t, t)
        loss = F.mse_loss(pred_x0, x_0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_epoch % vis_freq == 0:
            xts_tensor_list, steps, cd_loss = sample(
                model.eval(), scheduler, T, B, N, D, x_0, device, data_mean=data_mean, data_std=data_std)
            path = f"{base_dir}/plots/mlp_man/basic_{method}_{N: 04d}_MAN_1m_model{'-'.join(str(layer) for layer in mlp_layers)}/sample_ep_{i_epoch:06d}_gt0_train-idx-32d.json"
            save_sample_json(path, epoch, x_0.view(B, N, D), xts_tensor_list,
                             steps,  cd=cd_loss, data_mean=data_mean.view(N, D), data_std=data_std.view(N, D), config={
                "B": B,
                "N": N,
                "D": D,
                "T": T,
                "seed_value": seed_value,
                "method": method,
                "mlp_layers": mlp_layers,
                "time_embed_dim": time_embed_dim,
                "lr": lr,
                "radar_ch": radar_ch
            })
            writer.add_scalar(f"CD", cd_loss, epoch)

        tt.set_description_str(
            f"MSE = {loss.item():.2f}, CD = {cd_loss:.2f}")

        writer.add_scalar(f"Loss", loss.item(), epoch)
    writer.close()


def save_sample_json(path, epoch, gt_tensor, xts_tensor_list, steps,  cd=None, data_mean=None, data_std=None, config=None):
    """
    Save point cloud data and metadata in specified JSON format.

    Args:
        path (str): File path to save the JSON.
        epoch (int): Current epoch.
        gt_tensor (Tensor): Ground truth tensor of shape (1, N, 3).
        xts_tensor_list (List[Tensor]): List of tensors (B, N, 3) at different timesteps.
        steps (List[int]): List of timesteps.
        cd (float, optional): Chamfer Distance value.
        data_mean (Tensor or List, optional): Mean used for normalization.
        data_std (Tensor or List, optional): Std used for normalization.

    Returns:
        None
    """
    data = {
        "epoch": epoch,
        "gt": gt_tensor.detach().cpu().tolist(),
        "xts": [xt.detach().cpu().tolist() for xt in xts_tensor_list],
        "steps": steps,
    }
    if cd is not None:
        data["cd"] = float(cd)

    if data_mean is not None:
        data["data_mean"] = data_mean if isinstance(
            data_mean, list) else data_mean.detach().cpu().tolist()

    if data_std is not None:
        data["data_std"] = data_std if isinstance(
            data_std, list) else data_std.detach().cpu().tolist()
    if config is not None:
        data["config"] = config
    # print("data", data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def sample(model, scheduler, T, B, N, D, x_0, device, data_mean, data_std):
    scheduler.set_timesteps(T)
    x_t = torch.randn(B, N * D).to(device)
    xts_tensor_list = []
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = torch.full(
                (B,), t.cpu(), device=device, dtype=torch.long)
            x0_pred = model(x_t.to(device), t_tensor)
            x_t = scheduler.step(
                x0_pred.cpu(), t_tensor.cpu(), x_t.cpu()).prev_sample
            xts_tensor_list.append(x_t[0].view(N, D).cpu())

    sampled = x_t.to(device)
    # print("sampled",sampled.shape,sampled.view(B,N,-1).shape)#torch.Size([1, 384]) torch.Size([1, 128, 3])
    cd_final, _ = chamfer_distance(
        (sampled*data_std+data_mean).view(B, N, -1), (x_0*data_std+data_mean).view(B, N, -1))
    # cd_final, _ = chamfer_distance(sampled, x_0)
    # print("-----")
    # print(scheduler.timesteps[-1:])
    xts_concat = xts_tensor_list[::10] + xts_tensor_list[-1:]
    time_step_concat = torch.cat(
        [scheduler.timesteps[::10], scheduler.timesteps[-1:]], dim=0)
    return xts_concat, [int(a) for a in time_step_concat], cd_final


if __name__ == "__main__":
    main()
