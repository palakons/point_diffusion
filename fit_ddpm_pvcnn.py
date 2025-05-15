from truckscenes import TruckScenes
from model.mypvcnn import PVC2Model
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from pytorch3d.loss.chamfer import chamfer_distance
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import pytorch3d
# from pytorch3d.loss import chamfer_distance
# from model.point_cloud_model import PointCloudModel


def get_man_data(N, radar_ch):
    # /ist-nas/users/palakonk/singularity_
    data_root = "/data/palakons/new_dataset/MAN/mini/man-truckscenes"
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
    x_t = torch.randn(B, N, D).to(device)
    xts_tensor_list = []
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = torch.full(
                (B,), t.cpu(), device=device, dtype=torch.long)
            noise_pred = model(x_t.to(device), t_tensor)
            x_t = scheduler.step(
                noise_pred, t_tensor, x_t).prev_sample
            xts_tensor_list.append(x_t[0].cpu())

    sampled = x_t.to(device)
    cd_final, _ = chamfer_distance(
        sampled*data_std+data_mean, x_0*data_std+data_mean)
    # cd_final, _ = chamfer_distance(sampled, x_0)
    # print("-----")
    # print(scheduler.timesteps[-1:])
    xts_concat = xts_tensor_list[::10] + xts_tensor_list[-1:]
    time_step_concat = torch.cat(
        [scheduler.timesteps[::10], scheduler.timesteps[-1:]], dim=0)
    return xts_concat, [int(a) for a in time_step_concat], cd_final


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D = 1, 256, 3  # upto 800 points
    T = 100
    seed_value = 42
    method = "4_pvcnn"
    # mlp_layers = [256, 128]
    # mlp_layers = [2048, 1024]
    pvcnn_embed_dim = 64
    time_embed_dim = 64
    lr = 1e-4
    epochs = 3_000_001
    vis_freq = 1000
    radar_ch = "RADAR_LEFT_FRONT"
    # base_dir = "/ist-nas/users/palakonk/singularity_logs/"
    base_dir = "/home/palakons/logs/"  # singularity
    run_name = f"{method}_{N:04d}_MAN_1m_pvcnn_emb{pvcnn_embed_dim}"
    log_dir = f"{base_dir}tb_log/mlp_man/{run_name}"

    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(seed_value)
    # x_0 = torch.randn(B, N, D).to(device)
    x_0 = get_man_data(N, radar_ch=radar_ch).to(device).float()
    print("x_0", x_0)
    data_mean, data_std = x_0.mean(dim=[0, 1]), x_0.std(dim=[0, 1])
    print("data_mean", data_mean)
    print("data_std", data_std)
    x_0 = (x_0 - data_mean) / data_std
    # data_mean_after, data_std_after = x_0.mean(dim=[0, 1]), x_0.std(dim=[0, 1])
    # print("data_mean", data_mean_after)
    # print("data_std", data_std_after)
    # exit()

    model = PVC2Model(
        in_channels=3,
        out_channels=3,
        embed_dim=pvcnn_embed_dim,
        dropout=0.1,
        width_multiplier=1,
        voxel_resolution_multiplier=1,).to(device)
    """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
    scheduler = DDPMScheduler(num_train_timesteps=T,
                              prediction_type="epsilon",  # or "sample" or "v_prediction"
                              clip_sample=False,  # important for point clouds
                              )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tt = trange(epochs)
    cd_loss = 0
    for i_epoch, epoch in enumerate(tt):
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t = scheduler.add_noise(x_0, noise, t)
        model.train()
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_epoch % vis_freq == 0:
            xts_tensor_list, steps, cd_loss = sample(
                model.eval(), scheduler, T, B, N, D, x_0, device, data_mean=data_mean, data_std=data_std)
            path = f"{base_dir}/plots/mlp_man/basic_{method}_{N:04d}_MAN_1m_pvcnn_emb{pvcnn_embed_dim}/sample_ep_{i_epoch:06d}_gt0_train-idx-32d.json"
            save_sample_json(path, epoch, x_0, xts_tensor_list,
                             steps,  cd=cd_loss, data_mean=data_mean, data_std=data_std, config={
                                 "B": B,
                                 "N": N,
                                 "D": D,
                                 "T": T,
                                 "seed_value": seed_value,
                                 "method": method,
                                 "pvcnn_embed_dim": pvcnn_embed_dim,
                                 "time_embed_dim": time_embed_dim,
                                 "lr": lr,
                                 "radar_ch": radar_ch
                             })
            writer.add_scalar(f"CD", cd_loss, epoch)

        tt.set_description_str(
            f"MSE = {loss.item():.2f}, CD = {cd_loss:.2f}")

        writer.add_scalar(f"Loss", loss.item(), epoch)
    writer.close()


if __name__ == "__main__":
    main()
