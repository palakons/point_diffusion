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
from datetime import datetime
# from pytorch3d.loss import chamfer_distance
# from model.point_cloud_model import PointCloudModel
from man_dataset import MANDataset, custom_collate_fn_man
from torch.utils.data import Dataset, DataLoader, Subset
from matplotlib import pyplot as plt
import numpy as np


def get_man_data(M, N, camera_ch, radar_ch, device, img_size, batch_size, shuffle, n_val=2,n_pull=10):
    dataset = MANDataset(
        n_pull,
        N,
        0,
        depth_model="vits",
        random_offset=False,
        data_file="man-mini",  # man-full
        device=device,
        is_scaled=True,
        img_size=img_size,
        radar_channel=radar_ch,
        camera_channel=camera_ch)

    indices = list(range(len(dataset)))
    train_indices = indices[:M]
    val_indices = indices[-n_val:]

    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)
    print("train", train_indices, "val", val_indices)
    print("len", len(dataset_train), len(dataset_val))
    # print("batch_size", batch_size)

    dataloader_train, dataloader_val = (
        DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=custom_collate_fn_man,
            shuffle=shuffle,
        ),
        DataLoader(
            dataset_val,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=custom_collate_fn_man,
            shuffle=shuffle,
        ))
    return dataloader_train, dataloader_val

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


def sample(model, scheduler, T, B, N, D, gt_normed, device, data_mean, data_std, x_0_cond_normed):
    scheduler.set_timesteps(T)
    x_t = torch.randn(B, N, D).to(device)
    xts_tensor_list = []
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = model(torch.cat(
                [x_t, x_0_cond_normed], dim=-1), t_tensor)

            x_t = scheduler.step(
                noise_pred, t, x_t).prev_sample
            # print("x_t", x_t.shape) #torch.Size([2, 3, 3])

            xts_tensor_list.append(x_t.cpu())

    sampled = x_t.to(device)
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)
    # print("sampled", sampled.shape)
    # print("data_mean", data_mean.shape)
    # print("data_std", data_std.shape)
    # print("gt_normed", gt_normed.shape)
    # sampled torch.Size([2, 3, 3])
    # data_mean torch.Size([3])
    # data_std torch.Size([3])
    # gt_normed torch.Size([2, 3, 3])
    cd_final, _ = chamfer_distance(
        sampled*data_std+data_mean, gt_normed*data_std+data_mean, batch_reduction=None)
    # print("cd_final", cd_final.tolist())
    # cd_final tensor([7388.0283, 6812.9808], device='cuda:0', dtype=torch.float64)
    # cd_final, _ = chamfer_distance(sampled, x_0)
    # print("-----")
    # print(scheduler.timesteps[-1:])
    # torch.Size([11, 2, 3, 3])
    xts_concat = torch.stack(xts_tensor_list[::10] + xts_tensor_list[-1:])

    time_step_concat = torch.cat(
        [scheduler.timesteps[::10], scheduler.timesteps[-1:]], dim=0)
    return xts_concat, [int(a) for a in time_step_concat], cd_final


def save_checkpoint(model, optimizer,
                    train_cd_list,
                    val_cd_list,
                    cd_epochs,
                    train_loss_list, val_loss_list, epoch, base_dir, config, run_name):
    checkpint_dir = os.path.join(
        base_dir, "checkpoints_man")
    proc_id = os.getpid()
    checkpoint_fname = os.path.join(
        checkpint_dir, f"cp_{datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S')}-{run_name.replace('/', '_') }_host{os.uname().nodename}_proc{proc_id}.pth")
    db_fname = os.path.join(
        base_dir, 'checkpoints_man', 'db.txt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_cd_list': train_cd_list,

        'val_cd_list': val_cd_list,
        'cd_epochs': cd_epochs,
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'config': config,
    }
    os.makedirs(checkpint_dir, exist_ok=True)
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_fname)
    with open(db_fname, "a" if os.path.exists(db_fname) else "w") as f:
        data = {"fname": checkpoint_fname, "config": config}
        json.dump(data, f)
        f.write("\n")


def get_sinusoidal_embedding(timesteps, embedding_dim, device):
    assert embedding_dim % 2 == 0, f"embedding_dim({embedding_dim}) must be even"
    half_dim = embedding_dim // 2
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=device)
        * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 4
    B, N, D = 2, 128, 3  # 128, 3  # upto 800 points
    lr = 1e-4
    epochs = 300_001
    pvcnn_embed_dim = 64
    pvcnn_width_multiplier = 1  # 2,4,3
    pvcnn_voxel_resolution_multiplier = 1  # 0.5,2,4
    ablate_pvcnn_mlp = False
    ablate_pvcnn_cnn = False

    n_val = 2
    n_pull = 10
    method = "4_5_pvcnn_cond_m4b4_notnormed_cond"

    T = 100
    pc2_conditioning_dim = 64
    vis_freq = 1000
    radar_ch = "RADAR_LEFT_FRONT"
    camera_ch = "CAMERA_RIGHT_FRONT"
    img_size = 618
    seed_value = 42
    # base_dir = "/ist-nas/users/palakonk/singularity_logs/"
    base_dir = "/home/palakons/logs/"  # singularity
    # run_name = f"{method}_{N:04d}_MAN_sc{M}_emb{pvcnn_embed_dim}_wm{pvcnn_width_multiplier}_vrm{pvcnn_voxel_resolution_multiplier}_ablate_mlp{'T' if ablate_pvcnn_mlp else 'F'}_cnn{'T' if ablate_pvcnn_cnn else 'F'}"
    run_name = f"{method}_{N:04d}_MAN_sc{M}"
    log_dir = f"{base_dir}tb_log/mlp_man/{run_name}"
    assert M<=8, f"Support M<=8, now {M}"

    exp_config = {
        "M": M,
        "n_val": n_val,
        "B": B,
        "N": N,
        "D": D,
        "T": T,
        "seed_value": seed_value,
        "method": method,
        "n_pull": n_pull,
        "pvcnn_embed_dim": pvcnn_embed_dim,
        "pvcnn_width_multiplier": pvcnn_width_multiplier,
        "pvcnn_voxel_resolution_multiplier": pvcnn_voxel_resolution_multiplier,
        "pc2_conditioning_dim": pc2_conditioning_dim,
        # "time_embed_dim": time_embed_dim,
        "lr": lr,
        "radar_ch": radar_ch,
        "camera_ch": camera_ch,
        "img_size": img_size,
        "epochs": epochs,
        "run_name": run_name,
        "ablate_pvcnn_mlp": ablate_pvcnn_mlp,
        "ablate_pvcnn_cnn": ablate_pvcnn_cnn,
    }
    print("exp_config", exp_config)

    writer = SummaryWriter(log_dir=log_dir)
    torch.manual_seed(seed_value)
    dataloader_train, dataloader_val = get_man_data(
        M, N, camera_ch, radar_ch, device, img_size, B, shuffle=True, n_val=n_val, n_pull=n_pull)

    data_mean, data_std = dataloader_train.dataset.dataset.data_mean, dataloader_train.dataset.dataset.data_std
    print("data_mean", data_mean.tolist())
    print("data_std", data_std.tolist())

    all_cond_signal = get_sinusoidal_embedding(torch.tensor(
        range(M+n_val), device=device) , pc2_conditioning_dim, device=device)
    if False:
        fname = "/home/palakons/from_scratch/encoding.png"
        print("all_cond_signal", all_cond_signal)
        
        #make subplots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(all_cond_signal.cpu().numpy())
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(np.log(all_cond_signal.cpu()))
        plt.title("Conditioning signal")
        plt.savefig(fname)
        exit()
        # print("all_cond_signal", all_cond_signal.shape)

    model = PVC2Model(
        in_channels=D + pc2_conditioning_dim,
        out_channels=D,
        embed_dim=pvcnn_embed_dim,
        dropout=0.1,
        width_multiplier=pvcnn_width_multiplier,
        voxel_resolution_multiplier=pvcnn_voxel_resolution_multiplier,
        ablate_pvcnn_mlp=ablate_pvcnn_mlp,
        ablate_pvcnn_cnn=ablate_pvcnn_cnn
    ).to(device)
    """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
    scheduler = DDPMScheduler(num_train_timesteps=T,
                              prediction_type="epsilon",  # or "sample" or "v_prediction"
                              clip_sample=False,  # important for point clouds
                              )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tt = trange(epochs)
    cd_train_ave = 0
    train_cd_list = {}
    val_cd_list = {}
    cd_epochs = []
    train_loss_list = []
    val_loss_list = []

    id_list = {}

    for i_epoch, epoch in enumerate(tt):
        sum_loss = 0
        model.train()
        for i_batch, batch in enumerate(dataloader_train):
            (depths,
             radar_data,
             camera,
             image_rgb,
             frame_token,
             npoints,
             npoints_filtered) = batch
            B, N, D = radar_data.shape

            # make frame_id to make encoding
            if i_epoch == 0:
                for i, token in enumerate(frame_token):
                    train_cd_list[token] = []
                    if token not in id_list:
                        id_list[token] = len(id_list)
            frame_idx = torch.tensor([id_list[token] for token in frame_token],
                                     device=device)

            # condition basic by sinusoidal embedding
            x_0_cond = all_cond_signal[frame_idx].float().to(device)
            # print("x_0_cond", x_0_cond.shape)  # x_0_cond torch.Size([2, 4])
            # repeating the condition to match the radar data
            x_0_cond = x_0_cond.unsqueeze(1).repeat(1, N, 1)
            # print("x_0_cond", x_0_cond.shape)  # x_0_cond torch.Size([2, 4, 4])
            # print("frame_token", frame_token)
            # print("x_0_cond", x_0_cond)
            cond_mean, cond_std = x_0_cond.mean(
                dim=[0, 1]), x_0_cond.std(dim=[0, 1])
            cond_std[cond_std == 0] = 1
            # print("cond_mean", cond_mean)
            # print("cond_std", cond_std)
            x_0_cond_norm = (x_0_cond - cond_mean) / cond_std
            # print("x_0_cond_norm", x_0_cond_norm)

            x_0_data = radar_data.float().to(device)
            t = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(x_0_data)
            x_t = scheduler.add_noise(x_0_data, noise, t)
            x_t = torch.cat(
                # [x_t, x_0_cond_norm], dim=-1)
                [x_t, x_0_cond], dim=-1)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            sum_loss += loss.item() * B

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # writer.add_scalar(
        #     f"Loss/train", sum_loss/len(dataloader_train.dataset), epoch)
        writer.add_scalars(
            f"Loss", {"train/average": sum_loss/len(dataloader_train.dataset)}, epoch)

        train_loss_list.append(sum_loss/len(dataloader_train.dataset))
        with torch.no_grad():
            sum_loss = 0
            model.eval()
            for i_batch, batch in enumerate(dataloader_val):
                (depths,
                 radar_data,
                 camera,
                 image_rgb,
                 frame_token,
                 npoints,
                 npoints_filtered) = batch

                # make frame_id to make encoding
                if i_epoch == 0:
                    for i, token in enumerate(frame_token):

                        val_cd_list[token] = []
                        if token not in id_list:
                            id_list[token] = len(id_list)
                frame_idx = torch.tensor([id_list[token]
                                         for token in frame_token], device=device)

                B, N, D = radar_data.shape
                x_0_data = radar_data.float().to(device)

                x_0_cond = all_cond_signal[frame_idx].float().to(device)
                x_0_cond = x_0_cond.unsqueeze(1).repeat(1, N, 1)
                
                cond_mean, cond_std = x_0_cond.mean(
                    dim=[0, 1]), x_0_cond.std(dim=[0, 1])
                cond_std[cond_std == 0] = 1

                x_0_cond_norm = (x_0_cond - cond_mean) / cond_std

                t = torch.randint(0, T, (B,), device=device)
                noise = torch.randn_like(x_0_data)

                x_t = scheduler.add_noise(x_0_data, noise, t)

                x_t = torch.cat(
                    # [x_t, x_0_cond_norm], dim=-1)
                    [x_t, x_0_cond], dim=-1)

                pred_noise = model(x_t, t)
                loss = F.mse_loss(pred_noise, noise)
                sum_loss += loss.item() * B

            # writer.add_scalar(
            #     f"Loss/val", sum_loss/len(dataloader_val.dataset), epoch)
            writer.add_scalars(
                f"Loss", {"val/average": sum_loss/len(dataloader_val.dataset)}, epoch)
            val_loss_list.append(sum_loss/len(dataloader_val.dataset))
        if i_epoch % vis_freq == 0:
            cd_epochs.append(epoch)
            sum_train_cd = 0
            for i_batch, batch in enumerate(dataloader_train):
                (depths,
                 radar_data,
                 camera,
                 image_rgb,
                 frame_token,
                 npoints,
                 npoints_filtered) = batch

                frame_idx = torch.tensor([id_list[token]
                                         for token in frame_token], device=device)

                B, N, D = radar_data.shape
                x_0_data = radar_data.float().to(device)

                x_0_cond = all_cond_signal[frame_idx].float().to(device)
                x_0_cond = x_0_cond.unsqueeze(1).repeat(1, N, 1)

                cond_mean, cond_std = x_0_cond.mean(
                    dim=[0, 1]), x_0_cond.std(dim=[0, 1])
                cond_std[cond_std == 0] = 1

                x_0_cond_normed = (x_0_cond - cond_mean) / cond_std

                xts_tensor_list, steps, cd_loss = sample(
                    model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, x_0_cond_normed=x_0_cond)
                    # model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, x_0_cond_normed=x_0_cond_normed)

                sum_train_cd += cd_loss.sum().item()
                # print("len(xts_tensor_list)", len(xts_tensor_list))
                # print("xts[0]", xts_tensor_list[0].shape)
                # print("steps", steps)
                # print("cd_loss", cd_loss)
                for i_result, frame_tkn in enumerate(frame_token):
                    xts = xts_tensor_list[:, i_result, :, :]
                    cd = cd_loss[i_result]
                    # print("xts", xts.shape)
                    # print("cd", cd.item())
                    # print("frame_tkn", frame_tkn)

                    path = f"{base_dir}/plots/mlp_man/{run_name}/sample_ep_{i_epoch:06d}_gt0_train-idx-{frame_tkn[:3]}.json"
                    save_sample_json(path, epoch, x_0_data, xts,
                                     steps,  cd=cd.item(), data_mean=data_mean, data_std=data_std, config=exp_config)
                    # writer.add_scalar(
                    #     f"CD/train/{frame_tkn[:3]}", cd.item(), epoch)
                    writer.add_scalars(
                        f"CD", {f"train/{frame_tkn[:3]}": cd.item()}, epoch)
                    # print("cd", cd.item())

                    train_cd_list[frame_tkn].append(cd.item())
            cd_train_ave = sum_train_cd / len(dataloader_train.dataset)
            # writer.add_scalar(f"CD/train/mean", cd_train_ave, epoch)
            writer.add_scalars(
                f"CD", {"train/average": cd_train_ave}, epoch)

            # dupto here for val
            sum_val_cd = 0
            for i_batch, batch in enumerate(dataloader_val):
                (depths,
                 radar_data,
                 camera,
                 image_rgb,
                 frame_token,
                 npoints,
                 npoints_filtered) = batch

                frame_idx = torch.tensor([id_list[token]
                                         for token in frame_token], device=device)

                B, N, D = radar_data.shape
                x_0_data = radar_data.float().to(device)

                x_0_cond = all_cond_signal[frame_idx].float().to(device)
                x_0_cond = x_0_cond.unsqueeze(1).repeat(1, N, 1)

                cond_mean, cond_std = x_0_cond.mean(
                    dim=[0, 1]), x_0_cond.std(dim=[0, 1])
                cond_std[cond_std == 0] = 1

                x_0_cond_normed = (x_0_cond - cond_mean) / cond_std

                xts_tensor_list, steps, cd_loss = sample(
                    model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, x_0_cond_normed=x_0_cond)
                    # model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, x_0_cond_normed=x_0_cond_normed)

                sum_val_cd += cd_loss.sum().item()
                # print("len(xts_tensor_list)", len(xts_tensor_list))
                # print("xts[0]", xts_tensor_list[0].shape)
                # print("steps", steps)
                # print("cd_loss", cd_loss)
                for i_result, frame_tkn in enumerate(frame_token):
                    xts = xts_tensor_list[:, i_result, :, :]
                    cd = cd_loss[i_result]
                    # print("xts", xts.shape)
                    # print("cd", cd.item())
                    # print("frame_tkn", frame_tkn)

                    path = f"{base_dir}/plots/mlp_man/basic_{method}_{N:04d}_MAN_1m_pvcnn_emb{pvcnn_embed_dim}/sample_ep_{i_epoch:06d}_gt0_val-idx-{frame_tkn[:3]}.json"
                    save_sample_json(path, epoch, x_0_data, xts,
                                     steps,  cd=cd.item(), data_mean=data_mean, data_std=data_std, config=exp_config)
                    # writer.add_scalar(
                    #     f"CD/val/{frame_tkn[:3]}", cd.item(), epoch)
                    writer.add_scalars(
                        f"CD", {f"val/{frame_tkn[:3]}": cd.item()}, epoch)

                    val_cd_list[frame_tkn].append(cd.item())
            cd_val_ave = sum_val_cd / len(dataloader_val.dataset)
            # writer.add_scalar(f"CD/val/mean", cd_val_ave, epoch)
            writer.add_scalars(
                f"CD", {"val/average": cd_val_ave}, epoch)

        tt.set_description_str(
            f"MSE = {loss.item():.2f}, CD_tr = {cd_train_ave:.2f}, CD_val = {cd_val_ave:.2f}")

    writer.close()
    # Save the model
    print("Saving model...")
    save_checkpoint(model, optimizer,
                    train_cd_list=train_cd_list,
                    val_cd_list=val_cd_list,
                    cd_epochs=cd_epochs,
                    val_loss_list=val_loss_list,
                    train_loss_list=train_loss_list, epoch=epoch, base_dir=base_dir, config=exp_config, run_name=run_name)


if __name__ == "__main__":
    main()
