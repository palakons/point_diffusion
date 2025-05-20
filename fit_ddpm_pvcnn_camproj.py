import numpy as np
import cv2
import argparse
from truckscenes import TruckScenes
from model.mypvcnn import PVC2Model
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from pytorch3d.loss.chamfer import chamfer_distance
from model.feature_model_finetune import FeatureModel
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.structures import Pointclouds
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time

local_feature_cache = {}


def get_man_data(M, N, camera_ch, radar_ch, device, img_size, batch_size, shuffle, n_val=2, n_pull=10, flip_images=False):
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
        camera_channel=camera_ch,
        double_flip_images=flip_images,)

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


def sample(feature_model, model, scheduler, T, B, N, D, gt_normed, device, data_mean, data_std, img_size, camera, frame_token, image_rgb):
    scheduler.set_timesteps(T)
    x_t = torch.randn(B, N, D).to(device)
    xts_tensor_list = []

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            # print("x_t", x_t.shape)
            # x_t torch.Size([1, 4, 3])

            x_t_cond = get_camera_conditioning(img_size,
                                               feature_model, camera, frame_token, image_rgb, x_t, device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0)

            # x_t_cond torch.Size([1, 4, 384])
            # print("x_t_cond", x_t_cond.shape)

            noise_pred = model(torch.cat(
                [x_t, x_t_cond], dim=-1), t_tensor)

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


def plot_image_depth_projected(
    gt,
    fname,
    point_size=0.1,
    camera=None,
    image_rgb=None,
    depth_image=None,
    data_mean=torch.tensor([0, 0, 0]),
    data_std=torch.tensor([1, 1, 1]),
):

    data_mean = data_mean.float().cpu().numpy()
    data_std = data_std.float().cpu().numpy()

    # print("plot_sample_condition fname", fname)

    gt = gt.numpy()
    gt = gt * data_std + data_mean

    fig = plt.figure(figsize=(30, 10 * len(gt)))

    for i in range(len(gt)):
        world_points = torch.tensor(
            gt[i], dtype=torch.float32).to(camera[i].device)
        cam_coord = (
            camera[i].get_world_to_view_transform(
            ).transform_points(world_points)
        )  # N x3

        image_coord = camera[i].transform_points(world_points)  # N x3

        projected_points = image_coord[:, :2]  # N x2
        projected_points = projected_points.cpu().numpy()

        ax = fig.add_subplot(len(gt), 3, 3 * i + 1)
        imgg = image_rgb[i].cpu().numpy().transpose(1, 2, 0)
        # rotae 180 degrees
        imgg = imgg[:, ::-1, :]
        imgg = imgg[::-1, :, :]
        plot = ax.imshow(imgg)
        ax.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            marker="x",
            s=point_size,
            cmap="jet",
            c=cam_coord[:, 2].cpu().numpy(),
        )
        cax = inset_axes(ax, width="5%", height="50%",
                         loc="lower left", borderpad=1)
        fig.colorbar(plot, cax=cax)
        ax.axis("off")
        ax.set_title("image_rgb")

        ax = fig.add_subplot(len(gt), 3, 3 * i + 3, projection="3d")
        plot = ax.scatter(
            gt[i][:, 0],
            gt[i][:, 1],
            gt[i][:, 2],
            marker=",",
            s=point_size,
            cmap="jet",
            c=cam_coord[:, 2].cpu().numpy(),
        )
        cax = inset_axes(ax, width="5%", height="50%",
                         loc="lower left", borderpad=1)
        fig.colorbar(plot, cax=cax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect("equal")

    # plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def get_camera_conditioning(img_size, feature_model,  camera, frame_token, image_rgb, normed_points, device, raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1])):
    """
    points: (B, N, D) NIORMALIZED Point cloud data,


    DEFAULT PC^2 parameters
    raster_point_radius: float = 0.0075,
    raster_points_per_pixel: int = 1,
    bin_size: int = 0,
    scale_factor: float = 1.0

    """

    x_t_input = []
    # 1. get local features: VITS
    # local_features can have batch >1
    local_features = []
    for i, frame_tkn in enumerate(frame_token):
        if frame_tkn in local_feature_cache:
            local_feature = local_feature_cache[frame_tkn]
        else:
            local_feature = feature_model(
                image_rgb[i:i + 1].float().to(device)).to(device)
            local_feature_cache[frame_tkn] = local_feature
        local_features.append(local_feature)
        # distance_transform = compute_distance_transform(B=len(frame_token),
        #                                                 img_size=img_size, device=device) #all zeros
        # print("local_feature", local_feature.shape)
        # print("local_feature", local_feature)
        # print("distance_transform", distance_transform.shape)
        # print("distance_transform", distance_transform)
        # exit()
    local_features = torch.cat(local_features, dim=0).to(device)
    # print("local_features", local_features.shape)
    # local_features torch.Size([1, 384, 618, 618]) cpu

    # local_features = feature_model(
    #     x=image_rgb).to(device)  # (B, C, H, W)

    # 2. project features to point cloud
    B, C, H, W, device = *local_features.shape, local_features.device
    # local_features torch.Size([1, 384, 618, 618]) cpu
    # torch.Size([1, 384, 618, 618])
    # print("local_features", local_features.shape, local_features.device)

    raster_settings = PointsRasterizationSettings(
        # image_size=(943,943),
        image_size=(img_size, img_size),
        radius=raster_point_radius,
        points_per_pixel=raster_points_per_pixel,
        bin_size=bin_size,
    )

    R = raster_settings.points_per_pixel
    N = normed_points.shape[1]

    # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
    camera = camera.clone()
    camera.T = camera.T * scale_factor

    # Create rasterizer
    rasterizer = PointsRasterizer(
        cameras=camera, raster_settings=raster_settings).to(device)

    inv_yx_normed_points = normed_points.clone()
    # inv_yx_normed_points[:, :, 0] *= -1  # Flip X
    # inv_yx_normed_points[:, :, 2] *= -1  # Flip Z (if needed)

    # Associate points with features via rasterization
    fragments = rasterizer(Pointclouds(inv_yx_normed_points * point_std.to(
        device).float() + point_mean.to(device).float()).to(device))  # (B, H, W, R)
    # torch.Size([1, 618, 618, 1]) ,,,([1, 618, 618, 1])
    # print("fragments", fragments.idx.shape, fragments.zbuf.shape)
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    # torch.Size([1, 618, 618, 1])
    # print("visible_pixels", visible_pixels.shape)
    # count visible pixels
    num_visible_pixels = visible_pixels.sum()

    if False:
        print("num_visible_pixels", num_visible_pixels.item(),
              f"or {num_visible_pixels.item()/(H*W)*100:.2f}%", W, H)

        import matplotlib.pyplot as plt

        # Convert to NumPy
        visible_pixels_np = visible_pixels.squeeze(-1).cpu().numpy()

        # plot visible pixels and image_rgb side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(visible_pixels_np[0], cmap="gray")

        # plot image_rgb
        image_rgb_np = image_rgb.squeeze(
            0).cpu().numpy().transpose(1, 2, 0)  # Convert to NumPy
        ax[1].imshow(image_rgb_np)
        plt.savefig(
            f"/home/palakons/from_scratch/pix_unnormed_r{raster_point_radius}_test_im{img_size}_newper.png")
        plt.close()

        plot_image_depth_projected(
            normed_points.cpu(),
            f"/home/palakons/from_scratch/pix_unnormed_r{raster_point_radius}_projected_test_im{img_size}_newper.png",
            1,
            camera=camera,
            image_rgb=image_rgb.detach().cpu(),
            depth_image=None,
            data_mean=point_mean,
            data_std=point_std
        )
    # num_visible_pixels tensor(16, device='cuda:0') or percent tensor(4.1893e-05, device='cuda:0') 618 618

    points_to_visible_pixels = fragments_idx[visible_pixels]
    # ixels torch.Size([4])
    # print("points_to_visible_pixels", points_to_visible_pixels.shape)

    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(
        0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)
    # torch.Size([1, 618, 618, 1, 384])
    # print("local_features", local_features.shape)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * N, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    # torch.Size([4, 384])
    # print("local_features_proj", local_features_proj.shape)
    local_features_proj = local_features_proj.reshape(B, N, C)
    # torch.Size([1, 4, 384])
    # print("local_features_proj", local_features_proj.shape)
    # import pdb; pdb.set_trace()
    local_feature_mean, local_feature_std = local_features_proj.mean(
        dim=[0, 1]), local_features_proj.std(dim=[0, 1])
    local_feature_std[local_feature_std == 0] = 1
    local_features_proj = (local_features_proj -
                           local_feature_mean) / local_feature_std

    x_t_input.append(local_features_proj)

    # Concatenate together all the pointwise features
    x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

    return x_t_input


def compute_distance_transform(B, img_size, device):
    mask = torch.ones((B, 1, img_size, img_size), device=device)
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(device)
    return distance_transform


def main():
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-M", "--num_scenes", type=int, default=1,
                        help="Number of scenes to train on")
    parser.add_argument("-N", "--num_points", type=int, default=128,
                        help="Number of points to sample from each scene")
    parser.add_argument("-B", "--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("-E", "--epochs", type=int, default=300_001,
                        help="Number of epochs to train for")
    parser.add_argument("-LR", "--learning_rate", type=float, default=1e-4,

                        help="Learning rate for the optimizer")
    parser.add_argument("-m", "--method", type=str, default="6_pvcnn_camcond_reduce64_test_time",

                        help="Method name for the experiment")
    parser.add_argument("-pc2dim", "--pc2_conditioning_dim", type=int, default=64,
                        help="Conditioning dimension for PC^2")
    parser.add_argument("-pvcnndim", "--pvcnn_embed_dim", type=int, default=64,
                        help="Embedding dimension for PVCNN")
    parser.add_argument("-pvcnnwm", "--pvcnn_width_multiplier", type=int, default=1,
                        help="Width multiplier for PVCNN")
    parser.add_argument("-pvcnnvrm", "--pvcnn_voxel_resolution_multiplier", type=int, default=1,
                        help="Voxel resolution multiplier for PVCNN")
    parser.add_argument("-ablate_mlp", "--ablate_pvcnn_mlp", action='store_true',
                        help="Ablate MLP in PVCNN", default=False)
    parser.add_argument("-ablate_cnn", "--ablate_pvcnn_cnn", action='store_true',  # when argument is passed, it will be True
                        help="Ablate CNN in PVCNN", default=False)
    parser.add_argument("-vits_model", "--vits_model", type=str, default="vit_small_patch16_224_msn",
                        help="VITS feature model name")
    parser.add_argument("-finetune_vits", "--finetune_vits", action='store_true',
                        help="Finetune VITS model or not", default=False)
    parser.add_argument("-vits_checkpoint", "--vits_checkpoint", type=str, default=None,
                        help="Path to VITS checkpoint")
    parser.add_argument("-flip", "--flip_images", action='store_true',
                        help="Flip images or not", default=True)
    parser.add_argument("-img_size", "--img_size", type=int, default=618,
                        help="Image size for VITS model")
    parser.add_argument("-radar_ch", "--radar_channel", type=str, default="RADAR_LEFT_FRONT",
                        help="Radar channel to use")
    parser.add_argument("-camera_ch", "--camera_channel", type=str, default="CAMERA_RIGHT_FRONT",
                        help="Camera channel to use")
    parser.add_argument("-base_dir", "--base_dir", type=str, default="/home/palakons/logs/",
                        help="Base directory for saving logs and checkpoints")
    parser.add_argument("-sd", "--seed_value", type=int, default=42,
                        help="Seed value for random number generation")
    parser.add_argument("-vfreq", "--vis_freq", type=int, default=1000,
                        help="Frequency of visualization")
    parser.add_argument("-n_val", "--n_val", type=int, default=2,
                        help="Number of validation scenes")
    parser.add_argument("-n_pull", "--n_pull", type=int, default=10,
                        help="Number of scenes to pull from dataset")
    parser.add_argument("-t", "--T", type=int, default=100,
                        help="Number of timesteps for DDPM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # M = 1
    M = args.num_scenes
    # B, N, D = 1, 128, 3  # 128, 3  # upto 800 points
    B = args.batch_size
    N = args.num_points
    D = 3
    # lr = 1e-4
    lr = args.learning_rate
    # epochs = 300_001
    epochs = args.epochs
    # pc2_conditioning_dim = 64
    pc2_conditioning_dim = args.pc2_conditioning_dim
    vits_dim = 384

    # pvcnn_embed_dim = 64
    pvcnn_embed_dim = args.pvcnn_embed_dim
    # pvcnn_width_multiplier = 1  # 2,4,3
    pvcnn_width_multiplier = args.pvcnn_width_multiplier
    # pvcnn_voxel_resolution_multiplier = 1  # 0.5,2,4
    pvcnn_voxel_resolution_multiplier = args.pvcnn_voxel_resolution_multiplier
    # ablate_pvcnn_mlp = False
    ablate_pvcnn_mlp = args.ablate_pvcnn_mlp
    # ablate_pvcnn_cnn = False
    ablate_pvcnn_cnn = args.ablate_pvcnn_cnn
    # vits_model = "vit_small_patch16_224_msn"
    vits_model = args.vits_model
    # finetune_vits = False
    finetune_vits = args.finetune_vits
    # vits_checkpoint = None
    vits_checkpoint = args.vits_checkpoint
    # flip_images = True
    flip_images = args.flip_images

    # n_val = 1
    n_val = args.n_val
    # n_pull = 1
    n_pull = args.n_pull
    # method = "6_pvcnn_camcond_reduce64_test_time"
    method = args.method

    # T = 100
    T = args.T
    # vis_freq = 1000
    vis_freq = args.vis_freq
    # radar_ch = "RADAR_LEFT_FRONT"
    radar_ch = args.radar_channel
    # camera_ch = "CAMERA_RIGHT_FRONT"
    camera_ch = args.camera_channel
    # img_size = 618  # 943#
    img_size = args.img_size
    # seed_value = 42
    seed_value = args.seed_value
    # base_dir = "/ist-nas/users/palakonk/singularity_logs/"
    # base_dir = "/home/palakons/logs/"  # singularity
    base_dir = args.base_dir

    data_group = "mlp_man"
    run_name = f"{method}_{N:04d}_MAN_sc{M}_emb{pvcnn_embed_dim}_wm{pvcnn_width_multiplier}_vrm{pvcnn_voxel_resolution_multiplier}_ablate_mlp{'T' if ablate_pvcnn_mlp else 'F'}_cnn{'T' if ablate_pvcnn_cnn else 'F'}_flip{'T' if flip_images else 'F'}"
    log_dir = f"{base_dir}tb_log/{data_group}/{run_name}"
    dir_rev = 0
    while os.path.exists(log_dir):
        dir_rev += 1
        log_dir = f"{base_dir}tb_log/{data_group}/{run_name}_r{dir_rev:02d}"
    if dir_rev > 0:
        run_name += f"_r{dir_rev:02d}"
        print("run_name", run_name)
    assert M <= n_pull-n_val, f"n_pull {n_pull} should be greater than M {M} + n_val {n_val}"
    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()



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
        "vits_dim": vits_dim,
        "lr": lr,
        "radar_ch": radar_ch,
        "camera_ch": camera_ch,
        "img_size": img_size,
        "epochs": epochs,
        "run_name": run_name,
        "ablate_pvcnn_mlp": ablate_pvcnn_mlp,
        "ablate_pvcnn_cnn": ablate_pvcnn_cnn,
        "vits_model": vits_model,
        "finetune_vits": finetune_vits,
        "vits_checkpoint": vits_checkpoint,
        "flip_images": flip_images,
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
        # utc time
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        #also have a copy of this code
        "code": code,
    }
    print("exp_config", exp_config)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", json.dumps(exp_config, indent=4))
    torch.manual_seed(seed_value)
    dataloader_train, dataloader_val = get_man_data(
        M, N, camera_ch, radar_ch, device, img_size, B, shuffle=True, n_val=n_val, n_pull=n_pull, flip_images=flip_images)

    data_mean, data_std = dataloader_train.dataset.dataset.data_mean, dataloader_train.dataset.dataset.data_std
    print("data_mean", data_mean.tolist())
    print("data_std", data_std.tolist())

    model = PVC2Model(
        in_channels=D + vits_dim,
        out_channels=D,
        embed_dim=pvcnn_embed_dim,
        dropout=0.1,
        width_multiplier=pvcnn_width_multiplier,
        voxel_resolution_multiplier=pvcnn_voxel_resolution_multiplier,
        ablate_pvcnn_mlp=ablate_pvcnn_mlp,
        ablate_pvcnn_cnn=ablate_pvcnn_cnn,
        vits_proj_dim=pc2_conditioning_dim,
    ).to(device)
    """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """

    feature_model = FeatureModel(
        img_size, model_name=vits_model,
        global_pool='', finetune_vits=finetune_vits, vits_checkpoint=vits_checkpoint).to(device)
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

    st_time = time.time()
    for i_epoch, epoch in enumerate(tt):
        sum_loss = 0
        model.train()
        start_time = time.time()
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

            x_0_data = radar_data.float().to(device)
            noise = torch.randn_like(x_0_data)
            # print("noise", noise.shape)  # torch.Size([1, 4, 3])
            t = torch.randint(0, T, (B,), device=device)
            x_t_data = scheduler.add_noise(x_0_data, noise, t)
            add_noise_time = time.time()
            x_t_cond = get_camera_conditioning(
                img_size,
                feature_model, camera, frame_token, image_rgb, x_t_data,  device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0, point_mean=data_mean.to(device).float(), point_std=data_std.to(device).float())
            # print("x_t", x_t.shape)  # x_t torch.Size([1, 4, 387]) 3+384
            cond_time = time.time()
            if False:
                inter_mean, inter_std = x_t.mean(
                    dim=[0, 1]), x_t.std(dim=[0, 1])
                inter_std[inter_std == 0] = 1
                # print("inter_mean", inter_mean)
                # print("inter_std", inter_std)
                # save inter_mean and inter_std to csv
                ffname = "/home/palakons/from_scratch/stdmean.csv"
                with open(ffname, 'a') as f:
                    f.write(f"mean,{inter_mean.tolist()}\n")
                    f.write(f"std,{inter_std.tolist()}\n")
            x_t = torch.cat(
                [x_t_data, x_t_cond], dim=-1)

            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            sum_loss += loss.item() * B
            ml_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            back_time = time.time()
            optimizer.step()
            # print(f"epoch {epoch}  add_noise {add_noise_time-start_time:.2f} cond {cond_time-add_noise_time:.2f} ml {ml_time-cond_time:.2f} back {back_time-ml_time:.2f} step {time.time()-back_time:.2f}")
        train_done = time.time()
        # writer.add_scalar(
        #     f"Loss/train", sum_loss/len(dataloader_train.dataset), epoch)
        writer.add_scalars(
            f"Loss", {"train/average": sum_loss/len(dataloader_train.dataset)}, epoch)

        train_loss_list.append(sum_loss/len(dataloader_train.dataset))
        tensor_done = time.time()
        with torch.no_grad():
            sum_loss = 0
            model.eval()
            val_start = time.time()
            for i_batch, batch in enumerate(dataloader_val):
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
                        val_cd_list[token] = []
                        if token not in id_list:
                            id_list[token] = len(id_list)
                frame_idx = torch.tensor([id_list[token]
                                          for token in frame_token], device=device)

                x_0_data = radar_data.float().to(device)
                noise = torch.randn_like(x_0_data)
                t = torch.randint(0, T, (B,), device=device)

                x_t_data = scheduler.add_noise(x_0_data, noise, t)

                x_t_cond = get_camera_conditioning(
                    img_size,
                    feature_model, camera, frame_token, image_rgb, x_t_data, device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0)

                x_t = torch.cat(
                    [x_t_data, x_t_cond], dim=-1)

                pred_noise = model(x_t, t)
                loss = F.mse_loss(pred_noise, noise)
                sum_loss += loss.item() * B
            val_done = time.time()
            # writer.add_scalar(
            #     f"Loss/val", sum_loss/len(dataloader_val.dataset), epoch)
            writer.add_scalars(
                f"Loss", {"val/average": sum_loss/len(dataloader_val.dataset)}, epoch)
            val_loss_list.append(sum_loss/len(dataloader_val.dataset))
            val_tensor = time.time()
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

                xts_tensor_list, steps, cd_loss = sample(feature_model,
                                                         model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, image_rgb=image_rgb, camera=camera, img_size=img_size, frame_token=frame_token)

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

                    path = f"{base_dir}/plots/{data_group}/{run_name}/sample_ep_{i_epoch:06d}_gt0_train-idx-{frame_tkn[:3]}.json"
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

                xts_tensor_list, steps, cd_loss = sample(feature_model,
                                                         model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, img_size=img_size, camera=camera, image_rgb=image_rgb, frame_token=frame_token)

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

                    path = f"{base_dir}/plots/{data_group}/basic_{method}_{N:04d}_MAN_1m_pvcnn_emb{pvcnn_embed_dim}/sample_ep_{i_epoch:06d}_gt0_val-idx-{frame_tkn[:3]}.json"
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
        # print(
        #     f"Epoch {epoch}/{epochs},Time = {time.time()-start_time:.2f}s, Train time = {train_done-start_time:.2f}s, Tensor time = {tensor_done-train_done:.2f}s, Val time = {val_done-val_start:.2f}s , after val = {val_tensor-val_done:.2f}s")
        st_time = time.time()

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
