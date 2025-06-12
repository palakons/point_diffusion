import numpy as np
import cv2
import argparse
from truckscenes import TruckScenes
from model.mypvcnn import PVC2Model
from contextlib import nullcontext
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
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
from man_dataset import MANDataset, custom_collate_fn_man,combine_perspective_cameras
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
from model.pvcnn.modules.pvconv import PVConv
from mymdm import MDM

from pytorch3d.renderer.cameras import PerspectiveCameras
from truckscenes_devkit.mydev.guided_diffusion.models.unet import EncoderUNetModelNoTime

local_feature_cache = {}


class SinusoidalTimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # T x dim


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, time_embed_dim=64, model_layers=[]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, 128),
            nn.ReLU()
        )

        self.net = nn.Sequential()
        for i, dim in enumerate(model_layers):
            if i == 0:
                self.net.append(nn.Linear(input_dim + 128, dim))
            else:
                self.net.append(nn.Linear(model_layers[i - 1], dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(model_layers[-1], output_dim))
        print("SimpleDenoiser input_dim", input_dim)
        print("SimpleDenoiser model_layers", model_layers)
        print("SimpleDenoiser output_dim", output_dim)

    def forward(self, x_t, t):
        t_embed = self.time_mlp(t)  # (B, 128)
        #expand to [B,N,128]
        t_embed = t_embed.unsqueeze(1).expand(x_t.shape[0], x_t.shape[1], -1)
        x = torch.cat([x_t, t_embed], dim=-1)
        return self.net(x)

def get_man_data(M, N, camera_ch, radar_ch, device, img_size, batch_size, shuffle, n_val=2, n_pull=10, flip_images=False, coord_only=True):
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
        double_flip_images=flip_images,
        coord_only=coord_only,)

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
        "steps": steps if isinstance(
            steps, list) else steps.detach().cpu().tolist(),
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


def sample(args, id_list, all_cond_signal, feature_model,unet_cond_model, model, scheduler, T, B, N, D, gt_normed, device, data_mean, data_std, camera, frame_token, image_rgb,depth_image=None):
    scheduler.set_timesteps(T)
    x_t = torch.randn(B, N, D).to(device)
    xts_tensor_list = []

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, leave=False, desc="Sampling"):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            x_t_cond = get_conditioning(args, device, id_list, all_cond_signal, x_t, feature_model,unet_cond_model, camera, frame_token,
                                        image_rgb, depth_image,point_mean=data_mean, point_std=data_std)

            noise_pred = model(torch.cat(
                [x_t, x_t_cond], dim=-1), t_tensor)

            x_t = scheduler.step(
                noise_pred, t, x_t).prev_sample
            # print("x_t", x_t.shape) #torch.Size([2, 3, 3])

            xts_tensor_list.append(x_t.cpu())


    xts_concat = torch.stack(xts_tensor_list[::10] + xts_tensor_list[-1:])

    time_step_concat = torch.cat(
        [scheduler.timesteps[::10], scheduler.timesteps[-1:]], dim=0)

    sampled = x_t.to(device)
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)

    unormed_sampled = sampled * data_std + data_mean
    unormed_gt = gt_normed * data_std + data_mean

    cd_final, _ = chamfer_distance(
        unormed_sampled[:,:3], unormed_gt[:,:3], batch_reduction=None)
    # print("cd_final", cd_final)
    if D==7:
        ave_distance = torch.mean(torch.norm(unormed_sampled[:, :,:3] - unormed_gt[:, :,:3], dim=-1), dim=-1)

        # print("unormed_sampled", unormed_sampled.shape) #[B,N,7]
        # print("unormed_gt", unormed_gt.shape)#[B,N,7]
        # print("ave_distance", ave_distance)

        # print("ave_distance", torch.norm(torch.tensor([[[1,0,0],[1,0,1]]]).float(), dim=-1))

        vrel_mse = torch.mean((unormed_sampled[:,:, 3:6] - unormed_gt[:,:, 3:6]) ** 2, dim=[1,2])
        # print("vrel_mse",vrel_mse)
        rcs_mse = torch.mean((unormed_sampled[:,:, 6:] - unormed_gt[:,:, 6:]) ** 2, dim=[1,2])
        # print("rcs_mse", rcs_mse)
        return xts_concat, time_step_concat, cd_final, ave_distance, vrel_mse, rcs_mse


    return xts_concat, [int(a) for a in time_step_concat], cd_final,None,None,None


def save_checkpoint(model, optimizer,unet_cond_model,
                    train_cd_list,
                    val_cd_list,
                    cd_epochs,
                    train_loss_list, val_loss_list, epoch, base_dir, config, run_name, code=None):
    checkpint_dir = os.path.join(
        base_dir, "checkpoints_man")
    proc_id = os.getpid()
    checkpoint_fname = os.path.join(
        checkpint_dir, f"cp_{datetime.utcnow().strftime(f'%Y-%m-%d-%H-%M-%S')}-{run_name.replace('/', '_') }_host{os.uname().nodename}_proc{proc_id}.pth")
    db_fname = os.path.join(
        base_dir, 'checkpoints_man', 'db.txt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'unet_cond_model_state_dict': unet_cond_model.state_dict(),
        'epoch': epoch,
        'train_cd_list': train_cd_list,

        'val_cd_list': val_cd_list,
        'cd_epochs': cd_epochs,
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'config': config,
        'code': code,
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
    assert embedding_dim > 2, f"embedding_dim({embedding_dim}) must be greater than 2"
    half_dim = embedding_dim // 2
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=device)
        * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for 
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(
        np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
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

def get_cached_local_feature(frame_tkn,feature_model,image_rgb,device):
    if frame_tkn in local_feature_cache:
        local_feature = local_feature_cache[frame_tkn]
    else:
        local_feature = feature_model(
            image_rgb.float().to(device), return_upscaled_features=False).to(device)
        # print("local_feature", local_feature.shape)  # 1, 384, 38, 38
        local_feature_cache[frame_tkn] = local_feature
    return local_feature
def flattened_conditioning(img_size, feature_model,  camera, frame_token, image_rgb, normed_points, device, raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1]), stat_factor=0,feature_first_dim=384):
    """
    Flattened conditioning for the model.
    """
    local_features = [get_cached_local_feature(frame_token[i],feature_model,image_rgb[i:i + 1],device) for i in range(len(frame_token))]  # list of local 

    local_features = torch.cat(local_features, dim=0).to(device)
    # print("local_features", local_features.shape)  # 2, 384, 38, 38
    if feature_first_dim ==0: #zero like 
        local_features = torch.zeros(local_features.shape[0], 384, local_features.shape[2], local_features.shape[3]).to(device)
    else:
        local_features = local_features[:, :feature_first_dim, :, :]  # 2, 384, 38, 38
    flattened_feature = local_features.reshape(local_features.shape[0], -1).unsqueeze(1).expand(
        normed_points.shape[0], normed_points.shape[1], -1) # 2, 4, 384*38*38
    return flattened_feature


def get_camera_conditioning(img_size, feature_model,  camera, frame_token, image_rgb, normed_points, device, raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1]), stat_factor=0):
    """
    points: (B, N, D) NIORMALIZED Point cloud data,


    DEFAULT PC^2 parameters
    raster_point_radius: float = 0.0075,
    raster_points_per_pixel: int = 1,
    bin_size: int = 0,
    scale_factor: float = 1.0

    """

    # return torch.zeros((normed_points.shape[0], normed_points.shape[1], 384)).to(device)

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

            if False:  # inspect stat from VITs
                small_feature = feature_model(image_rgb[i:i + 1].float().to(
                    device),        return_type='features',        return_upscaled_features=False).to(device)
                # small f shape torch.Size([1, 384, 38, 38])
                print("small f shape", small_feature.shape)
                # make state plot maybe box plot
                fig = plt.figure()
                x = range(small_feature.shape[1])
                y = small_feature.mean(dim=[0, 2, 3]).cpu().numpy()
                e = small_feature.std(dim=[0, 2, 3]).cpu().numpy()
                plt.errorbar(x, y, e, linestyle='None', marker='^')
                foutput = f'/home/palakons/from_scratch/vitsstat_{frame_tkn[:3]}.png'
                csv_output = f'/home/palakons/from_scratch/vitsstat.csv'
                plt.savefig(foutput)
                with open(csv_output, 'a') as f:
                    f.write(f"{frame_tkn},{y.tolist()},{e.tolist()}\n")
            local_feature_cache[frame_tkn] = local_feature
            # print("saved local_feature_cache", frame_tkn, local_feature.shape)

        feature_std = [0.554600779307215, 0.537929526836671, 0.832839469288984, 0.688226256184848, 0.452582867763333, 0.537518454778647, 0.464593682714164, 0.437203680675777, 0.614834816260369, 0.655601191992269, 0.542814605340389, 0.462566489675832, 0.744393020987382, 0.876776283223533, 0.473776484246442, 0.479791087846992, 0.594711949907883, 0.734973297405673, 0.677756093967993, 0.53951509742763, 0.644789558296267, 0.490651733810398, 0.524435484338827, 0.557896917712952, 0.533363231022826, 0.71959571881595, 0.821669035022517, 0.565855233702463, 0.500869936704265, 0.510585705364982, 0.533950558633835, 0.205956655726138, 0.952407599157279, 0.511780020773942, 0.56430255506086, 0.562207853710966, 0.795055679274598, 0.615666753839968, 0.945309950513202, 0.613559614039172, 0.590138292706903, 0.49895146991849, 0.52194562529096, 0.700932696257224, 0.645224403412526, 0.703786150253158, 0.579022142050683, 0.624623335179884, 0.541606998019367, 0.562395923668777, 0.712495231613632, 0.507601968362934, 0.515441312415464, 0.590177789793812, 0.567988672620836, 0.520565201662483, 0.510816627602666, 0.76818350259686, 0.488852958039427, 0.603074701372574, 0.46656750301952, 0.81194133499194, 0.687751090612731, 0.667133535250435, 0.624663717254817, 0.501100785251869, 0.620949447506688, 0.521222296530126, 1.67230050691576, 0.485441180163265, 0.589241415067557, 0.515164438549794, 0.520352712238719, 0.880896855214973, 1.16444350669608, 0.687580878316299, 0.913020985910151, 0.512359715196589, 0.652899539791035, 0.707458523798391, 0.509679594532527, 0.615546800382041, 0.547993458047345, 0.563178761327105, 0.631949227786975, 0.577033913196875, 0.563387787793224, 0.492701939753098, 0.581811630397851, 0.672342205937316, 0.471475458153929, 0.50704151131252, 0.503301523131877, 0.642306739259866, 0.451836903203987, 0.480716980219188, 0.562270515229471, 0.653803090548096, 0.562978052076063, 0.563346664415329, 0.532772206803312, 0.527431049080062, 0.4761023406452, 0.524083223476265, 0.566022437609096, 1.548152653029, 0.604865754147375, 0.656138367437757, 0.48468315425612, 0.622373694495396, 0.529181036978646, 0.929824320635805, 0.728397374784218, 0.564818159127398, 0.819068114911161, 0.42646464294041, 0.479548894061933, 0.804648623161829, 0.505984334382126, 0.484642699356241, 0.574588198405022, 1.08833634024418, 0.750882806559891, 0.513093266138995, 0.878275534733682, 0.644788941699198, 0.520722742778852, 0.546955621186042, 1.68160238613943, 0.618811511669013, 0.662992918038035, 0.476514516424947, 0.482469792667734, 0.551356392203334, 0.520986069622267, 0.421094755888871, 0.500634569523893, 0.664449111306439, 0.615617366779416, 0.57880152930854, 0.572787578518645, 0.570851195816962, 0.49013428695875, 0.552836404944352, 0.641925781385396, 0.4094272150653, 0.524628281438394, 0.856564104672934, 0.490164841339981, 0.645508999452972, 0.533623380562076, 0.623014565239808, 0.499013523449759, 0.545682134956882, 0.436540272875005, 0.618129903541916, 0.672531805111521, 0.591653041976553, 0.597381332551848, 0.49251697419277, 0.769636827824785, 0.664115094589816, 0.64581527344475, 0.493935518755869, 0.667263958195943, 1.87137467701686, 0.540378827374847, 0.59453176214772, 0.48859332805871, 0.624468576333967, 0.94131336614222, 0.670532968605627, 0.432959148015179, 0.609893937943585, 0.575034437746438, 0.642640775551616, 0.858139086274925, 0.523537133672446, 0.602505686392768, 0.528739318409639, 0.870830715835544, 0.513424884595948, 0.44377540533562, 0.598058128526848, 0.642629683623617, 0.613665341998579, 0.730120219556931, 0.568148340445044, 0.547615655006926, 0.53145645069097, 0.574715628363727, 0.514170328299472,
                       0.612641261520206, 0.565704651683949, 1.13245636480507, 0.777454476756735, 0.504563255417161, 0.527662170049688, 0.431552674864039, 0.593123691635424, 0.57412734087274, 0.904694471089732, 0.719175959643606, 0.630864273991396, 0.835916793527776, 0.523286647277629, 0.61445780317757, 0.590514119765582, 0.483685810905256, 0.705717538734291, 0.493100933751349, 0.611206301330545, 0.655086160833398, 0.620564858345275, 0.956640767047478, 0.688105997754872, 0.478751253103875, 0.669987778934299, 0.574186293907827, 0.507343785972242, 0.498126890554616, 0.53289511879493, 0.517586019812623, 0.730427838892392, 0.526928948549853, 0.526536079997553, 0.48831458546086, 0.529265078969169, 0.724748374666826, 1.09791063510496, 0.555084413719637, 0.762617807307714, 0.55180389162783, 0.478251157689463, 0.547460057719871, 0.624541493122867, 0.491159311320825, 0.662116624604175, 0.692167320175707, 0.647739737982363, 0.588025194686042, 0.516939450041867, 0.566471112847149, 0.705086424386061, 0.620102945951659, 0.63946946398484, 0.539014912078646, 0.681229091935792, 0.735602753309541, 0.612551584950629, 0.614330878972411, 0.692412034010566, 0.559939206493927, 0.532494290160364, 0.553348841248039, 1.01148365465637, 0.463048907663738, 0.564136037959149, 0.471639895181079, 0.824563627612978, 0.935867237653415, 0.670625440507263, 0.543092746137182, 0.516630946115443, 0.655923087682109, 0.654188351723103, 0.817503528934331, 0.495194648577435, 0.579224073159393, 0.621843938110739, 0.578064071571682, 0.664213593203764, 0.586763039855949, 0.530875859909773, 0.52979508125196, 0.652535741005965, 0.524113711407826, 0.647883721950918, 0.536738993572899, 0.782758339218088, 0.52136919043644, 0.635436489350748, 0.886456202056285, 0.578480236652527, 0.690974439074642, 0.666654493777065, 0.454023396161785, 0.568800197053621, 1.01924407285946, 0.484387595287388, 0.468497700182548, 0.645429180479539, 0.497374330452479, 0.582123983818825, 0.498510346386763, 0.437306589935678, 0.566052043939652, 0.536745148956718, 0.836548292208024, 0.56888185501007, 0.769100324537892, 0.662762039124775, 0.573452520217722, 0.558909744814009, 0.486306120642619, 0.501285443009788, 0.750101173623306, 0.455186191605655, 0.778484193655075, 0.619117065378341, 0.53497685601224, 0.485536745871903, 0.845468638699109, 0.611285051894612, 0.441928331945319, 0.509283406031377, 0.445047167212771, 0.536150691318069, 0.542703947184545, 0.507821358342099, 0.567193649047402, 0.643185396221863, 0.695187938501485, 1.09485032880718, 0.537807280961283, 0.626833033811764, 0.573065803835496, 0.66059170874435, 0.331773219624289, 0.569081773547911, 0.451976450902335, 0.541241706186236, 0.88972055344132, 0.561618748858036, 0.620746092829937, 0.540066211903894, 0.66815777921847, 0.561914618867896, 0.54331900219027, 0.542688114339779, 0.523208028896104, 0.603413412084337, 0.53813414603346, 0.524032670830881, 0.523408245094213, 0.491012485637503, 0.58999431661377, 0.51054910961674, 0.546471673704773, 0.538977681276544, 0.626401930253993, 0.45114385060188, 0.671521226110925, 0.676284813889698, 0.507504168836228, 0.607263100365296, 0.786635769636837, 0.480546838358651, 0.580454413976386, 0.538946076305407, 0.652781187454422, 0.792867956031318, 0.750708484413498, 0.59981988538097, 0.903871064425602, 0.657712662806385, 0.59067164583514, 0.685034905444457, 0.664816433629482, 0.683258212960194, 0.467186489486786, 0.529094731170526, 0.704851080666024, 0.587151074461343, 0.701697518994184, 0.796182234221306, 0.550139815600896, 0.654282282306366, 0.68359494783661, 0.506101286276429, 0.460845698109934, 0.711649158215783, 0.487962830555745, 0.528769730909249]
        feature_mean = [0.421777687289497, 0.0189073552292856, -0.0995346066192723, 0.565484117377888, 0.441723788326436, -0.108666602522135, -0.356904604218222, 0.0966046047396954, 0.778061601248654, -0.374680158766833, -0.152837291359901, -0.0769238988445562, -1.21274781227111, 1.05199857191605, -0.490998479453, 0.443240184675563, -1.04878332398154, 0.29465274174105, 0.188328082588585, 0.256385975263335, -0.580074800686402, -0.444082157178358, 0.161515402861616, -0.85429742119529, 1.17128594355149, 1.01965436610308, 0.59930682182312, 0.224380382082678, -0.892479538917541, 0.790715856985612, 0.300800564614209, 0.00514381645586002, -0.234960117123343, -0.66159718686884, 0.146795686673034, 0.286257856271483, -0.4806390106678, 0.774282921444286, 0.143041005188768, -0.000775707716291609, -1.42830793424086, -1.06234429641203, 0.643289880319075, 0.076742246408354, -0.756370219317349, 0.0898628498596899, -1.58089023286646, -0.190413393757559, 0.573497517542405, 1.02234416116367, 1.4869450005618, -0.445749838243831, 0.624607162042097, -0.00027644448511479, 0.331359974362633, 0.495943928306753, 0.08205357092348, 0.218869465318593, 0.354780717329545, 0.481288647109811, 0.832005728374828, 0.463102294640107, -1.02428408644416, 0.252535738728263, 0.71551749381152, -0.100442851300944, -2.59949036078019, -0.404046210375699, -0.884089269421317, -0.00388156038454987, -0.632668245922435, 0.508617750623009, -0.420100913806395, -0.800552481954748, -1.25346048311753, 1.14976967464793, 0.227126645770939, 0.573168342763727, -0.467174123633991, 1.46247666532343, -0.306479084220799, 0.557158995758403, -1.40018227967348, -0.221813715317032, 0.614042769778858, 1.09720938855951, 1.49746463515541, -0.716294716704975, 0.403976765545931, -1.12212851914492, -1.06398805704983, -1.08543429591438, -0.509612893516367, 0.409335296262394, 0.959251685576004, -0.717626669190146, -0.71592518416318, -0.3250126730312, 0.0545013398778708, -0.499477730555968, -0.375995123928243, 0.732472631064328, 0.315157618034969, -0.679124805060299, 1.52181938561526, -0.151895708658478, 0.51705559275367, 0.16978399049152, -1.18136978149413, 0.604415833950042, 1.04762237180363, 2.89772696928544, 0.642916592684659, 0.448308272795243, 0.523191996596076, -2.24355589259754, 1.48607392744584, 0.786534807898781, -0.40287924896587, 0.0512661997398192, 0.587903922254389, 0.277461286295544, -0.738653459332205, 0.513643752444874, 2.13697537508878, 0.939538104967637, -0.0814818841489878, -1.59736467491496, -5.18619567697698, -0.44851451028477, -0.228693941777402, -0.369210804050619, 0.365206066857684, 0.650102181868119, -0.803491711616516, -0.532293585213748, -0.201308051293546, 0.412144216624173, -0.393940936435352, 1.10410324009982, 0.098650220104239, -0.0660845371471208, 1.14419175278056, -0.17324044487693, -0.0228313033215024, 0.251477559859102, -0.374627221714366, 0.249604137106375, 0.712218338792974, -0.433897674083709, -0.204857629808512, 0.526087414134632, 0.562428550286726, -0.315324924208901, -0.400966102426702, 0.589831325140866, 0.0210791428372348, -0.176420122385025, 0.546601219610734, 1.17288706519387, 0.327926341782916, -0.14711055565964, 0.121763433922421, -1.72036310759457, 0.258670349012721, -5.47392892837524, -0.0995844975113866, -0.123713026331229, -0.0497813052074475, 0.425344784151424, 0.274083371866833, 0.0809571949595752, -1.29930041053078, 1.73344296758825, 0.503374397754668, -0.206498981876806, 0.222842547026547, 0.176804549992084, -0.348824392665516, -0.373038619756698, -0.105019704184749, -0.0943091308528725, -0.477882393381812, -0.0350356834347952, -1.17898781733079, -0.243675860491665, 0.597651199861006, 0.761674171144312, 0.00463196965442462, -1.69159880551424, 0.0379811872474172, -
                        1.08851532502607, 1.40051312880082, -0.699713132598183, 1.71449094468897, -0.723799076947298, -0.0120831045576117, 0.908208072185516, 0.628958349878137, -0.765700042247772, -1.03319495374506, -0.180859078737822, -0.637368142604827, 1.50938520648262, 0.79251649162986, -0.0792580277405001, -0.673354582353071, 0.214839676564389, -0.355802126906134, 1.03219702568921, -0.732817563143643, -0.394166380167007, -0.0526209548962386, -0.21178913116455, 0.0850402777167882, 1.11553752422332, 0.551978571848435, -0.360704668543555, -0.713010890917344, 1.52760311690243, -0.262620891359719, 0.665216294201937, -0.800644592805342, -1.3425395488739, 1.1695739030838, -0.0594951767813075, -0.243145560676401, 1.03334767710078, -0.581391627138311, 0.517951783808795, -0.537768713452599, 0.536346646872433, -0.0588506145233457, -0.577887735583565, -0.131241666322404, -0.669529394669966, -0.0847837924957274, 0.586357214234092, -0.513572974638505, -1.36442070657556, -1.31314436955885, 0.557277909734032, -0.258777840570969, 0.0270111573521386, -0.0397197174077683, -1.00250174782493, -0.449676765636964, -1.21949599005959, -0.301929983225735, -0.0346633776683699, 0.631520363417538, -0.534365293654528, -0.283336715264753, 0.0234901154921812, 0.579537483778866, 0.238388728011738, -0.124873600900173, 1.63026039166883, -0.613270586187189, 3.25391275232488, 1.65676919980482, 0.850932652300053, 0.747917505827817, -1.32377887855876, -0.200187177820639, -0.913858153603293, -0.631852502172643, 0.692758841948075, 1.10261060974814, 0.434704062613574, 0.446560859680175, -0.639799562367526, 1.25077674605629, 0.651713934811678, 0.787934287027879, 0.226934608410705, 0.380652579394253, -1.06103372573852, 1.06220962784507, -0.251439261165532, -0.235914103009484, -0.612384135072881, -0.209325054491108, 0.224179855801842, 0.376402369954369, 0.629222138361497, 1.56410290978171, 0.115386533804915, -1.72617764906449, -1.62868805365129, 0.763167841867967, 0.31407265771519, 0.371359711343591, -1.27027526768771, -0.425340928814627, -0.501113376834175, 2.00023218718441, -0.0179394854402, -0.488821146163073, -0.354322609576312, -1.06758773326873, -0.373636757785623, 0.311607433990998, -1.18474172462116, 0.895847699858925, 0.137461977926167, -0.374713410030711, 0.640658389438282, -0.269959666512229, -1.02525901252573, -1.62485052238811, 0.594793081283569, -0.450251408598639, 1.15334000370719, 0.0525070877576416, 0.156775041060014, 0.431551299311898, -0.90034686977213, -0.463377811691977, 0.0650009793991391, -1.11583976853977, -1.11231204596432, 1.09098522229628, -1.51357526128942, 1.21786909753626, -0.271727693351832, -0.3392025151036, 1.63211283900521, -0.698503954844041, 0.392656873572956, -1.3857634934512, -1.12882099368355, -1.17452721162275, -0.338672540404579, 0.124583656306971, 0.483165933327241, -0.718958274884657, 0.54862522537058, 0.121930671347813, 1.04927043481306, 0.590841084718704, -0.08817688324912, 0.428348010236566, 0.172704559158195, 0.257355069572275, 0.732194759628989, -1.26525141976096, 0.748026165095242, -0.017966709353707, 0.435383707284927, -0.497205796566876, 1.44420852444388, -1.25229209119623, 1.28680310466072, -1.19660906358198, 0.0729191000150008, 1.5556114912033, -0.328946501016616, -0.0920295546000653, -0.214668563821099, 0.568817057392813, -0.313449274409901, -0.826672521504489, 0.68809058449485, -1.14215156165036, 0.28966334733096, 0.779496436769312, 0.317960397763685, -0.623332906853069, -0.509313984350724, -0.0773121758618137, -0.00130324065685274, 0.371918420899998, -0.305517341602932, 0.419162406162782, 0.770624794743277, 0.806475568901408, 1.50114837559786, -0.720405692403966, 0.110616782172159, -0.225093380971388, 0.394939904863184, 0.726588384671644, 0.305522161451252]
        feature_mean = torch.tensor(feature_mean).to(
            device).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        feature_std = torch.tensor(feature_std).to(
            device).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)

        adjusted_feature = (local_feature-feature_mean *
                            stat_factor) / (1 + stat_factor*(feature_std-1))

        local_features.append(adjusted_feature)
        # print("local_features", adjusted_feature.mean(), adjusted_feature.std())
        # print("all close", torch.allclose(adjusted_feature, local_feature))
        # exit()
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

    # Associate points with features via rasterization
    fragments = rasterizer(Pointclouds(normed_points * point_std.to(
        device).float() + point_mean.to(device).float()).to(device))  # (B, H, W, R)
    # torch.Size([1, 618, 618, 1]) ,,,([1, 618, 618, 1])
    # print("fragments", fragments.idx.shape, fragments.zbuf.shape)
    fragments_idx: torch.Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    # torch.Size([1, 618, 618, 1])
    # print("visible_pixels", visible_pixels.shape)
    # count visible pixels
    num_visible_pixels = visible_pixels.sum()

    if False:
        print("num_visible_pixels", num_visible_pixels.item(),
              f"or {num_visible_pixels.item()/(H*W)*100:.2f}%", W, H)

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
    if False:  # normzalizing teh projected feature
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


def get_argparse():

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
    parser.add_argument("-pvcnnwm", "--pvcnn_width_multiplier", type=float, default=1,
                        help="Width multiplier for PVCNN")
    parser.add_argument("-pvcnnvrm", "--pvcnn_voxel_resolution_multiplier", type=float, default=1,
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
                        help="Flip images or not", default=True) #problem, if we dont wanna flip images
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
    parser.add_argument("-statf", "--stat_factor", type=float, default=0.0,
                        help="Statistical factor for conditioning: 0 means no scaling, 1 means full scaling to 0-mean and 1-std, channel-wise")
    parser.add_argument("-pv_drop", "--pvcnn_dropout", type=float, default=0.1,
                        help="Dropout rate for PVCNN")
    parser.add_argument("-cond", "--cond_type", type=str, default="camera",
                        help="Conditioning type: camera, zero, id or empty")
    parser.add_argument("-data_group", "--data_group", type=str, default="pc2_man",
                        help="Data group where tensorboard and plots are saved")
    parser.add_argument("-model", "--model_name", type=str, default="pvcnn",
                        help="Model name for the experiment")
    parser.add_argument("-noise_img", "--noise_image", action='store_true', default=False,
                        help="Use noise image for conditioning")
    #stackedpvcnnn kernel size 3,
    parser.add_argument("-pvc_kernel", "--pvcnn_kernel_size", type=int, default=3,
                        help="Kernel size for PVCNN convolution layers")
    # pvc_attention
    parser.add_argument("-pvc_att", "--pvcnn_attention", action='store_true', default=False,
                        help="Use attention in PVCNN")
    #with_se
    parser.add_argument("-pvc_se", "--pvcnn_se", action='store_true', default=False,
                        help="Use Squeeze-and-Excitation in PVCNN")
    #with_se_relu
    parser.add_argument("-pvc_se_relu", "--pvcnn_se_relu", action='store_true', default=False,  
                        help="Use ReLU activation in Squeeze-and-Excitation in PVCNN")
    #normalize
    parser.add_argument("-pvc_norm", "--pvcnn_normalize", action='store_true', default=False,
                        help="Use normalization in PVCNN")  
    #eps
    parser.add_argument("-pvc_eps", "--pvcnn_eps", type=float, default=0,
                        help="Epsilon value for normalization in PVCNN")
    #num_blocks 
    parser.add_argument("-pvc_blocks", "--pvcnn_num_blocks", type=int, default=2,
                        help="Number of blocks in PVCNN")
    parser.add_argument("-resume", "--resume_from_checkpoint", action='store_true', default=False,
                        help="Resume training from checkpoint")
            
    #duplication augnentation minibathc factor
    parser.add_argument("-dup", "--duplication_augmentation_factor", type=int, default=1,
                        help="Duplication augmentation factor for the batch size")
    # get_all data, not coord_only
    parser.add_argument("-getvr", "--get_all_data", action='store_true', default=False,
                        help="Get all data from the dataset, not only coordinates")

    parser.add_argument("-mdm_size_mult", "--mdm_size_multiplier", type=float, default=1.0,
                        help="Multiplier for the size of the MDM model")
    #shuffle images
    parser.add_argument("-shuffle_img", "--shuffle_images", action='store_true', default=False, 
                        help="Shuffle images in the batch")



    args = parser.parse_args()
    return args


def get_conditioning(args, device, id_list, all_cond_signal, x_t_data, feature_model, unet_cond_model, camera, frame_token, image_rgb, depth_image,  raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1])):
    if args.cond_type == "camera":
        image_to_be_conditioned =  torch.randn_like(image_rgb, device=device) if args.noise_image else image_rgb
        return get_camera_conditioning(
            args.img_size,
            feature_model, camera, frame_token, image_to_be_conditioned, x_t_data,  device=device, raster_point_radius=raster_point_radius, raster_points_per_pixel=raster_points_per_pixel, bin_size=bin_size, scale_factor=scale_factor, point_mean=point_mean, point_std=point_std, stat_factor=args.stat_factor)
    if args.cond_type == "zero":
        return torch.zeros((x_t_data.shape[0], x_t_data.shape[1], args.pc2_conditioning_dim)).to(device)
    if args.cond_type == "id":
        assert id_list is not None and all_cond_signal is not None, "ID conditioning requires id_list and all_cond_signal to be provided"
        frame_idx = torch.tensor([id_list[token] for token in frame_token],
                                 device=device)
        cond = all_cond_signal[frame_idx].float().to(
            device).unsqueeze(1).repeat(1, args.num_points, 1)
        return cond
    if "flatten" in args.cond_type:

        if args.cond_type == "flatten" :
            feature_first_dim = 384
        elif args.cond_type == "flatten_zero":
            feature_first_dim = 0
        else:
            feature_first_dim = int(args.cond_type.split("_")[-1])
        return flattened_conditioning(
                args.img_size,
                feature_model, camera, frame_token, image_rgb, x_t_data,  device=device, raster_point_radius=raster_point_radius, raster_points_per_pixel=raster_points_per_pixel, bin_size=bin_size, scale_factor=scale_factor, point_mean=point_mean, point_std=point_std, stat_factor=args.stat_factor, feature_first_dim=feature_first_dim)
                
    if args.cond_type == "empty":
        out= torch.empty((x_t_data.shape[0], x_t_data.shape[1], 0)).to(device)
        return out
    
    if "unet" in args.cond_type :
        # print("Using UNet conditioning")
        # print("input image_rgb shape", image_rgb.shape) #torch.Size([1, 3, 618, 618])
        st_time = time.time()
        #print devices 
        # print(" model device", next(unet_cond_model.parameters()).device)
        # print(" image_rgb device", image_rgb.device)
        # print(args.cond_type,"image_rgb shape", image_rgb.shape, "device", image_rgb.device, "type", image_rgb.dtype)
        # print("depth_image shape", depth_image.shape, "device", depth_image.device, "type", depth_image.dtype)
        # unet_depth image_rgb shape torch.Size([2, 3, 618, 618]) device cuda:0 type torch.float32
        # depth_image shape torch.Size([2, 618, 618]) device cuda:0 type torch.float32     
        if args.cond_type == "unet":
            cond_unet_input = image_rgb
        elif args.cond_type == "unet_depth":
            cond_unet_input =depth_image.unsqueeze(1) 
        elif args.cond_type == "unet_imgdepth":
            cond_unet_input = torch.cat([image_rgb, depth_image.unsqueeze(1)], dim=1) # torch.Size([2, 4, 618, 618])
        out = unet_cond_model(cond_unet_input).to(device)  # (B, N, D)
        et_time = time.time()
        # print(f"UNet conditioning time for {args.cond_type} : {et_time - st_time:.4f} seconds")
        # print("unet_cond_model output shape", out.shape) #unet_cond_model output shape torch.Size([1, 128])
        #expand to (B, N, 128)
        out = out.unsqueeze(1).repeat(1, x_t_data.shape[1], 1)
        return out



def train_val_one_epoch(args, dataloader, model, optimizer, scheduler, feature_model, unet_cond_model,operation, epoch, id_list, all_cond_signal,   device, writer, loss_list):

    if operation == "train":
        model.train()
        if unet_cond_model is not None:
            unet_cond_model.train()
        cm = nullcontext()
    elif operation == "val":
        model.eval()
        if unet_cond_model is not None:
            unet_cond_model.eval()
        cm = torch.no_grad()
    else:
        raise ValueError(f"Unknown operation {operation}")

    with cm:
        sum_loss = 0
        for i_batch, batch in enumerate(dataloader):
            (depths,
                radar_data,
                camera,
                image_rgb,
                frame_token,
                npoints,
                npoints_filtered) = batch
        
            #if shuffle_images is True, shuffle the images in the batch
            if args.shuffle_images and operation == "train":
                if args.batch_size >1:
                    # print("Shuffling images in the batch")
                    shuffle_indices = torch.randperm(image_rgb.size(0))
                    print(f"Shuffle_indices {shuffle_indices}")
                    image_rgb = image_rgb[shuffle_indices]
                    depths = depths[shuffle_indices]
                    frame_token = [frame_token[i] for i in shuffle_indices]
                    camera = PerspectiveCameras(
                        focal_length=camera.focal_length[shuffle_indices],
                        principal_point=camera.principal_point[shuffle_indices],
                        R=camera.R[shuffle_indices],
                        T=camera.T[shuffle_indices],
                        image_size=camera.image_size[shuffle_indices],
                    )
                else:
                    print("No use shuffling images in the batch, batch size is 1")


            # print(f"shape radar_data {radar_data.shape}, image_rgb {image_rgb.shape}, frame_token {len(frame_token)} npoints {npoints}, npoints_filtered {npoints_filtered}")
            
            image_rgb = image_rgb.to(device)
            depths = depths.to(device)
            radar_data = radar_data.float().to(device)
            point_mean = dataloader.dataset.dataset.data_mean.to(device).float()
            point_std = dataloader.dataset.dataset.data_std.to(device).float()

            # print(f"shape radar_data {radar_data.shape}, image_rgb {image_rgb.shape}, frame_token {len(frame_token)} npoints {npoints}, npoints_filtered {npoints_filtered} point_mean {point_mean.shape}, point_std {point_std.shape}")
            #shape radar_data torch.Size([2, 4, 3]), 
            # image_rgb torch.Size([2, 3, 618, 618]), 
            # frame_token 2 
            # npoints tensor([800, 800]), 
            # npoitsnts_filtered tensor([585, 569]) 
            # point_mean torch.Size([3]), 
            # point_std torch.Size([3])

            # print("radar_data", radar_data)
            # print("image_rgb", image_rgb[:,:,:3,:3 ]) #torch.Size([2, 3, 3, 3])
            # print("depths", depths[:,:3,:3]) #torch.Size([2, 3, 3])
            # print("frame_token", frame_token)#frame_token ['deb7b3f332f042d49e7636d6e4959354', '32d2bcf46e734dffb14fe2e0a823d059']
            
            if args.cond_type!= "camera":
                x_t_cond = get_conditioning(args, device, id_list, all_cond_signal, radar_data,
                                        feature_model, unet_cond_model,camera, frame_token, image_rgb,   depths,point_mean=point_mean, point_std=point_std)
                #duplicate duplication_augmentation_factor times dim 0
                x_t_cond = x_t_cond.repeat(
                    args.duplication_augmentation_factor, 1, 1) if operation == "train" else x_t_cond
                    
            if operation == "train": # duplication_augmentation_factor
                # print(f"camera focal length {camera.focal_length.shape}, camera principal point {camera.principal_point.shape} R {camera.R.shape}, T {camera.T.shape} I {camera.image_size.shape}")
                # print(f"radar_data shape {radar_data.shape}, image_rgb shape {image_rgb.shape}, frame_token {len(frame_token)} npoints {npoints}, npoints_filtered {npoints_filtered}")

                radar_data = radar_data.repeat(
                    args.duplication_augmentation_factor, 1, 1)

                camera = combine_perspective_cameras([camera]*args.duplication_augmentation_factor)

                image_rgb = image_rgb.repeat(
                    args.duplication_augmentation_factor, 1, 1, 1)
                frame_token *=  args.duplication_augmentation_factor
                npoints = torch.tensor(list(npoints) * args.duplication_augmentation_factor)
                npoints_filtered = torch.tensor(
                    list(npoints_filtered) * args.duplication_augmentation_factor)
                
                # print(f"camera focal length {camera.focal_length.shape}, camera principal point {camera.principal_point.shape} R {camera.R.shape}, T {camera.T.shape} I {camera.image_size.shape}")
                # print(f"radar_data shape {radar_data.shape}, image_rgb shape {image_rgb.shape}, frame_token {len(frame_token)} npoints {npoints}, npoints_filtered {npoints_filtered}")
                
            # print("radar_data", radar_data)
            # print("camera length", camera.focal_length)
            # print("camera principal point", camera.principal_point)
            # print("camera R", camera.R)
            # print("camera T", camera.T)
            # print("image_rgb", image_rgb.shape) #torch.Size([2, 3, 3, 3])
            # print("Frame tokens", frame_token)

            B, N, D = radar_data.shape


            x_0_data = radar_data

            noise = torch.randn_like(x_0_data)
            tv_step_t = torch.randint(
                0, args.T, (B,), device=device)  # uniform
            x_t_data = scheduler.add_noise(x_0_data, noise, tv_step_t)

            if args.cond_type == "camera":
                x_t_cond = get_conditioning(args, device, id_list, all_cond_signal, x_t_data,
                                        feature_model, unet_cond_model,camera, frame_token, image_rgb,   depths,point_mean=dataloader.dataset.dataset.data_mean.to(device).float(), point_std=dataloader.dataset.dataset.data_std.to(device).float())

            x_t = torch.cat(
                [x_t_data, x_t_cond], dim=-1)

            # print(f"noise shape {noise.shape}")
            # print(f"tv_step_t {tv_step_t.shape}, {tv_step_t}")
            # print(f" x_t_data {x_t_data.shape}, x_0_data {x_0_data.shape}, x_t_cond {x_t_cond.shape}, x_t {x_t.shape}")

            # noise shape torch.Size([4, 4, 3])
            # tv_step_t torch.Size([4]), tensor([ 3, 69,  0, 24], device='cuda:0')
            # x_t_data torch.Size([4, 4, 3]), x_0_data torch.Size([4, 4, 3]), x_t_cond torch.Size([4, 4, 128]), x_t torch.Size([4, 4, 131])
            if operation == "train":
                optimizer.zero_grad()

            pred_noise = model(x_t, tv_step_t)
            # print(f"shapes noise {noise.shape}, pred_noise {pred_noise.shape}, x_t {x_t.shape}, x_0_data {x_0_data.shape}, x_t_cond {x_t_cond.shape}")
            loss = F.mse_loss(pred_noise, noise)

            sum_loss += loss.item() * B

            # print(f"pred_noise {pred_noise.shape}")
            # print(f"loss {loss.item():.4f}")
            # print("sum loss", sum_loss)

            # exit()
            if operation == "train":
                loss.backward()
                optimizer.step()
        ave_loss = sum_loss / len(dataloader.dataset)
        writer.add_scalars(
            f"Loss", {f"{operation}/average": ave_loss}, epoch)
        loss_list.append(ave_loss)
        return ave_loss, loss_list


def sample_cd_save(args, model, scheduler, dataloader, save_key, feature_model,unet_cond_model, device, epoch, id_list, all_cond_signal, run_name, exp_config, writer, cd_list):
    """save_key: train or val"""
    data_mean = dataloader.dataset.dataset.data_mean.to(device).float()
    data_std = dataloader.dataset.dataset.data_std.to(device).float()
    sum_cd = 0
    sum_ave_distance = 0
    sum_vrel_mse = 0
    sum_rcs_mse = 0
    for i_batch, batch in enumerate(dataloader):
        (depths,
            radar_data,
            camera,
            image_rgb,
            frame_token,
            npoints,
            npoints_filtered) = batch
        image_rgb = image_rgb.to(device)
        depths = depths.to(device)
        B, N, D = radar_data.shape
        x_0_data = radar_data.float().to(device)

        xts_tensor_list, steps, cd_loss,ave_distance, vrel_mse, rcs_mse = sample(args, id_list, all_cond_signal, feature_model, unet_cond_model.eval() if unet_cond_model is not None else None,
                                                 model.eval(), scheduler, args.T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, image_rgb=image_rgb, 
                                                 depth_image=depths, camera=camera, frame_token=frame_token)

        sum_cd += cd_loss.sum().item()
        if D==7:
            sum_ave_distance += ave_distance.sum().item()
            sum_vrel_mse += vrel_mse.sum().item()
            sum_rcs_mse += rcs_mse.sum().item()

        for i_result, frame_tkn in enumerate(frame_token):
            xts = xts_tensor_list[:, i_result, :, :]
            cd = cd_loss[i_result]

            path = f"{args.base_dir}/plots/{args.data_group}/{run_name}/sample_ep_{epoch:06d}_gt0_{save_key}-idx-{frame_tkn[:3]}.json"
            save_sample_json(path, epoch, x_0_data[i_result].unsqueeze(
                0), xts, steps,  cd=cd.item(), data_mean=data_mean, data_std=data_std, config=exp_config)
            writer.add_scalars(
                f"CD", {f"{save_key}/{frame_tkn[:3]}": cd.item()}, epoch)
            if D==7:
                writer.add_scalars(
                    f"ave_distance", {f"{save_key}/{frame_tkn[:3]}": ave_distance[i_result].item()}, epoch)
                writer.add_scalars(
                    f"vrel_mse", {f"{save_key}/{frame_tkn[:3]}": vrel_mse[i_result].item()}, epoch)
                writer.add_scalars(
                    f"rcs_mse", {f"{save_key}/{frame_tkn[:3]}": rcs_mse[i_result].item()}, epoch)
            if frame_tkn not in cd_list:
                cd_list[frame_tkn] = []
            cd_list[frame_tkn].append(cd.item())
    cd_ave=sum_cd / len(dataloader.dataset)
    if D==7:
        ave_distance = sum_ave_distance / len(dataloader.dataset)
        vrel_mse = sum_vrel_mse / len(dataloader.dataset)
        rcs_mse = sum_rcs_mse / len(dataloader.dataset)
        writer.add_scalars(
            f"mean_distance", {f"{save_key}/average": ave_distance}, epoch)
        writer.add_scalars(
            f"vrel_mse", {f"{save_key}/average": vrel_mse}, epoch)
        writer.add_scalars(
            f"rcs_mse", {f"{save_key}/average": rcs_mse}, epoch)

    writer.add_scalars(
                f"CD", {f"{save_key}/average": cd_ave}, epoch)
    return cd_ave, cd_list


class StackedPVConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, resolution=32, num_blocks=2, 
    num_classes=3,attention=False,
                 dropout=0.0, with_se=True, with_se_relu=True, normalize=True, eps=0,
                 ablate_pvcnn_mlp: bool = False,
                 ablate_pvcnn_cnn: bool = False):
        """ with_se=True for pc2 paper
        with_se_relu=True for pvcnn paper
        normalize=True for pvcnn paper
        eps=0 for numerical stability,per pc2 paper
        kernel_size=3 for pvcnn paper
        """
        super().__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(PVConv(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                resolution,
                dropout=dropout,
                attention=attention,
                with_se=with_se,
                with_se_relu=with_se_relu,
                normalize=normalize,
                eps=eps,
                ablate_pvcnn_mlp=ablate_pvcnn_mlp,
                ablate_pvcnn_cnn=ablate_pvcnn_cnn
            ))
        self.blocks = nn.ModuleList(layers)
        self.out_proj = nn.Linear(out_channels, num_classes)  # Optional: final linear
        self.embed_dim =128
        print("stacked PVC", 
              f"in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, resolution={resolution}, num_blocks={num_blocks}, num_classes={num_classes}, attention={attention}, dropout={dropout}, with_se={with_se}, with_se_relu={with_se_relu}, normalize={normalize}, eps={eps}, ablate_pvcnn_mlp={ablate_pvcnn_mlp}, ablate_pvcnn_cnn={ablate_pvcnn_cnn}")

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
        inputs=inputs.transpose(1, 2)
        t_emb = get_timestep_embedding(
            self.embed_dim, t, inputs.device).float()

        t_emb = t_emb[:, :, None].expand(-1, -1, inputs.shape[-1])
        coords = inputs[:, :3, :].contiguous()  # (B, 3, N)
        x = inputs  # (B, 3 + S, N)

        for block in self.blocks:
            x, coords, temb = block((x, coords, t_emb))
        # print("x", x.shape)
        x = self.out_proj(x.transpose(1, 2))
        return x # (B, N, num_classes)

def get_model(args,D, device):

    if args.cond_type == "camera":
        cond_dim = 384
    elif "flatten" in args.cond_type:

        if args.cond_type == "flatten":
            feature_first_dim = 384
        elif args.cond_type == "flatten_zero":
            feature_first_dim = 384
        else:
            feature_first_dim = int(args.cond_type.split("_")[-1])
        cond_dim = feature_first_dim*((args.img_size//16)**2)
        print("all feature_first_dim, flatten conditioning", feature_first_dim) 
    elif args.cond_type == "id":
        cond_dim = args.pc2_conditioning_dim
    elif args.cond_type == "empty":
        cond_dim = 0
    elif args.cond_type == "zero":
        cond_dim = args.pc2_conditioning_dim
    elif "unet" in args.cond_type :
        cond_dim = 128

    if args.model_name == "pvcnn":
        return  PVC2Model(
            in_channels=D+cond_dim,
            out_channels=D,
            embed_dim=args.pvcnn_embed_dim,
            dropout=args.pvcnn_dropout,
            width_multiplier=args.pvcnn_width_multiplier,
            voxel_resolution_multiplier=args.pvcnn_voxel_resolution_multiplier,
            ablate_pvcnn_mlp=args.ablate_pvcnn_mlp,
            ablate_pvcnn_cnn=args.ablate_pvcnn_cnn,
            natural_cond_dim=args.pc2_conditioning_dim
        ).to(device)
    elif args.model_name == "mlp2048":
        mlp_layers = [2048, 1024]
        return SimpleDenoiser(input_dim=D+cond_dim, output_dim= D,
                           time_embed_dim=args.pc2_conditioning_dim, model_layers=mlp_layers).to(device)
    elif args.model_name == "stacked_pvcnn":
        out_channels, num_blocks,voxel_resolution=32,args.pvcnn_num_blocks,32
        return StackedPVConvModel(
            in_channels=D+cond_dim,
            out_channels=int(out_channels*args.pvcnn_width_multiplier),
            kernel_size=args.pvcnn_kernel_size,
            resolution=int(voxel_resolution*args.pvcnn_voxel_resolution_multiplier),
            num_blocks=num_blocks,
            num_classes=D,
            dropout=args.pvcnn_dropout,
            attention=args.pvcnn_attention, #fasle for pc2 paper
            with_se=args.pvcnn_se, #True,  # pc2 paper
            with_se_relu=args.pvcnn_se_relu, #True,  # pvcnn paper
            normalize=args.pvcnn_normalize, #True,  # pvcnn paper
            eps=args.pvcnn_eps,  #0, numerical stability, per pc2 paper
            ablate_pvcnn_mlp=args.ablate_pvcnn_mlp,
            ablate_pvcnn_cnn=args.ablate_pvcnn_cnn
        ).to(device)

    elif args.model_name == "mdm":
        #import MDM /ist-nas/users/palakonk/singularity/home/palakons/from_scratch/truckscenes-devkit/mydev/guided_diffusion/models/mdm.py

        #condition per frame!

        if args.cond_type in ["camera", "empty"]:
            raise ValueError(f"Conditioning type {args.cond_type} is not supported for MDM model. Use flatten or id conditioning instead.")

        print("mdm Using MDMModel with cond_dim", cond_dim)

        # max_pc_len=128, 
        # in_channels=3, 
        # out_channels=3,
        # num_heads=6, 
        # ff_size=2048,
        # model_channels=512,
        # num_layers=3,
        # condition_dim=2,
        # dropout=0.1,
        return MDM(
            in_channels=D,
            out_channels=D,
            num_heads=int(8* args.mdm_size_multiplier), # 6 for small, 8 for medium, 12 for large
            ff_size=int(2048 * args.mdm_size_multiplier), # 2048 for small, 4096 for medium, 8192 for large
            model_channels=int(512 * args.mdm_size_multiplier), # 512 for small, 1024 for medium, 2048 for large
            num_layers=int(6 * args.mdm_size_multiplier), # 6 for small, 12 for medium, 24 for large
            condition_dim=cond_dim,
            dropout=args.pvcnn_dropout
        ).to(device)

            
    else:
        raise ValueError(f"Unknown model {args.model_name}")

def match_args(args, ckp_config,except_keys=["epochs", "method", "base_dir", "vis_freq", "data_group"]):
    """Check if the args match the checkpoint config"""
    for key, value in args.items():
        if key in except_keys:
            continue
        if key == "noise_image" and key not in ckp_config and value is False:
            continue
        if key not in ckp_config or ckp_config[key] != value:
            return False
    return True

def get_checkpoint_fname(args, db_fname):
    assert  os.path.exists(db_fname), f"Database file {db_fname} does not exist. Please run the database script first."
    with open(db_fname, "r") as f:
        # extract all lines to a list
        lines = f.readlines()
        f_candidate=None
        candinate_epoch = -1
        for line in tqdm(lines):
            ckp = json.loads(line)

            if  os.path.exists(ckp["fname"]) and "configs" in ckp["config"] :

                if match_args(args.__dict__, ckp["config"]["configs"]) and candinate_epoch < ckp["config"]["configs"]["epochs"] <= args.epochs:
                    f_candidate= ckp["fname"]
                    candinate_epoch = ckp["config"]["configs"]["epochs"]
                    print("Found candidate checkpoint", f_candidate, "for epoch", candinate_epoch)
        return f_candidate


def main():
    args = get_argparse()

    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = 7 if args.get_all_data else 3 # if not args.get_all_data, we only use xyz, otherwise we use all data (xyz + vrel + rcs)
    run_name = f"{args.method}_{args.num_points:04d}_sc{args.num_scenes}-b{args.batch_size }-p{args.n_pull}"
    log_dir = f"{args.base_dir}/tb_log/{args.data_group}/{run_name}"
    dir_rev = 0
    while os.path.exists(log_dir):
        dir_rev += 1
        log_dir = f"{args.base_dir}/tb_log/{args.data_group}/{run_name}_r{dir_rev:02d}"
    if dir_rev > 0:
        run_name += f"_r{dir_rev:02d}"
        print("run_name", run_name)
    # assert M <= n_pull-n_val, f"n_pull {n_pull} should be greater than M {M} + n_val {n_val}"
    exp_config = {
        "configs": args.__dict__,
        "run_name": run_name,
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
        # utc time
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), #start time
        "device": str(device),
    }
    # not dispaly "code" in the config
    print("exp_config", exp_config)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", json.dumps(exp_config, indent=4))
    torch.manual_seed(args.seed_value)
    dataloader_train, dataloader_val = get_man_data(args.num_scenes, args.num_points, args.camera_channel, args.radar_channel,
                                                    device, args.img_size, args.batch_size, shuffle=True, n_val=args.n_val, n_pull=args.n_pull, flip_images=args.flip_images,coord_only = not args.get_all_data)

    data_mean, data_std = dataloader_train.dataset.dataset.data_mean, dataloader_train.dataset.dataset.data_std
    print("data_mean", data_mean.tolist())
    print("data_std", data_std.tolist())


    model = get_model(args,D,device)
    """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """

    feature_model = FeatureModel(
        args.img_size, model_name=args.vits_model,
        global_pool='', finetune_vits=args.finetune_vits, vits_checkpoint=args.vits_checkpoint).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.T,
                              prediction_type="epsilon",  # or "sample" or "v_prediction"
                              clip_sample=False,  # important for point clouds
                              )

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(
    list(model.parameters()) + (list(unet_cond_model.parameters()) if unet_cond_model is not None else []),
    lr=args.learning_rate
)
    assert optimizer is not None, "Optimizer is None, please check the learning rate and model parameters"
    train_cd_list = {}
    val_cd_list = {}
    cd_epochs = []
    train_loss_list = []
    val_loss_list = []
    start_epoch = 0


    attention_resolutions="16,8"
    attention_ds = [args.img_size // int(res) for res in attention_resolutions.split(",")]
    channel_mult =(0.5, 1, 1, 2, 2, 4, 4)

    if "unet" in args.cond_type :
        if args.cond_type == "unet":
            cond_in_channels = 3
        elif args.cond_type == "unet_depth":
            cond_in_channels = 1
        elif args.cond_type == "unet_imgdepth":
            cond_in_channels = 4
        else:
            raise ValueError(f"Unknown UNet conditioning type {args.cond_type}")
        unet_cond_model =  EncoderUNetModelNoTime(
            image_size=args.img_size,
            in_channels=cond_in_channels, #ch of image which is 3 (RGB)
            model_channels=128,
            out_channels=128,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            dropout=args.pvcnn_dropout,
            channel_mult=channel_mult,
            use_checkpoint=args.resume_from_checkpoint,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            pool='adaptive'
        ).to(device)
    else:
        unet_cond_model = None
    
    print("MDM model", model)
    print("UNet conditioning model", unet_cond_model)
    exit()

    # Load checkpoint if exists
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        db_fname = f"{args.base_dir}/checkpoints_man/db.txt"
        ckp_fname = get_checkpoint_fname(args, db_fname)
        if ckp_fname is not None:
            fname = os.path.basename(ckp_fname)
            print("Loading checkpoint from", fname)
            ckp = torch.load(ckp_fname, map_location=device)

            start_epoch = ckp["epoch"]
            model.load_state_dict(ckp["model_state_dict"])
            if "unet_cond_model_state_dict" in ckp:
                unet_cond_model.load_state_dict(ckp["unet_cond_model_state_dict"])
            optimizer.load_state_dict(ckp["optimizer_state_dict"])
            train_cd_list = ckp.get("train_cd_list", {})
            val_cd_list = ckp.get("val_cd_list", {})
            cd_epochs = ckp.get("cd_epochs", [])
            train_loss_list = ckp.get("train_loss_list", [])
            val_loss_list = ckp.get("val_loss_list", [])
            print("Loaded checkpoint starting at epoch", start_epoch)
    
        
    tt = trange(start_epoch, args.epochs, desc="Training", unit="epoch", leave=True)

    if args.cond_type == "id":

        all_cond_signal = get_sinusoidal_embedding(torch.tensor(
            range(args.n_pull), device=device), args.pc2_conditioning_dim, device=device)
        # print("all_cond_signal", all_cond_signal)
        all_frame_id = sorted(
            [batch[4] for ds in [dataloader_train.dataset.dataset] for batch in ds])

        id_list = {token: i for i, token in enumerate(all_frame_id)}
        print("id_list", id_list)
    else:
        all_cond_signal = None
        id_list = None
    cd_train_ave = -1
    cd_val_ave = -1

    for epoch in tt:
        # print("all_cond_signal", all_cond_signal)
        train_loss, new_train_loss_list = train_val_one_epoch(args, dataloader_train, model, optimizer,
                                                              scheduler, feature_model, unet_cond_model,"train", epoch, id_list, all_cond_signal,   device, writer, train_loss_list)
        val_loss, new_val_loss_list = train_val_one_epoch(args, dataloader_val, model, optimizer,
                                                          scheduler, feature_model,unet_cond_model, "val", epoch, id_list, all_cond_signal,   device, writer, val_loss_list)
        if epoch % args.vis_freq == 0:
            cd_epochs.append(epoch)
            cd_train_ave, new_train_cd_list = sample_cd_save(args, model, scheduler, dataloader_train, "train", feature_model, unet_cond_model,
                                                             device, epoch, id_list, all_cond_signal, run_name, exp_config, writer, train_cd_list)
            cd_val_ave, new_val_cd_list = sample_cd_save(args, model, scheduler, dataloader_val, "val", feature_model, unet_cond_model,
                                                         device, epoch, id_list, all_cond_signal, run_name, exp_config, writer, val_cd_list,)

        tt.set_description_str(
            f"MSE = {train_loss:.2f}, CD_tr = {cd_train_ave:.2f}, CD_val = {cd_val_ave:.2f}")

    writer.close()
    # Save the model

    print("Saving model...")
    save_checkpoint(model, optimizer, unet_cond_model,
                    train_cd_list=train_cd_list,
                    val_cd_list=val_cd_list,
                    cd_epochs=cd_epochs,
                    val_loss_list=val_loss_list,
                    train_loss_list=train_loss_list, epoch=args.epochs, base_dir=args.base_dir, config=exp_config, run_name=run_name, code=code)


if __name__ == "__main__":
    main()
