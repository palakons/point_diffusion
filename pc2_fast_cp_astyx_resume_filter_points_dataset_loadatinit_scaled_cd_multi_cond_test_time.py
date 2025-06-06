import logging
from pytorch3d.structures import Pointclouds
import hydra
import glob
from datetime import datetime
from pathlib import Path
import inspect
import time
from pytorch3d.vis.plotly_vis import get_camera_wireframe

from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2,
    registry,
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from omegaconf import DictConfig, OmegaConf
from config.structured import CO3DConfig, DataloaderConfig, ProjectConfig

from dataset.exclude_sequence import EXCLUDE_SEQUENCE, LOW_QUALITY_SEQUENCE
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from diffusers import DDPMScheduler
from geomloss import SamplesLoss  # Install via pip install geomloss
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm, trange
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from model.point_cloud_model import PointCloudModel
from model.projection_model_time import PointCloudProjectionModel
# from model.projection_model import PointCloudProjectionModel
import os
from pytorch3d.implicitron.dataset.data_loader_map_provider import (
    SequenceDataLoaderMapProvider,
)
from torch.utils.data import SequentialSampler
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pynvml
import psutil
import platform
import json
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from pytorch3d.renderer.cameras import PerspectiveCameras

CSV_TIME_FILE = "/home/palakons/from_scratch/timer.csv"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
#     filename="/home/palakons/from_scratch/app.log",
#     filemode='a',  # Append mode.
# )
logger = logging.getLogger(__name__)

if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False


class AstyxDataset(Dataset):
    def __init__(
        self,
        M,
        N,
        depth_model,
        random_offset: bool = False,
        root_dir: str = "/data/palakons/dataset_astyx_hires2019/",
        device="cpu",
        is_scaled=False,
    ):
        # root_dir: path to the directory containing the json files
        self.root_dir = root_dir
        self.radar_dir = root_dir + "/radar_6455"
        self.depth_dir = root_dir + "/depth_front"
        self.camera_dir = root_dir + "/camera_front"
        # self.depth_models = ['vitl','vitb','vits']
        self.calibration_dir = root_dir + "/calibration"
        self.object_dir = root_dir + "/groundtruth_obj3d"
        self.M = M
        self.N = N
        self.depth_model = depth_model  # ['vitl','vitb','vits']
        self.random_offset = random_offset
        self.device = device
        self.is_scaled = is_scaled

        self.ids = []
        # list filesin the radar_dir
        for file in os.listdir(self.radar_dir):
            idx = int(file[:6])
            if idx not in self.ids:
                if not os.path.exists(
                    self.depth_dir + f"/{idx:06d}_{self.depth_model}.jpg"
                ):
                    print(
                        "File not found: ",
                        self.depth_dir + f"/{idx:06d}_{self.depth_model}.jpg",
                    )
                    continue

                if not os.path.exists(self.calibration_dir + f"/{idx:06d}.json"):
                    print("File not found: ",
                          self.calibration_dir + f"/{idx:06d}.json")
                    continue
                if not os.path.exists(self.camera_dir + f"/{idx:06d}.jpg"):
                    print("File not found: ",
                          self.camera_dir + f"/{idx:06d}.jpg")
                    continue
                if not os.path.exists(self.object_dir + f"/{idx:06d}.json"):
                    print("File not found: ",
                          self.object_dir + f"/{idx:06d}.json")
                    continue
                self.ids.append(idx)
        # randomly sample for M items
        self.active_ids = random.sample(self.ids, M)

        self.data_bank = []
        for i in range(len(self.active_ids)):
            self.data_bank.append(self.load_data(self.active_ids[i]))

        # stat for position data
        # print("self.data_bank[0][1]", self.data_bank[0][1].shape)
        # self.data_bank[0][1] torch.Size([128, 3])
        all_radar_positions = torch.stack(
            [d[1] for d in self.data_bank], dim=0)
        # print("all_radar_positions", all_radar_positions.shape)
        # # all_radar_positions torch.Size([10, 128, 3])

        # print("mean", all_radar_positions.shape)
        # print("mean", all_radar_positions.mean(axis=(0,1)))
        # print("std", all_radar_positions.shape)
        # print("std", all_radar_positions.std(axis=(0,1)))
        # store mean and std
        self.data_mean = all_radar_positions.mean(axis=(0, 1))
        self.data_std = all_radar_positions.std(axis=(0, 1))

    def load_data(self, idx):

        depth = {}
        image = plt.imread(
            self.depth_dir + f"/{idx:06d}_{self.depth_model}.jpg")
        # if image is of 3 channels, take only the first one
        # print("image.shape: ", image.shape)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if image.max() > 1.0:
            image = image / 255.0
        depth = torch.tensor(image, dtype=torch.float32)
        # print("depth: ", depth.shape)  # depth:  torch.Size([618, 2048])

        square_image_offset = (
            int((depth.shape[1] - depth.shape[0]) / 2)
            if not self.random_offset
            else random.randint(0, depth.shape[1] - depth.shape[0])
        )

        # square_image_offset = depth.shape[1] - depth.shape[0]
        # print("square_image_offset: ", square_image_offset)
        image_size_hw = (depth.shape[0], depth.shape[1])
        depth = depth[:,
                      square_image_offset: square_image_offset + depth.shape[0]]
        # load radar data, .txt
        df = pd.read_csv(
            self.radar_dir + f"/{idx:06d}.txt", sep=" ", skip_blank_lines=True
        )
        df = df.iloc[:, :3]
        # convert to tensor
        radar_data = torch.tensor(df.values)
        npoints_original = radar_data.shape[0]

        # load calibration data, .json
        with open(self.calibration_dir + f"/{idx:06d}.json") as f:
            calibrations = json.load(f)
            for c in calibrations["sensors"]:
                if c["sensor_uid"] == "camera_front":
                    new_camera_base = calib_to_camera_base(
                        c["calib_data"], [image_size_hw[0]] *
                        2, square_image_offset
                    )

        assert (
            new_camera_base is not None
        ), "new_camera_base is None (not found in calibration data)"

        # load camera data, .jpg
        camera_front = plt.imread(self.camera_dir + f"/{idx:06d}.jpg").transpose(
            2, 0, 1
        )
        if camera_front.max() > 1.0:
            camera_front = camera_front / 255.0
        camera_front = torch.tensor(camera_front, dtype=torch.float32)
        # print("camera_front: ", camera_front.shape) #camera_front:  torch.Size([3, 618, 2048])
        camera_front = camera_front[
            :, :, square_image_offset: square_image_offset + camera_front.shape[1]
        ]

        # filter points: keeps only points within the image
        # world_points = radar_data
        world_points = radar_data.clone().detach().float()
        # cam_coord = new_camera_base.get_world_to_view_transform().transform_points(
        #     torch.tensor(world_points, dtype=torch.float32))  # N x3
        image_coord = new_camera_base.transform_points(world_points)
        # torch.tensor(world_points, dtype=torch.float32))  # N x3
        points_uv = image_coord[:, :2]  # N x2

        # print("shapes: ", points_uv.shape)
        # print("min axis 0", points_uv.min(axis=0))
        # print("max axis 0", points_uv.max(axis=0))
        # print("filtering from to", square_image_offset,
        #   camera_front.shape[1]+square_image_offset)
        mask = (
            (points_uv[:, 1] >= 0)
            & (points_uv[:, 1] < camera_front.shape[1])
            & (points_uv[:, 0] >= 0)
            & (points_uv[:, 0] < camera_front.shape[1])
            # & (points_uv[:, 2] > 0)  # Ensure points are in front of the camera
        )
        # print("radar_data: ", radar_data.shape)
        filtered_radar_data = radar_data[mask]
        npoints_filtered = filtered_radar_data.shape[0]
        # print("filtered_radar_data: ", filtered_radar_data.shape)

        # image_coord_in = new_camera_base.transform_points(
        #     torch.tensor(radar_data[mask], dtype=torch.float32))  # N x3
        # image_coord_out = new_camera_base.transform_points(
        #     torch.tensor(radar_data[~mask], dtype=torch.float32))  # N x3

        # print("in", filtered_radar_data.shape, image_coord_in.shape)
        # print("min max in", image_coord_in.min(
        #     axis=0), image_coord_in.max(axis=0))
        # print("out", radar_data[~mask].shape, image_coord_out.shape)

        # print("radar_data: ", radar_data.shape)#radar_data:  torch.Size([2246, 3])
        # sample radar data points to N points
        filtered_radar_data = filtered_radar_data[
            torch.randperm(filtered_radar_data.size(0))[: self.N]
        ]

        while filtered_radar_data.shape[0] < self.N:
            filtered_radar_data = torch.cat(
                [
                    filtered_radar_data,
                    filtered_radar_data[
                        torch.randperm(filtered_radar_data.size(0))[
                            : self.N - filtered_radar_data.shape[0]
                        ]
                    ],
                ]
            )
        # print("radar_data: ", radar_data.shape)#radar_data:  torch.Size([N, 3])

        # load object data, .json
        with open(self.object_dir + f"/{idx:06d}.json") as f:
            objects = json.load(f)

        # print("shapes: ", depth.shape, filtered_radar_data.shape, camera_front.shape,len(objects),len(idx),npoints)
        print(
            f"idx {idx} npoints_original {npoints_original} npoints_filtered {npoints_filtered} square_image_offset {square_image_offset}"
        )
        return (
            depth,
            filtered_radar_data,
            new_camera_base,
            camera_front,
            objects,
            idx,
            npoints_original,
            npoints_filtered,
            square_image_offset,
        )

    def __len__(self):
        return len(self.active_ids)

    def __getitem__(self, idx):
        if not self.is_scaled:

            return self.data_bank[idx]

        st_time = time.time()
        (
            depth,
            filtered_radar_data,
            new_camera_base,
            camera_front,
            objects,
            idx,
            npoints_original,
            npoints_filtered,
            square_image_offset,
        ) = self.data_bank[idx]

        logger.info(
            f"{os.getpid()}, __getitem__/1_data_bank, {time.time()-st_time}")
        st_time = time.time()

        # print("mean", self.data_mean)
        # print("std", self.data_std)
        # print("filtered_radar_data", filtered_radar_data.shape)
        # print("filtered_radar_data", filtered_radar_data[0])

        # newd =(filtered_radar_data-self.data_mean)/self.data_std

        # print("newd", newd.shape)

        # print("filtered_radar_data", newd[0])
        # mean tensor([38.5517, -0.2240,  1.0346], dtype=torch.float64)                                                             std tensor([20.8821,  3.5820,  1.9252], dtype=torch.float64)                                                             filtered_radar_data torch.Size([128, 3])                                                                  filtered_radar_data tensor([73.2768, -1.1732,  3.0596], dtype=torch.float64)                                                             newd torch.Size([128, 3])                                                                                                                        filtered_radar_data tensor([ 1.6629, -0.2650,  1.0518], dtype=torch.float64)

        output = (
            depth,
            (filtered_radar_data - self.data_mean) / self.data_std,
            new_camera_base,
            camera_front,
            objects,
            idx,
            npoints_original,
            npoints_filtered,
            square_image_offset,
        )
        logger.info(
            f"{os.getpid()}, __getitem__/2_norm, {time.time()-st_time}")

        return output


def custom_collate_fn(batch):

    st_time = time.time()
    (
        depths,
        radar_data,
        camera_base_list,
        camera_fronts,
        objects,
        idxs,
        npoints,
        npoints_filtered,
        square_image_offset,
    ) = zip(*batch)

    logger.info(
        f"{os.getpid()}, custom_collate_fn/1_zip, {time.time()-st_time}")
    st_time = time.time()

    npoints_after = torch.as_tensor(npoints)
    npoints_filtered_after = torch.as_tensor(npoints_filtered)
    idxs_after = torch.as_tensor(idxs)
    square_image_offset_after = torch.as_tensor(square_image_offset)

    logger.info(
        f"{os.getpid()},custom_collate_fn/2_as_tensor,{time.time()-st_time}")
    st_time = time.time()

    objects_after = list(objects)
    camera_fronts_after = torch.stack(camera_fronts)
    radar_data_after = torch.stack(radar_data)
    depths_after = torch.stack(depths)

    logger.info(
        f"{os.getpid()},custom_collate_fn/3_stack,{time.time()-st_time}")
    st_time = time.time()

    focal_length = []
    principal_point = []
    R = []
    T = []
    image_sizes_hw = []
    for camera_base in camera_base_list:
        focal_length.append(camera_base.focal_length)

        principal_point.append(camera_base.principal_point)
        R.append(camera_base.R)
        T.append(camera_base.T)
        image_sizes_hw.append(camera_base.image_size)

    focal_lengths = torch.concat(focal_length)
    principal_points = torch.concat(principal_point)
    Rs = torch.concat(R)
    Ts = torch.concat(T)
    image_sizes_hw = torch.concat(image_sizes_hw)

    logger.info(
        f"{os.getpid()},custom_collate_fn/4_prepare,{time.time()-st_time}")
    st_time = time.time()

    # print("image_sizes_hw: ", image_sizes_hw)

    camera_bases = PerspectiveCameras(
        focal_length=focal_lengths,
        principal_point=principal_points,
        R=Rs,
        T=Ts,
        in_ndc=False,
        image_size=image_sizes_hw,
    )

    logger.info(
        f"{os.getpid()},custom_collate_fn/5_PerspectiveCameras,{time.time()-st_time}")

    return (
        depths_after,
        radar_data_after,
        camera_bases,
        camera_fronts_after,
        objects_after,
        idxs_after,
        npoints_after,
        npoints_filtered_after,
        square_image_offset_after,
    )


def custom_collate_fn_org(batch):
    (
        depths,
        radar_data,
        camera_base_list,
        camera_fronts,
        objects,
        idxs,
        npoints,
        npoints_filtered,
        square_image_offset,
    ) = zip(*batch)

    npoints_after = torch.tensor(npoints)
    npoints_filtered_after = torch.tensor(npoints_filtered)
    idxs_after = torch.tensor(idxs)
    square_image_offset_after = torch.tensor(square_image_offset)
    objects_after = list(objects)
    camera_fronts_after = torch.stack(camera_fronts)
    radar_data_after = torch.stack(radar_data)
    depths_after = torch.stack(depths)

    focal_length = []
    principal_point = []
    R = []
    T = []
    image_sizes_hw = []
    for camera_base in camera_base_list:
        focal_length.append(camera_base.focal_length)

        principal_point.append(camera_base.principal_point)
        R.append(camera_base.R)
        T.append(camera_base.T)
        image_sizes_hw.append(camera_base.image_size)

    focal_lengths = torch.concat(focal_length)
    principal_points = torch.concat(principal_point)
    Rs = torch.concat(R)
    Ts = torch.concat(T)
    image_sizes_hw = torch.concat(image_sizes_hw)

    # print("image_sizes_hw: ", image_sizes_hw)

    camera_bases = PerspectiveCameras(
        focal_length=focal_lengths,
        principal_point=principal_points,
        R=Rs,
        T=Ts,
        in_ndc=False,
        image_size=image_sizes_hw,
    )

    return (
        depths_after,
        radar_data_after,
        camera_bases,
        camera_fronts_after,
        objects_after,
        idxs_after,
        npoints_after,
        npoints_filtered_after,
        square_image_offset_after,
    )


def calib_to_camera_base(calibration_data, image_size_hw: tuple, offset: int = 0):

    # Convert intrinsic matrix K to tensor
    K = torch.tensor(calibration_data["K"])

    # Extract focal length and principal point from K
    focal_length = torch.tensor([[K[0, 0], K[1, 1]]])
    principal_point = torch.tensor([[K[0, 2] - offset, K[1, 2]]])

    # Convert extrinsic matrix T_to_ref_COS to tensor
    T_to_ref_COS = torch.tensor(calibration_data["T_to_ref_COS"])

    # Extract rotation (R) and translation (T) components
    R = T_to_ref_COS[:3, :3].unsqueeze(0)  # 3x3 rotation matrix
    RT_inv = torch.inverse(T_to_ref_COS)
    T = RT_inv[:3, 3].unsqueeze(0)  # 3x1 translation vector
    # T = T_to_ref_COS[:3, 3].unsqueeze(0)   # 3x1 translation vector

    # Create a PerspectiveCameras object
    return PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        in_ndc=False,
        image_size=[image_size_hw],
    )


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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


class PointCloudLoss(nn.Module):
    def __init__(self, npoints: int, emd_weight: float = 0.5):
        super().__init__()
        if not 0 <= emd_weight <= 1:
            raise ValueError("emd_weight must be between 0 and 1")
        self.npoints = npoints
        self.emd_weight = emd_weight
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y_true.size() == y_pred.size()
        assert y_true.size(2) == 3
        assert len(y_true.size()) == 3
        # y_true = y_true.view(-1, self.npoints, 3)
        # y_pred = y_pred.view(-1, self.npoints, 3)
        chamfer = chamfer_distance(y_true, y_pred)[0]
        if self.emd_weight == 0:
            return chamfer
        emd = self.sinkhorn_loss(y_true, y_pred)
        return (1 - self.emd_weight) * chamfer + self.emd_weight * emd


def extract_batch(cfg: ProjectConfig, batch, device):

    if cfg.dataset.type == "astyx":
        (
            depths,
            radar_data,
            camera,
            image_rgb,
            objects,
            idx,
            npoints,
            npoints_filtered,
            offsets,
        ) = batch
        depths = depths.to(device)
        camera = camera.to(device)
        mask = None
        image_rgb = image_rgb.to(device)
        points = radar_data.float().to(device)
        pc = Pointclouds(points=points)
    elif cfg.dataset.type == "co3dv2":
        batch = batch.to(device)
        pc = batch.sequence_point_cloud
        camera = batch.camera
        image_rgb = batch.image_rgb
        mask = batch.fg_probability
        depths = None
        idx = None

    return pc, camera, image_rgb, mask, depths, idx


# Training function
def train_one_epoch(
    dataloader,
    model,
    optimizer,
    scheduler,
    cfg: ProjectConfig,
    criterion,
    device,
    pcpm: PointCloudProjectionModel,
):
    assert pcpm is not None, "pcpm must be provided"
    model.train()

    data_mean = dataloader.dataset.data_mean.to(device).float()
    data_std = dataloader.dataset.data_std.to(device).float()

    batch_losses = []
    st_time = time.time()
    for batch in dataloader:

        logger.info(f"{os.getpid()},dataloader,{time.time()-st_time}")
        st_time = time.time()

        pc, camera, image_rgb, mask, depths, idx = extract_batch(
            cfg, batch, device)

        logger.info(f"{os.getpid()},extract_batch,{time.time()-st_time}")
        st_time = time.time()

        x_0 = pcpm.point_cloud_to_tensor(pc, normalize=True, scale=True)

        logger.info(
            f"{os.getpid()},point_cloud_to_tensor,{time.time()-st_time}")
        st_time = time.time()

        B, N, D = x_0.shape
        noise = torch.randn_like(x_0)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (B,),
            device=device,
            dtype=torch.long,
        )
        x_t = scheduler.add_noise(x_0, noise, timesteps)  # noisy_x
        # print("type x_t", type(x_t))
        # print("dtype x_t", x_t.dtype)

        x_t_input = apply_conditioning_to_xt(
            cfg,
            (x_t * data_std + data_mean) if cfg.dataset.is_scaled else x_t,
            camera,
            image_rgb,
            depths,
            mask,
            timesteps,
            pcpm,
        )

        logger.info(
            f"{os.getpid()},apply_conditioning_to_xt,{time.time()-st_time}")
        st_time = time.time()

        if cfg.dataset.is_scaled:
            # scale back the first 3 dimensions
            x_t_input[:, :, :3] = x_t[:, :, :3]

        optimizer.zero_grad()
        noise_pred = model(x_t_input, timesteps)

        logger.info(f"{os.getpid()},forward,{time.time()-st_time}")
        st_time = time.time()

        if not noise_pred.shape == noise.shape:
            raise ValueError(f"{noise_pred.shape} and {noise.shape} not equal")

        loss = criterion(noise_pred, noise)
        loss.backward()

        logger.info(f"{os.getpid()},backward,{time.time()-st_time}")
        st_time = time.time()

        optimizer.step()

        batch_losses.append(loss.item())  # float

        logger.info(f"{os.getpid()},step,{time.time()-st_time}")
        st_time = time.time()
    return batch_losses


def get_camera_wires_trans(cameras, scale: float = 1):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    # print("cameras.device", cameras.device)
    # print(torch.device("cuda"))
    # cameras.device cuda:0
    # cuda
    assert str(cameras.device).startswith("cuda"), "cameras should be on cuda"
    cam_wires_canonical = get_camera_wireframe(scale=scale).cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    return cam_wires_trans


def plot_image_depth(
    gt,
    cfg: ProjectConfig,
    fname,
    point_size=0.1,
    cam_wires_trans=None,
    image_rgb=None,
    depth_image=None,
    plt_title="",
):
    if cam_wires_trans is not None:
        assert cam_wires_trans.device == torch.device(
            "cpu"
        ), "cam_wires_trans should be on cpu"
    assert gt.device == torch.device("cpu"), "gt should be on cpu"

    plt_title = f"{cfg.run.name}"

    # plot_multi_gt(gts,  args, input_pc_file_list, fname=None)
    if fname is None:
        dir_name = f"plots/" + cfg.run.name
        # mkdir
        if not os.path.exists(dir_name):
            print("creating dir", dir_name)
            os.makedirs(dir_name)
        fname = f"{dir_name}/images-depths.png"
    # print("plot_sample_condition fname", fname)

    gt = gt.numpy()

    # print("gt", gt.shape)

    # make sure same number of items, gt, camera, image_rgb, mask
    assert (
        len(gt) == len(cam_wires_trans) == len(image_rgb)
    ), f"gt, camera, image_rgb should have same number of items: {len(gt)}, {len(cam_wires_trans)}, {len(image_rgb)}"
    fig = plt.figure(figsize=(30, 10 * len(gt)))

    for i in range(len(gt)):

        ax = fig.add_subplot(len(gt), 3, 3 * i + 1)
        ax.imshow(image_rgb[i].cpu().numpy().transpose(1, 2, 0))
        ax.axis("off")
        ax.set_title("image_rgb")
        if depth_image is not None:
            ax = fig.add_subplot(len(gt), 3, 3 * i + 2)
            # show gray scale image, take only the first layer
            #  depth_image torch.Size([618, 2048])
            # print("depth_image", depth_image[i].shape)
            plot = ax.imshow(depth_image[i].cpu().numpy())

            # make colorbar
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)

            # ax.imshow(depth_image[i].cpu().numpy().transpose(1, 2, 0))
            ax.axis("off")
            ax.set_title("depth image")
            # show
        # ax[x] is 3d plots

        # experiment plot projected points

        ax = fig.add_subplot(len(gt), 3, 3 * i + 3, projection="3d")
        plot = ax.scatter(
            gt[i][:, 0], gt[i][:, 1], gt[i][:, 2], marker=",", s=point_size
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if cam_wires_trans is not None:
            x_, z_, y_ = cam_wires_trans[i].numpy().T.astype(float)
            # print("coord", x_, y_, z_)
            (h,) = ax.plot(x_, y_, z_)
        ax.set_aspect("equal")

    plt.title(plt_title)
    # plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    exit()
    # print("saved image-mask", fname)


def plot_image_depth_projected(
    gt,
    cfg: ProjectConfig,
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
    if camera is not None:
        cam_wires_trans = get_camera_wires_trans(camera).detach().cpu()
        assert cam_wires_trans.device == torch.device(
            "cpu"
        ), "cam_wires_trans should be on cpu"
    assert gt.device == torch.device("cpu"), "gt should be on cpu"

    # plot_multi_gt(gts,  args, input_pc_file_list, fname=None)
    if fname is None:
        dir_name = f"plots/" + cfg.run.name
        # mkdir
        if not os.path.exists(dir_name):
            print("creating dir", dir_name)
            os.makedirs(dir_name)
        fname = f"{dir_name}/images-depths_projected.png"

    # print("plot_sample_condition fname", fname)

    gt = gt.numpy()

    if cfg.dataset.is_scaled:
        gt = gt * data_std + data_mean

    # print("gt", gt.shape)
    # make sure same number of items, gt, camera, image_rgb, mask
    assert (
        len(gt) == len(cam_wires_trans) == len(image_rgb)
    ), f"gt, camera, image_rgb should have same number of items: {len(gt)}, {len(cam_wires_trans)}, {len(image_rgb)}"
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
        plot = ax.imshow(image_rgb[i].cpu().numpy().transpose(1, 2, 0))
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

        if depth_image is not None:
            ax = fig.add_subplot(len(gt), 3, 3 * i + 2)
            # show gray scale image, take only the first layer
            #  depth_image torch.Size([618, 2048])
            # print("depth_image", depth_image[i].shape)
            plot = ax.imshow(depth_image[i].cpu().numpy())

            # make colorbar
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)

            # ax.imshow(depth_image[i].cpu().numpy().transpose(1, 2, 0))
            ax.axis("off")
            ax.set_title("depth image")
            # show
            # ax[x] is 3d plots

            ax.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                marker="x",
                s=point_size,
                cmap="jet",
                c=cam_coord[:, 2].cpu().numpy(),
            )
        # print("projected_points", projected_points.shape)
        # print("min max", projected_points.min(), projected_points.max())

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

        if cam_wires_trans is not None:
            x_, z_, y_ = cam_wires_trans[i].numpy().T.astype(float)
            # print("coord", x_, y_, z_)
            (h,) = ax.plot(x_, y_, z_)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    # print("saved image-mask", fname)


def plot_quadrants(
    point_list,
    color_list,
    name_list,
    fname,
    point_size=0.5,
    title="",
    cam_wires_trans=None,
    ax_lims=None,
    color_map_name="gist_rainbow",
):
    # assert camera is on cpu, not cuda
    assert cam_wires_trans.device == torch.device(
        "cpu"), "camera should be on cpu"
    fig_size_baseline = 10

    fig = plt.figure(figsize=(fig_size_baseline * 4 / 3, fig_size_baseline))
    ax = fig.add_subplot(221, projection="3d")
    for points, color, name in zip(point_list, color_list, name_list):
        assert (
            len(points.shape) == 2
        ), f"{name} points should have shape (N, 3), not {points.shape}"
        assert (
            points.shape[1] == 3
        ), f"{name} points should have shape (N, 3), not {points.shape}"
        params = {
            "zs": points[:, 2],
            "c": color,
            "marker": ",",
            "label": name,
            "s": point_size,
        }
        # if color is a string
        if not isinstance(color, str):  # color is a list
            params["cmap"] = color_map_name
        plot = ax.scatter(points[:, 0], points[:, 1], **params)

        # plot = ax.scatter(
        #     points[:, 0],
        #     points[:, 1],
        #     points[:, 2],
        #     c=color,
        #     marker=",",
        #     label=name,
        #     s=point_size,
        #     cmap=color_map_name,
        # )
        # if color is not string
        if not isinstance(color, str):
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if cam_wires_trans is not None:
        for wire in cam_wires_trans:
            # the Z and Y axes are flipped intentionally here!
            x_, z_, y_ = wire.numpy().T.astype(float)
            (h,) = ax.plot(x_, y_, z_, color="k", linewidth=0.3)
    ax.set_title(title)
    if ax_lims is not None:
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
        ax.set_zlim(ax_lims[2])
    ax.set_aspect("equal")
    x_equal_lim = ax.get_xlim()
    y_equal_lim = ax.get_ylim()
    z_equal_lim = ax.get_zlim()

    # next plot x-z, 2D
    ax = fig.add_subplot(222)
    for points, color, name in zip(point_list, color_list, name_list):

        params = {
            "x": points[:, 0],
            "y": points[:, 2],
            "c": color,
            "marker": ",",
            "label": name,
            "s": point_size,
        }
        # if color is a string
        if not isinstance(color, str):  # color is a list
            params["cmap"] = color_map_name
        plot = ax.scatter(**params)

        if not isinstance(color, str):
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)
    if cam_wires_trans is not None:
        for wire in cam_wires_trans:
            x_, z_, y_ = wire.numpy().T.astype(float)
            (h,) = ax.plot(x_, z_, color="k", linewidth=0.3)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    if ax_lims is not None:
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[2])
    ax.set_aspect("equal")

    # next plot y-z, 2D
    ax = fig.add_subplot(223)
    for points, color, name in zip(point_list, color_list, name_list):

        params = {
            "x": points[:, 1],
            "y": points[:, 2],
            "c": color,
            "marker": ",",
            "label": name,
            "s": point_size,
        }
        # if color is a string
        if not isinstance(color, str):  # color is a list
            params["cmap"] = color_map_name
        plot = ax.scatter(**params)

        if not isinstance(color, str):
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)
    if cam_wires_trans is not None:
        for wire in cam_wires_trans:
            x_, z_, y_ = wire.numpy().T.astype(float)
            (h,) = ax.plot(y_, z_, color="k", linewidth=0.3)
    ax.legend()
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    if ax_lims is not None:
        ax.set_xlim(ax_lims[1])
        ax.set_ylim(ax_lims[2])
    ax.set_aspect("equal")

    # next plot x-y, 2D
    ax = fig.add_subplot(224)
    for points, color, name in zip(point_list, color_list, name_list):

        params = {
            "x": points[:, 0],
            "y": points[:, 1],
            "c": color,
            "marker": ",",
            "label": name,
            "s": point_size,
        }
        # if color is a string
        if not isinstance(color, str):  # color is a list
            params["cmap"] = color_map_name
        plot = ax.scatter(**params)

        if not isinstance(color, str):
            cax = inset_axes(
                ax, width="5%", height="50%", loc="lower left", borderpad=1
            )
            fig.colorbar(plot, cax=cax)
    if cam_wires_trans is not None:
        for wire in cam_wires_trans:
            x_, z_, y_ = wire.numpy().T.astype(float)
            (h,) = ax.plot(x_, y_, color="k", linewidth=0.3)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if ax_lims is not None:
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
    ax.set_aspect("equal")

    # end all plots

    # plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return x_equal_lim, y_equal_lim, z_equal_lim


def plot_sample_condition(
    gt,
    xts,
    x0s,
    steps,
    cfg: ProjectConfig,
    epoch,
    fname,
    point_size=0.1,
    cam_wires_trans=None,
    image_rgb=None,
    depths=None,
    cd=None,
    data_mean=torch.tensor([0, 0, 0]),
    data_std=torch.tensor([1, 1, 1]),
    plt_title="",
):
    data_mean = data_mean.float().cpu().numpy()
    data_std = data_std.float().cpu().numpy()
    assert gt.shape[0] == 1, "gt should have shape (1, N, 3)"
    if cam_wires_trans is not None:
        assert cam_wires_trans.device == torch.device(
            "cpu"
        ), "cam_wires_trans should be on cpu"
    assert gt.device == torch.device("cpu"), "gt should be on cpu"
    assert xts.device == torch.device("cpu"), "xts should be on cpu"

    plt_title = f"{plt_title} {epoch}: {cfg.run.name}: "

    # add these parameters to the title, use a verybrief names
    plt_title += f"\nsd{cfg.run.seed} infer_step{cfg.run.num_inference_steps} im_sz{cfg.dataset.image_size} schedule{cfg.model.beta_schedule} dim{cfg.model.point_cloud_model_embed_dim}\nn{cfg.dataset.max_points} lr{cfg.optimizer.lr} loss{cfg.loss.loss_type} opt{cfg.optimizer.name}"
    if cd is not None:
        plt_title += f" -- CD{cd:.2f}"

    # plot_multi_gt(gts,  args, input_pc_file_list, fname=None)
    if fname is None:
        dir_name = f"plots/" + cfg.run.name
        # mkdir
        if not os.path.exists(dir_name):
            print("creating dir", dir_name)
            os.makedirs(dir_name)
        fname = f"{dir_name}/sample_ep_{epoch:05d}.png"
    print("plot_sample_condition fname", fname)

    gt = gt.numpy()[0]
    xt = xts[-1].numpy()

    if cfg.dataset.is_scaled:
        gt = gt * data_std + data_mean
        xt = xt * data_std + data_mean

    x_equal_lim, y_equal_lim, z_equal_lim = plot_quadrants(
        [gt, xt],
        ["g", "r"],
        ["gt", "xt"],
        fname.replace(".png", "_gt-xt.png"),
        point_size=1,
        title=plt_title,
        cam_wires_trans=cam_wires_trans,
    )

    plot_quadrants(
        [gt],
        ["g"],
        ["gt"],
        fname.replace(".png", "_gt.png"),
        point_size=1,
        title=plt_title,
        cam_wires_trans=cam_wires_trans,
        ax_lims=[x_equal_lim, y_equal_lim, z_equal_lim],
    )
    plot_quadrants(
        [xt],
        ["r"],
        ["xt"],
        fname.replace(".png", "_xt.png"),
        point_size=1,
        title=plt_title,
        cam_wires_trans=cam_wires_trans,
        ax_lims=[x_equal_lim, y_equal_lim, z_equal_lim],
    )

    color_map_name = "gist_rainbow"
    diff = gt[None, :, :] - xt[:, None, :]  # (100, 100, 3)
    dist = np.linalg.norm(diff, axis=-1)
    min_dist = np.min(dist, axis=0)
    plot_quadrants(
        [gt],
        [min_dist],
        ["error"],
        fname.replace(".png", "_gt_color_dist.png"),
        point_size=1,
        title=plt_title,
        cam_wires_trans=cam_wires_trans,
        ax_lims=[x_equal_lim, y_equal_lim, z_equal_lim],
        color_map_name=color_map_name,
    )


def to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def save_checkpoint(
    # (epoch, cd)
    model,
    optimizer,
    cfg,
    cd_list,
    loss_list,
    checkpoint_fname,
    db_fname,
):

    print("saving loss", loss_list)
    print("saving cd", cd_list)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": cfg,
        "cds": cd_list,
        "losses": loss_list,
    }
    torch.save(checkpoint, checkpoint_fname)

    with open(db_fname, "a" if os.path.exists(db_fname) else "w") as f:
        data = {"fname": checkpoint_fname, "args": to_dict(checkpoint["args"])}
        # dict to json text
        json.dump(data, f)
        f.write("\n")

    print("checkpoint saved at", checkpoint_fname)
    # print("saved at", db_fname)


def aggregate_ema(value_list, ema_factor):

    if len(value_list) == 1:
        return value_list
    ema = value_list[0]
    ema_list = [ema]
    for i, value in enumerate(value_list[1:]):
        ema = (1 - ema_factor) * ema + ema_factor * value
        ema_list.append(ema)
    return ema_list


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    cfg: ProjectConfig,
    device="cpu",
    start_epoch=0,
    criterion=nn.MSELoss(),
    writer=None,
    pcpm: PointCloudProjectionModel = None,
    CHECKPOINT_DIR="checkpoint_pc2",
    CHECKPOINT_DB_FILE="checkpoint_db.json",
    loss_ema_factors=[0.9, 0.95, 0.975, 0.99],
    cd_ema_factors=[0.69, 0.79, 0.89, 0.99],
    # {"epoch":[],  data:{scene_id: [cd]}}
    prev_cds={"epochs": [], "data": {}},
    prev_losses={"ave": []},  # {ave: [loss]}
):
    assert pcpm is not None, "pcpm must be provided"

    assert (
        len(prev_losses["ave"]) == start_epoch
    ), f"prev_cds and start_epoch should have same length"
    model.train()
    tqdm_range = trange(start_epoch, cfg.run.max_steps, desc="Epoch")
    # add run name and host name to checkpoint
    proc_id = os.getpid()
    checkpoint_fname = f"{CHECKPOINT_DIR}/cp_dm_{datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S')}-{cfg.run.name.replace('/', '_') }_{os.uname().nodename}_{proc_id}.pth"
    # mkdir if not exist
    if not os.path.exists(os.path.dirname(checkpoint_fname)):
        os.makedirs(os.path.dirname(checkpoint_fname))
    print("checkpoint to be saved at", checkpoint_fname)

    loss_emas = process_ema_prev_values(
        writer, list(loss_ema_factors) + [0], None, prev_losses, "Loss"
    )  # {"ave": [loss]}
    cd_emas = process_ema_prev_values(
        writer, list(cd_ema_factors) +
        [0], prev_cds["epochs"], prev_cds["data"], "CD"
    )  # {"epoch":[],  data:{scene_id: [cd]}}

    if start_epoch == cfg.run.max_steps:  # no training, checkpoint available
        print(
            "checkpoint already available at",
            checkpoint_fname,
            "goes directly to inference",
        )
        epoch = start_epoch
        batch = next(iter(dataloader))

        pc, camera, image_rgb, mask, depths, idx = extract_batch(
            cfg, batch, device)

        for i in range(len(pc)):

            sampled_point, xts, x0s, steps = sample(
                model,
                scheduler,
                cfg,
                camera=camera[i],
                depths=depths[i: i + 1] if depths is not None else None,
                image_rgb=image_rgb[i: i + 1],
                mask=mask[i: i + 1] if mask is not None else None,
                num_inference_steps=None,
                device=device,
                pcpm=pcpm,
                data_mean=dataloader.dataset.data_mean,
                data_std=dataloader.dataset.data_std,
            )

            pc_condition = pcpm.point_cloud_to_tensor(
                pc[i: i + 1], normalize=True, scale=True
            )

            cd_loss, _ = calculate_chamfer_distance(
                cfg.dataset.is_scaled,
                dataloader.dataset.data_mean,
                dataloader.dataset.data_std,
                sampled_point,
                pc_condition,
                device,
            )

            plot_sample_condition(
                pc_condition.cpu(),
                xts.cpu(),
                x0s.cpu(),
                steps,
                cfg,
                epoch,
                # None,
                f"plots/{cfg.run.name}/sample_ep_{epoch:05d}_gt{i}_idx{idx[i]}.png",
                0.1,
                cam_wires_trans=get_camera_wires_trans(
                    camera[i]).detach().cpu(),
                image_rgb=image_rgb[i: i + 1].detach().cpu(),
                # mask=mask[:1].detach().cpu() if mask is not None else None,
                depths=depths,
                cd=cd_loss.item(),
                data_mean=dataloader.dataset.data_mean,
                data_std=dataloader.dataset.data_std,
                plt_title=f"gt{i}_idx{idx[i]}",
            )

            plot_image_depth(
                pc_condition.cpu(),
                cfg,
                f"plots/{cfg.run.name}/images-depths_gt{i}_idx{idx[i]}.png",
                0.1,
                cam_wires_trans=get_camera_wires_trans(
                    camera[i]).detach().cpu(),
                image_rgb=image_rgb[i: i + 1].detach().cpu(),
                # depth_image=mask[:1].detach().cpu() if mask is not None else None,
                depth_image=(
                    depths[i: i + 1].detach().cpu() if depths is not None else None
                ),
                plt_title=f"gt{i}_idx{idx[i]}",
            )

        return loss_emas, cd_emas

    else:
        print("start from epoch", start_epoch, "to", cfg.run.max_steps)

        already_image_mask = False
        # {"epoch":[],  data:{scene_id: [cd]}}
        new_cds = {"epochs": [], "data": {}}
        new_losses = []
        for epoch in tqdm_range:
            batch_losses = train_one_epoch(
                dataloader, model, optimizer, scheduler, cfg, criterion, device, pcpm
            )
            losses = sum(batch_losses) / len(batch_losses)
            new_losses.append(losses)
            tqdm_range.set_description(f"loss: {losses:.4f}")

            for alpha in list(loss_ema_factors) + [0]:
                loss_emas["ave"][alpha] = (
                    losses
                    if loss_emas["ave"][alpha] is None
                    else ((1 - alpha) * losses + alpha * loss_emas["ave"][alpha])
                )
                writer.add_scalars(
                    "Loss/average", {f"ema_{alpha:.2e}": losses}, epoch)

            if (epoch + 1) % cfg.run.checkpoint_freq == 0:
                st_time = time.time()
                temp_epochs = cfg.run.max_steps
                cfg.run.max_steps = epoch + 1

                # combine prev_cds and new_cds

                combined_cds = {
                    "epochs": list(prev_cds["epochs"]) + list(new_cds["epochs"]),
                    "data": {
                        scene_id: (
                            list(prev_cds["data"][scene_id])
                            if len(prev_cds["epochs"]) > 0
                            else []
                        )
                        + list(new_cds["data"][scene_id])
                        for scene_id in new_cds["data"]
                    },
                }  # {"epoch":[],  data:{scene_id: [cd]}}

                save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    combined_cds,
                    {
                        "ave": list(prev_losses["ave"]) + list(new_losses)
                    },  # {ave: [loss]}
                    checkpoint_fname.replace(".pth", f"_{epoch}.pth"),
                    f"{CHECKPOINT_DIR}/{CHECKPOINT_DB_FILE}",
                )

                cfg.run.max_steps = temp_epochs

                logger.info(
                    f"{os.getpid()},save checkpoint,{time.time()-st_time}")
            if (epoch + 1) % cfg.run.vis_freq == 0:

                cd_list = []
                dir_name = f"plots/{cfg.run.name}"

                if not os.path.exists(dir_name):
                    print("creating dir", dir_name)
                    os.makedirs(dir_name)

                st_time = time.time()
                for batch in dataloader:

                    logger.info(
                        f"{os.getpid()},sample-viz/1_dataloader,{time.time()-st_time}")
                    st_time = time.time()

                    pc, camera, image_rgb, mask, depths, idx = extract_batch(
                        cfg, batch, device
                    )
                    logger.info(
                        f"{os.getpid()},sample-viz/2_extract_batch,{time.time()-st_time}")
                    st_time = time.time()

                    for i in range(len(pc)):

                        sampled_point, xts, x0s, steps = sample(
                            model,
                            scheduler,
                            cfg,
                            camera=camera[i],
                            image_rgb=image_rgb[i: i + 1],
                            depths=depths[i: i +
                                          1] if depths is not None else None,
                            mask=mask[i: i + 1] if mask is not None else None,
                            num_inference_steps=None,
                            device=device,
                            pcpm=pcpm,
                            data_mean=dataloader.dataset.data_mean,
                            data_std=dataloader.dataset.data_std,
                        )
                        logger.info(
                            f"{os.getpid()},sample-viz/3_sample,{time.time()-st_time}")
                        st_time = time.time()

                        pc_condition = pcpm.point_cloud_to_tensor(
                            pc[i: i + 1], normalize=True, scale=True
                        )
                        logger.info(
                            f"{os.getpid()},sample-viz/4_point_cloud_to_tensor,{time.time()-st_time}")
                        st_time = time.time()

                        cd_loss, _ = calculate_chamfer_distance(
                            cfg.dataset.is_scaled,
                            dataloader.dataset.data_mean,
                            dataloader.dataset.data_std,
                            sampled_point,
                            pc_condition,
                            device,
                        )
                        logger.info(
                            f"{os.getpid()},sample-viz/5_calculate_chamfer_distance,{time.time()-st_time}")
                        st_time = time.time()

                        cd_loss_item = cd_loss.item()
                        cd_list.append(cd_loss_item)
                        # new_cds.append((epoch, cd_loss_item))  # {"epochs": [], "data": {}}
                        idx_i = idx[i].item()

                        new_cds["epochs"].append(epoch)
                        if idx_i not in new_cds["data"]:
                            new_cds["data"][idx_i] = [cd_loss_item]
                        else:
                            new_cds["data"][idx_i].append(cd_loss_item)

                        if idx_i not in cd_emas:
                            print("idx[i]", idx_i)
                            cd_emas[idx_i] = {
                                k: None for k in list(cd_ema_factors) + [0]}

                        for alpha in list(cd_ema_factors) + [0]:
                            cd_emas[idx_i][alpha] = (
                                cd_loss_item
                                if cd_emas[idx_i][alpha] is None
                                else (
                                    (1 - alpha) * cd_loss_item
                                    + alpha * cd_emas[idx_i][alpha]
                                )
                            )
                            writer.add_scalars(
                                f"CD/scene_{idx_i:03d}",
                                {f"ema_{alpha:.2e}": cd_loss_item},
                                epoch,
                            )
                        logger.info(
                            f"{os.getpid()},sample-viz/6_tensorboard,{time.time()-st_time}")
                        st_time = time.time()
                        plot_sample_condition(
                            pc_condition.cpu(),
                            xts.cpu(),
                            x0s.cpu(),
                            steps,
                            cfg,
                            epoch,
                            fname=f"plots/{cfg.run.name}/sample_ep_{epoch:05d}_gt{i}_idx{idx_i}.png",
                            # None,
                            point_size=0.1,
                            cam_wires_trans=get_camera_wires_trans(camera[i])
                            .detach()
                            .cpu(),
                            image_rgb=image_rgb[i:i+1].detach().cpu(),
                            # mask=mask[:1].detach().cpu() if mask is not None else None,
                            depths=depths,
                            cd=cd_loss.item(),
                            data_mean=dataloader.dataset.data_mean,
                            data_std=dataloader.dataset.data_std,
                        )
                        logger.info(
                            f"{os.getpid()},sample-viz/7_plot_sample_condition,{time.time()-st_time}")
                        st_time = time.time()
                        if not already_image_mask:
                            plot_image_depth_projected(
                                pc_condition.cpu(),
                                cfg,
                                f"plots/{cfg.run.name}/images-depths_gt{i}_idx{idx_i}.png",
                                0.1,
                                camera=camera[i],
                                image_rgb=image_rgb[i:i+1].detach().cpu(),
                                # depth_image=mask[:1].detach().cpu( ) if mask is not None else None,
                                depth_image=(
                                    depths[i:i+1].detach().cpu()
                                    if depths is not None
                                    else None
                                ),
                                data_mean=dataloader.dataset.data_mean,
                                data_std=dataloader.dataset.data_std,
                            )
                            if False:
                                plot_image_depth(
                                    pc_condition.cpu(),
                                    cfg,
                                    None,
                                    0.1,
                                    cam_wires_trans=get_camera_wires_trans(
                                        camera[i])
                                    .detach()
                                    .cpu(),
                                    image_rgb=image_rgb[i:i+1].detach().cpu(),
                                    # depth_image=mask[:1].detach().cpu( ) if mask is not None else None,
                                    depth_image=(
                                        depths[i:i+1].detach().cpu()
                                        if depths is not None
                                        else None
                                    ),
                                )

                        logger.info(
                            f"{os.getpid()},sample-viz/8_plot_image_depth_projected,{time.time()-st_time}")
                        st_time = time.time()

                cd_ave = sum(cd_list)/len(cd_list)
                if "ave" not in cd_emas:
                    cd_emas["ave"] = {
                        k: None for k in list(cd_ema_factors) + [0]}
                for alpha in list(cd_ema_factors) + [0]:
                    cd_emas["ave"][alpha] = (
                        cd_ave
                        if cd_emas["ave"][alpha] is None
                        else (
                            (1 - alpha) * cd_ave
                            + alpha * cd_emas["ave"][alpha]
                        )
                    )
                    writer.add_scalars(
                        f"CD/average",
                        {f"ema_{alpha:.2e}": cd_ave},
                        epoch,
                    )

                logger.info(
                    f"{os.getpid()},sample-viz/8_tensorboard,{time.time()-st_time}")
                st_time = time.time()

                log_utils(log_type="dynamic", model=model,
                          writer=writer, epoch=epoch)

                logger.info(
                    f"{os.getpid()},sample-viz/9_log_utils,{time.time()-st_time}")
                st_time = time.time()

        # for handler in logger.handlers:
        #     handler.flush()

        combined_cds = {
            "epochs": list(prev_cds["epochs"]) + list(new_cds["epochs"]),
            "data": {
                scene_id: (
                    list(prev_cds["data"][scene_id])
                    if len(prev_cds["epochs"]) > 0
                    else []
                )
                + list(new_cds["data"][scene_id])
                for scene_id in new_cds["data"]
            },
        }  # {"epoch":[],  data:{scene_id: [cd]}}
        if True:
            save_checkpoint(
                model,
                optimizer,
                cfg,
                combined_cds,
                {"ave": list(prev_losses["ave"]) +
                 list(new_losses)},  # {ave: [loss]}
                checkpoint_fname.replace(".pth", f"_final.pth"),
                f"{CHECKPOINT_DIR}/{CHECKPOINT_DB_FILE}",
            )
        return loss_emas, cd_emas


def apply_conditioning_to_xt(
    cfg: ProjectConfig,
    x_t,
    camera,
    image_rgb,
    depths,
    mask,
    t,
    pcpm: PointCloudProjectionModel,
):

    if cfg.model.condition_source == "unconditional_filter":
        return x_t
    elif cfg.model.condition_source in ["image_rgb_filter", "depth_filter"]:
        if cfg.model.condition_source == "image_rgb_filter":  # or depth
            cond_data = image_rgb
        elif cfg.model.condition_source == "depth_filter":  # B,c,w,h
            # repeat number of channels
            # print("depths", depths.shape)
            # print("image_rgb", image_rgb.shape)
            cond_data = depths.repeat(1, 3, 1, 1)
            # print("depths_rep", depths_rep.shape)
        # print("shape x_t",x_t.shape)
        # print("T shape",t.shape)
        return pcpm.get_input_with_conditioning(
            x_t, camera=camera, image_rgb=cond_data, mask=mask,
            # t=torch.tensor([t])
            t=torch.tensor(t)
        )
    else:
        raise ValueError(
            f"cfg.model.condition_source {cfg.model.condition_source} not supported, must be 'image_rgb_filter' or 'depth_filter' or  'unconditional_filter'"
        )


# Sampling function


@torch.no_grad()
def sample(
    model,
    scheduler,
    cfg: ProjectConfig,
    camera=None,
    image_rgb=None,
    depths=None,
    mask=None,
    color_channels=None,
    predict_color=False,
    num_inference_steps=None,
    device="cpu",
    pcpm: PointCloudProjectionModel = None,
    data_mean=torch.tensor([0, 0, 0]),
    data_std=torch.tensor([1, 1, 1]),
):

    st_time = time.time()

    evolution_freq = cfg.run.evolution_freq

    data_mean = data_mean.to(device).float()
    data_std = data_std.to(device).float()

    assert (
        camera is not None and image_rgb is not None
    ), "camera, image_rgb must be provided"
    assert pcpm is not None, "pcpm must be provided"

    model.eval()
    num_inference_steps = num_inference_steps or scheduler.config.num_train_timesteps
    scheduler.set_timesteps(num_inference_steps)

    N = cfg.dataset.max_points
    B = 1 if image_rgb is None else image_rgb.shape[0]
    D = 3 + (color_channels if predict_color else 0)

    x_t = torch.randn(B, N, D, device=device)
    # print("dev x_t", x_t.device)
    accepts_offset = "offset" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )
    extra_set_kwargs = {"offset": 1} if accepts_offset else {}
    scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    # extra_step_kwargs = {"eta": eta} if accepts_eta else {}
    extra_step_kwargs = {}
    xs = []
    x0t = []
    steps = []
    logger.info(f"{os.getpid()}, sample/1_setup, {time.time()-st_time}")
    st_time = time.time()
    for i, t in enumerate(
        tqdm(scheduler.timesteps.to(device), desc="Sampling", leave=False)
    ):

        # Conditioning
        x_t_input = apply_conditioning_to_xt(
            cfg,
            (x_t * data_std + data_mean) if cfg.dataset.is_scaled else x_t,
            camera,
            image_rgb,
            depths,
            mask,
            t,
            pcpm,
        )
        logger.info(
            f"{os.getpid()}, sample/2_apply_conditioning_to_xt, {time.time()-st_time}")
        st_time = time.time()

        if cfg.dataset.is_scaled:
            # scale back the first 3 dimensions
            x_t_input[:, :, :3] = x_t[:, :, :3]

        noise_pred = model(x_t_input, t.reshape(1).expand(B))

        logger.info(
            f"{os.getpid()}, sample/3_model_inference, {time.time()-st_time}")
        st_time = time.time()

        x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs)

        logger.info(f"{os.getpid()}, sample/4_DM_step, {time.time()-st_time}")
        st_time = time.time()

        # Convert output back into a point cloud, undoing normalization and scaling
        output_prev = (
            pcpm.tensor_to_point_cloud(
                x_t.prev_sample, denormalize=True, unscale=True)
            .points_padded()
            .to(device)
        )
        output_original_sample = (
            pcpm.tensor_to_point_cloud(
                x_t.pred_original_sample, denormalize=True, unscale=True
            )
            .points_padded()
            .to(device)
        )

        x_t = x_t.prev_sample
        # print("dev x_t", x_t.device)
        # dev x_t cuda:0

        if (
            evolution_freq is not None and i % evolution_freq == 0
        ) or i == num_inference_steps - 1:
            xs.append(output_prev)
            steps.append(t)
            x0t.append(output_original_sample)

        logger.info(
            f"{os.getpid()}, sample/5_extraction, {time.time()-st_time}")
        st_time = time.time()
    if num_inference_steps == 1:
        xs.append(output_prev)
        steps.append(0)
        x0t.append(output_original_sample)

    xs = torch.concat(xs, dim=0)
    steps = torch.tensor(steps)
    x0t = torch.concat(x0t, dim=0)

    logger.info(f"{os.getpid()}, sample/6_tensor_ops, {time.time()-st_time}")
    st_time = time.time()

    return output_prev, xs, x0t, steps


# PVCNN-Based
class PVCNNDiffusionModel3D(nn.Module):
    def __init__(
        # pvcnnplusplus, pvcnn, simple
        self,
        data_dim,
        point_cloud_model_embed_dim=64,
        point_cloud_model="pvcnn",
        dropout=0.1,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__()

        self.in_channels = data_dim  # 3
        self.out_channels = 3
        self.scale_factor = 1.0
        self.dropout = dropout
        self.width_multiplier = width_multiplier
        self.voxel_resolution_multiplier = voxel_resolution_multiplier

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dropout=self.dropout,
            width_multiplier=self.width_multiplier,
            voxel_resolution_multiplier=self.voxel_resolution_multiplier,
        )

    def forward(self, x, t):
        # (B, N, 3) (B,) #x torch.Size([1, 100, 3]) t torch.Size([1])
        noise_pred = self.point_cloud_model(x, t)

        return noise_pred


def get_astyxdataset(cfg: ProjectConfig, device="cpu"):
    # cfg1.dataloader.batch_size
    print("batch_size", cfg.dataloader.batch_size)
    train_dataset = AstyxDataset(
        cfg.dataloader.num_scenes,
        cfg.dataset.max_points,
        "vits",
        device=device,
        is_scaled=cfg.dataset.is_scaled,
    )
    dataloader_train, dataloader_val, dataloader_vis = (
        DataLoader(
            train_dataset,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
        None,
        None,
    )
    return dataloader_train, dataloader_val, dataloader_vis


def get_pc2dataset(cfg):
    dataset_cfg: CO3DConfig = cfg.dataset
    # category:'car'
    # max_points: int = 16_384
    # image_size: int = 224
    # mask_images: bool = '${model.use_mask}'
    # use_mask: bool = True
    # restrict_model_ids: Optional[List] = None
    # subset_name: str = '80-20'
    # root: str = os.getenv('ASTYX_DATASET_ROOT')

    dataloader_cfg: DataloaderConfig = cfg.dataloader

    # Exclude bad and low-quality sequences
    exclude_sequence = []
    exclude_sequence.extend(EXCLUDE_SEQUENCE.get(dataset_cfg.category, []))
    exclude_sequence.extend(LOW_QUALITY_SEQUENCE.get(dataset_cfg.category, []))

    # Whether to load pointclouds
    kwargs = dict(
        remove_empty_masks=True,
        n_frames_per_sequence=1,
        load_point_clouds=True,
        max_points=dataset_cfg.max_points,
        image_height=dataset_cfg.image_size,
        image_width=dataset_cfg.image_size,
        mask_images=dataset_cfg.mask_images,
        exclude_sequence=exclude_sequence,
        pick_sequence=(
            ()
            if dataset_cfg.restrict_model_ids is None
            else dataset_cfg.restrict_model_ids
        ),
    )

    # Get dataset mapper
    dataset_map_provider_type = registry.get(
        JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2"
    )
    expand_args_fields(dataset_map_provider_type)
    dataset_map_provider = dataset_map_provider_type(
        category=dataset_cfg.category,
        subset_name=dataset_cfg.subset_name,
        dataset_root=dataset_cfg.root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig(kwargs),
    )

    # Get datasets
    datasets = dataset_map_provider.get_dataset_map()

    # print length of train, val, test
    print("len(train)", len(datasets["train"]))
    print("len(val)", len(datasets["val"]))
    print("len(test)", len(datasets["test"]))
    print("M", cfg.dataloader.num_scenes)
    # datasets["train"].frame_annots = datasets["train"].frame_annots[:M]
    # randoimly sample M point, using random.sample
    datasets["train"].frame_annots = random.sample(
        datasets["train"].frame_annots, cfg.dataloader.num_scenes
    )

    print("frame_annots", len(datasets["train"].frame_annots))
    print("seq_annots", len(datasets["train"].seq_annots))

    print("len(train)", len(datasets["train"]))

    # len(train) 144
    # len(val) 46
    # len(test) 144
    # M 1

    # pick only M first item in train,val,test

    # PATCH BUG WITH POINT CLOUD LOCATIONS!
    for dataset in (datasets["train"], datasets["val"]):
        for key, ann in dataset.seq_annots.items():
            correct_point_cloud_path = Path(dataset.dataset_root) / Path(
                *Path(ann.point_cloud.path).parts[-3:]
            )
            assert correct_point_cloud_path.is_file(), correct_point_cloud_path
            ann.point_cloud.path = str(correct_point_cloud_path)

    # Get dataloader mapper
    data_loader_map_provider_type = registry.get(
        SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider"
    )
    expand_args_fields(data_loader_map_provider_type)
    data_loader_map_provider = data_loader_map_provider_type(
        batch_size=dataloader_cfg.batch_size,
        num_workers=dataloader_cfg.num_workers,
    )

    # QUICK HACK: Patch the train dataset because it is not used but it throws an error
    if (
        len(datasets["train"]) == 0
        and len(datasets[dataset_cfg.eval_split]) > 0
        and dataset_cfg.restrict_model_ids is not None
        and cfg.run.job == "sample"
    ):
        datasets = DatasetMap(
            train=datasets[dataset_cfg.eval_split],
            val=datasets[dataset_cfg.eval_split],
            test=datasets[dataset_cfg.eval_split],
        )
        print(
            "Note: You used restrict_model_ids and there were no ids in the train set."
        )

    # Get dataloaders
    dataloaders = data_loader_map_provider.get_data_loader_map(datasets)
    dataloader_train = dataloaders["train"]
    dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

    # Replace validation dataloader sampler with SequentialSampler
    dataloader_val.batch_sampler.sampler = SequentialSampler(
        dataloader_val.batch_sampler.sampler.data_source
    )

    # Modify for accelerate
    dataloader_train.batch_sampler.drop_last = True
    dataloader_val.batch_sampler.drop_last = False

    return dataloader_train, dataloader_val, dataloader_vis


def get_model(cfg: ProjectConfig, device="cpu", pcpm=None):

    if cfg.model.condition_source == "unconditional_filter":
        data_dim = 3
    else:
        assert pcpm is not None, "pcpm must be provided"
        data_dim = pcpm.in_channels
    print("data_dim", data_dim)

    return PVCNNDiffusionModel3D(
        data_dim=data_dim,
        point_cloud_model_embed_dim=cfg.model.point_cloud_model_embed_dim,
        point_cloud_model=cfg.model.point_cloud_model,
        dropout=0.1,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ).to(device)


def get_loss(cfg: ProjectConfig):
    if cfg.loss.loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    elif cfg.loss.loss_type == "chamfer":
        return PointCloudLoss(npoints=cfg.dataset.max_points, emd_weight=0)
    elif cfg.loss.loss_type == "emd":
        return PointCloudLoss(npoints=cfg.dataset.max_points, emd_weight=1)
    else:
        raise ValueError("loss not supported")


def log_utils(log_type="static", model=None, writer=None, epoch=None):
    assert log_type in [
        "static",
        "dynamic",
    ], f"log_type {log_type} must be either 'static' or 'dynamic'"
    assert writer is not None, "writer must be provided"

    data = {}

    if log_type == "static":
        if model is None:
            raise ValueError("model must be provided for static logging")

        # model param_count
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        data["model/param_count_M"] = param_count

        # log ,max ram,
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        data["gpu/GB_ram"] = meminfo.total / 2**30
        pynvml.nvmlShutdown()
        # log GPU model, e.g., 3080, etc., qurying the GPU model
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        data["gpu/model"] = str(gpu_name)
        # CPU: model
        # CPU: model
        cpu_name = platform.processor()
        data["cpu/processor"] = str(cpu_name)  # x86_64
        # cpu commcerical name
        cpu_commercial_name = platform.processor()

        # CPU: core count
        cpu_core_count = psutil.cpu_count(logical=False)
        data["cpu/core_count"] = cpu_core_count
        # CPU: thread count
        cpu_thread_count = psutil.cpu_count(logical=True)
        data["cpu/thread_count"] = cpu_thread_count
        # CPU: max ram
        cpu_max_ram = psutil.virtual_memory().total / 2**30
        data["cpu/GB_ram"] = cpu_max_ram
        # print(data)

        data["host/name"] = os.uname().nodename

        gpu_mem_util = pynvml.nvmlDeviceGetMemoryInfo(handle)
        data["gpu/mem_utilization_total_GB"] = gpu_mem_util.total / 2**30

        for key, value in data.items():
            # if str, add_text, else add_scalar
            if isinstance(value, str):
                writer.add_text(key, value)
            else:
                writer.add_scalar(key, value)
    elif log_type == "dynamic":
        if epoch is None:
            raise ValueError("epoch must be provided for dynamic logging")
        # GPU utilization/temp/mem utilization,fan speed, power consumption

        # CPU utilization/temp/mem utilization,fan speed, power consumption

        # disk/network/swap utilization,

        # GPU utilization/temp/mem utilization, fan speed, power consumption
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        data["gpu/utilization"] = gpu_util.gpu

        gpu_temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU)
        data["gpu/temperature"] = gpu_temp

        gpu_mem_util = pynvml.nvmlDeviceGetMemoryInfo(handle)
        data["gpu/mem_utilization_percent"] = (
            gpu_mem_util.used / gpu_mem_util.total * 100
        )
        data["gpu/mem_utilization_GB"] = gpu_mem_util.used / 2**30
        try:
            gpu_fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            data["gpu/fan_speed"] = gpu_fan_speed
        except Exception as e:
            print("error", e)

        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle)
        data["gpu/power_consumption"] = gpu_power / 1000  # Convert to watts

        pynvml.nvmlShutdown()

        # CPU utilization/temp/mem utilization, fan speed, power consumption
        cpu_util = psutil.cpu_percent(interval=1)
        data["cpu/utilization"] = cpu_util
        # print(psutil.sensors_temperatures())

        # {'nvme': [shwtemp(label='Composite', current=35.85, high=84.85, critical=84.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=40.85, high=65261.85, critical=65261.85)], 'enp3s0': [shwtemp(label='MAC Temperature', current=51.124, high=None, critical=None)], 'k10temp': [shwtemp(label='Tctl', current=77.875, high=None, critical=None), shwtemp(label='Tdie', current=50.875, high=None, critical=None), shwtemp(label='Tccd1', current=68.75, high=None, critical=None), shwtemp(label='Tctl', current=76.75, high=None, critical=None), shwtemp(label='Tdie', current=49.75, high=None, critical=None)], 'iwlwifi_1': [shwtemp(label='', current=31.0, high=None, critical=None)]}
        for key, value in psutil.sensors_temperatures().items():
            # print(key,":")
            for sensor in value:
                # print(sensor.label, sensor.current, sensor.high, sensor.critical)
                data[f"temperature/{key}/{sensor.label}"] = sensor.current

        cpu_mem = psutil.virtual_memory()
        data["mem/utilization_percent"] = cpu_mem.percent

        # Disk utilization
        disk_usage = psutil.disk_usage("/")
        data["disk/utilization"] = disk_usage.percent

        # Network utilization
        net_io = psutil.net_io_counters()
        data["network/bytes_sent_GB"] = net_io.bytes_sent / 2**30
        data["network/bytes_recv_GB"] = net_io.bytes_recv / 2**30

        # Swap utilization
        swap = psutil.swap_memory()
        data["swap/utilization"] = swap.percent

        # print(data)
        for key, value in data.items():
            # if str, add_text, else add_scalar
            if isinstance(value, str):
                writer.add_text(key, value, epoch)
            else:
                writer.add_scalar(key, value, epoch)


def match_args_json(cfg1, cfg2):
    try:
        return (
            cfg1["run"]["seed"] == cfg2["run"]["seed"]
            and cfg1["run"]["num_inference_steps"] == cfg2["run"]["num_inference_steps"]
            and cfg1["dataset"]["image_size"] == cfg2["dataset"]["image_size"]
            and cfg1["dataset"]["is_scaled"] == cfg2["dataset"]["is_scaled"]
            and cfg1["model"]["beta_schedule"] == cfg2["model"]["beta_schedule"]
            and cfg1["model"]["point_cloud_model_embed_dim"]
            == cfg2["model"]["point_cloud_model_embed_dim"]
            and cfg1["dataset"]["category"] == cfg2["dataset"]["category"]
            and cfg1["dataset"]["type"] == cfg2["dataset"]["type"]
            and cfg1["dataset"]["max_points"] == cfg2["dataset"]["max_points"]
            and cfg1["optimizer"]["lr"] == cfg2["optimizer"]["lr"]
            and cfg1["dataloader"]["batch_size"] == cfg2["dataloader"]["batch_size"]
            and cfg1["dataloader"]["num_scenes"] == cfg2["dataloader"]["num_scenes"]
            and cfg1["loss"]["loss_type"] == cfg2["loss"]["loss_type"]
            and cfg1["optimizer"]["name"] == cfg2["optimizer"]["name"]
            and cfg1["optimizer"]["weight_decay"] == cfg2["optimizer"]["weight_decay"]
            and cfg1["optimizer"]["kwargs"]["betas"][0]
            == cfg2["optimizer"]["kwargs"]["betas"][0]
            and cfg1["optimizer"]["kwargs"]["betas"][1]
            == cfg2["optimizer"]["kwargs"]["betas"][1]
            and cfg1["model"]["condition_source"] == cfg2["model"]["condition_source"]
            # and cfg1.run.num_inference_steps == cfg2.run.num_inference_steps
            # and cfg1.dataset.image_size == cfg2.dataset.image_size
            # and cfg1.model.beta_schedule == cfg2.model.beta_schedule
            # and cfg1.model.point_cloud_model_embed_dim
            # == cfg2.model.point_cloud_model_embed_dim
            # and cfg1.dataset.category == cfg2.dataset.category
            # and cfg1.dataset.type == cfg2.dataset.type
            # and cfg1.dataset.max_points == cfg2.dataset.max_points
            # and cfg1.optimizer.lr == cfg2.optimizer.lr
            # and cfg1.dataloader.batch_size == cfg2.dataloader.batch_size
            # and cfg1.dataloader.num_scenes == cfg2.dataloader.num_scenes
            # and cfg1.loss.loss_type == cfg2.loss.loss_type
            # and cfg1.optimizer.name == cfg2.optimizer.name
            # and cfg1.optimizer.weight_decay == cfg2.optimizer.weight_decay
            # and cfg1.optimizer.kwargs.betas[0] == cfg2.optimizer.kwargs.betas[0]
            # and cfg1.optimizer.kwargs.betas[1] == cfg2.optimizer.kwargs.betas[1]
        )
    except KeyError as e:
        # print("key not found", e)
        return False


def get_checkpoint_fname_json(cfg: ProjectConfig, db_fname, CHECKPOINT_DIR):
    # check checkpoint
    db_fname = f"{CHECKPOINT_DIR}/{db_fname}"
    current_cp_fname = None
    max_epoch = 0

    # load file line by line

    print("db_fname", db_fname)
    print("exists", os.path.exists(db_fname))
    with open(db_fname, "r") as f:
        # extract all lines to a list
        lines = f.readlines()
        for line in tqdm(lines):
            dat = json.loads(line)
            # print("checking checkpoint", dat["fname"])
            if not os.path.exists(dat["fname"]):
                print(dat["fname"].split("/")[-1], "not found")
                continue
            print(".", end="")
            if match_args_json(dat["args"], to_dict(cfg)):
                if (
                    dat["args"]["run"]["max_steps"] <= cfg.run.max_steps
                    and dat["args"]["run"]["max_steps"] > max_epoch
                ):
                    max_epoch = dat["args"]["run"]["max_steps"]
                    current_cp_fname = dat["fname"]
                    print("this is ok", current_cp_fname)
                    # print("current_cp", current_cp_fname)
                else:
                    print(
                        "epoch in config",
                        dat["args"]["run"]["max_steps"],
                        " is less than max epoch in checkpoint",
                        cfg.run.max_steps,
                        "or max epoch in checkpoint is less than max epoch in checkpoint",
                        max_epoch,
                    )

    return current_cp_fname


def get_dataset(cfg: ProjectConfig, device="cpu"):

    if cfg.dataset.type == "astyx":
        dataloader_train, _, _ = get_astyxdataset(cfg, device)
    elif cfg.dataset.type == "co3dv2":
        dataloader_train, _, _ = get_pc2dataset(cfg)
    else:
        raise ValueError(
            f"dataset type {cfg.dataset.type} not supported, must be 'astyx' or 'co3dv2'"
        )
    return dataloader_train


def calculate_chamfer_distance(is_scaled, mean, std, gt, pred, device):
    mean = mean.to(device)
    std = std.to(device)
    gt = gt.to(device)
    pred = pred.to(device)

    # gt, pred: B,N,3
    if is_scaled:
        gt = gt * std + mean
        pred = pred * std + mean
    return chamfer_distance(gt, pred)


def process_ema_prev_values(writer, ema_factors, epochs, prev_values: dict, name):

    emas = {idx: {k: None for k in ema_factors} for idx in prev_values}
    if "ave" not in emas:
        emas["ave"] = {k: None for k in ema_factors}

    all_scenes = []

    for scene_id, losses in prev_values.items():  # {scene_id: [loss]}
        if len(losses) == 0:
            continue
        for alpha in ema_factors:
            ema_list = aggregate_ema(losses, alpha)
            emas[scene_id][alpha] = ema_list[-1]
            for epoch, ema in zip(
                epochs if epochs is not None else range(
                    len(ema_list)), ema_list
            ):
                writer.add_scalars(
                    (
                        f"{name}/scene_{scene_id:03d}"
                        if scene_id != "ave"
                        else f"{name}/average"
                    ),
                    {f"ema_{alpha:.2e}": ema},
                    epoch,
                )

        all_scenes.append(losses)
    # convert to tensor and average scroos scenes

    if len(all_scenes) == 0:
        return emas

    ave_all_scenes = torch.tensor(all_scenes).mean(dim=0)

    for alpha in ema_factors:
        ema_list = aggregate_ema(ave_all_scenes, alpha)

        emas["ave"][alpha] = ema_list[-1]
        for epoch, ema in zip(
            epochs if epochs is not None else range(len(ema_list)), ema_list
        ):
            writer.add_scalars(f"{name}/average",
                               {f"ema_{alpha:.2e}": ema}, epoch)
    return emas


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: ProjectConfig):

    log_filepath = f'/home/palakons/from_scratch/app_4.csv'
    file_handler = logging.FileHandler(log_filepath, mode='a')
    formatter = logging.Formatter(
        f'%(asctime)s,%(name)s,%(levelname)s,{cfg.run.name.replace("/", "_")},%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    st_time = time.time()
    # print(cfg)
    CHECKPOINT_DIR = "checkpoint_pc2"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_DB_FILE = "checkpoint_db_track_loss.json"

    set_seed(cfg.run.seed)
    tb_log_dir = "tb_log"
    log_dir = tb_log_dir + f"/{cfg.run.name}"
    writer = SummaryWriter(log_dir=log_dir)
    print("tensorboard log at", log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    pcpm = PointCloudProjectionModel(
        image_size=cfg.model.image_size,
        image_feature_model=cfg.model.image_feature_model,
        use_local_colors=cfg.model.use_local_colors,
        use_local_features=cfg.model.use_local_features,
        use_global_features=cfg.model.use_global_features,
        use_mask=cfg.model.use_mask,
        use_distance_transform=cfg.model.use_distance_transform,
        predict_shape=cfg.model.predict_shape,
        predict_color=cfg.model.predict_color,
        color_channels=cfg.model.color_channels,
        colors_mean=cfg.model.colors_mean,
        colors_std=cfg.model.colors_std,
        scale_factor=cfg.model.scale_factor,
        run_name=cfg.run.name.replace("/", "_"),
    ).to(device)

    dataloader_train = get_dataset(cfg, device=device)

    writer.add_text("data/ids", str(dataloader_train.dataset.active_ids))

    if False:

        batch = next(iter(dataloader_train))

        # depths, radar_data, camera, image_rgb, objects, idx, npoints, npoints_filtered, offsets = batch

        pc, camera, image_rgb, mask, depths = extract_batch(cfg, batch, device)
        fname = f"/home/palakons/from_scratch/plot_at_load.png"
        plot_image_depth_projected(
            radar_data[:1],
            None,
            fname,
            point_size=1,
            camera=camera.to(device)[0],
            image_rgb=image_rgb[:1],
            depth_image=depths[:1],
        )

    model = get_model(cfg, device=device, pcpm=pcpm)

    if cfg.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.kwargs.betas,
        )
    elif cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.kwargs.betas,
        )
    else:
        raise ValueError(
            f"optimizer {cfg.optimizer.name} not supported, pick Adam or AdamW"
        )
    # linear, scaled_linear, or squaredcos_cap_v2.
    assert cfg.run.diffusion_scheduler == "ddpm", "only ddpm supported"
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.run.num_inference_steps,
        beta_schedule=cfg.model.beta_schedule,
        prediction_type="epsilon",  # or "sample" or "v_prediction"
        clip_sample=False,
        beta_start=cfg.model.beta_start,
        beta_end=cfg.model.beta_end,
    )
    start_epoch = 0

    current_cp_fname = None

    if cfg.checkpoint.resume_training:
        if os.path.exists(f"{CHECKPOINT_DIR}/{CHECKPOINT_DB_FILE}"):
            current_cp_fname = get_checkpoint_fname_json(
                cfg, CHECKPOINT_DB_FILE, CHECKPOINT_DIR
            )
    else:
        print("CFG dictates not resuming training from checkpoint")

    prev_cds, prev_losses = {"epochs": [], "data": {}}, {"ave": []}
    if current_cp_fname is not None:
        current_cp = torch.load(current_cp_fname)
        print("loading checkpoint", current_cp_fname)
        model.load_state_dict(current_cp["model"])
        optimizer.load_state_dict(current_cp["optimizer"])
        start_epoch = current_cp["args"].run.max_steps
        print("start_epoch", start_epoch)

        if "cds" in current_cp:
            prev_cds = current_cp["cds"]
            prev_losses = current_cp["losses"]
            assert (
                len(prev_losses["ave"]) == start_epoch
            ), f"prev_losses length {len(prev_losses['ave'])} must be equal to start_epoch {start_epoch}"
    else:
        print("no checkpoint found")

    log_utils(log_type="static", model=model, writer=writer, epoch=None)

    assert cfg.loss.loss_type == "mse", "only mse supported"
    criterion = get_loss(cfg)

    print(f"time: before tain {time.time()-st_time:.2f}")
    # save into csv CSV_TIME_FILE

    logger.info(f"{os.getpid()},before train,{time.time()-st_time}")
    # Train the model
    loss_emas, cd_emas = train(
        model,
        # dataloader,
        dataloader_train,
        optimizer,
        scheduler,
        cfg,
        device=device,
        start_epoch=start_epoch,
        criterion=criterion,
        writer=writer,
        pcpm=pcpm,
        loss_ema_factors=[0.9, 0.95, 0.975, 0.99],
        prev_cds=prev_cds,
        prev_losses=prev_losses,
        CHECKPOINT_DB_FILE=CHECKPOINT_DB_FILE,
    )

    for handler in logger.handlers:
        handler.flush()
    exit()

    metric_dict = {
        **{f"Loss_ave/ema_{k:.2e}": loss_emas["ave"][k] for k in loss_emas["ave"]},
        **{f"CD_ave/ema_{k:.2e}": cd_emas["ave"][k] for k in cd_emas["ave"]},
    }

    # Sample from the model

    batch = next(iter(dataloader_train))

    pc, camera, image_rgb, mask, depths, idx = extract_batch(
        cfg, batch, device)

    samples = {}
    for i in [1, 5, 10, 50, 100, scheduler.config.num_train_timesteps]:

        samples[f"step{i}"] = sample(
            model,
            scheduler,
            cfg,
            camera=camera[0],
            image_rgb=image_rgb[:1],
            depths=depths[:1] if depths is not None else None,
            mask=mask[:1] if mask is not None else None,
            num_inference_steps=i,
            device=device,
            pcpm=pcpm,
            data_mean=dataloader_train.dataset.data_mean,
            data_std=dataloader_train.dataset.data_std,
        )

    if False:  # Evo plots
        # make the plot that will be logged to tb
        gt_cond_pc = pcpm.point_cloud_to_tensor(
            pc[:1], normalize=True, scale=True)
        # print("samples_updated", samples_updated.shape)
        # print("gt_cond_pc", gt_cond_pc.shape)
        # samples_updated torch.Size([1, 128, 3])
        # gt_cond_pc torch.Size([2, 128, 3])

        plt.figure(figsize=(10, 10))

        for key, value in samples.items():
            samples_updated = samples[key][0]

            cd_loss, _ = calculate_chamfer_distance(
                cfg.dataset.is_scaled,
                dataloader_train.dataset.data_mean,
                dataloader_train.dataset.data_std,
                gt_cond_pc,
                samples_updated,
                device,
            )

            print(
                key,
                "\t",
                f"CD: {cd_loss:.2f}",
            )

            error = []
            assert len(samples[key][1]) > 1, "need more than 1 sample to plot"
            for x in samples[key][1]:

                cd_loss, _ = calculate_chamfer_distance(
                    cfg.dataset.is_scaled,
                    dataloader_train.dataset.data_mean,
                    dataloader_train.dataset.data_std,
                    gt_cond_pc,
                    x.unsqueeze(0),
                    device,
                )

                error.append(cd_loss.item())
            # ax = plt.plot(error, label=key)
            plt.plot(
                [i / (len(error) - 1) for i in range(len(error))],
                error,
                label=key,
                marker=None,
            )
            # print()
        plt.legend()
        plt.title(
            f"Diffusion model: {cfg.run.max_steps } epochs, {cfg.run.num_inference_steps } timesteps, {cfg.model.beta_schedule } schedule",
        )
        plt.xlabel("Evolution steps ratio")
        plt.ylabel("Chamfer distance")
        # ylog
        plt.yscale("log")

        writer.add_figure(f"Evolution", plt.gcf(), cfg.run.max_steps)
        plt.close()

        # add cd to metric_dict
        metric_dict = {**{f"CD": cd_loss}, **metric_dict}

    print("done evo plots")

    hparam_dict = {
        "seed": cfg.run.seed,  #
        "epochs": cfg.run.max_steps,
        "num_inference_steps": cfg.run.num_inference_steps,  #
        "image_size": cfg.dataset.image_size,  #
        "beta_schedule": cfg.model.beta_schedule,  #
        "point_cloud_model_embed_dim": cfg.model.point_cloud_model_embed_dim,  #
        "dataset_cat": cfg.dataset.category,
        "dataset_source": cfg.dataset.type,
        "max_points": cfg.dataset.max_points,  #
        "lr": cfg.optimizer.lr,  #
        "batch_size": cfg.dataloader.batch_size,
        "num_scenes": cfg.dataloader.num_scenes,
        "loss_type": cfg.loss.loss_type,  #
        "optimizer": cfg.optimizer.name,  #
        "optimizer_decay": cfg.optimizer.weight_decay,
        "optimizer_beta_0": cfg.optimizer.kwargs.betas[0],
        "optimizer_beta_1": cfg.optimizer.kwargs.betas[1],
    }

    try:
        writer.add_hparams(hparam_dict, metric_dict)
    except Exception as e:
        print("error adding hparams", e)
    writer.close()

    for handler in logger.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
