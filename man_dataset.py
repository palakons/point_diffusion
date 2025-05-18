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
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer.cameras import PerspectiveCameras

from matplotlib import pyplot as plt

class MANDataset(Dataset):
    def __init__(
        self,
        M,
        N,
        scene_id,
        depth_model="vits",
        random_offset: bool = False,
        data_file: str = "man-mini",  # man-full
        device="cpu",
        is_scaled=True,
        img_size=618,
        radar_channel='RADAR_LEFT_FRONT',
        camera_channel='CAMERA_RIGHT_FRONT'
    ):
        self.M = M
        self.N = N
        self.depth_model = depth_model  # ['vitl','vitb','vits']
        self.random_offset = random_offset
        self.device = device
        self.is_scaled = is_scaled
        self.img_size = img_size
        self.data_file = data_file
        self.scene_id = scene_id
        self.radar_channel = radar_channel
        self.camera_channel = camera_channel
        
        # store mean and std

        if self.data_file == "man-mini":
            self.data_root = '/data/palakons/new_dataset/MAN/mini/man-truckscenes'
            trucksc = TruckScenes('v1.0-mini', self.data_root, False)
        elif self.data_file == "man-full":
            self.data_root = '/data/palakons/new_dataset/MAN/man-truckscenes'
            trucksc = TruckScenes('v1.0-trainval', self.data_root, False)
        else:
            raise ValueError(f"Unknown data_file: {self.data_file}")

        self.data_bank = []
        first_frame_token = trucksc.scene[self.scene_id]['first_sample_token']
        frame_token = first_frame_token
        i = 0

        while frame_token != "":

            i = i+1

            self.data_bank.append(self.load_data(trucksc, frame_token))
            print("loaded", i, "frame_token", frame_token)
            if len(self.data_bank) >= self.M:
                break
            frame_token = trucksc.get('sample', frame_token)['next']

        if len(self.data_bank) < self.M:
            print(
                f"Warning: only {len(self.data_bank)} samples found in scene {self.scene_id}")

        all_radar_positions = torch.stack(
            [d[1] for d in self.data_bank], dim=0)
        print("all_radar_positions", all_radar_positions.shape)  # 1x 16 x3
        # print("all_radar_positions", all_radar_positions)  # 1x 16 x3

        # calculate means
        dims = all_radar_positions.shape

        self.data_mean = all_radar_positions.reshape(-1, dims[-1]).mean(axis=0)
        self.data_std = all_radar_positions.reshape(-1, dims[-1]).std(axis=0)
        # std are 1s, same shape of data_mean
        if all_radar_positions.reshape(-1, dims[-1]).shape[0] == 1:
            self.data_std = torch.ones_like(self.data_mean)

        # print("self.data_mean", self.data_mean)
        # print("self.data_std", self.data_std)
        depth_image_dir = os.path.join(self.data_root, "depth_images")
        if not os.path.exists(depth_image_dir):
            os.makedirs(depth_image_dir)



    def inverse_SE3(self,T: torch.Tensor) -> torch.Tensor:
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = torch.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv
    def quat2mat(self,q):  # q in (w, x, y, z)
        # === Convert quaternions to rotation matrices ===
        return quaternion_to_matrix(torch.tensor([q[0], q[1], q[2], q[3]])[None])[0]


    def to_homogeneous(self,R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert rotation and translation to a 4x4 homogeneous transform."""
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def calib_to_camera_base_MAN(self,cam_calib, cam_pose, radar_calib, radar_pose, image_size_hw: tuple, offset: int = 0, image_size: int = 618, step=4):

        # === Build individual transforms as 4x4 homogeneous matrices ===
        T_radar2ego = self.to_homogeneous(self.quat2mat(radar_calib['rotation']), torch.tensor(
            radar_calib['translation'])) if step >= 1 else torch.eye(4)
        T_ego2global_r = self.to_homogeneous(self.quat2mat(radar_pose['rotation']), torch.tensor(
            radar_pose['translation'])) if step >= 2 else torch.eye(4)
        T_global2ego_c = self.to_homogeneous(self.quat2mat(cam_pose['rotation']), torch.tensor(
            cam_pose['translation'])) if step >= 3 else torch.eye(4)
        T_ego2cam = self.to_homogeneous(self.quat2mat(cam_calib['rotation']), torch.tensor(
            cam_calib['translation'])) if step >= 4 else torch.eye(4)

        # === Compose full radar → camera transform === but some conventions like Pytorch3D uses right matrix multiplication in computation procedure

        # T_radar2cam = T_radar2ego.sT @ T_ego2global_r.T @ torch.linalg.inv(T_global2ego_c).T @ torch.linalg.inv(T_ego2cam).T

        T_global2ego_c_inv = self.inverse_SE3(T_global2ego_c)
        T_ego2cam_inv = self.inverse_SE3(T_ego2cam)

        T_radar2cam = T_radar2ego.T @ T_ego2global_r.T @ T_global2ego_c_inv.T @ T_ego2cam_inv.T

        R = T_radar2cam[:3, :3].unsqueeze(0)  # (1, 3, 3)
        T = (T_radar2cam[3, :3].T).unsqueeze(0)   # (1, 3)

        # print("R", R,R.shape            )
        # print("R-1R", torch.linalg.inv(R) @ R)
        # print("T", T,T.shape)

        MAN_image_height = 943

        s = image_size/MAN_image_height
        # Convert intrinsic matrix K to tensor

        K = torch.tensor(cam_calib['camera_intrinsic'])

        K[0, 0] = K[0, 0] * s
        K[1, 1] = K[1, 1] * s

        K[0, 2] = K[0, 2] * s
        K[1, 2] = K[1, 2] * s

        offset = offset * s

        # Extract focal length and principal point from K
        focal_length = torch.tensor([[K[0, 0], K[1, 1]]])
        principal_point = torch.tensor([[K[0, 2] - offset, K[1, 2]]])

        # Create a PerspectiveCameras object
        return PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            in_ndc=False,
            image_size=[image_size_hw],
        )

    def load_data(self, trucksc, frame_token):

        frame = trucksc.get('sample', frame_token)
        cam = trucksc.get('sample_data', frame['data'][self.camera_channel])
        radar = trucksc.get('sample_data', frame['data'][self.radar_channel])

        # 1) load depth
        depth_image_path = os.path.join(self.data_root, "depth_images", cam['filename'].replace(
            '.jpg', f'_{self.depth_model}.png'))

        if not os.path.exists(depth_image_path):
            raise ValueError(f"Depth image not found: {depth_image_path}")

        depth_image = plt.imread(depth_image_path)
        self.original_image_size = (depth_image.shape[0], depth_image.shape[1])
        # print("original_image_size", original_image_size) # original_image_size (943, 1980)
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        if depth_image.max() > 1.0:
            depth_image = depth_image / 255.0
        depth_image = torch.tensor(depth_image, dtype=torch.float32)

        self.square_image_offset = (
            int((depth_image.shape[1] - depth_image.shape[0]) / 2)
            if not self.random_offset
            else random.randint(0, depth_image.shape[1] - depth_image.shape[0])
        )

        depth_image = depth_image[:,
                                  self.square_image_offset: self.square_image_offset + depth_image.shape[0]]

        depth_image = torch.nn.functional.interpolate(
            depth_image.unsqueeze(0).unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # 2) load radar
        # load pcd
        radar_pcd_path = os.path.join(self.data_root, radar['filename'])
        cloud = o3d.io.read_point_cloud(radar_pcd_path)   
        radar_data = torch.tensor(cloud.points)
        npoints_original = radar_data.shape[0]

        # 3) load calibration

        # print calibration
        cam_calib = trucksc.get('calibrated_sensor',
                                cam['calibrated_sensor_token'])
        radar_calib = trucksc.get(
            'calibrated_sensor', radar['calibrated_sensor_token'])
        cam_pose = trucksc.get('ego_pose', cam['ego_pose_token'])
        radar_pose = trucksc.get('ego_pose', radar['ego_pose_token'])

        cam_calib_obj = self.calib_to_camera_base_MAN(cam_calib, cam_pose, radar_calib, radar_pose, [self.original_image_size[0]] *
                                                 2, self.square_image_offset, self.img_size
                                                 )

        # 4) load camera
        image_rgb_path = os.path.join(self.data_root, cam['filename'])

        camera_front = plt.imread(image_rgb_path).transpose(
            2, 0, 1
        )
        if camera_front.max() > 1.0:
            camera_front = camera_front / 255.0
        camera_front = torch.tensor(camera_front, dtype=torch.float32)
        # print("camera_front: ", camera_front.shape) #camera_front:  torch.Size([3, 618, 2048])
        camera_front = camera_front[
            :, :, self.square_image_offset: self.square_image_offset + camera_front.shape[1]
        ]

        # resize image to self.img_size
        camera_front = torch.nn.functional.interpolate(
            camera_front.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        world_points = radar_data.clone().detach().float()

        image_coord = cam_calib_obj.transform_points(world_points)
        points_uv = image_coord[:, :2]  # N x2

        # 5) filter points
        mask = (
            (points_uv[:, 1] >= 0)
            & (points_uv[:, 1] < camera_front.shape[1])
            & (points_uv[:, 0] >= 0)
            & (points_uv[:, 0] < camera_front.shape[1])
            # & (points_uv[:, 2] > 0)  # Ensure points are in front of the camera
        )
        filtered_radar_data = radar_data[mask]
        npoints_filtered = filtered_radar_data.shape[0]

        # 6) randomly filter point or sample to have N points
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

        return (
            depth_image,
            filtered_radar_data,
            cam_calib_obj,
            camera_front,
            frame_token,
            npoints_original,
            npoints_filtered
        )

    def __len__(self):
        return len(self.data_bank)

    def __getitem__(self, idx):
        if not self.is_scaled:

            return self.data_bank[idx]

        (
            depth_image,
            filtered_radar_data,
            cam_calib_obj,
            camera_front,
            frame_token,
            npoints_original,
            npoints_filtered
        ) = self.data_bank[idx]

        point_out = (filtered_radar_data - self.data_mean) / self.data_std

        return (
            depth_image,
            point_out,
            cam_calib_obj,
            camera_front,
            frame_token,
            npoints_original,
            npoints_filtered
        )


def custom_collate_fn_man(batch):
    (
        depths,
        radar_data,
        camera_base_list,
        camera_fronts,
        frame_token,
        npoints_original,
        npoints_filtered
    ) = zip(*batch)

    npoints_after = torch.as_tensor(npoints_original)
    npoints_filtered_after = torch.as_tensor(npoints_filtered)

    frame_token_after = list(frame_token)

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
        camera_fronts_after, frame_token_after,
        npoints_after,
        npoints_filtered_after,
    )