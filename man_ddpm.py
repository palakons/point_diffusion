from truckscenes import TruckScenes
import pypcd4
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

if False:
    from depth_anything_3.api import DepthAnything3
from datetime import datetime

from scipy.spatial import cKDTree
from PIL import Image, ImageOps
from transformers import CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm, trange
import random
import textwrap

import time
import sys
import torchvision.transforms.functional as TF
from torch.nn import functional as F

if False:
    sys.path.insert(0, "/home/palakons/PointNeXt")
    from openpoints.cpp.chamfer_dist import ChamferFunction, ChamferDistanceL2

sys.path.insert(0, "/home/palakons/Wan2.2")


class SimplePerspectiveCamera:
    """Simple perspective camera replacement for PyTorch3D PerspectiveCameras"""

    def __init__(self, focal_length, principal_point, R, T, image_size):
        """
        focal_length: (1, 2) tensor [fx, fy]
        principal_point: (1, 2) tensor [cx, cy]
        R: (1, 3, 3) rotation matrix (world to camera)
        T: (1, 3) translation vector (world to camera)
        image_size: [[height, width]]
        """
        self.fx = focal_length[0, 0]
        self.fy = focal_length[0, 1]
        self.cx = principal_point[0, 0]
        self.cy = principal_point[0, 1]
        self.R = R[0]  # (3, 3)
        self.T = T[0]  # (3,)
        self.image_size = image_size[0]  # [height, width]

        # Build intrinsic matrix K
        self.K = torch.tensor(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=torch.float32,
        )

    def transform_points(self, points_in_radar_view):
        """
        Transform 3D points_in_radar_view points to image coordinates.
        ( radar points in radar view -> vehicle's view -> camera view -> image coordinates ) --> R|T
        points_in_radar_view: (N, 3) tensor of 3D points in radar view coordinates
        Returns: (N, 3) tensor of [u, v, depth] in image coordinates
        """
        # Convert to homogeneous coordinates
        N = points_in_radar_view.shape[0]

        # Transform to camera coordinates: p_cam = R * p_world + T
        camera_points = (self.R @ points_in_radar_view.T).T + self.T  # (N, 3)

        # Project to image plane using intrinsic matrix
        # p_image = K * p_cam
        image_coords_homo = (self.K @ camera_points.T).T  # (N, 3)

        # Normalize by depth (z-coordinate)
        depth = image_coords_homo[:, 2:3]  # (N, 1)
        uvz = torch.cat(
            [
                image_coords_homo[:, 0:1] / depth,  # u
                image_coords_homo[:, 1:2] / depth,  # v
                depth,  # z (depth)
            ],
            dim=1,
        )

        return uvz, camera_points

def make_dummy_lshape(n_p, device):
    # We will build up to n_p points by concatenating simple segments.
    # First make a 32-point prototype; if n_p differs, resample crudely.

    n1 = max(8, int(n_p * 0.50))   # horizontal arm
    n2 = max(6, int(n_p * 0.30))   # vertical arm
    n3 = max(4, n_p - n1 - n2)     # elevated short arm

    # horizontal arm: x in [-1, 1], y=0, z=0
    x1 = torch.linspace(-1.0, 1.0, n1, device=device)
    y1 = torch.zeros(n1, device=device)
    z1 = torch.zeros(n1, device=device)
    arm1 = torch.stack([x1, y1, z1], dim=1)

    # vertical arm: x=0, y in [0, 1], z=0
    x2 = torch.zeros(n2, device=device)
    y2 = torch.linspace(0.0, 1.0, n2, device=device)
    z2 = torch.zeros(n2, device=device)
    arm2 = torch.stack([x2, y2, z2], dim=1)

    # elevated short arm: x=0.4, y in [0, 0.6], z=0.5
    x3 = torch.full((n3,), 0.4, device=device)
    y3 = torch.linspace(0.0, 0.6, n3, device=device)
    z3 = torch.full((n3,), 0.5, device=device)
    arm3 = torch.stack([x3, y3, z3], dim=1)

    pts = torch.cat([arm1, arm2, arm3], dim=0)

    # exact length safety
    if pts.shape[0] > n_p:
        pts = pts[:n_p]
    elif pts.shape[0] < n_p:
        # pad by repeating last point if ever needed
        pad = pts[-1:].repeat(n_p - pts.shape[0], 1)
        pts = torch.cat([pts, pad], dim=0)

    # center and scale to roughly [-1, 1]
    pts = pts - pts.mean(dim=0, keepdim=True)
    pts = pts / pts.abs().max()

    return pts

def chamfer_distance(pred_uvz, gt_uvz, device):
    # convert to tensor
    pred_uvz = torch.tensor(pred_uvz, device=device)
    gt_uvz = torch.tensor(gt_uvz, device=device)
    # print("shape of pred_uvz:", pred_uvz.shape, type(pred_uvz))
    # print("shape of gt_uvz:", gt_uvz.shape, type(gt_uvz))

    # if dim1 is zero, return 0
    if pred_uvz.shape[1] == 0 or gt_uvz.shape[1] == 0:
        return np.inf
    """
    Compute bidirectional Chamfer Distance.

    Args:
        pred_uvz: (B, N, 3) - predicted UVZ
        gt_uvz: (B, M, 3) - ground truth UVZ

    Returns:
        chamfer_loss: scalar tensor
    """
    # Compute pairwise squared distances
    # pred: (B, N, 1, 3), gt: (B, 1, M, 3)
    pred_expanded = pred_uvz.unsqueeze(2)  # (B, N, 1, 3)
    gt_expanded = gt_uvz.unsqueeze(1)  # (B, 1, M, 3)

    # dist[b, i, j] = ||pred[b,i] - gt[b,j]||^2
    dist = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # (B, N, M)

    # Forward: nearest GT for each predicted point
    try:
        min_dist_pred_to_gt, _ = torch.min(dist, dim=2)  # (B, N)
        forward_loss = min_dist_pred_to_gt.mean()
    except RuntimeError as e:
        print("pred_expanded shape:", pred_expanded.shape)  # [1, 495, 1, 3]
        print(
            "gt_expanded shape:", gt_expanded.shape
        )  # gt_expanded shape: torch.Size([1, 1, 0, 3]) #[1, 1, 0, 3]
        print(f"dist shape: {dist.shape}")  # [1, 495, 0]
        print(f"Error in forward chamfer distance computation: {e}")
        return 0

    # Backward: nearest predicted for each GT point
    min_dist_gt_to_pred, _ = torch.min(dist, dim=1)  # (B, M)
    backward_loss = min_dist_gt_to_pred.mean()

    # Bidirectional Chamfer
    chamfer_loss = forward_loss + backward_loss

    return chamfer_loss


class MANDataset(Dataset):
    def __init__(
        self,
        scene_ids: list = [],
        data_file: str = "man-mini",  # man-full
        device: str = "cpu",
        radar_channel: str = "RADAR_LEFT_FRONT",
        camera_channel: str = "CAMERA_LEFT_FRONT",
        double_flip_images: bool = True,
        coord_only: bool = True,  # if True, only return coordinates of points
        depth_bins=96,  # Number of depth bins
        visualize_uvz=False,
        max_depth: float = 250.0,
        roi=None,  # (min_v, max_v, min_u, max_u) region of interest in pixel coordinates
        scaled_image_size: tuple = None,  # Target size for scaled images
        n_p: int = 0,  # number of point in each radar frame, if  0 no padding, otw, pick random n_p point, if not enough, pad by padding_value
        padding_value=0,
        get_clip=False,
        get_depth=False,
        get_occ_grid=False,
        get_camera=True,
        wan_vae: bool = False,
        wan_vae_checkpoint: str = "wan2_1_832x480.pth",
        wan_spec={"wan_frames":5, "wan_frame_mode":"repeat", "wan_frame_stride":1,"wan_edge_policy":"skip"},  # wan_frame_mode : repeat/center/past/future, wan_edge_policy: skip/pad
        viz_dir: str = "",
        grid_binary_range: str = "0-1",  # "0-1" or "neg1-1"
        keep_frames=0,
        get_bb=False,  # whether to get bounding box of radar points
        point_preset="uniform",  # "uniform" or "original", "l-shape"
        normalize_type="std",  # "std" or "minmax", if "std", return mean, std, if "minmax", return mean(min,max), range
        x_range=None, y_range=None, z_range=None,
        wan_preprocess_dir=None,
        trucksc = None, wan_vae21_object = None
    ):  # load all frames from scene_id, of data_file, particular radar and camera channel
        self.device = device
        self.data_file = data_file
        self.scene_ids = scene_ids
        self.radar_channel = radar_channel
        self.camera_channel = camera_channel
        self.double_flip_images = double_flip_images
        self.coord_only = coord_only  # if True, only return coordinates of points
        self.depth_model = "da3nested-giant-large"  # depth model name
        self.clip_model = "openai/clip-vit-large-patch14"  # clip model name
        self.original_image_size = (943, 1980)  # original image size (height, width)
        self.depth_bins = depth_bins
        self.max_depth = max_depth
        self.visualize_uvz = visualize_uvz
        self.roi = roi  # (min_v, max_v, min_u, max_u)
        self.scaled_image_size = scaled_image_size
        self.clip_features = []  # Store CLIP features for each frame
        self.n_p = n_p
        self.padding_value = padding_value
        self.wan_vae = wan_vae
        self.wan_vae_checkpoint = wan_vae_checkpoint
        self.viz_dir = viz_dir
        self.grid_binary_range = grid_binary_range
        self.keep_frames = keep_frames
        self.get_bb = get_bb
        self.vae21 = wan_vae21_object

        self.get_clip = get_clip
        self.get_depth = get_depth
        self.get_occ_grid = get_occ_grid
        self.get_camera = get_camera
        self.point_preset = point_preset
        self.normalize_type = normalize_type
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.wan_preprocess_dir = os.path.join(wan_preprocess_dir,self.data_file) if wan_preprocess_dir is not None else None
        self.wan_spec = wan_spec
        print("wan process dir:", self.wan_preprocess_dir)
        

        if self.wan_preprocess_dir is not None:
            os.makedirs(self.wan_preprocess_dir, exist_ok=True)

        if self.wan_vae and self.vae21 is None:
            print(f"Loading VAE from checkpoint: {self.wan_vae_checkpoint}")

            from wan.modules.vae2_1 import Wan2_1_VAE
            self.vae21 = Wan2_1_VAE(
                vae_pth=self.wan_vae_checkpoint,
                device=self.device,
            )
            print("VAE loaded successfully.")
        # store mean and std

        if self.data_file == "man-mini":
            print("Loading MAN mini dataset...")
            self.data_root = "/data/palakons/new_dataset/MAN/mini/man-truckscenes"
            trucksc = TruckScenes("v1.0-mini", self.data_root, False) if trucksc is None else trucksc
        elif self.data_file == "man-full":
            print("Loading MAN full dataset...")
            self.data_root = "/data/palakons/new_dataset/MAN/man-truckscenes"
            trucksc = TruckScenes("v1.0-trainval", self.data_root, False) if trucksc is None else trucksc
        else:
            raise ValueError(f"Unknown data_file: {self.data_file}")

        if len(self.scene_ids) == 0:  ##allscenes
            self.scene_ids = list(range(len(trucksc.scene)))
        print("Using all scene_ids:", self.scene_ids)

        if self.get_clip:
            img_files, feature_files = self.gather_missing_clip_features(
                self.scene_ids, trucksc
            )
            print("Processing CLIP features for missing frames...", len(img_files))
            self.process_clip(img_files, feature_files, batch_size=40)
        if self.get_depth:

            img_files, depth_files = self.gather_missing_depth_files(
                self.scene_ids, trucksc
            )
            print("Processing depth images for missing frames...", len(img_files))
            self.inferDA3_depth_image(img_files, depth_files, batch_size=128)

        self.data_bank = []
        image_cache = []  # Cache for loaded images to avoid redundant disk reads

        stredge_str = f"_str{self.wan_spec['wan_frame_stride']}_edge{self.wan_spec['wan_edge_policy']}" if self.wan_spec['wan_frame_mode'] != "repeat" else ""
        latent_id = f"fr{self.wan_spec['wan_frames']}_mode{self.wan_spec['wan_frame_mode']}{stredge_str}"
        for scene_id in tqdm(self.scene_ids, desc="Loading scenes"):
            if len(self.data_bank) >= self.keep_frames and self.keep_frames != 0:
                break
            first_frame_token = trucksc.scene[scene_id]["first_sample_token"]
            temp_token = first_frame_token
            num_frames = 0
            while temp_token != "":
                num_frames += 1
                temp_token = trucksc.get("sample", temp_token)["next"]
            frame_token = first_frame_token
            pbar = tqdm(desc=f"Scene {scene_id}", leave=False, total=num_frames)

            while frame_token != "" and (
                len(self.data_bank) < self.keep_frames or self.keep_frames == 0
            ):
                loaded_data = self.load_data(trucksc, frame_token, self.get_bb)
                # assert loaded_data is not None, f"Failed to load data for scene {scene_id} frame {pbar.n} token {frame_token}"
                if self.wan_vae and loaded_data is not None: #if wan, and cached, and the cahed wan tensor is [0], set loaded_data to None
                    preprocessed_file_path = os.path.join( self.wan_preprocess_dir, loaded_data["camera_file_name"].split("/")[-1].replace(".jpg", f"_{latent_id}.pt") ) 
                    # print(f"path {preprocessed_file_path}")
                    if os.path.exists(preprocessed_file_path):
                        try:
                            loaded_wan = torch.load(preprocessed_file_path)

                            if   torch.load(preprocessed_file_path).shape == torch.Size([1]) and torch.load(preprocessed_file_path)[0] == 0:
                                print(f"Preprocessed VAE latent for scene {scene_id} frame {pbar.n} is a placeholder, skipping loading data for this frame.len data_bank {len(self.data_bank)}")
                                print(f"type dtype shape device of cached preprocessed VAE latent for scene {scene_id} frame {pbar.n}: {type(loaded_wan)}, {loaded_wan.dtype}, {loaded_wan.shape}, {loaded_wan.device}")
                                loaded_data = None
                        except Exception as e:
                            print(f"Error loading preprocessed VAE latent for scene {scene_id} frame {pbar.n}: {e}. Skipping this frame.")
                            os.remove(preprocessed_file_path)  # Remove the corrupted file
                            loaded_wan = None
                        # loaded_wan = torch.load(preprocessed_file_path)
                        # print(f"type dtype shape device of cached preprocessed VAE latent for scene {scene_id} frame {pbar.n}: {type(loaded_wan)}, {loaded_wan.dtype}, {loaded_wan.shape}, {loaded_wan.device}")

                if loaded_data is not None:
                    image_cache.append({
                        "scene_id": scene_id,
                        "frame_index": pbar.n,
                        "frame_token": frame_token,
                        "camera_file_name": loaded_data["camera_file_name"],
                    }) #cache even no points
                    if (
                        self.point_preset == "uniform"
                    ):  # assign grid of x [0-200] y [-100, 100] z [-20, 20] to radar points, and only keep one point in each grid cell, if multiple points in one grid cell, keep the one with highest rcs
                        # mesh grid
                        n_point_axis = int(self.n_p ** (1 / 3)) + 1
                        x = np.linspace(0, 200, n_point_axis)
                        y = np.linspace(-100, 100, n_point_axis)
                        z = np.linspace(-20, 20, n_point_axis)
                        xx, yy, zz = np.meshgrid(x, y, z)
                        # reassign xx,yy zz to  loaded_data["filtered_radar_data"]
                        points = []
                        for i in range(self.n_p):
                            points.append(
                                [
                                    0 + xx.flatten()[i],
                                    -100 + yy.flatten()[i],
                                    -20 + zz.flatten()[i],
                                ]
                            )
                        assert (
                            len(points) == self.n_p
                        ), f"n_p is too large, please reduce n_p to less than {len(points)}"
                        assert len(points) == len(
                            loaded_data["filtered_radar_data"]
                        ), f"n_p is too large, please reduce n_p to less than {len(points)}"
                        loaded_data["filtered_radar_data"] = torch.tensor(
                            points, dtype=torch.float32, device=self.device
                        )
                    elif self.point_preset == "l-shape":  
                        loaded_data["filtered_radar_data"] = make_dummy_lshape(self.n_p, device=self.device)

                    self.data_bank.append(
                        {
                            **loaded_data,
                            "scene_id": scene_id,
                            "frame_index": pbar.n,
                        }
                    )
                else:
                    print(f"Failed to load data for scene {scene_id} frame {pbar.n} token {frame_token}, skipping this frame. len data_bank {len(self.data_bank)}")
                # print("loaded", i, "frame_token", frame_token)
                frame_token = trucksc.get("sample", frame_token)["next"]
                pbar.update(1)
            pbar.close()

        if self.wan_vae :
            assert os.path.exists(self.wan_preprocess_dir), f"WAN preprocess directory {self.wan_preprocess_dir} does not exist. Please run WAN preprocessing first to generate VAE latent files."

            print(f"cached {len(image_cache)} images for VAE preprocessing")
            # for item in tqdm(image_cache, desc="Preprocessing images for VAE"):
            #     print(f"Processing scene {item['scene_id']} frame {item['frame_index']} token {item['frame_token']}")
            missing_image_cache = []
            for item in tqdm(image_cache, desc="Preprocessing images for VAE"):
                preprocessed_file_path = os.path.join( self.wan_preprocess_dir, item["camera_file_name"].split("/")[-1].replace(".jpg", f"_{latent_id}.pt") ) 
                if not os.path.exists(preprocessed_file_path):
                    missing_image_cache.append(item)
            print(f'Already processed {len(image_cache)-len(missing_image_cache)} cached images, {len(missing_image_cache)} images missing for VAE preprocessing')


            wan_latent =self.process_camera_front_to_wan(missing_image_cache) #return list

            #save wan_latent to preprocessed_file_path
            print(f"Saving {len(missing_image_cache)} preprocessed VAE latent files to {self.wan_preprocess_dir}")
            for item, latent in tqdm(zip(missing_image_cache, wan_latent), desc="Saving preprocessed VAE latents", total=len(missing_image_cache)):
                preprocessed_file_path = os.path.join( self.wan_preprocess_dir, item["camera_file_name"].split("/")[-1].replace(".jpg", f"_{latent_id}.pt") ) 
                torch.save(latent.cpu(), preprocessed_file_path)
                # print(f"saved preprocessed VAE latent for scene {item['scene_id']} frame {item['frame_index']} token {item['frame_token']} to {preprocessed_file_path}")

            for idx_db,item in tqdm(enumerate(image_cache), desc="Loading preprocessed VAE latents", total=len(image_cache)):
                preprocessed_file_path = os.path.join( self.wan_preprocess_dir, item["camera_file_name"].split("/")[-1].replace(".jpg", f"_{latent_id}.pt") )  
                if os.path.exists(preprocessed_file_path):
                    wan_vae_latent = torch.load(preprocessed_file_path).to(self.device)
                else:
                    wan_vae_latent = torch.tensor([0], device=self.device)  # placeholder for missing latent

                self.data_bank[idx_db]['wan_vae_latent'] = wan_vae_latent

            print(f'len data_bank {len(self.data_bank)}')
            #clean up if record of databank with wan_vae_latent = tensor ([0]), skip
            print(f'len data_bank before {len(self.data_bank)}')
            self.data_bank = [d for d in self.data_bank if d['wan_vae_latent'].shape != torch.Size([1])]
            print(f'len data_bank after wan_vae_latent (valid edge) {len(self.data_bank)}')
            self.data_bank = [d for d in self.data_bank if d['filtered_radar_data'].shape[0] > 0]
            print(f'len data_bank after filtered_radar_data (no point) {len(self.data_bank)}')


        if False:
            ## Create GIF from PNG frames
            gif_output = f"/home/palakons/from_scratch/man_ds_sample/samples/{self.camera_channel}.gif"
            print(f"Creating GIF: {gif_output}")

            # High quality GIF with palette
            ret = os.system(
                f"ffmpeg -y -framerate 8 -pattern_type glob -i '/home/palakons/from_scratch/man_ds_sample/samples/{self.camera_channel}/*.png' -vf 'fps=8,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5' -loop 0 '{gif_output}'"
            )

            if ret == 0:
                print(f"✓ GIF created successfully: {gif_output}")
                os.system(
                    f"rm /home/palakons/from_scratch/man_ds_sample/samples/{self.camera_channel}/*.png"
                )
            else:
                print(f"✗ GIF creation failed, keeping PNG files")

            # remove all png files
            os.system(
                f"rm /home/palakons/from_scratch/man_ds_sample/samples/{self.camera_channel}/*.png"
            )

        if len(self.data_bank) >0:
            assert self.data_bank[0]["filtered_radar_data"].shape[1] in [
            3,
            7,
        ], "filtered_radar_data should have shape (N, 3) or (N, 7)"

        if len(self.data_bank) == 0:
            print("No valid frames loaded into data_bank. Please check your dataset and preprocessing steps.")
            self.data_mean = torch.zeros(3, device=self.device)
            self.data_std = torch.ones(3, device=self.device)
            self.uvz_mean = torch.zeros(3, device=self.device)
            self.uvz_std = torch.ones(3, device=self.device)
            all_radar_positions = torch.empty((0, 3), device=self.device)
        else:
            all_radar_positions = torch.cat(
            [d["filtered_radar_data"] for d in self.data_bank], dim=0
            )  # filtered_radar_data, d["filtered_radar_data"] will have different number of row (dim 0) for each frame
            print("all_radar_positions", all_radar_positions.shape)  # 1x 16 x1
            # for d in self.data_bank:
            #     print(f"frame_token {d['frame_token']} filtered_radar_data shape {d['filtered_radar_data'].shape}")
                

            # calculate means
            dims = all_radar_positions.shape

            if self.normalize_type == "std":
                self.data_mean = all_radar_positions.mean(axis=0)
                raw_std = all_radar_positions.std(axis=0)
                self.data_std = torch.ones_like(raw_std)

                # 1. XYZ (0:3): Use uniform scale (max of X,Y,Z std) to preserve geometry
                xyz_max_std = raw_std[:3].max()
                self.data_std[:3] = xyz_max_std if xyz_max_std > 0 else 1.0

                # 2. Doppler (3:6): Use uniform scale for velocity vectors if they exist
                if len(raw_std) >= 6:
                    doppler_max_std = raw_std[3:6].max()
                    self.data_std[3:6] = doppler_max_std if doppler_max_std > 0 else 1.0

                # 3. RCS (6): Use individual scale if it exists
                if len(raw_std) >= 7:
                    self.data_std[6] = raw_std[6] if raw_std[6] > 0 else 1.0

            elif self.normalize_type == "minmax":
                min_vals = all_radar_positions.min(axis=0).values
                max_vals = all_radar_positions.max(axis=0).values
                self.data_mean = (min_vals + max_vals) / 2
                raw_range = (max_vals - min_vals) / 2
                self.data_std = torch.ones_like(raw_range)
                xyz_max_range = raw_range[:3].max()
                self.data_std[:3] = xyz_max_range
                if len(raw_range) >= 6:
                    doppler_max_range = raw_range[3:6].max()
                    self.data_std[3:6] = doppler_max_range
                if len(raw_range) >= 7:
                    self.data_std[6] = raw_range[6]
            else:
                raise ValueError(
                    f"Unknown normalize_type: {self.normalize_type}, must be 'std' or 'minmax'"
                )

            # Avoid divide-by-zero
            self.data_std = torch.where(
                self.data_std == 0, torch.ones_like(self.data_std), self.data_std
            )

            print(
                "all_radar_positions data_mean",
                self.data_mean,
            )
            print(
                "all_radar_positions data_std",
                self.data_std,
            )

            # also the same for uvz mena/std
            all_uvz = torch.cat(
                [d["uvz"] for d in self.data_bank], dim=0
            )  # filtered_radar_data, d["filtered_radar_data"] will have different number of row (dim 0) for each frame
            self.uvz_mean = all_uvz.mean(axis=0)
            self.uvz_std = all_uvz.std(axis=0)
            if all_uvz.shape[0] == 1 or (self.uvz_std == 0).any():
                self.uvz_std = torch.where(
                    self.uvz_std == 0, torch.ones_like(self.uvz_std), self.uvz_std
                )

            # all_radar_positions torch.Size([1, 128, 3])
            # all_vrel torch.Size([1, 128, 3])
            # all_rcs torch.Size([1, 128, 1])
            # data_mean tensor([79.6537,  5.0939,  0.3102]) data_std tensor([36.9492, 28.5173,  2.8060])
            # vrel_mean tensor([-2.1093, -0.1453,  0.0066]) vrel_std tensor([0.3315, 0.7359, 0.1138])
            # rcs_mean tensor([-2.3594]) rcs_std tensor([7.6742])

        # Precompute histograms and visualize UVZ comparison
        if self.get_occ_grid:
            self.occupancy_grids = []
            for data in self.data_bank:
                depth_image = data["depth_image"]
                points_uvz = data["uvz"]
                camera_front = data["camera_front"]
                frame_token = data["frame_token"]

                # Scale images if scaled_image_size is provided
                if self.scaled_image_size is not None:
                    original_size = camera_front.shape[1:]  # (H, W)

                    depth_image, _ = self.crop_and_scale_last2(
                        depth_image,
                        self.calc_target_size_hw(
                            depth_image.shape,
                            self.scaled_image_size,
                            self.original_image_size,
                        ),
                        self.roi,
                        self.original_image_size,
                    )
                    camera_front, camera_front_mid = self.crop_and_scale_last2(
                        camera_front,
                        self.scaled_image_size,
                        self.roi,
                        self.original_image_size,
                    )

                    points_uvz[:, 0] *= (
                        self.scaled_image_size[1] / original_size[1]
                    )  # scale u
                    points_uvz[:, 1] *= (
                        self.scaled_image_size[0] / original_size[0]
                    )  # scale v

                    # Reassign scaled images back to the data dictionary
                    data["depth_image"] = depth_image
                    data["camera_front"] = camera_front
                    data["uvz"] = points_uvz

                # Create occupancy grid
                occupancy_grid = self.create_occupancy_grid(
                    depth_bins=self.depth_bins,
                    radar_points_uvz=points_uvz,
                    original_image_size=camera_front.shape[1:],
                    max_depth=self.max_depth,
                    output_range=self.grid_binary_range,
                )
                self.occupancy_grids.append(occupancy_grid)

                if self.visualize_uvz:
                    # Visualize UVZ comparison
                    original_uvz = data["uvz"]
                    # adjsut using original image size vs scaled image size
                    original_uvz[:, 0] = (
                        data["uvz"][:, 0]
                        / self.scaled_image_size[1]
                        * self.original_image_size[1]
                    )
                    original_uvz[:, 1] = (
                        data["uvz"][:, 1]
                        / self.scaled_image_size[0]
                        * self.original_image_size[0]
                    )
                    self.visualize_uvz_comparison(
                        frame_token
                        + f"{self.scaled_image_size[0]}x{self.scaled_image_size[1]}",
                        original_uvz,
                        self.occupancy_grid_to_uvz(
                            occupancy_grid,
                            image_roi=self.original_image_size,
                            max_depth=self.max_depth,
                            threshold=0.5,
                        ),
                        save_di=self.viz_dir,
                    )
    def process_camera_front_to_wan(self, image_cache):
        if len(image_cache) == 0:
            print("No images to process for WAN VAE.")
            return []
        wan_latents = []
        wan_ready_images = []
        w, h = 832, 480

        print("wan_spec:", self.wan_spec) #{'wan_frames': 5, 'wan_frame_mode': 'repeat', 'wan_frame_stride': 1, 'wan_edge_policy': 'skip'}
        for ic in tqdm(image_cache, desc="Processing images for VAE"):
            image_rgb_path = os.path.join(self.data_root, ic["camera_file_name"])
            img, _ = self.crop_and_scale_last2(
                    TF.to_tensor(Image.open(image_rgb_path).convert("RGB"))
                    .to(self.device)
                    .float().sub_(0.5)
                    .div_(0.5),
                    (h, w),
                    self.roi,
                    self.original_image_size,
                )
            wan_ready_images.append(img)
        wan_ready_images = torch.stack(wan_ready_images, dim=0)  # (B, C, H, W) wan_ready_images shape: torch.Size([50, 3, 480, 832])
        # should be [B, C, F, H, W] 
        #duplicate along frame dimension according to wan_spec
        if self.wan_spec["wan_frame_mode"] == "repeat":
            wan_tiled_images = wan_ready_images.unsqueeze_(2).repeat(1, 1, self.wan_spec["wan_frames"], 1, 1)  # (B, C, F, H, W)
        else:
            # wan_tiled_images 

            # need to take into acount of frames, mode, stride, edge policy ,with respect to the scene edge , as we have scene_id, frame_index, to tell the boundary
            # inside image_cache
            # "scene_id": scene_id,
            # "frame_index": pbar.n,
            # "frame_token": frame_token,
            # "camera_front": loaded_data["camera_front"],
            # "camera_file_name": loaded_data["camera_file_name"],

            #group min max indeces of a scene_id
            scene_id_indeces = {}
            for i, item in enumerate(image_cache):
                # print(f"Index {i} scene_id {item['scene_id']} frame_index {item['frame_index']} camera_file_name {item['camera_file_name']}")
                scene_id = item["scene_id"]
                if scene_id not in scene_id_indeces:
                    scene_id_indeces[scene_id] = []
                scene_id_indeces[scene_id].append(i)
            # print("scene_id_indeces:", scene_id_indeces)
            tiling_plan = []
            for ic_idx in range(len(image_cache)):
                item = image_cache[ic_idx]
                indeces_to_tile = []
                
                for f in range(self.wan_spec["wan_frames"]):
                    if self.wan_spec["wan_frame_mode"] == "center":
                        idx = ic_idx + (f - self.wan_spec["wan_frames"] // 2) * self.wan_spec["wan_frame_stride"]
                    elif self.wan_spec["wan_frame_mode"] == "past":
                        idx = ic_idx - (self.wan_spec["wan_frames"] - f - 1) * self.wan_spec["wan_frame_stride"]
                    elif self.wan_spec["wan_frame_mode"] == "future":
                        idx = ic_idx + f * self.wan_spec["wan_frame_stride"]
                    else:
                        raise ValueError(f"Unknown wan_frame_mode: {self.wan_spec['wan_frame_mode']}")
                    
                    # edge policy
                    if self.wan_spec["wan_edge_policy"] == "skip":
                        if idx < min(scene_id_indeces[item["scene_id"]]) or idx >= max(scene_id_indeces[item["scene_id"]]) + 1:
                            continue
                    elif self.wan_spec["wan_edge_policy"] == "pad":
                        # print(f"scene id {item['scene_id']} indeces: {scene_id_indeces[item['scene_id']]} ")
                        if idx < min(scene_id_indeces[item["scene_id"]]):
                            idx = min(scene_id_indeces[item["scene_id"]])
                        elif idx >= max(scene_id_indeces[item["scene_id"]]) + 1:
                            idx = max(scene_id_indeces[item["scene_id"]])
                    else:
                        raise ValueError(f"Unknown wan_edge_policy: {self.wan_spec['wan_edge_policy']}")

                    indeces_to_tile.append(idx)
                tiling_plan.append(indeces_to_tile)
                # print(f"Image {ic_idx} frame_index {item['frame_index']} indeces_to_tile: {indeces_to_tile}")
            print("tiling_plan:", tiling_plan)

            print(f"type dtype shape dev of wan_ready_images: {type(wan_ready_images)}, {wan_ready_images.dtype}, {wan_ready_images.shape}, {wan_ready_images.device}")
            wan_ready_images = wan_ready_images.cpu()
            print(f"type dtype shape dev of wan_ready_images: {type(wan_ready_images)}, {wan_ready_images.dtype}, {wan_ready_images.shape}, {wan_ready_images.device}")
            tiled_list = [
                torch.stack([wan_ready_images[idx] for idx in indeces], dim=1) for indeces in tiling_plan if len(indeces) == self.wan_spec["wan_frames"]
            ]
            ok_indices = [i for i, indeces in enumerate(tiling_plan) if len(indeces) == self.wan_spec["wan_frames"]]
            print(f"tiled_list shape: {[(t.shape) for t in tiled_list]}")  # tiled_list shape: [torch.Size([3, 5, 480, 832]), torch.Size([3, 5, 480, 832]), torch.Size([3, 5, 480, 832]) ...
            print(f"ok_indices: {ok_indices}")  # ok_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
            if len(tiled_list) == 0:
                print("No valid images to tile for WAN VAE. Returning empty list.")
                return []
            wan_tiled_images = torch.stack(tiled_list, dim=0)  
            # (B, C, F, H, W)
            print(f"wan_tiled_images shape after stack: {wan_tiled_images.shape}")  # tiled_list shape after stack: torch.Size([50, 3, 5, 480, 832])
       
        print("wan_tiled_images shape:", wan_tiled_images.shape )
        with torch.no_grad():
            BATCH = 16
            for i in tqdm(range(0, wan_tiled_images.shape[0], BATCH), desc=f"Encoding images batchsize {BATCH} with VAE"):
                batch_imgs = wan_tiled_images[i : i + BATCH].to(self.device)
                # print("batch_imgs shape:", batch_imgs.shape) #batch_imgs shape: torch.Size([2, 3, 5, 480, 832])
                outputs = self.vae21.encode(batch_imgs)  
                # print(f"type of outputs: {type(outputs)}, length of outputs: {len(outputs)}") #<class 'list'>, length of outputs: 32
                #must be list, make tensor
                output_tensor = torch.stack(outputs, dim=0)  # (B, latent_dim)
                # print("latents shape:", output_tensor.shape) #torch.Size([50, 16, 2, 60, 104])
                wan_latents.append(output_tensor.cpu())

        wan_latents = torch.cat(wan_latents, dim=0)  # (num_images, latent_dim)
        print("wan_latents shape:", wan_latents.shape   ) #wan_latents shape: torch.Size([50, 16, 2, 60, 104])
        #convert first dim to list
        if wan_latents.shape[0] != wan_ready_images.shape[0]:
            #use ok_indices to filter wan_latents
            wan_latent_list = [torch.zeros(1) ] * wan_ready_images.shape[0]
            for idx, ok_idx in enumerate(ok_indices):
                wan_latent_list[ok_idx] = wan_latents[idx]
        else:   
            wan_latent_list = [wan_latents[i] for i in range(wan_latents.shape[0])]
        print(f"lenz: {len(wan_latent_list)}")
        return wan_latent_list

    def inverse_SE3(self, T: torch.Tensor) -> torch.Tensor:
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = torch.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    def quat2mat(self, q):  # q in (w, x, y, z)
        # === Convert quaternions to rotation matrices ===
        """
        Convert quaternion to rotation matrix using scipy.
        q: [w, x, y, z] (scalar first convention)
        Returns: 3x3 rotation matrix as torch tensor
        """
        # scipy expects [x, y, z, w] (scalar last), so we need to reorder
        q_scipy = [q[1], q[2], q[3], q[0]]  # [w,x,y,z] -> [x,y,z,w]

        # Create rotation object and get matrix
        rot = R.from_quat(q_scipy)
        rotation_matrix = rot.as_matrix()

        # Convert to torch tensor
        return torch.tensor(rotation_matrix, dtype=torch.float32)

    def to_homogeneous(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert rotation and translation to a 4x4 homogeneous transform."""
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def calib_to_camera_base_MAN_no_crop(
        self,
        cam_calib,
        cam_pose,
        radar_calib,
        radar_pose,
        image_size_hw,
        step=4,
    ):
        """
        image_size_hw: tuple, (height, width), origninal image size
        """

        # === Build individual transforms as 4x4 homogeneous matrices ===
        T_radar2ego = (
            self.to_homogeneous(
                self.quat2mat(radar_calib["rotation"]),
                torch.tensor(radar_calib["translation"]),
            )
            if step >= 1
            else torch.eye(4)
        )
        T_ego2global_r = (
            self.to_homogeneous(
                self.quat2mat(radar_pose["rotation"]),
                torch.tensor(radar_pose["translation"]),
            )
            if step >= 2
            else torch.eye(4)
        )
        T_global2ego_c = (
            self.to_homogeneous(
                self.quat2mat(cam_pose["rotation"]),
                torch.tensor(cam_pose["translation"]),
            )
            if step >= 3
            else torch.eye(4)
        )
        T_ego2cam = (
            self.to_homogeneous(
                self.quat2mat(cam_calib["rotation"]),
                torch.tensor(cam_calib["translation"]),
            )
            if step >= 4
            else torch.eye(4)
        )

        # === Compose full radar → camera transform === but some conventions like Pytorch3D uses right matrix multiplication in computation procedure

        # T_radar2cam = T_radar2ego.sT @ T_ego2global_r.T @ torch.linalg.inv(T_global2ego_c).T @ torch.linalg.inv(T_ego2cam).T

        T_global2ego_c_inv = self.inverse_SE3(T_global2ego_c)
        T_ego2cam_inv = self.inverse_SE3(T_ego2cam)

        T_radar2cam = (
            T_radar2ego.T @ T_ego2global_r.T @ T_global2ego_c_inv.T @ T_ego2cam_inv.T
        )
        T_radar2cam_left_multiply = (
            T_ego2cam_inv @ T_global2ego_c_inv @ T_ego2global_r @ T_radar2ego
        )
        # print(f"T_radar2cam_left_multiply shape ")
        # print(T_radar2cam_left_multiply)
        R = T_radar2cam_left_multiply[:3, :3].unsqueeze(0)  # (1, 3, 3)
        T = (T_radar2cam_left_multiply[3, :3].T).unsqueeze(0)  # (1, 3)

        K = torch.tensor(cam_calib["camera_intrinsic"])

        # Extract focal length and principal point from K
        focal_length = torch.tensor([[K[0, 0], K[1, 1]]])
        principal_point = torch.tensor([[K[0, 2], K[1, 2]]])

        # Create a PerspectiveCameras object
        return SimplePerspectiveCamera(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            image_size=[image_size_hw],
        )

    def inferDA3_depth_image(
        self, image_paths, output_paths, batch_size=4
    ):  # output as h,w  (1 channel)
        if len(image_paths) == 0:
            return
        da3_model = DepthAnything3.from_pretrained(
            f"depth-anything/{self.depth_model}"
        ).to(device=self.device)
        # loop throuugh batch
        for i in trange(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i : i + batch_size]
            batch_output_paths = output_paths[i : i + batch_size]

            prediction = da3_model.inference(
                batch_image_paths,
                export_dir="output",
                export_format="npz",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
            )
            for j in range(len(batch_image_paths)):
                depth_image = (
                    prediction.depth[j, :, :] / 250
                )  # H x W, 1 = 250 meters (max automotive radar range)
                assert (
                    depth_image.max() <= 1.0 and depth_image.min() >= 0.0
                ), f"Depth image values should be in range [0, 1], but got min {depth_image.min()}, max {depth_image.max()}"
                # save image as gray scale
                plt.imsave(
                    batch_output_paths[j], depth_image, cmap="gray", vmin=0, vmax=1
                )
        return

    def visualize_data(
        self,
        camera_front,
        depth_image,
        points_uvz,
        frame_token,
        output_filename,
        rcs,
        radar_view_points,
        camera_view_points,
    ):
        fig, axs = plt.subplots(2, 3, figsize=(23, 10), dpi=75)

        # Show RGB image
        axs[0, 0].imshow(camera_front.permute(1, 2, 0).cpu().numpy())
        axs[0, 0].set_title("RGB Image")
        axs[0, 0].axis("off")

        # Show Depth image
        axs[1, 0].imshow(depth_image.cpu().numpy(), cmap="gray")
        axs[1, 0].set_title("Depth Image")
        axs[1, 0].axis("off")

        # Show projected radar points
        axs[0, 1].imshow(camera_front.permute(1, 2, 0).cpu().numpy())
        # color by the depth'
        colors = points_uvz[:, 2].cpu().numpy()
        axs[0, 1].scatter(
            points_uvz[:, 0].cpu().numpy(),
            points_uvz[:, 1].cpu().numpy(),
            c=colors,
            cmap="rainbow",
            s=5,
        )
        # plt.colorbar(axs[0,1].collections[0], ax=axs[0,1], label='Depth (m)')
        axs[0, 1].set_title("Projected Radar Points")
        axs[0, 1].axis("off")

        # Show projected radar points colored by RCS
        axs[1, 1].imshow(camera_front.permute(1, 2, 0).cpu().numpy())
        colors_rcs = rcs.squeeze().cpu().numpy()
        sc = axs[1, 1].scatter(
            points_uvz[:, 0].cpu().numpy(),
            points_uvz[:, 1].cpu().numpy(),
            c=colors_rcs,
            cmap="rainbow",
            s=5,
        )
        # plt.colorbar(sc, ax=axs[1,1], label='RCS')
        axs[1, 1].set_title("Projected Radar Points Colored by RCS")
        axs[1, 1].axis("off")

        # plot scatter depth vs RCS

        # from k onvert
        # inverse x
        axs[0, 2].scatter(
            radar_view_points[:, 1].cpu().numpy(),
            radar_view_points[:, 0].cpu().numpy(),
            c=colors_rcs,
            s=5,
            cmap="rainbow",
        )
        axs[0, 2].set_xlabel("Y (m)")
        # invert x
        # x between +- 150
        axs[0, 2].set_xlim(-150, 150)
        axs[0, 2].set_ylabel("X (m)")
        # set lim 0 to 250
        axs[0, 2].set_ylim(0, 250)
        axs[0, 2].set_title("Y vs X colored by RCS, Radar" "s view, Top view")
        axs[0, 2].set_aspect("equal", "box")

        # plot camera view
        axs[1, 2].scatter(
            camera_view_points[:, 0].cpu().numpy(),
            camera_view_points[:, 2].cpu().numpy(),
            c=colors_rcs,
            s=5,
            cmap="rainbow",
        )
        axs[1, 2].set_xlabel("X (m)")
        axs[1, 2].set_xlim(-150, 150)
        axs[1, 2].set_ylabel("Z (m)")
        axs[1, 2].set_ylim(0, 250)
        axs[1, 2].set_title("X vs Z colored by RCS, Camera" "s view, Top view")
        axs[1, 2].set_aspect("equal", "box")

        plt.suptitle(
            f'Frame Token: {frame_token} camera: {self.camera_channel.replace("CAMERA_", "")} radar: {self.radar_channel.replace("RADAR_", "")}',
            fontsize=16,
        )
        # tight layout
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

        return

    def calc_target_size_hw(
        self, image_size_hw, scaled_image_size, original_image_size
    ):
        assert scaled_image_size is not None
        # print("Calculating target size hw...")
        # print("image_size_hw:", image_size_hw)
        # print("scaled_image_size:", scaled_image_size)
        # print("original_image_size:", original_image_size)
        return (
            int(image_size_hw[0] / original_image_size[0] * scaled_image_size[0]),
            int(image_size_hw[1] / original_image_size[1] * scaled_image_size[1]),
        )

    def crop_and_scale_last2(
        self,
        x: torch.Tensor,
        target_size_hw,
        roi=None,
        original_size_hw=None,
        mode="bilinear",
        align_corners=False,
    ):
        """
        Crop + resize where spatial dims are always the last 2: (..., H, W).

        Args:
            x: torch.Tensor with shape (..., H, W)
            target_size_hw: (out_h, out_w)
            roi: (min_v, max_v, min_u, max_u) specified in pixels of original_size_hw
                 If None, use full extent.
            original_size_hw: (H0, W0) coordinate system for roi.
                 If None, roi is assumed to be in x's current (H,W) coordinates.
            mode: interpolate mode
            align_corners: for bilinear/bicubic

        Returns:
            torch.Tensor with shape (..., out_h, out_w)
        """
        if x is None:
            return None

        in_h, in_w = x.shape[-2], x.shape[-1]
        # print("x shape:", x.shape)
        # print("roi:", roi)
        # print("original_size_hw:", original_size_hw)
        # x shape: torch.Size([3, 943, 1980])
        # roi: [0, 236, 0, 495]
        # original_size_hw: torch.Size([943, 1980])

        # --- map ROI into x index space ---
        H0, W0 = original_size_hw
        if roi is None:
            roi = (0, H0, 0, W0)    
        
        min_v, max_v, min_u, max_u = roi
        h0 = int(min_v * in_h / H0)
        h1 = int(max_v * in_h / H0)
        w0 = int(min_u * in_w / W0)
        w1 = int(max_u * in_w / W0)

        # clamp
        h0 = max(0, min(in_h, h0))
        h1 = max(0, min(in_h, h1))
        w0 = max(0, min(in_w, w0))
        w1 = max(0, min(in_w, w1))

        # --- crop (view) ---
        # print("Cropping to h:", h0, h1, "w:", w0, w1, "h w", h1 - h0, w1 - w0)
        # Cropping to h: 0 236 w: 0 495 h w 236 495
        x = x[..., h0:h1, w0:w1]
        if target_size_hw is None:
            print("No target size provided, skipping resize.")
            return None, x

        # --- resize via interpolate (expects N,C,H,W) ---
        h, w = x.shape[-2], x.shape[-1]
        lead_shape = x.shape[:-2]
        lead = int(torch.tensor(lead_shape).prod().item()) if len(lead_shape) else 1

        x4 = x.reshape(lead, 1, h, w)  # treat as 1-channel
        y4 = F.interpolate(
            x4,
            size=tuple(target_size_hw),
            mode=mode,
            align_corners=align_corners if mode in ("bilinear", "bicubic") else None,
        )
        out_h, out_w = y4.shape[-2], y4.shape[-1]
        # print("Resized to h w:", out_h, out_w)
        # Resized to h w: 480 832
        return y4.reshape(*lead_shape, out_h, out_w), x

    def scale_image(self, image, target_size, is_rgb=False, roi=None):
        """
        Scale an image to the given target size.

        Args:
            image: torch.Tensor - The image to scale.
            target_size: tuple - The target size (height, width).
            is_rgb: bool - Whether the image is an RGB image.
            roi: tuple - Region of interest (min_v, max_v, min_u, max_u) to crop before scaling.

        Returns:
            torch.Tensor - The scaled image.
        """
        if is_rgb:
            return torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            return (
                torch.nn.functional.interpolate(
                    image.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

    def gather_missing_depth_files(self, scene_ids, trucksc):
        img_files, depth_files = [], []
        for scene_id in tqdm(self.scene_ids):
            first_frame_token = trucksc.scene[scene_id]["first_sample_token"]
            frame_token = first_frame_token
            while frame_token != "":
                frame = trucksc.get("sample", frame_token)
                frame_token = trucksc.get("sample", frame_token)["next"]
                cam = trucksc.get("sample_data", frame["data"][self.camera_channel])
                depth_path = os.path.join(
                    self.data_root,
                    "depth_images",
                    cam["filename"].replace(".jpg", f"_{self.depth_model}.png"),
                )
                if not os.path.exists(depth_path):
                    depth_files.append(depth_path)

                    os.makedirs(os.path.dirname(depth_files[-1]), exist_ok=True)
                    img_files.append(os.path.join(self.data_root, cam["filename"]))
        return img_files, depth_files

    def gather_missing_clip_features(self, scene_ids, trucksc):
        img_files, feature_files = [], []
        for scene_id in tqdm(self.scene_ids):
            first_frame_token = trucksc.scene[scene_id]["first_sample_token"]
            frame_token = first_frame_token
            while frame_token != "":
                frame = trucksc.get("sample", frame_token)
                frame_token = trucksc.get("sample", frame_token)["next"]
                cam = trucksc.get("sample_data", frame["data"][self.camera_channel])
                clip_path = os.path.join(
                    self.data_root,
                    "clip_features",
                    cam["filename"].replace(
                        ".jpg", f"_{self.clip_model.replace('openai/','')}.pth"
                    ),
                )
                if not os.path.exists(clip_path):
                    feature_files.append(clip_path)
                    os.makedirs(os.path.dirname(feature_files[-1]), exist_ok=True)
                    img_files.append(os.path.join(self.data_root, cam["filename"]))
        return img_files, feature_files

    def process_clip(self, img_files, feature_files, batch_size=4):
        def image_to_square_letterbox(image: np.array) -> np.array:
            """

            Args:
                image: np.array - The input image array of shape (H, W, C) with values in [0 - 255] int.

            Returns:
                np.array - The resized square image tensor of shape (desired_size, desired_size, C).
            """

            h, w, _ = image.shape
            assert (
                h <= w
            ), "Image height should be less than or equal to width for letterboxing."
            output_image = np.zeros((w, w, image.shape[2]), dtype=image.dtype)

            # print("image dtype:", image.dtype)  # image dtype: uint8

            # put original image in the center of output_image
            top_letter_box_height = (w - h) // 2
            output_image[top_letter_box_height : top_letter_box_height + h, :, :] = (
                image
            )

            # resize image to desired size while maintaining aspect ratio

            return output_image

        if len(img_files) == 0:
            return
        clip_image_processor = CLIPImageProcessor.from_pretrained(self.clip_model)
        clip_vision_model = CLIPVisionModel.from_pretrained(self.clip_model).to(
            device=self.device
        )

        for i in tqdm(
            range(0, len(img_files), batch_size), desc="Processing CLIP features"
        ):
            batch_img_files = img_files[i : i + batch_size]
            batch_feature_files = feature_files[i : i + batch_size]
            images = []
            for img_file in batch_img_files:
                image = plt.imread(img_file)
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                images.append(image_to_square_letterbox(image))

            with torch.no_grad():
                clip_vision_feature = clip_vision_model(
                    **clip_image_processor(images=images, return_tensors="pt").to(
                        device=self.device
                    )
                )
            # print("clip_vision_feature.last_hidden_state shape:", clip_vision_feature.last_hidden_state.shape)  # ([B, 257, 1024])

            tokens = clip_vision_feature.last_hidden_state
            for j in range(len(batch_img_files)):
                clip_path = batch_feature_files[j]
                torch.save(tokens[j].cpu(), clip_path)  # ([ 257, 1024])

    def _get_cam_bboxes_for_sample(
        self, trucksc, sample, cam_token: str, verbose: bool = False
    ):
        """
        Return a list of (ann_token, bbox) for annotations that belong to the given camera sample_data token.

        Note:
        MAN/nuScenes annotations live at the sample-level; some datasets include a
        'sensor_data_token' on the annotation to indicate which sensor it was created for.
        """
        out = []
        ann_tokens = sample["anns"]
        for ann_token in ann_tokens:
            ann = trucksc.get("sample_annotation", ann_token)

            if verbose:
                print(f"ann_token: {ann_token}")
                for k, v in ann.items():
                    print(f"  {k}: {v}")

            # trucksc.get_box expects an annotation token
            out.append((ann_token, trucksc.get_box(ann_token)))
        return out

    def load_data(self, trucksc, frame_token, get_bb=False):
        time_0 = time.time()

        da3_model = None

        frame = trucksc.get("sample", frame_token)
        cam = trucksc.get("sample_data", frame["data"][self.camera_channel])
        radar = trucksc.get("sample_data", frame["data"][self.radar_channel])

        # -1) get annotated Bouding Box of this nuScene-like frame
        if get_bb:
            cam_token = cam["token"]
            boxes = self._get_cam_bboxes_for_sample(
                trucksc, frame, cam_token, verbose=True
            )  # frame IS my_sample

            print(f"Frame token: {frame_token}")
            print(f"Camera token: {cam_token}")
            print(f"Found {len(boxes)} boxes")
            trucksc.render_sample_data(
                cam_token,
                out_path=f"/data/palakons/man_vaevoxelmetadit/plots/sample_data_rendered_{frame_token}.jpg",
            )
            for i, (ann_token, bbox) in enumerate(boxes):
                ann = trucksc.get("sample_annotation", ann_token)
                if ann["num_radar_pts"] > 0:
                    print(i, ann)
                    # 35
                    # {
                    #     "token": "f23838566e114745a5dcbf64da702b3e",
                    #     "sample_token": "32d2bcf46e734dffb14fe2e0a823d059",
                    #     "instance_token": "fac044e78dbd4ebcaf73dd47f1dfe0b6",
                    #     "visibility_token": "6adc1826065b452195ae6a76317b8ab5",
                    #     "attribute_tokens": ["eefd82c9b2f144e2b1c61a432254cff2"],
                    #     "translation": [
                    #         571683.6013194472,
                    #         5367861.168234938,
                    #         594.2107392291495,
                    #     ],
                    #     "size": [2.624609, 12.548973, 1.650928],
                    #     "rotation": [
                    #         0.39957834306376744,
                    #         -1.0842021724855044e-18,
                    #         0.0,
                    #         0.9166990497182892,
                    #     ],
                    #     "prev": "",
                    #     "next": "fd6e6d6cc299412a90268145d75ed51f",
                    #     "num_lidar_pts": 1584,
                    #     "num_radar_pts": 9,
                    #     "category_name": "vehicle.trailer",
                    # }
                    # get sample token
                    print("sample", trucksc.get("sample", ann["sample_token"]).keys())
                    print("instance", trucksc.get("instance", ann["instance_token"]))
                    print(
                        "visibility", trucksc.get("visibility", ann["visibility_token"])
                    )
                    print(
                        "attribute",
                        trucksc.get(
                            "attribute", ann["attribute_tokens"][0]
                        ),  # only one attribute
                    )
                    # sample dict_keys(['token', 'scene_token', 'timestamp', 'prev', 'next', 'data', 'anns'])
                    # instance {'token': 'fac044e78dbd4ebcaf73dd47f1dfe0b6', 'category_token': 'fec744eb30e5499ab654c8e9576658b0', 'nbr_annotations': 26, 'first_annotation_token': 'f23838566e114745a5dcbf64da702b3e', 'last_annotation_token': 'd7abb64014b945c0801f7772b905f7f9'}
                    # visibility {'token': '6adc1826065b452195ae6a76317b8ab5', 'level': 4, 'description': 'The object is 81% to 100% visible in panoramic view of all cameras.'}
                    # attribute {'token': 'eefd82c9b2f144e2b1c61a432254cff2', 'name': 'vehicle.parked', 'description': 'Vehicle is stationary (usually for longer duration) with no immediate intent to move.'}

            exit()

        # 0) load CLIP
        time_1 = time.time()
        if self.get_clip:
            clip_path = os.path.join(
                self.data_root,
                "clip_features",
                cam["filename"].replace(
                    ".jpg", f"_{self.clip_model.replace('openai/','')}.pth"
                ),
            )
            if not os.path.exists(clip_path):
                raise ValueError(
                    f"CLIP feature file not found: {clip_path}. Please run process_clip to generate CLIP features."
                )
            clip_feature = (
                torch.load(clip_path).squeeze(0).to(device=self.device)
            )  # (feature_dim,)

        # 1) load depth image

        time_2 = time.time()
        if self.get_depth:
            depth_image_path = os.path.join(
                self.data_root,
                "depth_images",
                cam["filename"].replace(".jpg", f"_{self.depth_model}.png"),
            )

            if not os.path.exists(depth_image_path):
                raise ValueError(
                    f"Depth image file not found: {depth_image_path}. Please run inferDA3_depth_image to generate depth images."
                )

            depth_image = plt.imread(depth_image_path)
            if depth_image.max() > 1.0:
                raise ValueError("Depth image should be in range [0, 1]")
                # depth_image = depth_image / 255.0
            depth_image = torch.tensor(depth_image, dtype=torch.float32)[:, :, 0]

        # print("depth_image shape:", depth_image.shape)  # depth_image shape: torch.Size([238, 504])
        time_3 = time.time()
        if self.get_depth:

            if self.double_flip_images:
                # rotate 180 degrees
                depth_image = torch.flip(depth_image, [1])
                depth_image = torch.flip(depth_image, [0])

        # 2) load radar
        # load pcd
        time_4 = time.time()
        radar_pcd_path = os.path.join(self.data_root, radar["filename"])

        radar_data = pypcd4.PointCloud.from_path(radar_pcd_path).pc_data
        points = np.array(
            [
                radar_data["x"],
                radar_data["y"],
                radar_data["z"],
                radar_data["vrel_x"],
                radar_data["vrel_y"],
                radar_data["vrel_z"],
                radar_data["rcs"],
            ],
            dtype=np.float64,
        ).T
        # print("points", points.shape)  # (7, N)
        time_5 = time.time()
        radar_data_all = torch.tensor(points, dtype=torch.float32)
        npoints_original = radar_data.shape[0]

        # 3) load calibration

        # print calibration
        cam_calib = trucksc.get(
            "calibrated_sensor", cam["calibrated_sensor_token"]
        )  # Coordinate system orientation as quaternion: w, x, y, z. / Coordinate system origin in meters: x, y, z.
        radar_calib = trucksc.get("calibrated_sensor", radar["calibrated_sensor_token"])
        cam_pose = trucksc.get("ego_pose", cam["ego_pose_token"])
        radar_pose = trucksc.get("ego_pose", radar["ego_pose_token"])
        time_6 = time.time()
        cam_calib_obj = self.calib_to_camera_base_MAN_no_crop(
            cam_calib,
            cam_pose,
            radar_calib,
            radar_pose,
            self.original_image_size,
        )

        # 4) load camera
        time_7 = time.time()
        if self.get_camera:
            
            image_rgb_path = os.path.join(self.data_root, cam["filename"])

            camera_front = plt.imread(image_rgb_path).transpose(
                2, 0, 1
            )  # PNG images are returned as float arrays (0-1). All other formats are returned as int arrays,
            camera_front = camera_front / 255.0
            camera_front = torch.tensor(camera_front, dtype=torch.float32)
            # print("camera_front: ", camera_front.shape) #camera_front:  torch.Size([3, 618, 2048])
        time_8 = time.time()
        if self.get_camera:
            if self.double_flip_images:
                # rotate 180 degrees
                camera_front = torch.flip(camera_front, [1])
                camera_front = torch.flip(camera_front, [2])

            # upscale depth image to match camera size
            # [238, 504, 4] depth_image



            if False: #test image clone
                print("Testing image clone to device...")
                if self.double_flip_images:
                    # rotate 180 degrees
                    camera_front = torch.flip(camera_front, [1])
                    camera_front = torch.flip(camera_front, [2])
                img1 = camera_front.clone().to(self.device).float() 

                # print shape dtype, device, type, min max
                print("img1 shape:", img1.shape)  # (3, H, W)
                print("img1 dtype:", img1.dtype)  # torch.float32
                print("img1 device:", img1.device)  # cuda:0
                print("img1 type:", type(img1))  # <class 'torch.Tensor'>
                print("img1 min max:", img1.min(), img1.max())  # min max values


                image_rgb_path = os.path.join(self.data_root, cam["filename"])
                img2 = (
                    TF.to_tensor(Image.open(image_rgb_path).convert("RGB"))
                    .to(self.device)
                    .float())
                
                print("img2 shape:", img2.shape)  # (3, 1, H, W)
                print("img2 dtype:", img2.dtype)  # torch.float32
                print("img2 device:", img2.device)  # cuda:0
                print("img2 type:", type(img2))  # <class 'torch.Tensor'>
                print("img2 min max:", img2.min(), img2.max())  # min max values

                # img1 shape: torch.Size([3, 943, 1980])
                # img1 dtype: torch.float32
                # img1 device: cuda:0
                # img1 type: <class 'torch.Tensor'>
                # img1 min max: tensor(0., device='cuda:0') tensor(1., device='cuda:0')

                # img2 shape: torch.Size([3, 943, 1980])
                # img2 dtype: torch.float32
                # img2 device: cuda:0
                # img2 type: <class 'torch.Tensor'>
                # img2 min max: tensor(0., device='cuda:0') tensor(1., device='cuda:0')

                # save the two images to /home/palakons/point_diffusion/output/sample/ as a.jpg and b.jpg
                Image.fromarray(                ((img1.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(                    np.uint8                )            ).save("/home/palakons/point_diffusion/output/sample/a.jpg")
                Image.fromarray(                ((img2.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(                    np.uint8                )            ).save("/home/palakons/point_diffusion/output/sample/b.jpg")

                print(f"all close, max diff: {(img1 - img2).abs().max()}")  # all close, max diff: tensor(0., device='cuda:0')

                exit()
        time_9 = time.time()
        if self.get_depth:
            if (
                depth_image.shape[0] != camera_front.shape[1]
                or depth_image.shape[1] != camera_front.shape[2]
            ):
                depth_image = (
                    torch.nn.functional.interpolate(
                        depth_image.unsqueeze(0).unsqueeze(0),
                        size=(camera_front.shape[1], camera_front.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            if self.original_image_size is None:
                self.original_image_size = camera_front.shape[1:]  # (H, W)
            else:
                assert (
                    self.original_image_size == camera_front.shape[1:]
                ), "All images must have the same size"
        time_10 = time.time()
        # points_in_radar_view = radar_data_all[:, :3].clone().detach().float()
        # rcs = radar_data_all[:, 6:7].clone().detach().float()
        # doppler = radar_data_all[:, 3:6].clone().detach().float()

        # print(f"shapes {radar_data_all.shape} {rcs.shape} {doppler.shape}")
        # radar_data_all_filter_distance = radar_data_all[:, :3].clone().detach().float()
        radar_data_all_filter = radar_data_all.clone().detach().float()
        #masking point with x, y and z in self.x_range, self.y_range and self.z_range
        for i,axs in enumerate([self.x_range, self.y_range, self.z_range]):
            if axs is not None:
                assert len(axs) == 2 and axs[0] < axs[1], "Range should be a tuple of (min, max) with min < max"
                radar_data_all_filter = radar_data_all_filter[
                    (radar_data_all_filter[:, i] >= axs[0])
                    & (radar_data_all_filter[:, i] <= axs[1])
                ]
        # print(f"shapes {radar_data_all_filter_distance.shape} {rcs.shape} {doppler.shape}")
        n_points_after_distance_filter = radar_data_all_filter.shape[0]

        image_coord, camera_view_points = cam_calib_obj.transform_points(
            radar_data_all_filter[:,:3]# x y z
        )
        points_uvz = image_coord[:, :3]  # N x3
        time_11 = time.time()

        if self.wan_vae and False: #original
            # infer

            w, h = 832, 480
            F = 5

            preprocessed_file_path = os.path.join( self.wan_preprocess_dir, cam["filename"].split("/")[-1].replace(".jpg", f"_preprocessed.pt") ) if self.wan_preprocess_dir is not None else None
            # print("preprocessed_file_path:", preprocessed_file_path)
            if self.wan_preprocess_dir is not None and os.path.exists(preprocessed_file_path):
                # print(f"Loading preprocessed image from {preprocessed_file_path}")
                wan_vae_latent = torch.load(preprocessed_file_path).to(self.device)
            else:
                image_rgb_path = os.path.join(self.data_root, cam["filename"])
                img, _ = self.crop_and_scale_last2(
                    TF.to_tensor(Image.open(image_rgb_path).convert("RGB"))
                    .to(self.device)
                    .float()
                    .sub_(0.5)
                    .div_(0.5),
                    (h, w),
                    self.roi,
                    self.original_image_size,
                )
                #save iamge to /home/palakons/point_diffusion/output/sample/a.jpg
                # Image.fromarray(                ((img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(                    np.uint8                )            ).save("/home/palakons/point_diffusion/output/sample/a.jpg") 
                img = img.unsqueeze(1)
                # print("img size after crop and scale:", img.shape)
                # print("minmax img before norm:", img.min(), img.max())
                # video0 = torch.concat(
                #     [
                #         torch.nn.functional.interpolate(
                #             TF.to_tensor(Image.open(image_rgb_path).convert("RGB"))
                #             .sub_(0.5)
                #             .div_(0.5)
                #             .to(self.device)[None]
                #             .cpu(),
                #             size=(h, w),
                #             mode="bicubic",
                #         ).transpose(0, 1),
                #     ]
                #     * F,
                #     dim=1,
                # )  # F,3,h,w
                # video0 = video0.to(self.device)
                video0 = img.repeat(1, F, 1, 1)

                # temp shape: torch.Size([3, 1, 480, 832])
                # min max temp before norm: tensor(0.) tensor(1.)
                # minmax temp norm: tensor(-1.1611) tensor(1.1844)
                # hw: (480, 832)
                # img size after crop and scale: torch.Size([3, 1, 480, 832])
                # minmax img before norm: tensor(-0.4431, device='cuda:0') tensor(0.2272, device='cuda:0')
                # video shape: torch.Size([3, 5, 480, 832])
                # video0 shape before to device: torch.Size([3, 5, 480, 832])

                videos = video0.unsqueeze(0)  # add batch dim

                # save_dir = "/home/palakons/from_scratch/output"
                # # save eachf frame in videos
                # for fi in range(video0.shape[1]):
                #     vframe = video0[:, fi, :, :].unsqueeze(0)
                #     vframe = (vframe * 0.5 + 0.5).clamp(0, 1)
                #     vframe_pil = TF.to_pil_image(vframe.squeeze().cpu())
                #     vframe_pil.save(
                #         os.path.join(
                #             save_dir,
                #             f"{frame_token}_{self.camera_channel.replace('CAMERA_','')}_wanvae_frame{fi}.png",
                #         )
                #     )
                # verified 29-jan-26

                wan_vae_latent = self.vae21.encode(videos.to(self.device))[0]
                if self.wan_preprocess_dir is not None:
                    torch.save(wan_vae_latent.cpu(), preprocessed_file_path)
                    print(f"Saved preprocessed image to {preprocessed_file_path}")
        time_115 = time.time()
        # 5) filter points
        # print("camera_front.shape", camera_front.shape) #[3, 943, 1980])
        mask = (
            (points_uvz[:, 1] >= 0)
            # & (points_uvz[:, 1] < camera_front.shape[1])
            & (points_uvz[:, 1] < 943)
            & (points_uvz[:, 0] >= 0)
            # & (points_uvz[:, 0] < camera_front.shape[2])
            & (points_uvz[:, 0] < 1980)
            & (points_uvz[:, 2] > 0)  # Ensure points are in front of the camera
        )
        # print(f"# points {radar_data_all.shape[0]} -(user-defined ROI)-> {radar_data_all_filter.shape[0]} -(visible in camera view)-> {points_uvz[mask].shape[0]}")
        time_12 = time.time()
        filtered_radar_data = radar_data_all_filter[mask]

        npoints_filtered = filtered_radar_data.shape[0]

        # print(f"shapes {filtered_radar_data.shape} {filtered_rcs.shape} {filtered_doppler.shape}")

        if self.coord_only:
            filtered_radar_data = filtered_radar_data[:, :3]
        # print(f"frame_token: {frame_token}, npoints_original: {npoints_original}, npoints_filtered: {npoints_filtered}")
        # print("filtered_radar_data", filtered_radar_data.shape)  # (N, 3) or (N, 7)

        # if random.random() < 0.01:
        time_13 = time.time()
        if False:
            # mkdir
            output_filename = os.path.join(
                "/home/palakons/from_scratch/man_ds_sample",
                cam["filename"].replace(".jpg", f"_{self.depth_model}_vis.png"),
            )
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            self.visualize_data(
                camera_front,
                depth_image,
                points_uvz[mask],
                frame_token,
                output_filename,
                rcs,
                filtered_radar_data,
                camera_view_points[mask],
            )
            output_filename = os.path.join(
                "/home/palakons/from_scratch/man_ds_sample",
                cam["filename"].replace(".jpg", f"_sdkvis.png"),
            )
            trucksc.render_pointcloud_in_image(
                frame_token,
                pointsensor_channel=self.radar_channel,
                camera_channel=self.camera_channel,
                out_path=output_filename,
                render_intensity=False,
                cmap="viridis",
                verbose=True,
            )
        # print("depth imahe shape", depth_image.shape) #depth imahe shape torch.Size([238, 504, 4])
        time_14 = time.time()
        # 6) keep or padding radar points to fixed number, self.n_p
        output_filtered_radar_data = filtered_radar_data
        output_uvz = points_uvz[mask]
        if output_uvz.shape[0] >0:
            if self.n_p == 0:
                pass
            elif output_uvz.shape[0] < self.n_p:
                # RANDOM RESAMPLING (Replacement)
                # This maintains the true distribution of the radar data instead of clustering at zero
                indices = torch.randint(0, output_uvz.shape[0], (self.n_p,))
                output_uvz = output_uvz[indices]
                output_filtered_radar_data = output_filtered_radar_data[indices]
            elif output_uvz.shape[0] > self.n_p:  # randomly pick self.n_p points
                indices = torch.randperm(output_uvz.shape[0])[: self.n_p]
                output_uvz = output_uvz[indices]
                output_filtered_radar_data = output_filtered_radar_data[indices]
        else:
            # if no points,  output empty tensor with shape (self.n_p, 3) or (self.n_p, 7) depending
            output_filtered_radar_data = output_filtered_radar_data[[]]
        time_15 = time.time()
        # calculate percentage of time spent in each step
        time_diff = {
            "load frame": time_1 - time_0,
            "load CLIP": time_2 - time_1,
            "load depth": time_3 - time_2,
            "load radar": time_5 - time_4,
            "load calib": time_6 - time_5,
            "load camera": time_8 - time_7,
            "upscale depth": time_9 - time_8,
            "calib transform": time_11 - time_10,
            "wan": time_12 - time_11,
            "visualization": time_14 - time_13,
            "final padding": time_15 - time_14,
        }
        total_time = sum(time_diff.values())
        # print time percentages
        # for k, v in time_diff.items():
        #     print(f"{k}: {v/total_time*100:.2f}%")
        
        output = {
            "filtered_radar_data": output_filtered_radar_data,
            "uvz": output_uvz,
            "frame_token": frame_token,
            "npoints_original": npoints_original,
            "npoints_after_distance_filter": n_points_after_distance_filter,
            "npoints_filtered": npoints_filtered,
            "camera_file_name": cam["filename"],
        }
        # if self.wan_vae:
        #     output["wan_vae_latent"] = wan_vae_latent  # add wan vae latent
        if self.get_clip:
            output["clip_feature"] = clip_feature  # add clip feature
        if self.get_camera:
            output["camera_front"] = camera_front  # add camera image
        if self.get_depth:
            output["depth_image"] = depth_image  # add depth image
        return output

    def __len__(self):
        return len(self.data_bank)

    def __getitem__(self, idx):
        if self.get_occ_grid:
            return {**self.data_bank[idx], "occupancy_grid": self.occupancy_grids[idx]}

        return self.data_bank[idx]

    def create_occupancy_grid(
        self,
        depth_bins,
        radar_points_uvz,
        original_image_size=(943, 1980),
        max_depth=250.0,
        output_range="0-1",  # or "neg1-1"
    ):
        """
        Create occupancy grid from radar points.

        Args:
            radar_points_uvz: (N, 3) - [u, v, depth]
            depth_bins: int - number of depth bins
            original_image_size: (H, W) - original image size
            max_depth: float - maximum depth value

        Returns:
            occupancy_grid: (D, H, W) - occupancy grid with values in {0, +1}
        """
        H_orig, W_orig = original_image_size
        D = depth_bins

        # Initialize occupancy grid
        occupancy_grid = torch.zeros(
            (D, H_orig, W_orig), device=radar_points_uvz.device
        )
        if output_range == "neg1-1":
            occupancy_grid -= 1  # initialize to -1
        elif output_range == "0-1":
            pass  # initialize to 0
        else:
            raise ValueError("output_range must be '0-1' or 'neg1-1'")

        if len(radar_points_uvz) == 0:
            return occupancy_grid

        # Convert depth to bin index
        depth_normalized = torch.clamp(radar_points_uvz[:, 2] / max_depth, 0, 1)
        depth_bins = (depth_normalized * (D - 1)).long()  #  floor??

        # Filter valid coordinates
        u = radar_points_uvz[:, 0].long()
        v = radar_points_uvz[:, 1].long()
        valid_mask = (u >= 0) & (u < W_orig) & (v >= 0) & (v < H_orig)
        u = u[valid_mask]
        v = v[valid_mask]
        depth_bins = depth_bins[valid_mask]

        # Fill occupancy grid
        for i in range(len(u)):
            occupancy_grid[depth_bins[i], v[i], u[i]] = 1.0

        return occupancy_grid

    def occupancy_grid_to_uvz(
        self,
        occupancy_grid,
        image_roi=(943, 1980),
        max_depth=250.0,
        threshold=0.5,
    ):
        """
        Convert occupancy grid back to uvz point cloud coordinates.

        Args:
            occupancy_grid: (D, H, W) - occupancy grid
            original_image_size: (H, W) - original image size, if 4 numbers given, it mean the specified ROI (min_v/height, max_v, min_u/width, max_u)
            max_depth: float - maximum depth value
            threshold: float - threshold for occupancy grid values

        Returns:
            uvz_points: (N, 3) - [u, v, depth] point cloud coordinates
        """
        D, H, W = occupancy_grid.shape
        if len(image_roi) == 4:
            min_v, max_v, min_u, max_u = image_roi
        elif len(image_roi) == 2:
            min_v, min_u = 0, 0
            max_v, max_u = image_roi
        else:
            raise ValueError("image_roi must be of length 2 or 4")
        W_orig, H_orig = max_u - min_u, max_v - min_v

        # Threshold occupancy grid
        binary_grid = (occupancy_grid > threshold).float()

        # Find non-zero locations
        nonzero_indices = torch.nonzero(binary_grid, as_tuple=False)
        if len(nonzero_indices) == 0:
            return torch.zeros((0, 3), device=occupancy_grid.device)

        depth_bins_idx = nonzero_indices[:, 0]
        v_idx = nonzero_indices[:, 1]
        u_idx = nonzero_indices[:, 2]

        # Convert back to original coordinates
        u = u_idx.float() / W * W_orig + min_u
        v = v_idx.float() / H * H_orig + min_v
        depth_normalized = depth_bins_idx.float() / (D - 1)
        depth = depth_normalized * max_depth

        return torch.stack([u, v, depth], dim=1)

    def visualize_uvz_comparison(
        self,
        frame_token,
        original_uvz,
        reconstructed_uvz,
        title="",
        save_dir="/home/palakons/from_scratch/man_ds_sample/recon_uvz",
        plotlims=None,  # {"u": (0, 1980), "v": (0, 943), "z": (0, 250)},
        marker_config={
            "original": {"color": "blue", "marker": "o", "size": 10, "alpha": 0.6},
            "reconstructed": {"color": "red", "marker": "x", "size": 10, "alpha": 0.6},
        },
        fig_size=(20, 20),
        device="cpu",
    ):
        """
        Visualize original vs reconstructed UVZ point clouds.

        Args:
            frame_token: str - unique identifier for the data sample
            original_uvz: (N, 3) tensor - [u, v, z]
            reconstructed_uvz: (M, 3) tensor - [u, v, z]
            save_dir: str - directory to save the plot
        """
        os.makedirs(save_dir, exist_ok=True)

        orig_np = (
            original_uvz.cpu().numpy()
            if torch.is_tensor(original_uvz)
            else original_uvz
        )
        recon_np = (
            reconstructed_uvz.cpu().numpy()
            if torch.is_tensor(reconstructed_uvz)
            else reconstructed_uvz
        )

        fig = plt.figure(figsize=fig_size)

        # Original point cloud views
        ax1 = fig.add_subplot(2, 2, 1)

        # plot actualy and predict in the same plot with different colors
        ax1.scatter(
            orig_np[:, 0],
            orig_np[:, 1],
            color=marker_config["original"]["color"],
            label="Original",
            s=marker_config["original"]["size"],
            alpha=marker_config["original"]["alpha"],
            marker=marker_config["original"]["marker"],
            rasterized=True,
        )
        ax1.scatter(
            recon_np[:, 0],
            recon_np[:, 1],
            color=marker_config["reconstructed"]["color"],
            label="Reconstructed",
            s=marker_config["reconstructed"]["size"],
            alpha=marker_config["reconstructed"]["alpha"],
            marker=marker_config["reconstructed"]["marker"],
            rasterized=True,
        )
        ax1.legend(loc="upper right")  # () is slow
        ax1.set_title(
            f"UV View (n=ori{len(orig_np)}, recon{len(recon_np)}) CD:{chamfer_distance(orig_np[None,:,:] , recon_np[None,:,:],device):.4f}"
        )
        ax1.axis("equal")
        if plotlims is not None:
            ax1.set_xlim(plotlims["u"])
            ax1.set_ylim(plotlims["v"])
        ax1.set_xlabel("U-Horizontal Pixel")
        ax1.set_ylabel("V-Vertical Pixel")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(
            orig_np[:, 0],
            orig_np[:, 2],
            color=marker_config["original"]["color"],
            label="Original",
            s=marker_config["original"]["size"],
            alpha=marker_config["original"]["alpha"],
            marker=marker_config["original"]["marker"],
            rasterized=True,
        )
        ax2.scatter(
            recon_np[:, 0],
            recon_np[:, 2],
            color=marker_config["reconstructed"]["color"],
            label="Reconstructed",
            s=marker_config["reconstructed"]["size"],
            alpha=marker_config["reconstructed"]["alpha"],
            marker=marker_config["reconstructed"]["marker"],
            rasterized=True,
        )
        ax2.legend(loc="upper right")
        ax2.set_title("UZ View")
        if plotlims is not None:
            ax2.set_xlim(plotlims["u"])
            ax2.set_ylim(plotlims["z"])
        ax2.set_xlabel("U-Horizontal Pixel")
        ax2.set_ylabel("Z-Depth (m)")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(
            orig_np[:, 2],
            orig_np[:, 1],
            color=marker_config["original"]["color"],
            label="Original",
            s=marker_config["original"]["size"],
            alpha=marker_config["original"]["alpha"],
            marker=marker_config["original"]["marker"],
            rasterized=True,
        )
        ax3.scatter(
            recon_np[:, 2],
            recon_np[:, 1],
            color=marker_config["reconstructed"]["color"],
            label="Reconstructed",
            s=marker_config["reconstructed"]["size"],
            alpha=marker_config["reconstructed"]["alpha"],
            marker=marker_config["reconstructed"]["marker"],
            rasterized=True,
        )
        ax3.legend(loc="upper right")
        ax3.set_title("VZ View")
        if plotlims is not None:
            ax3.set_ylim(plotlims["v"])
            ax3.set_xlim(plotlims["z"])
        ax3.set_ylabel("V-Vertical Pixel")
        ax3.set_xlabel("Z-Depth (m)")

        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        ax4.scatter(
            orig_np[:, 0],
            orig_np[:, 2],
            orig_np[:, 1],
            color=marker_config["original"]["color"],
            label="Original",
            s=marker_config["original"]["size"],
            alpha=marker_config["original"]["alpha"],
            marker=marker_config["original"]["marker"],
            rasterized=True,
        )
        ax4.scatter(
            recon_np[:, 0],
            recon_np[:, 2],
            recon_np[:, 1],
            color=marker_config["reconstructed"]["color"],
            label="Reconstructed",
            s=marker_config["reconstructed"]["size"],
            alpha=marker_config["reconstructed"]["alpha"],
            marker=marker_config["reconstructed"]["marker"],
            rasterized=True,
        )
        ax4.legend(loc="upper right")

        ax4.set_title(f"3D View")
        if plotlims is not None:
            ax4.set_xlim(plotlims["u"])
            ax4.set_zlim(plotlims["v"])
            ax4.set_ylim(plotlims["z"])
        ax4.set_xlabel("U-Horizontal Pixel")
        ax4.set_zlabel("V-Vertical Pixel")
        ax4.set_ylabel("Z-Depth (m)")

        if title:
            # make sure wrapt he line to 80 characters
            wrapped_title = "\n".join(textwrap.wrap(title, width=120))
            plt.suptitle(wrapped_title, fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"compare_uvz_{frame_token}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # print(f"Saved UVZ comparison plot to {save_path}")


def main():
    dataset = MANDataset(
        scene_id=0,
        data_file="man-mini",
        device="cpu",
        radar_channel="RADAR_RIGHT_FRONT",
        camera_channel="CAMERA_RIGHT_FRONT",
        double_flip_images=False,
        coord_only=False,
    )
    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Data sample {i}:")
        print(f"Depth Image Shape: {data[0].shape}")
        print(f"Filtered Radar Data Shape, N: {data[1].shape}")
        print(f"Camera Calibration Object: {data[2]}")
        print(f"Camera Front Image Shape: {data[3].shape}")
        print(f"Frame Token: {data[4]}")
        print(f"Number of Original Points: {data[5]}")
        print(f"Number of Filtered Points: {data[6]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
