from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
import time
import time_profiler as tp

from .feature_model import FeatureModel
from .model_utils import compute_distance_transform
import logging
log_filepath = "/home/palakons/from_scratch/app_2.csv"
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_filepath, mode='a')
formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class PointCloudProjectionModel(ModelMixin):
    
    def __init__(
        self,
        image_size: int,
        image_feature_model: str,
        use_local_colors: bool = True,
        use_local_features: bool = True,
        use_global_features: bool = False,
        use_mask: bool = True,
        use_distance_transform: bool = True,
        predict_shape: bool = True,
        predict_color: bool = False,
        process_color: bool = False,
        image_color_channels: int = 3,  # for the input image, not the points
        color_channels: int = 3,  # for the points, not the input image
        colors_mean: float = 0.5,
        colors_std: float = 0.5,
        scale_factor: float = 1.0,
        # Rasterization settings
        raster_point_radius: float = 0.0075,  # point size
        raster_points_per_pixel: int = 1,  # a single point per pixel, for now
        bin_size: int = 0,
    ):
        super().__init__()
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.use_local_colors = use_local_colors
        self.use_local_features = use_local_features
        self.use_global_features = use_global_features
        self.use_mask = use_mask
        self.use_distance_transform = use_distance_transform
        self.predict_shape = predict_shape
        self.predict_color = predict_color
        self.process_color = process_color
        self.image_color_channels = image_color_channels
        self.color_channels = color_channels
        self.colors_mean = colors_mean
        self.colors_std = colors_std

        # Types of conditioning that are used
        self.use_local_conditioning = self.use_local_colors or self.use_local_features or self.use_mask
        self.use_global_conditioning = self.use_global_features

        # Create feature model
        self.feature_model = FeatureModel(image_size, image_feature_model)

        # Input size
        self.in_channels = 3  # 3 for 3D point positions
        if self.use_local_colors:
            self.in_channels += self.image_color_channels
        if self.use_local_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_global_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_mask:
            self.in_channels += 2 if self.use_distance_transform else 1
        if self.process_color:  # whether color should be an input
            self.in_channels += self.color_channels
        
        # Output size
        self.out_channels = 0
        if self.predict_shape:
            self.out_channels += 3
        if self.predict_color:
            self.out_channels += self.color_channels

        # Save rasterization settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(image_size, image_size),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )
    
    def denormalize(self, x: Tensor, /, clamp: bool = True):
        x = x * self.colors_std + self.colors_mean
        return torch.clamp(x, 0, 1) if clamp else x

    def normalize(self, x: Tensor, /):
        x = (x - self.colors_mean) / self.colors_std
        return x

    def get_global_conditioning(self, image_rgb: Tensor):
        global_conditioning = []
        if self.use_global_features:
            global_conditioning.append(self.feature_model(image_rgb, 
                return_cls_token_only=True))  # (B, D)
        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning
    
    # from functools import lru_cache
    # @lru_cache(maxsize=None)
    def get_local_conditioning(self, image_rgb: Tensor, mask: Tensor):
        local_conditioning = []
        if self.use_local_colors:
            local_conditioning.append(self.normalize(image_rgb))
        if self.use_local_features:
            local_conditioning.append(self.feature_model(image_rgb))
        if self.use_mask:
            local_conditioning.append(mask.float())
        if self.use_distance_transform:
            if not self.use_mask:
                raise ValueError('No mask for distance transform?')
            if mask.is_floating_point():
                mask = mask > 0.5
            local_conditioning.append(compute_distance_transform(mask))
        local_conditioning = torch.cat(local_conditioning, dim=1)  # (B, D_cond, H, W)
        return local_conditioning

    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection(
        self, points: Tensor, camera: CamerasBase, local_features: Tensor,
    ):        
        tp.tic("spro")

        tp.tic("spro_setup1")
        B, C, H, W, device = *local_features.shape, local_features.device
        tp.toca("spro_setup1")
        tp.tic("spro_setup2")
        R = self.raster_settings.points_per_pixel
        tp.toca("spro_setup2")
        tp.tic("spro_setup3")
        N = points.shape[1]
        tp.toca("spro_setup3")
        
        # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
        tp.tic("spro_setup4")
        camera = camera.clone()
        tp.toca("spro_setup4")
        tp.tic("spro_setup5")
        camera.T = camera.T * self.scale_factor
        tp.toca("spro_setup5")

        tp.tic("spro_raster")
        # Create rasterizer
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings)
        tp.toca("spro_raster")

        # Associate points with features via rasterization
        tp.tic("spro_rasterize")
        fragments = rasterizer(Pointclouds(points))  # (B, H, W, R)
        tp.toca("spro_rasterize")

        tp.tic("spro_vis")
        fragments_idx: Tensor = fragments.idx.long()
        visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
        points_to_visible_pixels = fragments_idx[visible_pixels]
        tp.toca("spro_vis")

        tp.tic("spro_permute")
        # Reshape local features to (B, H, W, R, C)
        local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)
        tp.toca("spro_permute")
        
        # Get local features corresponding to visible points
        tp.tic("spro_zero")
        local_features_proj = torch.zeros(B * N, C, device=device)
        tp.toca("spro_zero")
        tp.tic("spro_assign")
        local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
        tp.toca("spro_assign")

        tp.tic("spro_reshape")
        local_features_proj = local_features_proj.reshape(B, N, C)
        tp.toca("spro_reshape")
        tp.toca("spro")
        
        # import pdb; pdb.set_trace()
        return local_features_proj
    
    def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
        """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
        points = pc.points_padded() * (self.scale_factor if scale else 1)
        if self.predict_color and pc.features_padded() is not None:
            colors = self.normalize(pc.features_padded()) if normalize else pc.features_padded()
            return torch.cat((points, colors), dim=2)
        else:
            return points
    
    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            assert x.shape[2] == 3
            return Pointclouds(points=points)

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        t: Optional[Tensor],
    ):
        import time,os
        """ Extracts local features from the input image and projects them onto the points 
            in the point cloud to obtain the input to the model. Then extracts global 
            features, replicates them across points, and concats them to the input."""
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations (and colors if and only if predicting color)
        x_t_input = [x_t]

        st_time =time.time()
        # Local conditioning
        if self.use_local_conditioning:

            # Get local features and check that they are the same size as the input image


            local_features = self.get_local_conditioning(image_rgb=image_rgb, mask=mask)
            if local_features.shape[-2:] != image_rgb.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {image_rgb.shape=}')
            
            logger.info(f"{os.getpid()}, get_input_with_conditioning/1_get_local_conditioning, {time.time()-st_time}")
            st_time =time.time()
            
            # Project local features. Here that we only need the point locations, not colors
            local_features_proj = self.surface_projection(points=x_t[:, :, :3],
                camera=camera, local_features=local_features)  # (B, N, D_local)

            x_t_input.append(local_features_proj)

            logger.info(f"{os.getpid()}, get_input_with_conditioning/2_surface_projection, {time.time()-st_time}")
            st_time =time.time()
        # Global conditioning
        if self.use_global_conditioning:

            # Get and repeat global features
            global_features = self.get_global_conditioning(image_rgb=image_rgb)  # (B, D_global)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, D_global, N)

            x_t_input.append(global_features)

        logger.info(f"{os.getpid()}, get_input_with_conditioning/3_use_global_conditioning, {time.time()-st_time}")
        st_time =time.time()
        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        logger.info(f"{os.getpid()}, get_input_with_conditioning/4_torch.cat, {time.time()-st_time}")
        return x_t_input

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()
