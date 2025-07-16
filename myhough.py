import math
import os
import os.path as osp
import pickle
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import pypcd4
import torch
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from PIL import Image
from plotly.subplots import make_subplots
from pyquaternion import Quaternion
from pytorch3d.transforms import axis_angle_to_matrix
from tqdm import tqdm, trange
from truckscenes import TruckScenes
from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import (BoxVisibility, transform_matrix,
                                              view_points)

def hill_climb(acc, start,visited):
    """
    acc: 5D tensor (theta, phi, x, y, z)
    start: starting coordinate as tuple (i, j, x, y, z)
    returns: peak coordinate and value
    """
    current = start
    while True:
        i, j, x, y, z = current
        # Define a small 5D patch around current
        theta_range = slice(max(0, i-1), min(acc.shape[0], i+2))
        phi_range   = slice(max(0, j-1), min(acc.shape[1], j+2))
        x_range     = slice(max(0, x-1), min(acc.shape[2], x+2))
        y_range     = slice(max(0, y-1), min(acc.shape[3], y+2))
        z_range     = slice(max(0, z-1), min(acc.shape[4], z+2))

        patch = acc[theta_range, phi_range, x_range, y_range, z_range]
        #add whole patch to visited
        for idx in np.ndindex(patch.shape):
            visited.add((theta_range.start + idx[0],
                         phi_range.start + idx[1],
                         x_range.start + idx[2],
                         y_range.start + idx[3],
                         z_range.start + idx[4]))
        max_val = patch.max()
        max_idx = torch.argmax(patch)

        # Convert flat index back to 5D offset
        offset = np.unravel_index(max_idx.cpu().numpy(), patch.shape)


        new = (theta_range.start + offset[0],
               phi_range.start + offset[1],
               x_range.start + offset[2],
               y_range.start + offset[3],
               z_range.start + offset[4])

        if new == current:
            return current, acc[current]
        current = new
        if current in visited:
            return current, None

def hough_closest_point_cuda_soft_voting(points_3d, table_sizes, param_ranges, dtype=torch.float32,show_dist=0,output_n_best=1,level=0,max_it = 1000,vote_filter_threshold=0.5):
    """
     --- float32 / 5 parameters / custom binning / return multiple lines

    Vectorized Hough voting for 3D lines parameterized by (theta, phi, qx, qy, qz).
    - points_3d: Nx3 array of 3D points
    - table_sizes: tuple of sizes for each parameter (n_theta, n_phi, n_qx, n_qy, n_qz)
    - param_ranges: tuple of ranges for each parameter
      (theta_range, phi_range, qx_range, qy_range, qz_range)
    - dtype: data type for computations (default: torch.float32)
    - show_dist: probability to show distance from point to line (default: 0, no output)
    - (theta, phi): direction (unit vector, theta angle on x-y plane, phi angle from the plane towards z-axis)
    - (qx, qy, qz): closest point along the line to origin (must satisfy q·d=0, but we vote in all 5D)
    """
    assert points_3d.ndim == 2 and points_3d.shape[1] == 3, f"Input must be a Nx3 array of 3D points. Got shape {points_3d.shape}."
    if points_3d.shape[0] < 2:
        print("At least two points are required to define a line.")
        return [], None

    # Unpack parameter ranges
    theta_range, phi_range, qx_range, qy_range, qz_range = param_ranges
    n_theta, n_phi, n_qx, n_qy, n_qz = table_sizes
    # print("Avail VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    # Create parameter grids
    theta_values = torch.linspace(theta_range[0], theta_range[1], n_theta, device='cuda', dtype=dtype)
    phi_values = torch.linspace(phi_range[0], phi_range[1], n_phi, device='cuda', dtype=dtype)
    qx_values = torch.linspace(qx_range[0], qx_range[1], n_qx, device='cuda', dtype=dtype)
    qy_values = torch.linspace(qy_range[0], qy_range[1], n_qy, device='cuda', dtype=dtype)
    qz_values = torch.linspace(qz_range[0], qz_range[1], n_qz, device='cuda', dtype=dtype)

    # Meshgrid for direction
    theta_grid, phi_grid = torch.meshgrid(theta_values, phi_values, indexing='ij')
    dx = torch.cos(theta_grid) * torch.cos(phi_grid)
    dy = torch.sin(theta_grid) * torch.cos(phi_grid)
    dz = torch.sin(phi_grid)

    dirs = torch.stack([dx, dy, dz], axis=-1).reshape(-1, 3)  # (n_theta*n_phi, 3)
    d_norm = torch.norm(dirs, dim=1, keepdim=True)
    d_unit = dirs / d_norm # (n_theta*n_phi, 3) unit vectors meshed 
    # print(n_theta, n_phi, n_qy, n_qz, n_qx,n_theta*n_phi* n_qy* n_qz* n_qx /1e9*4, "GB of Hough tensor used") 
    # Prepare Hough tensor
    hough_tensor = torch.zeros((n_theta, n_phi, n_qx, n_qy, n_qz), dtype=dtype, device='cuda')
    # For each point, vote for all directions and closest points
    dist_from_point_to_line = []
    for point in tqdm(points_3d,  desc=f" level {level} Processing points"):
        p = torch.tensor(point, device='cuda', dtype=dtype).unsqueeze(0) #current point coordinates
        pxd = torch.cross(p, d_unit, dim=1)# cross product to all directions, thetas and phis
        q = torch.cross(d_unit, pxd, dim=1)  # vector from origin to closest point on the line

        # Soft voting: distribute votes to 8 neighboring bins in qx, qy, qz (trilinear interpolation)
        qx_cont = (q[:, 0] - qx_range[0]) / (qx_range[1] - qx_range[0]) * (n_qx - 1)
        qy_cont = (q[:, 1] - qy_range[0]) / (qy_range[1] - qy_range[0]) * (n_qy - 1)
        qz_cont = (q[:, 2] - qz_range[0]) / (qz_range[1] - qz_range[0]) * (n_qz - 1)

        qx_floor = torch.floor(qx_cont).long()
        qy_floor = torch.floor(qy_cont).long()
        qz_floor = torch.floor(qz_cont).long()
        qx_ceil = torch.clamp(qx_floor + 1, max=n_qx - 1)
        qy_ceil = torch.clamp(qy_floor + 1, max=n_qy - 1)
        qz_ceil = torch.clamp(qz_floor + 1, max=n_qz - 1)

        wx = qx_cont - qx_floor.float()
        wy = qy_cont - qy_floor.float()
        wz = qz_cont - qz_floor.float()

        valid = (qx_floor >= 0) & (qx_floor < n_qx) & (qy_floor >= 0) & (qy_floor < n_qy) & (qz_floor >= 0) & (qz_floor < n_qz)
        theta_idx = torch.arange(n_theta, device='cuda').repeat_interleave(n_phi)
        phi_idx = torch.arange(n_phi, device='cuda').repeat(n_theta)

        if show_dist >  random.random():
            for i in range(len(qx_idx)):
                dir_vec = torch.tensor([ np.cos(theta_values[theta_idx[i]].item()) * np.cos(phi_values[phi_idx[i]].item()),
                                np.sin(theta_values[theta_idx[i]].item()) * np.cos(phi_values[phi_idx[i]].item()),
                                np.sin(phi_values[phi_idx[i]].item())], device='cuda', dtype=dtype)
                dir_vec /= torch.norm(dir_vec)

                pos_org = torch.tensor([qx_values[qx_idx[i]-1].item(), qy_values[qy_idx[i]-1].item(), qz_values[qz_idx[i]-1].item()], device='cuda', dtype=dtype)
                dist_from_point_to_line.append( torch.norm(torch.cross(p -  pos_org, dir_vec.unsqueeze(0)), dim=1).item() )      

        theta_idx = theta_idx[valid]
        phi_idx = phi_idx[valid]
        qx_floor = qx_floor[valid]
        qy_floor = qy_floor[valid]
        qz_floor = qz_floor[valid]
        qx_ceil = qx_ceil[valid]
        qy_ceil = qy_ceil[valid]
        qz_ceil = qz_ceil[valid]
        wx = wx[valid]
        wy = wy[valid]
        wz = wz[valid]

        # Trilinear interpolation: distribute vote to 8 neighboring voxels
        # Corner 000 (floor, floor, floor)
        weight = (1 - wx) * (1 - wy) * (1 - wz)
        hough_tensor[theta_idx, phi_idx, qx_floor, qy_floor, qz_floor] += weight
        # Corner 001 (floor, floor, ceil)
        weight = (1 - wx) * (1 - wy) * wz
        hough_tensor[theta_idx, phi_idx, qx_floor, qy_floor, qz_ceil] += weight
        # Corner 010 (floor, ceil, floor)
        weight = (1 - wx) * wy * (1 - wz)
        hough_tensor[theta_idx, phi_idx, qx_floor, qy_ceil, qz_floor] += weight
        # Corner 011 (floor, ceil, ceil)
        weight = (1 - wx) * wy * wz
        hough_tensor[theta_idx, phi_idx, qx_floor, qy_ceil, qz_ceil] += weight
        # Corner 100 (ceil, floor, floor)
        weight = wx * (1 - wy) * (1 - wz)
        hough_tensor[theta_idx, phi_idx, qx_ceil, qy_floor, qz_floor] += weight
        # Corner 101 (ceil, floor, ceil)
        weight = wx * (1 - wy) * wz
        hough_tensor[theta_idx, phi_idx, qx_ceil, qy_floor, qz_ceil] += weight
        # Corner 110 (ceil, ceil, floor)
        weight = wx * wy * (1 - wz)
        hough_tensor[theta_idx, phi_idx, qx_ceil, qy_ceil, qz_floor] += weight
        # Corner 111 (ceil, ceil, ceil)
        weight = wx * wy * wz
        hough_tensor[theta_idx, phi_idx, qx_ceil, qy_ceil, qz_ceil] += weight
    if show_dist > 0:
        print("dist_from_point_to_line:", sum(dist_from_point_to_line)/len(dist_from_point_to_line))  
        
    max_idx = np.unravel_index(torch.argmax(hough_tensor, axis=None).cpu().numpy(), hough_tensor.shape)
    max_theta = theta_values[max_idx[0]].cpu().numpy()
    max_phi = phi_values[max_idx[1]].cpu().numpy()
    max_qx = qx_values[max_idx[2]].cpu().numpy()
    max_qy = qy_values[max_idx[3]].cpu().numpy()
    max_qz = qz_values[max_idx[4]].cpu().numpy()
    

    peaks = [max_idx]
    print("best peak found:", max_idx, "with value:", hough_tensor[max_idx].item())
    visited = set()
    while len(peaks) < output_n_best-1:  
        if max_it <= 0:
            print("Max iterations reached, stopping search for peaks.")
            break
        max_it -= 1
        start = tuple(torch.randint(s, (1,)).item() for s in hough_tensor.shape)
        if start in visited:
            continue
        visited.add(start)

        peak, value = hill_climb(hough_tensor, start,visited)
        if hough_tensor[start].item()>1:
            print(max_it,"from",start, "value ", hough_tensor[start].item(),"visited",len(visited),"target",hough_tensor[max_idx].item())
        if value is not None and peak not in peaks and  value.item() >= hough_tensor[max_idx].item() * vote_filter_threshold:
            print("Found peak:", peak, "with value:", value.item(),"len peaks:", len(peaks))
            peaks.append(peak)
        else:
            if value is not None and value.item() >1:
                print("not append peak:", peak, "with value:", value.item(), "len peaks:", len(peaks))
            # print("not append peak:", peak, "with value:", value.item() if value is not None else "None","len peaks:", len(peaks))

    output= []
    for pk in peaks:
        #add into output
        theta_idx, phi_idx, qx_idx, qy_idx, qz_idx = pk
        h_result = {'thetaphi':(theta_values[theta_idx].item(), phi_values[phi_idx].item()),
                    "p0":( qx_values[qx_idx].item(), qy_values[qy_idx].item(), qz_values[qz_idx].item()), 
                    'votes': hough_tensor[pk].item(),'resolution':{
            'theta': (theta_range[1] - theta_range[0]) / table_sizes[0],
            'phi': (phi_range[1] - phi_range[0]) / table_sizes[1],
            'qx': (qx_range[1] - qx_range[0]) / table_sizes[2],
            'qy': (qy_range[1] - qy_range[0]) / table_sizes[3],
            'qz': (qz_range[1] - qz_range[0]) / table_sizes[4]
                        }}
        output.append(h_result)


    #free hough_tensor
    del hough_tensor
    torch.cuda.empty_cache()

    if level == 0:
        return output
    for i in range(len(output)):
        #need to narrow down the range of parameters around the found line
        resolutions = [(theta_range[1] - theta_range[0])/ table_sizes[0],
                       (phi_range[1] - phi_range[0]) / table_sizes[1],
                       (qx_range[1] - qx_range[0]) / table_sizes[2],
                       (qy_range[1] - qy_range[0]) / table_sizes[3],
                       (qz_range[1] - qz_range[0]) / table_sizes[4]]
        sub_param_ranges = [(theta_values[max_idx[0]].item() + np.array([-.5, .5]) * table_sizes[0]/2*resolutions[0]).tolist(),
                            (phi_values[max_idx[1]].item() + np.array([-.5, .5]) * table_sizes[1]/2* resolutions[1]).tolist(),
                            (qx_values[max_idx[2]].item() + np.array([-.5, .5]) * table_sizes[2]/2* resolutions[2]).tolist(),
                            (qy_values[max_idx[3]].item() + np.array([-.5, .5]) * table_sizes[3]/2* resolutions[3]).tolist(),
                            (qz_values[max_idx[4]].item() + np.array([-.5, .5]) * table_sizes[4]/2* resolutions[4]).tolist()]

        rehough_output= hough_closest_point_cuda_soft_voting(points_3d, table_sizes, sub_param_ranges, dtype=torch.float32,show_dist=0,output_n_best=1, level=level-1)

        output[i]['thetaphi'] = [ a for a in rehough_output[0]['thetaphi']]
        output[i]['p0'] = [ a for a in rehough_output[0]['p0']]
        output[i]['resolution'] = rehough_output[0]['resolution']
    return output

def hough_closest_point_cuda(points_3d, table_sizes, param_ranges, dtype=torch.float32,show_dist=0,output_n_best=1,level=0):
    """
     --- float32 / 5 parameters / custom binning / return multiple lines

    Vectorized Hough voting for 3D lines parameterized by (theta, phi, qx, qy, qz).
    - points_3d: Nx3 array of 3D points
    - table_sizes: tuple of sizes for each parameter (n_theta, n_phi, n_qx, n_qy, n_qz)
    - param_ranges: tuple of ranges for each parameter
      (theta_range, phi_range, qx_range, qy_range, qz_range)
    - dtype: data type for computations (default: torch.float32)
    - show_dist: probability to show distance from point to line (default: 0, no output)
    - (theta, phi): direction (unit vector, theta angle on x-y plane, phi angle from the plane towards z-axis)
    - (qx, qy, qz): closest point along the line to origin (must satisfy q·d=0, but we vote in all 5D)
    """
    assert points_3d.ndim == 2 and points_3d.shape[1] == 3, f"Input must be a Nx3 array of 3D points. Got shape {points_3d.shape}."
    if points_3d.shape[0] < 2:
        print("At least two points are required to define a line.")
        return [], None

    # Unpack parameter ranges
    theta_range, phi_range, qx_range, qy_range, qz_range = param_ranges
    n_theta, n_phi, n_qx, n_qy, n_qz = table_sizes
    # print("Avail VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    # Create parameter grids
    theta_values = torch.linspace(theta_range[0], theta_range[1], n_theta, device='cuda', dtype=dtype)
    phi_values = torch.linspace(phi_range[0], phi_range[1], n_phi, device='cuda', dtype=dtype)
    qx_values = torch.linspace(qx_range[0], qx_range[1], n_qx, device='cuda', dtype=dtype)
    qy_values = torch.linspace(qy_range[0], qy_range[1], n_qy, device='cuda', dtype=dtype)
    qz_values = torch.linspace(qz_range[0], qz_range[1], n_qz, device='cuda', dtype=dtype)

    # Meshgrid for direction
    theta_grid, phi_grid = torch.meshgrid(theta_values, phi_values, indexing='ij')
    dx = torch.cos(theta_grid) * torch.cos(phi_grid)
    dy = torch.sin(theta_grid) * torch.cos(phi_grid)
    dz = torch.sin(phi_grid)

    dirs = torch.stack([dx, dy, dz], axis=-1).reshape(-1, 3)  # (n_theta*n_phi, 3)
    d_norm = torch.norm(dirs, dim=1, keepdim=True)
    d_unit = dirs / d_norm # (n_theta*n_phi, 3) unit vectors meshed 
    # print(n_theta, n_phi, n_qy, n_qz, n_qx,n_theta*n_phi* n_qy* n_qz* n_qx /1e9*4, "GB of Hough tensor used") 
    # Prepare Hough tensor
    hough_tensor = torch.zeros((n_theta, n_phi, n_qx, n_qy, n_qz), dtype=dtype, device='cuda')
    # For each point, vote for all directions and closest points
    dist_from_point_to_line = []
    for point in tqdm(points_3d,  desc="Processing points"):
        p = torch.tensor(point, device='cuda', dtype=dtype).unsqueeze(0) #current point coordinates
        pxd = torch.cross(p, d_unit, dim=1)# cross product to all directions, thetas and phis
        q = torch.cross(d_unit, pxd, dim=1)  # vector from origin to closest point on the line

        # Bin qy, qz, qx - make tensors contiguous to avoid performance warning
        qx_idx = torch.bucketize(q[:, 0].contiguous(), qx_values)
        qy_idx = torch.bucketize(q[:, 1].contiguous(), qy_values)
        qz_idx = torch.bucketize(q[:, 2].contiguous(), qz_values)

        # Only keep valid indices
        valid = (qy_idx > 0) & (qy_idx < n_qy) & (qz_idx > 0) & (qz_idx < n_qz) & (qx_idx > 0) & (qx_idx < n_qx)
        theta_idx = torch.arange(n_theta, device='cuda').repeat_interleave(n_phi)
        phi_idx = torch.arange(n_phi, device='cuda').repeat(n_theta)

        if show_dist >  random.random():
            for i in range(len(qx_idx)):
                dir_vec = torch.tensor([ np.cos(theta_values[theta_idx[i]].item()) * np.cos(phi_values[phi_idx[i]].item()),
                                np.sin(theta_values[theta_idx[i]].item()) * np.cos(phi_values[phi_idx[i]].item()),
                                np.sin(phi_values[phi_idx[i]].item())], device='cuda', dtype=dtype)
                dir_vec /= torch.norm(dir_vec)

                pos_org = torch.tensor([qx_values[qx_idx[i]-1].item(), qy_values[qy_idx[i]-1].item(), qz_values[qz_idx[i]-1].item()], device='cuda', dtype=dtype)
                dist_from_point_to_line.append( torch.norm(torch.cross(p -  pos_org, dir_vec.unsqueeze(0)), dim=1).item() )      

        theta_idx = theta_idx[valid]
        phi_idx = phi_idx[valid]
        qy_idx = qy_idx[valid] - 1 # because bucktize bins are 1-based
        qz_idx = qz_idx[valid] - 1
        qx_idx = qx_idx[valid] - 1

        hough_tensor[theta_idx, phi_idx, qx_idx, qy_idx, qz_idx] += 1
    if show_dist > 0:
        print("dist_from_point_to_line:", sum(dist_from_point_to_line)/len(dist_from_point_to_line))  
        
    output= []
    max_idx = np.unravel_index(torch.argmax(hough_tensor, axis=None).cpu().numpy(), hough_tensor.shape)
    max_theta = theta_values[max_idx[0]].cpu().numpy()
    max_phi = phi_values[max_idx[1]].cpu().numpy()
    max_qx = qx_values[max_idx[2]].cpu().numpy()
    max_qy = qy_values[max_idx[3]].cpu().numpy()
    max_qz = qz_values[max_idx[4]].cpu().numpy()

    h_result = {'thetaphi':(max_theta.item(), max_phi.item()),"p0":( max_qx.item(), max_qy.item(), max_qz.item()), 
                    'votes': hough_tensor[max_idx].item(),'resolution':{
        'theta': (theta_range[1] - theta_range[0]) / table_sizes[0],
        'phi': (phi_range[1] - phi_range[0]) / table_sizes[1],
        'qx': (qx_range[1] - qx_range[0]) / table_sizes[2],
        'qy': (qy_range[1] - qy_range[0]) / table_sizes[3],
        'qz': (qz_range[1] - qz_range[0]) / table_sizes[4]
                    }}
    output.append( h_result)

    peaks = []
    print("best peak found:", max_idx, "with value:", hough_tensor[max_idx].item())
    max_it = 1000
    while len(peaks) < output_n_best-1:  
        if max_it <= 0:
            print("Max iterations reached, stopping search for peaks.")
            break
        max_it -= 1
        start = tuple(torch.randint(s, (1,)).item() for s in hough_tensor.shape)

        peak, value = hill_climb(hough_tensor, start)
        if peak not in peaks and  value.item() >= hough_tensor[max_idx].item() * 0.5:
            print(max_it,"from",start, "value ", hough_tensor[start].item(),end="")
            print("Found peak:", peak, "with value:", value.item(),"len peaks:", len(peaks))
            peaks.append(peak)
    for pk in peaks:
        #add into output
        theta_idx, phi_idx, qx_idx, qy_idx, qz_idx = pk
        h_result = {'thetaphi':(theta_values[theta_idx].item(), phi_values[phi_idx].item()),
                    "p0":( qx_values[qx_idx].item(), qy_values[qy_idx].item(), qz_values[qz_idx].item()), 
                    'votes': hough_tensor[pk].item(),'resolution':{
            'theta': (theta_range[1] - theta_range[0]) / table_sizes[0],
            'phi': (phi_range[1] - phi_range[0]) / table_sizes[1],
            'qx': (qx_range[1] - qx_range[0]) / table_sizes[2],
            'qy': (qy_range[1] - qy_range[0]) / table_sizes[3],
            'qz': (qz_range[1] - qz_range[0]) / table_sizes[4]
                        }}
        output.append(h_result)


    #free hough_tensor
    del hough_tensor
    torch.cuda.empty_cache()

    if level == 0:
        return output
    for i in range(len(output)):
        #need to narrow down the range of parameters around the found line
        resolutions = [(theta_range[1] - theta_range[0])/ table_sizes[0],
                       (phi_range[1] - phi_range[0]) / table_sizes[1],
                       (qx_range[1] - qx_range[0]) / table_sizes[2],
                       (qy_range[1] - qy_range[0]) / table_sizes[3],
                       (qz_range[1] - qz_range[0]) / table_sizes[4]]
        sub_param_ranges = [(theta_values[max_idx[0]].item() + np.array([-.5, .5]) * table_sizes[0]/2*resolutions[0]).tolist(),
                            (phi_values[max_idx[1]].item() + np.array([-.5, .5]) * table_sizes[1]/2* resolutions[1]).tolist(),
                            (qx_values[max_idx[2]].item() + np.array([-.5, .5]) * table_sizes[2]/2* resolutions[2]).tolist(),
                            (qy_values[max_idx[3]].item() + np.array([-.5, .5]) * table_sizes[3]/2* resolutions[3]).tolist(),
                            (qz_values[max_idx[4]].item() + np.array([-.5, .5]) * table_sizes[4]/2* resolutions[4]).tolist()]

        rehough_output= hough_closest_point_cuda(points_3d, table_sizes, sub_param_ranges, dtype=torch.float32,show_dist=0,output_n_best=1, level=level-1)

        output[i]['thetaphi'] = [ a for a in rehough_output[0]['thetaphi']]
        output[i]['p0'] = [ a for a in rehough_output[0]['p0']]
        output[i]['resolution'] = rehough_output[0]['resolution']
    return output