import math
import os
import os.path as osp
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
from tqdm import tqdm, trange
from truckscenes import TruckScenes
from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import (BoxVisibility, transform_matrix,
                                              view_points)

from pytorch3d.transforms import axis_angle_to_matrix
import pickle,random
from myhough import  hough_closest_point_cuda



def get_camera_token(trucksc,seq_id = 0,i_frame = 5, camera_channel = "CAMERA_LEFT_FRONT"):
    i=0
        
    sample_token = trucksc.scene[seq_id]['first_sample_token']

    while i< i_frame:
        sample_token =  trucksc.get('sample', sample_token)['next']
        i+=1

    sample_record = trucksc.get('sample', sample_token)
    camera_token=sample_record['data'][camera_channel]
    return camera_token

def get_rtk_man_ego(camera_token ):
    """
    Get the rotation, translation, intrinsics and image filename from a camera sample data record.
    The Dataset schema says 'All extrinsic parameters are given with respect to the ego vehicle body frame.'

    Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle. All extrinsic parameters are given with respect to the ego vehicle body frame. All camera images come undistorted and rectified.

    calibrated_sensor {
        "token":                   <str> -- Unique record identifier.
        "sensor_token":            <str> -- Foreign key pointing to the sensor type.
        "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
        "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
        "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
    }
    """
    #1. forward or backward transform? 
    cam = trucksc.get('sample_data', camera_token)
    ego_pose =trucksc.get('ego_pose', cam['ego_pose_token']) #Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system

    cs_record = trucksc.get('calibrated_sensor', cam['calibrated_sensor_token']) #All extrinsic parameters are given with respect to the ego vehicle body frame.
    r = Quaternion(cs_record['rotation']).rotation_matrix # Quaternion() accepts [w, x, y, z] format
    # print("translation:", cs_record['translation']) #row vector
    t = np.array(cs_record['translation']) #

    r_ego = Quaternion(ego_pose['rotation']).rotation_matrix
    t_ego = np.array(ego_pose['translation'])


    k = np.array(cs_record['camera_intrinsic'])
    img_file =cam['filename']
    return {"rotation": r, "translation": t, "intrinsics": k, "image_file": img_file,
            "rotation_ego": r_ego, "translation_ego": t_ego}
def line_to_dir(line):
    theta, phi= line[:2]
    return np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

def line_to_dir_multi(line):
    theta, phi= line['thetaphi']
    return np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])



def count_inpoints(points, hough_line , max_inpoint_radius):
    """
    Count inpoints within a certain radius from the Hough line.
    Args:
        points: Nx3 array of points in 3D space.
        hough_line: A line defined by the Hough transform, typically in the form [theta, phi, y0, z0]. angles in radians.
        max_inpoint_radius: Maximum distance from the line to consider a point as an inpoint.
    """
    #assert points shapes
    assert points.ndim == 2 and points.shape[1] == 3, f"Input must be a Nx3 array of 3D points. Got shape {points.shape}."

    inpoints_idx = []
    theta, phi, y0, z0 = hough_line
    for point in points:
        x, y, z = point

        # Calculate the direction vector of the line
        dx = np.cos(theta) * np.cos(phi)
        dy = np.sin(theta) * np.cos(phi)
        dz = np.sin(phi)

        # Calculate the distance from the point to the line
        # Using the cross-product formula in vector format

        v = np.array([x, y - y0, z - z0])
        d = np.array([dx, dy, dz])
        distance = np.linalg.norm(np.cross(v, d)) / np.linalg.norm(d)

        inpoints_idx.append(distance <= max_inpoint_radius)

    return np.array(inpoints_idx, dtype=np.bool_)

def plot_lines(lines, title="Lines", color='blue'):
    for line in lines:
        theta, phi, y0, z0 = line
        x0 = 0  # Assuming x0 is always 0 for these lines
        direction = line_to_dir(line)
        x1 = x0 + direction[0] * 10  # Extend the line in the direction
        y1 = y0 + direction[1] * 10
        z1 = z0 + direction[2] * 10
        fig.add_trace(go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode='lines',
            line=dict(color=color, width=2),
            name=f'Line: theta={theta*180/np.pi:.2f}, phi={phi*180/np.pi:.2f}, y0={y0:.2f}, z0={z0:.2f}',
        ))
    
def plot_line_from_points(line_points, title="Line from Points", color='blue', mode='lines',size=1, rc=None):
    for i,points in enumerate(line_points):
        if points.shape[0] < 2:
            continue
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        #if marker, amke small marker
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            line=dict(color=color, width=2),
            marker=dict(size=size),  # Uncomment if you want to use markers
            name=title if isinstance(title, str) else title[i]
        ), row=1 if rc is None else rc[0], col=1 if rc is None else rc[1])

def gen_point_from_line(line, length=10):
    theta, phi, y0, z0 = line
    x0 = 0  # Assuming x0 is always 0 for these lines
    direction = line_to_dir(line)
    x1 = x0 + direction[0] * length  # Extend the line in the direction
    y1 = y0 + direction[1] * length
    z1 = z0 + direction[2] * length
    return np.array([[x0, y0, z0],[x1, y1, z1]])

def get_masked_colmap_points(points,colors, mask_out_value,threshold=25, filt_type="out"):
    #unsqueeze colors to 2D
    colors = colors[ np.newaxis,:]
    # print("colors :", colors)
    colors *= 255  # Scale colors to [0, 255] range
    hsv_colors = cv2.cvtColor(colors.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask_out_value_hsv = cv2.cvtColor(np.array([[mask_out_value]], dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0]
    print("mask_out_value:", mask_out_value)
    print("mask_out_value_hsv:", mask_out_value_hsv)
    #mask hue component to not. diff more than 25
    colors = colors.reshape(-1, 3)/255  # Reshape to Nx3
    colors = colors.astype(np.float32)  # Ensure colors are float32 for cv2

    hsv_colors = hsv_colors.reshape(-1, 3)  # Reshape to Nx3

    mask = np.abs(hsv_colors[:, 0]/255 - mask_out_value_hsv[0]/255) <= threshold/255  +1e-6
    print("hsv_colors",hsv_colors)
    print("hsv_colors[:, 0]:", hsv_colors[:, 0])
    print("minus:", np.abs(hsv_colors[:, 0]/255 - mask_out_value_hsv[0]/255) )
    print("thres",threshold/255+1e-6 )
    print("mask shape:", mask.shape,mask)

    #min max h s and v
    print("min max hsv colors:", hsv_colors.min(axis=0), hsv_colors.max(axis=0))
    print("min max colors:", colors.min(axis=0), colors.max(axis=0))

    if filt_type == "out":
        mask = ~mask
    masked_points = points[mask]  # Keep only points not masked out
    masked_colors = colors[mask]  # Keep only colors not masked out

    if False:
        masked_points = points 
        masked_colors = colors//255
        # set masked point to bright red
        masked_colors[mask] = [1, 0, 0]  

    return masked_points, masked_colors

#do hough

def line_to_dir(line):
    theta, phi= line[:2]
    return np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

def plot_lines_5d_matplot_multi(ax,lines, title="Lines", color='blue'):
    
    for i,line in enumerate(lines):
        #max_theta, max_phi, max_qy, max_qz, max_qx
        theta, phi, =line['thetaphi']
        x0,y0, z0 = line['p0']
        direction = line_to_dir_multi(line)
        ax.plot([x0 - direction[0] * 10,  # Extend line in both directions
                 x0 + direction[0] * 10],
                [y0 - direction[1] * 10,  # Extend line in both directions
                 y0 + direction[1] * 10],
                [z0 - direction[2] * 10,  # Extend line in both directions
                 z0 + direction[2] * 10],
                color=color if isinstance(color,str) else color[i],
                linewidth=2,
                label=f'{title}: {theta*180/np.pi:.2f},{phi*180/np.pi:.2f}({x0:.2f},{y0:.2f},{z0:.2f})/{line["votes"]:.2f}')
        #plot point p0
        ax.scatter(x0, y0, z0, color=color if isinstance(color,str) else color[i], s=50, 
                   marker='X')

def plot_lines_5d_matplot(ax,lines, title="Lines", color='blue'):
    for i,line in enumerate(lines):
        #max_theta, max_phi, max_qy, max_qz, max_qx
        theta, phi, x0,y0, z0 = line
        direction = line_to_dir(line)
        ax.plot([x0 - direction[0] * 10,  # Extend line in both directions
                 x0 + direction[0] * 10],
                [y0 - direction[1] * 10,  # Extend line in both directions
                 y0 + direction[1] * 10],
                [z0 - direction[2] * 10,  # Extend line in both directions
                 z0 + direction[2] * 10],
                color=color if isinstance(color,str) else color[i],
                linewidth=2,
                label=f'{title}: theta={theta*180/np.pi:.2f}, phi={phi*180/np.pi:.2f}, x0={x0:.2f} y0={y0:.2f}, z0={z0:.2f}')

def plot_lines_5d(lines, title="Lines", color='blue',rc=None):
    for i,line in enumerate(lines):
        #max_theta, max_phi, max_qy, max_qz, max_qx
        theta, phi, x0,y0, z0 = line
        direction = line_to_dir(line)
        fig.add_trace(go.Scatter3d(
            x=[x0 - direction[0] * 10,  # Extend line in both directions
               x0 + direction[0] * 10],
            y=[y0 - direction[1] * 10,  # Extend line in both directions
               y0 + direction[1] * 10],
            z=[z0 - direction[2] * 10,  # Extend line in both directions
               z0 + direction[2] * 10],
            mode='lines',
            line=dict(color=color if isinstance(color,str) else color[i] , width=2),
            name=f'{title}: theta={theta*180/np.pi:.2f}, phi={phi*180/np.pi:.2f}, x0={x0:.2f} y0={y0:.2f}, z0={z0:.2f}',
        ), row=1 if rc is None else rc[0], col=1 if rc is None else rc[1])

def voxel_downsample_index(points, voxel_size):
    coords = (points / voxel_size).astype(int)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return unique_idx

# mask_out_value =[80,110,120]
mask_out_value =[138,125,97]
mask_in_value=[58,53,25]

colamp_output_file =f'/data/palakons/colmap_data/seq_0/dense/fused.ply'
if os.path.exists(colamp_output_file):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(colamp_output_file)
    o3d.visualization.draw_geometries([pcd])
    #plot matplotlib
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print("points.shape:", points.shape)


#random mask for 10% of points
mask = np.random.rand(points.shape[0]) <= .5
points = points[mask]
colors = colors[mask]
# print("colors after masking:", colors)
print("points shape:", points.shape)
mask_thresh=.1
# masked_points, masked_colors = get_masked_colmap_points(points,colors, mask_in_value,mask_thresh,filt_type="in")
masked_colors = colors
masked_points = points

print("masked_points shape:", masked_points.shape)
print("masked_colors shape:", masked_colors.shape)
#filter out z<0
# masked_colors = masked_colors[masked_points[:, 2] > 0]
# masked_points = masked_points[masked_points[:, 2] > 0]
# #x<4
# masked_colors = masked_colors[masked_points[:, 0] > 4]
# masked_points = masked_points[masked_points[:, 0] > 4]
print("masked_points shape:", masked_points.shape)
print("masked_colors shape:", masked_colors.shape)


if False:
    # set np seed
    np.random.seed(42)
    random.seed(42)  

    alpha = np.random.rand(5000) * 0.5 + 0.0  # Random alpha values between 0.5 and 1.0
    fix_test_line  = [45/180*np.pi, 45/180*np.pi, 0, 0, 0] #theta, phi, x0,y0, z0
    fix_test_line  = [random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] #theta, phi, x0,y0, z0
    _, _, x0,y0, z0 = fix_test_line
    direction = line_to_dir(fix_test_line)
    p0 =np.array([x0, y0, z0])

    rand_point = np.array([x0, y0, z0]) + direction * alpha[:, np.newaxis]*10 
    #add white noise
    masked_points = rand_point + np.random.normal(0, 0.1, rand_point.shape)  # Adding some noise to the points
    print("masked_points shape after adding noise:", masked_points.shape)
    #color randomly
    masked_colors = np.random.rand(masked_points.shape[0], 3)  #
    masked_colors = masked_colors   # Scale colors to [0, 255] range
    masked_colors = masked_colors.reshape(-1, 3)  # Reshape to Nx


# param_ranges = [[0, np.pi], [0, np.pi], [2.5, 3.5], [-1.5, -.5], [2.5, 3.5]]
# param_ranges = [[0, np.pi], [0, np.pi], [-5, 5], [-5, 5], [-5, 5]]
param_ranges = [[0, np.pi], [0, np.pi], [-20, 20], [-20, 20], [-20, 20]]
# param_ranges = [[np.pi/4, np.pi/2], [3/4*np.pi, 5/4*np.pi], [5, 15], [-5, 0], [5, 15]]
h_level =0
output_n_best = 10
vote_filter_threshold=.1
vote_method='hard'
smooth_kernel = 9
point_prune_voxel_size = .5  # Voxel size for pruning points
# availble_vram = torch.cuda.get_device_properties(0).total_memory / 1e6*.9 #MiB
availble_vram = 10240*.9 #MiB
# for rat in np.linspace(3, 55,10 , endpoint=False):
if point_prune_voxel_size > 0:
    downsample_index =voxel_downsample_index(masked_points, point_prune_voxel_size)
    print("downsample_index shape:", downsample_index.shape)
    masked_points = masked_points[downsample_index, :]
    masked_colors = masked_colors[downsample_index, :]
for rat in [65]:#np.linspace(131, 645,8, endpoint=False):
    time0 =     time.time()
    # rat**2*num_dist_var**3 =availble_vram
    num_dist_var = ((availble_vram * 1024 * 1024 / 4) / rat**2)**(1/3)

    table_sizes = np.array([int(rat),int(rat),int(num_dist_var),int(num_dist_var),int(num_dist_var)])
    print("table_sizes:", table_sizes,table_sizes[0]* table_sizes[1]* table_sizes[2]* table_sizes[3]* table_sizes[4] * 4 / 1024 / 1024, "MiB")
    print(f"resolution: {(param_ranges[0][1] - param_ranges[0][0]) / table_sizes[0]*180/np.pi:.2f} deg, {(param_ranges[1][1] - param_ranges[1][0]) / table_sizes[1]*180/np.pi:.2f} deg, "
          f"{(param_ranges[2][1] - param_ranges[2][0]) / table_sizes[2]:.2f}, "
          f"{(param_ranges[3][1] - param_ranges[3][0]) / table_sizes[3]:.2f}, "
          f"{(param_ranges[4][1] - param_ranges[4][0]) / table_sizes[4]:.2f}")
    
    h_line  = hough_closest_point_cuda(masked_points, table_sizes, param_ranges, dtype=torch.float32, show_dist=0.00, output_n_best=output_n_best, level=h_level, max_it=10000, vote_filter_threshold=vote_filter_threshold, vote_method=vote_method, smooth_kernel=smooth_kernel)
    time1 = time.time()
    # print(f"h_line: {len(h_line)} {h_line} ")
    for i in range(len(h_line)):
        print(f"Line {i+1}: theta={h_line[i]['thetaphi'][0]*180/np.pi:.2f}, phi={h_line[i]['thetaphi'][1]*180/np.pi:.2f}, "
              f"p0=({h_line[i]['p0'][0]:.2f}, {h_line[i]['p0'][1]:.2f}, {h_line[i]['p0'][2]:.2f}), "
              f"votes={h_line[i]['votes']:.2f} resolution={h_line[i]['resolution']}")
    #savet hline to file
    if False:
        h_line_file = "/data/palakons/colmap_data/seq_0/"+ f"double_hough_line_colmap_{rat:04d}_{len(masked_points):05d}.json"
        with open(h_line_file, 'w') as f:
            # print("saving",[h_line[j]['thetaphi']+ h_line[j]['p0'] for j in range(len(h_line))])
            import json
            json.dump({
                "hough_line": [h_line[j]['thetaphi']+ h_line[j]['p0'] for j in range(len(h_line))],
                "table_sizes": table_sizes.tolist(),
                "param_ranges": param_ranges,
                "masked_points": masked_points.tolist(),
                "masked_colors": masked_colors.tolist(),
            }, f, indent=4)

    print("now plotting the lines")

    #lets have 4 plot, in 3 differen viewing plans and perspectives


    fig_png_fname = "/data/palakons/colmap_data/seq_0/"+ f"hough_{rat:04d}_{len(masked_points):05d}_{mask_thresh:03.6f}_hlevel_{h_level}_nbest_{output_n_best}_vth_{vote_filter_threshold:.2f}_{smooth_kernel}_{vote_method}_{point_prune_voxel_size:.2f}.png"

    fig,ax = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(16, 16))

    fig.suptitle(f"Hough Lines smooth{smooth_kernel}_vote{vote_method}_prune{point_prune_voxel_size:.2f} (rat={rat}, points={len(masked_points)} {h_line[0]['thetaphi'][0]*180/np.pi:.2f}, {h_line[0]['thetaphi'][1]*180/np.pi:.2f} deg, ({h_line[0]['p0'][0]:.2f}, {h_line[0]['p0'][1]:.2f}, {h_line[0]['p0'][2]:.2f}) angle resolution: {h_line[0]['resolution']['theta']*180/np.pi:.4f},{h_line[0]['resolution']['phi']*180/np.pi:.4f}, ({h_line[0]['resolution']['qx']:.4f}, {h_line[0]['resolution']['qy']:.4f}, {h_line[0]['resolution']['qz']:.4f}) time:{time1-time0:.2f} sec)")



    name =  ["perspective","xy plane","xz plane","yz plane"]
    views_ele_azm = [
        (30, -45),  # Perspective view
        (90, -90),   # Top-down view (XY plane)
        (0, -90),    # Side view (XZ plane)
        (0, 0)    # Side view (YZ plane)
    ]
    d_plot=20
    # Get n colors from matplotlib colormap
    cmap = plt.cm.gist_rainbow  # You can use any colormap: viridis, plasma, tab10, Set1, etc.
    color_list = [cmap(i / max(1, len(h_line) - 1)) for i in range(len(h_line))]
    for i in range(2):
        for j in range(2):
            ax[i,j].scatter(masked_points[:, 0], masked_points[:, 1], masked_points[:, 2], c=  masked_colors, s=2, alpha=0.5)
            ax[i, j].set_title(name[i*2+j])
            ax[i, j].set_xlabel('X')
            ax[i, j].set_ylabel('Y')
            ax[i, j].set_zlabel('Z')
            ax[i, j].set_xlim([-d_plot, d_plot])
            ax[i, j].set_ylim([-d_plot, d_plot])
            ax[i, j].set_zlim([-d_plot, d_plot])    
            ax[i, j].view_init(elev=views_ele_azm[i*2+j][0], azim=views_ele_azm[i*2+j][1])  # Set viewing angle
            plot_lines_5d_matplot_multi(ax[i,j], h_line, title="", color= color_list)
            # Add a legend to the first subplot, multiple rows of legend
    ax[0,0].legend(loc='upper right' , fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

            
    #tight
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    fig.savefig(fig_png_fname, dpi=300)
    print(f"Figure saved to {fig_png_fname}")