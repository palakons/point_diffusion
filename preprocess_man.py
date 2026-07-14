import sys

import torch
import os
import numpy as np
import pickle
from io_dataset import normalize_data,make_man_pc
# from fitone_dfs_cond_log import parse_args
from tqdm import trange
from truckscenes import TruckScenes

sys.path.insert(0, "/home/palakons/Wan2.2")
from wan.modules.vae2_1 import Wan2_1_VAE

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a point cloud diffusion model on a single shape."
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of points in the point cloud"
    )
    parser.add_argument("--B", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--n_scene",
        type=int,
        default=1,
        help="Number of scenes to use from the dataset (for multi-scene datasets)",
    )
    parser.add_argument(
        "--T", type=int, default=1000, help="Number of diffusion steps during training"
    )
    parser.add_argument(
        "--T_infer",
        type=int,
        default=50,
        help="Number of diffusion steps during inference",
    )
    parser.add_argument(
        "--ddpm_iteration",
        type=int,
        default=10000 * 3,
        help="Number of training iteration",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for the output GIF"
    )
    parser.add_argument(
        "--cond_mode",
        type=str,
        default="pdnorm_only",
        help="Time conditioning mode for the model: 'pdnorm_only', 'feat_add', 'hybrid', 'feat_concat'",
    )
    parser.add_argument(
        "--shape_name",
        type=str,
        default="realman",
        help="Shape to train on: 'realman' or 'various', 'realman_dense', 'man_proper_split'"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Whether to run training or just inference test. Options: 'train', 'test'",
    )
    parser.add_argument(
        "--cond_method",
        type=str,
        default="scene_id",
        help="Conditioning method for multi-scene training: 'scene_id' (simple learnable embedding), 'wan' (use Wan's VAE latent)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="man-mini",
        help="Data file to use for MANDataset when shape_name is 'realman'. Options: 'man-mini', 'man-full', or path to custom data file compatible with MANDataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PTv3Dnsr",
        help="Model architecture to use: 'SetTxDnsr', 'PTv3Dnsr'",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save and load checkpoints",
    )
    parser.add_argument(
        "--train_rcs_doppler",
        action="store_true",
        help="Whether to train on RCS and Doppler features",
    )

    parser.add_argument(
        "--loss_weight_position",
        type=float,
        default=1.0,
        help="Loss weight for position feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_doppler",
        type=float,
        default=1.0,
        help="Loss weight for Doppler feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--loss_weight_rcs",
        type=float,
        default=1.0,
        help="Loss weight for RCS feature, only used if --train_rcs_doppler is set",
    )
    parser.add_argument(
        "--num_train_log",
        type=int,
        default=100,
        help="Logging frequency (in steps) during training",
    )
    parser.add_argument(
        "--num_checkpoints_save",
        type=int,
        default=1000,
        help="Checkpoint saving frequency (in steps) during training",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=1000,
        help="Number of evaluation points to use during training",
    )
    parser.add_argument(
        "--lambda_cd",
        type=float,
        default=0.0,
        help="Weight for the CD loss term",
    )
    parser.add_argument(
        "--cd_mode",
        type=str,
        default="xyz_attr",
        help="Chamfer Distance mode: 'xyz_attr' (use chamfer with attribute loss), 'cd5d' (treat doppler and rcs as extra dimensions in chamfer)",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=1.0,
        help="Weight for the MSE loss term",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The type of prediction the model makes: 'epsilon' (predict noise), 'sample' (predict x0 directly)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        help="The sampling method to use during inference: 'ddpm' (standard DDPM sampling), 'ddim' (DDIM sampling)",
    )
    parser.add_argument(
        "--wan_frames", 
        type=int,
        default=5,
        help="Number of frames to use for Wan's VAE latent conditioning"
    )   
    parser.add_argument(
        "--wan_frame_mode",
        type=str,
        default="repeat",
        help="Mode for handling Wan's VAE latent frames: 'repeat/center/past/future'"
    )
    parser.add_argument(
        "--wan_frame_stride",
        type=int,
        default=1,
        help="Stride for selecting frames for Wan's VAE latent conditioning"
    )
    parser.add_argument(
        "--wan_edge_policy",
        type=str,
        default="skip",
        help="Policy for handling edge frames in Wan's VAE latent conditioning: 'skip/pad'"
    )
    parser.add_argument(
        "--set_cond_type",
        type=str,
        default="film",
        choices=["film", "xattn", "film-xattn"],
        help="Conditioning type for SetTxDnsr: 'film', 'xattn', or 'film-xattn'",
    )
    parser.add_argument(
        "--scale_eps2x0_conversion",
        action="store_true",
        help="During epsilon to x0 conversion, scale the predicted epsilon by the standard deviation of the noise added at each timestep, as suggested in some implementations to improve stability. This is only applied when prediction_type is 'epsilon' and the CD loss is used."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default_exp_name",
        help="Optional experiment name to use in logging. If not provided, a name will be generated based on the other arguments.",
    )
    parser.add_argument(
        "--set_tx_dim",
        type=int,
        default=64,
        help="Dimension of the Set Transformer features in the SetTxDnsr model",
    )
    parser.add_argument(
        "--use_condition_pooling",
        action="store_true",
        help="Pool WAN condition spatially before FiLM/XAttn conditioning.",
    )

    parser.add_argument(
        "--condition_pool_kernel",
        type=int,
        default=4,
        help="Spatial pooling kernel/stride for WAN condition.",
    )
    parser.add_argument(
        "--use_wan_pos_emb",
        action="store_true",
        help="Add learned 2D positional embeddings to WAN tokens for cross-attention.",
    )
    parser.add_argument(
        "--man_one_distribution",
        action="store_true",
        help="Use a single distribution for all scenes in the MAN dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val_scene_id",
        type=int,
        default=-1,
        help="Scene ID to use for validation",
    )

    parser.add_argument(
        "--from_scene_id",
        type=int,
        default=0,
        help="Starting scene ID for training (inclusive)",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="Number of scenes to use for training",
    )
    parser.add_argument(
        "--sensor_side",
        type=str,
        default="left",
        help="Sensor side to use for training: 'left' or 'right'",
    )
    parser.add_argument(
        "--output_unnormalized",
        action="store_true",
        help="If set, output unnormalized point clouds instead of normalized ones.",
    )

    args = parser.parse_args()


    return args

if __name__ == "__main__":
        

    args = parse_args()

    print(f"args done: {[k for k in vars(args)]}")

    system_key = "ddpm_cond_slow"
    wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth"
    checkpoint_dir = f"/data/palakons/{system_key}/cache_unnorm/"
    os.makedirs(checkpoint_dir, exist_ok=True)  

    print(f"Checkpoint directory: {checkpoint_dir}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wan_spec = {"wan_frames": args.wan_frames, "wan_frame_mode": args.wan_frame_mode, "wan_frame_stride": args.wan_frame_stride, "wan_edge_policy": args.wan_edge_policy}
    print(f"dev {device}")
    print(f"wan spec {wan_spec}")

    if args.data_file == "man-mini":
        print(f"loading tc mini")
        tc_full = None
        tc_mini = TruckScenes("v1.0-mini", "/data/palakons/new_dataset/MAN/mini/man-truckscenes", False) 
    else:
        print(f"loading tc full")
        tc_mini =None
        tc_full = TruckScenes("v1.0-trainval", "/data/palakons/new_dataset/MAN/man-truckscenes", False) 
    total_scenes = 10 if args.data_file == "man-mini" else 597
    
    csv_meta_fname = f"man_meta_data_{args.cond_method}_{args.N}_{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}.csv"
    csv_meta_path = os.path.join(checkpoint_dir, csv_meta_fname)

    print(f"totla sc {total_scenes}")
    print(f"csv_meta_fname, {csv_meta_fname}")
    if not os.path.exists(csv_meta_path):
        with open(csv_meta_path, "w") as f:
            f.write("scene_id,data_file,sensor_side,is_normalized,total_points_original_mean,total_points_after_distance_filter_mean,total_points_visible_mean,total_points_original_std,total_points_after_distance_filter_std,total_points_visible_std,x0sbn3_mean_x,x0sbn3_mean_y,x0sbn3_mean_z,x0sbn3_max_half_range,doppler_mean,doppler_max_half_range,rcs_mean,rcs_max_half_range,wan_cond_max_abs\n")

    print(f"Loading VAE from checkpoint: {wan_vae_checkpoint}")

    wan_vae21_object = Wan2_1_VAE(
        vae_pth=wan_vae_checkpoint,
        device=device,
    )
    print("VAE loaded successfully.")

    for sc_id in trange(args.from_scene_id,min(args.from_scene_id + args.num_scenes, total_scenes), desc="Processing scenes", total=min(args.num_scenes, total_scenes - args.from_scene_id)):

         
        cache_fname = f"man_{args.data_file}_{sc_id}{f'_right' if args.sensor_side =='right' else ''}_{args.cond_method}_{args.N}_{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}{'_unnorm' if args.output_unnormalized else ''}.pkl"
        cache_path = os.path.join(checkpoint_dir, cache_fname)
        print(f"Processing scene {sc_id} with cache path: {cache_path}")


        if os.path.exists(cache_path):
            print(f"Cache file {cache_path} already exists. Loading from cache.")
            loaded_data = pickle.load(open(cache_path, "rb"))
            mands, frame_ids_loaded = loaded_data
            print(f"shapes of loaded mands: {[m.shape if m is not None else None for m in mands]} len(frame_ids_loaded['token']): {len(frame_ids_loaded['token'])}") # should be [n_scene, num_points, 3]
            all_32 =True
            for a in frame_ids_loaded['token']:
                 if len(a)!=32:
                    all_32 = False
                    print(f"Len should be 32, but got {len(a)} for frame token: {a}.")
            if all_32:
                print(f"All frame tokens have length 32.: Good!")
        else:
            norm_record ={'scene_id': sc_id, 'data_file': args.data_file, 'sensor_side': args.sensor_side,"is_normalized": not args.output_unnormalized}
            print (f"Cache file {cache_path} does not exist. Creating new cache.")

            # try:   
            if True: 
                mands_org = make_man_pc(
                    num_points=args.N,scene_ids= [sc_id],
                    n_scene=40,
                    is_dense=True,
                    device=device,
                    data_file=args.data_file,
                    wan_spec=wan_spec,
                    get_wan_cond=True,
                    # get_wan_cond=False,
                    radar_channel = "RADAR_LEFT_FRONT" if args.sensor_side == "left" else "RADAR_RIGHT_FRONT",
                    camera_channel = "CAMERA_LEFT_FRONT" if args.sensor_side == "left" else "CAMERA_RIGHT_FRONT",
                    trucksc= tc_mini if args.data_file == "man-mini" else tc_full   ,
                    wan_vae21_object = wan_vae21_object 
                )
            # except Exception as e:
            #     print(f"Error processing scene {sc_id} {args.data_file} {args.sensor_side}: {e}. Skipping this scene.")
            #     continue


            frame_ids = {'token':[mands_org[2][i]['frame_token'] for i in range(len(mands_org[0]))],"scene_id":[mands_org[2][i]['scene_id'] for i in range(len(mands_org[0]))],"frame_index":[mands_org[2][i]['frame_index'] for i in range(len(mands_org[0]))]} 
            mands = [mands_org[0], mands_org[1], torch.norm(mands_org[3], p=2, dim=-1, keepdim=True), mands_org[4]] #x0sbn3, wan_cond, doppler, rcs
            total_scenes_num = mands[0].shape[0]
            print(f"sizes of train_ds: {mands[0].shape}") # should be [n_scene, num_points, 3]


            npoints_original, n_points_after_distance_filter, npoints_filtered = [mands_org[2][i]['npoints_original'] for i in range(len(mands_org[2]))], [mands_org[2][i]['npoints_after_distance_filter'] for i in range(len(mands_org[2]))], [mands_org[2][i]['npoints_filtered'] for i in range(len(mands_org[2]))]

            

            norm_record['total_points_original_mean'] = np.mean(npoints_original)
            norm_record['total_points_after_distance_filter'] = np.mean(n_points_after_distance_filter)
            norm_record['total_points_visible'] = np.mean(npoints_filtered)
            norm_record['total_points_original_std'] = np.std(npoints_original)
            norm_record['total_points_after_distance_filter_std'] = np.std(n_points_after_distance_filter)
            norm_record['total_points_visible_std'] = np.std(npoints_filtered)

            if args.cond_mode =='none':
                print("cond_mode is 'none', setting wan_cond to zeros.")
                mands[1] =  torch.zeros(( total_scenes_num , 1), device='cpu').float()

            else:
                if args.cond_method == 'wan':
                    assert  mands[1] is not None, "wan_cond is None but cond_method is 'wan'."
                    try:
                        wan_max = mands[1].abs().max() 
                    except Exception as e:
                        print(f"Error computing max absolute value of wan_cond: {e}. Setting wan_max to 1 to avoid division by zero.")
                        wan_max = 1.0
                    if wan_max == 0:
                        print("Warning: max absolute value of wan_cond in mands is 0, which may cause division by zero during normalization. Setting wan_max to 1 to avoid this issue.")
                        wan_max = 1.0
                    norm_record['wan_cond_max_abs'] = wan_max

                    if not args.output_unnormalized:
                        mands[1] /= wan_max
                            
                        print(f"max abs wan_cond after normalization in ds: {wan_max}") # check the max abs value of wan_cond after normalization

                elif args.cond_method == 'scene_id': #actually frame_id
                    mands[1] = ((torch.arange(0, total_scenes_num, device='cpu').float() / total_scenes_num)*2-1).unsqueeze(1)  # [n_scene], normalized to [-1,1]
                        
            for i, name in enumerate(["x0sbn3", "wan_cond", "doppler","rcs"]): #assume no None
                # print(f"{name} {i}  shape: {mands[i].shape}, dtype: {mands[i].dtype}, device: {mands[i].device}") 
                mands[i] = mands[i].cpu()
                # print(f"{name} {i}  shape: {mands[i].shape}, dtype: {mands[i].dtype}, device: {mands[i].device}")  #must be on cpu for the following preprocessing steps to save GPU memory??
                assert mands[i].device == torch.device('cpu'), f"{name}  {i} is not on CPU, but on {mands[i].device}. Please move it to CPU before proceeding."

            for data_name,idx in zip(["x0sbn3","doppler","rcs"], [0,2,3]):
                mean, max_half_range = None, None
                    
                updated_data, mean_, max_half_range_ = normalize_data(mands[idx], save_filename_title=(f"/home/palakons/point_diffusion/output/sample/{args.data_file}_{sc_id}_{data_name}_data_normalization.png", f"{args.data_file} sc{sc_id} {data_name} data normalization,"))
                
                if mands[idx].shape[0]>0:
                    mean_ = mean_.view(-1) 
                    max_half_range_ = max_half_range_.view(-1)
                    if data_name == "x0sbn3": 
                        norm_record[f"{data_name}_mean_x"] = mean_[0].item()
                        norm_record[f"{data_name}_mean_y"] = mean_[1].item()
                        norm_record[f"{data_name}_mean_z"] = mean_[2].item()
                    else:   
                        norm_record[f"{data_name}_mean"] = mean_.item()
                    norm_record[f"{data_name}_max_half_range"] = max_half_range_.item()
                else:
                    if data_name == "x0sbn3": 
                        norm_record[f"{data_name}_mean_x"] = 0
                        norm_record[f"{data_name}_mean_y"] =0
                        norm_record[f"{data_name}_mean_z"] = 0
                    else:   
                        norm_record[f"{data_name}_mean"] = 0
                    norm_record[f"{data_name}_max_half_range"] = 1
                    
                if not args.output_unnormalized:
                    mands[idx] =updated_data 

            with open(cache_path, "wb") as f:            
                pickle.dump([ mands,frame_ids], f)
            for i, name in enumerate(["x0sbn3", "wan_cond", "doppler","rcs"]): #assume no None
                print(f"{name} {i}  shape: {mands[i].shape}, dtype: {mands[i].dtype}, device: {mands[i].device}")
            for key in frame_ids:
                print(f"frame_ids[{key}] length: {len(frame_ids[key])}")

            with open(csv_meta_path, "a") as f:
                f.write(f"{norm_record['scene_id']},{norm_record['data_file']},{norm_record['sensor_side']},{norm_record['is_normalized']},{norm_record['total_points_original_mean']},{norm_record['total_points_after_distance_filter']},{norm_record['total_points_visible']},{norm_record['total_points_original_std']},{norm_record['total_points_after_distance_filter_std']},{norm_record['total_points_visible_std']},{norm_record['x0sbn3_mean_x']},{norm_record['x0sbn3_mean_y']},{norm_record['x0sbn3_mean_z']},{norm_record['x0sbn3_max_half_range']},{norm_record['doppler_mean']},{norm_record['doppler_max_half_range']},{norm_record['rcs_mean']},{norm_record['rcs_max_half_range']},{norm_record.get('wan_cond_max_abs', 'N/A')}\n")

        print(f"Saved frame IDs to {cache_path}")