import torch
import os
import pickle
from io_dataset import normalize_data,make_man_pc
# from fitone_dfs_cond_log import parse_args
from tqdm import trange


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
    args = parser.parse_args()


    return args

if __name__ == "__main__":
        

    args = parse_args()



    system_key = "ddpm_cond_5"
    checkpoint_dir = f"/data/palakons/{system_key}/checkpoints/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wan_spec = {"wan_frames": args.wan_frames, "wan_frame_mode": args.wan_frame_mode, "wan_frame_stride": args.wan_frame_stride, "wan_edge_policy": args.wan_edge_policy}


    for sc_id in trange(args.from_scene_id, args.from_scene_id + args.num_scenes, desc="Processing scenes"):
        cache_fname = f"man_{args.data_file}_{sc_id}_{args.cond_method}_{args.N}_{args.cond_mode}_{args.wan_frames}_{args.wan_frame_mode}_{args.wan_frame_stride}_{args.wan_edge_policy}.pkl"
        cache_path = os.path.join(checkpoint_dir, cache_fname)
            
        mands = make_man_pc(
            num_points=args.N,scene_ids= [sc_id],
            n_scene=40,
            is_dense=True,
            device=device,
            data_file=args.data_file,
            wan_spec=wan_spec,
            get_wan_cond=True
        )


        frame_ids = {"train": {'token':[mands[2][i]['frame_token'][:5] for i in range(len(mands[0]))],"scene_id":[mands[2][i]['scene_id'] for i in range(len(mands[0]))],"frame_index":[mands[2][i]['frame_index'] for i in range(len(mands[0]))]} }
        mands = [mands[0], mands[1], torch.norm(mands[3], p=2, dim=-1, keepdim=True), mands[4]] #x0sbn3, wan_cond, doppler, rcs

        
        frame_count={"train": mands[0].shape[0], "eval": 0, "test": 0}
        print(f"sizes of train_ds: {mands[0].shape}") # should be [n_scene, num_points, 3]

        if os.path.exists(cache_path):
            print(f"Cache file {cache_path} already exists. Loading from cache.")
            continue

        if args.cond_mode =='none':
            print("cond_mode is 'none', setting wan_cond to zeros.")
            mands[1] =  torch.zeros(( mands[0].shape[0] , 1), device='cpu').float()

        else:
            if args.cond_method == 'wan':
                assert  mands[1] is not None, "wan_cond is None but cond_method is 'wan'."
                wan_max = mands[1].abs().max() 
                if wan_max == 0:
                    print("Warning: max absolute value of wan_cond in mands is 0, which may cause division by zero during normalization. Setting wan_max to 1 to avoid this issue.")
                    wan_max = 1.0

                mands[1] /= wan_max
                    
                print(f"max abs wan_cond after normalization in ds: {mands[1].abs().max() if mands[1] is not None else 'N/A'}") # check the max abs value of wan_cond after normalization

            elif args.cond_method == 'scene_id': #actually frame_id
                total_scenes_num = frame_count["train"] + frame_count["eval"] + frame_count["test"]
                mands[1] = ((torch.arange(0, 0+mands[0].shape[0], device='cpu').float() / total_scenes_num)*2-1).unsqueeze(1)  # [n_scene], normalized to [-1,1]
        
        
        for i, name in enumerate(["x0sbn3", "wan_cond", "doppler","rcs"]): #assume no None
            print(f"{name} {i}  shape: {mands[i].shape}, dtype: {mands[i].dtype}, device: {mands[i].device}") 
            mands[i] = mands[i].cpu()
            print(f"{name} {i}  shape: {mands[i].shape}, dtype: {mands[i].dtype}, device: {mands[i].device}")  #must be on cpu for the following preprocessing steps to save GPU memory??
            assert mands[i].device == torch.device('cpu'), f"{name}  {i} is not on CPU, but on {mands[i].device}. Please move it to CPU before proceeding."

        for data_name,idx in zip(["x0sbn3","doppler","rcs"], [0,2,3]):
            mean, max_half_range = None, None
            
            mands[idx], mean_, max_half_range_ = normalize_data(mands[idx], save_filename_title=(f"/home/palakons/point_diffusion/output/sample/{args.data_file}_{sc_id}_{data_name}_data_normalization.png", f"{args.data_file} sc{sc_id} {data_name} data normalization"))

        with open(cache_path, "wb") as f:            
            pickle.dump([ mands,frame_ids], f)
        print(f"Saved frame IDs to {cache_path}")