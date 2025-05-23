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


def sample(feature_model, model, scheduler, T, B, N, D, gt_normed, device, data_mean, data_std, img_size, camera, frame_token, image_rgb, stat_factor=0.):
    scheduler.set_timesteps(T)
    x_t = torch.randn(B, N, D).to(device)
    xts_tensor_list = []

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            # print("x_t", x_t.shape)
            # x_t torch.Size([1, 4, 3])

            # x_t_cond = get_camera_conditioning(
            x_t_cond = flattened_conditioning(
                img_size,
                                               feature_model, camera, frame_token, image_rgb, x_t, device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0,stat_factor =stat_factor)

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
                    train_loss_list, val_loss_list, epoch, base_dir, config, run_name, code=None):
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

def flattened_conditioning(img_size, feature_model,  camera, frame_token, image_rgb, normed_points, device, raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1]),stat_factor=0):
    """
    Flattened conditioning for the model.
    """
    local_features = []
    for i, frame_tkn in enumerate(frame_token):
        if frame_tkn in local_feature_cache:
            local_feature = local_feature_cache[frame_tkn]
        else:
            local_feature = feature_model(
                image_rgb[i:i + 1].float().to(device), return_upscaled_features=False).to(device)
            # print("local_feature", local_feature.shape) #1, 384, 38, 38
                
            local_feature_cache[frame_tkn] = local_feature
            # print("saved local_feature_cache", frame_tkn, local_feature.shape)

        local_features.append(local_feature)
    local_features = torch.cat(local_features, dim=0).to(device)
    # print("local_features", local_features.shape) #2, 384, 38, 38
    flattened_feature = local_features.reshape(local_features.shape[0], -1) 
    #expand to [B,N,384*38*38]'
    # print("flattened_feature", flattened_feature.shape) #2, 384*38*38
    flattened_feature = flattened_feature.unsqueeze(1).expand(normed_points.shape[0], normed_points.shape[1], -1)
    # print("flattened_feature", flattened_feature.shape) #2, 4, 384*38*38
    return flattened_feature

def get_camera_conditioning(img_size, feature_model,  camera, frame_token, image_rgb, normed_points, device, raster_point_radius: float = 0.0075, raster_points_per_pixel: int = 1, bin_size: int = 0, scale_factor: float = 1.0, point_mean=torch.tensor([0, 0, 0]), point_std=torch.tensor([1, 1, 1]),stat_factor=0):
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
                
            if False: #inspect stat from VITs
                small_feature = feature_model(                image_rgb[i:i + 1].float().to(device),        return_type = 'features',        return_upscaled_features = False).to(device)
                print("small f shape",small_feature.shape) #small f shape torch.Size([1, 384, 38, 38])
                #make state plot maybe box plot
                fig = plt.figure()
                x = range(small_feature.shape[1])
                y = small_feature.mean(dim=[0,2,3]).cpu().numpy()
                e = small_feature.std(dim=[0,2,3]).cpu().numpy()
                plt.errorbar(x, y, e, linestyle='None', marker='^')
                foutput = f'/home/palakons/from_scratch/vitsstat_{frame_tkn[:3]}.png'
                csv_output = f'/home/palakons/from_scratch/vitsstat.csv'
                plt.savefig(foutput)
                with open(csv_output, 'a') as f:
                    f.write(f"{frame_tkn},{y.tolist()},{e.tolist()}\n")
            local_feature_cache[frame_tkn] = local_feature
            # print("saved local_feature_cache", frame_tkn, local_feature.shape)

        feature_std=[0.554600779307215,0.537929526836671,0.832839469288984,0.688226256184848,0.452582867763333,0.537518454778647,0.464593682714164,0.437203680675777,0.614834816260369,0.655601191992269,0.542814605340389,0.462566489675832,0.744393020987382,0.876776283223533,0.473776484246442,0.479791087846992,0.594711949907883,0.734973297405673,0.677756093967993,0.53951509742763,0.644789558296267,0.490651733810398,0.524435484338827,0.557896917712952,0.533363231022826,0.71959571881595,0.821669035022517,0.565855233702463,0.500869936704265,0.510585705364982,0.533950558633835,0.205956655726138,0.952407599157279,0.511780020773942,0.56430255506086,0.562207853710966,0.795055679274598,0.615666753839968,0.945309950513202,0.613559614039172,0.590138292706903,0.49895146991849,0.52194562529096,0.700932696257224,0.645224403412526,0.703786150253158,0.579022142050683,0.624623335179884,0.541606998019367,0.562395923668777,0.712495231613632,0.507601968362934,0.515441312415464,0.590177789793812,0.567988672620836,0.520565201662483,0.510816627602666,0.76818350259686,0.488852958039427,0.603074701372574,0.46656750301952,0.81194133499194,0.687751090612731,0.667133535250435,0.624663717254817,0.501100785251869,0.620949447506688,0.521222296530126,1.67230050691576,0.485441180163265,0.589241415067557,0.515164438549794,0.520352712238719,0.880896855214973,1.16444350669608,0.687580878316299,0.913020985910151,0.512359715196589,0.652899539791035,0.707458523798391,0.509679594532527,0.615546800382041,0.547993458047345,0.563178761327105,0.631949227786975,0.577033913196875,0.563387787793224,0.492701939753098,0.581811630397851,0.672342205937316,0.471475458153929,0.50704151131252,0.503301523131877,0.642306739259866,0.451836903203987,0.480716980219188,0.562270515229471,0.653803090548096,0.562978052076063,0.563346664415329,0.532772206803312,0.527431049080062,0.4761023406452,0.524083223476265,0.566022437609096,1.548152653029,0.604865754147375,0.656138367437757,0.48468315425612,0.622373694495396,0.529181036978646,0.929824320635805,0.728397374784218,0.564818159127398,0.819068114911161,0.42646464294041,0.479548894061933,0.804648623161829,0.505984334382126,0.484642699356241,0.574588198405022,1.08833634024418,0.750882806559891,0.513093266138995,0.878275534733682,0.644788941699198,0.520722742778852,0.546955621186042,1.68160238613943,0.618811511669013,0.662992918038035,0.476514516424947,0.482469792667734,0.551356392203334,0.520986069622267,0.421094755888871,0.500634569523893,0.664449111306439,0.615617366779416,0.57880152930854,0.572787578518645,0.570851195816962,0.49013428695875,0.552836404944352,0.641925781385396,0.4094272150653,0.524628281438394,0.856564104672934,0.490164841339981,0.645508999452972,0.533623380562076,0.623014565239808,0.499013523449759,0.545682134956882,0.436540272875005,0.618129903541916,0.672531805111521,0.591653041976553,0.597381332551848,0.49251697419277,0.769636827824785,0.664115094589816,0.64581527344475,0.493935518755869,0.667263958195943,1.87137467701686,0.540378827374847,0.59453176214772,0.48859332805871,0.624468576333967,0.94131336614222,0.670532968605627,0.432959148015179,0.609893937943585,0.575034437746438,0.642640775551616,0.858139086274925,0.523537133672446,0.602505686392768,0.528739318409639,0.870830715835544,0.513424884595948,0.44377540533562,0.598058128526848,0.642629683623617,0.613665341998579,0.730120219556931,0.568148340445044,0.547615655006926,0.53145645069097,0.574715628363727,0.514170328299472,0.612641261520206,0.565704651683949,1.13245636480507,0.777454476756735,0.504563255417161,0.527662170049688,0.431552674864039,0.593123691635424,0.57412734087274,0.904694471089732,0.719175959643606,0.630864273991396,0.835916793527776,0.523286647277629,0.61445780317757,0.590514119765582,0.483685810905256,0.705717538734291,0.493100933751349,0.611206301330545,0.655086160833398,0.620564858345275,0.956640767047478,0.688105997754872,0.478751253103875,0.669987778934299,0.574186293907827,0.507343785972242,0.498126890554616,0.53289511879493,0.517586019812623,0.730427838892392,0.526928948549853,0.526536079997553,0.48831458546086,0.529265078969169,0.724748374666826,1.09791063510496,0.555084413719637,0.762617807307714,0.55180389162783,0.478251157689463,0.547460057719871,0.624541493122867,0.491159311320825,0.662116624604175,0.692167320175707,0.647739737982363,0.588025194686042,0.516939450041867,0.566471112847149,0.705086424386061,0.620102945951659,0.63946946398484,0.539014912078646,0.681229091935792,0.735602753309541,0.612551584950629,0.614330878972411,0.692412034010566,0.559939206493927,0.532494290160364,0.553348841248039,1.01148365465637,0.463048907663738,0.564136037959149,0.471639895181079,0.824563627612978,0.935867237653415,0.670625440507263,0.543092746137182,0.516630946115443,0.655923087682109,0.654188351723103,0.817503528934331,0.495194648577435,0.579224073159393,0.621843938110739,0.578064071571682,0.664213593203764,0.586763039855949,0.530875859909773,0.52979508125196,0.652535741005965,0.524113711407826,0.647883721950918,0.536738993572899,0.782758339218088,0.52136919043644,0.635436489350748,0.886456202056285,0.578480236652527,0.690974439074642,0.666654493777065,0.454023396161785,0.568800197053621,1.01924407285946,0.484387595287388,0.468497700182548,0.645429180479539,0.497374330452479,0.582123983818825,0.498510346386763,0.437306589935678,0.566052043939652,0.536745148956718,0.836548292208024,0.56888185501007,0.769100324537892,0.662762039124775,0.573452520217722,0.558909744814009,0.486306120642619,0.501285443009788,0.750101173623306,0.455186191605655,0.778484193655075,0.619117065378341,0.53497685601224,0.485536745871903,0.845468638699109,0.611285051894612,0.441928331945319,0.509283406031377,0.445047167212771,0.536150691318069,0.542703947184545,0.507821358342099,0.567193649047402,0.643185396221863,0.695187938501485,1.09485032880718,0.537807280961283,0.626833033811764,0.573065803835496,0.66059170874435,0.331773219624289,0.569081773547911,0.451976450902335,0.541241706186236,0.88972055344132,0.561618748858036,0.620746092829937,0.540066211903894,0.66815777921847,0.561914618867896,0.54331900219027,0.542688114339779,0.523208028896104,0.603413412084337,0.53813414603346,0.524032670830881,0.523408245094213,0.491012485637503,0.58999431661377,0.51054910961674,0.546471673704773,0.538977681276544,0.626401930253993,0.45114385060188,0.671521226110925,0.676284813889698,0.507504168836228,0.607263100365296,0.786635769636837,0.480546838358651,0.580454413976386,0.538946076305407,0.652781187454422,0.792867956031318,0.750708484413498,0.59981988538097,0.903871064425602,0.657712662806385,0.59067164583514,0.685034905444457,0.664816433629482,0.683258212960194,0.467186489486786,0.529094731170526,0.704851080666024,0.587151074461343,0.701697518994184,0.796182234221306,0.550139815600896,0.654282282306366,0.68359494783661,0.506101286276429,0.460845698109934,0.711649158215783,0.487962830555745,0.528769730909249]
        feature_mean=[0.421777687289497,0.0189073552292856,-0.0995346066192723,0.565484117377888,0.441723788326436,-0.108666602522135,-0.356904604218222,0.0966046047396954,0.778061601248654,-0.374680158766833,-0.152837291359901,-0.0769238988445562,-1.21274781227111,1.05199857191605,-0.490998479453,0.443240184675563,-1.04878332398154,0.29465274174105,0.188328082588585,0.256385975263335,-0.580074800686402,-0.444082157178358,0.161515402861616,-0.85429742119529,1.17128594355149,1.01965436610308,0.59930682182312,0.224380382082678,-0.892479538917541,0.790715856985612,0.300800564614209,0.00514381645586002,-0.234960117123343,-0.66159718686884,0.146795686673034,0.286257856271483,-0.4806390106678,0.774282921444286,0.143041005188768,-0.000775707716291609,-1.42830793424086,-1.06234429641203,0.643289880319075,0.076742246408354,-0.756370219317349,0.0898628498596899,-1.58089023286646,-0.190413393757559,0.573497517542405,1.02234416116367,1.4869450005618,-0.445749838243831,0.624607162042097,-0.00027644448511479,0.331359974362633,0.495943928306753,0.08205357092348,0.218869465318593,0.354780717329545,0.481288647109811,0.832005728374828,0.463102294640107,-1.02428408644416,0.252535738728263,0.71551749381152,-0.100442851300944,-2.59949036078019,-0.404046210375699,-0.884089269421317,-0.00388156038454987,-0.632668245922435,0.508617750623009,-0.420100913806395,-0.800552481954748,-1.25346048311753,1.14976967464793,0.227126645770939,0.573168342763727,-0.467174123633991,1.46247666532343,-0.306479084220799,0.557158995758403,-1.40018227967348,-0.221813715317032,0.614042769778858,1.09720938855951,1.49746463515541,-0.716294716704975,0.403976765545931,-1.12212851914492,-1.06398805704983,-1.08543429591438,-0.509612893516367,0.409335296262394,0.959251685576004,-0.717626669190146,-0.71592518416318,-0.3250126730312,0.0545013398778708,-0.499477730555968,-0.375995123928243,0.732472631064328,0.315157618034969,-0.679124805060299,1.52181938561526,-0.151895708658478,0.51705559275367,0.16978399049152,-1.18136978149413,0.604415833950042,1.04762237180363,2.89772696928544,0.642916592684659,0.448308272795243,0.523191996596076,-2.24355589259754,1.48607392744584,0.786534807898781,-0.40287924896587,0.0512661997398192,0.587903922254389,0.277461286295544,-0.738653459332205,0.513643752444874,2.13697537508878,0.939538104967637,-0.0814818841489878,-1.59736467491496,-5.18619567697698,-0.44851451028477,-0.228693941777402,-0.369210804050619,0.365206066857684,0.650102181868119,-0.803491711616516,-0.532293585213748,-0.201308051293546,0.412144216624173,-0.393940936435352,1.10410324009982,0.098650220104239,-0.0660845371471208,1.14419175278056,-0.17324044487693,-0.0228313033215024,0.251477559859102,-0.374627221714366,0.249604137106375,0.712218338792974,-0.433897674083709,-0.204857629808512,0.526087414134632,0.562428550286726,-0.315324924208901,-0.400966102426702,0.589831325140866,0.0210791428372348,-0.176420122385025,0.546601219610734,1.17288706519387,0.327926341782916,-0.14711055565964,0.121763433922421,-1.72036310759457,0.258670349012721,-5.47392892837524,-0.0995844975113866,-0.123713026331229,-0.0497813052074475,0.425344784151424,0.274083371866833,0.0809571949595752,-1.29930041053078,1.73344296758825,0.503374397754668,-0.206498981876806,0.222842547026547,0.176804549992084,-0.348824392665516,-0.373038619756698,-0.105019704184749,-0.0943091308528725,-0.477882393381812,-0.0350356834347952,-1.17898781733079,-0.243675860491665,0.597651199861006,0.761674171144312,0.00463196965442462,-1.69159880551424,0.0379811872474172,-1.08851532502607,1.40051312880082,-0.699713132598183,1.71449094468897,-0.723799076947298,-0.0120831045576117,0.908208072185516,0.628958349878137,-0.765700042247772,-1.03319495374506,-0.180859078737822,-0.637368142604827,1.50938520648262,0.79251649162986,-0.0792580277405001,-0.673354582353071,0.214839676564389,-0.355802126906134,1.03219702568921,-0.732817563143643,-0.394166380167007,-0.0526209548962386,-0.21178913116455,0.0850402777167882,1.11553752422332,0.551978571848435,-0.360704668543555,-0.713010890917344,1.52760311690243,-0.262620891359719,0.665216294201937,-0.800644592805342,-1.3425395488739,1.1695739030838,-0.0594951767813075,-0.243145560676401,1.03334767710078,-0.581391627138311,0.517951783808795,-0.537768713452599,0.536346646872433,-0.0588506145233457,-0.577887735583565,-0.131241666322404,-0.669529394669966,-0.0847837924957274,0.586357214234092,-0.513572974638505,-1.36442070657556,-1.31314436955885,0.557277909734032,-0.258777840570969,0.0270111573521386,-0.0397197174077683,-1.00250174782493,-0.449676765636964,-1.21949599005959,-0.301929983225735,-0.0346633776683699,0.631520363417538,-0.534365293654528,-0.283336715264753,0.0234901154921812,0.579537483778866,0.238388728011738,-0.124873600900173,1.63026039166883,-0.613270586187189,3.25391275232488,1.65676919980482,0.850932652300053,0.747917505827817,-1.32377887855876,-0.200187177820639,-0.913858153603293,-0.631852502172643,0.692758841948075,1.10261060974814,0.434704062613574,0.446560859680175,-0.639799562367526,1.25077674605629,0.651713934811678,0.787934287027879,0.226934608410705,0.380652579394253,-1.06103372573852,1.06220962784507,-0.251439261165532,-0.235914103009484,-0.612384135072881,-0.209325054491108,0.224179855801842,0.376402369954369,0.629222138361497,1.56410290978171,0.115386533804915,-1.72617764906449,-1.62868805365129,0.763167841867967,0.31407265771519,0.371359711343591,-1.27027526768771,-0.425340928814627,-0.501113376834175,2.00023218718441,-0.0179394854402,-0.488821146163073,-0.354322609576312,-1.06758773326873,-0.373636757785623,0.311607433990998,-1.18474172462116,0.895847699858925,0.137461977926167,-0.374713410030711,0.640658389438282,-0.269959666512229,-1.02525901252573,-1.62485052238811,0.594793081283569,-0.450251408598639,1.15334000370719,0.0525070877576416,0.156775041060014,0.431551299311898,-0.90034686977213,-0.463377811691977,0.0650009793991391,-1.11583976853977,-1.11231204596432,1.09098522229628,-1.51357526128942,1.21786909753626,-0.271727693351832,-0.3392025151036,1.63211283900521,-0.698503954844041,0.392656873572956,-1.3857634934512,-1.12882099368355,-1.17452721162275,-0.338672540404579,0.124583656306971,0.483165933327241,-0.718958274884657,0.54862522537058,0.121930671347813,1.04927043481306,0.590841084718704,-0.08817688324912,0.428348010236566,0.172704559158195,0.257355069572275,0.732194759628989,-1.26525141976096,0.748026165095242,-0.017966709353707,0.435383707284927,-0.497205796566876,1.44420852444388,-1.25229209119623,1.28680310466072,-1.19660906358198,0.0729191000150008,1.5556114912033,-0.328946501016616,-0.0920295546000653,-0.214668563821099,0.568817057392813,-0.313449274409901,-0.826672521504489,0.68809058449485,-1.14215156165036,0.28966334733096,0.779496436769312,0.317960397763685,-0.623332906853069,-0.509313984350724,-0.0773121758618137,-0.00130324065685274,0.371918420899998,-0.305517341602932,0.419162406162782,0.770624794743277,0.806475568901408,1.50114837559786,-0.720405692403966,0.110616782172159,-0.225093380971388,0.394939904863184,0.726588384671644,0.305522161451252]
        feature_mean = torch.tensor(feature_mean).to(device).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        feature_std = torch.tensor(feature_std).to(device).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)

        adjusted_feature = (local_feature-feature_mean*stat_factor) / (1 +stat_factor*(feature_std-1))

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
    parser.add_argument("-statf", "--stat_factor", type=float, default=0.0,
    help="Statistical factor for conditioning: 0 means no scaling, 1 means full scaling to 0-mean and 1-std, channel-wise")
    args = parser.parse_args()

    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()

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
    vits_dim = 384*(618//16)*(618//16)  # 384*618//16*618//16


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
    stat_factor = args.stat_factor

    data_group = "pc2_man"
    run_name = f"{method}_{N:04d}_sc{M}-b{B}-p{n_pull}"
    log_dir = f"{base_dir}tb_log/{data_group}/{run_name}"
    dir_rev = 0
    while os.path.exists(log_dir):
        dir_rev += 1
        log_dir = f"{base_dir}tb_log/{data_group}/{run_name}_r{dir_rev:02d}"
    if dir_rev > 0:
        run_name += f"_r{dir_rev:02d}"
        print("run_name", run_name)
    # assert M <= n_pull-n_val, f"n_pull {n_pull} should be greater than M {M} + n_val {n_val}"

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
        "stat_factor": stat_factor,
    }
    # not dispaly "code" in the config
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
            t = torch.randint(0, T, (B,), device=device) #uniform
            x_t_data = scheduler.add_noise(x_0_data, noise, t)
            add_noise_time = time.time()
            # x_t_cond = get_camera_conditioning(
            x_t_cond = flattened_conditioning(
                img_size,
                feature_model, camera, frame_token, image_rgb, x_t_data,  device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0, point_mean=data_mean.to(device).float(), point_std=data_std.to(device).float(), stat_factor=stat_factor)
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

                # x_t_cond = get_camera_conditioning(
                x_t_cond = flattened_conditioning(
                    img_size,
                    feature_model, camera, frame_token, image_rgb, x_t_data, device=device, raster_point_radius=0.0075, raster_points_per_pixel=1, bin_size=0, scale_factor=1.0,stat_factor=stat_factor)

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
                                                         model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, image_rgb=image_rgb, camera=camera, img_size=img_size, frame_token=frame_token,stat_factor=stat_factor)

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
                    save_sample_json(path, epoch, x_0_data[i_result].unsqueeze(
                        0), xts, steps,  cd=cd.item(), data_mean=data_mean, data_std=data_std, config=exp_config)
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
                                                         model.eval(), scheduler, T, B, N, D, x_0_data, device, data_mean=data_mean, data_std=data_std, img_size=img_size, camera=camera, image_rgb=image_rgb, frame_token=frame_token,stat_factor=stat_factor)

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

                    path = f"{base_dir}/plots/{data_group}/{run_name}/sample_ep_{i_epoch:06d}_gt0_val-idx-{frame_tkn[:3]}.json"
                    save_sample_json(path, epoch, x_0_data[i_result].unsqueeze(0), xts,
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
                    train_loss_list=train_loss_list, epoch=epoch, base_dir=base_dir, config=exp_config, run_name=run_name, code=code)


if __name__ == "__main__":
    main()
