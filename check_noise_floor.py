import os
import numpy as np
import pypcd4
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from truckscenes import TruckScenes

from matplotlib import pyplot as plt

import torch
from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance


DATA_ROOT = "/data/palakons/new_dataset/MAN/man-truckscenes"
VERSION = "v1.0-trainval"
RADAR_CHANNEL = "RADAR_RIGHT_FRONT"
RADAR_CHANNEL = "LIDAR_TOP_FRONT"


def quat_to_R(q):
    """MAN quaternion [w,x,y,z] -> rotation matrix."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def load_radar_xyz(trucksc, sample_data):
    path = os.path.join(DATA_ROOT, sample_data["filename"])
    pc = pypcd4.PointCloud.from_path(path).pc_data

    return np.stack(
        [pc["x"], pc["y"], pc["z"]],
        axis=1,
    ).astype(np.float64)


def sensor_to_global(trucksc, sd, xyz):
    """
    radar sensor -> ego -> global
    """
    calib = trucksc.get(
        "calibrated_sensor",
        sd["calibrated_sensor_token"],
    )
    pose = trucksc.get(
        "ego_pose",
        sd["ego_pose_token"],
    )

    R_calib = quat_to_R(calib["rotation"])
    t_calib = np.asarray(calib["translation"])

    R_pose = quat_to_R(pose["rotation"])
    t_pose = np.asarray(pose["translation"])

    xyz_ego = xyz @ R_calib.T + t_calib
    xyz_global = xyz_ego @ R_pose.T + t_pose

    return xyz_global


def symmetric_nn_distance(P, Q):
    """
    Symmetric nearest-neighbour distance.

    Returns distances in METRES, not squared metres.
    """
    if len(P) == 0 or len(Q) == 0:
        return None

    tree_Q = cKDTree(Q)
    d_PQ, _ = tree_Q.query(P, k=1)

    tree_P = cKDTree(P)
    d_QP, _ = tree_P.query(Q, k=1)

    return {
        "mean": 0.5 * (d_PQ.mean() + d_QP.mean()),
        "median": np.median(np.concatenate([d_PQ, d_QP])),
        "p68": np.percentile(np.concatenate([d_PQ, d_QP]), 68),
        "p90": np.percentile(np.concatenate([d_PQ, d_QP]), 90),
        "p95": np.percentile(np.concatenate([d_PQ, d_QP]), 95),
        "forward_mean": d_PQ.mean(),
        "backward_mean": d_QP.mean(),
        "n0": len(P),
        "n1": len(Q),
    }


def plot_points_xy_multi(points, title=None, out_file=None,label=None, xlim=None, ylim=None):
    # points: list of (N,2) arrays
    # label: list of labels
    plt.figure(figsize=(10, 10))
    for i, p in enumerate(points):
        if label:
            plt.scatter(p[:, 0], p[:, 1], s=1, label=label[i])
        else:
            plt.scatter(p[:, 0], p[:, 1], s=1, label=f"Points {i}")
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    plt.legend()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()
def plot_points_xy(P, Q, title=None, out_file=None):
    plot_points_xy_multi([P, Q], title=title, out_file=out_file)


def main():

    trucksc = TruckScenes(VERSION, DATA_ROOT, False)

    scene = trucksc.scene[0]

    sample = trucksc.get(
        "sample",
        scene["first_sample_token"],
    )
    # print(f"channels={list(sample['data'].keys())}") #channels=['RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE', 'RADAR_RIGHT_FRONT', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_LEFT_BACK', 'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR', 'CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK']
    
    sd_token = sample["data"][RADAR_CHANNEL]

    results = []

    timestamp_prev = None
    timestamp_curr = None
    first_time = True
    count = 0
    points = []
    points_local = []
    while sd_token:
        count += 1
        timestamp_prev = timestamp_curr
        sd0 = trucksc.get("sample_data", sd_token)
        timestamp_curr = sd0["timestamp"]

        next_token = sd0["next"]
        if not next_token:
            break

        sd1 = trucksc.get("sample_data", next_token)

        P0 = load_radar_xyz(trucksc, sd0)
        P1 = load_radar_xyz(trucksc, sd1)

        P0_global = sensor_to_global(trucksc, sd0, P0)
        P1_global = sensor_to_global(trucksc, sd1, P1)
        points.append(P0_global)
        points_local.append(P0)
        stats_raw = symmetric_nn_distance(P0, P1)

        stats_global = symmetric_nn_distance(
            P0_global,
            P1_global,
        )
        print(
            f"raw={stats_raw['median']:.3f} m "
            f"aligned={stats_global['median']:.3f} m"
        )

        if first_time:
            first_time = False
            P2 = P0.copy()
            P2[:, 0] += 0.5
            P2[:, 0] -= 0.5

            print(symmetric_nn_distance(P0, P2))

            tree = cKDTree(P1_global)
            d, idx = tree.query(P0_global)

            plt.hist(d, bins=np.linspace(0, 20, 100))
            plt.xlabel("Nearest-neighbor distance [m]")
            plt.ylabel("Count")
            plt_out_file  = f"/home/palakons/point_diffusion/output/nn_distance_hist_{sd0['token']}.png"
            plt.savefig(plt_out_file)
            plt.close()


        if stats_global is not None:
            dt = (sd1["timestamp"] - sd0["timestamp"]) / 1e6

            stats_global["dt"] = dt
            stats_global["token0"] = sd0["token"]
            stats_global["token1"] = sd1["token"]

            results.append(stats_global)

        sd_token = next_token
    print(f"Processed {count} radar frames, {len(results)} results, points={len(points)}")
    plot_points_xy_multi(points, title="Lidar points in global frame", out_file="/home/palakons/point_diffusion/output/lidar_points_global.png")
    plot_points_xy_multi(points_local, title="Lidar points in local frame", out_file="/home/palakons/point_diffusion/output/lidar_points_local.png")  
    
    #plot N latest frames of global radar points, stride 1
    N = 5
    minmax = {"x": [np.inf, -np.inf], "y": [np.inf, -np.inf]}
    for p in points:
        minmax["x"][0] = min(minmax["x"][0], p[:, 0].min())
        minmax["x"][1] = max(minmax["x"][1], p[:, 0].max())
        minmax["y"][0] = min(minmax["y"][0], p[:, 1].min())
        minmax["y"][1] = max(minmax["y"][1], p[:, 1].max())
    print(minmax)


    for i in range(len(points)-N):
        # calculate CD

        plot_points_xy_multi([points[j] for j in range(i, i+N)], title=f"Lidar points in global frame, {i} to {i+N} frames", out_file=f"/home/palakons/point_diffusion/output/lidar_points_global_{i}.png", label=[f"Fr:{j} CD{pt3d_chamfer_distance(torch.from_numpy(points[i]).unsqueeze(0).float(), torch.from_numpy(points[j]).unsqueeze(0).float() )[0].item():.4f}" for j in range(i, i+N)], xlim=minmax["x"], ylim=minmax["y"])

    os.system('ffmpeg -y -framerate 20 -i /home/palakons/point_diffusion/output/lidar_points_global_%d.png -vf "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a" -loop 0 /home/palakons/point_diffusion/output/lidar_points_global.gif')

    for r in results[:20]:
        print(
            f"dt={r['dt']:.3f}s "
            f"N={r['n0']:4d}->{r['n1']:4d} "
            f"mean={r['mean']:.3f}m "
            f"P50={r['median']:.3f}m "
            f"P90={r['p90']:.3f}m "
            f"P95={r['p95']:.3f}m"
        )
if __name__ == "__main__":
    main()