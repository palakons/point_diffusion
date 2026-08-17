# rerun_streamer.py

import os
import pickle

import numpy as np
import pypcd4
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import rerun as rr
import rerun.blueprint as rrb

from truckscenes import TruckScenes


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = "/data/palakons/new_dataset/MAN/man-truckscenes"
VERSION = "v1.0-trainval"
SCENE_ID = 408

IPADDR = "10.204.190.239"
RERUN_URL = f"rerun+http://{IPADDR}:9876/proxy"

system_key = "ddpm_cond_slow"
exp_name = (
    "xattn-B16_dim512_samplemse-lr1e-4-constantm1-"
    "smooth-lazy-1norm-center1-cameraframe-xyzonly-30090"
)

n_multi = 10

inference_dir = (
    f"/data/palakons/{system_key}/{exp_name}/"
    f"inference_sc{SCENE_ID}-multi{n_multi}"
)

PRED_PATH = os.path.join(
    inference_dir,
    "rerun_predictions_correct_cond.pkl",
)
SIDE= "RIGHT"
RADAR = f"RADAR_{SIDE}_FRONT"
CAMERA = f"CAMERA_{SIDE}_FRONT"

FPS = 20.0 #native frames
V_MAX = 100 / 3.6      # 27.78 m/s
BASE_RADIUS = 0.25
PERSISTENCE_K = 2

N_BARS = 4*2*4

Y_MAX = np.log10(101.0)   # ≈ 2.0043

X_RANGE = [0, 50]
Y_RANGE = [-50, 50]
Z_RANGE = [-2, 2]


# =============================================================================
# BLUEPRINT
# =============================================================================

def log_axes():

    # Camera-coordinate axes:
    #
    #   +X = right
    #   +Y = down
    #   +Z = forward
    #
    # But default viewer looks along Y,
    # so screen looks like X-Z with +Z up.

    axis_len = 10.0

    origins = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.float32)

    vectors = np.array([
        [axis_len, 0, 0],   # X
        [0, axis_len, 0],   # Y
        [0, 0, axis_len],   # Z
    ], dtype=np.float32)

    colors = np.array([
        [255,   0,   0],    # X = red
        [  0, 255,   0],    # Y = green
        [  0,   0, 255],    # Z = blue
    ], dtype=np.uint8)

    labels = ["+X", "+Y", "+Z"]

    endpoints = np.array([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ], dtype=np.float32)

    for root in [
        "gt",
        "gt_soft",
        "pred",
    ]:

        # Axes arrows
        rr.log(
            f"{root}/axes",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=colors,
                radii=0.04,
            ),
            static=True,
        )

        # X / Y / Z labels
        rr.log(
            f"{root}/axis_labels",
            rr.Points3D(
                endpoints,
                colors=colors,
                radii=rr.Radius.ui_points(4),
                labels=labels,
            ),
            static=True,
        )


def setup_view():

    eye = rrb.EyeControls3D(
        position=(0, -70, 25),
        look_target=(0, 0, 25),
        eye_up=(0, 0, 1),
    )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(
                        name="Image",
                        origin="/image",
                    ),

                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial3DView(
                                name="GT discrete",
                                origin="/gt",
                                eye_controls=eye,
                            ),

                            rrb.Vertical(
                                contents=[
                                    rrb.Spatial3DView(
                                        name="GT soft",
                                        origin="/soft_bins",
                                        eye_controls=eye,
                                    ),

                                    rrb.BarChartView(
                                        name="Persistence histogram",
                                        origin="/soft_bins",
                                        plot_legend=rrb.PlotLegend(
                                            visible=False,
                                        ),
                                    )
                                    
                                ],
                                row_shares=[3, 1],
                            ),

                            rrb.Spatial3DView(
                                name="Pred",
                                origin="/pred",
                                eye_controls=eye,
                            ),
                        ],
                        column_shares=[1, 1, 1],
                    ),
                ],
                row_shares=[1.0, 1.0],
            ),

            collapse_panels=True,
        )
    )

# =============================================================================
# BASIC GEOMETRY
# =============================================================================

def quat_to_R(q):
    return Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]
    ).as_matrix()


def sensor_to_world(trucksc, sd):

    calib = trucksc.get(
        "calibrated_sensor",
        sd["calibrated_sensor_token"],
    )

    pose = trucksc.get(
        "ego_pose",
        sd["ego_pose_token"],
    )

    Rc = quat_to_R(calib["rotation"])
    tc = np.asarray(calib["translation"])

    Rp = quat_to_R(pose["rotation"])
    tp = np.asarray(pose["translation"])

    return (
        Rp @ Rc,
        Rp @ tc + tp,
    )


def points_to_world(trucksc, sd, xyz):
    R, t = sensor_to_world(trucksc, sd)
    return xyz @ R.T + t


def points_world_to_sensor(trucksc, sd, xyz):
    R, t = sensor_to_world(trucksc, sd)
    return (xyz - t) @ R


# =============================================================================
# DATA
# =============================================================================

def load_image(sd):
    return np.asarray(
        Image.open(
            os.path.join(
                DATA_ROOT,
                sd["filename"],
            )
        ).convert("RGB")
    )


def load_radar(sd):

    pc = pypcd4.PointCloud.from_path(
        os.path.join(
            DATA_ROOT,
            sd["filename"],
        )
    ).pc_data

    xyz = np.stack(
        [
            pc["x"],
            pc["y"],
            pc["z"],
        ],
        axis=1,
    ).astype(np.float64)

    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= X_RANGE[0])
        & (xyz[:, 0] <= X_RANGE[1])
        & (xyz[:, 1] >= Y_RANGE[0])
        & (xyz[:, 1] <= Y_RANGE[1])
        & (xyz[:, 2] >= Z_RANGE[0])
        & (xyz[:, 2] <= Z_RANGE[1])
    )

    return xyz[mask]


def get_stream(trucksc, scene, channel):

    sample = trucksc.get(
        "sample",
        scene["first_sample_token"],
    )

    token = sample["data"][channel]

    out = []

    while token:

        sd = trucksc.get(
            "sample_data",
            token,
        )

        out.append(
            (
                int(sd["timestamp"]),
                token,
            )
        )

        token = sd["next"]

    return out


def nearest(timestamp, stream):

    return min(
        stream,
        key=lambda x: abs(
            x[0] - timestamp
        ),
    )


# =============================================================================
# PERSISTENCE
# =============================================================================

def radar_world(
    trucksc,
    stream,
    idx,
):
    _, token = stream[idx]

    sd = trucksc.get(
        "sample_data",
        token,
    )

    return points_to_world(
        trucksc,
        sd,
        load_radar(sd),
    )


def persistence(
    trucksc,
    stream,
    idx,
):

    xyz = radar_world(
        trucksc,
        stream,
        idx,
    )

    hits = np.zeros(
        len(xyz),
        dtype=int,
    )

    for j in range(
        idx - PERSISTENCE_K,
        idx + PERSISTENCE_K + 1,
    ):

        if j == idx:
            continue

        neighbor = radar_world(
            trucksc,
            stream,
            j,
        )

        d, _ = cKDTree(
            neighbor
        ).query(
            xyz,
            k=1,
        )

        frame_offset = abs(j - idx)
        dt = frame_offset / FPS

        radius = (
            BASE_RADIUS
            + V_MAX * dt
        )

        hits += (
            d <= radius
        ).astype(int)

    return xyz, hits

def soft_persistence_binned_colors(
    score,
    n_bins=N_BARS,
):
    score = np.clip(
        np.asarray(score),
        0.0,
        1.0,
    )

    # Bin edges:
    # N_BARS=4 -> [0, .25, .5, .75, 1]
    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    # Each point -> bin index 0 ... n_bins-1
    bin_idx = np.digitize(
        score,
        edges[1:-1],
        right=False,
    )

    # Representative persistence value for each bin.
    # Use centers for color.
    bin_centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    bin_colors = soft_persistence_colors(
        bin_centers
    )

    # Assign every point exactly its bin's color.
    point_colors = bin_colors[bin_idx]

    return point_colors, bin_idx, edges, bin_centers, bin_colors

def soft_persistence(
    trucksc,
    stream,
    idx,
):

    xyz = radar_world(
        trucksc,
        stream,
        idx,
    )

    score = np.zeros(
        len(xyz),
        dtype=np.float32,
    )

    n_neighbors = 0

    for j in range(
        idx - PERSISTENCE_K,
        idx + PERSISTENCE_K + 1,
    ):

        if j == idx:
            continue

        neighbor = radar_world(
            trucksc,
            stream,
            j,
        )

        d, _ = cKDTree(
            neighbor
        ).query(
            xyz,
            k=1,
        )

        # -----------------------------------------------------
        # lag-dependent motion allowance
        # -----------------------------------------------------
        frame_offset = abs(j - idx)

        dt = frame_offset / FPS

        radius = (
            BASE_RADIUS
            + V_MAX * dt
        )

        sigma = radius / 2.0

        # -----------------------------------------------------
        # Gaussian persistence
        # -----------------------------------------------------
        s = np.exp(
            -(d ** 2)
            / (2.0 * sigma ** 2)
        )

        s[d > radius] = 0.0

        score += s
        n_neighbors += 1

    score /= n_neighbors

    return xyz, score

def persistence_colors(hits):

    total = 2 * PERSISTENCE_K

    colors = np.zeros(
        (len(hits), 3),
        dtype=np.uint8,
    )

    score = hits / total

    low = score <= 0.5

    # red -> yellow
    colors[low, 0] = 255
    colors[low, 1] = (
        255
        * score[low]
        / 0.5
    ).astype(np.uint8)

    # yellow -> green
    colors[~low, 0] = (
        255
        * (1 - score[~low])
        / 0.5
    ).astype(np.uint8)

    colors[~low, 1] = 255

    return colors

def soft_persistence_colors(score):

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    colors = np.zeros(
        (len(score), 3),
        dtype=np.uint8,
    )

    low = score <= 0.5

    # red -> yellow
    colors[low, 0] = 255
    colors[low, 1] = (
        255
        * score[low]
        / 0.5
    ).astype(np.uint8)

    # yellow -> green
    colors[~low, 0] = (
        255
        * (1.0 - score[~low])
        / 0.5
    ).astype(np.uint8)

    colors[~low, 1] = 255

    return colors
def persistence_bins(score):
    score = np.clip(score, 0.0, 1.0)

    edges = np.linspace(
        0.0,
        1.0,
        N_BARS + 1,
    )

    bin_idx = np.digitize(
        score,
        edges[1:-1],
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    colors = soft_persistence_colors(
        centers
    )

    return bin_idx, edges, centers, colors
# =============================================================================
# LOG ONE FRAME
# =============================================================================

def log_frame(
    trucksc,
    radar_stream,
    camera_stream,
    idx,
    pred_by_token,
):

    timestamp, radar_token = (
        radar_stream[idx]
    )

    radar_sd = trucksc.get(
        "sample_data",
        radar_token,
    )

    _, camera_token = nearest(
        timestamp,
        camera_stream,
    )

    camera_sd = trucksc.get(
        "sample_data",
        camera_token,
    )


    # -------------------------------------------------------------------------
    # IMAGE
    # -------------------------------------------------------------------------

    rr.log(
        "image/rgb",
        rr.Image(
            load_image(camera_sd)
        ),
    )


    # -------------------------------------------------------------------------
    # GT + persistence
    # -------------------------------------------------------------------------

    xyz_world, hits = persistence(
        trucksc,
        radar_stream,
        idx,
    ) #xyz_world is in WORLD_COORDINATE frame

    xyz_gt = points_world_to_sensor(
        trucksc,
        camera_sd,
        xyz_world,
    )

    rr.log(
        "gt/points",
        rr.Points3D(
            xyz_gt, #camera-coordinate frame
            colors=persistence_colors(
                hits
            ),
            radii=rr.Radius.ui_points(
                3
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # GT + soft persistence
    # -------------------------------------------------------------------------

    xyz_world_soft, soft_score = soft_persistence(
        trucksc,
        radar_stream,
        idx,
    )

    # -------------------------------------------------------------------------
    # SOFT-PERSISTENCE HISTOGRAM
    # log10(1 + count), count clipped to 100
    # -------------------------------------------------------------------------

    xyz_gt_soft = points_world_to_sensor(
        trucksc,
        camera_sd,
        xyz_world_soft,
    )

    # point_colors, bin_idx, edges, bin_centers, bin_colors = (
    #     soft_persistence_binned_colors(
    #         soft_score,
    #         N_BARS,
    #     )
    # ) 
    bin_idx, edges, centers, bin_colors = persistence_bins(
        soft_score
    )

    hist = np.bincount(
        bin_idx,
        minlength=N_BARS,
    )

    hist_log = np.log10(
        1.0 + np.clip(hist, 0, 100)
    )

    bar_width = 0.9 / N_BARS

    for b in range(N_BARS):

        root = f"soft_bins/bin_{b:02d}_min{edges[b]:.2f}_max{edges[b+1]:.2f}"

        mask = bin_idx == b

        # ---------------------------------------------------------
        # points belonging to this persistence bin
        # ---------------------------------------------------------
        rr.log(
            f"{root}/points",
            rr.Points3D(
                xyz_gt_soft[mask],
                colors=bin_colors[b],
                radii=rr.Radius.ui_points(3),
            ),
        )

        # ---------------------------------------------------------
        # corresponding histogram bar
        # ---------------------------------------------------------
        rr.log(
            f"{root}/bar",
            rr.BarChart(
                values=[hist_log[b]],
                abscissa=[centers[b]],
                color=bin_colors[b],
                widths=bar_width,
            ),
        )




    # -------------------------------------------------------------------------
    # PRED
    # -------------------------------------------------------------------------

    sample_token = radar_sd[
        "sample_token"
    ]

    if sample_token in pred_by_token:

        item = pred_by_token[
            sample_token
        ]

        if isinstance(item, dict):
            pred = item[
                "pred_xyz_camera" 
            ]
        else:
            pred = item

        rr.log(
            "pred/points",
            rr.Points3D(
                pred, #camera-coordinate frame
                colors=[255, 0, 0],
                radii=rr.Radius.ui_points(
                    3
                ),
            ),
        )

    else:

        rr.log(
            "pred/points",
            rr.Clear(
                recursive=True
            ),
        )


    # -------------------------------------------------------------------------
    # TIMESTAMP
    # -------------------------------------------------------------------------

    t_sec = timestamp / 1e6

    text = (
        f"t = {t_sec:.3f} s\n"
        f"radar #{idx}"
    )

    rr.log(
        "image/info",
        rr.TextDocument(text),
    )

    rr.log(
        "gt/info",
        rr.TextDocument(text),
    )

    rr.log(
        "gt_soft/info",
        rr.TextDocument(text),
    )

    rr.log(
        "pred/info",
        rr.TextDocument(text),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    rr.init(
        f"MAN_scene_{SCENE_ID}"
    )

    rr.connect_grpc(
        RERUN_URL
    )

    setup_view()
    log_axes()


    # -------------------------------------------------------------------------
    # Prediction file
    # -------------------------------------------------------------------------

    assert os.path.exists(
        PRED_PATH
    ), PRED_PATH

    with open(
        PRED_PATH,
        "rb",
    ) as f:
        pred_by_token = pickle.load(f)

    print(
        f"predictions: "
        f"{len(pred_by_token)}"
    )


    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    trucksc = TruckScenes(
        VERSION,
        DATA_ROOT,
        False,
    )

    scene = trucksc.scene[
        SCENE_ID
    ]

    radar_stream = get_stream(
        trucksc,
        scene,
        RADAR,
    )

    camera_stream = get_stream(
        trucksc,
        scene,
        CAMERA,
    )

    t0 = radar_stream[0][0]


    # -------------------------------------------------------------------------
    # Native RADAR timeline
    # -------------------------------------------------------------------------

    for idx in range(
        PERSISTENCE_K,
        len(radar_stream)
        - PERSISTENCE_K,
    ):

        timestamp, _ = (
            radar_stream[idx]
        )

        rr.set_time(
            "scene_time",
            duration=(
                timestamp - t0
            ) / 1e6,
        )

        log_frame(
            trucksc,
            radar_stream,
            camera_stream,
            idx,
            pred_by_token,
        )

    print("done")


if __name__ == "__main__":
    main()