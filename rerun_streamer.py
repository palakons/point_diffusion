# rerun_streamer.py

import os

import numpy as np
import pypcd4
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import rerun as rr
from truckscenes import TruckScenes
import rerun.blueprint as rrb
from collections import defaultdict
import torch
from pytorch3d.loss import chamfer_distance as cd_pt3d_l2sq_m2


def chamfer_pt3d_sq_m2(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return np.nan

    P = torch.as_tensor(P, dtype=torch.float32).unsqueeze(0)
    Q = torch.as_tensor(Q, dtype=torch.float32).unsqueeze(0)

    cd, _ = cd_pt3d_l2sq_m2(P, Q)

    return float(cd)

CD_RESULTS = defaultdict(
    lambda: defaultdict(
        lambda: {
            "cd_m": [],
            "pt3d_cd_sq_m2": [],
            "retain_pct": [],
        }
    )
)

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = "/data/palakons/new_dataset/MAN/man-truckscenes"
VERSION = "v1.0-trainval"

SCENE_ID = 3

RERUN_URL = "rerun+http://10.204.190.135:9876/proxy"

RADARS = [
    # "RADAR_RIGHT_BACK",
    # "RADAR_RIGHT_SIDE",
    "RADAR_RIGHT_FRONT",
    "RADAR_LEFT_FRONT",
    # "RADAR_LEFT_SIDE",
    # "RADAR_LEFT_BACK",
]

CAMERAS = [
    "CAMERA_LEFT_FRONT",
    # "CAMERA_LEFT_BACK",
    "CAMERA_RIGHT_FRONT",
    # "CAMERA_RIGHT_BACK",
]

CHANNELS = RADARS + CAMERAS

# Compare against K previous + K future sweeps.
PERSISTENCE_K = 2

# Return in another sweep counts as persistent if within this distance.
PERSISTENCE_RADIUS = 0.5

# =============================================================================
# RADAR ROI
# =============================================================================

X_RANGE = [0, 50]       # forward [m]
Y_RANGE = [-50, 50]     # lateral [m]
Z_RANGE = [-2, 2]       # vertical [m]



def setup_views():
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    # =========================================================
                    # LEFT: 3D radar scene
                    # =========================================================
                    rrb.Spatial3DView(
                        name="MAN 3D Scene",
                        origin="/world",
                    ),

                    # =========================================================
                    # RIGHT
                    # =========================================================
                    rrb.Vertical(
                        contents=[
                            # Dashcams
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(
                                        name="Left Front Camera",
                                        origin="/world/CAMERA_LEFT_FRONT/image",
                                    ),

                                    rrb.Spatial2DView(
                                        name="Right Front Camera",
                                        origin="/world/CAMERA_RIGHT_FRONT/image",
                                    ),
                                ]
                            ),

                            # CD plots
                            *[
                                rrb.TimeSeriesView(
                                    name=f"{channel} — persistence CD",
                                    origin=f"/metrics/cd/{channel}",
                                    axis_y=rrb.ScalarAxis(
                                        range=(0.0, 3.0),
                                        zoom_lock=True,
                                    ),
                                    plot_legend=rrb.PlotLegend(
                                        visible=True,
                                    ),
                                )
                                for channel in RADARS
                            ],
                        ]
                    ),
                ],

                # 3D gets ~2/3 of width
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )
    )
def setup_plots():
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                contents=[
                    rrb.TimeSeriesView(
                        name=f"{channel} — persistence CD",
                        origin=f"/metrics/cd/{channel}",
                        axis_y=rrb.ScalarAxis(
                            range=(0.0, 3.0),
                            zoom_lock=True,
                        ),
                        plot_legend=rrb.PlotLegend(
                            visible=True,
                        ),
                    )
                    for channel in RADARS
                ]
            )
        )
    )
def chamfer_m(P, Q):
    """Symmetric nearest-neighbor Chamfer, in metres."""
    if len(P) == 0 or len(Q) == 0:
        return np.nan

    d_pq, _ = cKDTree(Q).query(P, k=1)
    d_qp, _ = cKDTree(P).query(Q, k=1)

    return 0.5 * (d_pq.mean() + d_qp.mean())

def print_cd_summary():

    print("\n" + "=" * 100)
    print("CONSECUTIVE REAL↔REAL RADAR REPEATABILITY")
    print("=" * 100)

    for channel, levels in CD_RESULTS.items():

        print(f"\n{channel}")

        print(
            f"{'level':12s} "
            f"{'retain%':>8s} "
            f"{'P25[m]':>9s} "
            f"{'P50[m]':>9s} "
            f"{'P75[m]':>9s} "
            f"{'P90[m]':>9s} "
            f"{'IQR[m]':>9s} "
            f"{'PT3D P50[m²]':>15s}"
        )

        for label, result in levels.items():

            cd = summarize(result["cd_m"])
            cd_sq = summarize(result["pt3d_cd_sq_m2"])

            if cd is None:
                continue

            retain = np.median(
                result["retain_pct"]
            )

            print(
                f"{label:12s} "
                f"{retain:8.1f} "
                f"{cd['p25']:9.3f} "
                f"{cd['p50']:9.3f} "
                f"{cd['p75']:9.3f} "
                f"{cd['p90']:9.3f} "
                f"{cd['iqr']:9.3f} "
                f"{cd_sq['p50']:15.4f}"
            )

def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    p25, p50, p75, p90 = np.percentile(
        values,
        [25, 50, 75, 90],
    )

    return {
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "iqr": p75 - p25,
        "n": len(values),
    }

def log_persistence_cd(
    trucksc,
    channel,
    stream,
    frame_idx,
):
    if (
        frame_idx < PERSISTENCE_K
        or frame_idx + 1 >= len(stream) - PERSISTENCE_K
    ):
        return

    xyz0, hits0 = radar_frame_with_persistence(
        trucksc, stream, frame_idx
    )

    xyz1, hits1 = radar_frame_with_persistence(
        trucksc, stream, frame_idx + 1
    )

    if xyz0 is None or xyz1 is None:
        return

    total = 2 * PERSISTENCE_K

    # threshold 0 = ALL
    for min_hits in range(total + 1):

        if min_hits == 0:
            label = "all"
            P = xyz0
            Q = xyz1
        else:
            label = f"ge_{min_hits}_of_{total}"
            P = xyz0[hits0 >= min_hits]
            Q = xyz1[hits1 >= min_hits]

        if len(P) == 0 or len(Q) == 0:
            continue

        cd_m = chamfer_m(P, Q)
        cd_sq = chamfer_pt3d_sq_m2(P, Q)

        # Average retained fraction across the two compared sweeps.
        retain_pct = 50.0 * (
            len(P) / len(xyz0)
            + len(Q) / len(xyz1)
        )

        # Store for final statistics.
        result = CD_RESULTS[channel][label]

        result["cd_m"].append(cd_m)
        result["pt3d_cd_sq_m2"].append(cd_sq)
        result["retain_pct"].append(retain_pct)

        # Rerun plots.
        rr.log(
            f"metrics/cd_m/{channel}/{label}",
            rr.Scalars(cd_m),
        )

        rr.log(
            f"metrics/cd_sq_m2/{channel}/{label}",
            rr.Scalars(cd_sq),
        )
def filter_xyz(
    xyz,
    x_range=None,
    y_range=None,
    z_range=None,
):
    mask = np.ones(len(xyz), dtype=bool)

    for dim, limits in enumerate([x_range, y_range, z_range]):
        if limits is not None:
            mask &= (
                (xyz[:, dim] >= limits[0])
                & (xyz[:, dim] <= limits[1])
            )

    return xyz[mask]

# =============================================================================
# BASIC DATASET FUNCTIONS
# =============================================================================

def quat_to_R(q):
    """MAN [w,x,y,z] -> 3x3 rotation matrix."""
    return Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]
    ).as_matrix()


def load_points(sd):
    pc = pypcd4.PointCloud.from_path(
        os.path.join(DATA_ROOT, sd["filename"])
    ).pc_data

    xyz = np.stack(
        [pc["x"], pc["y"], pc["z"]],
        axis=1,
    ).astype(np.float64)

    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    return filter_xyz(
        xyz,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
    )


def load_image(sd):
    return np.asarray(
        Image.open(
            os.path.join(DATA_ROOT, sd["filename"])
        ).convert("RGB")
    )


def sensor_to_world(trucksc, sd):
    """
    Return transform:

        p_world = R @ p_sensor + t
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
    t_calib = np.asarray(calib["translation"], dtype=np.float64)

    R_pose = quat_to_R(pose["rotation"])
    t_pose = np.asarray(pose["translation"], dtype=np.float64)

    R = R_pose @ R_calib
    t = R_pose @ t_calib + t_pose

    return R, t, calib


def points_to_world(trucksc, sd, xyz):
    R, t, _ = sensor_to_world(trucksc, sd)
    return xyz @ R.T + t


def get_stream(trucksc, scene, channel):
    """All native sample_data frames for one sensor."""

    sample = trucksc.get(
        "sample",
        scene["first_sample_token"],
    )

    token = sample["data"][channel]
    stream = []

    while token:
        sd = trucksc.get("sample_data", token)

        stream.append(
            (int(sd["timestamp"]), channel, token)
        )

        token = sd["next"]

    return stream


# =============================================================================
# PERSISTENCE
# =============================================================================

def persistence_scores(current, neighbors):
    """
    For every current point:

        score =
        fraction of neighboring sweeps containing a point
        within PERSISTENCE_RADIUS.

    All inputs are already in WORLD coordinates.
    """

    hits = np.zeros(len(current), dtype=np.int32)

    for neighbor in neighbors:
        tree = cKDTree(neighbor)

        distance, _ = tree.query(
            current,
            k=1,
        )

        hits += distance <= PERSISTENCE_RADIUS

    return hits


def radar_frame_with_persistence(
    trucksc,
    stream,
    frame_idx,
):
    """
    Returns:
        xyz_world
        hits

    hits ranges:
        0 .. 2*K

    Scene-edge frames are skipped so every point always has
    exactly 2*K comparison sweeps.
    """

    K = PERSISTENCE_K

    if frame_idx < K or frame_idx >= len(stream) - K:
        return None, None

    _, _, token = stream[frame_idx]

    sd = trucksc.get("sample_data", token)

    xyz = points_to_world(
        trucksc,
        sd,
        load_points(sd),
    )

    neighbors = []

    for j in range(frame_idx - K, frame_idx + K + 1):

        if j == frame_idx:
            continue

        _, _, neighbor_token = stream[j]

        neighbor_sd = trucksc.get(
            "sample_data",
            neighbor_token,
        )

        neighbor_xyz = points_to_world(
            trucksc,
            neighbor_sd,
            load_points(neighbor_sd),
        )

        neighbors.append(neighbor_xyz)

    hits = persistence_scores(
        xyz,
        neighbors,
    )

    return xyz, hits


def persistence_color(hits, total):
    """
    0/total     -> red
    total/2     -> yellow
    total/total -> green
    """

    score = hits / total

    if score <= 0.5:
        return [
            255,
            int(255 * score / 0.5),
            0,
        ]

    return [
        int(255 * (1.0 - score) / 0.5),
        255,
        0,
    ]


# =============================================================================
# CAMERA
# =============================================================================

def setup_camera_pinhole(trucksc, channel, sd):
    """Camera intrinsic parameters are static."""

    _, _, calib = sensor_to_world(
        trucksc,
        sd,
    )

    K = np.asarray(
        calib["camera_intrinsic"],
        dtype=np.float64,
    )

    with Image.open(
        os.path.join(DATA_ROOT, sd["filename"])
    ) as img:
        width, height = img.size

    rr.log(
        f"world/{channel}/image",
        rr.Pinhole(
            image_from_camera=K,
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
        static=True,
    )


def log_camera(trucksc, channel, sd):
    """
    Camera itself moves directly in WORLD coordinates.
    """

    R, t, _ = sensor_to_world(
        trucksc,
        sd,
    )

    rr.log(
        f"world/{channel}",
        rr.Transform3D(
            translation=t,
            mat3x3=R,
        ),
    )

    rr.log(
        f"world/{channel}/image",
        rr.Image(load_image(sd)),
    )


# =============================================================================
# RADAR
# =============================================================================

def log_radar(
    trucksc,
    channel,
    stream,
    frame_idx,
):
    xyz_world, hits = radar_frame_with_persistence(
        trucksc,
        stream,
        frame_idx,
    )

    if xyz_world is None:
        return

    total = 2 * PERSISTENCE_K

    # Separate entity per persistence level.
    # Rerun lets you show/hide each independently.
    for n_hits in range(total + 1):

        mask = hits == n_hits

        rr.log(
            f"world/radar/{channel}/"
            f"persistence_{n_hits}_of_{total}",
            rr.Points3D(
                xyz_world[mask],
                colors=persistence_color(
                    n_hits,
                    total,
                ),
                radii=rr.Radius.ui_points(2.5),
            ),
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    rr.init(
        f"MAN_scene_{SCENE_ID}_persistence_filtered_chart",
    )

    rr.connect_grpc(RERUN_URL)





    setup_views()

    trucksc = TruckScenes(
        VERSION,
        DATA_ROOT,
        False,
    )

    scene = trucksc.scene[SCENE_ID]

    # -------------------------------------------------------------------------
    # Build sensor streams
    # -------------------------------------------------------------------------

    streams = {
        channel: get_stream(
            trucksc,
            scene,
            channel,
        )
        for channel in CHANNELS
    }

    for channel, stream in streams.items():
        print(
            f"{channel:22s}: {len(stream)} frames"
        )

    # Fast token -> frame index lookup for radar persistence.
    radar_index = {
        channel: {
            token: i
            for i, (_, _, token) in enumerate(stream)
        }
        for channel, stream in streams.items()
        if channel.startswith("RADAR_")
    }

    # -------------------------------------------------------------------------
    # Camera intrinsics
    # -------------------------------------------------------------------------

    for channel in CAMERAS:

        first_token = streams[channel][0][2]

        first_sd = trucksc.get(
            "sample_data",
            first_token,
        )


        print(f"with sd {first_sd}, checking camera intrinsics, should be same for all frames in a stream")
        calib = trucksc.get(
            "calibrated_sensor",
            first_sd["calibrated_sensor_token"],
        )

        print(channel)
        print(calib)

        setup_camera_pinhole(
            trucksc,
            channel,
            first_sd,
        )

    # -------------------------------------------------------------------------
    # Merge everything chronologically
    # -------------------------------------------------------------------------

    events = [
        event
        for stream in streams.values()
        for event in stream
    ]

    events.sort(key=lambda x: x[0])

    t0 = events[0][0]

    # -------------------------------------------------------------------------
    # Stream
    # -------------------------------------------------------------------------

    for i, (timestamp, channel, token) in enumerate(events):

        rr.set_time(
            "scene_time",
            duration=(timestamp - t0) / 1e6,
        )

        sd = trucksc.get(
            "sample_data",
            token,
        )
        if channel.startswith("RADAR_"):

            frame_idx = radar_index[channel][token]

            log_radar(
                trucksc,
                channel,
                streams[channel],
                frame_idx,
            )

            log_persistence_cd(
                trucksc,
                channel,
                streams[channel],
                frame_idx,
            )

        else:

            log_camera(
                trucksc,
                channel,
                sd,
            )

        if i % 100 == 0:
            print(
                f"{i:5d}/{len(events):5d} "
                f"{(timestamp - t0)/1e6:6.2f}s "
                f"{channel}"
            )
    # setup_plots()
    print_cd_summary()
    print("done")


if __name__ == "__main__":
    main()