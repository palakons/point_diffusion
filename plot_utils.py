
from pytorch3d.loss import chamfer_distance as pt3d_chamfer_distance
import math
import numpy as np

import torch
import torch.nn.functional as F

def plot_pc(
    pc, gt, title="Point Cloud", fname="pc.png", azm=45, progress=None, elev=30
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pc = pc.cpu().numpy()
    gt = gt.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color="blue", label="Noisy", marker="o")
    ax.scatter(
        gt[:, 0], gt[:, 1], gt[:, 2], color="red", label="Ground Truth", marker="^"
    )
    ax.legend()
    # square aspect ratio
    ax.set_box_aspect([1, 1, 1])
    # set limits
    lim = 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # label
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azm)
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cax.barh(0, progress, color="dodgerblue")
        cax.set_xlim(0, 1)
        cax.axis("off")
    plt.savefig(fname)
    plt.close()


def plot_pc_batch(
    pc,
    gt,
    title="Point Cloud Batch",
    fname="pc_batch.png",
    azm=45,
    progress=None,
    elev=30,
    max_cols=4,
    batch_titles=None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    pc = pc.cpu().numpy()
    if gt is not None:
        gt = gt.cpu().numpy()
    batch_size = pc.shape[0]
    has_attr = pc.shape[-1] >= 5 and gt is not None and gt.shape[-1] >= 5
    n_cols = min(max_cols, batch_size)
    n_rows = int(math.ceil(batch_size / n_cols))
    # make sure 3d plot
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )

    for idx in range(batch_size):
        try:
            ax = axs[idx // n_cols, idx % n_cols]
            ax.scatter(
                pc[idx, :, 0],
                pc[idx, :, 1],
                pc[idx, :, 2],
                color="blue",
                label="Noisy",
                marker="o",
            )
            if gt is not None:
                ax.scatter(
                    gt[idx, :, 0],
                    gt[idx, :, 1],
                    gt[idx, :, 2],
                    color="red",
                    label="Ground Truth",
                    marker="^",
                )
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_zlim(-1.0, 1.0)

            cd_string = ""
            if gt is not None:
                if has_attr:

                    # i have to compute the gt-pc correlation, pc_idx, wihci point in pc corresponds to which point in gt,, by find the NN between pc and gt, 
                    # then from this paring, identify corresponding doppler and rcs values, compute error, plot histogram of error, and compute mse loss for doppler and rcs, and add to title

                    pc_idx = np.argmin(
                        np.linalg.norm(
                            pc[idx, :, :3][:, None, :] - gt[idx, :, :3][None, :, :], axis=-1
                        ),
                        axis=1,
                    )

                    doppler_err = pc[idx, :, 3] - gt[idx, pc_idx, 3]
                    rcs_err = pc[idx, :, 4] - gt[idx, pc_idx, 4]
                    xyz_err = np.linalg.norm(pc[idx, :, :3] - gt[idx, pc_idx, :3], axis=-1)
                    position_loss = np.mean(xyz_err**2)
                    
                    minmax_doppler = [-1, 1]
                    minmax_rcs = [-1, 1]
                    bins_doppler = np.linspace(minmax_doppler[0], minmax_doppler[1], 21, endpoint=True)
                    bins_rcs = np.linspace(minmax_rcs[0], minmax_rcs[1], 21, endpoint=True)

                    axins_l = inset_axes(ax, width="38%", height="20%", loc="lower left", borderpad=1)
                    # min max [-.5, .5] for doppler,  
                    axins_l.hist(doppler_err,  #bins=bins_doppler, 
                                alpha=0.75, color="tab:orange", log=True)
                    axins_l.axvline(0.0, color="black", linewidth=1)
                    axins_l.tick_params(axis='both', which='major', labelsize=7)
                    #set ticklabel_format
                    axins_l.ticklabel_format(axis='x', style='plain')
                    # axins_l.set_xticks([])
                    # axins_l.set_yticks([])
                    axins_l.set_title(f"doppler L: {doppler_err.mean():.1e}", fontsize=7)
                    # axins_l.set_title(f"doppler [{minmax_doppler[0]}, {minmax_doppler[1]}] L: {doppler_loss.item():.1e}", fontsize=7)

                    axins_r = inset_axes(ax, width="38%", height="20%", loc="lower right", borderpad=1)
                    #[-1, 1] for rcs,
                    axins_r.hist(rcs_err, #bins=bins_rcs,
                                alpha=0.75, color="tab:green",  log=True)
                    axins_r.axvline(0.0, color="black", linewidth=1)
                    #axis ticks alebls should ahve size 7 font, and in normal number nor "r" format
                    axins_r.tick_params(axis='both', which='major', labelsize=7)
                    axins_r.ticklabel_format(axis='x', style='plain')
                    # axins_r.set_xticks([])
                    # axins_r.set_yticks([])
                    # axins_r.set_title(f"rcs [{minmax_rcs[0]}, {minmax_rcs[1]}] L: {rcs_loss.item():.1e}", fontsize=7)
                    axins_r.set_title(f"rcs L: {rcs_err.mean():.1e}", fontsize=7)

                    #add top-left inset, plot 2d scatter of doppler vs rcs, with limits [-0.5, 0.5] for doppler and [-1, 1] for rcs, one set for pred, another for gt
                    axins_tl = inset_axes(ax, width="38%", height="20%", loc="upper left", borderpad=1)
                    axins_tl.scatter(pc[idx, :, 3], pc[idx, :, 4], alpha=0.75, color="tab:blue", label="Noisy", marker="o", s=10)
                    axins_tl.scatter(gt[idx, :, 3], gt[idx, :, 4], alpha=0.75, color="tab:red", label="GT", marker="^", s=10)


                    # axins_tl.set_xlim(minmax_doppler)
                    # axins_tl.set_ylim(minmax_rcs)
                    # axins_tl.set_xticks([])
                    # axins_tl.set_yticks([])
                    #set x y label
                    axins_tl.set_xlabel("Doppler", fontsize=6)
                    axins_tl.set_ylabel("RCS", fontsize=6)
                    axins_tl.set_title("Doppler vs RCS", fontsize=7)
                    # axins_tl.legend(fontsize=6)


                    # rmse_doppler = np.sqrt(np.mean(doppler_err**2))
                    # rmse_rcs = np.sqrt(np.mean(rcs_err**2))
                    cd = pt3d_chamfer_distance(
                        torch.from_numpy(pc[idx : idx + 1, :, :3]), torch.from_numpy(gt[idx : idx + 1, :, :3])
                    )[0]
                    cd_string = f"CD:{cd.item():.1e} L:{position_loss.item():.1e}"  
                else:
                    cd = pt3d_chamfer_distance(
                        torch.from_numpy(pc[idx : idx + 1]), torch.from_numpy(gt[idx : idx + 1])
                    )[0]
                    cd_string = f"CD: {cd.item():.1e}"
            if batch_titles and len(batch_titles) == batch_size:
                ax.set_title(f"{batch_titles[idx]} {cd_string}", fontsize=10)
            else:
                ax.set_title(cd_string, fontsize=10)
            ax.view_init(elev=elev, azim=azm)
        except Exception as e:
            print(f"Error plotting index {idx}: {e}")

    for idx in range(batch_size, n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis("off")

    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    fig.suptitle(wrapped_title, fontsize=14)

    if batch_size > 0:
        axs[0, 0].legend()
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        axs[0, 0].set_zlabel("Z")
    plt.tight_layout()
    if progress is not None:
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cax.barh(0, progress, color="dodgerblue")
        cax.set_xlim(0, 1)
        cax.axis("off")
    plt.savefig(fname)
    plt.close()

def azm_easing(step, total_steps, style="cosine"):
    # Ease in and out from 0 to 360 degrees
    progress = step / total_steps
    if style == "linear":
        eased = progress
    elif style == "cosine":
        eased = 0.5 - 0.5 * math.cos(progress * math.pi)  # Cosine easing
    elif style == "quadratic":
        eased = (
            2 * progress**2 if progress < 0.5 else 1 - 2 * (1 - progress) ** 2
        )  # Quadratic easing
    elif style == "exponential":
        eased = (
            0.5 * (2 ** (10 * (progress - 1)))
            if progress < 0.5
            else 1 - 0.5 * (2 ** (-10 * progress))
        )  # Exponential easing
    else:
        raise ValueError(f"Unknown easing style: {style}")
    return eased * 360
from pytorch3d.ops import knn_points

def calculate_pointset_stat(
    pc,
    gt,
    bin=20,
    prefix="",
    fscore_thresholds=(0.5, 1.0, 2.0, 3.0, 5.0),
):
    assert pc.device == gt.device, \
        f"Device mismatch: pc on {pc.device}, gt on {gt.device}"

    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix += "_"

    pc_xyz = pc[:, :, :3]
    gt_xyz = gt[:, :, :3]

    cd_xyz = pt3d_chamfer_distance(
        pc_xyz,
        gt_xyz,
    )[0]

    centroid_dist = (
        pc_xyz.mean(dim=1) - gt_xyz.mean(dim=1)
    ).norm(dim=-1).mean()

    # -------- F-score --------
    # PyTorch3D KNN distances are squared L2 distances.
    d_pc_gt = knn_points(
        pc_xyz, gt_xyz, K=1, norm=2
    ).dists[..., 0]

    d_gt_pc = knn_points(
        gt_xyz, pc_xyz, K=1, norm=2
    ).dists[..., 0]

    fscore_data = {}

    for r in fscore_thresholds:
        r2 = r ** 2

        precision_b = (d_pc_gt <= r2).float().mean(dim=1)
        recall_b = (d_gt_pc <= r2).float().mean(dim=1)

        fscore_b = (
            2 * precision_b * recall_b
            / (precision_b + recall_b + 1e-8)
        )

        tag = f"{r:g}m"

        fscore_data.update({
            f"{prefix}precision_{tag}":
                precision_b.mean().item(),
            f"{prefix}recall_{tag}":
                recall_b.mean().item(),
            f"{prefix}fscore_{tag}":
                fscore_b.mean().item(),
        })


    gt_range = np.linalg.norm(gt[:, :, :3].cpu().numpy(), axis=-1)
    gt_range_range = (gt_range.min(), gt_range.max())
    range_hist_error = np.histogram(np.linalg.norm(pc[:, :, :3].cpu().numpy(), axis=-1), bins=bin, range=gt_range_range)[0] - np.histogram(np.linalg.norm(gt[:, :, :3].cpu().numpy(), axis=-1), bins=bin, range=gt_range_range)[0]

    gt_azm = np.arctan2(gt[:, :, 1].cpu().numpy(), gt[:, :, 0].cpu().numpy())
    gt_azm_range = (gt_azm.min(), gt_azm.max())
    azm_hist_error = np.histogram(np.arctan2(pc[:, :, 1].cpu().numpy(), pc[:, :, 0].cpu().numpy()), bins=bin, range=gt_azm_range)[0] - np.histogram(np.arctan2(gt[:, :, 1].cpu().numpy(), gt[:, :, 0].cpu().numpy()), bins=bin, range=gt_azm_range)[0]

    #lets do hist on x-y plane, resolution 50/.15 = 333
    # print(f"shape {pc[:, :, 0].cpu().numpy().shape}") 8,128
    pc_hist = np.histogram2d(pc[:, :, 0].cpu().numpy().flatten(), pc[:, :, 1].cpu().numpy().flatten(), bins=(bin, bin), range=(gt_range_range, gt_azm_range))[0]
    gt_hist = np.histogram2d(gt[:, :, 0].cpu().numpy().flatten(), gt[:, :, 1].cpu().numpy().flatten(), bins=(bin, bin), range=(gt_range_range, gt_azm_range))[0]

    x_y_occupancy_error = pc_hist - gt_hist

    binary_x_y_occupancy_error = (pc_hist > 0).astype(int) - (gt_hist > 0).astype(int)

    #range_hist_error, azm_hist_error, x_y_occupancy_error , range_hist_error  must be summarized to single number, lets do MAE for now
    range_hist_error = np.mean(np.abs(range_hist_error))
    azm_hist_error = np.mean(np.abs(azm_hist_error))
    x_y_occupancy_error = np.mean(np.abs(x_y_occupancy_error))
    binary_x_y_occupancy_error = np.mean(np.abs(binary_x_y_occupancy_error))

    logged_data = {f'{prefix}xyz_cd': cd_xyz.item(), f'{prefix}centroid_error': centroid_dist.item(), f'{prefix}range_hist_error': range_hist_error, f'{prefix}azm_hist_error': azm_hist_error, f'{prefix}x_y_occupancy_error': x_y_occupancy_error, f'{prefix}binary_x_y_occupancy_error': binary_x_y_occupancy_error}
    logged_data.update(fscore_data)
    if pc.shape[-1] >3 and gt.shape[-1] >3:
        gt_doppler_range = (gt[:, :, 3].min().item(), gt[:, :, 3].max().item())
        gt_rcs_range = (gt[:, :, 4].min().item(), gt[:, :, 4].max().item())
        doppler_mae = F.l1_loss(pc[:, :, 3:4].cpu(), gt[:, :, 3:4].to(pc.device).cpu())
        rcs_mae = F.l1_loss(pc[:, :, 4:].cpu(), gt[:, :, 4:].to(pc.device).cpu())
        doppler_hist_error = np.histogram(pc[:, :, 3].cpu().numpy().flatten(), bins=bin, range=gt_doppler_range)[0] - np.histogram(gt[:, :, 3].cpu().numpy().flatten(), bins=bin, range=gt_doppler_range)[0]
        rcs_hist_error = np.histogram(pc[:, :, 4].cpu().numpy().flatten(), bins=bin, range=gt_rcs_range)[0] - np.histogram(gt[:, :, 4].cpu().numpy().flatten(), bins=bin, range=gt_rcs_range)[0]
        # doppler_hist_error, rcs_hist_error need to be sumarized to single number, MSE? MAE? RMSE? lets do MAE for now
        doppler_hist_error = np.mean(np.abs(doppler_hist_error))
        rcs_hist_error = np.mean(np.abs(rcs_hist_error))
        
        doppler_sign_accuracy = ((pc[:, :, 3] > 0).int() == (gt[:, :, 3] > 0).int().to(pc.device)).float().mean().item()
        # print(f"min max doppler {pc[:, :, 3].min().item(), pc[:, :, 3].max().item()}, {gt[:, :, 3].min().item(), gt[:, :, 3].max().item()}")

        logged_data.update({f'{prefix}doppler_mae': doppler_mae.item(), f'{prefix}rcs_mae': rcs_mae.item(), f'{prefix}doppler_hist_mae': doppler_hist_error, f'{prefix}rcs_hist_mae': rcs_hist_error, f'{prefix}doppler_sign_accuracy': doppler_sign_accuracy})
    else:
        logged_data.update({f'{prefix}doppler_mae': None, f'{prefix}rcs_mae': None, f'{prefix}doppler_hist_mae': None, f'{prefix}rcs_hist_mae': None, f'{prefix}doppler_sign_accuracy': None})

    return logged_data

