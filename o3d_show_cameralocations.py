import sys, os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def get_colors(inp, colormap, vmin=None, vmax=None):
    if vmin == None:
        vmin = np.nanmin(inp)
    if vmax == None:
        vmax = np.nanmax(inp)
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def plot_4attributes():
    # make 4 panel plot showing color scatters for variance, reprojection_error, reconstruction_uncertainty, image_count
    fg, ax = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True, dpi=300)
    im0 = ax[0, 0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=tiepoints_array[:, 4],
        s=0.1,
        vmin=np.percentile(tiepoints_array[:, 4], 2),
        vmax=np.percentile(tiepoints_array[:, 4], 98),
        cmap=plt.cm.viridis,
    )
    # ax[0,0].set_xlabel('X Coordinate [m]')
    ax[0, 0].set_ylabel("Y Coordinate [m]")
    h0 = plt.colorbar(im0, ax=ax[0, 0])
    h0.set_label("Variance")

    im1 = ax[0, 1].scatter(
        xy[:, 0],
        xy[:, 1],
        c=tiepoints_array[:, 8],
        s=0.1,
        vmin=np.percentile(tiepoints_array[:, 8], 2),
        vmax=np.percentile(tiepoints_array[:, 8], 98),
        cmap=plt.cm.magma,
    )
    # ax[0,1].set_xlabel('X Coordinate [m]')
    # ax[0,1].set_ylabel('Y Coordinate [m]')
    h1 = plt.colorbar(im1, ax=ax[0, 1])
    h1.set_label("Reprojection error [pixels]")

    im2 = ax[1, 0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=tiepoints_array[:, 9],
        s=0.1,
        vmin=np.percentile(tiepoints_array[:, 9], 2),
        vmax=np.percentile(tiepoints_array[:, 9], 98),
        cmap=plt.cm.plasma,
    )
    ax[1, 0].set_xlabel("X Coordinate [m]")
    ax[1, 0].set_ylabel("Y Coordinate [m]")
    h2 = plt.colorbar(im2, ax=ax[1, 0])
    h2.set_label("Reconstruction uncertainty")

    im3 = ax[1, 1].scatter(
        xy[:, 0],
        xy[:, 1],
        c=tiepoints_array[:, 10],
        s=0.1,
        vmin=3,  # np.percentile(tiepoints_array[:,10],2),
        vmax=12,  # np.percentile(tiepoints_array[:,10],98),
        cmap=plt.cm.tab10,
    )
    ax[1, 1].set_xlabel("X Coordinate [m]")
    # ax[1,0].set_ylabel('Y Coordinate [m]')
    h3 = plt.colorbar(im3, ax=ax[1, 1])
    h3.set_label("Image-pair count")

    fg.suptitle("Tiepoint attributes for %s" % csv_fname.split(".")[0], fontsize=18)
    fg.tight_layout()
    fg.savefig(csv_fname.split(".")[0] + ".png", dpi=300)


csv_fname = sys.argv[1]

# read data and determine number of columns
# with open(csv_fname) as f:
#    ncols = len(f.readline().split(","))
f = open(csv_fname)
lines = f.readlines()
ncols = len(lines[5].split(","))
f.close()

camera_data = np.loadtxt(csv_fname, delimiter=",", skiprows=2, usecols=range(1, ncols))
# X_est,Y_est,Z_est,Yaw_est,Pitch_est,Roll_est,X_var,Y_var,Z_var,Yaw_var,Pitch_var,Roll_var

camera_labels = np.genfromtxt(
    csv_fname, delimiter=",", skip_header=2, dtype="str", usecols=0
)

# xy = camera_data[:, 0:2]
# plot_4attributes()

# create open3d point clouds
xyz = camera_data[:, 0:3]
xyz_var = np.sqrt(
    camera_data[:, 3] ** 2 + camera_data[:, 4] ** 2 + camera_data[:, 5] ** 2
)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
rgb = get_colors(
    xyz_var,
    plt.cm.viridis,
    vmin=np.percentile(xyz_var, 2),
    vmax=np.percentile(xyz_var, 98),
)
pcd.colors = o3d.utility.Vector3dVector(rgb[:, 0:3])
o3d.io.write_point_cloud(
    csv_fname.split(".")[0] + "_variance.ply", pcd, compressed=True
)
# o3d.visualization.draw_geometries([pcd])
print("pcd:", pcd)
