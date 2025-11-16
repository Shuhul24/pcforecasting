import os
import math
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

def range_projection(
    current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
):
    """Project a pointcloud into a spherical projection, range image.

    Args:
      current_vertex: raw point clouds

    Returns:
      proj_range: projected range image with depth, each pixel contains the corresponding depth
      proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
      proj_intensity: each pixel contains the corresponding intensity
      proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # print(fov_up, fov_down, max_range, proj_H, proj_W)
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    # # we use a maximum range threshold
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # print(np.degrees(np.max(pitch)), np.degrees(np.min(pitch)))

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full(
        (proj_H, proj_W), -1, dtype=np.float32
    )  # [H,W] range (-1 is no data)
    proj_vertex = np.full(
        (proj_H, proj_W, 4), -1, dtype=np.float32
    )  # [H,W] index (-1 is no data)
    proj_idx = np.full(
        (proj_H, proj_W), -1, dtype=np.int32
    )  # [H,W] index (-1 is no data)
    proj_intensity = np.full(
        (proj_H, proj_W), -1, dtype=np.float32
    )  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]
    ).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx


class projection:
    """Projection class for getting a 3D point cloud from range images"""

    def __init__(self, cfg):
        """Init

        Args:
            cfg (dict): Parameters
        """
        self.cfg = cfg

        fov_up = (
            self.cfg["DATA_CONFIG"]["FOV_UP"] / 180.0 * np.pi
        )  # field of view up in radians
        fov_down = (
            self.cfg["DATA_CONFIG"]["FOV_DOWN"] / 180.0 * np.pi
        )  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radian
        W = self.cfg["DATA_CONFIG"]["WIDTH"]
        H = self.cfg["DATA_CONFIG"]["HEIGHT"]

        h = torch.arange(0, H).view(H, 1).repeat(1, W)
        w = torch.arange(0, W).view(1, W).repeat(H, 1)
        yaw = np.pi * (1.0 - 2 * torch.true_divide(w, W))
        pitch = (1.0 - torch.true_divide(h, H)) * fov - abs(fov_down)
        self.x_fac = torch.cos(pitch) * torch.cos(yaw)
        self.y_fac = torch.cos(pitch) * torch.sin(yaw)
        self.z_fac = torch.sin(pitch)

    def get_valid_points_from_range_view(self, range_view, use_batch = False):
        """Reproject from range image to valid 3D points

        Args:
            range_view (torch.tensor): Range image with size (H,W)

        Returns:
            torch.tensor: Valid 3D points with size (N,3)
        """

        if use_batch:
            B, H, W = range_view.shape
            # points = torch.full((B,H,W,3), 1e3).type_as(range_view)
            points = torch.zeros(B, H, W, 3).type_as(range_view)
            points[:, :, :, 0] = range_view * self.x_fac.type_as(range_view)
            points[:, :, :, 1] = range_view * self.y_fac.type_as(range_view)
            points[:, :, :, 2] = range_view * self.z_fac.type_as(range_view)

            points[range_view < 0,:] = torch.tensor([1e3 ,1e3 ,1e3], dtype=torch.float32).to('cuda')

            return points
        
        H, W = range_view.shape
        points = torch.zeros(H, W, 3).type_as(range_view)
        points[:, :, 0] = range_view * self.x_fac.type_as(range_view)
        points[:, :, 1] = range_view * self.y_fac.type_as(range_view)
        points[:, :, 2] = range_view * self.z_fac.type_as(range_view)
        return points[range_view > 0.0]


    def get_mask_from_output(self, output):
        """Get mask from logits

        Args:
            output (dict): Output dict with mask_logits as key

        Returns:
            mask: Predicted mask containing per-point probabilities
        """
        mask = nn.Sigmoid()(output["mask_logits"])
        return mask

    def get_target_mask_from_range_view(self, range_view):
        """Ground truth mask

        Args:
            range_view (torch.tensor): Range image of size (H,W)

        Returns:
            torch.tensor: Target mask of valid points
        """
        target_mask = torch.zeros(range_view.shape).type_as(range_view)
        target_mask[range_view != -1.0] = 1.0
        return target_mask

    def get_masked_range_view(self, output):
        """Get predicted masked range image

        Args:
            output (dict): Dictionary containing predicted mask logits and ranges

        Returns:
            torch.tensor: Maskes range image in which invalid points are mapped to -1.0
        """
        ##Changed this part
        mask = self.get_mask_from_output(output)
        masked_range_view = output["rv"].clone()

        # Set invalid points to -1.0 according to mask
        masked_range_view[mask < self.cfg["MODEL"]["MASK_THRESHOLD"]] = -1.0
        return masked_range_view


def plot_two_point_clouds(pcd1, pcd2):
    """
    Plots two Open3D point clouds in a single interactive visualization window.

    This function sets different colors for each point cloud to make them
    distinguishable and then uses the Open3D visualization utility to display
    them together.

    Args:
        pcd1 (o3d.geometry.PointCloud): The first point cloud to plot.
        pcd2 (o3d.geometry.PointCloud): The second point cloud to plot.
    """
    # Check if both point clouds have points before proceeding
    if not pcd1.has_points() and not pcd2.has_points():
        print("Both point clouds are empty. Nothing to visualize.")
        return
    elif not pcd1.has_points():
        print("First point cloud is empty. Visualizing only the second one.")
        o3d.visualization.draw_geometries([pcd2], window_name="Single Point Cloud")
        return
    elif not pcd2.has_points():
        print("Second point cloud is empty. Visualizing only the first one.")
        o3d.visualization.draw_geometries([pcd1], window_name="Single Point Cloud")
        return

    # Assign different colors to make them distinguishable
    # Set the first point cloud to blue
    pcd1.paint_uniform_color([0, 0, 1])  # RGB color for blue

    # Set the second point cloud to red
    pcd2.paint_uniform_color([1, 0, 0])  # RGB color for red

    # Create a list containing both point cloud objects
    geometries_to_plot = [pcd1, pcd2]

    # Visualize the list of geometries. Open3D will handle displaying them
    # in the same window.
    o3d.visualization.draw_geometries(geometries_to_plot, window_name="Two Point Clouds")

    print("Visualization window closed.")