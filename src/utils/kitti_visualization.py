#!/usr/bin/env python3
"""
KITTI Point Cloud and Range Image Visualization Script with Video Generation and Pose Integration
Generates both 2D range image visualizations and 3D point cloud visualizations
Uses matplotlib for headless server compatibility
Modified to show only BEV and 3D view with red points
Added video generation functionality at 10 fps
Added pose loading and transformation functionality
Added multi-frame visualization with different colors
Updated with tqdm for better progress tracking and reduced verbosity
Updated to include proper LiDAR-to-Camera transformation chain
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import open3d as o3d
import cv2
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
matplotlib.use('Agg')  # Use non-interactive backend for headless servers

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_poses(poses_path, sequence, dataset_path=None):
    """
    Load poses from KITTI poses file
    
    Args:
        poses_path: Path to poses directory or specific poses file
        sequence: Sequence number (e.g., "00")
        dataset_path: Path to KITTI dataset root (alternative location)
    
    Returns:
        poses: List of 4x4 transformation matrices
    """
    # Try multiple possible locations for pose files
    possible_paths = []
    
    # Standard KITTI poses directory structure
    if poses_path:
        possible_paths.append(os.path.join(poses_path, f"{sequence}.txt"))
    
    # KITTI dataset sequences structure
    if dataset_path:
        possible_paths.extend([
            os.path.join(dataset_path, "sequences", sequence, "poses.txt"),
            os.path.join(dataset_path, "dataset", "sequences", sequence, "poses.txt")
        ])
    
    # If poses_path looks like a full dataset path, try sequences subdirectory
    if poses_path and "kitti" in poses_path.lower():
        possible_paths.extend([
            os.path.join(poses_path, "sequences", sequence, "poses.txt"),
            os.path.join(poses_path, "dataset", "sequences", sequence, "poses.txt")
        ])

    pose_file = None
    for path in possible_paths:
        if os.path.exists(path):
            pose_file = path
            print(f"✓ Found poses at: {pose_file}")
            break
    
    if pose_file is None:
        print(f"⚠ Pose file not found. Tried locations:")
        for path in possible_paths:
            print(f"    {path}")
        return None
    
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            # Parse the 12 values from each line
            values = list(map(float, line.strip().split()))
            if len(values) != 12:
                continue
            
            # Reshape into 3x4 matrix and add bottom row [0, 0, 0, 1]
            pose_3x4 = np.array(values).reshape(3, 4)
            pose_4x4 = np.vstack([pose_3x4, [0, 0, 0, 1]])
            poses.append(pose_4x4)
    
    return poses

def load_calibration(dataset_path, sequence):
    """
    Load calibration data from sequence directory.
    
    Args:
        dataset_path: Path to KITTI dataset
        sequence: Sequence number (e.g., "00")
    
    Returns:
        calib: Dictionary containing calibration matrices
    """
    # Try multiple possible locations for calibration files
    possible_paths = [
        Path(dataset_path) / 'sequences' / sequence / 'calib.txt',
        Path(dataset_path) / 'dataset' / 'sequences' / sequence / 'calib.txt'
    ]
    
    calib_file = None
    for path in possible_paths:
        if path.exists():
            calib_file = path
            print(f"✓ Found calibration at: {calib_file}")
            break
    
    if calib_file is None:
        print(f"⚠ Calibration file not found. Tried locations:")
        for path in possible_paths:
            print(f"    {path}")
        return None
    
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, *values = line.strip().split()
                calib[key.rstrip(':')] = np.array([float(v) for v in values])
    
    # Velodyne to camera transforms
    if 'Tr' in calib:
        # Transform from velodyne to cam0
        T_cam0_velo = np.vstack([calib['Tr'].reshape(3, 4), [0, 0, 0, 1]])
        calib['T_cam0_velo'] = T_cam0_velo
    else:
        print(f"⚠ 'Tr' transformation not found in calibration file")
        # Use identity transformation as fallback
        calib['T_cam0_velo'] = np.eye(4)
    
    return calib

def transform_points(points, pose, T_cam_velo=None):
    """
    Transform points using the proper chain: LiDAR → Camera → World
    
    Args:
        points: Nx4 array (x, y, z, intensity)
        pose: 4x4 camera-to-world transformation matrix
        T_cam_velo: 4x4 LiDAR-to-camera transformation matrix
    
    Returns:
        transformed_points: Nx4 array with transformed coordinates
    """
    if points.shape[0] == 0:
        return points
    
    # Step 1: Transform from LiDAR to camera coordinates (if calibration available)
    if T_cam_velo is not None:
        # Convert to homogeneous coordinates
        points_homo = np.column_stack([points[:, :3], np.ones(points.shape[0])])
        
        # Transform LiDAR → Camera
        points_cam = (T_cam_velo @ points_homo.T).T
    else:
        # If no calibration, assume points are already in camera coordinates
        points_cam = np.column_stack([points[:, :3], np.ones(points.shape[0])])
    
    # Step 2: Transform from camera to world coordinates
    points_world = (pose @ points_cam.T).T
    
    # Return x, y, z, intensity (keep original intensity)
    return np.column_stack([points_world[:, :3], points[:, 3]])

def load_range_data(filepath):
    """Load range data from .npy file"""
    return np.load(filepath)

def load_xyz_data(filepath):
    """Load XYZ data from .npy file (projected coordinates)"""
    return np.load(filepath)

def load_intensity_data(filepath):
    """Load intensity data from .npy file"""
    return np.load(filepath)

def normalize_data(data, percentile_clip=None):
    """Normalize data to [0, 1] range with optional percentile clipping"""
    if percentile_clip:
        lower, upper = np.percentile(data, [percentile_clip, 100 - percentile_clip])
        data = np.clip(data, lower, upper)
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)

def create_custom_colormap():
    """Create a custom colormap for better visualization"""
    colors = ['#ffffff', '#ff0000', '#ff5500', '#ffff00', '#55ff00',
              '#00ffff', '#0055ff', '#0000ff', '#000055', '#000033']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def range_image_to_pointcloud(proj_range, proj_xyz, proj_intensity, max_range=80.0):
    """
    Convert range image back to 3D point cloud
    
    Args:
        proj_range: Range image (H, W)
        proj_xyz: XYZ coordinates (H, W, 3 or 4)
        proj_intensity: Intensity values (H, W)
        max_range: Maximum range value
    
    Returns:
        points: Nx4 array (x, y, z, intensity)
    """
    H, W = proj_range.shape
    
    # Create mask for valid points (range > 0)
    valid_mask = proj_range > 0
    
    # Extract XYZ coordinates (only first 3 channels)
    xyz_coords = proj_xyz[:, :, :3] if proj_xyz.shape[2] >= 3 else proj_xyz
    
    # Extract valid points
    valid_xyz = xyz_coords[valid_mask]
    valid_intensity = proj_intensity[valid_mask]
    valid_range = proj_range[valid_mask]
    
    # Filter out points that are too far or invalid
    range_mask = (valid_range > 0) & (valid_range < max_range)
    
    points_xyz = valid_xyz[range_mask]
    points_intensity = valid_intensity[range_mask]
    
    # Combine xyz and intensity
    points = np.column_stack([points_xyz, points_intensity])
    
    return points

def load_frame_pointcloud(data_path, sequence, frame_idx, max_range=80.0):
    """
    Load point cloud data for a specific frame
    
    Args:
        data_path: Path to processed data
        sequence: Sequence number
        frame_idx: Frame index
        max_range: Maximum range value
    
    Returns:
        points: Nx4 array (x, y, z, intensity) or None if failed
    """
    frame_str = str(frame_idx).zfill(6)
    range_file = os.path.join(data_path, sequence, "processed", "range", f"{frame_str}.npy")
    xyz_file = os.path.join(data_path, sequence, "processed", "xyz", f"{frame_str}.npy")
    intensity_file = os.path.join(data_path, sequence, "processed", "intensity", f"{frame_str}.npy")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [range_file, xyz_file, intensity_file]):
        return None
    
    try:
        # Load data
        range_data = load_range_data(range_file)
        xyz_data = load_xyz_data(xyz_file)
        intensity_data = load_intensity_data(intensity_file)
        
        # Convert to point cloud
        points = range_image_to_pointcloud(range_data, xyz_data, intensity_data, max_range=max_range)
        return points
    except Exception as e:
        return None

def create_multi_frame_visualization(points_list, colors, labels, output_path, title="Multi-Frame Point Cloud"):
    """
    Create 3D point cloud visualization with multiple frames in different colors using Open3D
    
    Args:
        points_list: List of Nx4 arrays (x, y, z, intensity)
        colors: List of RGB color tuples for each frame
        labels: List of labels for each frame
        output_path: Path to save the image
        title: Title for the visualization
    """
    if not points_list or all(points.shape[0] == 0 for points in points_list):
        return False
    
    try:
        # Create combined Open3D point cloud
        combined_pcd = o3d.geometry.PointCloud()
        combined_points = []
        combined_colors = []
        
        for i, (points, color) in enumerate(zip(points_list, colors)):
            if points.shape[0] == 0:
                continue
            
            # Add points
            combined_points.append(points[:, :3])
            
            # Add colors (repeat color for all points in this frame)
            frame_colors = np.tile(color, (points.shape[0], 1))
            combined_colors.append(frame_colors)
        
        if not combined_points:
            return False
        
        # Combine all points and colors
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Use headless rendering
        vis = o3d.visualization.rendering.OffscreenRenderer(800, 600)
        
        # Set up the scene
        vis.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background
        
        # Add the point cloud to the scene
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 1.2
        vis.scene.add_geometry("pointcloud", combined_pcd, mat)
        
        # Set up camera - slight elevation view
        bounds = combined_pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        
        # Position camera with slight elevation
        camera_distance = np.linalg.norm(extent) * 1.8
        camera_pos = center + np.array([camera_distance * 0.5, camera_distance * 0.5, camera_distance * 0.4])
        
        vis.setup_camera(30.0, center, camera_pos, [0, 0, 1])
        
        # Render and save
        img = vis.render_to_image()
        o3d.io.write_image(output_path, img)
        
        return True
        
    except Exception as e:
        # Fallback to matplotlib 3D rendering
        return create_multi_frame_matplotlib_viz(points_list, colors, labels, output_path, title)

def create_multi_frame_matplotlib_viz(points_list, colors, labels, output_path, title="Multi-Frame Point Cloud"):
    """
    Fallback multi-frame visualization using matplotlib when Open3D fails
    """
    if not points_list or all(points.shape[0] == 0 for points in points_list):
        return False
    
    # Subsample points for faster rendering
    viz_points_list = []
    for points in points_list:
        if points.shape[0] > 20000:
            indices = np.random.choice(points.shape[0], 5000, replace=False)
            viz_points_list.append(points[indices])
        else:
            viz_points_list.append(points)
    
    # Create combined BEV and 3D visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Bird's eye view (XY plane)
    ax1 = fig.add_subplot(121)
    for i, (points, color, label) in enumerate(zip(viz_points_list, colors, labels)):
        if points.shape[0] == 0:
            continue
        x, y = points[:, 0], points[:, 1]
        ax1.scatter(x, y, c=[color], s=0.5, alpha=0.7, label=label)
    
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Left)')
    ax1.set_title('Bird\'s Eye View (XY)')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # 3D view
    ax2 = fig.add_subplot(122, projection='3d')
    for i, (points, color, label) in enumerate(zip(viz_points_list, colors, labels)):
        if points.shape[0] == 0:
            continue
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax2.scatter(x, y, z, c=[color], s=0.5, alpha=0.7, label=label)
    
    ax2.set_xlabel('X (Forward)')
    ax2.set_ylabel('Y (Left)')
    ax2.set_zlabel('Z (Up)')
    ax2.set_title('3D View')
    ax2.legend()
    
    # Set viewing angle
    ax2.view_init(elev=5, azim=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

def create_open3d_visualization(points, output_path, title="Point Cloud"):
    """
    Create 3D point cloud visualization using Open3D headless rendering
    
    Args:
        points: Nx4 array (x, y, z, intensity)
        output_path: Path to save the image
        title: Title for the visualization
    """
    if points.shape[0] == 0:
        return False
    
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # xyz coordinates
        
        # Set all points to red color
        red_color = np.array([[1.0, 0.0, 0.0]] * points.shape[0])  # RGB red for all points
        pcd.colors = o3d.utility.Vector3dVector(red_color)
        
        # Use headless rendering
        vis = o3d.visualization.rendering.OffscreenRenderer(800, 600)
        
        # Set up the scene
        vis.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background
        
        # Add the point cloud to the scene
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 1.2
        vis.scene.add_geometry("pointcloud", pcd, mat)
        
        # Set up camera - slight elevation view
        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        
        # Position camera with slight elevation
        camera_distance = np.linalg.norm(extent) * 1.8
        camera_pos = center + np.array([camera_distance * 0.5, camera_distance * 0.5, camera_distance * 0.4])
        
        vis.setup_camera(30.0, center, camera_pos, [0, 0, 1])
        
        # Render and save
        img = vis.render_to_image()
        o3d.io.write_image(output_path, img)
        
        return True
        
    except Exception as e:
        # Fallback to matplotlib 3D rendering
        return create_fallback_3d_visualization(points, output_path, title)

def create_fallback_3d_visualization(points, output_path, title="Point Cloud"):
    """
    Fallback 3D visualization using matplotlib when Open3D fails
    
    Args:
        points: Nx4 array (x, y, z, intensity)
        output_path: Path to save the image
        title: Title for the visualization
    """
    if points.shape[0] == 0:
        return False
    
    # Subsample if too many points
    if points.shape[0] > 50000:
        indices = np.random.choice(points.shape[0], 10000, replace=False)
        points_viz = points[indices]
    else:
        points_viz = points
    
    x, y, z = points_viz[:, 0], points_viz[:, 1], points_viz[:, 2]
    
    # Create 3D plot with matplotlib
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='red', s=0.5, alpha=0.6)
    
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    ax.set_title('3D View')
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=5, azim=45)
    
    # Remove grid and axes for cleaner look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

def create_matplotlib_pointcloud_viz(points, output_path, title="Point Cloud", 
                                   subsample_factor=10):
    """
    Create point cloud visualization with BEV (matplotlib) and 3D view (Open3D)
    
    Args:
        points: Nx4 array (x, y, z, intensity)
        output_path: Path to save the image
        title: Title for the visualization
        subsample_factor: Factor to subsample points for faster rendering
    """
    if points.shape[0] == 0:
        return False
    
    # Subsample points for faster rendering
    if points.shape[0] > 50000:
        indices = np.random.choice(points.shape[0], 
                                 points.shape[0] // subsample_factor, 
                                 replace=False)
        points_viz = points[indices]
    else:
        points_viz = points
    
    # Extract coordinates
    x, y, z = points_viz[:, 0], points_viz[:, 1], points_viz[:, 2]
    
    # Create figure with BEV only
    fig = plt.figure(figsize=(8, 8))
    
    # Bird's eye view (XY plane)
    ax1 = fig.add_subplot(1, 1, 1)
    scatter1 = ax1.scatter(x, y, c='red', s=0.5, alpha=0.6)
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Left)')
    ax1.set_title('Bird\'s Eye View (XY)')
    ax1.set_aspect('equal')
    
    plt.suptitle(f'{title} - {points.shape[0]} points', fontsize=16)
    plt.tight_layout()
    
    # Save matplotlib part (BEV only)
    bev_output_path = output_path.replace('.png', '_bev.png')
    plt.savefig(bev_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create Open3D 3D visualization
    open3d_output_path = output_path.replace('.png', '_3d.png')
    create_open3d_visualization(points, open3d_output_path, title)
    
    # Create combined image with both views
    try:
        bev_img = cv2.imread(bev_output_path)
        o3d_img = cv2.imread(open3d_output_path)
        
        # Resize images to same height if needed
        if bev_img.shape[0] != o3d_img.shape[0]:
            target_height = min(bev_img.shape[0], o3d_img.shape[0])
            bev_img = cv2.resize(bev_img, (int(bev_img.shape[1] * target_height / bev_img.shape[0]), target_height))
            o3d_img = cv2.resize(o3d_img, (int(o3d_img.shape[1] * target_height / o3d_img.shape[0]), target_height))
        
        # Combine images horizontally
        combined_img = np.concatenate([bev_img, o3d_img], axis=1)
        cv2.imwrite(output_path, combined_img)
        
        # Clean up temporary files
        os.remove(bev_output_path)
        os.remove(open3d_output_path)
        
    except Exception as e:
        return True
    
    return True

def get_available_frames(data_path, sequence):
    """Get list of available frame files in the sequence"""
    range_dir = os.path.join(data_path, sequence, "processed", "range")
    if not os.path.exists(range_dir):
        return []
    
    # Get all .npy files and extract frame numbers
    npy_files = glob.glob(os.path.join(range_dir, "*.npy"))
    frame_numbers = []
    
    for file in npy_files:
        filename = os.path.basename(file)
        if filename.endswith('.npy') and len(filename) == 10:  # XXXXXX.npy
            try:
                frame_num = int(filename[:6])
                frame_numbers.append(frame_num)
            except ValueError:
                continue
    
    return sorted(frame_numbers)

def visualize_frame(data_path, sequence, frame_idx, save_path, cfg=None, video_mode=False, 
                   poses=None, poses_path=None, multi_frame=False, dataset_path=None):
    """
    Visualize both 2D range images and 3D point cloud for a single frame or multiple frames
    
    Args:
        data_path: Path to processed data
        sequence: Sequence number (e.g., "01")
        frame_idx: Frame index (e.g., 0 for 000000.npy)
        save_path: Path to save the visualization
        cfg: Configuration dictionary
        video_mode: If True, save to temporary directory for video creation
        poses: List of pose matrices (loaded externally)
        poses_path: Path to poses directory (alternative to poses)
        multi_frame: If True, plot frame_idx+1 in blue and frame_idx+2 in green
        dataset_path: Path to KITTI dataset for calibration loading
    """
    # Load poses if needed
    if poses is None and poses_path is not None:
        poses = load_poses(poses_path, sequence, dataset_path)
    
    # Load calibration data
    calib = None
    T_cam_velo = None
    if dataset_path is not None:
        calib = load_calibration(dataset_path, sequence)
        if calib is not None and 'T_cam0_velo' in calib:
            T_cam_velo = calib['T_cam0_velo']
    
    # Get configuration parameters
    max_range = cfg.get("DATA_CONFIG", {}).get("MAX_RANGE", 80.0) if cfg else 80.0
    
    # Load primary frame
    frame_str = str(frame_idx).zfill(6)
    points = load_frame_pointcloud(data_path, sequence, frame_idx, max_range)
    
    if points is None:
        return False
    
    # Apply pose transformation if available
    if poses is not None and frame_idx < len(poses):
        points = transform_points(points, poses[frame_idx], T_cam_velo)
    
    # Set up save paths based on mode
    if video_mode:
        # Create temporary directory for video frames
        temp_dir = os.path.join(save_path, "temp_video_frames")
        os.makedirs(temp_dir, exist_ok=True)
        if multi_frame:
            pcd_output_path = os.path.join(temp_dir, f'pointcloud_seq_{sequence}_multiframe_{frame_str}.png')
        else:
            pcd_output_path = os.path.join(temp_dir, f'pointcloud_seq_{sequence}_frame_{frame_str}.png')
        range_output_filename = os.path.join(temp_dir, f'range_images_seq_{sequence}_frame_{frame_str}.png')
    else:
        # Create output directory for regular saves
        os.makedirs(save_path, exist_ok=True)
        if multi_frame:
            pcd_output_path = os.path.join(save_path, f'pointcloud_seq_{sequence}_multiframe_{frame_str}.png')
        else:
            pcd_output_path = os.path.join(save_path, f'pointcloud_seq_{sequence}_frame_{frame_str}.png')
        range_output_filename = os.path.join(save_path, f'range_images_seq_{sequence}_frame_{frame_str}.png')
    
    # Handle multi-frame visualization
    if multi_frame:
        points_list = [points]
        colors = [(1.0, 0.0, 0.0)]  # Red for primary frame
        labels = [f"Frame {frame_idx}"]
        
        # Load and transform frame i+1 (blue)
        if frame_idx + 1 < len(poses) if poses else True:
            points_next = load_frame_pointcloud(data_path, sequence, frame_idx + 1, max_range)
            if points_next is not None:
                if poses is not None and frame_idx + 1 < len(poses):
                    points_next = transform_points(points_next, poses[frame_idx + 1], T_cam_velo)
                points_list.append(points_next)
                colors.append((0.0, 0.0, 1.0))  # Blue
                labels.append(f"Frame {frame_idx + 1}")
        
        # Load and transform frame i+2 (green)
        if frame_idx + 2 < len(poses) if poses else True:
            points_next2 = load_frame_pointcloud(data_path, sequence, frame_idx + 2, max_range)
            if points_next2 is not None:
                if poses is not None and frame_idx + 2 < len(poses):
                    points_next2 = transform_points(points_next2, poses[frame_idx + 2], T_cam_velo)
                points_list.append(points_next2)
                colors.append((0.0, 1.0, 0.0))  # Green
                labels.append(f"Frame {frame_idx + 2}")
        
        # Create multi-frame visualization
        success_pcd = create_multi_frame_visualization(points_list, colors, labels, 
                                                      pcd_output_path, 
                                                      f"Sequence {sequence} Multi-Frame Starting at {frame_str}")
    else:
        # Single frame visualization
        success_pcd = create_matplotlib_pointcloud_viz(points, pcd_output_path, 
                                                      f"Sequence {sequence} Frame {frame_str}")
    
    # Create 2D range image visualizations only for primary frame (and only if not in multi-frame video mode)
    if not (multi_frame and video_mode):
        # Load original range data for 2D visualization
        range_file = os.path.join(data_path, sequence, "processed", "range", f"{frame_str}.npy")
        intensity_file = os.path.join(data_path, sequence, "processed", "intensity", f"{frame_str}.npy")
        
        if os.path.exists(range_file) and os.path.exists(intensity_file):
            range_data = load_range_data(range_file)
            intensity_data = load_intensity_data(intensity_file)
            
            # Normalize data for better visualization
            range_norm = normalize_data(range_data, percentile_clip=2)
            intensity_norm = normalize_data(intensity_data, percentile_clip=2)
            
            # Create custom colormap
            custom_cmap = create_custom_colormap()
            
            # Create 2D visualization - only Range and Intensity
            fig, axes = plt.subplots(2, 1, figsize=(18, 4))
            fig.suptitle(f'KITTI Sequence {sequence} - Frame {frame_str} - Range Image Visualization', fontsize=16)
            
            # Range
            im1 = axes[0].imshow(range_norm, cmap=custom_cmap, aspect='auto')
            axes[0].set_title(f'Range\nMin: {range_data.min():.2f}, Max: {range_data.max():.2f}')
            axes[0].set_xlabel('Width (Azimuth)')
            axes[0].set_ylabel('Height (Elevation)')
            plt.colorbar(im1, ax=axes[0])
            
            # Intensity
            im2 = axes[1].imshow(intensity_norm, cmap='hot', aspect='auto')
            axes[1].set_title(f'Intensity\nMin: {intensity_data.min():.2f}, Max: {intensity_data.max():.2f}')
            axes[1].set_xlabel('Width (Azimuth)')
            axes[1].set_ylabel('Height (Elevation)')
            plt.colorbar(im2, ax=axes[1])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the 2D visualization
            plt.savefig(range_output_filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    return success_pcd  # Return success status

def create_video_from_images(image_pattern, output_video_path, fps=10):
    """
    Create video from sequence of images with progress bar
    
    Args:
        image_pattern: Pattern to match images (e.g., '/path/range_images_seq_00_frame_*.png')
        output_video_path: Path to save output video
        fps: Frames per second
    
    Returns:
        bool: Success status
    """
    # Get sorted list of images
    images = sorted(glob.glob(image_pattern))
    
    if not images:
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(images[0])
    if first_img is None:
        return False
    
    height, width, channels = first_img.shape
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        return False
    
    # Add images to video with progress bar
    for image_path in tqdm(images, desc="Creating video", unit="frames"):
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        # Resize image if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
    
    # Release everything
    out.release()
    return True

def main():
    """Main function to generate visualizations and videos"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KITTI Point Cloud and Range Image Visualization with Poses')
    parser.add_argument('--video', action='store_true', help='Generate videos from all frames')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 1, 2, 3, 4], 
                       help='Frame indices to process (ignored if --video is used)')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence to process')
    parser.add_argument('--fps', type=int, default=10, help='Video framerate (default: 10)')
    parser.add_argument('--multi-frame', action='store_true', 
                       help='Plot frame i (red), i+1 (blue), and i+2 (green) together')
    parser.add_argument('--poses-path', type=str, default='/DATA/vishwesh/kitti/poses',
                       help='Path to poses directory')
    parser.add_argument('--dataset-path', type=str, default='/DATA/vishwesh/kitti',
                       help='Path to KITTI dataset root (for calibration)')
    args = parser.parse_args()
    
    # Configuration paths
    config_path = "/home/vishwesh/PPMFNet/configs/parameters.yml"
    data_path = "/DATA/vishwesh/kitti/processed_data"
    save_path = "/home/vishwesh/PPMFNet/media"
    
    sequence = args.sequence
    
    print(f"🚀 KITTI Visualization Generator with LiDAR-Camera Transform")
    print(f"Sequence: {sequence} | Video: {args.video} | Multi-frame: {args.multi_frame}")
    
    # Try to load config
    try:
        cfg = load_config(config_path)
        max_range = cfg.get('DATA_CONFIG', {}).get('MAX_RANGE', 80.0)
        print(f"✓ Config loaded | Max range: {max_range}m")
    except Exception as e:
        print(f"⚠ Config failed, using defaults")
        cfg = None
    
    # Load poses
    poses = load_poses(args.poses_path, sequence, args.dataset_path)
    if poses is not None:
        print(f"✓ Loaded {len(poses)} poses")
    else:
        print(f"⚠ No poses found - using original coordinates")
    
    # Load calibration data
    calib = load_calibration(args.dataset_path, sequence)
    if calib is not None:
        if 'T_cam0_velo' in calib:
            print(f"✓ Calibration loaded - LiDAR-to-Camera transform available")
            print(f"  Transform matrix shape: {calib['T_cam0_velo'].shape}")
        else:
            print(f"⚠ Calibration loaded but no LiDAR-to-Camera transform found")
    else:
        print(f"⚠ No calibration found - skipping LiDAR-to-Camera transform")
    
    # Determine frames to process
    if args.video:
        # Get all available frames
        frames = get_available_frames(data_path, sequence)
        if not frames:
            print(f"❌ No frames found for sequence {sequence}")
            return
        
        # Limit frames if poses are available to avoid index errors
        if poses is not None:
            max_frame = min(max(frames), len(poses) - 1)
            frames = [f for f in frames if f <= max_frame]
        
        print(f"📹 Video mode: {len(frames)} frames ({frames[0]}-{frames[-1]}) @ {args.fps} fps")
    else:
        frames = args.frames
        
        # Check if requested frames are within pose limits
        if poses is not None:
            max_frame = len(poses) - 1
            valid_frames = [f for f in frames if f <= max_frame]
            if len(valid_frames) != len(frames):
                frames = valid_frames
                print(f"⚠ Limited to frames: {frames}")
        
        print(f"🖼 Image mode: {len(frames)} frames")
    
    # Generate visualizations for each frame
    successful_frames = []
    failed_frames = []
    
    # Process frames with progress bar
    for frame_idx in tqdm(frames, desc="Processing frames", unit="frame"):
        success = visualize_frame(data_path, sequence, frame_idx, save_path, cfg, 
                                video_mode=args.video, poses=poses, 
                                multi_frame=args.multi_frame, dataset_path=args.dataset_path)
        
        if success:
            successful_frames.append(frame_idx)
        else:
            failed_frames.append(frame_idx)
    
    # Create videos if requested
    if args.video and successful_frames:
        print(f"\n🎬 Creating videos...")
        
        # Use temporary directory for video creation
        temp_dir = os.path.join(save_path, "temp_video_frames")
        
        if args.multi_frame:
            # Create multi-frame point cloud video
            pcd_pattern = os.path.join(temp_dir, f'pointcloud_seq_{sequence}_multiframe_*.png')
            pcd_video_path = os.path.join(save_path, f'pointcloud_seq_{sequence}_multiframe_video.mp4')
            pcd_success = create_video_from_images(pcd_pattern, pcd_video_path, args.fps)
            range_success = True  # Skip range video for multi-frame mode
        else:
            # Create range image video
            range_pattern = os.path.join(temp_dir, f'range_images_seq_{sequence}_frame_*.png')
            range_video_path = os.path.join(save_path, f'range_images_seq_{sequence}_video.mp4')
            range_success = create_video_from_images(range_pattern, range_video_path, args.fps)
            
            # Create point cloud video
            pcd_pattern = os.path.join(temp_dir, f'pointcloud_seq_{sequence}_frame_*.png')
            pcd_video_path = os.path.join(save_path, f'pointcloud_seq_{sequence}_video.mp4')
            pcd_success = create_video_from_images(pcd_pattern, pcd_video_path, args.fps)
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            with tqdm(desc="Cleaning temp files", unit="file") as pbar:
                try:
                    shutil.rmtree(temp_dir)
                    pbar.update(1)
                except Exception as e:
                    pass
    
    # Summary
    print(f"\n✅ SUMMARY")
    print(f"Successfully processed: {len(successful_frames)} frames")
    if failed_frames:
        print(f"Failed: {len(failed_frames)} frames")
    
    if poses is not None:
        print("✓ Pose transformations applied")
    
    if calib is not None and 'T_cam0_velo' in calib:
        print("✓ LiDAR-to-Camera transformations applied")
        print("  Transform chain: LiDAR → Camera → World")
    else:
        print("⚠ No LiDAR-to-Camera transform - points may be in wrong coordinate system")
    
    if args.video:
        print(f"\n🎥 Generated videos:")
        if args.multi_frame:
            print(f"   • pointcloud_seq_{sequence}_multiframe_video.mp4")
            print(f"     (Multi-frame: red=frame i, blue=i+1, green=i+2)")
        else:
            print(f"   • range_images_seq_{sequence}_video.mp4")
            print(f"   • pointcloud_seq_{sequence}_video.mp4")
        if poses is not None:
            print("     (All frames transformed to world coordinates)")
    
    print(f"\n💾 Output saved to: {save_path}")
    print("🏁 Done!")

if __name__ == "__main__":
    main()