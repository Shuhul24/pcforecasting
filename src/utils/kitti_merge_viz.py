#!/usr/bin/env python3
"""
KITTI Point Cloud Visualization Script - Modified for N-Frame Black Point Rendering
Merges n-frames and renders all points as black for combined visualization
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

def create_nframe_black_visualization(points_list, output_path, title="N-Frame Merged Point Cloud"):
    """
    Create 3D point cloud visualization with n-frames merged as black points using Open3D
    
    Args:
        points_list: List of Nx4 arrays (x, y, z, intensity) - one for each frame
        output_path: Path to save the image
        title: Title for the visualization
    
    Returns:
        bool: Success status
    """
    if not points_list or all(points.shape[0] == 0 for points in points_list):
        return False
    
    try:
        # Merge all point clouds
        combined_points = []
        total_points = 0
        
        for points in points_list:
            if points.shape[0] > 0:
                combined_points.append(points[:, :3])  # Only XYZ coordinates
                total_points += points.shape[0]
        
        if not combined_points:
            return False
        
        # Combine all points into single array
        all_points = np.vstack(combined_points)
        
        # Create Open3D point cloud
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Set all points to black color
        black_colors = np.zeros((total_points, 3))  # RGB black for all points
        combined_pcd.colors = o3d.utility.Vector3dVector(black_colors)
        
        # Use headless rendering
        vis = o3d.visualization.rendering.OffscreenRenderer(800, 600)
        
        # Set up the scene with white background for contrast with black points
        vis.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background
        
        # Add the point cloud to the scene
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 0.01  # Smaller points for dense visualization
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
        print(f"Open3D rendering failed: {e}")
        # Fallback to matplotlib 3D rendering
        return create_nframe_matplotlib_viz(points_list, output_path, title)

def create_nframe_matplotlib_viz(points_list, output_path, title="N-Frame Merged Point Cloud"):
    """
    Fallback n-frame visualization using matplotlib when Open3D fails
    
    Args:
        points_list: List of Nx4 arrays (x, y, z, intensity) - one for each frame
        output_path: Path to save the image
        title: Title for the visualization
    
    Returns:
        bool: Success status
    """
    if not points_list or all(points.shape[0] == 0 for points in points_list):
        return False
    
    # Merge and subsample points for faster rendering
    combined_points = []
    total_points = 0
    
    for points in points_list:
        if points.shape[0] > 0:
            # Subsample each frame if too many points
            if points.shape[0] > 10000:
                indices = np.random.choice(points.shape[0], 5000, replace=False)
                subsampled = points[indices]
            else:
                subsampled = points
            combined_points.append(subsampled)
            total_points += subsampled.shape[0]
    
    if not combined_points:
        return False
    
    # Merge all points
    all_points = np.vstack(combined_points)
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    
    # Create combined BEV and 3D visualization
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    
    # Bird's eye view (XY plane)
    ax1 = fig.add_subplot(121)
    ax1.scatter(x, y, c='black', s=0.3, alpha=0.7)
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Left)')
    ax1.set_title('Bird\'s Eye View (XY)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 3D view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x, y, z, c='black', s=0.3, alpha=0.7)
    ax2.set_xlabel('X (Forward)')
    ax2.set_ylabel('Y (Left)')
    ax2.set_zlabel('Z (Up)')
    ax2.set_title('3D View')
    
    # Set viewing angle
    ax2.view_init(elev=10, azim=45)
    
    plt.suptitle(f'{title}\nMerged {len(points_list)} frames - {total_points:,} points', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
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

def visualize_nframe_merged(data_path, sequence, start_frame, n_frames, save_path, cfg=None, 
                           poses=None, dataset_path=None):
    """
    Visualize n-frames merged together as black points
    
    Args:
        data_path: Path to processed data
        sequence: Sequence number (e.g., "01")
        start_frame: Starting frame index
        n_frames: Number of frames to merge
        save_path: Path to save the visualization
        cfg: Configuration dictionary
        poses: List of pose matrices
        dataset_path: Path to KITTI dataset for calibration loading
    
    Returns:
        bool: Success status
    """
    # Load calibration data
    calib = None
    T_cam_velo = None
    if dataset_path is not None:
        calib = load_calibration(dataset_path, sequence)
        if calib is not None and 'T_cam0_velo' in calib:
            T_cam_velo = calib['T_cam0_velo']
    
    # Get configuration parameters
    max_range = cfg.get("DATA_CONFIG", {}).get("MAX_RANGE", 80.0) if cfg else 80.0
    
    # Load and transform all frames
    points_list = []
    successful_frames = []
    
    print(f"Loading {n_frames} frames starting from frame {start_frame}...")
    
    for i in range(n_frames):
        frame_idx = start_frame + i
        
        # Load frame point cloud
        points = load_frame_pointcloud(data_path, sequence, frame_idx, max_range)
        
        if points is None:
            print(f"⚠ Failed to load frame {frame_idx}")
            continue
        
        # Apply pose transformation if available
        if poses is not None and frame_idx < len(poses):
            points = transform_points(points, poses[frame_idx], T_cam_velo)
        
        points_list.append(points)
        successful_frames.append(frame_idx)
    
    if not points_list:
        print(f"❌ No frames could be loaded")
        return False
    
    # Create output path
    os.makedirs(save_path, exist_ok=True)
    output_filename = os.path.join(save_path, 
                                  f'merged_pointcloud_seq_{sequence}_frames_{start_frame:06d}-{start_frame+n_frames-1:06d}_black.png')
    
    # Create visualization
    title = f"Sequence {sequence} - Merged Frames {start_frame}-{start_frame+n_frames-1}"
    success = create_nframe_black_visualization(points_list, output_filename, title)
    
    if success:
        print(f"✓ Successfully merged {len(successful_frames)} frames and saved as black points")
        print(f"💾 Saved to: {output_filename}")
        
        # Print statistics
        total_points = sum(points.shape[0] for points in points_list)
        print(f"📊 Total points: {total_points:,}")
        print(f"📋 Frames used: {successful_frames}")
        
        if poses is not None:
            print("✓ Pose transformations applied (world coordinates)")
        if T_cam_velo is not None:
            print("✓ LiDAR-to-Camera transformations applied")
    else:
        print(f"❌ Failed to create visualization")
    
    return success

def main():
    """Main function to generate n-frame merged visualizations with black points"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KITTI N-Frame Merged Visualization with Black Points')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence to process')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame index')
    parser.add_argument('--n-frames', type=int, default=5, help='Number of frames to merge')
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
    start_frame = args.start_frame
    n_frames = args.n_frames
    
    print(f"🚀 KITTI N-Frame Merged Visualization with Black Points")
    print(f"Sequence: {sequence} | Start Frame: {start_frame} | N-Frames: {n_frames}")
    
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
        
        # Check if we have enough poses for requested frames
        if start_frame + n_frames > len(poses):
            max_n_frames = len(poses) - start_frame
            if max_n_frames <= 0:
                print(f"❌ Start frame {start_frame} exceeds available poses ({len(poses)})")
                return
            print(f"⚠ Limiting to {max_n_frames} frames due to pose availability")
            n_frames = max_n_frames
    else:
        print(f"⚠ No poses found - using original coordinates")
    
    # Check frame availability
    available_frames = get_available_frames(data_path, sequence)
    if not available_frames:
        print(f"❌ No frames found for sequence {sequence}")
        return
    
    # Verify requested frames are available
    requested_frames = list(range(start_frame, start_frame + n_frames))
    unavailable = [f for f in requested_frames if f not in available_frames]
    if unavailable:
        print(f"⚠ Some requested frames are not available: {unavailable}")
        available_requested = [f for f in requested_frames if f in available_frames]
        if not available_requested:
            print(f"❌ None of the requested frames are available")
            return
        n_frames = len(available_requested)
        print(f"⚠ Proceeding with {n_frames} available frames")
    
    # Generate merged visualization
    success = visualize_nframe_merged(data_path, sequence, start_frame, n_frames, save_path, 
                                    cfg, poses, args.dataset_path)
    
    # Summary
    print(f"\n✅ SUMMARY")
    if success:
        print(f"✓ Successfully created merged visualization with {n_frames} frames")
        print(f"✓ All points rendered as black")
        if poses is not None:
            print("✓ Pose transformations applied (world coordinates)")
        print(f"💾 Output saved to: {save_path}")
    else:
        print(f"❌ Failed to create merged visualization")
    
    print("🏁 Done!")

if __name__ == "__main__":
    main()