#!/usr/bin/env python3
import os
import time
import argparse
import yaml
import subprocess
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Fix the path issue for pyTorchChamferDistance
# Change to the src directory so relative paths work correctly
original_dir = os.getcwd()
src_dir = '/home/soham/garments/preet/here/PPMFNet/src'
os.chdir(src_dir)

from src.datasets.datasets_kitti import KittiOdometryModule
from src.utils.projection import projection
from src.models.chamfer import chamfer_dist_pc  # Use the new point cloud chamfer distance

# Change back to original directory
os.chdir(original_dir)

class GTPointCloudForecaster:
    """Ground truth pose-based point cloud forecasting using range images"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.chamfer_distance = chamfer_dist_pc(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize projection utility for range image to point cloud conversion
        self.projection = projection(cfg)
        
    def range_data_to_point_cloud(self, data):
        """Convert range image data to point cloud
        
        Args:
            data: (4, H, W) tensor containing [range, x, y, z] channels
            
        Returns:
            points: (N, 3) tensor of valid 3D points
        """
        # data shape: (4, H, W) where channels are [range, x, y, z]
        range_img = data[0]  # (H, W)
        x_img = data[1]      # (H, W)  
        y_img = data[2]      # (H, W)
        z_img = data[3]      # (H, W)
        
        # Find valid pixels (where range > 0)
        valid_mask = range_img > 0
        
        if not valid_mask.any():
            return torch.zeros((0, 3), device=data.device)
        
        # Extract valid 3D coordinates
        x_valid = x_img[valid_mask]
        y_valid = y_img[valid_mask] 
        z_valid = z_img[valid_mask]
        
        # Stack into point cloud format (N, 3)
        points = torch.stack([x_valid, y_valid, z_valid], dim=1)
        
        return points
    # def range_data_to_point_cloud(self, data):
    #     """
    #     Convert a batch of range images to point clouds.

    #     Args:
    #         data: (B, 4, H, W) tensor containing [range, x, y, z] channels

    #     Returns:
    #         points_list: list of length B where each element is a (N_i, 3) tensor of valid 3D points
    #     """
    #     B, F,_, H, W = data.shape
    #     range_img = data[:, 0, :, :]  # (B, H, W)
    #     x_img = data[:, 1, :, :]      # (B, H, W)
    #     y_img = data[:, 2, :, :]      # (B, H, W)
    #     z_img = data[:, 3, :, :]      # (B, H, W)

    #     valid_mask = range_img > 0    # (B, H, W)

    #     points_list = []
    #     for b in range(B):
    #         mask = valid_mask[b]  # (H, W)
    #         if not mask.any():
    #             points_list.append(torch.zeros((0, 3), device=data.device))
    #             continue

    #         x_valid = x_img[b][mask]
    #         y_valid = y_img[b][mask]
    #         z_valid = z_img[b][mask]
    #         points = torch.stack([x_valid, y_valid, z_valid], dim=1)
    #         points_list.append(points)

    #     return points_list

    def transform_point_cloud(self, points, from_pose, to_pose, calibration):
        """Transform point cloud from one camera frame to another
        
        Args:
            points: (N, 3) or (N, 4) tensor of points in lidar coordinates
            from_pose: (4, 4) camera-to-world transform matrix
            to_pose: (4, 4) camera-to-world transform matrix  
            calibration: (4, 4) lidar-to-camera transform matrix
            
        Returns:
            transformed_points: (N, 3) transformed points in lidar coordinates
        """
        device = points.device
        
        # Take only xyz coordinates if points have intensity
        if points.shape[1] == 4:
            xyz_points = points[:, :3]  # (N, 3)
        else:
            xyz_points = points
            
        N = xyz_points.shape[0]
        
        if N == 0:
            return torch.zeros((0, 3), device=device)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(N, 1, device=device)
        points_homo = torch.cat([xyz_points, ones], dim=1)  # (N, 4)
        
        # Transform sequence:
        # 1. Lidar -> Camera (from_frame): calibration
        # 2. Camera -> World: from_pose  
        # 3. World -> Camera (to_frame): to_pose^-1
        # 4. Camera -> Lidar: calibration^-1
        # Inside transform_point_cloud, before the inversion
        if torch.isnan(to_pose).any() or torch.isinf(to_pose).any():
            print("!!! Warning: NaN or Inf found in to_pose matrix.")

        # Check if the matrix is singular before inverting
        det_to_pose = torch.linalg.det(to_pose)
        if torch.isclose(det_to_pose, torch.tensor(0.0)):
            print(f"!!! Warning: to_pose matrix is singular or nearly singular! Determinant: {det_to_pose.item()}")

        # to_pose_inv = torch.linalg.inv(to_pose)
        # Add similar checks for the 'calibration' matrix
        calib_inv = torch.linalg.inv(calibration)
        to_pose_inv = torch.linalg.inv(to_pose)
        
        # Combined transformation matrix
        transform_matrix = calib_inv @ to_pose_inv @ from_pose @ calibration
        
        # Apply transformation: (4, 4) @ (4, N) -> (4, N)
        transformed_points_homo = transform_matrix @ points_homo.T  # (4, N)
        transformed_points = transformed_points_homo[:3].T  # (N, 3)
        
        return transformed_points
    
    def forecast_point_cloud(self, current_data, current_pose, target_pose, calibration):
        """Forecast point cloud at target timestep using current range data and poses
        
        Args:
            current_data: (4, H, W) current range image data (at timestep t)
            current_pose: (4, 4) current camera pose (at timestep t)
            target_pose: (4, 4) target camera pose (at timestep t+k)
            calibration: (4, 4) lidar-to-camera calibration
            
        Returns:
            predicted_pc: (N, 3) predicted point cloud
        """
        # Convert range image to point cloud
        current_pc = self.range_data_to_point_cloud(current_data)
        if torch.isnan(current_pc).any():
            print("!!! Warning: NaN values produced by range_data_to_point_cloud.")
        # Transform point cloud from current frame to target frame
        transformed_pc = self.transform_point_cloud(
            current_pc, current_pose, target_pose, calibration
        )
        
        return transformed_pc
    
    def evaluate_forecasting(self, data_loader):
        """Evaluate ground truth pose forecasting on the dataset using range images
        
        Args:
            data_loader: DataLoader for test data
            
        Returns:
            dict: Evaluation metrics
        """
        # Store distances for each timestep separately
        timestep_distances = {f't+{i+1}': [] for i in range(5)}
        all_chamfer_distances = []
        
        # Set up progress bar
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                   desc="GT Pose Forecasting (Range->Range)", unit="batch")
        
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # Move data to CUDA
                past_data = batch['past_data'].to(self.device)  # (B, 5, 4, H, W)
                fut_data = batch['fut_data'].to(self.device)    # (B, 5, 4, H, W)
                past_poses = batch['past_poses'].to(self.device)  # (B, 5, 4, 4)
                fut_poses = batch['fut_poses'].to(self.device)    # (B, 5, 4, 4)
                calibration = batch['calibration'][0].to(self.device)  # (4, 4) - same for all in batch
                
                B = past_data.shape[0]
                T_fut = fut_data.shape[1]  # Number of future timesteps (should be 5)
                
                batch_distances = []
                
                # Process each sample in the batch
                for b in range(B):
                    # Use the last timestep from past_data as current range data (t)
                    current_data = past_data[b, -1]  # (4, H, W) - last past timestep
                    current_pose = past_poses[b, -1]  # (4, 4) - last past pose
                    
                    sample_distances = []
                    
                    # Collect predicted and target point clouds for all future timesteps
                    predicted_pcs = []
                    target_pcs = []
                    
                    # Evaluate each future timestep
                    for t_fut in range(T_fut):
                        # Get target pose for this future timestep
                        target_pose = fut_poses[b, t_fut]  # (4, 4)
                        
                        # Predict point cloud for this future timestep using range data
                        pred_pc = self.forecast_point_cloud(
                            current_data, current_pose, target_pose, calibration
                        )
                        
                        # Get ground truth point cloud for this future timestep
                        gt_data = fut_data[b][t_fut].to(self.device)  
                        gt_pc = self.range_data_to_point_cloud(gt_data)

                        predicted_pcs.append(pred_pc)
                        target_pcs.append(gt_pc)
                    
                    # Now compute chamfer distances for all timesteps at once
                    for t_fut in range(T_fut):
                        pred_pc = predicted_pcs[t_fut].unsqueeze(0)  # (1, N, 3)
                        gt_pc = target_pcs[t_fut].unsqueeze(0)       # (1, M, 3)
                        
                        # Compute chamfer distance using point cloud method
                        chamfer_distances, chamfer_tensor = self.chamfer_distance(
                            pred_pc, gt_pc
                        )
                        # breakpoint()
                        cd_value = chamfer_distances[0].item()
                        sample_distances.append(cd_value)
                        timestep_distances[f't+{t_fut+1}'].append(cd_value)
                    
                    # Average over future timesteps for this sample
                    avg_distance = np.mean(sample_distances)
                    batch_distances.append(avg_distance)
                
                # Average over samples in this batch
                batch_avg_distance = np.mean(batch_distances)
                all_chamfer_distances.append(batch_avg_distance)
                
                # Update progress bar with current means
                current_total_mean = np.mean(all_chamfer_distances)
                t1_mean = np.mean(timestep_distances['t+1']) if timestep_distances['t+1'] else 0
                t5_mean = np.mean(timestep_distances['t+5']) if timestep_distances['t+5'] else 0
                
                pbar.set_postfix({
                    'Total': f'{current_total_mean:.4f}',
                    't+1': f'{t1_mean:.4f}',
                    't+5': f'{t5_mean:.4f}'
                })
        
        pbar.close()
        
        # Compute final metrics for each timestep and overall
        results = {
            'mean_chamfer_distance': np.mean(all_chamfer_distances),
            'std_chamfer_distance': np.std(all_chamfer_distances),
            'all_distances': all_chamfer_distances
        }
        
        # Add per-timestep metrics
        for timestep, distances in timestep_distances.items():
            results[f'mean_chamfer_{timestep}'] = np.mean(distances)
            results[f'std_chamfer_{timestep}'] = np.std(distances)
        
        return results

def load_kitti_data(split='test'):
    """Load KITTI Odometry data for train, validation, and test splits"""
    
    # Configuration paths
    config_path = "/home/soham/garments/preet/here/PPMFNet/configs/parameters.yml"
    data_path = "/DATA/common/kitti/processed_data"
    
    # Load configuration
    print("Loading configuration from:", config_path)
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Update data path in config
    cfg["DATA_CONFIG"]["PROCESSED_PATH"] = data_path
    # Add raw data path for point clouds
    cfg["DATA_CONFIG"]["RAW_PATH"] = "/DATA/soham/robotics/kitti/dataset/sequences"
    
    # Initialize the data module
    print("Initializing KITTI Odometry Data Module...")
    data_module = KittiOdometryModule(cfg)

    # Setup data loaders
    data_module.setup()
    if split=='test':
        test_loader = data_module.test_dataloader()
    if split=='val':
        test_loader = data_module.val_dataloader()
    # Get test data loader
    

    return data_module, test_loader, cfg

def main():
    try:
        data_module, test_loader, cfg = load_kitti_data()

        # Initialize the GT forecaster
        print("Initializing GT Point Cloud Forecaster (Range->Range)...")
        forecaster = GTPointCloudForecaster(cfg)
        
        # Move chamfer distance module to CUDA
        forecaster.chamfer_distance.to('cuda')
        
        # Run evaluation
        print("\nRunning GT pose forecasting evaluation (Range Image -> Range Image)...")
        results = forecaster.evaluate_forecasting(test_loader)
        
        # Print results
        print("\n" + "="*50)
        print("GT POSE FORECASTING RESULTS (Range->Range)")
        print("="*50)
        print(f"Overall Mean Chamfer Distance: {results['mean_chamfer_distance']:.6f}")
        print(f"Overall Std Chamfer Distance: {results['std_chamfer_distance']:.6f}")
        print(f"Number of batches evaluated: {len(results['all_distances'])}")
        print("\nPer-Timestep Results:")
        print("-" * 30)
        for i in range(1, 6):
            timestep = f't+{i}'
            mean_key = f'mean_chamfer_{timestep}'
            std_key = f'std_chamfer_{timestep}'
            print(f"{timestep}: {results[mean_key]:.6f} ± {results[std_key]:.6f}")
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        print(f"Total samples processed: {sum(len(distances) for distances in [results['all_distances']])}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()