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
src_dir = os.path.join(original_dir, 'src')
os.chdir(src_dir)

from datasets.datasets_kitti import KittiOdometryModule
from utils.projection import projection
from models.chamfer import chamfer_dist_pc  # Use the new point cloud chamfer distance

# Change back to original directory
os.chdir(original_dir)

class GTPointCloudForecaster:
    """Ground truth pose-based point cloud forecasting using raw point clouds"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.chamfer_distance = chamfer_dist_pc(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        calib_inv = torch.linalg.inv(calibration)
        to_pose_inv = torch.linalg.inv(to_pose)
        
        # Combined transformation matrix
        transform_matrix = calib_inv @ to_pose_inv @ from_pose @ calibration
        
        # Apply transformation: (4, 4) @ (4, N) -> (4, N)
        transformed_points_homo = transform_matrix @ points_homo.T  # (4, N)
        transformed_points = transformed_points_homo[:3].T  # (N, 3)
        
        return transformed_points
    
    def forecast_point_cloud(self, current_pc, current_pose, target_pose, calibration):
        """Forecast point cloud at target timestep using current point cloud and poses
        
        Args:
            current_pc: (N, 3) or (N, 4) current point cloud (at timestep t)
            current_pose: (4, 4) current camera pose (at timestep t)
            target_pose: (4, 4) target camera pose (at timestep t+k)
            calibration: (4, 4) lidar-to-camera calibration
            
        Returns:
            predicted_pc: (N, 3) predicted point cloud
        """
        # Transform point cloud from current frame to target frame
        transformed_pc = self.transform_point_cloud(
            current_pc, current_pose, target_pose, calibration
        )
        
        return transformed_pc
    
    def evaluate_forecasting(self, data_loader):
        """Evaluate ground truth pose forecasting on the dataset using raw point clouds
        
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
                   desc="GT Pose Forecasting (PC->PC)", unit="batch")
        
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # Move pose data to CUDA
                past_poses = batch['past_poses'].to(self.device)  # (B, 5, 4, 4)
                fut_poses = batch['fut_poses'].to(self.device)    # (B, 5, 4, 4)
                calibration = batch['calibration'][0].to(self.device)  # (4, 4) - same for all in batch
                
                # Get raw point clouds - they're structured as lists
                past_pc = batch['past_pc']  # List of length B, each element is list of length 5 (timesteps)
                fut_pc = batch['fut_pc']    # List of length B, each element is list of length 5 (timesteps)
                
                B = len(past_pc)
                T_fut = len(fut_pc[0])  # Number of future timesteps (should be 5)
                
                batch_distances = []
                
                # Process each sample in the batch
                for b in range(B):
                    # Use the last timestep from past_pc as current point cloud (t)
                    current_pc = past_pc[b][-1].to(self.device)  # (N, 4) - last past timestep
                    current_pose = past_poses[b, -1]             # (4, 4) - last past pose
                    
                    sample_distances = []
                    
                    # Collect predicted and target point clouds for all future timesteps
                    predicted_pcs = []
                    target_pcs = []
                    
                    # Evaluate each future timestep
                    for t_fut in range(T_fut):
                        # Get target pose for this future timestep
                        target_pose = fut_poses[b, t_fut]  # (4, 4)
                        
                        # Predict point cloud for this future timestep using raw PC
                        pred_pc = self.forecast_point_cloud(
                            current_pc, current_pose, target_pose, calibration
                        )
                        
                        # Get ground truth point cloud for this future timestep
                        gt_pc = fut_pc[b][t_fut].to(self.device)  # (M, 3) or (M, 4)
                        
                        # Take only xyz coordinates if points have intensity
                        if pred_pc.shape[1] > 3:
                            pred_pc = pred_pc[:, :3]
                        if gt_pc.shape[1] > 3:
                            gt_pc = gt_pc[:, :3]
                        
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

def load_kitti_data():
    """Load KITTI Odometry data for train, validation, and test splits"""
    
    # Configuration paths
    config_path = "/home/vishwesh/PPMFNet/configs/parameters.yml"
    data_path = "/DATA/vishwesh/kitti/processed_data"
    
    # Load configuration
    print("Loading configuration from:", config_path)
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Update data path in config
    cfg["DATA_CONFIG"]["PROCESSED_PATH"] = data_path
    # Add raw data path for point clouds
    cfg["DATA_CONFIG"]["RAW_PATH"] = "/DATA/vishwesh/kitti/dataset/sequences"
    
    # Initialize the data module
    print("Initializing KITTI Odometry Data Module...")
    data_module = KittiOdometryModule(cfg)

    # Setup data loaders
    data_module.setup()

    # Get test data loader
    test_loader = data_module.test_dataloader()

    return data_module, test_loader, cfg

def main():
    try:
        data_module, test_loader, cfg = load_kitti_data()

        # Initialize the GT forecaster
        print("Initializing GT Point Cloud Forecaster (PC->PC)...")
        forecaster = GTPointCloudForecaster(cfg)
        
        # Move chamfer distance module to CUDA
        forecaster.chamfer_distance.to('cuda')
        
        # Run evaluation
        print("\nRunning GT pose forecasting evaluation (Point Cloud -> Point Cloud)...")
        results = forecaster.evaluate_forecasting(test_loader)
        
        # Print results
        print("\n" + "="*50)
        print("GT POSE FORECASTING RESULTS (PC->PC)")
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