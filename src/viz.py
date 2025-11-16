#!/usr/bin/env python3
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
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

class PointCloudVisualizer:
    """Visualize point cloud transformations with chamfer distance computation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize chamfer distance calculator
        self.chamfer_calculator = chamfer_dist_pc(cfg)
        
        # Get projection parameters for range image conversion
        self.fov_up = cfg["DATA_CONFIG"]["FOV_UP"] * np.pi / 180  # Convert to radians
        self.fov_down = cfg["DATA_CONFIG"]["FOV_DOWN"] * np.pi / 180
        self.H = cfg["DATA_CONFIG"]["HEIGHT"]  # Range image height
        self.W = cfg["DATA_CONFIG"]["WIDTH"]   # Range image width

    def compute_chamfer_distance(self, pc1, pc2):
        """Compute chamfer distance between two point clouds
        
        Args:
            pc1: (N, 3) tensor - first point cloud
            pc2: (M, 3) tensor - second point cloud
            
        Returns:
            chamfer_dist: float - chamfer distance
        """
        if pc1.shape[0] == 0 or pc2.shape[0] == 0:
            return float('inf')
        
        # Ensure both point clouds are on the same device
        pc1 = pc1.to(self.device)
        pc2 = pc2.to(self.device)
        
        # Add batch dimension
        pc1_batch = pc1.unsqueeze(0)  # (1, N, 3)
        pc2_batch = pc2.unsqueeze(0)  # (1, M, 3)
        
        try:
            chamfer_distances, chamfer_distances_tensor = self.chamfer_calculator(pc1_batch, pc2_batch)
            return chamfer_distances[0].item()
        except Exception as e:
            print(f"Warning: Chamfer distance computation failed: {e}")
            return float('inf')

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
    
    def transform_to_world_coordinates(self, points, pose, calibration):
        """Transform point cloud from lidar coordinates to world coordinates
        
        Args:
            points: (N, 3) or (N, 4) tensor of points in lidar coordinates
            pose: (4, 4) camera-to-world transform matrix
            calibration: (4, 4) lidar-to-camera transform matrix
            
        Returns:
            world_points: (N, 3) points in world coordinates
        """
        device = points.device
        
        # Take only xyz coordinates if points have intensity
        if points.shape[1] == 4:
            xyz_points = points[:, :3]  # (N, 3)
        else:
            xyz_points = points
            
        N = xyz_points.shape[0]
        
        # Convert to homogeneous coordinates
        ones = torch.ones(N, 1, device=device)
        points_homo = torch.cat([xyz_points, ones], dim=1)  # (N, 4)
        
        # Transform sequence:
        # 1. Lidar -> Camera: calibration
        # 2. Camera -> World: pose
        transform_matrix = pose @ calibration
        
        # Apply transformation: (4, 4) @ (4, N) -> (4, N)
        world_points_homo = transform_matrix @ points_homo.T  # (4, N)
        world_points = world_points_homo[:3].T  # (N, 3)
        
        return world_points

    def plot_comprehensive_bev_with_chamfer(self, current_pc, fut_pcs, current_pose, fut_poses, 
                                          calibration, current_data, fut_data, save_path):
        """Plot comprehensive 4x5 BEV visualization with chamfer distances
        
        Args:
            current_pc: (N, 3) current point cloud
            fut_pcs: List of (N, 3) future point clouds
            current_pose: (4, 4) current pose
            fut_poses: List of (4, 4) future poses
            calibration: (4, 4) calibration matrix
            current_data: (4, H, W) current range image data
            fut_data: List of (4, H, W) future range image data
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(4, 5, figsize=(30, 24))
        
        # Row titles
        row_titles = [
            'BEV: Current PC + Future PC (World) + Chamfer Distance',
            'BEV: Transformed Current PC + Future PC (Local) + Chamfer Distance',
            'BEV: Current Data + Future Data (World) + Chamfer Distance',
            'BEV: Transformed Current Data + Future Data (Local) + Chamfer Distance'
        ]
        
        # Convert range data to point clouds
        current_pc_from_data = self.range_data_to_point_cloud(current_data)
        fut_pcs_from_data = [self.range_data_to_point_cloud(data) for data in fut_data]
        
        # Transform to world coordinates
        current_pc_world = self.transform_to_world_coordinates(current_pc, current_pose, calibration)
        fut_pcs_world = [self.transform_to_world_coordinates(pc, pose, calibration) 
                        for pc, pose in zip(fut_pcs, fut_poses)]
        
        current_data_world = self.transform_to_world_coordinates(current_pc_from_data, current_pose, calibration)
        fut_data_world = [self.transform_to_world_coordinates(pc, pose, calibration) 
                         for pc, pose in zip(fut_pcs_from_data, fut_poses)]
        
        # Calculate global bounds for consistent scaling (using X and Z for BEV in world coordinates)
        all_points = [current_pc_world] + fut_pcs_world + [current_data_world] + fut_data_world
        all_x = torch.cat([pc[:, 0] for pc in all_points if pc.shape[0] > 0])
        all_z = torch.cat([pc[:, 2] for pc in all_points if pc.shape[0] > 0])
        x_min, x_max = all_x.min().item() - 10, all_x.max().item() + 10
        z_min, z_max = all_z.min().item() - 10, all_z.max().item() + 10
        
        # Store chamfer distances for summary
        chamfer_results = {
            'world_pc': [],
            'local_pc': [],
            'world_data': [],
            'local_data': []
        }
        
        for t in range(5):  # 5 future timesteps
            # Row 0: Current PC + Future PC (World) - X vs Z
            ax = axes[0, t]
            current_pc_np = current_pc_world.cpu().numpy()
            ax.scatter(current_pc_np[:, 0], current_pc_np[:, 2], 
                      c='blue', s=0.1, alpha=0.6, label='Current PC')
            
            chamfer_dist_world_pc = float('inf')
            if t < len(fut_pcs_world):
                fut_pc_np = fut_pcs_world[t].cpu().numpy()
                ax.scatter(fut_pc_np[:, 0], fut_pc_np[:, 2], 
                          c='red', s=0.1, alpha=0.6, label=f'Future PC t+{t+1}')
                
                # Compute chamfer distance for world coordinates (using X,Z points for BEV comparison)
                current_world_xz = current_pc_world[:, [0, 2]]  # X, Z coordinates
                future_world_xz = fut_pcs_world[t][:, [0, 2]]   # X, Z coordinates
                # Pad with zeros for Y coordinate to make it 3D for chamfer calculation
                current_world_3d = torch.cat([current_world_xz[:, :1], torch.zeros(current_world_xz.shape[0], 1, device=current_world_xz.device), current_world_xz[:, 1:]], dim=1)
                future_world_3d = torch.cat([future_world_xz[:, :1], torch.zeros(future_world_xz.shape[0], 1, device=future_world_xz.device), future_world_xz[:, 1:]], dim=1)
                chamfer_dist_world_pc = self.compute_chamfer_distance(current_world_3d, future_world_3d)
                chamfer_results['world_pc'].append(chamfer_dist_world_pc)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            title_text = f't+{t+1}'
            if chamfer_dist_world_pc != float('inf'):
                title_text += f'\nCD: {chamfer_dist_world_pc:.3f}m'
            ax.set_title(title_text)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            if t == 0:
                ax.legend(markerscale=10)
                ax.set_ylabel(row_titles[0], fontsize=10, rotation=90, labelpad=20)
            
            # Row 1: Transformed Current PC + Future PC (Future Local Frame) - X vs Y BEV
            ax = axes[1, t]
            chamfer_dist_local_pc = float('inf')
            if t < len(fut_poses):
                # Transform current PC to future frame
                transformed_current = self.transform_point_cloud(
                    current_pc, current_pose, fut_poses[t], calibration
                )
                transformed_np = transformed_current.cpu().numpy()
                ax.scatter(transformed_np[:, 0], transformed_np[:, 1], 
                          c='blue', s=0.1, alpha=0.6, label='Transformed Current')
                
                # Future PC in its own frame (identity transform) - X vs Y BEV view
                fut_pc_np = fut_pcs[t].cpu().numpy()
                ax.scatter(fut_pc_np[:, 0], fut_pc_np[:, 1], 
                          c='red', s=0.1, alpha=0.6, label=f'Future PC t+{t+1}')
                
                # Compute chamfer distance for local coordinates (using X,Y points for BEV comparison)
                transformed_current_xy = transformed_current[:, [0, 1]]  # X, Y coordinates
                future_local_xy = fut_pcs[t][:, [0, 1]]                 # X, Y coordinates
                # Pad with zeros for Z coordinate
                transformed_current_3d = torch.cat([transformed_current_xy, torch.zeros(transformed_current_xy.shape[0], 1, device=transformed_current_xy.device)], dim=1)
                future_local_3d = torch.cat([future_local_xy, torch.zeros(future_local_xy.shape[0], 1, device=future_local_xy.device)], dim=1)
                chamfer_dist_local_pc = self.compute_chamfer_distance(transformed_current_3d, future_local_3d)
                chamfer_results['local_pc'].append(chamfer_dist_local_pc)
                
                # Calculate local bounds for X vs Y BEV
                if transformed_np.shape[0] > 0 and fut_pc_np.shape[0] > 0:
                    local_x = np.concatenate([transformed_np[:, 0], fut_pc_np[:, 0]])
                    local_y = np.concatenate([transformed_np[:, 1], fut_pc_np[:, 1]])
                    local_x_min, local_x_max = local_x.min() - 5, local_x.max() + 5
                    local_y_min, local_y_max = local_y.min() - 5, local_y.max() + 5
                    ax.set_xlim(local_x_min, local_x_max)
                    ax.set_ylim(local_y_min, local_y_max)
            
            title_text = f't+{t+1}'
            if chamfer_dist_local_pc != float('inf'):
                title_text += f'\nCD: {chamfer_dist_local_pc:.3f}m'
            ax.set_title(title_text)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            if t == 0:
                ax.legend(markerscale=10)
                ax.set_ylabel(row_titles[1], fontsize=10, rotation=90, labelpad=20)
            
            # Row 2: Current Data + Future Data (World) - X vs Z
            ax = axes[2, t]
            current_data_np = current_data_world.cpu().numpy()
            ax.scatter(current_data_np[:, 0], current_data_np[:, 2], 
                      c='blue', s=0.1, alpha=0.6, label='Current Data')
            
            chamfer_dist_world_data = float('inf')
            if t < len(fut_data_world):
                fut_data_np = fut_data_world[t].cpu().numpy()
                ax.scatter(fut_data_np[:, 0], fut_data_np[:, 2], 
                          c='red', s=0.1, alpha=0.6, label=f'Future Data t+{t+1}')
                
                # Compute chamfer distance for world data (X,Z coordinates)
                current_data_xz = current_data_world[:, [0, 2]]
                future_data_xz = fut_data_world[t][:, [0, 2]]
                # Pad with zeros for Y coordinate
                current_data_3d = torch.cat([current_data_xz[:, :1], torch.zeros(current_data_xz.shape[0], 1, device=current_data_xz.device), current_data_xz[:, 1:]], dim=1)
                future_data_3d = torch.cat([future_data_xz[:, :1], torch.zeros(future_data_xz.shape[0], 1, device=future_data_xz.device), future_data_xz[:, 1:]], dim=1)
                chamfer_dist_world_data = self.compute_chamfer_distance(current_data_3d, future_data_3d)
                chamfer_results['world_data'].append(chamfer_dist_world_data)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            title_text = f't+{t+1}'
            if chamfer_dist_world_data != float('inf'):
                title_text += f'\nCD: {chamfer_dist_world_data:.3f}m'
            ax.set_title(title_text)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            if t == 0:
                ax.legend(markerscale=10)
                ax.set_ylabel(row_titles[2], fontsize=10, rotation=90, labelpad=20)
            
            # Row 3: Transformed Current Data + Future Data (Future Local Frame) - X vs Y BEV
            ax = axes[3, t]
            chamfer_dist_local_data = float('inf')
            if t < len(fut_poses) and t < len(fut_pcs_from_data):
                # Transform current data to future frame
                transformed_current_data = self.transform_point_cloud(
                    current_pc_from_data, current_pose, fut_poses[t], calibration
                )
                transformed_data_np = transformed_current_data.cpu().numpy()
                ax.scatter(transformed_data_np[:, 0], transformed_data_np[:, 1], 
                          c='blue', s=0.1, alpha=0.6, label='Transformed Current Data')
                
                # Future data in its own frame - X vs Y BEV view
                fut_data_local_np = fut_pcs_from_data[t].cpu().numpy()
                ax.scatter(fut_data_local_np[:, 0], fut_data_local_np[:, 1], 
                          c='red', s=0.1, alpha=0.6, label=f'Future Data t+{t+1}')
                
                # Compute chamfer distance for local data (X,Y coordinates)
                transformed_data_xy = transformed_current_data[:, [0, 1]]
                future_data_local_xy = fut_pcs_from_data[t][:, [0, 1]]
                # Pad with zeros for Z coordinate
                transformed_data_3d = torch.cat([transformed_data_xy, torch.zeros(transformed_data_xy.shape[0], 1, device=transformed_data_xy.device)], dim=1)
                future_data_local_3d = torch.cat([future_data_local_xy, torch.zeros(future_data_local_xy.shape[0], 1, device=future_data_local_xy.device)], dim=1)
                chamfer_dist_local_data = self.compute_chamfer_distance(transformed_data_3d, future_data_local_3d)
                chamfer_results['local_data'].append(chamfer_dist_local_data)
                
                # Calculate local bounds for X vs Y BEV
                if transformed_data_np.shape[0] > 0 and fut_data_local_np.shape[0] > 0:
                    local_x = np.concatenate([transformed_data_np[:, 0], fut_data_local_np[:, 0]])
                    local_y = np.concatenate([transformed_data_np[:, 1], fut_data_local_np[:, 1]])
                    local_x_min, local_x_max = local_x.min() - 5, local_x.max() + 5
                    local_y_min, local_y_max = local_y.min() - 5, local_y.max() + 5
                    ax.set_xlim(local_x_min, local_x_max)
                    ax.set_ylim(local_y_min, local_y_max)
            
            title_text = f't+{t+1}'
            if chamfer_dist_local_data != float('inf'):
                title_text += f'\nCD: {chamfer_dist_local_data:.3f}m'
            ax.set_title(title_text)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            if t == 0:
                ax.legend(markerscale=10)
                ax.set_ylabel(row_titles[3], fontsize=10, rotation=90, labelpad=20)
        
        # Set appropriate labels for each row
        # Rows 0 and 2 (World coordinates): X vs Z
        for ax in axes[0, :]:  # Row 0
            ax.set_xlabel('X (m) - Forward/Backward')
        for ax in axes[2, :]:  # Row 2
            ax.set_xlabel('X (m) - Forward/Backward')
            
        # Rows 1 and 3 (Local coordinates): X vs Y
        for ax in axes[1, :]:  # Row 1
            ax.set_xlabel('X (m) - Forward/Backward')
        for ax in axes[3, :]:  # Row 3
            ax.set_xlabel('X (m) - Forward/Backward')
        
        # Y-axis labels
        for i, ax in enumerate(axes[:, 0]):
            if i in [0, 2]:  # World coordinate rows (X vs Z)
                ax.set_ylabel(ax.get_ylabel() + '\nZ (m) - Left/Right', fontsize=10)
            else:  # Local coordinate rows (X vs Y)
                ax.set_ylabel(ax.get_ylabel() + '\nY (m) - Left/Right', fontsize=10)
        
        # Add summary text with chamfer distance statistics
        summary_text = "Chamfer Distance Summary (mean ± std):\n"
        for key, values in chamfer_results.items():
            if values:
                valid_values = [v for v in values if v != float('inf')]
                if valid_values:
                    mean_cd = np.mean(valid_values)
                    std_cd = np.std(valid_values)
                    summary_text += f"{key}: {mean_cd:.3f} ± {std_cd:.3f}m\n"
                else:
                    summary_text += f"{key}: No valid distances\n"
            else:
                summary_text += f"{key}: No data\n"
        
        # plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
        #            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Point Cloud Visualization with Chamfer Distances (CD in meters)', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive visualization with chamfer distances to: {save_path}")
        
        # Print chamfer distance summary
        print("\n📊 Chamfer Distance Summary:")
        for key, values in chamfer_results.items():
            if values:
                valid_values = [v for v in values if v != float('inf')]
                if valid_values:
                    mean_cd = np.mean(valid_values)
                    std_cd = np.std(valid_values)
                    print(f"  {key}: {mean_cd:.3f} ± {std_cd:.3f}m (over {len(valid_values)} timesteps)")
                    print(f"    Individual: {[f'{v:.3f}' for v in valid_values]}")
                else:
                    print(f"  {key}: No valid distances")
            else:
                print(f"  {key}: No data")
        
        plt.show()
        return chamfer_results
    
    def visualize_transformations(self, data_loader, save_dir, batch_idx=0):
        """Visualize point cloud transformations for a specific sample with chamfer distances
        
        Args:
            data_loader: DataLoader to get batches from
            save_dir: Directory to save visualizations
            batch_idx: Specific batch index to process (default: 0)
        """
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get specific batch
            target_batch = None
            for current_idx, batch in enumerate(data_loader):
                if current_idx == batch_idx:
                    target_batch = batch
                    break
            
            if target_batch is None:
                print(f"❌ Batch index {batch_idx} not found in data loader!")
                return
            
            print(f"Processing batch {batch_idx} with chamfer distance computation...")
            
            # Move data to device
            past_poses = target_batch['past_poses'].to(self.device)  # (B, 5, 4, 4)
            fut_poses = target_batch['fut_poses'].to(self.device)    # (B, 5, 4, 4)
            calibration = target_batch['calibration'][0].to(self.device)  # (4, 4)
            
            # Get raw point clouds
            past_pc = target_batch['past_pc']  # List of length B
            fut_pc = target_batch['fut_pc']    # List of length B
            
            # Get range image data
            past_data = target_batch['past_data'].to(self.device)  # (B, 5, 4, H, W)
            fut_data = target_batch['fut_data'].to(self.device)    # (B, 5, 4, H, W)
            
            # Process first sample in batch
            b = 0
            current_pc = past_pc[b][-1].to(self.device)  # (N, 4) - last past timestep
            current_pose = past_poses[b, -1]             # (4, 4) - last past pose
            current_data = past_data[b, -1]              # (4, H, W) - last past data
            
            print(f"Current PC shape: {current_pc.shape}")
            print(f"Current pose shape: {current_pose.shape}")
            print(f"Current data shape: {current_data.shape}")
            
            # Prepare future data
            fut_pcs_list = []
            fut_poses_list = []
            fut_data_list = []
            
            T_fut = fut_poses.shape[1]  # Number of future timesteps
            
            for t_fut in range(T_fut):
                fut_pcs_list.append(fut_pc[b][t_fut][:, :3].to(self.device))  # Take only xyz
                fut_poses_list.append(fut_poses[b, t_fut])
                fut_data_list.append(fut_data[b, t_fut])
            
            # Create comprehensive visualization with chamfer distances
            save_path = os.path.join(save_dir, f"comprehensive_bev_4x5_chamfer_batch_{batch_idx}.png")
            chamfer_results = self.plot_comprehensive_bev_with_chamfer(
                current_pc[:, :3],  # Take only xyz
                fut_pcs_list,
                current_pose,
                fut_poses_list,
                calibration,
                current_data,
                fut_data_list,
                save_path
            )
            
            print(f"✅ Comprehensive visualization with chamfer distances completed for batch {batch_idx}!")
            return chamfer_results
                

def load_kitti_data():
    """Load KITTI Odometry data"""
    
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

    # Get test data loader with batch size 1 for visualization
    test_loader = data_module.test_dataloader()

    return data_module, test_loader, cfg

def main():
    try:
        # Load data
        data_module, test_loader, cfg = load_kitti_data()

        # Initialize visualizer
        print("Initializing Point Cloud Visualizer...")
        visualizer = PointCloudVisualizer(cfg)
        
        # Create visualization
        save_dir = "/home/vishwesh/PPMFNet/media"
        
        # You can specify which batch to visualize here
        target_batch_idx = 72
        
        print(f"\nCreating comprehensive visualizations with chamfer distances for batch {target_batch_idx}...")
        print(f"Save directory: {save_dir}")
        
        chamfer_results = visualizer.visualize_transformations(test_loader, save_dir, batch_idx=target_batch_idx)
        
        if chamfer_results:
            print(f"\n✅ Comprehensive visualization with chamfer distances completed successfully for batch {target_batch_idx}!")
            print("\n📈 Final Chamfer Distance Analysis:")
            for key, values in chamfer_results.items():
                avg_chamfer = np.mean([v for v in values if v != float('inf')])
                print(f"{key:12}: avg = {avg_chamfer:.4f}, values = {values}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()