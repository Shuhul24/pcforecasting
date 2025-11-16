import torch
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# from utils import data_util
# from utils import yaml_util, model_util,viz_util
# from inference import postprocess
import datetime
import argparse
from pytorch3d.transforms import so3_log_map, so3_exp_map
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import wandb
from preet.utils import range_projection
from preet.utils import projection
from preet.prior_util import load_contact_module, load_noise_scheduler
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")
torch.set_printoptions(precision=10)
torch.manual_seed(0)
np.random.seed(0)
# from src.utils.projection import projection
# import scenepic as sp
from src.gt_pose_forecast_1 import load_kitti_data, GTPointCloudForecaster

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_file', type=str, default="config_files/train_contact_config.yaml")
args = parser.parse_args()
print(args)
from src.models.chamfer import cham_dist
from tqdm import tqdm
from preet.unetdiff import FrameByFrameDiffusion
# Import your specific dataloader, functions, and config
# from your_project.dataset import YourDatasetClass 
from torch.utils.data import DataLoader
# from your_project.utils import pcf, range_projection
# from your_project.config import cfg
from src.gt_pose_forecast_1 import load_kitti_data, GTPointCloudForecaster
def precompute_and_save():
    """
    Loops through the dataset once to precompute and save the deterministic
    pose-based forecasts.
    """
    data_module, test_loader, cfg = load_kitti_data(split='test')
    pcf=GTPointCloudForecaster(cfg)
    Projection = projection(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup ---
    # 1. Create a directory to store the precomputed files
    save_dir = os.path.join('/DATA2/shuhul/kitti', "precomputed_forecasts_train")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving precomputed data to: {save_dir}")

    # 2. Load your dataset. IMPORTANT: Use shuffle=False to keep order!
    
    # --- Precomputation Loop ---
    for batch in tqdm(test_loader, desc="Precomputing Forecasts"):
        # We need a way to identify each sample to create a unique filename.
        # Your batch should contain some metadata, like a frame ID or index.
        # Here, I'm assuming batch['metadata'] exists.
        metadata = batch['meta'] 

        past_data = batch['past_data'].to(device)
        past_poses = batch['past_poses'].to(device)
        fut_poses = batch['fut_poses'].to(device)
        calibration = batch['calibration'][0].to(device)
        
        B, T_fut, H, W = past_data.shape[0], fut_poses.shape[1], 64, 2048
        predicted_range_batch = torch.zeros((B, T_fut, H, W)).to(device)

        with torch.no_grad():
            for b in range(B):
                current_data = past_data[b, -1]  
                current_pose = past_poses[b, -1]
                for t_fut in range(T_fut):
                    target_pose = fut_poses[b, t_fut]
                    
                    pred_pc = pcf.forecast_point_cloud(
                        current_data, current_pose, target_pose, calibration
                    )
                    intensity_placeholder = torch.zeros((pred_pc.shape[0], 1), device=pred_pc.device)
                    pred_pc_with_intensity = torch.cat([pred_pc, intensity_placeholder], dim=1)
                    range_image0 = range_projection(
                            pred_pc_with_intensity.cpu().numpy(),
                            fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                            fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                            proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                            proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                            max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
                        )
                    range_image_tensor = torch.from_numpy(range_image0[0]).float().to(device)
                    predicted_range_batch[b, t_fut] = range_image_tensor
        
        # --- Save to Disk ---
        # Convert to NumPy and save each sample in the batch individually
        predicted_range_numpy = predicted_range_batch.cpu().numpy()
        for b in range(B):
            # Create a unique filename based on the sample's metadata
            seq, scan_idx = metadata[b]
            sample_id = f"{seq}_{scan_idx}" # Creates a clean filename like "0_2263"
            save_path = os.path.join(save_dir, f"{sample_id}.npy")
            np.save(save_path, predicted_range_numpy[b])

if __name__ == '__main__':
    precompute_and_save()