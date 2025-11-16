# # File: compute_stats.py
# import torch
# from tqdm import tqdm
# # --- Import your specific Dataloader and Config ---
# from torch.utils.data import DataLoader
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# from src.gt_pose_forecast_1 import load_kitti_data, GTPointCloudForecaster

# def compute_normalization_stats():
#     """
#     Performs a two-pass calculation over the training dataset to find the
#     mean and standard deviation of the target delta (`naction`).
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     data_module, test_loader, cfg = load_kitti_data()
#     # 1. Load your TRAINING dataset
#     # dataset = KittiOdometryRaw(cfg=cfg, split='train')
#     # dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

#     # --- Pass 1: Calculate the Mean ---
#     sum_of_elements = torch.tensor(0.0, dtype=torch.float64, device=device)
#     num_elements = 0
#     for batch in tqdm(test_loader, desc="Pass 1/2 (Mean)"):
#         fut_data = batch['fut_data'].to(device)
#         predicted_range = batch['predicted_range'].to(device)
#         naction = fut_data[:, :, 0, :, :] - predicted_range
#         sum_of_elements += torch.sum(naction)
#         num_elements += naction.numel()
#     mean = sum_of_elements / num_elements

#     # --- Pass 2: Calculate the Standard Deviation ---
#     sum_of_squared_diffs = torch.tensor(0.0, dtype=torch.float64, device=device)
#     # Re-initialize the dataloader to iterate from the beginning
#     dataloader_pass2 = data_module.test_loader()
#     for batch in tqdm(dataloader_pass2, desc="Pass 2/2 (Std Dev)"):
#         fut_data = batch['fut_data'].to(device)
#         predicted_range = batch['predicted_range'].to(device)
#         naction = fut_data[:, :, 0, :, :] - predicted_range
#         sum_of_squared_diffs += torch.sum((naction - mean) ** 2)
#     std = torch.sqrt(sum_of_squared_diffs / num_elements)

#     # --- Save the stats to a file ---
#     stats = {'mean': mean.float(), 'std': std.float()}
#     save_path = "norm_stats.pt"
#     torch.save(stats, save_path)
#     print(f"\nStats saved to '{save_path}': Mean={mean.item()}, Std={std.item()}")

# if __name__ == '__main__':
#     compute_normalization_stats()
# File: compute_stats.py
import torch
from tqdm import tqdm
# --- Import your specific Dataloader and Config ---
from torch.utils.data import DataLoader
from src.datasets.datasets_kitti import KittiOdometryModule # Make sure this path is correct
# from configs import cfg # Your config file
import yaml
def compute_normalization_stats():
    """
    Performs a two-pass calculation over the training dataset to find the
    mean and standard deviation of the target delta (`naction`).
    """
    # Use a relative path to the config file within your project
    config_path = os.path.join(parent_dir, "configs", "parameters.yml")
    
    # Load configuration
    print("Loading configuration from:", config_path)
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg["DATA_CONFIG"]["PROCESSED_PATH"] = "/DATA2/shuhul/kitti/processed_data"
    cfg["DATA_CONFIG"]["RAW_PATH"] = "/DATA2/shuhul/kitti/dataset/sequences"
    # 1. Load your TRAINING dataset
    # dataset = KittiOdometryRaw(cfg=cfg, split='test')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    data_module = KittiOdometryModule(cfg)

    # Setup data loaders
    data_module.setup()

    # Get test data loader
    test_loader = data_module.test_dataloader()
    # --- Pass 1: Calculate the Mean ---
    sum_of_elements = torch.tensor(0.0, dtype=torch.float64, device=device)
    num_elements = 0
    for batch in tqdm(test_loader, desc="Pass 1/2 (Mean)"):
        fut_data = batch['fut_data'].to(device)
        predicted_range = batch['predicted_range'].to(device)
        naction = fut_data[:, :, 0, :, :] - predicted_range
        sum_of_elements += torch.sum(naction)
        num_elements += naction.numel()
    mean = sum_of_elements / num_elements
    data_module = KittiOdometryModule(cfg)

    # Setup data loaders
    data_module.setup()

    # Get test data loader
    test_loader = data_module.test_dataloader()
    # --- Pass 2: Calculate the Standard Deviation ---
    sum_of_squared_diffs = torch.tensor(0.0, dtype=torch.float64, device=device)
    # Re-initialize the dataloader to iterate from the beginning
    # dataloader_pass2 = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    for batch in tqdm(test_loader, desc="Pass 2/2 (Std Dev)"):
        fut_data = batch['fut_data'].to(device)
        predicted_range = batch['predicted_range'].to(device)
        naction = fut_data[:, :, 0, :, :] - predicted_range
        sum_of_squared_diffs += torch.sum((naction - mean) ** 2)
    std = torch.sqrt(sum_of_squared_diffs / num_elements)

    # --- Save the stats to a file ---
    stats = {'mean': mean.float(), 'std': std.float()}
    save_path = "norm_stats.pt"
    torch.save(stats, save_path)
    print(f"\nStats saved to '{save_path}': Mean={mean.item()}, Std={std.item()}")

if __name__ == '__main__':
    compute_normalization_stats()