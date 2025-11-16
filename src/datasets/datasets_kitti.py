#!/usr/bin/env python3

import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys

from src.utils.projection import projection
from src.utils.preprocess_data import prepare_data, compute_mean_and_std
from src.utils.utils import load_files

class KittiOdometryModule:
    """A PyTorch module for KITTI Odometry"""

    def __init__(self, cfg):
        """Method to initialize the Kitti Odometry dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        self.cfg = cfg
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

    def prepare_data(self):
        """Call prepare_data method to generate npy range images from raw LiDAR data"""
        if self.cfg["DATA_CONFIG"]["GENERATE_FILES"]:
            prepare_data(self.cfg)

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        ########## Point dataset splits
        train_set = KittiOdometryRaw(self.cfg, split="train")

        val_set = KittiOdometryRaw(self.cfg, split="val")

        test_set = KittiOdometryRaw(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            
            timeout=0,
            persistent_workers=True,
            collate_fn=self.custom_collate_fn
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True,
            collate_fn=self.custom_collate_fn
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True,
            collate_fn=self.custom_collate_fn
        )
        self.test_iter = iter(self.test_loader)

        # Optionally compute statistics of training data
        if self.cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"]:
            compute_mean_and_std(self.cfg, self.train_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        """Return training dataloader"""
        if self.train_loader is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")
        return self.train_loader

    def val_dataloader(self):
        """Return validation dataloader"""
        if self.valid_loader is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")
        return self.valid_loader

    def test_dataloader(self):
        """Return test dataloader"""
        if self.test_loader is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")
        return self.test_loader

    def get_train_iterator(self):
        """Get training data iterator"""
        if self.train_iter is None:
            raise RuntimeError("Must call setup() before accessing iterators")
        return self.train_iter

    def get_valid_iterator(self):
        """Get validation data iterator"""
        if self.valid_iter is None:
            raise RuntimeError("Must call setup() before accessing iterators")
        return self.valid_iter

    def custom_collate_fn(self, batch):
        """Custom collate function to handle variable-sized point clouds"""
        # Separate different types of data
        past_data = torch.stack([item['past_data'] for item in batch])
        fut_data = torch.stack([item['fut_data'] for item in batch])
        past_poses = torch.stack([item['past_poses'] for item in batch])
        fut_poses = torch.stack([item['fut_poses'] for item in batch])
        calibration = torch.stack([item['calibration'] for item in batch])
        meta = [item['meta'] for item in batch]
        predicted_range=torch.stack([item['predicted_range'] for item in batch])
        # Keep point clouds as lists (can't stack due to variable sizes)
        past_pc = [item['past_pc'] for item in batch]
        fut_pc = [item['fut_pc'] for item in batch]
        
        return {
            'past_data': past_data,
            'fut_data': fut_data,
            'past_pc': past_pc,
            'fut_pc': fut_pc,
            'past_poses': past_poses,
            'fut_poses': fut_poses,
            'calibration': calibration,
            'meta': meta,
            'predicted_range':predicted_range
        }


class KittiOdometryRaw(Dataset):
    """Dataset class for range image-based point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = self.cfg["DATA_CONFIG"]["PROCESSED_PATH"]
        # Add raw data path - you may need to adjust this path in your config
        self.raw_data_path = self.cfg["DATA_CONFIG"].get("RAW_PATH", "/DATA2/shuhul/kitti/dataset/sequences")
        
        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.n_channels = 4

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]

        # Projection class for mapping from range image to 3D point cloud
        self.projection = projection(self.cfg)

        if split == "train":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        elif split == "val":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        elif split == "test":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")
        # breakpoint()
        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames_range = {}
        self.filenames_xyz = {}
        self.filenames_intensity = {}
        self.filenames_semantic = {}
        self.filenames_velodyne = {}  # Add raw point cloud files

        # Store poses and calibration data for each sequence
        self.poses = {}
        self.calibrations = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        self.precompute_dir = os.path.join('/DATA2/shuhul/kitti/', "precomputed_forecasts_train")

        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            scan_path_range = os.path.join(self.root_dir, seqstr, "processed", "range")
            self.filenames_range[seq] = load_files(scan_path_range)

            scan_path_xyz = os.path.join(self.root_dir, seqstr, "processed", "xyz")
            self.filenames_xyz[seq] = load_files(scan_path_xyz)

            # Add raw velodyne point cloud files
            velodyne_path = os.path.join(self.raw_data_path, seqstr, "velodyne")
            self.filenames_velodyne[seq] = load_files(velodyne_path)

            # Load poses and calibration for this sequence
            pose_file = os.path.join(self.root_dir, seqstr, "processed", "poses.txt")
            calib_file = os.path.join(self.root_dir, seqstr, "processed", "calib.txt")
            
            self.poses[seq] = self.load_poses(pose_file)
            self.calibrations[seq] = self.load_calib(calib_file)

            # Get number of sequences based on number of past and future steps
            n_samples_sequence = max(
                0,
                len(self.filenames_range[seq])
                - self.n_past_steps
                - self.n_future_steps
                + 1,
            )
            # breakpoint()
            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.n_past_steps + sample_idx - 1
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence

    def load_poses(self, pose_file):
        """Load pose data from poses.txt file
        
        Args:
            pose_file (str): Path to poses.txt file
            
        Returns:
            list: List of 4x4 pose matrices (W_cam0 - world to camera0 transform)
        """
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

    def load_calib(self, calib_file):
        """Load calibration data from calib.txt file and return as tensor
        
        Args:
            calib_file (str): Path to calib.txt file
            
        Returns:
            torch.Tensor: 4x4 transformation matrix from velodyne to cam0
        """
        with open(calib_file, 'r') as f:
            for line in f:
                if line.strip():
                    key, *values = line.strip().split()
                    if key.rstrip(':') == 'Tr':
                        # Transform from velodyne to cam0 (4x4 matrix)
                        tr_values = np.array([float(v) for v in values])
                        cam0_velo = np.vstack([tr_values.reshape(3, 4), [0, 0, 0, 1]])
                        return torch.tensor(cam0_velo, dtype=torch.float32)
        
        # If no 'Tr' found, return identity matrix
        return torch.eye(4, dtype=torch.float32)

    def load_velodyne_scan(self, filename):
        """Load point cloud from .bin file
        
        Args:
            filename (str): Path to .bin file
            
        Returns:
            torch.Tensor: Point cloud tensor of shape (N, 4) where each point is [x, y, z, intensity]
        """
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # Each point has [x, y, z, intensity]
        return torch.tensor(scan, dtype=torch.float32)

    def get_poses_for_sample(self, seq, scan_idx):
        """Get pose data for past and future steps of a sample
        
        Args:
            seq (int): Sequence number
            scan_idx (int): Current scan index
            
        Returns:
            tuple: (past_poses, future_poses)
        """
        poses_seq = self.poses[seq]
        
        # Past poses
        past_poses = []
        for t in range(self.n_past_steps):
            pose_idx = scan_idx - self.n_past_steps + 1 + t
            past_poses.append(poses_seq[pose_idx])
        
        # Future poses
        future_poses = []
        for t in range(self.n_future_steps):
            pose_idx = scan_idx + 1 + t
            future_poses.append(poses_seq[pose_idx])
        
        return past_poses, future_poses

    def get_calibration(self, seq):
        """Get calibration data for a sequence
        
        Args:
            seq (int): Sequence number
            
        Returns:
            torch.Tensor: 4x4 transformation matrix
        """
        return self.calibrations[seq]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load and concatenate range image channels and raw point clouds

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        # breakpoint()
        seq, scan_idx = self.idx_mapper[idx]

        # Load past data
        past_data = torch.empty(
            [self.n_past_steps, self.n_channels, self.height, self.width]
        )

        from_idx = scan_idx - self.n_past_steps + 1
        to_idx = scan_idx
        past_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        past_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        past_filenames_velodyne = self.filenames_velodyne[seq][from_idx : to_idx + 1]

        # Load past raw point clouds
        past_pc = []
        for t in range(self.n_past_steps):
            past_data[t, 0, :, :] = self.load_range(past_filenames_range[t])
            past_data[t, 1:4, :, :] = self.load_xyz(past_filenames_xyz[t])
            # Load raw point cloud
            pc = self.load_velodyne_scan(past_filenames_velodyne[t])
            past_pc.append(pc)

        # Load future data
        fut_data = torch.empty(
            [self.n_future_steps, self.n_channels, self.height, self.width]
        )

        from_idx = scan_idx + 1
        to_idx = scan_idx + self.n_future_steps
        fut_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        fut_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        fut_filenames_velodyne = self.filenames_velodyne[seq][from_idx : to_idx + 1]

        # Load future raw point clouds
        fut_pc = []
        for t in range(self.n_future_steps):
            fut_data[t, 0, :, :] = self.load_range(fut_filenames_range[t])
            fut_data[t, 1:4, :, :] = self.load_xyz(fut_filenames_xyz[t])
            # Load raw point cloud
            pc = self.load_velodyne_scan(fut_filenames_velodyne[t])
            fut_pc.append(pc)

        # Get pose data
        past_poses, future_poses = self.get_poses_for_sample(seq, scan_idx)
        
        # Get calibration data (now returns tensor directly)
        calib = self.get_calibration(seq)
        sample_id = f"[{int(seq)}, {scan_idx}]"
        # breakpoint()
        precomputed_path = os.path.join(self.precompute_dir, f"{sample_id}.npy")
        
        try:
            # Load the .npy file and convert it to a tensor
            predicted_range = np.load(precomputed_path)
            predicted_range_tensor = torch.from_numpy(predicted_range).float()
        except FileNotFoundError:
            # If a file is missing, print a warning and return a zero tensor
            # This makes your training more robust to errors.
            print(f"WARNING: Precomputed file not found at {precomputed_path}")
            predicted_range_tensor = torch.zeros((self.n_future_steps, self.height, self.width))
        item = {
            "past_data": past_data, 
            "fut_data": fut_data, 
            "past_pc": past_pc,  # List of point cloud tensors
            "fut_pc": fut_pc,    # List of point cloud tensors
            "past_poses": torch.tensor(np.stack(past_poses), dtype=torch.float32),
            "fut_poses": torch.tensor(np.stack(future_poses), dtype=torch.float32),
            "calibration": calib,  # Now a 4x4 tensor
            "meta": (seq, scan_idx),
            "predicted_range": predicted_range_tensor
        }
        return item

    def load_range(self, filename):
        """Load .npy range image as (1,height,width) tensor"""
        rv = torch.Tensor(np.load(filename)).float()
        return rv

    def load_xyz(self, filename):
        """Load .npy xyz values as (3,height,width) tensor"""
        xyz = torch.Tensor(np.load(filename)).float()[:, :, :3]
        xyz = xyz.permute(2, 0, 1)
        return xyz


if __name__ == "__main__":
    config_filename = "./configs/parameters.yml"
    cfg = yaml.safe_load(open(config_filename))
    data = KittiOdometryModule(cfg)
    data.prepare_data()
    data.setup()

    item = data.val_dataloader().dataset.__getitem__(0)

    def normalize(image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image

    # Print pose and calibration information
    print("Past poses shape:", item["past_poses"].shape)
    print("Future poses shape:", item["fut_poses"].shape)
    print("Calibration shape:", item["calibration"].shape)
    
    # Print point cloud information
    print(f"Number of past point clouds: {len(item['past_pc'])}")
    print(f"Number of future point clouds: {len(item['fut_pc'])}")
    
    if len(item['past_pc']) > 0:
        print(f"Past PC[0] shape: {item['past_pc'][0].shape}")
        print(f"Past PC[0] min/max: {item['past_pc'][0].min():.3f} / {item['past_pc'][0].max():.3f}")
    
    if len(item['fut_pc']) > 0:
        print(f"Future PC[0] shape: {item['fut_pc'][0].shape}")
        print(f"Future PC[0] min/max: {item['fut_pc'][0].min():.3f} / {item['fut_pc'][0].max():.3f}")

    print("Velodyne to Camera0 transform:")
    print(item["calibration"])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(30, 30 * 4 * 64 / 2048))

    axs[0].imshow(normalize(item["fut_data"][0, 0, :, :].numpy()))
    axs[0].set_title("Range")
    axs[1].imshow(normalize(item["fut_data"][0, 1, :, :].numpy()))
    axs[1].set_title("X")
    axs[2].imshow(normalize(item["fut_data"][0, 2, :, :].numpy()))
    axs[2].set_title("Y")
    axs[3].imshow(normalize(item["fut_data"][0, 3, :, :].numpy()))
    axs[3].set_title("Z")

    plt.tight_layout()
    plt.show()
    
    # Optionally visualize point cloud statistics
    if len(item['fut_pc']) > 0:
        pc = item['fut_pc'][0]
        print(f"\nPoint cloud statistics for future frame 0:")
        print(f"  Total points: {pc.shape[0]}")
        print(f"  X range: [{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}]")
        print(f"  Y range: [{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}]")
        print(f"  Z range: [{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}]")
        print(f"  Intensity range: [{pc[:, 3].min():.3f}, {pc[:, 3].max():.3f}]")