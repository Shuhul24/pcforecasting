#!/usr/bin/env python3
import os
import time
import argparse
import yaml
import subprocess
import sys

from datasets.datasets_kitti import KittiOdometryModule

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
    
    print("Configuration loaded successfully!")
    print(f"Data path: {data_path}")
    print(f"Training sequences: {cfg['DATA_CONFIG']['SPLIT']['TRAIN']}")
    print(f"Validation sequences: {cfg['DATA_CONFIG']['SPLIT']['VAL']}")
    print(f"Test sequences: {cfg['DATA_CONFIG']['SPLIT']['TEST']}")
    print(f"Batch size: {cfg['TRAIN']['BATCH_SIZE']}")
    print(f"Number of workers: {cfg['DATA_CONFIG']['DATALOADER']['NUM_WORKER']}")
    print(f"Past steps: {cfg['MODEL']['N_PAST_STEPS']}")
    print(f"Future steps: {cfg['MODEL']['N_FUTURE_STEPS']}")
    print("-" * 60)
    
    # Initialize the data module
    print("Initializing KITTI Odometry Data Module...")
    data_module = KittiOdometryModule(cfg)
    
    # Prepare data (if needed)
    print("Preparing data...")
    start_time = time.time()
    data_module.prepare_data()
    prepare_time = time.time() - start_time
    print(f"Data preparation completed in {prepare_time:.2f} seconds")
    
    # Setup data loaders
    print("Setting up data loaders...")
    start_time = time.time()
    data_module.setup()
    setup_time = time.time() - start_time
    print(f"Data loader setup completed in {setup_time:.2f} seconds")
    print("-" * 60)
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("-" * 60)
    
    # Test loading a batch from each split
    print("Testing data loading...")
    
    try:
        # Load a training batch
        print("Loading training batch...")
        start_time = time.time()
        train_batch = next(iter(train_loader))
        train_time = time.time() - start_time
        print(f"✓ Training batch loaded successfully in {train_time:.3f} seconds")
        print(f"  Past data shape: {train_batch['past_data'].shape}")
        print(f"  Future data shape: {train_batch['fut_data'].shape}")
        print(f"  Past poses shape: {train_batch['past_poses'].shape}")
        print(f"  Future poses shape: {train_batch['fut_poses'].shape}")
        print(f"  Meta info: {train_batch['meta']}")
        
        # Load a validation batch
        print("\nLoading validation batch...")
        start_time = time.time()
        val_batch = next(iter(val_loader))
        val_time = time.time() - start_time
        print(f"✓ Validation batch loaded successfully in {val_time:.3f} seconds")
        print(f"  Past data shape: {val_batch['past_data'].shape}")
        print(f"  Future data shape: {val_batch['fut_data'].shape}")
        print(f"  Past poses shape: {val_batch['past_poses'].shape}")
        print(f"  Future poses shape: {val_batch['fut_poses'].shape}")
        print(f"  Meta info: {val_batch['meta']}")
        
        # Load a test batch
        print("\nLoading test batch...")
        start_time = time.time()
        test_batch = next(iter(test_loader))
        test_time = time.time() - start_time
        print(f"✓ Test batch loaded successfully in {test_time:.3f} seconds")
        print(f"  Past data shape: {test_batch['past_data'].shape}")
        print(f"  Future data shape: {test_batch['fut_data'].shape}")
        print(f"  Past poses shape: {test_batch['past_poses'].shape}")
        print(f"  Future poses shape: {test_batch['fut_poses'].shape}")
        print(f"  Meta info: {test_batch['meta']}")
        
        print("-" * 60)
        print("✅ All data loaders working correctly!")
        
        # Test iterating through a few batches
        print("\nTesting iteration through training data...")
        batch_count = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            batch_count += 1
            if i >= 2:  # Test first 3 batches
                break
        
        iteration_time = time.time() - start_time
        print(f"✓ Successfully iterated through {batch_count} training batches in {iteration_time:.3f} seconds")
        
        return data_module, train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("Please check:")
        print("1. Data path exists and contains processed sequences")
        print("2. Required utility modules are available")
        print("3. File permissions are correct")
        raise e

def main():
    """Main function to run the data loading test"""
    print("=" * 60)
    print("KITTI Odometry Data Loading Test")
    print("=" * 60)
    
    try:
        data_module, train_loader, val_loader, test_loader = load_kitti_data()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Data loading completed successfully!")
        print("You can now use the data loaders for training/validation/testing.")
        print("=" * 60)
        
        # Optional: Save some sample data info
        sample_item = train_loader.dataset[0]
        print(f"\nSample data ranges:")
        print(f"Past data - min: {sample_item['past_data'].min():.3f}, max: {sample_item['past_data'].max():.3f}")
        print(f"Future data - min: {sample_item['fut_data'].min():.3f}, max: {sample_item['fut_data'].max():.3f}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()