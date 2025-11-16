import os
import time
import argparse
import yaml

from utils.preprocess_data import prepare_data
from utils.process_nuscenes import prepare_data_trainval

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./preprocess_data.py")
    parser.add_argument(
        "--dataset", "-d", type=str, default="kitti", help="Name of dataset: kitti or nuscenes"
    )
    parser.add_argument(
        "--dataset_path", "-dp", type=str, default=None, help="Path to the raw dataset"
    )
    parser.add_argument(
        "--processed_path", "-pp", type=str, default=None, help="Path to save the processed dataset to"
    )

    args, unparsed = parser.parse_known_args()

    config_filename = "./configs/parameters.yml" if args.dataset=="kitti" else "./configs/nuscenes_parameters.yml"
    cfg = yaml.safe_load(open(config_filename))

    cfg["DATA_CONFIG"]["RAW_DATASET_PATH"] = args.dataset_path
    cfg["DATA_CONFIG"]["PROCESSED_PATH"] = args.processed_path

    if args.dataset=="kitti":
        prepare_data(cfg)
    elif args.dataset=="nuscenes":
        prepare_data_trainval(cfg)


