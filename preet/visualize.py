import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from plyfile import PlyData

def read_ply_points(filepath):
    """Reads x, y, z coordinates from a .ply file."""
    try:
        with open(filepath, 'rb') as f:
            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        return points
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.array([])

def find_and_pair_files(folder_path):
    """Finds and pairs ground truth and prediction .ply files."""
    gt_files = sorted(glob.glob(os.path.join(folder_path, "*_gt_pc.ply")))
    file_pairs = []

    for gt_path in gt_files:
        pred_path = gt_path.replace("_gt_pc.ply", "_pred_pc.ply")
        if os.path.exists(pred_path):
            base_name = os.path.basename(gt_path).replace("_gt_pc.ply", "")
            file_pairs.append((gt_path, pred_path, base_name))
        else:
            print(f"Warning: Missing prediction file for {os.path.basename(gt_path)}")
    return file_pairs

def save_gt_pred_plots(folder_path, output_dir="output_plots_gt_pred", max_points=15000):
    """
    Saves side-by-side 2D scatter plots for Ground Truth vs. Prediction.
    """
    file_pairs = find_and_pair_files(folder_path)
    if not file_pairs:
        print("No matching GT/Pred file pairs found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving GT vs. Pred plots to '{output_dir}/'")

    for gt_path, pred_path, base_name in file_pairs:
        gt_points = read_ply_points(gt_path)
        pred_points = read_ply_points(pred_path)

        if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
            continue

        if gt_points.shape[0] > max_points:
            gt_points = gt_points[np.random.choice(gt_points.shape[0], max_points, replace=False)]
        if pred_points.shape[0] > max_points:
            pred_points = pred_points[np.random.choice(pred_points.shape[0], max_points, replace=False)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Side-by-Side Top-Down View: {base_name}", fontsize=16)

        ax1.scatter(gt_points[:, 0], gt_points[:, 1], s=1, c='blue')
        ax1.set_title("Ground Truth")
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], s=1, c='red')
        ax2.set_title("Prediction")

        for ax in [ax1, ax2]:
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_aspect('equal', adjustable='box')

        all_points = np.vstack((gt_points, pred_points))
        min_bound, max_bound = all_points.min(axis=0), all_points.max(axis=0)
        for ax in [ax1, ax2]:
            ax.set_xlim(min_bound[0], max_bound[0])
            ax.set_ylim(min_bound[1], max_bound[1])

        save_path = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Saved GT/Pred plot for {base_name}")

def save_timeseries_plots(folder_path, file_suffix, output_dir, plot_title_prefix, max_points=15000):
    """
    Groups files by sample and saves a debug image with each timestep on a separate subplot.
    """
    files = glob.glob(os.path.join(folder_path, f"*{file_suffix}"))
    if not files: return

    # Create a dedicated directory for these debug plots
    debug_output_dir = output_dir + "_debug"
    os.makedirs(debug_output_dir, exist_ok=True)
    print(f"\nSaving DEBUG timeseries plots to '{debug_output_dir}/'")

    grouped_files = defaultdict(list)
    for f in files:
        match = re.search(r"batch(\d+)_sample(\d+)", os.path.basename(f))
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            grouped_files[key].append(f)

    for (batch_idx, sample_idx), filepaths in grouped_files.items():
        def get_timestep(filepath):
            match = re.search(r"timestep(\d+)", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        filepaths.sort(key=get_timestep)
        
        num_timesteps = len(filepaths)
        if num_timesteps == 0: continue

        # Create a figure with one subplot per timestep
        fig, axes = plt.subplots(1, num_timesteps, figsize=(5 * num_timesteps, 5))
        if num_timesteps == 1: axes = [axes] # Ensure axes is always a list
        
        fig.suptitle(f"{plot_title_prefix} Debug: Batch {batch_idx}, Sample {sample_idx}", fontsize=16)
        
        all_points_in_sample = []
        loaded_points_list = []

        # First pass: load all points to determine global bounds
        for filepath in filepaths:
            points = read_ply_points(filepath)
            if points.shape[0] > 0:
                if points.shape[0] > max_points:
                    points = points[np.random.choice(points.shape[0], max_points, replace=False)]
                all_points_in_sample.append(points)
                loaded_points_list.append(points)
            else:
                loaded_points_list.append(np.array([]))

        if not all_points_in_sample:
            plt.close(fig)
            continue
        
        # Determine unified axis limits
        global_points = np.vstack(all_points_in_sample)
        min_bound, max_bound = global_points.min(axis=0), global_points.max(axis=0)

        # Second pass: plot each point cloud on its own subplot
        for i, (ax, points) in enumerate(zip(axes, loaded_points_list)):
            if points.shape[0] > 0:
                ax.scatter(points[:, 0], points[:, 1], s=1, c='blue')
            
            timestep = get_timestep(filepaths[i])
            ax.set_title(f"Timestep {timestep}")
            ax.set_xlim(min_bound[0], max_bound[0])
            ax.set_ylim(min_bound[1], max_bound[1])
            ax.set_aspect('equal', adjustable='box')

        base_name = f"batch{batch_idx}_sample{sample_idx}"
        save_path = os.path.join(debug_output_dir, f"{base_name}_debug.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Saved DEBUG plot for {base_name}")
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Set the path to your folder containing ALL .ply files
    FOLDER_PATH = "/home/soham/garments/preet/here/PPMFNet/preet/saved_output"
    # Set the maximum number of points to plot per point cloud
    MAX_DISPLAY_POINTS = 15000
    # -----------------------------------------------------------
    
    # if not os.path.isdir(FOLDER_PATH) or FOLDER_PATH == "/home/soham/garments/preet/here/saved_outputs":
    #      print("Error: Please update the 'FOLDER_PATH' variable to a valid directory.")
    # else:
        # 1. Generate the original side-by-side GT vs. Prediction plots
    save_gt_pred_plots(FOLDER_PATH, max_points=MAX_DISPLAY_POINTS)

    # 2. Generate the Odometry timeseries plots
    save_timeseries_plots(
        folder_path=FOLDER_PATH,
        file_suffix="_odo_pc.ply",
        output_dir="output_plots_odo",
        plot_title_prefix="Odometry",
        max_points=MAX_DISPLAY_POINTS
    )

    # 3. Generate the Input timeseries plots
    save_timeseries_plots(
        folder_path=FOLDER_PATH,
        file_suffix="_input_pc.ply",
        output_dir="output_plots_input",
        plot_title_prefix="Input",
        max_points=MAX_DISPLAY_POINTS
    )
    save_timeseries_plots(
        folder_path=FOLDER_PATH,
        file_suffix="_gt_pc.ply",
        output_dir="output_plots_gt",
        plot_title_prefix="gt",
        max_points=MAX_DISPLAY_POINTS
    )
    save_timeseries_plots(
        folder_path=FOLDER_PATH,
        file_suffix="_pred_pc.ply",
        output_dir="output_plots_pred",
        plot_title_prefix="pred",
        max_points=MAX_DISPLAY_POINTS
    )
# {0: tensor(0.3248646557)}
# {0: tensor(0.3025129139)}
# {0: tensor(0.3234318197)}
# {0: tensor(0.3509832323)}
# def read_ply_points(filepath):
#     """Reads x, y, z coordinates from a .ply file."""
#     try:
#         with open(filepath, 'rb') as f:
#             plydata = PlyData.read(f)
#             vertex = plydata['vertex']
#             points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
#         return points
#     except Exception as e:
#         print(f"Error reading {filepath}: {e}")
#         return np.array([])

# def find_and_pair_files(folder_path):
#     """Finds and pairs ground truth and prediction .ply files."""
#     gt_files = sorted(glob.glob(os.path.join(folder_path, "*_gt_pc.ply")))
#     file_pairs = []

#     for gt_path in gt_files:
#         pred_path = gt_path.replace("_gt_pc.ply", "_pred_pc.ply")
#         if os.path.exists(pred_path):
#             base_name = os.path.basename(gt_path).replace("_gt_pc.ply", "")
#             file_pairs.append((gt_path, pred_path, base_name))
#         else:
#             print(f"Warning: Missing prediction file for {os.path.basename(gt_path)}")
#     return file_pairs

# def save_matplotlib_2d_plots(folder_path, output_dir="output_plots", max_points=10000):
#     """
#     Loads pairs of point clouds and saves 2D scatter plots (top-down view) to files.
#     """
#     file_pairs = find_and_pair_files(folder_path)
    
#     if not file_pairs:
#         print(f"No matching file pairs found in '{folder_path}'")
#         return

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Found {len(file_pairs)} pairs. Saving plots to '{output_dir}/' directory.")

#     for gt_path, pred_path, base_name in file_pairs:
#         # Load point cloud data
#         gt_points = read_ply_points(gt_path)
#         pred_points = read_ply_points(pred_path)

#         if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
#             print(f"  - Skipping {base_name} due to an empty or unreadable point cloud.")
#             continue

#         # --- Subsample the points for performance ---
#         if gt_points.shape[0] > max_points:
#             indices = np.random.choice(gt_points.shape[0], max_points, replace=False)
#             gt_points = gt_points[indices]
        
#         if pred_points.shape[0] > max_points:
#             indices = np.random.choice(pred_points.shape[0], max_points, replace=False)
#             pred_points = pred_points[indices]

#         # --- Create the 2D plots ---
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#         fig.suptitle(f"Side-by-Side Top-Down View: {base_name}", fontsize=16)

#         # Plot Ground Truth (X vs Y)
#         ax1.scatter(gt_points[:, 0], gt_points[:, 1], s=1, c='blue', label='Ground Truth')
#         ax1.set_title("Ground Truth")
#         ax1.set_xlabel("X coordinate")
#         ax1.set_ylabel("Y coordinate")
#         ax1.set_aspect('equal', adjustable='box')

#         # Plot Prediction (X vs Y)
#         ax2.scatter(pred_points[:, 0], pred_points[:, 1], s=1, c='red', label='Prediction')
#         ax2.set_title("Prediction")
#         ax2.set_xlabel("X coordinate")
#         ax2.set_ylabel("Y coordinate")
#         ax2.set_aspect('equal', adjustable='box')

#         # --- Unify axis limits for fair comparison ---
#         all_points = np.vstack((gt_points, pred_points))
#         min_bound = all_points.min(axis=0)
#         max_bound = all_points.max(axis=0)
        
#         ax1.set_xlim(min_bound[0], max_bound[0])
#         ax1.set_ylim(min_bound[1], max_bound[1])
        
#         ax2.set_xlim(min_bound[0], max_bound[0])
#         ax2.set_ylim(min_bound[1], max_bound[1])

#         # --- Save the figure and close it ---
#         save_path = os.path.join(output_dir, f"{base_name}.png")
#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close(fig) # Close the figure to free memory
        
#         print(f"  - Saved plot for {base_name} to {save_path}")

# if __name__ == "__main__":
#     # --- IMPORTANT ---
#     # --- Set the path to your folder containing the .ply files ---
#     FOLDER_PATH = "/home/soham/garments/preet/here/saved_outputs"
#     # --- Set the maximum number of points to plot for performance ---
#     MAX_DISPLAY_POINTS = 15000
#     # -----------------------------------------------------------

#     save_matplotlib_2d_plots(FOLDER_PATH, max_points=MAX_DISPLAY_POINTS)