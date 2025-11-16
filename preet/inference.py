import torch
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import torch
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# from utils import data_util
# from utils import yaml_util, model_util,viz_util
# from inference import postprocess
import datetime
import argparse
from PIL import Image
from pytorch3d.transforms import so3_log_map, so3_exp_map
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import wandb
from src.utils.utils import load_files, range_projection
from preet.prior_util import load_contact_module, load_noise_scheduler
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")
torch.set_printoptions(precision=10)
torch.manual_seed(0)
np.random.seed(0)
from src.utils.projection import projection
# import scenepic as sp
from src.gt_pose_forecast_1 import load_kitti_data, GTPointCloudForecaster
from src.models.chamfer import cham_dist
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_file', type=str, default="config_files/train_contact_config.yaml")
args = parser.parse_args()
import open3d as o3d
print(args)
from tqdm import tqdm
from preet.unetdiff import FrameByFrameDiffusion
def train(cfg, dataloader, noise_scheduler, model, start_epochs, num_epochs, use_wandb=True,visualize=True):
    pcf=GTPointCloudForecaster(cfg)
    chamfer_distance_og = cham_dist(cfg)
    loss_L1=nn.L1Loss(reduction="mean")
    stats = torch.load("norm_stats.pt")
    mean = stats['mean'].to(device)
    std = stats['std'].to(device)
    loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
    print(f"Loaded normalization stats: Mean={mean.item():.4f}, Std={std.item():.4f}")
    model.eval()
    for epoch_idx in range(0, start_epochs + num_epochs):
        loss_acc = 0
        loss_pred_acc=0
        loss_mask_acc=0
        loss_gt=0
        loss_chamfer=0
        epoch_start_time = datetime.datetime.now()
        Projection = projection(cfg)
        i=0
        # breakpoint()
        # batch loop
        # start= datetime.datetime.now()
        # for bn, nbatch in enumerate(dataloader): # debug use the same nbatch, nbatch contains 
        timestep_distances = {f't+{i+1}': [] for i in range(5)}
        
        all_chamfer_distances = []
        # data_module, test_loader, cfg = load_kitti_datavga()
        # Set up progress bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                   desc="GT Pose Forecasting (Range->Range)", unit="batch")
        
        zero = torch.tensor(0.0, device=device)
        minus1=torch.tensor(-1.0, device=device)
        
        total_chamfer = 0.0   # <-- Move outside batch loop
        total_count = 0       # <-- Move outside batch loop
        # model.train()
        for batch_idx, batch in pbar:
            with torch.no_grad():

                past_data = batch['past_data'].to(device)
                fut_data = batch['fut_data'].to(device)
                past_poses = batch['past_poses'].to(device)
                fut_poses = batch['fut_poses'].to(device)
                calibration = batch['calibration'][0].to(device)
                # breakpoint()
                predicted_range=batch["predicted_range"].to(device)
                # breakpoint()
                B = past_data.shape[0]
                T_fut = fut_data.shape[1]
                            
                nbatch_norm={}
                nbatch_norm['action'] = fut_data[:,:,0,:,:]
                # naction = nbatch_norm['action']
                naction_gt = nbatch_norm['action']
                # frame_position = torch.randint(0, 5, (4,))
                target_mask = Projection.get_target_mask_from_range_view(naction_gt)
                naction=naction_gt-predicted_range
                # naction = (naction - mean) / std

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (naction.shape[0],), device=device
                ).long()
                # forward process
                # --- 1. SETUP ---
# Your model should be in evaluation mode
                model.eval()

                # Define inference parameters
                num_inference_steps = 50
                guidance_scale = 0.01 # A common starting point, can be tuned

                # Create the conditioning input your model expects
                conditional_input = torch.cat([past_data[:, :, 0], predicted_range], dim=1)

                # ✅ Create the unconditional input (a tensor of zeros)
                unconditional_input = torch.zeros_like(conditional_input)

                # ✅ Combine them into a single batch for an efficient cond pass
                combined_input = torch.cat([conditional_input, unconditional_input], dim=0)

                # ✅ Set the scheduler's timesteps for the denoising loop
                noise_scheduler.set_timesteps(num_inference_steps)
                noise = torch.randn(naction.shape, device=device).float()
                # ✅ Start with pure random noise
                # The shape should match your desired output shape (B, T_fut, H, W)
                sample_pred = torch.randn(naction_gt.shape, device=device).float()
                # sample_pred = model(noise, timesteps, 
                #                 obj_feat=conditional_input)
                # AFTER (consistent with training single-step)
                noise = torch.randn(naction.shape, device=device).float()
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                        (naction.shape[0],), device=device).long()
                # create the noisy sample exactly like training
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                output = model(noisy_actions, timesteps, obj_feat=conditional_input)
                sample_pred,mask= output[:,:,0,:,:],output[:,:,1,:,:]
                # sample_pred = sample_pred * std + mean
                final_inter=sample_pred+predicted_range[:]
                # if torch.isnan(final_inter).any():
                #     print("NaN detected in 'final' tensor calculation!")
                #     # Also check the inputs to be sure
                #     print("Is sample_pred NaN?", torch.isnan(sample_pred).any())
                #     print("Is predicted_range NaN?", torch.isnan(predicted_range).any())
                if batch_idx % 500 == 0 and torch.isnan(final_inter).any():
                    print("NaN detected!")
                
                final = torch.where(final_inter < 0, minus1, final_inter)
                # final = torch.where(0>final_inter > -0.5, zero, final_inter)
                # === Compute L2 losses ===
                masked_rv=Projection.get_masked_range_view(final,mask)
                sample_pred[naction_gt == -1.0] = -1.0
                naction[naction_gt == -1.0] = -1.0
                loss_delta = loss_L1(sample_pred,naction)
                loss_mask=loss_bce(mask,target_mask)
                loss_pred=loss_delta+loss_mask
                masked_rv[naction_gt == -1.0] = -1.0
                # naction_gt[naction_gt == -1.0] = -1.0
                final[naction_gt==-1.0]=-1.0
                loss_range=loss_L1(final,naction_gt)
                # final=masked_rv
                # if torch.isnan(mask).any() or torch.isinf(mask).any():
                #     print("NaN or Inf found in MASK tensor!")
                # loss_mask = nn.BCEWithLogitsLoss()(mask, target_mask)
                # print('pred_loss:',loss_L1(sample_pred+predicted_range, naction_gt))
                loss2=loss_L1(naction_gt,sample_pred+predicted_range[:])
                loss2=loss_L1(naction_gt,final)
                print('predicted',loss2)
                loss2=loss_L1(naction_gt,final_inter)
                
                loss2=loss_L1(naction_gt,predicted_range)
                print('odo:',loss2)
                # breakpoint()
                # λ_rot=0.1
                # loss = loss_pred 
                if  epoch_idx>-1:
                    

                    batch_loss_chamfer = torch.tensor(0.0, device=device)
                    num_frames = 0
                    for b in range(B):
                        for t in range(T_fut):
                            # Get the predicted point cloud from the final range image output
                            reprojected_point_clouds = Projection.get_valid_points_from_range_view(
                                final[b, t], use_batch=False
                            )

                            # Get the ground truth point cloud
                            gt_data = fut_data[b, t,0].to(device)
                            # ✅ 2. Corrected the slicing on gt_data.
                            # It should be handled consistently with how `final[b,t]` is processed.
                            gt_pc = Projection.get_valid_points_from_range_view(
                                gt_data, use_batch=False
                            )
                            
                            # breakpoint()
                            # Calculate the Chamfer distance for the current frame
                            chamfer_distance_val, _ = pcf.chamfer_distance(
                                reprojected_point_clouds.unsqueeze(0), gt_pc.unsqueeze(0)
                            )
                            print(chamfer_distance_val)
                            # ✅ 3. Append the scalar result of this frame's distance to the list.
                            # .item() is used to get the Python number from the tensor.
                            batch_loss_chamfer = batch_loss_chamfer + chamfer_distance_val[0]
                            num_frames += 1

                    # 3. Calculate the average loss for the batch
                    if num_frames > 0:
                        loss_chamfer_distance = batch_loss_chamfer / num_frames
                        total_chamfer += batch_loss_chamfer.item() 
                        total_count += num_frames

                    else:
                        loss_chamfer_distance = torch.tensor(0.0, device=device)
                    loss=4*loss_chamfer_distance+0.25*loss_pred
                    
                    loss_chamfer+=loss_chamfer_distance.item()
                else:
                    loss=loss_pred 
                # loss = nn.functional.mse_loss(sample_pred, naction_gt) 
                
                # Set model back to training mode
                
                output_dir='/home/soham/garments/preet/here/PPMFNet/preet/saved_output'
                    # Save range images
                for b in range(B):
                    for t in range(T_fut):
                        # Save ground truth range image
                        gt_range = naction_gt[b, t].cpu().numpy()
                        gt_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_gt_range.npy")
                        np.save(gt_filename, gt_range)
                        gt_min = gt_range.min()
                        gt_max = gt_range.max()

                        # Avoid division by zero if the image is constant
                        if gt_max > gt_min:
                            normalized_range = (gt_range - gt_min) / (gt_max - gt_min) * 255.0
                        else:
                            normalized_range = np.zeros_like(gt_range)

                        normalized_range = normalized_range.astype(np.uint8)

                        # Create output directory if it doesn't exist
                        # output_dir = 'output_directory_path'  # Set this accordingly
                        # os.makedirs(output_dir, exist_ok=True)

                        # Save as PNG
                        gt_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_gt_range.png")
                        Image.fromarray(normalized_range).save(gt_filename)
                        # Save predicted range image
                        pred_range = final[b, t].detach().cpu().numpy()
                        pred_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_pred_range.npy")
                        np.save(pred_filename, pred_range)
                        gt_min = pred_range.min()
                        gt_max = pred_range.max()

                        # Avoid division by zero if the image is constant
                        if gt_max > gt_min:
                            normalized_range = (pred_range - gt_min) / (gt_max - gt_min) * 255.0
                        else:
                            normalized_range = np.zeros_like(pred_range)

                        normalized_range = normalized_range.astype(np.uint8)

                        # Create output directory if it doesn't exist
                        # output_dir = 'output_directory_path'  # Set this accordingly
                        # os.makedirs(output_dir, exist_ok=True)

                        # Save as PNG
                        gt_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_pred_range.png")
                        Image.fromarray(normalized_range).save(gt_filename)
                        # Convert to point clouds and save
                        # You need the mask or threshold; here we use values > 0
                        gt_pc = Projection.get_valid_points_from_range_view(
                            naction_gt[b, t, :, :].cpu()
                        )

                            # Create open3d point cloud object
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(gt_pc)

                        # Save point cloud
                        pcd_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_gt_pc.ply")
                        o3d.io.write_point_cloud(pcd_filename, pcd)

                        # Similarly for ground truth point cloud
                        output_points = Projection.get_valid_points_from_range_view(
                            final[b, t, :, :].cpu()
                        )

                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(output_points.numpy())

                        gt_pcd_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_pred_pc.ply")
                        o3d.io.write_point_cloud(gt_pcd_filename, pcd_gt)
                        output_points = Projection.get_valid_points_from_range_view(
                            predicted_range[b, t, :, :].cpu()
                        )

                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(output_points)

                        gt_pcd_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_odo_pc.ply")
                        o3d.io.write_point_cloud(gt_pcd_filename, pcd_gt)
                        output_points = Projection.get_valid_points_from_range_view(
                            past_data[b, t,0, :, :].cpu()
                        )

                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(output_points)

                        gt_pcd_filename = os.path.join(output_dir, f"batch{batch_idx}_sample{b}_timestep{t}_input_pc.ply")
                        o3d.io.write_point_cloud(gt_pcd_filename, pcd_gt)
            breakpoint()
            loss_pred_acc+=loss_pred.item()
            # loss_mask_acc+=loss_mask.item()
            loss_gt+=loss2.item()
            loss_acc += loss.item()
        loss_acc /= len(dataloader)
        loss_pred_acc /= len(dataloader)
        loss_mask_acc /= len(dataloader)
        loss_gt /= len(dataloader)
        if total_count>0:
            loss_chamfer=total_chamfer/total_count
        print("Epoch %d, avg loss is %f, pred_loss is %f, mask_loss is %f, gt is %f, chamfer is %f" %(epoch_idx , loss_acc, loss_pred_acc, loss_mask_acc, loss_gt, loss_chamfer), flush=True)

    print(loss_chamfer/len(dataloader)
              )
                # final can now be used as the output prediction
            #     output_result = final
            #     # batch_distances=[]
        
            #     batch_distances = []
            #     loss_chamfer_distance=0
            #     for b in range(B):
            #         sample_distances = []
            #         for t_fut in range(T_fut):
            #             gt_data = fut_data[b][t_fut].to(device)
            #             gt_pc = pcf.range_data_to_point_cloud(gt_data)
            #             gt_pc = Projection.get_valid_points_from_range_view(
            #                 naction_gt[b, t_fut, :, :]
            #             )

            #             pred_pc = Projection.get_valid_points_from_range_view(
            #                 final[b, t_fut, :, :]
            #             )

            #             chamfer_distances, _ = pcf.chamfer_distance(
            #                 pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
            #             )
            #             # cd_value = chamfer_distances[0]
            #             loss_chamfer_distance+= sum([cd for cd in chamfer_distances.values()]) / len(
            #     chamfer_distances
            # )
            #     print(chamfer_distances, loss_chamfer_distance)
            #     loss_c+=(loss_chamfer_distance/5)
            #     batch_distances=[]
        
            #     for b in range(B):
            #         sample_distances=[]
            #         timestep_distances = {f't+{i+1}': [] for i in range(5)}
            #         for t_fut in range(T_fut):
            #             # Get target pose for this future timestep
            #             target_pose = fut_poses[b, t_fut]  # (4, 4)
                        
            #             # Predict point cloud for this future timestep using range data
            #             # pred_pc = forecast_point_cloud(
            #             #     current_data, current_pose, target_pose, calibration
            #             # )
                        
            #             # Get ground truth point cloud for this future timestep
            #             gt_data = fut_data[b][t_fut].to(device)  
            #             gt_pc = pcf.range_data_to_point_cloud(gt_data)

            #             # predicted_pcs.append(pred_pc)
            #             # target_pcs.append(gt_pc)
                    
            #         # Now compute chamfer distances for all timesteps at once
            #         # for t_fut in range(T_fut):
            #             # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
            #             # final[b, t_fut, :, :] = torch.where(final[b, t_fut, :, :] == 0, torch.tensor(-1.0, device=final.device), final[b, t_fut, :, :])

            #             pred_pc = Projection.get_valid_points_from_range_view(
            #             (sample_pred+predicted_range[:])[b, t_fut, :, :]
            #         )
            #             # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                        
            #             # Compute chamfer distance using point cloud method
            #             chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
            #                 pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
            #             )
                        
            #             cd_value = chamfer_distances[0].item()
            #             sample_distances.append(cd_value)
            #             timestep_distances[f't+{t_fut+1}'].append(cd_value)
                
            #         # Average over future timesteps for this sample_pred
            #         avg_distance = np.mean(sample_distances)
            #         batch_distances.append(avg_distance)
            #     print(batch_distances)
            #     batch_distances=[]
                
            #     for b in range(B):
            #         sample_distances=[]
            #         timestep_distances = {f't+{i+1}': [] for i in range(5)}
            #         for t_fut in range(T_fut):
            #             # Get target pose for this future timestep
            #             target_pose = fut_poses[b, t_fut]  # (4, 4)
                        
            #             # Predict point cloud for this future timestep using range data
            #             # pred_pc = forecast_point_cloud(
            #             #     current_data, current_pose, target_pose, calibration
            #             # )
                        
            #             # Get ground truth point cloud for this future timestep
            #             gt_data = fut_data[b][t_fut].to(device)  
            #             gt_pc = pcf.range_data_to_point_cloud(gt_data)

            #             # predicted_pcs.append(pred_pc)
            #             # target_pcs.append(gt_pc)
            #             pred_pc = Projection.get_valid_points_from_range_view(
            #                 predicted_range[b, t_fut, :, :]
            #             )
            #         # Now compute chamfer distances for all timesteps at once
            #         # for t_fut in range(T_fut):
            #             # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
            #             pred_pc = Projection.get_valid_points_from_range_view(
            #             naction_gt[b, t_fut, :, :]
            #         )
            #             # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                        
            #             # Compute chamfer distance using point cloud method
            #             chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
            #                 pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
            #             )
                        
            #             cd_value = chamfer_distances[0].item()
            #             sample_distances.append(cd_value)
            #             timestep_distances[f't+{t_fut+1}'].append(cd_value)
                
            #         # Average over future timesteps for this sample_pred
            #         avg_distance = np.mean(sample_distances)
            #         batch_distances.append(avg_distance)
            #     print(batch_distances)
        # print("Average Chamfer Distance over Epoch:", loss_c)
if __name__ == '__main__':
    start_overall = datetime.datetime.now()
    
    # Load configuration and data
    data_module, test_loader, cfg = load_kitti_data(split='test')
    
    # Initialize wandb for logging if needed (optional)
    wandb.init(config=cfg, project='diffusion_model',
               name='diff_model_inference', dir='/home/soham/garments/preet/here/PPMFNet/logs/preet')
    
    print("Starting inference at:", datetime.datetime.now())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and scheduler
    contact_model, _, _ = load_contact_module(cfg, device, test_loader)  # optimizer and scheduler unused in inference
    checkpoint = torch.load('/home/soham/garments/preet/here/PPMFNet/checkpoints/model_epoch_7.pth')
    contact_model = torch.compile(contact_model)
    contact_model.load_state_dict(checkpoint)
    # contact_model = torch.compile(contact_model)
    noise_scheduler = load_noise_scheduler(cfg)
    train(cfg, test_loader, 
            noise_scheduler, contact_model, 
            0, 200)