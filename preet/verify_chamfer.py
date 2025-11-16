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
def train(cfg,save_dir, dataloader, noise_scheduler, model, optimizer, lr_scheduler, start_epochs, num_epochs, use_wandb=True,visualize=True):
    pcf=GTPointCloudForecaster(cfg)
    chamfer_distance_og = cham_dist(cfg)
    loss_L1=nn.L1Loss(reduction="mean")
    for epoch_idx in range(0, start_epochs + num_epochs):
        loss_acc = 0
        loss_pred_acc=0
        loss_mask_acc=0
        loss_gt=0
        loss_chamfer=0
        epoch_start_time = datetime.datetime.now()
        Projection = projection(cfg)
        i=0
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
        
        # --- Start of Corrections ---

        # ✅ 1. Imports added for nn.L1Loss and tqdm.
        # Assuming other necessary imports like pcf and range_projection are already present.

        # ✅ 2. Initialize global accumulators before the loop.
        # This ensures metrics are calculated over the entire dataset, not just the last batch.
        total_chamfer = 0.0
        total_l1_loss = 0.0
        total_count = 0

        # ✅ 3. Define the loss function once outside the loop.
        # loss_L1 = nn.L1Loss(reduction="sum") # Use "sum" to accumulate total error before averaging.

        # ✅ 4. The progress bar now wraps a placeholder for your data loader.
        # Replace `your_data_loader` with your actual DataLoader object.
        # pbar = tqdm(enumerate(your_data_loader), total=len(your_data_loader))

        # --- Original Code with Corrections ---

        for batch_idx, batch in pbar:
            past_data = batch['past_data'].to(device)
            fut_data = batch['fut_data'].to(device)
            past_poses = batch['past_poses'].to(device)
            fut_poses = batch['fut_poses'].to(device)
            calibration = batch['calibration'][0].to(device)
            
            B = past_data.shape[0]
            T_fut = fut_data.shape[1]
            # This line seems unused in the provided snippet, but kept as requested.
            predicted_range = torch.zeros((B, T_fut, 64, 2048)).to(device)

            # ✅ 5. Renamed 'loss_gt' for clarity; it now correctly accumulates for the batch.
            batch_l1_loss = 0.0
            
            # Process each sample in batch
            for b in range(B):
                current_data = past_data[b, -1]  
                current_pose = past_poses[b, -1]  
                
                # ✅ 6. These accumulators are correctly reset for each sample in the batch.
                sample_distances = []
                sample_l1_loss = 0.0
                
                for t_fut in range(T_fut):
                    target_pose = fut_poses[b, t_fut]
                    gt_data = fut_data[b, t_fut]
                    gt_pc = pcf.range_data_to_point_cloud(gt_data)

                    with torch.no_grad():
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
                        reprojected_point_clouds = Projection.get_valid_points_from_range_view(
range_image_tensor, use_batch=False
)                       
                        # range_image_tensor_gt = torch.from_numpy(gt_data[0]).float().to(device)
#                         reprojected_point_clouds_gt = Projection.get_valid_points_from_range_view(
# gt_data[0], use_batch=False
# )
                        # breakpoint()
                        # print((range_image_tensor-gt_data[0]).max())
                        # ✅ 7. This line calculates L1 loss for the current timestep.
                        loss_pred = loss_L1(range_image_tensor, gt_data[0])
                        # breakpoint()
                        # ✅ 8. Accumulate the L1 loss for the current sample.
                        sample_l1_loss += loss_pred.item()
                        # breakpoint()
                        chamfer_distances, _ = pcf.chamfer_distance(
                            reprojected_point_clouds.unsqueeze(0), gt_pc.unsqueeze(0)
                        )
                        sample_distances.append(chamfer_distances[0].item())

                # ✅ 9. Logic corrected to calculate the average for the current sample 'b'.
                # The original `loss_gt/=(T_fut)` was corrected to a proper assignment.
                avg_l1_b = sample_l1_loss / T_fut
                avg_chamfer_b = sum(sample_distances) / len(sample_distances)
                
                # The original print statements now show the correct per-sample averages.
                print(f"Sample {b} Avg Chamfer: {avg_chamfer_b:.6f}")
                print(f"Sample {b} Avg L1 Loss: {avg_l1_b:.6f}")

                # ✅ 10. Accumulate sums for global averaging. This is more numerically stable.
                total_l1_loss += sample_l1_loss
                total_chamfer += sum(sample_distances)
                total_count += len(sample_distances)

        # --- Final Metrics Calculation ---

        # ✅ 11. Final calculations and print statements are moved outside the loop.
        # They now compute and display the true global averages over the whole dataset.
        global_avg_l1 = total_l1_loss / total_count
        global_avg_chamfer = total_chamfer / total_count

        print(f"📊 Global Average L1 Loss (entire dataset): {global_avg_l1:.6f}")
        print(f"🌐 Global Average Chamfer Distance (entire dataset): {global_avg_chamfer:.6f}")
            # breakpoint()
                    # target_pcs.append(tgt_seq)
                # range_forecast=range_projection(pred_pc)
                # nbatch_norm['action'] = torch.cat([nbatch_norm['obs']['obj_feat_pred'], nbatch_norm['obs']['curr_global_states_pred']], dim=-1) # normalize obs
            
        #     nbatch_norm={}
        #     nbatch_norm['action'] = fut_data[:,:,0,:,:]
        #     # naction = nbatch_norm['action']
        #     naction_gt = nbatch_norm['action']
        #     # frame_position = torch.randint(0, 5, (4,))
        #     target_mask = Projection.get_target_mask_from_range_view(naction_gt)
        #     naction=torch.cat([naction_gt-predicted_range,target_mask],dim=1)
        #     # noise = torch.randn(naction.shape, device=device).float()

        #     # sample a diffusion iteration for each data point
        #     timesteps = torch.randint(
        #         0, noise_scheduler.config.num_train_timesteps,
        #         (naction.shape[0],), device=device
        #     ).long()
        #     # forward process
        #     input=torch.cat([past_data[:,:,0],predicted_range[:]],dim=1)
        #     noise = torch.randn(naction.shape, device=device).float()
        #     noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
        #     output = model(noisy_actions, timesteps, 
        #                         obj_feat=input)
        #     sample_pred, mask = output[:,:past_data.shape[1]], output[:,past_data.shape[1]:]
            
        #     # final_inter=target_mask*(sample_pred+predicted_range[:])
        #     final_inter=target_mask*(sample_pred+predicted_range[:])
        #     if torch.isnan(final_inter).any():
        #         print("NaN detected in 'final' tensor calculation!")
        #         # Also check the inputs to be sure
        #         print("Is sample_pred NaN?", torch.isnan(sample_pred).any())
        #         print("Is predicted_range NaN?", torch.isnan(predicted_range).any())
        #     final = torch.where(final_inter <= 0, zero, final_inter)
        #     # === Compute L2 losses ===
        #     loss_pred = loss_L1(final, target_mask*(naction_gt))
        #     if torch.isnan(mask).any() or torch.isinf(mask).any():
        #         print("NaN or Inf found in MASK tensor!")
        #     loss_mask = nn.BCEWithLogitsLoss()(mask, target_mask)

        #     loss2=loss_L1(target_mask*(naction_gt),target_mask*(predicted_range[:]))
            
        #     # λ_rot=0.1
        #     # loss = loss_pred 
        #     if  epoch_idx>2:
        #         batch_distances=[]
        
        #         batch_distances = []
        #         loss_chamfer_distance=0
        #         chd,cht=chamfer_distance_og(final,target_mask, fut_data, n_samples=-1)
        #         loss_chamfer_distance = sum([cd for cd in chd.values()]) / len(
        #         chd
        #     )
        #     #     for b in range(B):
        #     #         sample_distances = []
        #     #         for t_fut in range(T_fut):
        #     #             gt_data = fut_data[b][t_fut].to(device)
        #     #             gt_pc = pcf.range_data_to_point_cloud(gt_data)
        #     #             final = torch.where(final <= 0, torch.tensor(-1.0, device=final.device), final)
        #     #             pred_pc = Projection.get_valid_points_from_range_view(
        #     #                 final[b, t_fut, :, :]
        #     #             )
                        
        #     #             chamfer_distances, _ = pcf.chamfer_distance(
        #     #                 pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
        #     #             )
        #     #             # cd_value = chamfer_distances[0]
        #     #             loss_chamfer_distance+= sum([cd for cd in chamfer_distances.values()]) / len(
        #     #     chamfer_distances
        #     # )
        #                 # sample_distances.append(cd_value)

        #             # sample_distances = torch.stack(sample_distances)  # (T_fut,)
        #             # avg_distance = sample_distances.mean()           # scalar tensor
        #             # batch_distances.append(avg_distance)

        #         # batch_distances = torch.stack(batch_distances)       # (B,)
        #         # loss = 0.5*(0.5*(loss_pred) + loss_mask) + 3*(loss_chamfer_distance)
        #         loss=(loss_pred) + loss_mask + (loss_chamfer_distance)
        #         # loss_chamfer+=loss_chamfer_distance.item()/B
        #         # loss = loss_pred + loss_mask+torch.tensor(batch_distances, device=loss_pred.device).mean()
        #         # loss_chamfer+=torch.tensor(batch_distances, device=loss_pred.device).mean().item()
        #         loss_chamfer+=loss_chamfer_distance.item()
        #     else:
        #         loss=loss_pred + loss_mask
        #     # loss = nn.functional.mse_loss(sample_pred, naction_gt) 
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     loss_pred_acc+=loss_pred.item()
        #     loss_mask_acc+=loss_mask.item()
        #     loss_gt+=loss2.item()
        #     loss_acc += loss.item()
        #     # i += 1
        #     # if i % 100 == 0:
                
        #     #     end = datetime.datetime.now()
        #     #     print("Epoch %d, batch %d, loss is %f, time for one batch is %s" %(epoch_idx, bn, loss.item(), str((end - start)/i)), flush=True)
        #     # break
        #     # print(nbatch_norm['aux']['category'][0])
        #     lr_scheduler.step()
        # batch_distances=[]
        
        # for b in range(B):
        #     sample_distances=[]
        #     timestep_distances = {f't+{i+1}': [] for i in range(5)}
        #     for t_fut in range(T_fut):
        #         # Get target pose for this future timestep
        #         target_pose = fut_poses[b, t_fut]  # (4, 4)
                
        #         # Predict point cloud for this future timestep using range data
        #         # pred_pc = forecast_point_cloud(
        #         #     current_data, current_pose, target_pose, calibration
        #         # )
                
        #         # Get ground truth point cloud for this future timestep
        #         gt_data = fut_data[b][t_fut].to(device)  
        #         gt_pc = pcf.range_data_to_point_cloud(gt_data)

        #         # predicted_pcs.append(pred_pc)
        #         # target_pcs.append(gt_pc)
            
        #     # Now compute chamfer distances for all timesteps at once
        #     # for t_fut in range(T_fut):
        #         # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
        #         # final[b, t_fut, :, :] = torch.where(final[b, t_fut, :, :] == 0, torch.tensor(-1.0, device=final.device), final[b, t_fut, :, :])

        #         pred_pc = Projection.get_valid_points_from_range_view(
        #         (sample_pred+predicted_range[:])[b, t_fut, :, :]
        #         # (sample_pred)[b, t_fut, :, :]
        #     )
        #         # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                
        #         # Compute chamfer distance using point cloud method
        #         chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
        #             pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
        #         )
                
        #         cd_value = chamfer_distances[0].item()
        #         sample_distances.append(cd_value)
        #         timestep_distances[f't+{t_fut+1}'].append(cd_value)
        
        #     # Average over future timesteps for this sample
        #     avg_distance = np.mean(sample_distances)
        #     batch_distances.append(avg_distance)
        # print(batch_distances)
        # batch_distances=[]
        
        # for b in range(B):
        #     sample_distances=[]
        #     timestep_distances = {f't+{i+1}': [] for i in range(5)}
        #     for t_fut in range(T_fut):
        #         # Get target pose for this future timestep
        #         target_pose = fut_poses[b, t_fut]  # (4, 4)
                
        #         # Predict point cloud for this future timestep using range data
        #         # pred_pc = forecast_point_cloud(
        #         #     current_data, current_pose, target_pose, calibration
        #         # )
                
        #         # Get ground truth point cloud for this future timestep
        #         gt_data = fut_data[b][t_fut]
        #         gt_pc = pcf.range_data_to_point_cloud(gt_data)

        #         # predicted_pcs.append(pred_pc)
        #         # target_pcs.append(gt_pc)
        #         pred_pc = Projection.get_valid_points_from_range_view(
        #             predicted_range[b, t_fut, :, :]
        #         )
        #     # Now compute chamfer distances for all timesteps at once
        #     # for t_fut in range(T_fut):
        #         # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
        #         pred_pc = Projection.get_valid_points_from_range_view(
        #         naction_gt[b, t_fut, :, :]
        #     )
        #         # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                
        #         # Compute chamfer distance using point cloud method
        #         chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
        #             pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
        #         )
                
        #         cd_value = chamfer_distances[0].item()
        #         sample_distances.append(cd_value)
        #         timestep_distances[f't+{t_fut+1}'].append(cd_value)
        
        #     # Average over future timesteps for this sample
        #     avg_distance = np.mean(sample_distances)
        #     batch_distances.append(avg_distance)
        # print(batch_distances)
        
        # # end of batch loop 
        # loss_acc /= len(dataloader)
        # loss_pred_acc /= len(dataloader)
        # loss_mask_acc /= len(dataloader)
        # loss_gt /= len(dataloader)
        # loss_chamfer/=len(dataloader)
        # if use_wandb:
        #     wandb.log({"Train/Loss": loss_acc}, step=epoch_idx)
        # else:
        #     writer.add_scalar('Loss/train', loss_acc, epoch_idx)
        # epoch_end_time = datetime.datetime.now()
        # print("Epoch %d, avg loss is %f, pred_loss is %f, mask_loss is %f, gt is %f, chamfer is %f" %(epoch_idx , loss_acc, loss_pred_acc, loss_mask_acc, loss_gt, loss_chamfer), flush=True)
        # if epoch_idx < 5:
        #     print(" -------------- time for one epoch is %s ------" %(str(epoch_end_time - epoch_start_time)), flush=True)

        # if epoch_idx % 2 == 1 :
        #     # model_util.save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, 
        #                                                             # os.path.join(save_dir, "model_epoch_%d.pth" %(epoch_idx)))
        #     save_dir='/home/soham/garments/preet/here/PPMFNet/checkpoints'
        #     torch.save(model.state_dict(), os.path.join(save_dir, "model_epoch_%d.pth" % epoch_idx))


    if use_wandb:
        wandb.run.finish()

    model_path = os.path.join(save_dir, "model_final.pth")
    # model_util.save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, model_path)
    # print("saving trained model at path: " + model_path, flush=True)
    return model

if __name__ == '__main__':
    start_overall = datetime.datetime.now()
    # cfg = yaml_util.load_yaml(args.yaml_file)
    # exp_name = cfg["exp_name"]
    # save_dir = os.path.join(cfg["save_root_dir"], exp_name)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # if not cfg["use_wandb"]:

    writer = SummaryWriter(os.path.join('/home/soham/garments/preet/here/PPMFNet/logs/preet', "runs"))
    # yaml_util.save_yaml(os.path.join(save_dir, "config.yaml"), cfg)
    # if cfg["use_wandb"]:
    #     # Loggers
    data_module, test_loader, cfg = load_kitti_data()
    wandb.init(config=cfg, project='diffusion_model',
                    name='diff_model', dir='/home/soham/garments/preet/here/PPMFNet/logs/preet')
    
    start_time = datetime.datetime.now()    
    print("starting time is: ")
    print(start_time)
    # train_dataset = load_contact_dataset(cfg)
    # train_dataloader = DataLoader(test_loader, batch_size=64, shuffle=True, num_workers=1, drop_last=True)      
    print("************************ In total, dataloading takes time: " + str(datetime.datetime.now() - start_time) + "************************")
    # mesh_dict = train_dataset.mesh_dict #categories
    # stat_dict = data_util.load_stat_dict(cfg["stat_dict_path"], device) #mean,stds for normalizing global states,action,obj_feat from assets/contact_norm+stats.pkl
    # motion_dict=data_util.load_stat_dict(cfg["motion_dict_path"], device)
    # breakpoint()
    contact_model, optimizer, lr_scheduler = load_contact_module(cfg, device, test_loader)
    noise_scheduler = load_noise_scheduler(cfg)
    # stat_dict=torch.load('/home/soham/garments/preet/here/PPMFNet/checkpoints/model_epoch_9.pth')
    # contact_model.load(state_dict=stat_dict)
    # contact_model.load_state_dict(stat_dict)
    save_dir='/home/soham/garments/preet/here/PPMFNet/save'
    # if cfg["num_epochs"] != 0:
    # contact_model = FrameByFrameDiffusion(max_frames=5).to(device)
    
    train(cfg,save_dir, test_loader, 
            noise_scheduler, contact_model, optimizer, lr_scheduler,
            0, 200)


    end_overall = datetime.datetime.now()
    print("Whole program execution time is: ")
    print(end_overall-start_overall)



