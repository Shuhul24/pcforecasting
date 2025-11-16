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
    stats = torch.load("norm_stats.pt")
    mean = stats['mean'].to(device)
    std = stats['std'].to(device)
    print(f"Loaded normalization stats: Mean={mean.item():.4f}, Std={std.item():.4f}")

    for epoch_idx in range(42, start_epochs + num_epochs):
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
        minus1=torch.tensor(-1.0, device=device)
        # batch = next(iter(dataloader))
        total_chamfer = 0.0   # <-- Move outside batch loop
        total_count = 0       # <-- Move outside batch loop
        model.train()
        for batch_idx, batch in pbar:
        # for step in range(1000):
            # if batch_idx % 100 != 0:
            #     continue
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
            # predicted_range = torch.zeros((B, T_fut, 64, 2048)).to(device)
            
            # # Process each sample in batch
            # for b in range(B):
            #     current_data = past_data[b, -1]  
            #     current_pose = past_poses[b, -1]  
            #     sample_distances = []
                
            #     for t_fut in range(T_fut):
            #         target_pose = fut_poses[b, t_fut]
            #         gt_data = fut_data[b, t_fut]
            #         # gt_pc = pcf.range_data_to_point_cloud(gt_data)

            #         with torch.no_grad():
            #             pred_pc = pcf.forecast_point_cloud(
            #                 current_data, current_pose, target_pose, calibration
            #             )

            #             intensity_placeholder = torch.zeros((pred_pc.shape[0], 1), device=pred_pc.device)
            #             pred_pc_with_intensity = torch.cat([pred_pc, intensity_placeholder], dim=1)

            #             range_image0 = range_projection(
            #                 pred_pc_with_intensity.cpu().numpy(),
            #                 fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
            #                 fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
            #                 proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
            #                 proj_W=cfg["DATA_CONFIG"]["WIDTH"],
            #                 max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            #             )

            #             range_image_tensor = torch.from_numpy(range_image0[0]).float().to(device)
            #             # reprojected_point_clouds = Projection.get_valid_points_from_range_view(
            #             #     range_image_tensor, use_batch=False
            #             # )
            #             # breakpoint()
            #             predicted_range[b, t_fut] = range_image_tensor
        #                 chamfer_distances, _ = pcf.chamfer_distance(
        #                     pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
        #                 )

        #                 sample_distances.append(chamfer_distances[0])
                
        #         avg_chamfer_b = sum(sample_distances) / len(sample_distances)
        #         # print(f"[Batch {batch_idx}] Avg Chamfer Distance for sample {b}: {avg_chamfer_b:.6f}")

        #         # ✅ Accumulate globally
        #         total_chamfer += sum(sample_distances)
        #         total_count += len(sample_distances)

        # # ✅ Compute global average after all batches
        # global_avg_chamfer = total_chamfer / total_count
        # print(f"🌐 Global Average Chamfer Distance (entire dataset): {global_avg_chamfer:.6f}")

            # breakpoint()
                    # target_pcs.append(tgt_seq)
                # range_forecast=range_projection(pred_pc)
                # nbatch_norm['action'] = torch.cat([nbatch_norm['obs']['obj_feat_pred'], nbatch_norm['obs']['curr_global_states_pred']], dim=-1) # normalize obs
            
            nbatch_norm={}
            nbatch_norm['action'] = fut_data[:,:,0,:,:]
            # naction = nbatch_norm['action']
            naction_gt = nbatch_norm['action']
            # frame_position = torch.randint(0, 5, (4,))
            # target_mask = Projection.get_target_mask_from_range_view(naction_gt)
            naction=naction_gt-predicted_range
            # naction = (naction - mean) / std

            # breakpoint()
            # noise = torch.randn(naction.shape, device=device).float()

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (naction.shape[0],), device=device
            ).long()
            # forward process
            input=torch.cat([past_data[:,:,0],predicted_range[:]],dim=1)
            noise = torch.randn(naction.shape, device=device).float()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
            sample_pred = model(noisy_actions, timesteps, 
                                obj_feat=input)
            # sample_pred  = output[:,:past_data.shape[1]], output[:,past_data.shape[1]:]
            
            # final_inter=target_mask*(sample_pred+predicted_range[:])
            # breakpoint()
            # sample_pred = sample_pred * std + mean
            final_inter=sample_pred+predicted_range[:]
            if torch.isnan(final_inter).any():
                print("NaN detected in 'final' tensor calculation!")
                # Also check the inputs to be sure
                print("Is sample_pred NaN?", torch.isnan(sample_pred).any())
                print("Is predicted_range NaN?", torch.isnan(predicted_range).any())
            final = torch.where(final_inter < 0, minus1, final_inter)
            # final = torch.where(0>final_inter > -0.5, zero, final_inter)
            # === Compute L2 losses ===
            loss_pred = loss_L1(sample_pred, naction)
            # if torch.isnan(mask).any() or torch.isinf(mask).any():
            #     print("NaN or Inf found in MASK tensor!")
            # loss_mask = nn.BCEWithLogitsLoss()(mask, target_mask)
            # print('pred_loss:',loss_L1(sample_pred+predicted_range, naction_gt))
            loss2=loss_L1(naction_gt,sample_pred+predicted_range[:])
            # breakpoint()
            # λ_rot=0.1
            # loss = loss_pred 
            if  epoch_idx>40:
                

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
                # breakpoint()
            #     for b in range(B):
            #         sample_distances = []
            #         for t_fut in range(T_fut):
            #             gt_data = fut_data[b][t_fut].to(device)
            #             gt_pc = pcf.range_data_to_point_cloud(gt_data)
            #             final = torch.where(final <= 0, torch.tensor(-1.0, device=final.device), final)
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
                        # sample_distances.append(cd_value)

                    # sample_distances = torch.stack(sample_distances)  # (T_fut,)
                    # avg_distance = sample_distances.mean()           # scalar tensor
                    # batch_distances.append(avg_distance)

                # batch_distances = torch.stack(batch_distances)       # (B,)
                # loss = 0.5*(0.5*(loss_pred) + loss_mask) + 3*(loss_chamfer_distance)
                loss=loss_chamfer_distance+loss_pred
                # loss_chamfer+=loss_chamfer_distance.item()/B
                # loss = loss_pred + loss_mask+torch.tensor(batch_distances, device=loss_pred.device).mean()
                # loss_chamfer+=torch.tensor(batch_distances, device=loss_pred.device).mean().item()
                loss_chamfer+=loss_chamfer_distance.item()
            else:
                loss=loss_pred 
            # loss = nn.functional.mse_loss(sample_pred, naction_gt) 
            # print(loss)
             # Set model back to training mode
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_pred_acc+=loss_pred.item()
            # loss_mask_acc+=loss_mask.item()
            loss_gt+=loss2.item()
            loss_acc += loss.item()
            # i += 1
            # if i % 100 == 0:
                
            #     end = datetime.datetime.now()
            #     print("Epoch %d, batch %d, loss is %f, time for one batch is %s" %(epoch_idx, bn, loss.item(), str((end - start)/i)), flush=True)
            # break
            # print(nbatch_norm['aux']['category'][0])
        # model.eval() # Set model to evaluation mode
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         past_data = batch['past_data'].to(device)
        #         fut_data = batch['fut_data'].to(device)
        #         past_poses = batch['past_poses'].to(device)
        #         fut_poses = batch['fut_poses'].to(device)
        #         calibration = batch['calibration'][0].to(device)
        #         predicted_range=batch["predicted_range"].to(device)
        #         B = past_data.shape[0]
        #         T_fut = fut_data.shape[1]
        #         nbatch_norm={}
        #         nbatch_norm['action'] = fut_data[:,:,0,:,:]
        #         naction_gt = nbatch_norm['action']
        #         naction=naction_gt-predicted_range
        #         naction = (naction - mean) / std
        #         timesteps = torch.randint(
        #             0, noise_scheduler.config.num_train_timesteps,
        #             (naction.shape[0],), device=device
        #         ).long()
        #         input=torch.cat([past_data[:,:,0],predicted_range[:]],dim=1)
        #         noise = torch.randn(naction.shape, device=device).float()
        #         noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
        #         sample_pred = model(noisy_actions, timesteps, 
        #                             obj_feat=input)
        #         sample_pred = sample_pred * std + mean
        #         final_inter=sample_pred+predicted_range[:]
        #         final = torch.where(final_inter < 0, minus1, final_inter)
        #         loss_pred_val = loss_L1(sample_pred, naction)
        #         total_val_loss += loss_pred_val.item()
        # avg_val_loss = total_val_loss / len(val_loader)
            lr_scheduler.step()
        batch_distances=[]
        
        for b in range(B):
            sample_distances=[]
            timestep_distances = {f't+{i+1}': [] for i in range(5)}
            for t_fut in range(T_fut):
                # Get target pose for this future timestep
                target_pose = fut_poses[b, t_fut]  # (4, 4)
                
                # Predict point cloud for this future timestep using range data
                # pred_pc = forecast_point_cloud(
                #     current_data, current_pose, target_pose, calibration
                # )
                
                # Get ground truth point cloud for this future timestep
                gt_data = fut_data[b][t_fut].to(device)  
                gt_pc = pcf.range_data_to_point_cloud(gt_data)

                # predicted_pcs.append(pred_pc)
                # target_pcs.append(gt_pc)
            
            # Now compute chamfer distances for all timesteps at once
            # for t_fut in range(T_fut):
                # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
                # final[b, t_fut, :, :] = torch.where(final[b, t_fut, :, :] == 0, torch.tensor(-1.0, device=final.device), final[b, t_fut, :, :])

                pred_pc = Projection.get_valid_points_from_range_view(
                (sample_pred+predicted_range[:])[b, t_fut, :, :]
                # (sample_pred)[b, t_fut, :, :]
            )
                # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                
                # Compute chamfer distance using point cloud method
                chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
                    pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
                )
                
                cd_value = chamfer_distances[0].item()
                sample_distances.append(cd_value)
                timestep_distances[f't+{t_fut+1}'].append(cd_value)
        
            # Average over future timesteps for this sample
            avg_distance = np.mean(sample_distances)
            batch_distances.append(avg_distance)
        print(batch_distances)
        batch_distances=[]
        
        for b in range(B):
            sample_distances=[]
            timestep_distances = {f't+{i+1}': [] for i in range(5)}
            for t_fut in range(T_fut):
                # Get target pose for this future timestep
                target_pose = fut_poses[b, t_fut]  # (4, 4)
                
                # Predict point cloud for this future timestep using range data
                # pred_pc = forecast_point_cloud(
                #     current_data, current_pose, target_pose, calibration
                # )
                
                # Get ground truth point cloud for this future timestep
                gt_data = fut_data[b][t_fut]
                gt_pc = pcf.range_data_to_point_cloud(gt_data)

                # predicted_pcs.append(pred_pc)
                # target_pcs.append(gt_pc)
                pred_pc = Projection.get_valid_points_from_range_view(
                    predicted_range[b, t_fut, :, :]
                )
            # Now compute chamfer distances for all timesteps at once
            # for t_fut in range(T_fut):
                # pred_pc = pcf.range_data_to_point_cloud(final[b,t_fut].unsqueeze(0))  # (1, N, 3)
                pred_pc = Projection.get_valid_points_from_range_view(
                naction_gt[b, t_fut, :, :]
            )
                # gt_pc = target_pcs[:,t_fut].unsqueeze(0)       # (1, M, 3)
                
                # Compute chamfer distance using point cloud method
                chamfer_distances, chamfer_tensor = pcf.chamfer_distance(
                    pred_pc.unsqueeze(0), gt_pc.unsqueeze(0)
                )
                
                cd_value = chamfer_distances[0].item()
                sample_distances.append(cd_value)
                timestep_distances[f't+{t_fut+1}'].append(cd_value)
        
            # Average over future timesteps for this sample
            avg_distance = np.mean(sample_distances)
            batch_distances.append(avg_distance)
        print(batch_distances)
        
        # end of batch loop 
        loss_acc /= len(dataloader)
        loss_pred_acc /= len(dataloader)
        loss_mask_acc /= len(dataloader)
        loss_gt /= len(dataloader)
        if total_count>0:
            loss_chamfer=total_chamfer/total_count
        if use_wandb:
            wandb.log({"Train/Loss": loss_acc}, step=epoch_idx)
        else:
            writer.add_scalar('Loss/train', loss_acc, epoch_idx)
        epoch_end_time = datetime.datetime.now()
        print("Epoch %d, avg loss is %f, pred_loss is %f, mask_loss is %f, gt is %f, chamfer is %f" %(epoch_idx , loss_acc, loss_pred_acc, loss_mask_acc, loss_gt, loss_chamfer), flush=True)
        if epoch_idx < 5:
            print(" -------------- time for one epoch is %s ------" %(str(epoch_end_time - epoch_start_time)), flush=True)

        if epoch_idx % 2 == 1 :
            # model_util.save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, 
                                                                    # os.path.join(save_dir, "model_epoch_%d.pth" %(epoch_idx)))
            save_dir='/home/soham/garments/preet/here/PPMFNet/checkpoints'
            torch.save(model.state_dict(), os.path.join(save_dir, "model_epoch_%d.pth" % epoch_idx))


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
    data_module, test_loader, cfg = load_kitti_data(split='test')
    # data_module, val_loader, cfg = load_kitti_data(split='val')
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
    stat_dict=torch.load('/home/soham/garments/preet/here/PPMFNet/checkpoints/model_epoch_41.pth')
    # contact_model.load(state_dict=stat_dict)
    contact_model.load_state_dict(stat_dict)
    save_dir='/home/soham/garments/preet/here/PPMFNet/save'
    # if cfg["num_epochs"] != 0:
    # contact_model = FrameByFrameDiffusion(max_frames=5).to(device)
    
    train(cfg,save_dir, test_loader, 
            noise_scheduler, contact_model, optimizer, lr_scheduler,
            0, 200)


    end_overall = datetime.datetime.now()
    print("Whole program execution time is: ")
    print(end_overall-start_overall)



