import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import datetime
import argparse
from pytorch3d.transforms import so3_log_map, so3_exp_map
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import wandb
from preet.utils import range_projection
from preet.utils import projection
# from preet.prior_util import load_contact_module, load_noise_scheduler
from src.utils.projection import projection as SrcProjection

from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")
torch.set_printoptions(precision=10)
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
    loss_bce = nn.BCEWithLogitsLoss(reduction="mean")    
    stats = torch.load("/home/shuhul/ppmf/norm_stats.pt")
    mean = stats['mean'].to(device)
    std = stats['std'].to(device)
    print(f"Loaded normalization stats: Mean={mean.item():.4f}, Std={std.item():.4f}")

    Projection = SrcProjection(cfg)
    zero = torch.tensor(0.0, device=device)
    minus1=torch.tensor(-1.0, device=device)
    loss_L1_no_reduction = nn.L1Loss(reduction="none")
    for epoch_idx in range(0, start_epochs + num_epochs):
        loss_acc = 0
        loss_pred_acc=0
        loss_mask_acc=0
        loss_gt=0
        loss_chamfer=0
        epoch_start_time = datetime.datetime.now()
        loss_range_acc=0
        i=0
        timestep_distances = {f't+{i+1}': [] for i in range(5)}
        
        all_chamfer_distances = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                   desc="GT Pose Forecasting (Range->Range)", unit="batch")
        
        
        # batch = next(iter(dataloader))
        total_chamfer = 0.0   # <-- Move outside batch loop
        total_count = 0       # <-- Move outside batch loop
        model.train()
        for batch_idx, batch in pbar:
        # for step in range(1000):
            # if batch_idx != 0:
            #     continue
            past_data = batch['past_data'].to(device)
            fut_data = batch['fut_data'].to(device)
            fut_poses = batch['fut_poses'].to(device)
            predicted_range=batch["predicted_range"].to(device)
            B = past_data.shape[0]
            T_fut = fut_data.shape[1]

            nbatch_norm={}
            nbatch_norm['action'] = fut_data[:,:,0,:,:]

            naction_gt = nbatch_norm['action']

            target_mask = Projection.get_target_mask_from_range_view(naction_gt)
            naction=naction_gt-predicted_range
            norm_naction = (naction - mean) / std
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (naction.shape[0],), device=device
            ).long()

            input=torch.cat([past_data[:,:,0],predicted_range[:]],dim=1)
            noise = torch.randn(naction.shape, device=device).float()
            noisy_actions = noise_scheduler.add_noise(norm_naction, noise, timesteps)
            output = model(noisy_actions, timesteps, 
                                obj_feat=input)

            sample_pred,mask= output[:,:,0,:,:],output[:,:,1,:,:]

            pixelwise_loss = loss_L1_no_reduction(sample_pred, norm_naction)

            masked_loss = pixelwise_loss * target_mask
            loss_delta = masked_loss.sum() / (target_mask.sum() + 1e-8)

            loss_mask = loss_bce(mask, target_mask)

            # This is your final, correct loss
            loss_pred = loss_delta + loss_mask
            with torch.no_grad():
                final_delta=sample_pred*std+mean
                final_inter=final_delta+predicted_range[:]
                # if torch.isnan(final_inter).any():
                #     print("NaN detected in 'final' tensor calculation!")
                #     # Also check the inputs to be sure
                #     print("Is sample_pred NaN?", torch.isnan(sample_pred).any())
                #     print("Is predicted_range NaN?", torch.isnan(predicted_range).any())
                if batch_idx % 500 == 0 and torch.isnan(final_inter).any():
                    print("NaN detected!")
                
                final = torch.where(final_inter < 0, zero, final_inter)
                # final = torch.where(0>final_inter > -0.5, zero, final_inter)
                # === Compute L2 losses ===
                masked_rv=Projection.get_masked_range_view(final,mask)
                
                final_delta[naction_gt == -1.0] = -1.0
                naction[naction_gt == -1.0] = -1.0
                loss_delta_unnorm = loss_L1(final_delta,naction)
                # loss_mask=loss_bce(mask,target_mask)
                # loss_pred=loss_delta+loss_mask
                # masked_rv[naction_gt == -1.0] = -1.0
                # naction_gt[naction_gt == -1.0] = -1.0
                final[naction_gt==-1.0]=-1.0
                loss_range=loss_L1(final,naction_gt)
                final=masked_rv
                # if torch.isnan(mask).any() or torch.isinf(mask).any():
                #     print("NaN or Inf found in MASK tensor!")
                # loss_mask = nn.BCEWithLogitsLoss()(mask, target_mask)
                # print('pred_loss:',loss_L1(sample_pred+predicted_range, naction_gt))
                loss2=loss_L1(naction_gt,sample_pred+predicted_range[:])
            # breakpoint()
            # λ_rot=0.1
            # loss = loss_pred 
            if  epoch_idx>10:
                

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
                        
                        chamfer_distance_val, _ = pcf.chamfer_distance(
                            reprojected_point_clouds.unsqueeze(0), gt_pc.unsqueeze(0)
                        )

                        batch_loss_chamfer = batch_loss_chamfer + chamfer_distance_val[0]
                        num_frames += 1

                # 3. Calculate the average loss for the batch
                if num_frames > 0:
                    loss_chamfer_distance = batch_loss_chamfer / num_frames
                    total_chamfer += batch_loss_chamfer.item() 
                    total_count += num_frames

                else:
                    loss_chamfer_distance = torch.tensor(0.0, device=device)
 
                loss=loss_chamfer_distance+loss_pred

                loss_chamfer+=loss_chamfer_distance.item()
            else:
                loss=loss_pred 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_range_acc+=loss_range.item()
            loss_pred_acc+=loss_delta_unnorm.item()
            loss_mask_acc+=loss_mask.item()
            loss_gt+=loss2.item()
            loss_acc += loss.item()
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
        loss_range_acc/=len(dataloader)
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
        print("Epoch %d, avg loss is %f, pred_loss is %f, mask_loss is %f, gt is %f, chamfer is %f, range is %f" %(epoch_idx , loss_acc, loss_pred_acc, loss_mask_acc, loss_gt, loss_chamfer,loss_range_acc), flush=True)
        if epoch_idx < 5:
            print(" -------------- time for one epoch is %s ------" %(str(epoch_end_time - epoch_start_time)), flush=True)

        if epoch_idx % 2 == 1 :
            checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_epoch_%d.pth" % epoch_idx))


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
    
    writer = SummaryWriter(os.path.join(current_dir, "runs"))
    # yaml_util.save_yaml(os.path.join(save_dir, "config.yaml"), cfg)
    # if cfg["use_wandb"]:
    #     # Loggers
    data_module, test_loader, cfg = load_kitti_data(split='test', project_root=parent_dir)
    # data_module, val_loader, cfg = load_kitti_data(split='val', project_root=parent_dir)
    wandb.init(config=cfg, project='diffusion_model', name='diff_model', dir=current_dir)
    
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
    # stat_dict=torch.load('/home/soham/garments/preet/here/PPMFNet/checkpoints/model_epoch_29.pth')
    # contact_model.load(state_dict=stat_dict)
    contact_model = torch.compile(contact_model)
    # contact_model.load_state_dict(stat_dict) # If you use this, ensure the path is correct
    save_dir=os.path.join(parent_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)
    # if cfg["num_epochs"] != 0:
    # contact_model = FrameByFrameDiffusion(max_frames=5).to(device)
    
    train(cfg,save_dir, test_loader, 
            noise_scheduler, contact_model, optimizer, lr_scheduler,
            0, 200)


    end_overall = datetime.datetime.now()
    print("Whole program execution time is: ")
    print(end_overall-start_overall)
