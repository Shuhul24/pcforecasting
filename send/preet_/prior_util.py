import os
import sys
sys.path.append("../utils")
sys.path.append("../model_arch")
from preet.transformer_encoder_without_mask import ContactTransformer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from utils import model_util
from diffusers.optimization import get_scheduler
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from dataset.ours_contact_data import ObjectContactData
from torch.optim.lr_scheduler import ReduceLROnPlateau


# def load_contact_dataset(cfg,split="train"):
#     train_dataset = None
#     train_dataset = ObjectContactData(split=split,
#                                       base_dir=cfg["base_dir"],
#                                       end_frame=cfg["end_frame"],
#                                       pred_horizon=cfg["pred_horizon"],
#                                       return_aux_info=False) 
#     return train_dataset

def load_contact_module(cfg, device, train_dataloader=None):
    encoder_input_dim = 0 #flattened bps points 512*3
    # input_dim = cfg["num_bps_points"] * 2 #left and right 512*2
    # prev_input_dim = cfg["num_bps_points"] * (cfg["num_features"] + 2) 
    # # part based, dim x 2
    # input_dim *= 2 #top and bottom
    # prev_input_dim *= 2
    encoder_input_dim *= 2 #top and bottom
    # global_state_dim = cfg["global_state_dim"]
    contact_model = ContactTransformer(input_feats=524288,
                                        latent_dim=512, #512
                                        ff_size=2048,
                                        num_layers=6,
                                        num_heads=8,
                                        dropout=0.1,
                                        activation='gelu',
                                        bps_input_dim=encoder_input_dim + 7+1200, #3072+7
                                        # bps_input_dim=encoder_input_dim + 7, #3072+7
                                        pred_horizon=64,
                                        diffusion_step_embed_dim=512,
                                        )


    contact_model.to(device)

    if train_dataloader is not None:
        optimizer = torch.optim.AdamW(
        params=contact_model.parameters(),
        lr=4e-4, weight_decay=0.0)
        # Coine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=2000,
            num_training_steps=len(train_dataloader) * cfg['TRAIN']["MAX_EPOCH"]
    )
        # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)


        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    else:
        optimizer = None
        lr_scheduler = None

    # if cfg["load_pretrained"]:
    #     model_file = os.path.join(cfg["save_root_dir"], cfg["exp_name"], cfg["pretrained_model_path"])
    #     if os.path.exists(model_file):
    #         print("--- loading pretrained model from --- " + model_file, flush=True)
    #         contact_model, _, optimizer, lr_scheduler, start_epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
    #             contact_model, optimizer, lr_scheduler, model_path=model_file)
    #         print("loaded model at epoch: ", start_epoch)
    #         cfg["start_epochs"] = start_epoch + 1
    #         print("*****************updating starte epoch to :  %d" %cfg["start_epochs"])
    #     else:
    #         print("pretrained contact model not found, retraining a new model")
    return contact_model, optimizer, lr_scheduler


def load_noise_scheduler(cfg):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=50,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=False, 
        prediction_type="sample",
    )
    return noise_scheduler



