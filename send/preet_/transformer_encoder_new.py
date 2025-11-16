import numpy as np
import torch
import torch.nn as nn
from preet.encoder import MLP_Encoder
from preet.encoder import FrameEncoder, FrameDecoder
class ContactTransformer(nn.Module):
    def __init__(self, 
                 input_feats, 
                 latent_dim=256, 
                 ff_size=1024, 
                 num_layers=8, 
                 num_heads=4, 
                 dropout=0.1,
                 activation="gelu",  
                 bps_input_dim=3072, 
                 pred_horizon=64, 
                 diffusion_step_embed_dim=256,
                 max_len=65):
        super().__init__()
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.pred_horizon = pred_horizon 
        self.input_process = InputProcess(self.input_feats, self.latent_dim) #(2048,512)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=6000)
        self.diffusion_step_embed_dim = diffusion_step_embed_dim #512
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.output_process = OutputProcess(self.input_feats, self.latent_dim) #(2048,512)
        self.output_process_mask = OutputProcess(self.input_feats, self.latent_dim)
        self.bps_encoder = MLP_Encoder(bps_input_dim, latent_dim=self.latent_dim)  #(3079,512)
        self.frame_encoder_obj = FrameEncoder(patch_size=16, emb_dim=1024)
        self.frame_encoder_sample = FrameEncoder(patch_size=16, emb_dim=1024)
        self.frame_decoder= FrameDecoder(patch_size=16, emb_dim=1024, out_channels=1)
        self.frame_decoder_mask= FrameDecoder(patch_size=16, emb_dim=1024, out_channels=1)
        self.max_range=85.0
        self.min_range=-85.0
    def forward(self, sample, timestep, obj_feat=None):
        B, F, T, D = obj_feat.shape  # placeholder for shape consistency

        # Encode object feature frames
        tokens, ps = self.frame_encoder_obj(obj_feat)  # (B, F, T, D)
        B, F, T, D = tokens.shape
        tokens = tokens.reshape(B, F,T* D)
        tokens_input = self.input_process(tokens)
        tokens = tokens_input.permute(1, 0, 2)

        # Time embedding
        timesteps = timestep.expand(sample.shape[0])
        emb = self.embed_timestep(timesteps).permute(1, 0, 2)  # (1, B, D)

        # Encode sample frames
        tokens_sample = self.frame_encoder_sample(sample)[0]
        tokens_sample = tokens_sample.reshape(B, F,T*D)
        tokens_sample_input = self.input_process(tokens_sample)
        sample_tokens = tokens_sample_input.permute(1, 0, 2)

        # Combine tokens
        x = sample_tokens + tokens
        xseq = torch.cat((emb, x), dim=0)
        xseq = self.sequence_pos_encoder(xseq)

        # Transformer processing
        output1 = self.seqTransEncoder(xseq)[1:1 + 2*F]
        # print(output1.shape)
        output = self.output_process(output1[:int(F/2)])
        output = output.permute(1, 0, 2)
        # print(output.shape,tokens_sample.shape)
        output = output.view((B, int(F/2), T, D))
        output_mask = self.output_process_mask(output1[int(F/2):])
        output_mask = output_mask.permute(1, 0, 2)
        output_mask = output_mask.view((B, int(F/2), T, D))
        # Decode range and mask separately
        output_final = self.frame_decoder(output, ps).squeeze(2)        # (B, 1, H, W)
        output_final_mask = self.frame_decoder_mask(output_mask, ps).squeeze(2) 
        # pred_mask_logits = self.frame_decoder_mask(output, ps).squeeze(2)  # (B, 1, H, W)
        pred_img=output_final
        pred_mask_logits=output_final_mask
        # Apply activations
        # pred_range = self.min_range + torch.sigmoid(pred_img) * (self.max_range - self.min_range)
        # pred_mask = torch.sigmoid(pred_mask_logits)  # output mask probabilities in [0,1]

        return torch.cat([pred_img,pred_mask_logits],dim=1)

    # def forward(self, 
    #             sample, 
    #             timestep, 
    #             obj_feat=None,
    #             global_cond=None):
    #     """
    #     x: 
    #         [batch_size,  max_frames, input_feats], denoted x_t in the paper # [1, 32, 1200]
    #          MDM assumes x is [batch_size, njoints, nfeats, max_frames]; hence we should transpose
    #     timesteps: [batch_size] (int)
    #     """
        

    #     # frames = torch.randn(8, 5, 128, 128)  # (B, F, H, W)
    #     tokens,ps = self.frame_encoder(obj_feat)       # (B, 5, T, D)
    #     # tokens=obj_feat
    #     # Flatten frames into long sequence
    #     B, F, T, D = tokens.shape
    #     tokens = tokens.reshape(B, F*T, D)       # (B, 5*T, D)
    #     tokens_input= self.input_process(tokens)
    #     tokens = tokens_input.permute(1, 0, 2)      # (5*T, B, D) for transformer

    #     timesteps = timestep.expand(sample.shape[0]) #just repeat values incase timesteps are lesser than needed.
    #     emb = self.embed_timestep(timesteps)  # diff timestep encoding [B, 1, D],torch.Size([100, 1, 512])
    #     # global_states = global_cond["curr_global_states"]
    #     # obj_feat = torch.cat([obj_feat, global_states], axis=-1) #torch.Size([100, 64, 3079])
    #     # obj_feat = self.bps_encoder(obj_feat) # B, T, D ,torch.Size([100, 64, 512])
    #     # obj_feat = obj_feat.permute(1, 0, 2) # T, B, D
    #     tokens_sample = self.frame_encoder(sample)[0] 
    #     tokens_sample = tokens_sample.reshape(B, F*T,D)
    #     tokens_sample_input=self.input_process(tokens_sample)
        
    #     sample_tokens = tokens_sample_input.permute(1, 0, 2) # B, T, D -> T, B, D #torch.Size([64, 100, 2048])
    #     emb = emb.permute(1, 0, 2) #(1,100,512)

    #     # x = self.input_process(sample_tokens) # linear layer # [T, B, D]
    #     # x_input=self.input_process(tokens)
    #     # x = x + x_input  #contact points+ bps+gs+scale
    #     x=sample_tokens+tokens
    #     xseq = torch.cat((emb, x), dim=0)
    #     # xseq = torch.cat((emb, x), axis=0)  # [T+1,B,D], torch.Size([65, 100, 512]) 
    #     xseq = self.sequence_pos_encoder(xseq) #torch.Size([65, 100, 512]) , position of each 65(Time) is encoded
    #     output = self.seqTransEncoder(xseq)[1:1+F*T]  #torch.Size([64, 100, 512]) # skip PE of time embedding, src_key_padding_mask=~maskseq)  # [bs, seqlen, d]
    #     output = self.output_process(output)  #torch.Size([100, 64, 2048]) # [bs, njoints, nfeats, nframes]
    #     # assert output.shape[0] == F*T
    #     output = output.permute(1, 0, 2)               # (B, T, D)
        
    #     H, W = sample.shape[-2:]                       # recover spatial dims
    #     output = output.view((B, F, T,-1) )
    #     pred_img = self.frame_decoder(output,ps)   # (B, 1, 1, H, W)
    #     pred_img = pred_img.squeeze(2)               # (B, 1, H, W)

    #     return self.min_range+nn.Sigmoid()(pred_img)*(self.max_range-self.min_range)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [T, B, D]
        '''
        x = x + self.pe[:x.shape[0], :] # addition
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim #512
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim), #(512,512)
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim), #(512,512)
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.kpEmbedding = nn.Linear(self.input_feats, self.latent_dim)


    def forward(self, x):
        '''x: [T, B, D]
        '''
        x = self.kpEmbedding(x)
        return x

class OutputProcess(nn.Module):
    def __init__(self,input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.kpFinal = nn.Linear(self.latent_dim, self.input_feats)


    def forward(self, output):
        output = self.kpFinal(output) # T, B, D
        output = output.permute(1,0,2)  # [B, T, D]
        return output

