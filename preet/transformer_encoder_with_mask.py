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
        self.input_process = InputProcess(512, self.latent_dim) #(2048,512)
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
        self.output_process = OutputProcess(512, self.latent_dim) #(2048,512)
        # self.output_process_mask = OutputProcess(self.input_feats, self.latent_dim)
        # self.bps_encoder = MLP_Encoder(bps_input_dim, latent_dim=self.latent_dim)  #(3079,512)
        self.frame_encoder_obj = FrameEncoder(patch_size=32, emb_dim=512)
        self.frame_encoder_odo = FrameEncoder(patch_size=32, emb_dim=512)
        self.frame_decoder= FrameDecoder(patch_size=32, emb_dim=512, out_channels=2)
        # self.frame_decoder_mask= FrameDecoder(patch_size=32, emb_dim=512, out_channels=1)
        self.max_range=86.0
        self.min_range=-86.0
        self.past_frame_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.odometry_frame_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.sample_frame_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))

    def forward(self, sample, timestep, obj_feat=None):
        B = sample.shape[0]
        self.n_past_steps=5
        # --- 1. Encode Conditioning Frames ---
        # Split the conditioning input into past observations and the kinematic forecast
        past_frames = obj_feat[:, :self.n_past_steps]
        forecast_frames = obj_feat[:, self.n_past_steps:]

        # Use separate encoders for each data type
        # Output shape for each: (B, F, T, D_enc) where T is num_patches
        past_tokens, ps = self.frame_encoder_obj(past_frames)
        forecast_tokens, _ = self.frame_encoder_odo(forecast_frames)

        # ✅ Flatten the Frame (F) and Patch (T) dimensions into a single sequence
        # Shape becomes (B, F*T, D_enc)
        past_tokens = past_tokens.reshape(B, -1, self.frame_encoder_obj.emb_dim)
        forecast_tokens = forecast_tokens.reshape(B, -1, self.frame_encoder_odo.emb_dim)

        past_tokens = past_tokens + self.past_frame_token
        forecast_tokens = forecast_tokens + self.odometry_frame_token # Renamed for clarity
        
        conditioning_tokens = torch.cat([past_tokens, forecast_tokens], dim=1)

        # --- 2. Encode the Noisy Sample Frames ---
        sample_tokens, _ = self.frame_encoder_obj(sample)
        # breakpoint()
        # ✅ Flatten the Frame (F) and Patch (T) dimensions
        sample_tokens = sample_tokens.reshape(B, -1, self.frame_encoder_obj.emb_dim)
        
        # Project to latent dim and add token
        # sample_tokens = self.input_process(sample_tokens)
        sample_tokens = sample_tokens + self.sample_frame_token
        # breakpoint()
        # --- 3. Prepare Time Embedding ---
        timesteps = timestep.expand(B)
        time_emb = self.embed_timestep(timesteps) # Shape: (B, D_latent)
        # Add a sequence dimension to concatenate with other tokens
        time_emb = time_emb# Shape: (B, 1, D_latent)
        # breakpoint()
        # --- 4. Assemble Final Sequence for Transformer ---
        # Concatenate along the sequence dimension (dim=1)
        # Final sequence: [time, sample, past, forecast]
        # breakpoint()
        x=sample_tokens+past_tokens+forecast_tokens
        full_sequence = torch.cat([time_emb, x], dim=1)
        full_sequence = self.sequence_pos_encoder(full_sequence)
        
        # Permute from (Batch, SeqLen, Dim) to (SeqLen, Batch, Dim) for the transformer
        xseq = full_sequence.permute(1, 0, 2)

        # --- 5. Transformer Processing ---
        transformer_output = self.seqTransEncoder(xseq)
        
        # Permute back to (Batch, SeqLen, Dim)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # --- 6. Decode the Output ---
        # Extract only the tokens corresponding to the original sample
        # We skip the time_emb token (at index 0)
        num_sample_tokens = sample_tokens.shape[1]
        output_sample_tokens = transformer_output[:, 1 : 1 + num_sample_tokens]
        
        # Project back to the original patch embedding dimension
        # output_tokens = self.output_process(output_sample_tokens)

        # Reshape from a flat sequence back to (B, F, T, D_enc)
        n_future_frames = sample.shape[1]
        n_patches_per_frame = ps[0] * ps[1]
        output_tokens_grid = output_sample_tokens.reshape(B, n_future_frames, n_patches_per_frame, -1)
        # breakpoint()
        # Decode the patch tokens back into an image
        output_data = self.frame_decoder(output_tokens_grid, ps)
        # pred_img=self.min_range + nn.Sigmoid()(pred_img) * (
        #     self.max_range - self.min_range
        # )
        return output_data

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

