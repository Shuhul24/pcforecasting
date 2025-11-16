

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels
from diffusers import UNet2DConditionModel

min_range=-70.0
max_range=70.0
class TemporalEncoder(nn.Module):
    def __init__(
        self,
        frame_encoder: nn.Module = None,
        frame_feature_dim: int = 512,
        cross_attention_dim: int = 512,
        max_frames: int = 100,
        n_transformer_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_learned_pos: bool = True
    ):
        super().__init__()
        resnet = tvmodels.resnet18(pretrained=False)

        # Change first conv layer to accept 1 channel instead of 3
        resnet.conv1 = nn.Conv2d(
            in_channels=1,  # grayscale
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias is not None
        )
        # Frame encoder: default -> resnet18 trunk
        if frame_encoder is None:
            # resnet = tvmodels.resnet18(weights='DEFAULT')  # Updated syntax
            # Remove final fc layer and avgpool to get feature maps
            self.frame_encoder = nn.Sequential(*list(resnet.children())[:-2])  # Keep until conv layers
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Add explicit pooling
            self._frame_feature_dim = 512
        else:
            self.frame_encoder = frame_encoder
            self.global_pool = None
            self._frame_feature_dim = frame_feature_dim

        # Project frame features to cross_attention_dim
        self.frame_proj = nn.Linear(self._frame_feature_dim, cross_attention_dim)

        # Positional encoding
        self.use_learned_pos = use_learned_pos
        if use_learned_pos:
            self.pos_embedding = nn.Embedding(max_frames, cross_attention_dim)
        else:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_frames, cross_attention_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cross_attention_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        self.layer_norm = nn.LayerNorm(cross_attention_dim)
        self.cross_attention_dim = cross_attention_dim

    def forward(self, past_frames):
        """
        Args:
            past_frames: Tensor (B, F, C, H, W)
        Returns:
            temporal_context: Tensor (B, F, cross_attention_dim)
        """
        B, F = past_frames.shape[0], past_frames.shape[1]
        device = past_frames.device

        # Reshape to process all frames at once
        x = past_frames.reshape(B * F, *past_frames.shape[2:])

        # Extract features
        feat = self.frame_encoder(x)  # (B*F, 512, H', W')
        
        # Global pooling if needed
        if self.global_pool is not None:
            feat = self.global_pool(feat)  # (B*F, 512, 1, 1)
        
        # Flatten spatial dimensions
        feat = feat.view(feat.shape[0], feat.shape[1], -1).mean(dim=-1)  # (B*F, 512)
        
        # Project to cross_attention_dim
        feat = self.frame_proj(feat)  # (B*F, cross_attention_dim)
        
        # Reshape back to (B, F, cross_attention_dim)
        features = feat.view(B, F, self.cross_attention_dim)

        # Add positional encoding
        if self.use_learned_pos:
            pos_idx = torch.arange(F, device=device).unsqueeze(0).expand(B, -1)
            pos = self.pos_embedding(pos_idx)
        else:
            pos = self.pos_encoding[:F].unsqueeze(0).expand(B, -1, -1).to(device)
        
        features = features + pos

        # Apply transformer
        temporal_context = self.temporal_transformer(features)
        temporal_context = self.layer_norm(temporal_context)

        return temporal_context

    @staticmethod
    def _create_sinusoidal_encoding(max_len, d_model):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class FrameByFrameDiffusion(nn.Module):
    def __init__(self, max_frames=100, embed_dim=512):
        super().__init__()
        
        # Core 2D U-Net configuration
        self.unet2d = UNet2DConditionModel(
            sample_size=64,
            in_channels=1,
            out_channels=2,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            cross_attention_dim=512,
            attention_head_dim=8
        )
        
        # Temporal conditioning encoder
        self.temporal_encoder = TemporalEncoder(
            cross_attention_dim=512,  # Match UNet's cross_attention_dim
            max_frames=max_frames
        )
        
        # Frame positional embedding (for additional conditioning)
        self.frame_position_embedding = nn.Embedding(max_frames, embed_dim)
        
        # Project positional embedding to match expected input size
        # UNet2DConditionModel expects class_labels to be a scalar or match time_embed_dim
        self.pos_proj = nn.Linear(embed_dim, self.unet2d.config.cross_attention_dim)
        
    def forward(self, noisy_frame, timestep, condition_frames, frame_position):
        """
        Args:
            noisy_frame: (B, C, H, W) - Current frame with noise
            timestep: (B,) - Diffusion timestep
            condition_frames: (B, F, C, H, W) - Past frames for conditioning
            frame_position: (B,) - Position of current frame in sequence
        Returns:
            noise_pred: (B, C, H, W) - Predicted noise
        """
        # Get temporal context from past frames
        temporal_context = self.temporal_encoder(condition_frames)  # (B, F, 512)
        
        # Get positional embedding
        pos_embed = self.frame_position_embedding(frame_position)  # (B, embed_dim)
        pos_embed = self.pos_proj(pos_embed)  # (B, time_embed_dim)
        
        # Forward through UNet with conditioning
        noise_pred = self.unet2d(
            sample=noisy_frame,
            timestep=timestep,
            encoder_hidden_states=temporal_context,  # Cross-attention conditioning
            class_labels=pos_embed  # Additional positional conditioning
        ).sample  # Extract sample from output
        
        return noise_pred[:,:1], noise_pred[:,1:]  # Return two channels separately
