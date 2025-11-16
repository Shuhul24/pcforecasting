import torch
import torch.nn as nn
class VideoPatchEmbed(nn.Module):
    def __init__(self, in_channels=1, emb_dim=1024, patch_size=(2, 16, 16)):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, F, H, W)
        x = self.proj(x)  # (B, D, F', H', W')
        B, D, Fp, Hp, Wp = x.shape
        # x = x.flatten(2).transpose(1, 2)  # (B, F'*H'*W', D)
        return x, (Fp, Hp, Wp)
class SpatiotemporalPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_frames, max_height, max_width):
        super().__init__()
        self.time_embed = nn.Parameter(torch.randn(max_frames, emb_dim))
        self.spatial_embed = nn.Parameter(torch.randn(max_height * max_width, emb_dim))

    def forward(self, tokens, frame_shape):
        Fp, Hp, Wp = frame_shape
        B, C,N, D = tokens.shape  # N = Fp*Hp*Wp

        time_embed = self.time_embed[:Fp]  # (Fp, D)
        spatial_embed = self.spatial_embed[:Hp*Wp]  # (Hp*Wp, D)

        # Ensure embeddings match tokens shape
        # assert D == time_embed.size(1), "Feature dimension mismatch in time embedding"
        # assert D == spatial_embed.size(1), "Feature dimension mismatch in spatial embedding"

        time_embed = time_embed.unsqueeze(1).repeat(1, Hp*Wp, 1).unsqueeze(0).repeat(B, 1, 1,1)
        spatial_embed = spatial_embed.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1).repeat(1,Fp,1,1)

        return tokens + time_embed.permute(0,3,1,2) + spatial_embed.permute(0,3,1,2)

class VideoPatchDecoder(nn.Module):
    def __init__(self, emb_dim=512, out_channels=1, patch_size=(2, 16, 16)):
        super().__init__()
        self.proj = nn.ConvTranspose3d(
            in_channels=emb_dim,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, tokens, frame_shape):
        Fp, Hp, Wp = frame_shape
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).contiguous().view(B, D, Fp, Hp, Wp)
        x = self.proj(x)
        return x  # (B, out_channels, F, H, W)

class FrameEncoder(nn.Module):
    def __init__(self, patch_size=32, emb_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.proj = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, F, H, W)
        B, F, H, W = x.shape
        H_patches, W_patches = H // self.patch_size, W // self.patch_size
        # breakpoint()
        x = x.reshape(B*F, 1, H, W)                 # (B*F, 1, H, W)
        x = self.proj(x)                         # (B*F, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B*F, T, D), T = H_patches*W_patches
        x = x.reshape(B, F, -1, self.emb_dim)       # (B, F, T, D)

        return x, (H_patches, W_patches)
class FrameDecoder(nn.Module):
    def __init__(self, patch_size=32, emb_dim=512, out_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
        # --- Store out_channels ---
        self.out_channels = out_channels 
        
        self.proj = nn.ConvTranspose2d(
            in_channels=emb_dim,
            out_channels=self.out_channels, # Use it here
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, patch_shape):
        """
        x: (B, F, T, D)
        patch_shape: (H_patches, W_patches) from encoder
        """
        B, F, T, D = x.shape
        H_patches, W_patches = patch_shape
        assert H_patches * W_patches == T, f"Mismatch: {H_patches}×{W_patches} != {T}"

        x = x.reshape(B*F, T, D).transpose(1, 2)         # (B*F, D, T)
        x = x.reshape(B*F, D, H_patches, W_patches)      # (B*F, D, H/P, W/P)
        
        # This proj layer now outputs (B*F, self.out_channels, H, W)
        x = self.proj(x)
        
        # --- Use self.out_channels instead of 1 ---
        x = x.view(B, F, self.out_channels, x.shape[-2], x.shape[-1]) 
        
        return x



class MLP_Encoder(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, latent_dim, layer1_dim=1024, layer2_dim=1024):
    super().__init__()
    self.input_dim = input_dim,
    self.layers = nn.Sequential(
      nn.Linear(input_dim, layer1_dim),
      nn.ReLU(),
      nn.Linear(layer1_dim, layer2_dim),
      nn.ReLU(),
      nn.Linear(layer2_dim, latent_dim)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
