# GroupNorm https://arxiv.org/pdf/1803.08494
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math

from .simpleunet import SinusoidalPosEmb, ResBlock, SimpleUNet

class ResBlock_k5(ResBlock):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__(in_c, out_c, time_emb_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 5, padding=2) 
        self.conv2 = nn.Conv2d(out_c, out_c, 5, padding=2) 
        self.time_emb_dim = time_emb_dim 

class ResBlock_k7(ResBlock):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__(in_c, out_c, time_emb_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 7, padding=3) 
        self.conv2 = nn.Conv2d(out_c, out_c, 7, padding=3) 
        self.time_emb_dim = time_emb_dim  
        
class ResBlock_dilated1(ResBlock):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__(in_c, out_c, time_emb_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, dilation=1) 
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, dilation=1) 

class ResBlock_dilated2(ResBlock):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__(in_c, out_c, time_emb_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, dilation=2) 
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, dilation=2) 

class Unet_k5(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)
        self.inc = ResBlock_k5(in_channels, base_dim)
        self.enc2 = ResBlock_k5(base_dim, base_dim * 2)
        self.bot1 = ResBlock_k5(base_dim * 2, base_dim * 4)
        self.bot2 = ResBlock_k5(base_dim * 4, base_dim * 2)
        self.dec1 = ResBlock_k5(base_dim * 4, base_dim * 2)
        self.dec2 = ResBlock_k5(base_dim * 3, base_dim)

class Unet_k7(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)
        self.inc = ResBlock_k7(in_channels, base_dim)
        self.enc2 = ResBlock_k7(base_dim, base_dim * 2)
        self.bot1 = ResBlock_k7(base_dim * 2, base_dim * 4)
        self.bot2 = ResBlock_k7(base_dim * 4, base_dim * 2)
        self.dec1 = ResBlock_k7(base_dim * 4, base_dim * 2)
        self.dec2 = ResBlock_k7(base_dim * 3, base_dim)

class Unet_dilated(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)
        self.inc = ResBlock(in_channels, base_dim)
        self.enc2 = ResBlock_dilated1(base_dim, base_dim * 2)
        self.bot1 = ResBlock_dilated1(base_dim * 2, base_dim * 4)
        self.bot2 = ResBlock_dilated1(base_dim * 4, base_dim * 2)
        self.dec1 = ResBlock_dilated1(base_dim * 4, base_dim * 2)
        self.dec2 = ResBlock(base_dim * 3, base_dim)
        
class Unet_dilated2(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)
        self.inc = ResBlock(in_channels, base_dim)
        self.enc2 = ResBlock_dilated2(base_dim, base_dim * 2)
        self.bot1 = ResBlock_dilated2(base_dim * 2, base_dim * 4)
        self.bot2 = ResBlock_dilated2(base_dim * 4, base_dim * 2)
        self.dec1 = ResBlock_dilated2(base_dim * 4, base_dim * 2)
        self.dec2 = ResBlock(base_dim * 3, base_dim)
        
class Unet_bottleneck2_1(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)

        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.bot1 = ResBlock(base_dim * 2, base_dim * 1, time_dim)
        self.bot2 = ResBlock(base_dim * 1, base_dim * 2, time_dim)

class Unet_bottleneck2_2(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__(in_channels, out_channels, base_dim)

        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.bot1 = ResBlock(base_dim * 2, base_dim * 2, time_dim)
        self.bot2 = ResBlock(base_dim * 2, base_dim * 2, time_dim)




class SimpleUNet(nn.Module):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        super().__init__()
        
        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential( SinusoidalPosEmb(base_dim), nn.Linear(base_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        # Encoder
        self.inc = ResBlock(in_channels, base_dim, time_dim); self.down1 = nn.AvgPool2d(2)         
        self.enc2 = ResBlock(base_dim, base_dim * 2, time_dim); self.down2 = nn.AvgPool2d(2)

        # Bottleneck
        self.bot1 = ResBlock(base_dim * 2, base_dim * 4, time_dim)
        self.bot2 = ResBlock(base_dim * 4, base_dim * 2, time_dim)

        # Decoder
        
        ## Symetric
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec1 = ResBlock(base_dim * 4, base_dim * 2, time_dim) # in 256, out 128

        ## not symmetric - (ResBlock)  128 channels in the upsampling (base_dim * 2)
        ## ResBlock takes 192 channels (128 up + 64 skip) and outputs 64
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec2 = ResBlock(base_dim * 3, base_dim, time_dim) 
        
        ## Salida: Logits (no softmax)
        self.outc = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x, t):
        ## x shape: [Batch, 18, L, L]
        t_emb = self.time_mlp(t)
        
        x1 = self.inc(x, t_emb)      # [B, 64, L, L]
        x2 = self.down1(x1)          # [B, 64, L/2, L/2]
        x2 = self.enc2(x2, t_emb)    # [B, 128, L/2, L/2]
        x3 = self.down2(x2)          # [B, 128, L/4, L/4]
        
        x3 = self.bot1(x3, t_emb)   # [B, 256, L/8, L/8]
        x3 = self.bot2(x3, t_emb)    # [B, 128, L/4, L/4]
        
        x_up1 = self.up1(x3)         # [B, 128, L/2, L/2]
        x_up1 = tr.cat([x_up1, x2], dim=1) # Concat: 128 + 128 = 256
        x_up1 = self.dec1(x_up1, t_emb)       # [B, 128, L/2, L/2]

        x_up2 = self.up2(x_up1)      # [B, 128, L, L] (no channel reduction)
        x_up2 = tr.cat([x_up2, x1], dim=1) # Concat: 128 + 64 = 192
        x_up2 = self.dec2(x_up2, t_emb)       # [B, 64, L, L] (ResBlock reduces channels)
        
        logits = self.outc(x_up2)    # [B, 2, L, L]
        return logits