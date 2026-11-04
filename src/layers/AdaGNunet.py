# GroupNorm https://arxiv.org/pdf/1803.08494
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math
from .simpleunet import SinusoidalPosEmb, SimpleUNet

class ResBlockAdaGN1(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.act1  = nn.SiLU()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c * 2))
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act2  = nn.SiLU()

        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        h = self.conv2(h)

        scale, shift = self.mlp(time_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act2(h)

        return h + self.shortcut(x)
    
class ResBlockAdaGN2(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.act1  = nn.SiLU()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c * 2))
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act2  = nn.SiLU()

        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)

        scale, shift = self.mlp(time_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm1(h) * (1 + scale) + shift
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.shortcut(x)

class AdaGN1_UNet(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        """
        in_channels=18: 16 (Pair Rep) + 2 (Noisy One-Hot Contact Map)
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__(self, in_channels=18, out_channels=2, base_dim=64)
        
        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.inc  = ResBlockAdaGN1(in_channels, base_dim, time_dim) 
        self.enc2 = ResBlockAdaGN1(base_dim, base_dim * 2, time_dim) 
        self.bot1 = ResBlockAdaGN1(base_dim * 2, base_dim * 4, time_dim)
        self.bot2 = ResBlockAdaGN1(base_dim * 4, base_dim * 2, time_dim) 
        self.dec1 = ResBlockAdaGN1(base_dim * 4, base_dim * 2, time_dim) # in 256, out 128 
        self.dec2 = ResBlockAdaGN1(base_dim * 3, base_dim, time_dim) 

class AdaGN2_UNet(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        """
        in_channels=18: 16 (Pair Rep) + 2 (Noisy One-Hot Contact Map)
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__(self, in_channels=18, out_channels=2, base_dim=64)
        
        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.inc  = ResBlockAdaGN2(in_channels, base_dim, time_dim) 
        self.enc2 = ResBlockAdaGN2(base_dim, base_dim * 2, time_dim) 
        self.bot1 = ResBlockAdaGN2(base_dim * 2, base_dim * 4, time_dim)
        self.bot2 = ResBlockAdaGN2(base_dim * 4, base_dim * 2, time_dim) 
        self.dec1 = ResBlockAdaGN2(base_dim * 4, base_dim * 2, time_dim) # in 256, out 128 
        self.dec2 = ResBlockAdaGN2(base_dim * 3, base_dim, time_dim) 
