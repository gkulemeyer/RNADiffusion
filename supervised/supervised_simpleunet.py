# GroupNorm https://arxiv.org/pdf/1803.08494
import torch as tr
import torch.nn as nn 

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.act1  = nn.SiLU()

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act2  = nn.SiLU()

        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h + self.shortcut(x)
 
class BackboneSimpleUNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=2, base_dim=64):
        """
        in_channels=16: 16 (Pair Rep) 
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__()
 

        # Encoder
        self.inc = ResBlock(in_channels, base_dim)
        self.down1 = nn.AvgPool2d(2)

        self.enc2 = ResBlock(base_dim, base_dim * 2)
        self.down2 = nn.AvgPool2d(2)

        # Bottleneck
        self.bot1 = ResBlock(base_dim * 2, base_dim * 4)
        self.bot2 = ResBlock(base_dim * 4, base_dim * 2)

        # Decoder

        ## Symetric
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec1 = ResBlock(base_dim * 4, base_dim * 2) # in 256, out 128

        ## not symmetric - (ResBlock)  128 channels in the upsampling (base_dim * 2)
        ## ResBlock takes 192 channels (128 up + 64 skip) and outputs 64
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec2 = ResBlock(base_dim * 3, base_dim)

        ## Salida: Logits (no softmax)
        self.outc = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x):
        ## x shape: [Batch, 16, L, L] 

        x1 = self.inc(x)         # [B, 64, L, L]
        x2 = self.down1(x1)      # [B, 64, L/2, L/2]
        x2 = self.enc2(x2)       # [B, 128, L/2, L/2]
        x3 = self.down2(x2)      # [B, 128, L/4, L/4]

        x3 = self.bot1(x3)       # [B, 256, L/8, L/8]
        x3 = self.bot2(x3)       # [B, 128, L/4, L/4]

        x_up1 = self.up1(x3)     # [B, 128, L/2, L/2]
        x_up1 = tr.cat([x_up1, x2], dim=1) # Concat: 128 + 128 = 256
        x_up1 = self.dec1(x_up1) # [B, 128, L/2, L/2]

        x_up2 = self.up2(x_up1)  # [B, 128, L, L] (no channel reduction)
        x_up2 = tr.cat([x_up2, x1], dim=1) # Concat: 128 + 64 = 192
        x_up2 = self.dec2(x_up2) # [B, 64, L, L] (ResBlock reduces channels)

        logits = self.outc(x_up2) # [B, 2, L, L]
        return logits
    
    
    