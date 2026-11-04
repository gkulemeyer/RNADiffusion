# GroupNorm https://arxiv.org/pdf/1803.08494
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math
from .simpleunet import SinusoidalPosEmb, ResBlock, SimpleUNet

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tr.exp(tr.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tr.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 

class LayerNorm2d(nn.Module):
    """LayerNorm sobre la dimensión de canales para tensores (B, C, H, W), como en ConvNeXt."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(tr.ones(num_channels))
        self.bias = nn.Parameter(tr.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / tr.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, mult=4, layer_scale_init=1e-6):
        super().__init__()
        # 1) Convolución depthwise 7x7 (mezcla espacial, canal a canal)
        self.dwconv = nn.Conv2d(in_c, in_c, kernel_size=7, padding=3, groups=in_c)
        self.norm = LayerNorm2d(in_c)

        # Inyección del tiempo 
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_c)
        )

        # 2) MLP invertido con convoluciones 1x1 (mezcla entre canales)
        self.pwconv1 = nn.Conv2d(in_c, out_c * mult, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(out_c * mult, out_c, kernel_size=1)
 
        self.gamma = (
            nn.Parameter(layer_scale_init * tr.ones(out_c))
            if layer_scale_init > 0 else None
        )

        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb):
        h = self.dwconv(x)
        h = self.norm(h)

        # Time injection
        time_condition = self.mlp(time_emb)[:, :, None, None]
        h = h + time_condition

        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)

        if self.gamma is not None:
            h = h * self.gamma[:, None, None]

        return h + self.shortcut(x)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "num_heads debe dividir a dim"
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
 
    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)              # [B, heads, N, d]
        out = F.scaled_dot_product_attention(q, k, v)     # flash / mem-efficient en GPU
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)
  
class AttentionBlock(nn.Module):
    """Self-attention estilo DDPM/ADM: GroupNorm -> MHSA -> residual."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = MultiHeadSelfAttention(channels, num_heads)
        # Proyección de salida a cero: el bloque empieza como identidad
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)
 
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)       # [B, H*W, C]
        h = self.attn(h)
        return x + h.transpose(1, 2).reshape(B, C, H, W)  
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
  
def sincos_pos_emb_2d(h, w, dim, device):
    """Embedding posicional 2D senoidal fijo, calculado al vuelo (vale para cualquier L)."""
    assert dim % 4 == 0
    quarter = dim // 4
    omega = 1.0 / (10000 ** (tr.arange(quarter, device=device).float() / quarter))
    ys = tr.arange(h, device=device).float()[:, None] * omega[None]
    xs = tr.arange(w, device=device).float()[:, None] * omega[None]
    emb_y = tr.cat([ys.sin(), ys.cos()], dim=1)                        # [h, dim/2]
    emb_x = tr.cat([xs.sin(), xs.cos()], dim=1)                        # [w, dim/2]
    emb = tr.cat([
        emb_y[:, None, :].expand(h, w, -1),
        emb_x[None, :, :].expand(h, w, -1),
    ], dim=-1)                                                         # [h, w, dim]
    return emb.reshape(h * w, dim)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, cond_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
 
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
 
        # adaLN-Zero: shift, scale y gate para atención y MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
 
    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
 
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
 
 
class DiTStack(nn.Module):
    """Aplana el mapa una vez, aplica `depth` bloques DiT y lo reconstruye."""
    def __init__(self, channels, cond_dim, depth=4, num_heads=8, mlp_ratio=4.0, use_pos_emb=True):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.blocks = nn.ModuleList([
            DiTBlock(channels, cond_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
 
    def forward(self, x, time_emb):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)              # [B, H*W, C]
        if self.use_pos_emb:
            tokens = tokens + sincos_pos_emb_2d(H, W, C, x.device).to(tokens.dtype)[None]
        for blk in self.blocks:
            tokens = blk(tokens, time_emb)
        return tokens.transpose(1, 2).reshape(B, C, H, W)
 

class convnextblock_unet(nn.Module):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64):
        """
        in_channels=18: 16 (Pair Rep) + 2 (Noisy One-Hot Contact Map)
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__()
        
        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.inc = ConvNeXtBlock(in_channels, base_dim, time_dim)
        self.down1 = nn.AvgPool2d(2)         

        self.enc2 = ConvNeXtBlock(base_dim, base_dim * 2, time_dim)
        self.down2 = nn.AvgPool2d(2)

        # Bottleneck
        self.bot1 = ConvNeXtBlock(base_dim * 2, base_dim * 4, time_dim)
        self.bot2 = ConvNeXtBlock(base_dim * 4, base_dim * 2, time_dim)

        # Decoder
        
        ## Symetric
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec1 = ConvNeXtBlock(base_dim * 4, base_dim * 2, time_dim) # in 256, out 128

        ## not symmetric - (ResBlock)  128 channels in the upsampling (base_dim * 2)
        ## ResBlock takes 192 channels (128 up + 64 skip) and outputs 64
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec2 = ConvNeXtBlock(base_dim * 3, base_dim, time_dim) 
        
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
        x_up2 = self.dec2(x_up2, t_emb)       # [B, 64, L, L] (ConvNeXtBlock reduces channels)
        
        logits = self.outc(x_up2)    # [B, 2, L, L]
        return logits

class SelfAttn_UNet(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64, num_heads=8):
        """
        in_channels=18: 16 (Pair Rep) + 2 (Noisy One-Hot Contact Map)
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_dim=base_dim)
 
        # Bottleneck: bot1 -> attn -> bot2
        self.attn = AttentionBlock(base_dim * 4, num_heads)
 
    def forward(self, x, t):
        ## x shape: [Batch, 18, L, L]
        t_emb = self.time_mlp(t)
 
        x1 = self.inc(x, t_emb)      # [B, 64, L, L]
        x2 = self.down1(x1)          # [B, 64, L/2, L/2]
        x2 = self.enc2(x2, t_emb)    # [B, 128, L/2, L/2]
        x3 = self.down2(x2)          # [B, 128, L/4, L/4]
 
        x3 = self.bot1(x3, t_emb)    # [B, 256, L/4, L/4]
        x3 = self.attn(x3)           # [B, 256, L/4, L/4]
        x3 = self.bot2(x3, t_emb)    # [B, 128, L/4, L/4]
 
        x_up1 = self.up1(x3)                    # [B, 128, L/2, L/2]
        x_up1 = tr.cat([x_up1, x2], dim=1)      # Concat: 128 + 128 = 256
        x_up1 = self.dec1(x_up1, t_emb)         # [B, 128, L/2, L/2]
 
        x_up2 = self.up2(x_up1)                 # [B, 128, L, L]
        x_up2 = tr.cat([x_up2, x1], dim=1)      # Concat: 128 + 64 = 192
        x_up2 = self.dec2(x_up2, t_emb)         # [B, 64, L, L]
 
        logits = self.outc(x_up2)    # [B, 2, L, L]
        return logits


class DiT_UNet(SimpleUNet):
    def __init__(self, in_channels=18, out_channels=2, base_dim=64,
                 depth=4, num_heads=8, mlp_ratio=4.0, use_pos_emb=True):
        """
        in_channels=18: 16 (Pair Rep) + 2 (Noisy One-Hot Contact Map)
        out_channels=2: Logits: 0 (no-contact), 1 (contact)
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_dim=base_dim)
 
        # Bottleneck: bot1 -> DiT x depth -> bot2
        time_dim = base_dim * 4
        self.dit = DiTStack(base_dim * 4, time_dim, depth, num_heads, mlp_ratio, use_pos_emb)
 
    def forward(self, x, t):
        ## x shape: [Batch, 18, L, L]
        t_emb = self.time_mlp(t)
 
        x1 = self.inc(x, t_emb)      # [B, 64, L, L]
        x2 = self.down1(x1)          # [B, 64, L/2, L/2]
        x2 = self.enc2(x2, t_emb)    # [B, 128, L/2, L/2]
        x3 = self.down2(x2)          # [B, 128, L/4, L/4]
 
        x3 = self.bot1(x3, t_emb)    # [B, 256, L/4, L/4]
        x3 = self.dit(x3, t_emb)     # [B, 256, L/4, L/4]
        x3 = self.bot2(x3, t_emb)    # [B, 128, L/4, L/4]
 
        x_up1 = self.up1(x3)                    # [B, 128, L/2, L/2]
        x_up1 = tr.cat([x_up1, x2], dim=1)      # Concat: 128 + 128 = 256
        x_up1 = self.dec1(x_up1, t_emb)         # [B, 128, L/2, L/2]
 
        x_up2 = self.up2(x_up1)                 # [B, 128, L, L]
        x_up2 = tr.cat([x_up2, x1], dim=1)      # Concat: 128 + 64 = 192
        x_up2 = self.dec2(x_up2, t_emb)         # [B, 64, L, L]
 
        logits = self.outc(x_up2)    # [B, 2, L, L]
        return logits