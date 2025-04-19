import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Downsample(nn.Module):
    """Downsample with Conv + PixelUnshuffle"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim//2, kernel_size=3, padding=1)
        self.unshuffle = nn.PixelUnshuffle(2)
        
    def forward(self, x):
        return self.unshuffle(self.conv(x))

class LayerNorm(nn.Module):
    """LayerNorm that works with 4D tensors"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, h)

class ConvFFN(nn.Module):
    """Conv-based Feed Forward Network"""
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden_dim = dim * expansion
        
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Efficient Self-Attention"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(1, heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim*3, 1)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3)
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', h=h)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """Transformer Block with Attention + FFN"""
    def __init__(self, dim, heads=8, ffn_expansion=4):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = ConvFFN(dim, ffn_expansion)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ConvTransformer(nn.Module):
    """Conv + Transformer Fusion Block"""
    def __init__(self, dim, heads=8, ffn_expansion=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.transformer = TransformerBlock(dim, heads, ffn_expansion)
        self.reduce = nn.Conv2d(dim*2, dim, 1)
        self.out = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        conv = self.act(self.conv(x))
        trans = self.transformer(x)
        return self.act(self.out(self.reduce(torch.cat([conv, trans], 1))))

class RawFormer(nn.Module):
    """Simplified RawFormer Architecture"""
    def __init__(self, in_ch=1, out_ch=3, dim=48, heads=[8,8,8,8], ffn_exp=2):
        super().__init__()
        
        # Encoder
        self.embed = nn.Conv2d(in_ch*4, dim, 3, padding=1)
        self.enc1 = ConvTransformer(dim, heads[0], ffn_exp)
        self.down1 = Downsample(dim)
        self.enc2 = ConvTransformer(dim*2, heads[1], ffn_exp)
        self.down2 = Downsample(dim*2)
        self.enc3 = ConvTransformer(dim*4, heads[2], ffn_exp)
        self.down3 = Downsample(dim*4)
        self.bottleneck = ConvTransformer(dim*8, heads[3], ffn_exp)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(dim*8, dim*4, 2, 2)
        self.dec1 = ConvTransformer(dim*4, heads[2], ffn_exp)
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, 2)
        self.dec2 = ConvTransformer(dim*2, heads[1], ffn_exp)
        self.up3 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.dec3 = ConvTransformer(dim, heads[0], ffn_exp)
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(dim, out_ch*4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        # Encoder
        x = F.pixel_unshuffle(x, 2)
        x = self.embed(x)
        
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.bottleneck(self.down3(x3))
        
        # Decoder
        x = self.dec1(torch.cat([self.up1(x4), x3], 1))
        x = self.dec2(torch.cat([self.up2(x), x2], 1))
        x = self.dec3(torch.cat([self.up3(x), x1], 1))
        
        return self.out(x)
