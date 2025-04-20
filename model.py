import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import os
from tqdm import tqdm
import imageio
import glob

class BasicOps:
    @staticmethod
    def to_3d(x):
        return rearrange(x, 'b c h w -> b (h w) c')

    @staticmethod
    def to_4d(x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.PixelUnshuffle(2)
        )
        
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        return BasicOps.to_4d(self.norm(BasicOps.to_3d(x)), h, h)

class ConvFFN(nn.Module):
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
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(1, heads, 1, 1))
        
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim*3, 1),
            nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3)
        )
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = rearrange(attn.softmax(dim=-1) @ v, 'b head c (h w) -> b (head c) h w', h=h)
        return self.proj(out)

class TransformerBlock(nn.Module):
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
    def __init__(self, dim, heads=8, ffn_expansion=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.transformer = TransformerBlock(dim, heads, ffn_expansion)
        self.reduce = nn.Conv2d(dim*2, dim, 1)
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        conv = self.conv(x)
        trans = self.transformer(x)
        return self.out(self.reduce(torch.cat([conv, trans], 1)))

class RawFormer(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, dim=48, heads=[8]*4, ffn_exp=2):
        super().__init__()
        
        # Encoder
        self.embed = nn.Conv2d(in_ch*4, dim, 3, padding=1)
        self.encoder = nn.ModuleList([
            ConvTransformer(dim * (2**i), heads[i], ffn_exp)
            for i in range(3)
        ])
        self.downsamples = nn.ModuleList([
            Downsample(dim * (2**i))
            for i in range(3)
        ])
        self.bottleneck = ConvTransformer(dim*8, heads[3], ffn_exp)
        
        # Decoder
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(dim * (2**(3-i)), dim * (2**(2-i)), 2, 2)
            for i in range(3)
        ])
        self.decoder = nn.ModuleList([
            ConvTransformer(dim * (2**(2-i)), heads[2-i], ffn_exp)
            for i in range(3)
        ])
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(dim, out_ch*4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        # Encoder
        x = F.pixel_unshuffle(x, 2)
        x = self.embed(x)
        
        features = []
        for enc, down in zip(self.encoder, self.downsamples):
            features.append(x)
            x = enc(x)
            x = down(x)
        
        x = self.bottleneck(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder)):
            x = up(x)
            x = dec(torch.cat([x, features[-1-i]], 1))
        
        return self.output(x)

