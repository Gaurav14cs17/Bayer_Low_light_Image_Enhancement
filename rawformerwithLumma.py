import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Luminance + Chrominance extraction -----
class BayerLumaChroma(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("r_w", torch.tensor(0.299))
        self.register_buffer("g_w", torch.tensor(0.587))
        self.register_buffer("b_w", torch.tensor(0.114))

    def forward(self, x):
        r = x[:, 0:1]
        g = 0.5 * (x[:, 1:2] + x[:, 2:3])
        b = x[:, 3:4]
        y = self.r_w * r + self.g_w * g + self.b_w * b
        cr = r - y
        cb = b - y
        return y, cr, cb

# ----- Frequency Split -----
def frequency_split(x, kernel_size=15):
    low = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    high = x - low
    return low, high

# ----- FLCA (Frequency-aware Luma-Chroma Attention) -----
class FLCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_attn = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.Sigmoid())
        self.high_attn = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.Tanh())
        self.chroma_attn = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.Sigmoid())

    def forward(self, feat, y, cr, cb):
        y_low, y_high = frequency_split(y)
        low_a = self.low_attn(y_low)
        high_a = self.high_attn(y_high)
        chroma_a = self.chroma_attn(torch.cat([cr, cb], dim=1))
        feat = feat * (1 + low_a) + feat * high_a + feat * chroma_a
        return feat

# ----- Transformer Encoder Block -----
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_attn, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + x_attn
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).reshape(B, C, H, W)

# ----- Conv block -----
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# ----- Hybrid Transformer-UNet with FLCA -----
class Transformer_FLCA_UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base_ch=32):
        super().__init__()
        self.luma_chroma = BayerLumaChroma()

        # Encoder CNN stages
        self.enc1 = conv_block(in_ch, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)

        self.pool = nn.MaxPool2d(2)

        # Transformer stages with FLCA
        self.trans1 = TransformerBlock(base_ch)
        self.flca1 = FLCA(base_ch)

        self.trans2 = TransformerBlock(base_ch*2)
        self.flca2 = FLCA(base_ch*2)

        self.trans3 = TransformerBlock(base_ch*4)
        self.flca3 = FLCA(base_ch*4)

        # Bottleneck
        self.bottleneck = TransformerBlock(base_ch*8)
        self.flca_bottleneck = FLCA(base_ch*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y, cr, cb = self.luma_chroma(x)

        # Encoder + Transformer + FLCA
        e1 = self.enc1(x)
        e1 = self.trans1(e1)
        e1 = self.flca1(e1, y, cr, cb)

        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        e2 = self.trans2(e2)
        e2 = self.flca2(e2, y, cr, cb)

        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e3 = self.trans3(e3)
        e3 = self.flca3(e3, y, cr, cb)

        b = self.pool(e3)
        b = self.bottleneck(b)
        b = self.flca_bottleneck(b, y, cr, cb)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# ----- Test -----
if __name__ == "__main__":
    model = Transformer_FLCA_UNet()
    dummy = torch.rand(1, 4, 256, 256)
    out = model(dummy)
    print(out.shape)  # Expected: [1, 4, 256, 256]
