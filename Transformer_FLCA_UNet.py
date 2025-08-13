import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Bayer Luma + Chroma
# -----------------------
class BayerLumaChroma(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("r_w", torch.tensor(0.299, dtype=torch.float32))
        self.register_buffer("g_w", torch.tensor(0.587, dtype=torch.float32))
        self.register_buffer("b_w", torch.tensor(0.114, dtype=torch.float32))

    def forward(self, x):
        # x: [B,4,H,W] (R, G1, G2, B)
        r = x[:, 0:1]
        g = 0.5 * (x[:, 1:2] + x[:, 2:3])
        b = x[:, 3:4]
        y = self.r_w * r + self.g_w * g + self.b_w * b
        # per-image normalization helps stability
        y = y / (y.amax(dim=(2,3), keepdim=True).clamp_min(self.eps))
        cr = r - y
        cb = b - y
        return y, cr, cb

# -----------------------
# Frequency split
# -----------------------
def frequency_split(x, kernel_size=3):
    pad = kernel_size // 2
    low = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    high = x - low
    return low, high

# -----------------------
# Frequency-aware Luma–Chroma Attention (FLCA)
# -----------------------
class FLCA(nn.Module):
    def __init__(self, channels, r_ratio=8):
        super().__init__()
        # depthwise for low cost spatial cues
        self.low_attn  = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        self.high_attn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, groups=1, bias=False),
            nn.Tanh()
        )
        self.chroma_attn = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # channel attention (SE) driven by luminance stats
        hidden = max(8, channels // r_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )
        # learnable balances
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, y, cr, cb):
        B, C, H, W = feat.shape
        # resize guidance to feat size
        y  = F.interpolate(y,  size=(H,W), mode='bilinear', align_corners=False)
        cr = F.interpolate(cr, size=(H,W), mode='bilinear', align_corners=False)
        cb = F.interpolate(cb, size=(H,W), mode='bilinear', align_corners=False)

        y_low, y_high = frequency_split(y, kernel_size=3)
        a_low  = self.low_attn(y_low)     # [B,C,H,W]
        a_high = self.high_attn(y_high)   # [B,C,H,W]
        a_chr  = self.chroma_attn(torch.cat([cr, cb], dim=1))  # [B,C,H,W]

        # combined spatial attention
        spatial = 1 + self.alpha * a_low + self.beta * a_high + self.gamma * a_chr
        x = feat * spatial

        # channel attention (SE) on the modulated features
        ch = self.se(x)
        x = x * ch
        return x

# -----------------------
# Residual Conv Block (no BN), with dilation
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, c, dilation=1, residual_scale=0.2):
        super().__init__()
        pad = dilation
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1)
        )
        self.scale = residual_scale

    def forward(self, x):
        return x + self.body(x) * self.scale

# -----------------------
# ResBlock + Channel Attention
# -----------------------
class ResCA(nn.Module):
    def __init__(self, c, dilation=1, residual_scale=0.2, r_ratio=8):
        super().__init__()
        self.rb = ResBlock(c, dilation=dilation, residual_scale=residual_scale)
        hidden = max(8, c // r_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.rb(x)
        return x * self.se(x) + x  # lightweight residual-on-residual

# -----------------------
# Transformer Bottleneck (token-LN; shape-safe)
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
        self.res_scale = 0.2

    def forward(self, x):
        # x: [B,C,H,W] -> [B,HW,C]
        B, C, H, W = x.shape
        t = x.permute(0,2,3,1).reshape(B, H*W, C)
        t = t + self.attn(self.ln1(t), self.ln1(t), self.ln1(t))[0] * self.res_scale
        t = t + self.mlp(self.ln2(t)) * self.res_scale
        return t.view(B, H, W, C).permute(0,3,1,2)

# -----------------------
# Encoder Stage (multi-block + FLCA)
# -----------------------
class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        blocks = []
        for i in range(num_blocks):
            # alternate dilation for wider RF
            blocks += [ResCA(out_ch, dilation=1 if i%2==0 else 2, residual_scale=0.2)]
        self.blocks = nn.Sequential(*blocks)
        self.flca = FLCA(out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, y, cr, cb):
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.flca(x, y, cr, cb)
        skip = x
        x = self.down(x)
        return x, skip

# -----------------------
# Decoder Stage
# -----------------------
class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.GELU(),
            ResCA(out_ch, dilation=1, residual_scale=0.2),
            ResCA(out_ch, dilation=2, residual_scale=0.2),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

# -----------------------
# Full Model
# -----------------------
class Transformer_FLCA_UNet(nn.Module):
    """
    Better PSNR variant:
      - Deeper residual conv backbone w/ SE (no BN)
      - FLCA at each encoder stage
      - Transformer only at bottleneck
      - Global residual if shapes match (helps PSNR)
    """
    def __init__(self, in_ch=4, out_ch=4, base=48, blocks=(3,3,3), heads=4):
        super().__init__()
        self.luma = BayerLumaChroma()

        # Encoder (3 scales)
        self.enc1 = EncoderStage(in_ch,      base,     num_blocks=blocks[0])
        self.enc2 = EncoderStage(base,       base*2,   num_blocks=blocks[1])
        self.enc3 = EncoderStage(base*2,     base*4,   num_blocks=blocks[2])

        # Bottleneck (reduce once more for global transformer)
        self.down_bott = nn.Conv2d(base*4, base*4, 3, stride=2, padding=1)
        self.trans = TransformerBlock(base*4, num_heads=heads, mlp_ratio=4.0)
        self.up_bott = nn.ConvTranspose2d(base*4, base*4, kernel_size=2, stride=2)

        # Decoder
        self.dec3 = DecoderStage(base*4, base*4, base*4)  # fuse with enc3 skip
        self.dec2 = DecoderStage(base*4, base*2, base*2)  # fuse with enc2 skip
        self.dec1 = DecoderStage(base*2, base,   base)    # fuse with enc1 skip

        # Final mapping
        self.tail = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base//2, out_ch, 1)
        )

        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        # guidance
        y, cr, cb = self.luma(x)

        # encoder
        x1, s1 = self.enc1(x,  y, cr, cb)   # -> base, skip s1
        x2, s2 = self.enc2(x1, y, cr, cb)   # -> base*2, skip s2
        x3, s3 = self.enc3(x2, y, cr, cb)   # -> base*4, skip s3

        # bottleneck transformer
        b = self.down_bott(x3)
        b = self.trans(b)
        b = self.up_bott(b)
        if b.shape[-2:] != x3.shape[-2:]:
            b = F.interpolate(b, size=x3.shape[-2:], mode='bilinear', align_corners=False)

        # decoder
        d3 = self.dec3(b,  s3)      # -> base*4
        d2 = self.dec2(d3, s2)      # -> base*2
        d1 = self.dec1(d2, s1)      # -> base

        out = self.tail(d1)

        # global residual (helps PSNR when in/out channels match)
        if self.in_ch == self.out_ch:
            out = out + x
        return out

# -----------------------
# Quick test
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer_FLCA_UNet(in_ch=4, out_ch=4, base=48, blocks=(3,3,3), heads=4).to(device)
    model.eval()
    for hw in [(512,512), (128,128), (68,68)]:
        inp = torch.randn(1, 4, *hw, device=device)
        with torch.no_grad():
            out = model(inp)
        print(f"Input {hw} -> Output {tuple(out.shape)}")
