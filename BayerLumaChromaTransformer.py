import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

# -----------------------
# Bayer Luma + Chroma
# -----------------------
class BayerLumaChroma(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # make these learnable if you want; fixed here for stability
        self.register_buffer("r_w", torch.tensor(0.299, dtype=torch.float32))
        self.register_buffer("g_w", torch.tensor(0.587, dtype=torch.float32))
        self.register_buffer("b_w", torch.tensor(0.114, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,4,H,W] in RGGB order
        r = x[:, 0:1]
        g = 0.5 * (x[:, 1:2] + x[:, 2:3])
        b = x[:, 3:4]
        y = self.r_w * r + self.g_w * g + self.b_w * b
        # Normalize per-image for numerical stability
        denom = y.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        y = y / denom
        cr = r - y
        cb = b - y
        return y, cr, cb

# -----------------------
# Multi-scale frequency split
# -----------------------
class MultiScaleFrequencySplit(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int] = (7, 15, 31)):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W]
        returns: concatenated highs [B, len(kernel_sizes), H, W]
        """
        highs = []
        for k in self.kernel_sizes:
            if k <= 1:
                low = x
            else:
                pad = k // 2
                low = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
            highs.append(x - low)
        return torch.cat(highs, dim=1)

# -----------------------
# FLCA block
# -----------------------
class FLCA(nn.Module):
    def __init__(self, channels: int, freq_kernels: Sequence[int] = (7, 15, 31)):
        """
        channels: number of channels in feature map 'feat'
        freq_kernels: kernels that MultiScaleFrequencySplit will use
        """
        super().__init__()
        self.freq_kernels = list(freq_kernels)
        self.freq_split = MultiScaleFrequencySplit(self.freq_kernels)

        # low (1) -> channels
        self.low_attn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.Sigmoid()
        )

        # high (len(freq_kernels)) -> channels
        high_in = len(self.freq_kernels)
        self.high_attn = nn.Sequential(
            nn.Conv2d(high_in, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.Tanh()
        )

        # chroma (Cr, Cb) -> channels
        self.chroma_attn = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.Sigmoid()
        )

        # small residual conv to refine
        self.refine = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, feat: torch.Tensor, y: torch.Tensor, cr: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, Hf, Wf]; y/cr/cb: [B,1,H,W] full-res
        Hf, Wf = feat.shape[2], feat.shape[3]
        # resize guidance
        y_r = F.interpolate(y, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cr_r = F.interpolate(cr, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cb_r = F.interpolate(cb, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # low/high components
        low = F.avg_pool2d(y_r, kernel_size=15, stride=1, padding=15 // 2)  # single low map
        highs = self.freq_split(y_r)  # multiscale highs concatenated

        low_a = self.low_attn(low)            # [B, C, Hf, Wf]
        high_a = self.high_attn(highs)        # [B, C, Hf, Wf]
        chroma_a = self.chroma_attn(torch.cat([cr_r, cb_r], dim=1))  # [B, C, Hf, Wf]

        combined = 1.0 + low_a + high_a + chroma_a
        out = feat * combined
        out = out + self.refine(out)
        return out

# -----------------------
# Resolution-agnostic Transformer block
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.local_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # depthwise conv
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, L, C]
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        local = self.local_enhance(x)  # [B, C, H, W]
        # combine token path and local path
        x_comb = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2) + local

        tokens2 = x_comb.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens2 = tokens2 + self.mlp(self.norm2(tokens2))
        x_final = tokens2.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_final

# -----------------------
# Small conv block used in encoder/decoder (no BN for PSNR; InstanceNorm is OK)
# -----------------------
def conv_block(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

# -----------------------
# Encoder stage
# -----------------------
class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, freq_kernels: Sequence[int] = (7, 15, 31)):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(conv_block(out_ch, out_ch))
        self.blocks = nn.Sequential(*blocks)
        self.trans = TransformerBlock(out_ch)
        self.flca = FLCA(out_ch, freq_kernels=freq_kernels)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # downsample

    def forward(self, x: torch.Tensor, y: torch.Tensor, cr: torch.Tensor, cb: torch.Tensor):
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.trans(x)
        x = self.flca(x, y, cr, cb)
        skip = x
        x = self.down(x)
        return x, skip

# -----------------------
# Decoder stage
# -----------------------
class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

# -----------------------
# Bottleneck block (wrap conv-down, transformer, FLCA, conv-up)
# -----------------------
class BottleneckBlock(nn.Module):
    def __init__(self, ch: int, freq_kernels: Sequence[int] = (7, 15, 31)):
        super().__init__()
        self.conv_down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)   # reduce spatial by 2
        self.trans = TransformerBlock(ch)
        self.flca = FLCA(ch, freq_kernels=freq_kernels)
        self.conv_up = nn.ConvTranspose2d(ch, ch, 2, stride=2)       # restore spatial

    def forward(self, x: torch.Tensor, y: torch.Tensor, cr: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.trans(x)
        x = self.flca(x, y, cr, cb)
        x = self.conv_up(x)
        return x

# -----------------------
# Full network
# -----------------------
class Transformer_FLCA_UNet_Full(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 48, freq_kernels: Sequence[int] = (7, 15, 31)):
        super().__init__()
        self.luma = BayerLumaChroma()
        self.freq_kernels = list(freq_kernels)

        # Encoder stages
        self.enc1 = EncoderStage(in_ch, base, num_blocks=2, freq_kernels=self.freq_kernels)
        self.enc2 = EncoderStage(base, base * 2, num_blocks=2, freq_kernels=self.freq_kernels)
        self.enc3 = EncoderStage(base * 2, base * 4, num_blocks=2, freq_kernels=self.freq_kernels)

        # Bottleneck
        self.bottleneck = BottleneckBlock(base * 4, freq_kernels=self.freq_kernels)

        # Decoder stages
        self.dec3 = DecoderStage(base * 4, base * 4, base * 4)
        self.dec2 = DecoderStage(base * 4, base * 2, base * 2)
        self.dec1 = DecoderStage(base * 2, base, base)

        # final
        self.tail = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base // 2, out_ch, 1)
        )

        # residual projection if needed
        self.res_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _adjust_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        raw: [B, 4, H, W] (RGGB)
        returns: [B, out_ch, H, W]
        """
        y, cr, cb = self.luma(raw)  # full-res guidance [B,1,H,W]

        # encoder
        x1, s1 = self.enc1(raw, y, cr, cb)  # x1: downsampled; s1 skip at full res of stage1
        x2, s2 = self.enc2(x1, y, cr, cb)
        x3, s3 = self.enc3(x2, y, cr, cb)

        # bottleneck (accepts x3 and guidance)
        b = self.bottleneck(x3, y, cr, cb)

        # decoder
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.tail(d1)

        # add global residual (project input if necessary and resize)
        res = raw if self.res_proj is None else self.res_proj(raw)
        res = F.interpolate(res, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + res
        return out

# -----------------------
# Quick multi-size test
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer_FLCA_UNet_Full(in_ch=4, out_ch=4, base=32, freq_kernels=(7, 15, 31)).to(device)
    model.eval()

    for hw in [(68, 68)]:
        inp = torch.randn(1, 4, hw[0], hw[1], device=device)
        with torch.no_grad():
            out = model(inp)
        print(f"Input {hw} -> Output {tuple(out.shape)}")
