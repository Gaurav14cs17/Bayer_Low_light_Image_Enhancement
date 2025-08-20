import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ptflops import get_model_complexity_info


# -----------------------
# utils
# -----------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):
    """
    Down Shuffle function, same as nn.PixelUnshuffle().
    Input: variable of size (B × C × H × W)
    Output: down-shuffled var of size (B × (C*r^2) × H/r × W/r)
    """
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return (var.contiguous()
               .view(b, c, out_h, r, out_w, r)
               .permute(0, 1, 3, 5, 2, 4)
               .contiguous()
               .view(b, out_channel, out_h, out_w)
               .contiguous())


# -----------------------
# Wavelet: 2x2 orthogonal Haar
# -----------------------
class HaarDWT(nn.Module):
    """
    Orthogonal 2x2 Haar analysis filter bank with stride=2.
    For odd H/W, uses reflect padding on right/bottom to make them even.
    Returns: LL, (LH, HL, HH) each with spatial size ceil(H/2) x ceil(W/2).
    """
    def __init__(self):
        super().__init__()
        h = torch.tensor([1.0,  1.0]) / math.sqrt(2.0)
        g = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        LL = torch.outer(h, h)
        LH = torch.outer(h, g)
        HL = torch.outer(g, h)
        HH = torch.outer(g, g)
        filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filt", filt)  # non-trainable

    def forward(self, x: torch.Tensor):
        """
        x: [B,C,H,W]
        y: [B,4C,H/2,W/2] (ceil if odd, due to reflect pad)
        """
        B, C, H, W = x.shape
        # reflect-pad to even H/W so 2x2 stride-2 covers the grid
        pad_h = H & 1  # 1 if odd else 0
        pad_w = W & 1
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # (left,right,top,bottom)

        filt = self.filt.repeat(C, 1, 1, 1)  # [4C,1,2,2]
        y = F.conv2d(x, filt, stride=2, padding=0, groups=C)  # [B,4C,H2,W2]
        H2, W2 = y.shape[-2], y.shape[-1]
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return LL, (LH, HL, HH)


# -----------------------
# Bayer Luma + Chroma from RGGB planes
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
# Frequency-aware Luma–Chroma Attention (FLCA)
# -----------------------
class FLCA(nn.Module):
    def __init__(self, channels, r_ratio=8, eps=1e-8):
        super().__init__()
        self.dwt = HaarDWT()
        self.eps = eps
        # per-spatial attention heads (1 -> C)
        self.low_attn  = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.high_attn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.Tanh()
        )
        # chroma guidance (2 -> C)
        self.chroma_attn = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # channel attention (SE) driven by the modulated features
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
        B, C, Hf, Wf = feat.shape

        # Wavelet subbands on luminance
        LL, (LH, HL, HH) = self.dwt(y)
        y_high_mag = torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + self.eps)  # [B,1,Hy/2,Wy/2]
        y_low = LL  # [B,1,Hy/2,Wy/2]

        # Resize guidance to feature map size
        y_low  = F.interpolate(y_low,  size=(Hf, Wf), mode='bilinear', align_corners=False)
        y_high = F.interpolate(y_high_mag, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cr     = F.interpolate(cr, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cb     = F.interpolate(cb, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # Spatial attentions
        a_low  = self.low_attn(y_low)          # [B,C,Hf,Wf]
        a_high = self.high_attn(y_high)        # [B,C,Hf,Wf]
        a_chr  = self.chroma_attn(torch.cat([cr, cb], dim=1))  # [B,C,Hf,Wf]

        # Combined spatial attention
        spatial = 1 + self.alpha * a_low + self.beta * a_high + self.gamma * a_chr
        x = feat * spatial

        # Channel attention (SE)
        ch = self.se(x)
        x = x * ch
        return x


# -----------------------
# Core blocks
# -----------------------
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # reduce channels by 2, then PixelUnshuffle x4 -> net x2 channels (matches your schedule)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return downshuffle(self.body(x), 2)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class conv_ffn(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                         dilation=1, groups=hidden_features)
        self.pointwise2 = torch.nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    from restormer
    input size: (B,C,H,W)
    output size: (B,C,H,W)
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = conv_ffn(dim, dim * ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Conv_Transformer(nn.Module):
    """
    conv branch -> replaced by FLCA
    """
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2):
        super().__init__()
        self.FLCA = FLCA(in_channel)  # new branch
        self.Transformer = TransformerBlock(
            dim=in_channel, num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor, bias=True
        )
        self.channel_reduce = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1)
        self.Conv_out = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, feat, y, cr, cb):
        flca_feat = self.FLCA(feat, y, cr, cb)
        trans = self.Transformer(feat)
        x = torch.cat([flca_feat, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x


# -----------------------
# RawFormer with FLCA
# -----------------------
class RawFormer(nn.Module):
    """
    RawFormer model with FLCA.
    Args:
        inp_channels (int): Input channel number, 1 for RAW Bayer.
        out_channels (int): Output channel number, e.g., 3 for RGB.
        dim (int): Base embedding dimension.
        num_heads (list): Transformer heads per stage.
        ffn_expansion_factor (int): FFN expansion.
    """

    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8, 8, 8, 8], ffn_expansion_factor=2):
        super(RawFormer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        # Bayer guidance
        self.luma_chroma = BayerLumaChroma()

        # Input packing (x -> 4 planes via downshuffle) then embed
        self.embedding = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.conv_tran1 = Conv_Transformer(dim,     num_heads[0], ffn_expansion_factor)
        self.down1      = Downsample(dim)          # -> 2*dim
        self.conv_tran2 = Conv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)
        self.down2      = Downsample(dim * 2)      # -> 4*dim
        self.conv_tran3 = Conv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)
        self.down3      = Downsample(dim * 4)      # -> 8*dim
        self.conv_tran4 = Conv_Transformer(dim * 8, num_heads[3], ffn_expansion_factor)

        # Decoder
        self.up1 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1, 1)
        self.conv_tran5 = Conv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.conv_tran6 = Conv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)

        self.up3 = nn.ConvTranspose2d(dim * 2, dim * 1, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim * 1, 1, 1)
        self.conv_tran7 = Conv_Transformer(dim, num_heads[0], ffn_expansion_factor)

        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # Pack Bayer to 4 channels at half-res
        x_ds = downshuffle(x, 2)  # [B,4,H/2,W/2]

        # Compute guidance from Bayer planes (same res as x_ds)
        y, cr, cb = self.luma_chroma(x_ds)  # [B,1,H/2,W/2], [B,1,H/2,W/2], [B,1,H/2,W/2]

        # Embed
        x0 = self.embedding(x_ds)  # [B,dim,H/2,W/2]

        # Encoder
        conv_tran1 = self.conv_tran1(x0, y, cr, cb)          # [B,dim,H/2,W/2]
        pool1 = self.down1(conv_tran1)                       # [B,2*dim,H/4,W/4]

        conv_tran2 = self.conv_tran2(pool1, y, cr, cb)       # [B,2*dim,H/4,W/4]
        pool2 = self.down2(conv_tran2)                       # [B,4*dim,H/8,W/8]

        conv_tran3 = self.conv_tran3(pool2, y, cr, cb)       # [B,4*dim,H/8,W/8]
        pool3 = self.down3(conv_tran3)                       # [B,8*dim,H/16,W/16]

        conv_tran4 = self.conv_tran4(pool3, y, cr, cb)       # [B,8*dim,H/16,W/16]

        # Decoder
        up1 = self.up1(conv_tran4)                           # [B,4*dim,H/8,W/8]
        concat1 = torch.cat([up1, conv_tran3], 1)            # [B,8*dim,H/8,W/8]
        concat1 = self.channel_reduce1(concat1)              # [B,4*dim,H/8,W/8]
        conv_tran5 = self.conv_tran5(concat1, y, cr, cb)     # [B,4*dim,H/8,W/8]

        up2 = self.up2(conv_tran5)                           # [B,2*dim,H/4,W/4]
        concat2 = torch.cat([up2, conv_tran2], 1)            # [B,4*dim,H/4,W/4]
        concat2 = self.channel_reduce2(concat2)              # [B,2*dim,H/4,W/4]
        conv_tran6 = self.conv_tran6(concat2, y, cr, cb)     # [B,2*dim,H/4,W/4]

        up3 = self.up3(conv_tran6)                           # [B,dim,H/2,W/2]
        concat3 = torch.cat([up3, conv_tran1], 1)            # [B,2*dim,H/2,W/2]
        concat3 = self.channel_reduce3(concat3)              # [B,dim,H/2,W/2]
        conv_tran7 = self.conv_tran7(concat3, y, cr, cb)     # [B,dim,H/2,W/2]

        conv_out = self.lrelu(self.conv_out(conv_tran7))     # [B,4*out,H/2,W/2]
        out = self.pixelshuffle(conv_out)                    # [B,out,H,W]
        return out


# -----------------------
# quick check
# -----------------------
if __name__ == "__main__":
    model = RawFormer(dim=48)
    # FLOPs/params on a RAW Bayer input (B,1,512,512)
    ops, params = get_model_complexity_info(model, (1, 512, 512),
                                            as_strings=True, print_per_layer_stat=False, verbose=False)
    print(ops, params)
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
