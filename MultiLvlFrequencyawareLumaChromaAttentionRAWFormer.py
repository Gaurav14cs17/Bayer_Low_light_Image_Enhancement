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
    def __init__(self):
        super().__init__()
        h = torch.tensor([1.0,  1.0]) / math.sqrt(2.0)
        g = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        LL = torch.outer(h, h)
        LH = torch.outer(h, g)
        HL = torch.outer(g, h)
        HH = torch.outer(g, g)
        filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filt", filt)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        pad_h = H & 1
        pad_w = W & 1
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        filt = self.filt.repeat(C, 1, 1, 1)  # [4C,1,2,2]
        y = F.conv2d(x, filt, stride=2, padding=0, groups=C)  # [B,4C,H2,W2]
        H2, W2 = y.shape[-2], y.shape[-1]
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return LL, (LH, HL, HH)


# -----------------------
# Bayer Luma + Chroma (packed RGGB planes)
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
        # normalize per-image to avoid scale issues
        y = y / (y.amax(dim=(2,3), keepdim=True).clamp_min(self.eps))
        cr = r - y
        cb = b - y
        return y, cr, cb


# -----------------------
# Multi-Level FLCA (Pyramid) with gated residuals (stabilized)
# -----------------------
class FLCA_Pyramid(nn.Module):
    """
    Multi-level FLCA with gates and magnitude-limited residuals to protect color.
    """
    def __init__(self, channels, levels=2, r_ratio=8, eps=1e-8, max_residual_scale=0.2):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.eps = eps
        self.max_residual_scale = float(max_residual_scale)  # maximum allowed residual magnitude (relative)
        self.dwt = HaarDWT()

        # per-level spatial attentions
        self.low_attn  = nn.ModuleList([nn.Sequential(nn.Conv2d(1, channels, 3, padding=1, bias=False), nn.Sigmoid()) for _ in range(levels)])
        self.high_attn = nn.ModuleList([nn.Sequential(nn.Conv2d(1, channels, 3, padding=1, bias=False), nn.Tanh())    for _ in range(levels)])

        # per-level gate heads (2 -> 2 scalars)
        self.freq_gate_head = nn.ModuleList([nn.Conv2d(2, 2, kernel_size=1, bias=True) for _ in range(levels)])

        # chroma attention and gate
        self.chroma_attn = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.chroma_gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # channel attention (SE)
        hidden = max(8, channels // r_ratio)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

        # small projector predicting residual delta (kept small by tanh * scale)
        self.res_proj = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 1))

    @staticmethod
    def _hf_mag(LH, HL, HH, eps):
        return torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + eps)

    def _pyramid_y(self, y):
        lows, highs = [], []
        cur = y
        for _ in range(self.levels):
            LL, (LH, HL, HH) = self.dwt(cur)
            lows.append(LL)
            highs.append(self._hf_mag(LH, HL, HH, self.eps))
            cur = LL
        return lows, highs

    def forward(self, feat, y, cr, cb, return_ll_anchor=False):
        """
        feat: [B,C,Hf,Wf]
        y:    [B,1,Hy,Wy]
        cr,cb: [B,1,Hy,Wy]
        """
        B, C, Hf, Wf = feat.shape
        x = feat
        lows, highs = self._pyramid_y(y)  # lists length = levels

        # progressive residual fusion with gates; residuals magnitude-limited
        for l in range(self.levels):
            y_low  = F.interpolate(lows[l],  size=(Hf, Wf), mode='bilinear', align_corners=False)
            y_high = F.interpolate(highs[l], size=(Hf, Wf), mode='bilinear', align_corners=False)

            a_low  = self.low_attn[l](y_low)    # [B,C,Hf,Wf]
            a_high = self.high_attn[l](y_high)  # [B,C,Hf,Wf]

            # gates produced from pooled low/high stats -> stable sigmoid
            low_g  = F.adaptive_avg_pool2d(y_low, 1)   # [B,1,1,1]
            high_g = F.adaptive_avg_pool2d(y_high,1)   # [B,1,1,1]
            g_in   = torch.cat([low_g, high_g], dim=1) # [B,2,1,1]
            gates  = torch.sigmoid(self.freq_gate_head[l](g_in))  # [B,2,1,1]
            alpha_l = gates[:, 0:1]  # (B,1,1,1)
            beta_l  = gates[:, 1:2]

            # spatial modulation
            spatial = alpha_l * a_low + beta_l * a_high  # [B,C,Hf,Wf]

            # predict residual, but limit magnitude:
            raw_res = self.res_proj(x * spatial)        # arbitrary scale
            # limit magnitude via tanh and a small multiplier -> ensures we don't overwrite LL anchor
            res = torch.tanh(raw_res) * self.max_residual_scale
            x = x + res  # residual fusion

        # chroma residual, with gated magnitude-limiting
        cr_r = F.interpolate(cr, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cb_r = F.interpolate(cb, size=(Hf, Wf), mode='bilinear', align_corners=False)
        a_chr = self.chroma_attn(torch.cat([cr_r, cb_r], dim=1))  # [B,C,Hf,Wf]
        chr_mag = torch.sqrt(cr_r.pow(2) + cb_r.pow(2) + self.eps)  # [B,1,Hf,Wf]
        gamma = torch.sigmoid(self.chroma_gate(F.adaptive_avg_pool2d(chr_mag,1)))  # [B,1,1,1]
        raw_res_c = self.res_proj(x * (gamma * a_chr))
        res_c = torch.tanh(raw_res_c) * self.max_residual_scale
        x = x + res_c

        # channel attention
        ch = self.se(x)
        x = x * ch

        if return_ll_anchor:
            return x, lows[-1]
        return x


# -----------------------
# Core blocks (Transformer, FFN etc.)
# -----------------------
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class conv_ffn(nn.Module):
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
        super().__init__()
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
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = conv_ffn(dim, dim * ffn_expansion_factor, dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Conv_Transformer(nn.Module):
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2, flca_levels=2):
        super().__init__()
        self.FLCA = FLCA_Pyramid(in_channel, levels=flca_levels)
        self.Transformer = TransformerBlock(dim=in_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=True)
        self.channel_reduce = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1)
        self.Conv_out = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
    def forward(self, feat, y, cr, cb):
        flca_feat = self.FLCA(feat, y, cr, cb)
        trans = self.Transformer(feat)
        x = torch.cat([flca_feat, trans], dim=1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x


# -----------------------
# Color anchor correction (per-channel mean correction)
# -----------------------
def color_anchor_correction_rgb(out_rgb, input_packed_bayer, alpha=0.12):
    """
    out_rgb: [B,3,H,W] (model output RGB)
    input_packed_bayer: [B,4,h,w] (packed RGGB at half res)
    alpha: strength of correction (0..1). small values are safer.
    Returns corrected RGB image.
    """
    # simple linear demosaic at packed resolution
    R = input_packed_bayer[:,0:1]
    G = 0.5 * (input_packed_bayer[:,1:2] + input_packed_bayer[:,2:3])
    B = input_packed_bayer[:,3:4]
    in_rgb = torch.cat([R, G, B], dim=1)  # [B,3,h,w]
    # upsample to out size
    in_rgb_full = F.interpolate(in_rgb, size=(out_rgb.shape[2], out_rgb.shape[3]), mode='bilinear', align_corners=False)
    in_mean = in_rgb_full.mean(dim=(2,3), keepdim=True)   # [B,3,1,1]
    out_mean = out_rgb.mean(dim=(2,3), keepdim=True)      # [B,3,1,1]
    # add small correction towards input mean
    corrected = out_rgb + alpha * (in_mean - out_mean)
    return corrected


# -----------------------
# Color-consistency loss
# -----------------------
def color_consistency_loss_rgb(pred_rgb, input_packed_bayer):
    """
    pred_rgb: [B,3,H,W]
    input_packed_bayer: [B,4,h,w]
    Returns L2 loss between per-channel means.
    """
    R = input_packed_bayer[:,0:1]
    G = 0.5 * (input_packed_bayer[:,1:2] + input_packed_bayer[:,2:3])
    B = input_packed_bayer[:,3:4]
    in_rgb = torch.cat([R, G, B], dim=1)
    in_rgb_full = F.interpolate(in_rgb, size=(pred_rgb.shape[2], pred_rgb.shape[3]), mode='bilinear', align_corners=False)
    in_mean = in_rgb_full.mean(dim=(2,3))
    out_mean = pred_rgb.mean(dim=(2,3))
    return F.mse_loss(out_mean, in_mean)


# -----------------------
# Full RawFormer with the above protections
# -----------------------
class RawFormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8,8,8,8], ffn_expansion_factor=2, flca_levels=2):
        super(RawFormer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.luma_chroma = BayerLumaChroma()
        self.embedding = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, stride=1, padding=1)

        # encoder / decoder
        self.conv_tran1 = Conv_Transformer(dim, num_heads[0], ffn_expansion_factor, flca_levels)
        self.down1      = nn.Sequential(nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv_tran2 = Conv_Transformer(dim*2, num_heads[1], ffn_expansion_factor, flca_levels)
        self.down2      = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv_tran3 = Conv_Transformer(dim*4, num_heads[2], ffn_expansion_factor, flca_levels)
        self.down3      = nn.Sequential(nn.Conv2d(dim*4, dim*2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv_tran4 = Conv_Transformer(dim*8, num_heads[3], ffn_expansion_factor, flca_levels)

        # decoder ups
        self.up1 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim*8, dim*4, 1, 1)
        self.conv_tran5 = Conv_Transformer(dim*4, num_heads[2], ffn_expansion_factor, flca_levels)

        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim*4, dim*2, 1, 1)
        self.conv_tran6 = Conv_Transformer(dim*2, num_heads[1], ffn_expansion_factor, flca_levels)

        self.up3 = nn.ConvTranspose2d(dim*2, dim*1, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim*2, dim*1, 1, 1)
        self.conv_tran7 = Conv_Transformer(dim, num_heads[0], ffn_expansion_factor, flca_levels)

        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)

        # Haar for explicit LL anchor and optional luminance nudge
        self.haar = HaarDWT()

    @staticmethod
    def simple_demosaic_from_packed(x_ds):
        # x_ds: [B,4,h,w] (R,G1,G2,B) at packed resolution
        R = x_ds[:, 0:1]
        G = 0.5 * (x_ds[:, 1:2] + x_ds[:, 2:3])
        B = x_ds[:, 3:4]
        return torch.cat([R, G, B], dim=1)  # [B,3,h,w]

    def forward(self, x):
        # x: [B,1,H,W] raw Bayer
        x_ds = downshuffle(x, 2)  # [B,4,H/2,W/2]
        y, cr, cb = self.luma_chroma(x_ds)  # guidance at packed res

        # compute deep LL anchor (2-level by default; adjust if you used different flca_levels)
        cur = y
        ll_anchor = None
        for _ in range(2):
            LL, (LH, HL, HH) = self.haar(cur)
            cur = LL
            ll_anchor = LL  # final LL is deepest low-res

        # embed
        x0 = self.embedding(x_ds)  # [B,dim,h,w]

        # stage 1
        conv_tran1 = self.conv_tran1(x0, y, cr, cb)          # [B,dim,h,w]
        pool1 = downshuffle(self.down1(conv_tran1), 2)      # replicate previous Downsample behaviour: conv then pixelunshuffle

        conv_tran2 = self.conv_tran2(pool1, y, cr, cb)
        pool2 = downshuffle(self.down2(conv_tran2), 2)

        conv_tran3 = self.conv_tran3(pool2, y, cr, cb)
        pool3 = downshuffle(self.down3(conv_tran3), 2)

        conv_tran4 = self.conv_tran4(pool3, y, cr, cb)

        up1 = self.up1(conv_tran4)
        concat1 = torch.cat([up1, conv_tran3], 1)
        concat1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1, y, cr, cb)

        up2 = self.up2(conv_tran5)
        concat2 = torch.cat([up2, conv_tran2], 1)
        concat2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2, y, cr, cb)

        up3 = self.up3(conv_tran6)
        concat3 = torch.cat([up3, conv_tran1], 1)
        concat3 = self.channel_reduce3(concat3)
        conv_tran7 = self.conv_tran7(concat3, y, cr, cb)

        conv_out = self.lrelu(self.conv_out(conv_tran7))
        out = self.pixelshuffle(conv_out)  # [B,3,H,W]

        # 1) Per-channel mean color anchor correction (simple and powerful)
        out = color_anchor_correction_rgb(out, x_ds, alpha=0.12)

        # 2) Optional tiny luminance nudge from LL anchor (very small to prevent hue shifts)
        if ll_anchor is not None:
            # upsample ll_anchor to full res
            ll_up = F.interpolate(ll_anchor, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=False)
            # convert out RGB -> Y (using same weights)
            r_w, g_w, b_w = 0.299, 0.587, 0.114
            out_y = r_w * out[:,0:1] + g_w * out[:,1:2] + b_w * out[:,2:3]
            # small residual towards ll anchor luminance
            y_residual = (ll_up - out_y) * 0.03  # very small coefficient
            out = out + torch.cat([y_residual, y_residual, y_residual], dim=1)

        return out


# -----------------------
# quick check
# -----------------------
if __name__ == "__main__":
    model = RawFormer(dim=48, flca_levels=2)
    # FLOPs/params on a RAW Bayer input (B,1,512,512)
    ops, params = get_model_complexity_info(model, (1, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print("FLOPs, Params:", ops, params)
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
