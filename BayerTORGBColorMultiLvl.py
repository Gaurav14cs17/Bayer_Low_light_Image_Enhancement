


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------
# utils
# -----------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):
    """
    Down shuffle with safe reflect-padding when H/W not divisible by r.
    Input: (B × C × H × W)
    Output: (B × (C*r^2) × H/r × W/r)
    """
    b, c, h, w = var.size()
    pad_h = (r - (h % r)) % r
    pad_w = (r - (w % r)) % r
    if pad_h or pad_w:
        var = F.pad(var, (0, pad_w, 0, pad_h), mode='reflect')
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
        h = torch.tensor([1.0,  1.0], dtype=torch.float32) / math.sqrt(2.0)
        g = torch.tensor([1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
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
        y = F.conv2d(x, filt, stride=2, padding=0, groups=C)
        H2, W2 = y.shape[-2], y.shape[-1]
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return LL, (LH, HL, HH)

# -----------------------
# Enhanced Bayer Processing for True Color
# -----------------------
class EnhancedBayerProcessor(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # Learnable white balance gains (camera-typical init: R/B slightly > G)
        self.wb_gains = nn.Parameter(torch.tensor([1.8, 1.0, 1.0, 1.6], dtype=torch.float32))  # R, G1, G2, B

        # Learnable color matrix (3x4: 3x3 matrix + 3x1 bias)
        self.color_matrix = nn.Parameter(torch.cat([torch.eye(3, 3, dtype=torch.float32),
                                                    torch.zeros(3, 1, dtype=torch.float32)], dim=1))

        # Demosaic refinement on linear RGB
        self.demosaic_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

        # Chroma extraction (Cr, Cb) using R, G, B, Y
        self.chroma_extractor = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh()
        )

        # BT.709 luminance weights
        self.register_buffer("y_weights", torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32))

    def forward(self, x):
        # x: [B,4,H,W] (R, G1, G2, B)

        # Positive WB gains to avoid color sign flips
        gains = F.softplus(self.wb_gains) + 1e-6
        wb_x = x * gains.view(1, 4, 1, 1)

        # Linear demosaic (pack -> RGB) before color matrix
        r = wb_x[:, 0:1]
        g = 0.5 * (wb_x[:, 1:2] + wb_x[:, 2:3])
        b = wb_x[:, 3:4]
        rgb = torch.cat([r, g, b], dim=1)  # [B,3,H,W]

        # Apply 3x3 color matrix + bias
        rgb_perm = rgb.permute(0, 2, 3, 1)                # [B,H,W,3]
        M    = self.color_matrix[:, :3]                   # [3,3]
        bias = self.color_matrix[:, 3].view(1, 1, 1, 3)   # [1,1,1,3]
        rgb_lin = torch.matmul(rgb_perm, M.t()) + bias    # [B,H,W,3]
        rgb_linear = rgb_lin.permute(0, 3, 1, 2).contiguous()

        # Luminance (normalized per-image to stabilize training)
        y = torch.sum(rgb_linear * self.y_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        y = y / (y.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps))

        # Chroma (learned)
        chroma_in = torch.cat([r, g, b, y], dim=1)
        chroma = self.chroma_extractor(chroma_in)
        cr, cb = chroma[:, 0:1], chroma[:, 1:2]

        # Demosaic refinement (residual)
        refined_rgb = rgb_linear + self.demosaic_refine(rgb_linear)

        return y, cr, cb, refined_rgb

# -----------------------
# Camera-Aware Color Correction
# -----------------------
class CameraAwareColorCorrection(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # Positive gamma
        self.gamma_param = nn.Parameter(torch.tensor(2.2, dtype=torch.float32))

        # Color transform (1x1 MLP)
        self.color_transform = nn.Sequential(
            nn.Conv2d(out_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

        # Tone curve per-channel (outputs multiplicative modulation in [0.8, 1.2])
        self.tone_curve = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,3,H,W]
        gamma = F.softplus(self.gamma_param) + 1e-6
        x = torch.pow(torch.clamp(x, 0, 1), 1.0 / gamma)

        x = self.color_transform(x)

        # Apply per-channel tone curve multiplicatively (preserves structure/details)
        out_ch = []
        for i in range(x.shape[1]):
            ch = x[:, i:i+1]
            mod = self.tone_curve(ch)                   # [0,1]
            scale = 0.8 + 0.4 * mod                     # [0.8,1.2]
            out_ch.append(torch.clamp(ch * scale, 0, 1))
        x = torch.cat(out_ch, dim=1)

        return torch.clamp(x, 0, 1)

# -----------------------
# Enhanced FLCA with Color Awareness (with pyramid)
# -----------------------
class EnhancedFLCA(nn.Module):
    def __init__(self, channels, levels=2, r_ratio=8, eps=1e-8):
        """
        channels: number of feature channels
        levels: number of DWT levels to build the pyramid
        """
        super().__init__()
        self.dwt = HaarDWT()
        self.eps = eps
        self.levels = max(1, int(levels))

        # Color-aware attention
        self.color_attention = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1),  # Y, Cr, Cb, R, G
            nn.Sigmoid()
        )

        # Frequency attention
        self.low_attn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.high_attn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=True),
            nn.Tanh()
        )

        # Channel attention (SE)
        hidden = max(8, channels // r_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

        # small projector predicting residual delta
        self.res_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def _hf_mag(self, LH, HL, HH, eps):
        """Stable magnitude of high-frequency subbands."""
        return torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + eps)

    def _pyramid_y(self, y):
        """
        Build a multi-level wavelet pyramid for luminance y.
        Returns:
            lows  - list of LL bands (level 0 = original -> then downsampled LLs)
            highs - list of HF magnitude maps (corresponding to each level)
        """
        lows, highs = [], []
        cur = y
        for _ in range(self.levels):
            LL, (LH, HL, HH) = self.dwt(cur)
            lows.append(LL)
            highs.append(self._hf_mag(LH, HL, HH, self.eps))
            cur = LL
        return lows, highs

    def forward(self, feat, y, cr, cb, rgb_guide):
        B, C, Hf, Wf = feat.shape

        # Build pyramid
        lows, highs = self._pyramid_y(y)  # lists length = self.levels

        # Guidance signals
        y_low = lows[-1]
        hf_resized = [F.interpolate(h, size=(Hf, Wf), mode='bilinear', align_corners=False) for h in highs]
        y_high = torch.stack(hf_resized, dim=0).mean(dim=0) if len(hf_resized) > 1 else hf_resized[0]

        # Resize guidance to feature map size
        y_resized   = F.interpolate(y,        size=(Hf, Wf), mode='bilinear', align_corners=False)
        y_low       = F.interpolate(y_low,    size=(Hf, Wf), mode='bilinear', align_corners=False)
        cr_resized  = F.interpolate(cr,       size=(Hf, Wf), mode='bilinear', align_corners=False)
        cb_resized  = F.interpolate(cb,       size=(Hf, Wf), mode='bilinear', align_corners=False)
        rgb_resized = F.interpolate(rgb_guide, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # Combined color guidance (Y, Cr, Cb, R, G)
        color_guidance = torch.cat(
            [y_resized, cr_resized, cb_resized, rgb_resized[:, 0:1], rgb_resized[:, 1:2]],
            dim=1
        )
        color_attn = self.color_attention(color_guidance)

        # Frequency attention (bounded)
        freq_attn = torch.tanh(self.low_attn(y_low) + self.high_attn(y_high))

        # Combined attention
        spatial = 1.0 + color_attn + freq_attn
        x = feat * spatial

        # Residual projection (clamped for stability)
        raw_res = self.res_proj(x)
        res = torch.tanh(raw_res) * 0.2
        x = x + res

        # Channel attention
        ch = self.se(x)
        x = x * ch
        return x

# -----------------------
# Core blocks
# -----------------------
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, max(1, n_feat // 2), kernel_size=3, stride=1, padding=1, bias=False),
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise  = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                    dilation=1, groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer  = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.log_temperature = nn.Parameter(torch.zeros(num_heads, 1, 1))
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
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.log_temperature.exp()
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=True):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = conv_ffn(dim, dim * ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class EnhancedConv_Transformer(nn.Module):
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2, flca_levels=2):
        super().__init__()
        self.FLCA = EnhancedFLCA(in_channel, levels=flca_levels)
        self.Transformer = TransformerBlock(dim=in_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=True)
        self.channel_reduce = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)
        self.Conv_out = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, feat, y, cr, cb, rgb_guide):
        flca_feat = self.FLCA(feat, y, cr, cb, rgb_guide)
        trans = self.Transformer(feat)
        x = torch.cat([flca_feat, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x

# -----------------------
# Final TrueColorRawFormer
# -----------------------
class TrueColorRawFormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8, 8, 8, 8], ffn_expansion_factor=2, flca_levels=2):
        super().__init__()

        # Enhanced Bayer processing
        self.bayer_processor = EnhancedBayerProcessor()

        # Input processing (Bayer is packed into 4 channels first)
        self.embedding = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, padding=1)

        # Encoder with enhanced blocks
        self.conv_tran1 = EnhancedConv_Transformer(dim,     num_heads[0], ffn_expansion_factor, flca_levels)
        self.down1      = Downsample(dim)
        self.conv_tran2 = EnhancedConv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor, flca_levels)
        self.down2      = Downsample(dim * 2)
        self.conv_tran3 = EnhancedConv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor, flca_levels)
        self.down3      = Downsample(dim * 4)
        self.conv_tran4 = EnhancedConv_Transformer(dim * 8, num_heads[3], ffn_expansion_factor, flca_levels)

        # Decoder
        self.up1 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.conv_tran5 = EnhancedConv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor, flca_levels)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.conv_tran6 = EnhancedConv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor, flca_levels)

        self.up3 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim, 1)
        self.conv_tran7 = EnhancedConv_Transformer(dim, num_heads[0], ffn_expansion_factor, flca_levels)

        # Output with proper color processing
        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.color_correction = CameraAwareColorCorrection(out_channels)

    def forward(self, x):
        # x: [B,1,H,W] raw Bayer
        x_ds = downshuffle(x, 2)  # [B,4,H/2,W/2]

        # Process Bayer planes for guidance
        y, cr, cb, rgb_guide = self.bayer_processor(x_ds)  # all at H/2, W/2

        # Feature extraction
        x0 = self.embedding(x_ds)

        # Encoder
        conv_tran1 = self.conv_tran1(x0, y, cr, cb, rgb_guide)
        pool1 = self.down1(conv_tran1)
        conv_tran2 = self.conv_tran2(pool1, y, cr, cb, rgb_guide)
        pool2 = self.down2(conv_tran2)
        conv_tran3 = self.conv_tran3(pool2, y, cr, cb, rgb_guide)
        pool3 = self.down3(conv_tran3)
        conv_tran4 = self.conv_tran4(pool3, y, cr, cb, rgb_guide)

        # Decoder
        up1 = self.up1(conv_tran4)
        concat1 = torch.cat([up1, conv_tran3], 1)
        concat1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1, y, cr, cb, rgb_guide)

        up2 = self.up2(conv_tran5)
        concat2 = torch.cat([up2, conv_tran2], 1)
        concat2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2, y, cr, cb, rgb_guide)

        up3 = self.up3(conv_tran6)
        concat3 = torch.cat([up3, conv_tran1], 1)
        concat3 = self.channel_reduce3(concat3)
        conv_tran7 = self.conv_tran7(concat3, y, cr, cb, rgb_guide)

        conv_out = F.relu(self.conv_out(conv_tran7), inplace=False)  # [B, 4*out, H/2, W/2]
        out = self.pixelshuffle(conv_out)                           # [B, out, H, W]
        out = self.color_correction(out)
        return out

# -----------------------
# Specialized SID Loss Function
# -----------------------
class SIDColorLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        # Structural loss
        mse_loss = F.mse_loss(pred, target)

        # Color consistency loss (LAB)
        pred_lab   = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        lab_loss = F.l1_loss(pred_lab, target_lab)

        # Perceptual color loss
        color_loss = self.color_angular_loss(pred, target)

        return self.alpha * mse_loss + self.beta * lab_loss + self.gamma * color_loss

    def rgb_to_lab(self, rgb):
        # Expect rgb in [0,1]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        # linearize sRGB
        def to_linear(c):
            mask = (c > 0.04045).float()
            return (((c + 0.055) / 1.055) ** 2.4) * mask + (c / 12.92) * (1 - mask)

        r_l = to_linear(rgb[:, 0:1])
        g_l = to_linear(rgb[:, 1:2])
        b_l = to_linear(rgb[:, 2:3])

        x = 0.412453 * r_l + 0.357580 * g_l + 0.180423 * b_l
        y = 0.212671 * r_l + 0.715160 * g_l + 0.072169 * b_l
        z = 0.019334 * r_l + 0.119193 * g_l + 0.950227 * b_l

        # Normalize by reference white (D65)
        x = x / 0.950456
        y = y / 1.000000
        z = z / 1.088754

        def f(t):
            return torch.where(t > 0.008856, torch.pow(t, 1/3), 7.787 * t + 16/116)

        fx, fy, fz = f(x), f(y), f(z)

        L = 116 * fy - 16
        A = 500 * (fx - fy)
        B = 200 * (fy - fz)

        return torch.cat([L, A, B], dim=1)

    def color_angular_loss(self, pred, target):
        pred_norm   = F.normalize(pred, dim=1, eps=self.eps)
        target_norm = F.normalize(target, dim=1, eps=self.eps)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        angular_loss = 1 - cosine_sim.mean()
        return angular_loss

# -----------------------
# Utility functions for SID dataset
# -----------------------
def preprocess_sid_raw(raw_tensor, bit_depth=14):
    """Preprocess RAW input for SID dataset"""
    raw_tensor = raw_tensor / (2**bit_depth - 1)
    return raw_tensor

def postprocess_sid_output(output_tensor):
    """Postprocess output for SID dataset"""
    output_tensor = torch.clamp(output_tensor, 0, 1)
    return output_tensor

def visualize_color_stats(tensor, name=""):
    """Debug function to visualize color statistics"""
    print(f"{name} - Mean: {tensor.mean().item():.4f}, "
          f"Std: {tensor.std().item():.4f}, "
          f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    if tensor.shape[1] == 3:  # RGB
        r, g, b = tensor[:,0], tensor[:,1], tensor[:,2]
        print(f"  R: {r.mean().item():.4f}, G: {g.mean().item():.4f}, B: {b.mean().item():.4f}")

# -----------------------
# quick check
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = TrueColorRawFormer(dim=48, flca_levels=2)

    # Test with sample input (non-negative to mimic raw)
    sample_input = torch.rand(1, 1, 512, 512)
    output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")

    # Check color stats
    visualize_color_stats(output, "Model Output")

    # Test loss function
    target = torch.rand_like(output)
    loss_fn = SIDColorLoss()
    loss = loss_fn(output, target)
    print(f"Test loss: {loss.item():.6f}")

    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
