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
    Input: (B × C × H × W)
    Output: (B × (C*r^2) × H/r × W/r)
    """
    b, c, h, w = var.size()
    assert h % r == 0 and w % r == 0, "downshuffle: H and W must be divisible by r"
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
        h = torch.tensor([1.0,  1.0], dtype=torch.float32) / math.sqrt(2.0)
        g = torch.tensor([1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
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
        pad_h = H & 1
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
# Enhanced Bayer Processing for True Color
# -----------------------
class EnhancedBayerProcessor(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # Learnable white balance gains (camera-specific)
        self.wb_gains = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))  # R, G1, G2, B

        # Learnable color matrix (3x4 for RGB conversion: 3x3 matrix + 3x1 offset in last column)
        self.color_matrix = nn.Parameter(torch.eye(3, 4, dtype=torch.float32))

        # Demosaicing refinement
        self.demosaic_refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.Softplus()  # keeps positives
        )

        # Chroma extraction with proper color difference
        self.chroma_extractor = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),  # Outputs Cr, Cb
            nn.Tanh()
        )

        # FIX: register luminance weights as a buffer to avoid device/dtype pitfalls
        self.register_buffer("y_weights", torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32))  # BT.709

    def forward(self, x):
        # x: [B,4,H,W] (R, G1, G2, B)

        # Apply white balance
        wb_x = x * self.wb_gains.view(1, 4, 1, 1)

        # Refine demosaicing
        refined = self.demosaic_refine(wb_x)

        # Extract proper RGB components
        r = refined[:, 0:1]
        g = 0.5 * (refined[:, 1:2] + refined[:, 2:3])
        b = refined[:, 3:4]

        # Form RGB image
        rgb = torch.cat([r, g, b], dim=1)  # [B,3,H,W]

        # Apply color matrix (3x4): first 3 columns are 3x3, last column is bias
        M = self.color_matrix[:, :3]                          # [3,3]
        rgb_linear = torch.einsum('ij,bjhw->bihw', M, rgb)    # [B,3,H,W]
        if self.color_matrix.size(1) == 4:
            bias = self.color_matrix[:, 3].view(1, 3, 1, 1)   # [1,3,1,1]
            rgb_linear = rgb_linear + bias

        # Luminance (BT.709), normalized per-image for stability
        y = torch.sum(rgb_linear * self.y_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        y = y / (y.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps))

        # Proper chroma components (learned deltas)
        chroma_input = torch.cat([r, g, b, y], dim=1)         # [B,4,H,W]
        chroma = self.chroma_extractor(chroma_input)
        cr, cb = torch.chunk(chroma, 2, dim=1)

        return y, cr, cb, rgb_linear


# -----------------------
# Camera-Aware Color Correction
# -----------------------
class CameraAwareColorCorrection(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # Learnable gamma correction
        self.gamma = nn.Parameter(torch.tensor(2.2, dtype=torch.float32))

        # Color transformation matrix (small 1x1 conv MLP)
        self.color_transform = nn.Sequential(
            nn.Conv2d(out_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

        # Tone curve adjustment (per-channel, applied independently)
        self.tone_curve = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply gamma correction (safe range)
        x_corrected = torch.pow(torch.clamp(x, 0, 1), 1.0 / self.gamma)

        # Apply color transformation
        x_transformed = self.color_transform(x_corrected)

        # Apply tone curve per channel
        tone_adjusted = []
        for i in range(x_transformed.shape[1]):
            channel = x_transformed[:, i:i+1]
            adjusted = self.tone_curve(channel)
            tone_adjusted.append(adjusted)

        x_final = torch.cat(tone_adjusted, dim=1)
        return torch.clamp(x_final, 0, 1)


# -----------------------
# Enhanced FLCA with Color Awareness
# -----------------------
class EnhancedFLCA(nn.Module):
    def __init__(self, channels, r_ratio=8, eps=1e-8):
        super().__init__()
        self.dwt = HaarDWT()
        self.eps = eps

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

    def forward(self, feat, y, cr, cb, rgb_guide):
        B, C, Hf, Wf = feat.shape

        # Wavelet subbands on luminance
        LL, (LH, HL, HH) = self.dwt(y)
        y_high_mag = torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + self.eps)
        y_low = LL

        # Resize guidance to feature map size
        # FIX: also resize 'y' itself to match spatial size
        y_resized   = F.interpolate(y,        size=(Hf, Wf), mode='bilinear', align_corners=False)
        y_low       = F.interpolate(y_low,    size=(Hf, Wf), mode='bilinear', align_corners=False)
        y_high      = F.interpolate(y_high_mag, size=(Hf, Wf), mode='bilinear', align_corners=False)
        cr_resized  = F.interpolate(cr,       size=(Hf, Wf), mode='bilinear', align_corners=False)
        cb_resized  = F.interpolate(cb,       size=(Hf, Wf), mode='bilinear', align_corners=False)
        rgb_resized = F.interpolate(rgb_guide, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # Combined color guidance (Y, Cr, Cb, R, G)
        color_guidance = torch.cat(
            [y_resized, cr_resized, cb_resized, rgb_resized[:, 0:1], rgb_resized[:, 1:2]],
            dim=1
        )
        color_attn = self.color_attention(color_guidance)

        # Frequency attention
        freq_attn = self.low_attn(y_low) + self.high_attn(y_high)

        # Combined attention
        spatial = 1 + color_attn + freq_attn
        x = feat * spatial

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
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    from Restormer
    input: (B,C,H,W) -> output: (B,C,H,W)
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


# -----------------------
# Enhanced Conv_Transformer
# -----------------------
class EnhancedConv_Transformer(nn.Module):
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2):
        super().__init__()
        self.FLCA = EnhancedFLCA(in_channel)
        self.Transformer = TransformerBlock(
            dim=in_channel, num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor, bias=True
        )
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
# Final RawFormer with True Color Correction
# -----------------------
class TrueColorRawFormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8, 8, 8, 8], ffn_expansion_factor=2):
        super().__init__()

        # Enhanced Bayer processing
        self.bayer_processor = EnhancedBayerProcessor()

        # Input processing (Bayer is packed into 4 channels first)
        self.embedding = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, padding=1)

        # Encoder with enhanced blocks
        self.conv_tran1 = EnhancedConv_Transformer(dim,     num_heads[0], ffn_expansion_factor)
        self.down1      = Downsample(dim)
        self.conv_tran2 = EnhancedConv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)
        self.down2      = Downsample(dim * 2)
        self.conv_tran3 = EnhancedConv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)
        self.down3      = Downsample(dim * 4)
        self.conv_tran4 = EnhancedConv_Transformer(dim * 8, num_heads[3], ffn_expansion_factor)

        # Decoder
        self.up1 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.conv_tran5 = EnhancedConv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.conv_tran6 = EnhancedConv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)

        self.up3 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim, 1)
        self.conv_tran7 = EnhancedConv_Transformer(dim, num_heads[0], ffn_expansion_factor)

        # Output with proper color processing
        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.color_correction = CameraAwareColorCorrection(out_channels)

    def forward(self, x):
        # Pack Bayer to 4 channels at half-res
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

        # Final output with proper color processing
        conv_out = F.relu(self.conv_out(conv_tran7), inplace=True)  # [B, 4*out, H/2, W/2]
        out = self.pixelshuffle(conv_out)                           # [B, out, H, W]
        out = self.color_correction(out)
        return out


# -----------------------
# Specialized SID Loss Function
# -----------------------
class SIDColorLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        # Structural loss
        mse_loss = F.mse_loss(pred, target)

        # Color consistency loss
        pred_lab   = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        lab_loss = F.l1_loss(pred_lab, target_lab)

        # Perceptual color loss
        color_loss = self.color_angular_loss(pred, target)

        return self.alpha * mse_loss + self.beta * lab_loss + self.gamma * color_loss

    def rgb_to_lab(self, rgb):
        # Expect rgb in [0,1]
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # RGB to XYZ (sRGB / D65)
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # Normalize by reference white
        x = x / 0.950456
        z = z / 1.088754

        def f(t):
            return torch.where(t > 0.008856, torch.pow(t, 1/3), 7.787 * t + 16/116)

        fx, fy, fz = f(x), f(y), f(z)

        L = 116 * fy - 16
        A = 500 * (fx - fy)
        B = 200 * (fy - fz)

        return torch.cat([L, A, B], dim=1)

    def color_angular_loss(self, pred, target):
        # Angular similarity loss for color
        pred_norm   = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
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
    model = TrueColorRawFormer(dim=48)

    # Test with sample input
    sample_input = torch.randn(1, 1, 512, 512)
    output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")

    # FLOPs/params on a RAW Bayer input (B,1,512,512)
    ops, params = get_model_complexity_info(model, (1, 512, 512),
                                            as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {ops}, Params: {params}")
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    # Check color stats
    visualize_color_stats(output, "Model Output")

    # Test loss function
    target = torch.rand_like(output)
    loss_fn = SIDColorLoss()
    loss = loss_fn(output, target)
    print(f"Test loss: {loss.item():.6f}")
