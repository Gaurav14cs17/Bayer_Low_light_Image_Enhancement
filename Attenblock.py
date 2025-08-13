import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Utility re-arrange helpers
# ----------------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):
    """
    Down Shuffle function, same as nn.PixelUnshuffle().
    Input: variable of size (B, C, H, W)
    Output: down-shuffled var of size (B, C*r^2, H/r, W/r)
    """
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()

# ----------------------------
# Small modules
# ----------------------------
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return downshuffle(self.body(x), 2)

class LayerNorm(nn.Module):
    """
    Apply LayerNorm over channels for 2D feature maps.
    """
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ConvFFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   dilation=1, groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.pointwise2(x)
        return x

# ----------------------------
# RGB -> Luma helper (normalized)
# ----------------------------
def rgb_to_luma(x_rgb, eps=1e-6):
    """
    x_rgb: Bx3xHxW (linear RGB or approximate RGB)
    returns normalized luma in [0,1] per-image
    """
    r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y_min = y.amin(dim=(2, 3), keepdim=True)
    y_max = y.amax(dim=(2, 3), keepdim=True)
    y = (y - y_min) / (y_max - y_min + eps)
    return y

# ----------------------------
# Bayer -> Luma estimator (supports common CFA patterns)
# ----------------------------
class BayerLuma(nn.Module):
    def __init__(self, pattern='rggb'):
        super().__init__()
        self.pattern = pattern.lower()
        assert self.pattern in ['rggb', 'bggr', 'grbg', 'gbrg'], "Invalid Bayer pattern"

        # Create fixed convolution kernels for each channel (3x3 kernels)
        r_k = self._create_kernel('r')
        g_k = self._create_kernel('g')
        b_k = self._create_kernel('b')

        self.register_buffer('r_weight', r_k)
        self.register_buffer('g_weight', g_k)
        self.register_buffer('b_weight', b_k)

        # Luma coefficients
        self.register_buffer('luma_coeff', torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1))

    def _create_kernel(self, channel):
        """
        Return a 1x1 conv kernel of shape (1,1,3,3) that extracts channel-specific samples
        according to the Bayer pattern. For green we place 0.5 on both green positions.
        """
        kernel = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        # Map 2x2 Bayer sampling to 3x3 kernel center positions (center is (1,1)).
        # We place the 2x2 pattern at center top-left (positions (0,0),(0,1),(1,0),(1,1))
        if self.pattern == 'rggb':
            # top-left R, top-right G, bottom-left G, bottom-right B
            if channel == 'r':
                kernel[0, 0, 0, 0] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 1] = 0.5
                kernel[0, 0, 1, 0] = 0.5
            elif channel == 'b':
                kernel[0, 0, 1, 1] = 1.0
        elif self.pattern == 'bggr':
            # top-left B, top-right G, bottom-left G, bottom-right R
            if channel == 'r':
                kernel[0, 0, 1, 1] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 1] = 0.5
                kernel[0, 0, 1, 0] = 0.5
            elif channel == 'b':
                kernel[0, 0, 0, 0] = 1.0
        elif self.pattern == 'grbg':
            # top-left G, top-right R, bottom-left B, bottom-right G
            if channel == 'r':
                kernel[0, 0, 0, 1] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 0] = 0.5
                kernel[0, 0, 1, 1] = 0.5
            elif channel == 'b':
                kernel[0, 0, 1, 0] = 1.0
        elif self.pattern == 'gbrg':
            # top-left G, top-right B, bottom-left R, bottom-right G
            if channel == 'r':
                kernel[0, 0, 1, 0] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 0] = 0.5
                kernel[0, 0, 1, 1] = 0.5
            elif channel == 'b':
                kernel[0, 0, 0, 1] = 1.0
        return kernel

    def forward(self, bayer):
        """
        Input: Bayer pattern image (B, 1, H, W)
        Output: Luma estimate (B, 1, H, W) normalized to [0,1] per-image
        """
        # Separate channels with 3x3 convolution (padding=1 to avoid size change)
        r = F.conv2d(bayer, self.r_weight, padding=1)
        g = F.conv2d(bayer, self.g_weight, padding=1)
        b = F.conv2d(bayer, self.b_weight, padding=1)

        # Stack channels and compute luma
        rgb = torch.cat([r, g, b], dim=1)  # Bx3xHxW
        luma = torch.sum(rgb * self.luma_coeff, dim=1, keepdim=True)  # Bx1xHxW

        # Normalize per-image
        luma_min = luma.amin(dim=(2, 3), keepdim=True)
        luma_max = luma.amax(dim=(2, 3), keepdim=True)
        luma_norm = (luma - luma_min) / (luma_max - luma_min + 1e-6)
        return luma_norm

# ----------------------------
# Luminance-Aware MHSA
# ----------------------------
class LumaCond(nn.Module):
    def __init__(self, heads, dim_head, per_head=True):
        super().__init__()
        self.per_head = per_head
        out_ch = heads * dim_head if per_head else dim_head
        hidden = max(16, out_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gamma_q = nn.Conv2d(hidden, out_ch, 1)
        self.beta_q = nn.Conv2d(hidden, out_ch, 1)
        self.gamma_k = nn.Conv2d(hidden, out_ch, 1)
        self.beta_k = nn.Conv2d(hidden, out_ch, 1)
        self.gamma_v = nn.Conv2d(hidden, out_ch, 1)
        self.beta_v = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, L):
        h = self.net(L)
        gq, bq = self.gamma_q(h), self.beta_q(h)
        gk, bk = self.gamma_k(h), self.beta_k(h)
        gv, bv = self.gamma_v(h), self.beta_v(h)
        gq, bq = gq.mean(dim=(2, 3), keepdim=True), bq.mean(dim=(2, 3), keepdim=True)
        gk, bk = gk.mean(dim=(2, 3), keepdim=True), bk.mean(dim=(2, 3), keepdim=True)
        gv, bv = gv.mean(dim=(2, 3), keepdim=True), bv.mean(dim=(2, 3), keepdim=True)
        return (gq, bq, gk, bk, gv, bv)

class LuminanceAwareMHSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, bias=True, luma_bias=True):
        super().__init__()
        self.heads = heads
        if dim_head is None:
            assert dim % heads == 0, "dim must be divisible by heads or provide dim_head"
            dim_head = dim // heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Conv2d(dim, inner, 1, bias=bias)
        self.to_k = nn.Conv2d(dim, inner, 1, bias=bias)
        self.to_v = nn.Conv2d(dim, inner, 1, bias=bias)
        self.proj = nn.Conv2d(inner, dim, 1, bias=bias)
        self.scale = dim_head ** -0.5
        self.luma_cond = LumaCond(heads, dim_head)
        self.luma_bias = luma_bias
        if luma_bias:
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, luma=None, rgb_input=None):
        B, C, H, W = x.shape

        if luma is None:
            assert rgb_input is not None and rgb_input.shape[1] == 3, \
                "Provide luma or rgb_input for luminance."
            luma = rgb_to_luma(rgb_input)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        gq, bq, gk, bk, gv, bv = self.luma_cond(luma)

        def reshape_film(t):
            return t.view(B, self.heads, self.dim_head, 1, 1)

        def apply_film(t, g, b):
            return g * t.view(B, self.heads, self.dim_head, H, W) + b

        q = apply_film(q, *map(reshape_film, (gq, bq)))
        k = apply_film(k, *map(reshape_film, (gk, bk)))
        v = apply_film(v, *map(reshape_film, (gv, bv)))

        q = q.flatten(3).transpose(2, 3)  # B, heads, N, dim_head
        k = k.flatten(3).transpose(2, 3)  # B, heads, N, dim_head
        v = v.flatten(3).transpose(2, 3)  # B, heads, N, dim_head

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.luma_bias:
            invL = 1.0 - luma
            invL = F.avg_pool2d(invL, kernel_size=3, stride=1, padding=1)
            invL = invL.view(B, 1, -1)  # B, 1, N
            invL = invL - invL.mean(dim=-1, keepdim=True)
            
            # Reshape bias to match attention logits shape
            bias = invL.unsqueeze(1)  # B, 1, 1, N
            bias = bias.repeat(1, self.heads, 1, 1)  # B, heads, 1, N
            
            attn_logits = attn_logits + self.alpha * bias

        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous()
        out = out.view(B, self.heads * self.dim_head, H, W)
        return self.proj(out)

# ----------------------------
# Transformer block using LuminanceAwareMHSA
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, dim_head=None):
        super(TransformerBlock, self).__init__()
        if dim_head is None:
            assert dim % num_heads == 0, "dim must be divisible by num_heads or provide dim_head"
            dim_head = dim // num_heads
        self.norm1 = LayerNorm(dim)
        self.attn = LuminanceAwareMHSA(dim=dim, heads=num_heads, dim_head=dim_head, bias=bias, luma_bias=True)
        self.norm2 = LayerNorm(dim)
        self.ffn = ConvFFN(dim, dim * ffn_expansion_factor, dim)

    def forward(self, x, rgb_for_luma=None, luma=None):
        x = x + self.attn(self.norm1(x), luma=luma, rgb_input=rgb_for_luma)
        x = x + self.ffn(self.norm2(x))
        return x

# ----------------------------
# Conv + Transformer wrapper
# ----------------------------
class ConvTransformer(nn.Module):
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2, dim_head=None):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.transformer = TransformerBlock(dim=in_channel, num_heads=num_heads,
                                         ffn_expansion_factor=ffn_expansion_factor, bias=True, dim_head=dim_head)
        self.channel_reduce = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=1, stride=1)
        self.conv_out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rgb_for_luma=None, luma=None):
        conv = self.lrelu(self.conv(x))
        trans = self.transformer(x, rgb_for_luma=rgb_for_luma, luma=luma)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.conv_out(x))
        return x

# ----------------------------
# RawFormer Model (integrated Bayer Luma)
# ----------------------------
class RawFormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8,8,8,8],
                 ffn_expansion_factor=2, bayer_pattern='rggb'):
        super(RawFormer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        # Bayer-aware luma estimator
        self.bayer_luma = BayerLuma(bayer_pattern)

        self.embedding = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.conv_tran1 = ConvTransformer(dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor)
        self.down1 = Downsample(dim)

        self.conv_tran2 = ConvTransformer(dim * 2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor)
        self.down2 = Downsample(dim * 2)

        self.conv_tran3 = ConvTransformer(dim * 4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor)
        self.down3 = Downsample(dim * 4)

        self.conv_tran4 = ConvTransformer(dim * 8, num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor)

        # Decoder
        self.up1 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1, 1)
        self.conv_tran5 = ConvTransformer(dim * 4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.conv_tran6 = ConvTransformer(dim * 2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor)

        self.up3 = nn.ConvTranspose2d(dim * 2, dim * 1, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim * 1, 1, 1)
        self.conv_tran7 = ConvTransformer(dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor)

        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x, rgb_for_luma=None, luma=None):
        """
        x: RAW Bayer (B,1,H,W)
        rgb_for_luma: optional RGB image (B,3,H,W) to compute luma (if provided)
        luma: optional precomputed luma (B,1,H,W)
        """
        # If no luma provided, compute from rgb_for_luma if present else from bayer
        if luma is None:
            if rgb_for_luma is not None:
                luma = rgb_to_luma(rgb_for_luma)
            else:
                luma = self.bayer_luma(x)

        x = downshuffle(x, 2)
        x = self.embedding(x)

        conv_tran1 = self.conv_tran1(x, rgb_for_luma=rgb_for_luma, luma=luma)
        pool1 = self.down1(conv_tran1)

        conv_tran2 = self.conv_tran2(pool1, rgb_for_luma=rgb_for_luma, luma=luma)
        pool2 = self.down2(conv_tran2)

        conv_tran3 = self.conv_tran3(pool2, rgb_for_luma=rgb_for_luma, luma=luma)
        pool3 = self.down3(conv_tran3)

        conv_tran4 = self.conv_tran4(pool3, rgb_for_luma=rgb_for_luma, luma=luma)

        up1 = self.up1(conv_tran4)
        concat1 = torch.cat([up1, conv_tran3], 1)
        concat1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1, rgb_for_luma=rgb_for_luma, luma=luma)

        up2 = self.up2(conv_tran5)
        concat2 = torch.cat([up2, conv_tran2], 1)
        concat2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2, rgb_for_luma=rgb_for_luma, luma=luma)

        up3 = self.up3(conv_tran6)
        concat3 = torch.cat([up3, conv_tran1], 1)
        concat3 = self.channel_reduce3(concat3)
        conv_tran7 = self.conv_tran7(concat3, rgb_for_luma=rgb_for_luma, luma=luma)

        conv_out = self.lrelu(self.conv_out(conv_tran7))
        out = self.pixelshuffle(conv_out)
        return out

# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawFormer(inp_channels=1, out_channels=3, dim=48,
                     num_heads=[8,8,8,8], ffn_expansion_factor=2,
                     bayer_pattern='rggb').to(device)

    # Test with synthetic Bayer RAW input (RGGB)
    B, H, W = 1, 128, 128
    bayer = torch.zeros(B, 1, H, W, device=device)
    # synthetic values at Bayer positions
    bayer[:, :, 0::2, 0::2] = 0.8    # R
    bayer[:, :, 0::2, 1::2] = 0.6    # G
    bayer[:, :, 1::2, 0::2] = 0.6    # G
    bayer[:, :, 1::2, 1::2] = 0.3    # B

    # Forward RAW-only
    with torch.no_grad():
        out_raw = model(bayer)
    print("RAW Input shape:", bayer.shape)
    print("RAW->RGB Output shape:", out_raw.shape)
    print("RAW Output range:", float(out_raw.min()), float(out_raw.max()))

    # Test with synthetic RGB input for luma path (optional)
    rgb = torch.rand(B, 3, H, W, device=device)
    luma_from_rgb = rgb_to_luma(rgb)
    print("RGB luma shape:", luma_from_rgb.shape, "min/max:", float(luma_from_rgb.min()), float(luma_from_rgb.max()))

    # Forward with explicit rgb_for_luma (using the same bayer input for data; only luma uses rgb)
    with torch.no_grad():
        out_rgb_luma = model(bayer, rgb_for_luma=rgb)
    print("RAW + rgb_for_luma -> Output shape:", out_rgb_luma.shape)
