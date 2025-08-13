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
    """Pixel unshuffle implementation"""
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
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return downshuffle(self.body(x), 2)

class LayerNorm(nn.Module):
    """2D LayerNorm for channels"""
    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ConvFFN(nn.Module):
    """Feed-forward network with depthwise conv"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                   stride=1, padding=1, groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pointwise2(x)
        return x

# ----------------------------
# Luminance processing
# ----------------------------
def rgb_to_luma(x_rgb, eps=1e-6):
    """Convert RGB to normalized luma [0,1]"""
    r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y_min = y.amin(dim=(2, 3), keepdim=True)
    y_max = y.amax(dim=(2, 3), keepdim=True)
    return (y - y_min) / (y_max - y_min + eps)

class BayerLuma(nn.Module):
    """Bayer pattern to luma converter"""
    def __init__(self, pattern='rggb'):
        super().__init__()
        self.pattern = pattern.lower()
        assert self.pattern in ['rggb', 'bggr', 'grbg', 'gbrg']

        # Create channel extraction kernels (3x3)
        self.register_buffer('r_weight', self._create_kernel('r'))
        self.register_buffer('g_weight', self._create_kernel('g'))
        self.register_buffer('b_weight', self._create_kernel('b'))
        self.register_buffer('luma_coeff', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _create_kernel(self, channel):
        kernel = torch.zeros(1, 1, 3, 3)
        # Explicit element assignments to avoid tricky advanced indexing
        if self.pattern == 'rggb':
            if channel == 'r':
                kernel[0, 0, 0, 0] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 1] = 0.5
                kernel[0, 0, 1, 0] = 0.5
            elif channel == 'b':
                kernel[0, 0, 1, 1] = 1.0
        elif self.pattern == 'bggr':
            if channel == 'b':
                kernel[0, 0, 0, 0] = 1.0
            elif channel == 'g':
                kernel[0, 0, 0, 1] = 0.5
                kernel[0, 0, 1, 0] = 0.5
            elif channel == 'r':
                kernel[0, 0, 1, 1] = 1.0
        elif self.pattern == 'grbg':
            if channel == 'g':
                kernel[0, 0, 0, 0] = 0.5
                kernel[0, 0, 1, 1] = 0.5
            elif channel == 'r':
                kernel[0, 0, 0, 1] = 1.0
            elif channel == 'b':
                kernel[0, 0, 1, 0] = 1.0
        elif self.pattern == 'gbrg':
            if channel == 'g':
                kernel[0, 0, 0, 0] = 0.5
                kernel[0, 0, 1, 1] = 0.5
            elif channel == 'b':
                kernel[0, 0, 0, 1] = 1.0
            elif channel == 'r':
                kernel[0, 0, 1, 0] = 1.0
        return kernel

    def forward(self, bayer):
        # bayer: Bx1xHxW
        r = F.conv2d(bayer, self.r_weight, padding=1)
        g = F.conv2d(bayer, self.g_weight, padding=1)
        b = F.conv2d(bayer, self.b_weight, padding=1)
        rgb = torch.cat([r, g, b], dim=1)
        luma = torch.sum(rgb * self.luma_coeff, dim=1, keepdim=True)
        luma_min = luma.amin(dim=(2, 3), keepdim=True)
        luma_max = luma.amax(dim=(2, 3), keepdim=True)
        return (luma - luma_min) / (luma_max - luma_min + 1e-6)

# ----------------------------
# Luminance-Aware Attention
# ----------------------------
class LumaCond(nn.Module):
    """Luma conditioning network"""
    def __init__(self, heads, dim_head):
        super().__init__()
        hidden = max(16, heads * dim_head // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(hidden, heads * dim_head, 1)
        self.beta = nn.Conv2d(hidden, heads * dim_head, 1)

    def forward(self, L):
        h = self.net(L)
        return self.gamma(h), self.beta(h)

class LuminanceAwareMHSA(nn.Module):
    """Multi-head self-attention with luminance conditioning (memory-efficient luma bias)"""
    def __init__(self, dim, heads=8, dim_head=None, bias=True, luma_bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head if dim_head else dim // heads
        inner = heads * self.dim_head

        self.to_qkv = nn.Conv2d(dim, inner * 3, 1, bias=bias)
        self.proj = nn.Conv2d(inner, dim, 1, bias=bias)
        self.scale = self.dim_head ** -0.5

        self.luma_cond = LumaCond(heads, self.dim_head)
        self.luma_bias = luma_bias
        if luma_bias:
            # alpha scales the per-position bias added to queries (cheap)
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, luma=None, rgb_input=None):
        B, C, H, W = x.shape
        N = H * W

        # Get luma if not provided
        if luma is None:
            if rgb_input is not None:
                luma = rgb_to_luma(rgb_input)
            else:
                raise ValueError("Either luma or rgb_input must be provided")

        # QKV projections
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        # Luma conditioning (feature-wise affine)
        gamma, beta = self.luma_cond(luma)  # gamma/beta at luma spatial size
        gamma = rearrange(gamma, 'b (h d) x y -> b h (x y) d', h=self.heads)
        beta = rearrange(beta, 'b (h d) x y -> b h (x y) d', h=self.heads)
        q = gamma * q + beta
        k = gamma * k + beta
        v = gamma * v + beta

        # Luma bias — cheap O(N) implementation:
        # compute invL per position, pool-smoothed; center it and add to queries as scalar bias.
        if self.luma_bias:
            invL = 1.0 - luma  # B x 1 x h' x w' (luma spatial must match q/k/v spatial)
            # Ensure invL has same spatial resolution as q/k/v: the caller must pass correctly pooled luma.
            invL = F.avg_pool2d(invL, 3, padding=1, stride=1)  # slight smoothing, optional
            invL = rearrange(invL, 'b 1 h w -> b (h w)')        # B x N
            invL = invL - invL.mean(dim=-1, keepdim=True)       # center
            # expand to (B, heads, N, 1) to add to q (per-position scalar)
            invL_q = invL.unsqueeze(1).unsqueeze(-1)           # B x 1 x N x 1 -> will broadcast to heads
            q = q + self.alpha * invL_q  # broadcasts to (B, heads, N, d)

        # Attention
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        return self.proj(out)

# ----------------------------
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=True, dim_head=None):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LuminanceAwareMHSA(dim, num_heads, dim_head, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = ConvFFN(dim, dim * ffn_expansion_factor, dim)

    def forward(self, x, rgb_for_luma=None, luma=None):
        x = x + self.attn(self.norm1(x), luma=luma, rgb_input=rgb_for_luma)
        x = x + self.ffn(self.norm2(x))
        return x

# ----------------------------
# Main Model
# ----------------------------
class RawFormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=3, dim=48, num_heads=[8,8,8,8],
                 ffn_expansion_factor=2, bayer_pattern='rggb'):
        super().__init__()

        self.bayer_luma = BayerLuma(bayer_pattern)
        self.embedding = nn.Conv2d(inp_channels*4, dim, 3, padding=1)

        # Encoder
        self.enc1 = TransformerBlock(dim, num_heads[0], ffn_expansion_factor)
        self.down1 = Downsample(dim)

        self.enc2 = TransformerBlock(dim*2, num_heads[1], ffn_expansion_factor)
        self.down2 = Downsample(dim*2)

        self.enc3 = TransformerBlock(dim*4, num_heads[2], ffn_expansion_factor)
        self.down3 = Downsample(dim*4)

        self.bottleneck = TransformerBlock(dim*8, num_heads[3], ffn_expansion_factor)

        # Decoder + projection layers to match channel sizes after concat
        self.up1 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.proj1 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.dec1 = TransformerBlock(dim*4, num_heads[2], ffn_expansion_factor)

        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.proj2 = nn.Conv2d(dim*6, dim*2, kernel_size=1)
        self.dec2 = TransformerBlock(dim*2, num_heads[1], ffn_expansion_factor)

        self.up3 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.proj3 = nn.Conv2d(dim*3, dim, kernel_size=1)
        self.dec3 = TransformerBlock(dim, num_heads[0], ffn_expansion_factor)

        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels*4, 3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, rgb_for_luma=None, luma=None):
        # Get full-resolution luma (HxW)
        if luma is None:
            if rgb_for_luma is not None:
                luma = rgb_to_luma(rgb_for_luma)
            else:
                luma = self.bayer_luma(x)

        # Precompute pooled lumas for each stage to match spatial sizes:
        # original raw: H x W
        # after downshuffle & embedding: H/2 x W/2  -> enc1  -> luma_p2
        # after down1: H/4 -> enc2 -> luma_p4
        # after down2: H/8 -> enc3 -> luma_p8
        # after down3: H/16 -> bottleneck -> luma_p16
        luma_p2 = F.avg_pool2d(luma, kernel_size=2)   # H/2
        luma_p4 = F.avg_pool2d(luma, kernel_size=4)   # H/4
        luma_p8 = F.avg_pool2d(luma, kernel_size=8)   # H/8
        luma_p16 = F.avg_pool2d(luma, kernel_size=16) # H/16

        # Encoder
        x = downshuffle(x, 2)            # B x (C*4) x H/2 x W/2
        x = self.embedding(x)           # B x dim x H/2 x W/2

        x1 = self.enc1(x, luma=luma_p2)                                    # B x dim x H/2 x W/2
        d1 = self.down1(x1)                                                  # B x dim*2 x H/4 x W/4

        x2 = self.enc2(d1, luma=luma_p4)                                     # B x dim*2 x H/4 x W/4
        d2 = self.down2(x2)                                                  # B x dim*4 x H/8 x W/8

        x3 = self.enc3(d2, luma=luma_p8)                                     # B x dim*4 x H/8 x W/8
        d3 = self.down3(x3)                                                  # B x dim*8 x H/16 x W/16

        x = self.bottleneck(d3, luma=luma_p16)                               # B x dim*8 x H/16 x W/16

        # Decoder
        u1 = self.up1(x)                                                      # B x dim*4 x H/8 x W/8
        cat1 = torch.cat([u1, x3], dim=1)                                     # B x dim*8 x H/8 x W/8
        p1 = self.proj1(cat1)                                                  # B x dim*4 x H/8 x W/8
        x = self.dec1(p1, luma=luma_p8)

        u2 = self.up2(x)                                                      # B x dim*2 x H/4 x W/4
        cat2 = torch.cat([u2, x2], dim=1)                                     # B x dim*6 x H/4 x W/4
        p2 = self.proj2(cat2)                                                  # B x dim*2 x H/4 x W/4
        x = self.dec2(p2, luma=luma_p4)

        u3 = self.up3(x)                                                      # B x dim x H/2 x W/2
        cat3 = torch.cat([u3, x1], dim=1)                                     # B x dim*3 x H/2 x W/2
        p3 = self.proj3(cat3)                                                  # B x dim x H/2 x W/2
        x = self.dec3(p3, luma=luma_p2)

        return self.output(x)


# Test
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RawFormer().to(device)
    x = torch.randn(1, 1, 128, 128).to(device)
    out = model(x)
    print(out.shape)  # Should be (1, 3, 128, 128)
