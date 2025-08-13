import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange

# -------------------- BayerLuma Module --------------------
class BayerLuma(nn.Module):
    """
    Compute luminance from 4-plane RGGB Bayer input:
    Channel 0: R
    Channel 1: G1
    Channel 2: G2
    Channel 3: B
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # ITU-R BT.601 luma weights
        self.register_buffer("r_w", torch.tensor(0.299, dtype=torch.float32))
        self.register_buffer("g_w", torch.tensor(0.587, dtype=torch.float32))
        self.register_buffer("b_w", torch.tensor(0.114, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[1] == 4, f"Expected 4-channel RGGB input, got {x.shape[1]}"
        r = x[:, 0:1]
        g1 = x[:, 1:2]
        g2 = x[:, 2:3]
        b = x[:, 3:4]
        g_avg = 0.5 * (g1 + g2)
        luma = self.r_w * r + self.g_w * g_avg + self.b_w * b
        luma = luma / (luma.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + self.eps)
        return luma

# -------------------- Utility Functions --------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

# -------------------- Model Components --------------------
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False)
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, 1)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, 1)
        self.act_layer = act_layer()
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
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
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
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class TransformerBlock(nn.Module):
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
    def __init__(self, in_channel, num_heads=8, ffn_expansion_factor=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.Transformer = TransformerBlock(in_channel, num_heads, ffn_expansion_factor, bias=True)
        self.channel_reduce = nn.Conv2d(in_channel * 2, in_channel, 1, 1)
        self.Conv_out = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
    def forward(self, x):
        conv = self.lrelu(self.conv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        return self.lrelu(self.Conv_out(x))

# -------------------- RawFormer with BayerLuma --------------------
class RawFormer(nn.Module):
    def __init__(self, inp_channels=4, out_channels=3, dim=48, num_heads=[8,8,8,8], ffn_expansion_factor=2):
        super(RawFormer, self).__init__()
        self.luma_layer = BayerLuma()
        self.embedding = nn.Conv2d(inp_channels + 1, dim, 3, 1, 1)  # extra channel for luma
        self.conv_tran1 = Conv_Transformer(dim, num_heads[0], ffn_expansion_factor)
        self.down1 = Downsample(dim)
        self.conv_tran2 = Conv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)
        self.down2 = Downsample(dim * 2)
        self.conv_tran3 = Conv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)
        self.down3 = Downsample(dim * 4)
        self.conv_tran4 = Conv_Transformer(dim * 8, num_heads[3], ffn_expansion_factor)
        self.up1 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1, 1)
        self.conv_tran5 = Conv_Transformer(dim * 4, num_heads[2], ffn_expansion_factor)
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.conv_tran6 = Conv_Transformer(dim * 2, num_heads[1], ffn_expansion_factor)
        self.up3 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim, 1, 1)
        self.conv_tran7 = Conv_Transformer(dim, num_heads[0], ffn_expansion_factor)
        self.conv_out = nn.Conv2d(dim, out_channels * 4, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        luma = self.luma_layer(x)
        x = torch.cat([x, luma], dim=1)
        x = downshuffle(x, 2)
        x = self.embedding(x)
        conv_tran1 = self.conv_tran1(x)
        pool1 = self.down1(conv_tran1)
        conv_tran2 = self.conv_tran2(pool1)
        pool2 = self.down2(conv_tran2)
        conv_tran3 = self.conv_tran3(pool2)
        pool3 = self.down3(conv_tran3)
        conv_tran4 = self.conv_tran4(pool3)
        up1 = self.up1(conv_tran4)
        concat1 = torch.cat([up1, conv_tran3], 1)
        concat1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1)
        up2 = self.up2(conv_tran5)
        concat2 = torch.cat([up2, conv_tran2], 1)
        concat2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2)
        up3 = self.up3(conv_tran6)
        concat3 = torch.cat([up3, conv_tran1], 1)
        concat3 = self.channel_reduce3(concat3)
        conv_tran7 = self.conv_tran7(concat3)
        conv_out = F.leaky_relu(self.conv_out(conv_tran7), 0.2)
        out = self.pixelshuffle(conv_out)
        return out

if __name__ == "__main__":
    model = RawFormer(dim=48)
    ops, params = get_model_complexity_info(model, (4, 512, 512), as_strings=True, print_per_layer_stat=False)
    print(ops, params)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
