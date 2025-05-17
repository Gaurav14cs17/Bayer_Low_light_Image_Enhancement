import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class BasicOps:
    @staticmethod
    def to_3d(x):
        return rearrange(x, 'b c h w -> b (h w) c')

    @staticmethod
    def to_4d(x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return BasicOps.to_4d(self.norm(BasicOps.to_3d(x)), h, w)


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)  # (batch,1,features)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2) - 1
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)
        return wavelet_weighted.sum(dim=2)

    def forward(self, x):
        if x.dim() > 2:
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        else:
            x_flat = x

        wavelet_out = self.wavelet_transform(x_flat)
        linear_out = F.linear(x_flat, self.weight)

        combined = wavelet_out + linear_out
        combined = self.bn(combined)

        if x.dim() > 2:
            combined = combined.view(b, h, w, self.out_features).permute(0, 3, 1, 2)

        return combined


class KANAttention(nn.Module):
    def __init__(self, dim, heads=8, wavelet_type='mexican_hat'):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(1, heads, 1, 1))

        self.qkv = nn.Sequential(
            KANLinear(dim, dim * 3, wavelet_type),
            nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        )
        self.proj = KANLinear(dim, dim, wavelet_type)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = rearrange(attn.softmax(dim=-1) @ v, 'b head c (h w) -> b (head c) h w', h=h)
        return self.proj(out)


class KANFFN(nn.Module):
    def __init__(self, dim, expansion=4, wavelet_type='mexican_hat'):
        super().__init__()
        hidden_dim = dim * expansion

        self.net = nn.Sequential(
            KANLinear(dim, hidden_dim, wavelet_type),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            KANLinear(hidden_dim, dim, wavelet_type)
        )

    def forward(self, x):
        return self.net(x)


class KANTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, ffn_expansion=4, wavelet_type='mexican_hat'):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = KANAttention(dim, heads, wavelet_type)
        self.norm2 = LayerNorm(dim)
        self.ffn = KANFFN(dim, ffn_expansion, wavelet_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KANConvTransformer(nn.Module):
    def __init__(self, dim, heads=8, ffn_expansion=2, wavelet_type='mexican_hat'):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.transformer = KANTransformerBlock(dim, heads, ffn_expansion, wavelet_type)
        self.reduce = KANLinear(dim * 2, dim, wavelet_type)
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        conv = self.conv(x)
        trans = self.transformer(x)
        combined = torch.cat([conv, trans], 1)
        reduced = self.reduce(combined)
        return self.out(reduced)


class WavKANRawFormer(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=48, heads=[8, 16, 32, 32], ffn_exp=2, wavelet_type='mexican_hat'):
        super().__init__()

        # Encoder
        self.embed = nn.Conv2d(in_ch * 4, dim, 3, padding=1)
        self.encoder = nn.ModuleList([
            KANConvTransformer(dim * (2 ** i), heads[i], ffn_exp, wavelet_type)
            for i in range(3)
        ])
        self.downsamples = nn.ModuleList([
            Downsample(dim * (2 ** i))
            for i in range(3)
        ])
        self.bottleneck = KANConvTransformer(dim * 8, heads[3], ffn_exp, wavelet_type)

        # Decoder
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(dim * 8, dim * 4, 2, 2),
            nn.ConvTranspose2d(dim * 8, dim * 2, 2, 2),
            nn.ConvTranspose2d(dim * 4, dim, 2, 2)
        ])


        self.decoder = nn.ModuleList([
            KANConvTransformer(dim*8, dim * 4, ffn_exp, wavelet_type),
            KANConvTransformer(dim*4, dim * 2, ffn_exp, wavelet_type),
            KANConvTransformer(dim * 2, dim, ffn_exp, wavelet_type)
        ])

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(dim*2, out_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        # Encoder
        x = F.pixel_unshuffle(x, 2)
        x = self.embed(x)

        features = []
        for enc, down in zip(self.encoder, self.downsamples):
            features.append(x)
            x = enc(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder)):
            x = up(x)
            x = torch.cat([x, features[-(i + 1)]], dim=1)
            x = dec(x)


        return self.output(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WavKANRawFormer(in_ch=3, out_ch=3, wavelet_type='mexican_hat').to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(1, 3, 32, 32).to(device)
    out = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {out.shape}")
