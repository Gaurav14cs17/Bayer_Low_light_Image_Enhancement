import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange
import blocks  
import numbers
from mamba_ssm import Mamba

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.rep_conv1 = Conv2d_BN(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.rep_conv2 = Conv2d_BN(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x2 = self.dwconv(x)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity

    @torch.no_grad()
    def fuse(self):
        conv = self.rep_conv1.fuse()  ##Conv_BN
        conv1 = self.rep_conv2.fuse()  ##Conv_BN

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False) 
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
		

class WM(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=9,  # Block expansion factor
        )
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c))
       
        y = self.model1(x).permute(0, 2, 1) 
        output = y.reshape(b, c, h, w)
        return self.smooth(output)
	
class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class WMB(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(WMB, self).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.illu = Illumination_Estimator(dim, n_fea_in=dim+1, n_fea_out=dim)
        self.ffab = blocks.FFAB(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.mb = WM(dim)

    def forward(self, input_):
        global m
        x = input_
        n, c, h, w = x.shape
        #print(f"\n[WMB] Input stats: mean={input_.mean():.4f}, std={input_.std():.4f}, min={input_.min():.4f}, max={input_.max():.4f}")
		
        x = self.norm1(input_)
        #print(f"[Norm1] Output stats: mean={x.mean():.4f}, std={x.std():.4f}")

        x = data_transform(x)
        input_dwt = self.DWT(x)
        
        # Log DWT outputs
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        #print(f"[DWT] LL stats: mean={input_LL.mean():.4f}, max={input_LL.max():.4f}")
        #print(f"[DWT] High stats: mean={input_high.mean():.4f}, max={input_high.max():.4f}")

        input_LL, _ = self.illu(input_LL)
        input_LL = self.ffab(input_LL)
        #print(f"[FFAB] Output stats: mean={input_LL.mean():.4f}, has_nan={torch.isnan(input_LL).any()}")

        input_high = self.mb(input_high)
        #print(f"[WM] Output stats: mean={input_high.mean():.4f}, has_nan={torch.isnan(input_high).any()}")

        output = self.IWT(torch.cat((input_LL, input_high), dim=0))
        output = inverse_data_transform(output)
        #print(f"[IWT] Output stats: mean={output.mean():.4f}, has_nan={torch.isnan(output).any()}")

        x = x + output
        x = x + self.ffn(self.norm2(x))
        return x

'''		
class WMB(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(WMB, self).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.illu = Illumination_Estimator(dim, n_fea_in=dim+1, n_fea_out=dim)
        self.ffab = blocks.FFAB(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.mb = WM(dim)

    def forward(self, input_):
        global m
        x = input_
        n, c, h, w = x.shape
        x = self.norm1(x)
        x = data_transform(x)
        input_dwt = self.DWT(x)
        # input_LL=A [B,C,H/2,W/2]   input_high0={V,H,D} [3B,C,H/2,W/2]
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        input_LL, input_image = self.illu(input_LL)
        input_LL = self.ffab(input_LL)
        input_high = self.mb(input_high)

        output = self.IWT(torch.cat((input_LL, input_high), dim=0))
        output = inverse_data_transform(output)

        x = x + output
        x = x + self.ffn(self.norm2(x))
        return x
'''

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def downshuffle(var, r):
    """
    Down Shuffle function, same as nn.PixelUnshuffle().
    Input: variable of size (1 × H × W)
    Output: down-shuffled var of size (r^2 × H/r × W/r)
    """
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),)
                                  #nn.PixelUnshuffle(2))

    def forward(self, x):
        return downshuffle(self.body(x),2)

'''
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
'''

class conv_ffn(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
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

        # transposed self-attention with attention map of shape (C×C)
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
    H, W could be different
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = conv_ffn(dim, dim*ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

'''
class Conv_Transformer(nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """
    def __init__(self, in_channel,num_heads=8,ffn_expansion_factor=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.Transformer = TransformerBlock(dim=in_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=True)
        self.channel_reduce = nn.Conv2d(in_channels=in_channel*2,out_channels=in_channel,kernel_size=1,stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
    def forward(self, x):
        conv = self.lrelu(self.conv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x
'''
		
class Conv_Transformer(nn.Module):
    """
    With WMB block
    """	
    def __init__(self, in_channel, num_heads=1, ffn_expansion_factor=2.66):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.Transformer = WMB(dim=in_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=True, LayerNorm_type='WithBias')
        self.channel_reduce = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=1, stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv = self.lrelu(self.conv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x

		

class RawFormer(nn.Module):
    """
    RawFormer model.
    Args:
        inp_channels (int): Input image channel number, 1 for RAW images, 3 for RGB images.
        out_channels (int): Output image channel number, 3 for RGB images.
        dim (int): Embedding layer (Conv 3×3) dimension, 32/48/64 for small/base/large.
        num_heads (list): Transformer blocks, [8,8,8,8] by default.
        ffn_expansion_factor (int): Feed-forward network expansion factor, 2 by default.
    """

    def __init__(self,inp_channels=1,out_channels=3,dim=48,num_heads=[8,8,8,8],ffn_expansion_factor=2):
        super(RawFormer, self).__init__()
        # self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.embedding = nn.Conv2d(inp_channels*4,dim,kernel_size=3, stride=1, padding=1)
        self.conv_tran1 = Conv_Transformer(dim,num_heads[0],ffn_expansion_factor)
        self.down1 = Downsample(dim)
        self.conv_tran2 = Conv_Transformer(dim*2,num_heads[1],ffn_expansion_factor)
        self.down2 = Downsample(dim*2)
        self.conv_tran3 = Conv_Transformer(dim*4,num_heads[2],ffn_expansion_factor)
        self.down3 = Downsample(dim*4)
        self.conv_tran4 = Conv_Transformer(dim*8,num_heads[3],ffn_expansion_factor)

        self.up1 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim*8,dim*4,1,1)
        self.conv_tran5 = Conv_Transformer(dim*4,num_heads[2],ffn_expansion_factor)
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.conv_tran6 = Conv_Transformer(dim*2,num_heads[1],ffn_expansion_factor)
        self.up3 = nn.ConvTranspose2d(dim*2, dim*1, 2, stride=2)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim * 1, 1, 1)
        self.conv_tran7 = Conv_Transformer(dim,num_heads[0],ffn_expansion_factor)
        self.conv_out = nn.Conv2d(dim, out_channels*4, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # x = self.pixelunshuffle(x)
        x = torch.clamp(x, 0, 1)  # Ensure input is valid
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

        conv_out = self.lrelu(self.conv_out(conv_tran7))

        out = self.pixelshuffle(conv_out)
        return torch.clamp(out, 0, 1)  # Force valid output range

def register_grad_hook(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"Gradient: {name} - max={grad.abs().max():.4f}, mean={grad.mean():.4f}, nan={torch.isnan(grad).any()}")
            )

if __name__ == "__main__":
    model = RawFormer(dim=48)
    #register_grad_hook(model)
    ops, params = get_model_complexity_info(model, (1,512,512), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(ops, params)
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
