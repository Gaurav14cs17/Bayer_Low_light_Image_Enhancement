import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_luma(x_rgb, eps=1e-6):
    # x_rgb: Bx3xHxW in linear RGB (if possible)
    r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # normalize to [0,1] per-image for stability
    y_min = y.amin(dim=(2,3), keepdim=True)
    y_max = y.amax(dim=(2,3), keepdim=True)
    y = (y - y_min) / (y_max - y_min + eps)
    return y  # Bx1xHxW

class LumaCond(nn.Module):
    """
    Small CNN that maps luminance (Bx1xHxW) to FiLM scales/biases
    for Q, K, V across heads (per-channel or per-head).
    """
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
        self.beta_q  = nn.Conv2d(hidden, out_ch, 1)
        self.gamma_k = nn.Conv2d(hidden, out_ch, 1)
        self.beta_k  = nn.Conv2d(hidden, out_ch, 1)
        self.gamma_v = nn.Conv2d(hidden, out_ch, 1)
        self.beta_v  = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, L, H, W):
        h = self.net(L)  # BxhiddenxHxW
        gq, bq = self.gamma_q(h), self.beta_q(h)
        gk, bk = self.gamma_k(h), self.beta_k(h)
        gv, bv = self.gamma_v(h), self.beta_v(h)
        # spatially average to get stable per-(head*ch) FiLM params
        gq, bq = gq.mean(dim=(2,3), keepdim=True), bq.mean(dim=(2,3), keepdim=True)
        gk, bk = gk.mean(dim=(2,3), keepdim=True), bk.mean(dim=(2,3), keepdim=True)
        gv, bv = gv.mean(dim=(2,3), keepdim=True), bv.mean(dim=(2,3), keepdim=True)
        return (gq, bq, gk, bk, gv, bv)

class LuminanceAwareMHSA(nn.Module):
    """
    Multi-Head Self-Attention with:
      - FiLM conditioning of Q/K/V from luminance
      - Luminance-biased attention logits (optional)
    """
    def __init__(self, dim, heads=8, dim_head=64, bias=True, luma_bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Conv2d(dim, inner, 1, bias=bias)
        self.to_k = nn.Conv2d(dim, inner, 1, bias=bias)
        self.to_v = nn.Conv2d(dim, inner, 1, bias=bias)
        self.proj = nn.Conv2d(inner, dim, 1, bias=bias)
        self.scale = dim_head ** -0.5
        self.luma_cond = LumaCond(heads, dim_head, per_head=True)
        self.luma_bias = luma_bias
        if luma_bias:
            # learnable strength for luminance bias
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def _rearrange_heads(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.heads, self.dim_head, H, W)  # B h d H W
        x = x.flatten(3)  # B h d (HW)
        x = x.transpose(2, 3)  # B h (HW) d
        return x, H, W

    def forward(self, x, luma=None, rgb_input=None):
        # x: BxCxHxW features. luma: Bx1xHxW (optional), or compute from rgb_input.
        B, C, H, W = x.shape

        if luma is None:
            assert rgb_input is not None and rgb_input.shape[1] == 3, \
                "Provide luma or rgb_input for luminance."
            luma = rgb_to_luma(rgb_input)

        # Q/K/V
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # ----- (A) FiLM conditioning from luminance -----
        gq, bq, gk, bk, gv, bv = self.luma_cond(luma, H, W)  # each: Bx(inner)x1x1
        # reshape to (B, heads, dim_head, 1, 1) for broadcasting
        def reshape_film(t):
            B_, CH, _, _ = t.shape
            return t.view(B_, self.heads, self.dim_head, 1, 1)

        gq, bq = reshape_film(gq), reshape_film(bq)
        gk, bk = reshape_film(gk), reshape_film(bk)
        gv, bv = reshape_film(gv), reshape_film(bv)

        def apply_film(t, g, b):
            B_, CH, H_, W_ = t.shape
            t = t.view(B_, self.heads, self.dim_head, H_, W_)
            t = g * t + b
            return t  # B h d H W

        q = apply_film(q, gq, bq)
        k = apply_film(k, gk, bk)
        v = apply_film(v, gv, bv)

        # flatten spatial
        q = q.flatten(3).transpose(2, 3)  # B h (HW) d
        k = k.flatten(3).transpose(2, 3)  # B h (HW) d
        v = v.flatten(3).transpose(2, 3)  # B h (HW) d

        # attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B h N N

        # ----- (B) Luminance-biased attention (optional) -----
        if self.luma_bias:
            # inverse-luminance: focus darker areas, smooth with avgpool
            invL = 1.0 - luma
            invL = F.avg_pool2d(invL, kernel_size=3, stride=1, padding=1)
            invL = invL.view(B, 1, -1)  # Bx1xN
            # Normalize to zero-mean per image for stable bias
            invL = invL - invL.mean(dim=-1, keepdim=True)
            # Broadcast to heads: bias(i->j) depends on dst pixel j
            bias = invL.unsqueeze(1).repeat(1, self.heads, invL.shape[-1], 1)  # B h N N (col-wise)
            attn_logits = attn_logits + self.alpha * bias

        attn = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn, v)  # B h N d
        # fold heads
        out = out.transpose(2, 3).contiguous()  # B h d N
        out = out.view(B, self.heads * self.dim_head, H, W)
        return self.proj(out)


# Example inside a Transformer block
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn  = LuminanceAwareMHSA(dim, heads=8, dim_head=64, luma_bias=True)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn   = nn.Sequential(
            nn.Conv2d(dim, 4*dim, 1), nn.GELU(),
            nn.Conv2d(4*dim, dim, 1)
        )

    def forward(self, x, rgb_for_luma=None, luma=None):
        a = self.attn(self.norm1(x), luma=luma, rgb_input=rgb_for_luma)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x

