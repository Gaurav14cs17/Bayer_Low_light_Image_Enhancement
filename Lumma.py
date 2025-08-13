import torch
import torch.nn as nn

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

        # Luminance coefficients in ITU-R BT.601 (linear RGB domain)
        self.register_buffer("r_w", torch.tensor(0.299, dtype=torch.float32))
        self.register_buffer("g_w", torch.tensor(0.587, dtype=torch.float32))
        self.register_buffer("b_w", torch.tensor(0.114, dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 4, H, W] in RGGB planes
               x[:,0] -> R
               x[:,1] -> G1
               x[:,2] -> G2
               x[:,3] -> B

        Returns:
            Luma: Tensor of shape [B, 1, H, W]
        """
        assert x.shape[1] == 4, f"Expected 4-channel RGGB input, got {x.shape[1]} channels"

        r = x[:, 0:1]
        g1 = x[:, 1:2]
        g2 = x[:, 2:3]
        b = x[:, 3:4]

        g_avg = 0.5 * (g1 + g2)

        luma = self.r_w * r + self.g_w * g_avg + self.b_w * b
        luma = luma / (luma.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + self.eps)

        return luma
