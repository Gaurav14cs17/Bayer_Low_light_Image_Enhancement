import rawpy
from torch.utils.data import Dataset
import torch
import numpy as np
import random

def bayer_downshuffle(var, bayer_pattern, r=2):
    """
    Space-to-Depth for Bayer pattern.
    Input:
        var: tensor of size (B,1,H,W)
        bayer_pattern: 2x2 array from rawpy.raw_pattern
        r: downscale factor (default 2)
    Output:
        down-shuffled tensor (B,4,H/2,W/2)
        Channels order: R,G1,G2,B according to pattern
    """
    if r != 2:
        raise ValueError("Bayer S2D only supports r=2")

    b, c, h, w = var.size()

    # Map Bayer pattern to channel order R,G1,G2,B
    pattern_map = {
        (0,1,1,2): ['R','G1','G2','B'],  # RGGB
        (2,1,1,0): ['B','G2','G1','R'],  # BGGR
        (1,0,2,1): ['G1','R','B','G2'],  # GRBG
        (1,2,0,1): ['G2','B','R','G1'],  # GBRG
    }
    key = tuple(bayer_pattern.flatten())
    if key not in pattern_map:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

    # Extract 4 channels
    top_left     = var[:, :, 0::2, 0::2]
    top_right    = var[:, :, 0::2, 1::2]
    bottom_left  = var[:, :, 1::2, 0::2]
    bottom_right = var[:, :, 1::2, 1::2]

    mapping = pattern_map[key]
    channel_dict = {'R': top_left, 'G1': top_right, 'G2': bottom_left, 'B': bottom_right}
    result = torch.cat([channel_dict[ch] for ch in ['R','G1','G2','B']], dim=1)
    return result

def downshuffle(var, r=2, bayer_pattern=None):
    """
    General Space-to-Depth.
    If single channel Bayer, use bayer_downshuffle.
    """
    b, c, h, w = var.size()
    if c == 1 and bayer_pattern is not None:
        return bayer_downshuffle(var, bayer_pattern, r)
    # General PixelUnshuffle
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
              .permute(0,1,3,5,2,4).contiguous() \
              .view(b, out_channel, out_h, out_w)

class load_data_SID(Dataset):
    """SID Dataset with Bayer-aware S2D and metadata exposure scaling."""

    def __init__(self, short_expo_files, long_expo_files, patch_size=512, training=True, s2d_r=2):
        self.training = training
        self.patch_size = patch_size
        self.short_files = short_expo_files
        self.long_files  = long_expo_files
        self.s2d_r = s2d_r

        print(f"\n...... {'Train' if self.training else 'Test'} files loading ......")
        print(f"Total files: {len(self.short_files)}\n")

    def __len__(self):
        return len(self.short_files)

    def __getitem__(self, idx):
        # --------------------------
        # Load short exposure (Bayer)
        # --------------------------
        raw_short = rawpy.imread(self.short_files[idx])
        short_expo_time = raw_short.raw_image_visible_metadata.exposure
        img_short = raw_short.raw_image_visible.astype(np.float32).copy()
        bayer_pattern = raw_short.raw_pattern  # 2x2 pattern
        raw_short.close()

        # --------------------------
        # Load long exposure (RGB)
        # --------------------------
        raw_long = rawpy.imread(self.long_files[idx])
        long_expo_time = raw_long.raw_image_visible_metadata.exposure
        img_long = raw_long.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16
        ).astype(np.float32) / 65535.0
        raw_long.close()

        # --------------------------
        # Exposure scaling for short
        # --------------------------
        scale = long_expo_time / short_expo_time
        img_short = np.maximum(img_short - 512, 0) / (16383 - 512) * scale

        H, W = img_short.shape

        # --------------------------
        # Crop patch if training
        # --------------------------
        if self.training and H >= self.patch_size and W >= self.patch_size:
            i = random.randint(0, (H - self.patch_size)//2)*2
            j = random.randint(0, (W - self.patch_size)//2)*2
            img_short = img_short[i:i+self.patch_size, j:j+self.patch_size]
            img_long  = img_long[i:i+self.patch_size, j:j+self.patch_size, :]
            # Random flips
            if random.random() > 0.5:
                img_short = np.fliplr(img_short).copy()
                img_long  = np.fliplr(img_long).copy()
            if random.random() < 0.2:
                img_short = np.flipud(img_short).copy()
                img_long  = np.flipud(img_long).copy()

        # --------------------------
        # Convert to torch tensors
        # --------------------------
        img_short = torch.from_numpy(img_short).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        img_long  = torch.from_numpy(np.transpose(img_long, (2,0,1))).float()       # (3,H,W)

        # --------------------------
        # Apply Bayer-aware S2D
        # --------------------------
        img_short_s2d = downshuffle(img_short, self.s2d_r, bayer_pattern)  # (1,4,H/2,W/2)

        return img_short_s2d, img_long
