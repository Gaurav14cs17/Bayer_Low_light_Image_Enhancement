import rawpy
from torch.utils.data import Dataset
import tqdm
import random
import imageio
import numpy as np
import torch

# ------------------------------
# Bayer channel correction
# ------------------------------
def correct_bayer_channels(rgb, pattern="RGGB"):
    pattern = pattern.upper()
    if pattern == "BGGR":
        rgb = rgb[..., [2, 1, 0]]  # Swap R/B
    elif pattern == "GBRG":
        rgb = rgb[..., [1, 0, 2]]
    elif pattern == "GRBG":
        rgb = rgb[..., [0, 2, 1]]
    # RGGB → keep as is
    return rgb

def auto_correct_rb(rgb):
    r_mean = rgb[..., 0].mean()
    b_mean = rgb[..., 2].mean()
    if r_mean < b_mean:
        rgb = rgb[..., [2, 1, 0]]
    return rgb

# ------------------------------
# SID Dataset
# ------------------------------
def image_read_SID(short_expo_files, long_expo_files, bayer_pattern="RGGB"):
    short_list = []
    long_list = []

    for i in tqdm.tqdm(range(len(short_expo_files))):
        # Load short exposure raw
        raw = rawpy.imread(short_expo_files[i])
        img_short = raw.raw_image_visible.copy()
        raw.close()
        short_list.append(img_short)

        # Load long exposure RGB GT
        raw = rawpy.imread(long_expo_files[i])
        img_long = raw.postprocess(use_camera_wb=True, half_size=False,
                                   no_auto_bright=True, output_bps=16).copy()
        raw.close()
        # Fix Bayer channel order and R/B swap
        img_long = correct_bayer_channels(img_long, bayer_pattern)
        img_long = auto_correct_rb(img_long)
        long_list.append(img_long)

    return short_list, long_list

class load_data_SID(Dataset):
    def __init__(self, short_expo_files, long_expo_files, patch_size=512, training=True, bayer_pattern="RGGB"):
        self.training = training
        self.patch_size = patch_size
        self.bayer_pattern = bayer_pattern

        if self.training:
            print('\n...... Train files loading\n')
        else:
            print('\n...... Test files loading\n')

        self.short_list, self.long_list = image_read_SID(short_expo_files, long_expo_files, bayer_pattern)
        print(f"\nFiles loaded: {len(self.short_list)} samples\n")

    def __len__(self):
        return len(self.short_list)

    def __getitem__(self, idx):
        img_short = self.short_list[idx].astype(np.float32)
        img_long = self.long_list[idx].astype(np.float32) / 65535.0  # normalize GT

        # Exposure scaling for raw
        ap = 300 if self.long_list[idx][-7] == '3' else 100
        img_short = np.maximum(img_short - 512, 0) / (16383 - 512) * ap

        H, W = img_short.shape

        # Crop and augmentation
        if self.training and H >= self.patch_size and W >= self.patch_size:
            i = random.randint(0, (H - self.patch_size) // 2) * 2
            j = random.randint(0, (W - self.patch_size) // 2) * 2
            img_short = img_short[i:i+self.patch_size, j:j+self.patch_size]
            img_long = img_long[i:i+self.patch_size, j:j+self.patch_size, :]

            if random.random() > 0.5:
                img_short = np.fliplr(img_short).copy()
                img_long = np.fliplr(img_long).copy()
            if random.random() < 0.2:
                img_short = np.flipud(img_short).copy()
                img_long = np.flipud(img_long).copy()

        img_short_tensor = torch.from_numpy(img_short).float().unsqueeze(0)  # (1,H,W)
        img_long_tensor = torch.from_numpy(np.transpose(img_long, [2,0,1])).float()  # (3,H,W)
        return img_short_tensor, img_long_tensor

# ------------------------------
# MCR Dataset
# ------------------------------
def image_read_MCR(train_c_path, train_rgb_path, bayer_pattern="RGGB"):
    inp_list = []
    gt_list = []

    for i in tqdm.tqdm(range(len(train_c_path))):
        color_raw = imageio.imread(train_c_path[i])
        inp_list.append(color_raw.astype(np.float32))

        gt_rgb = imageio.imread(train_rgb_path[i])
        gt_rgb = correct_bayer_channels(gt_rgb.astype(np.float32), bayer_pattern)
        gt_rgb = auto_correct_rb(gt_rgb)
        gt_rgb /= 255.0  # normalize
        gt_list.append(gt_rgb)

    return inp_list, gt_list

class load_data_MCR(Dataset):
    def __init__(self, train_c_path, train_rgb_path, patch_size=512, training=True, bayer_pattern="RGGB"):
        self.training = training
        self.patch_size = patch_size
        self.bayer_pattern = bayer_pattern

        if self.training:
            print('\n...... Train files loading\n')
        else:
            print('\n...... Test files loading\n')

        self.inp_list, self.gt_list = image_read_MCR(train_c_path, train_rgb_path, bayer_pattern)
        print(f"\nFiles loaded: {len(self.gt_list)} samples\n")

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        inp_raw = self.inp_list[idx]
        gt_rgb = self.gt_list[idx]

        # Exposure scaling
        img_num = int(str(idx)[-3:])  # replace your original logic here
        img_expo = 1000  # default if not available
        gt_expo = 12287 if img_num < 500 else 1023
        amp = gt_expo / img_expo
        inp_raw = inp_raw / 255 * amp

        H, W = inp_raw.shape

        # Random crop & augmentation
        if self.training and H >= self.patch_size and W >= self.patch_size:
            i = random.randint(0, (H - self.patch_size)//2) * 2
            j = random.randint(0, (W - self.patch_size)//2) * 2
            inp_raw = inp_raw[i:i+self.patch_size, j:j+self.patch_size]
            gt_rgb = gt_rgb[i:i+self.patch_size, j:j+self.patch_size, :]

            if random.random() > 0.5:
                inp_raw = np.fliplr(inp_raw).copy()
                gt_rgb = np.fliplr(gt_rgb).copy()
            if random.random() < 0.2:
                inp_raw = np.flipud(inp_raw).copy()
                gt_rgb = np.flipud(gt_rgb).copy()

        inp_tensor = torch.from_numpy(inp_raw).float().unsqueeze(0)
        gt_tensor = torch.from_numpy(np.transpose(gt_rgb, [2,0,1])).float()
        return inp_tensor, gt_tensor
