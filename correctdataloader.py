import rawpy
from torch.utils.data import Dataset
import tqdm
import random
import imageio
import numpy as np
import torch
import os
import time
from PIL import Image


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SID_Dataset(Dataset):
    """Improved SID Dataset with proper RAW processing and memory management"""

    def __init__(self, short_expo_files, long_expo_files, patch_size=512, training=True,
                 use_camera_wb=False, gt_png=False):
        """
        short_expo_files: list of short exposure RAW file paths
        long_expo_files: list of long exposure RAW file paths
        patch_size: size for random cropping during training
        training: whether in training mode
        use_camera_wb: whether to use camera white balance
        gt_png: whether GT is in PNG format (demosaiced)
        """
        self.training = training
        self.patch_size = patch_size
        self.use_camera_wb = use_camera_wb
        self.gt_png = gt_png

        # Store file paths instead of loading all images to memory
        self.short_files = short_expo_files
        self.long_files = long_expo_files

        # Timing meters
        self.raw_read_time = AverageMeter()
        self.raw_process_time = AverageMeter()

        print(f'\n...... {"Train" if training else "Test"} files initialized ({len(self.short_files)} pairs)\n')

    def __len__(self):
        return len(self.short_files)

    def pack_raw(self, raw):
        """Pack Sony RAW image into 4-channel format"""
        black = np.array(raw.black_level_per_channel)[:, None, None]
        white = raw.white_level
        im = raw.raw_image_visible.astype(np.float32)
        im = (im - black.min()) / (white - black.min())
        im = np.clip(im, 0, 1)

        im = np.expand_dims(im, axis=2)
        H, W, _ = im.shape
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out

    def __getitem__(self, idx):
        start_time = time.time()
        
        # Load and process short exposure RAW
        with rawpy.imread(self.short_files[idx]) as raw:
            # Extract exposure information from filename
            filename = os.path.basename(self.short_files[idx])
            short_exp = float(filename[9:-4])
            long_exp = float(os.path.basename(self.long_files[idx])[9:-4])
            ratio = min(long_exp / short_exp, 300)
            
            # Pack RAW and apply ratio (only once)
            img_short = self.pack_raw(raw) * ratio
        
        self.raw_read_time.update(time.time() - start_time)
        
        start_time = time.time()

        # Load ground truth
        if self.gt_png:
            img_long = np.array(Image.open(self.long_files[idx]), dtype=np.float32) / 255.0
        else:
            with rawpy.imread(self.long_files[idx]) as raw:
                img_long = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                img_long = np.float32(img_long / 65535.0)

        self.raw_process_time.update(time.time() - start_time)

        # Clamp values
        img_short = np.minimum(img_short, 1.0)

        # Convert to channel-first format
        img_short = img_short.transpose(2, 0, 1)  # HWC to CHW
        img_long = img_long.transpose(2, 0, 1)

        H, W = img_short.shape[1], img_short.shape[2]

        # Data augmentation for training
        if self.training and self.patch_size:
            # Ensure crop coordinates are even for Bayer pattern alignment
            yy = random.randint(0, (H - self.patch_size) // 2) * 2
            xx = random.randint(0, (W - self.patch_size) // 2) * 2

            img_short = img_short[:, yy:yy + self.patch_size, xx:xx + self.patch_size]

            # GT might be different resolution
            if img_long.shape[1] == H * 2:
                img_long = img_long[:, yy * 2:(yy + self.patch_size) * 2, xx * 2:(xx + self.patch_size) * 2]
            else:
                img_long = img_long[:, yy:yy + self.patch_size, xx:xx + self.patch_size]

            # Random flips
            if random.random() > 0.5:
                img_short = np.flip(img_short, axis=2)
                img_long = np.flip(img_long, axis=2)
            if random.random() > 0.5:
                img_short = np.flip(img_short, axis=1)
                img_long = np.flip(img_long, axis=1)
            if random.random() > 0.5:
                img_short = np.transpose(img_short, (0, 2, 1))
                img_long = np.transpose(img_long, (0, 2, 1))

        # Convert to torch tensors
        img_short_tensor = torch.from_numpy(img_short).float()
        img_long_tensor = torch.from_numpy(img_long).float()

        return img_short_tensor, img_long_tensor


class MCR_Dataset(Dataset):
    """Improved MCR Dataset with proper processing"""

    def __init__(self, train_c_path, train_rgb_path, patch_size=512, training=True):
        self.training = training
        self.patch_size = patch_size
        self.train_c_path = train_c_path
        self.train_rgb_path = train_rgb_path

        print(f'\n...... {"Train" if training else "Test"} files initialized ({len(self.train_c_path)} pairs)\n')

    def __len__(self):
        return len(self.train_c_path)

    def __getitem__(self, idx):
        # Load images on-the-fly to save memory
        inp_raw_image = imageio.imread(self.train_c_path[idx]).astype(np.float32)
        gt_rgb_image = imageio.imread(self.train_rgb_path[idx]).astype(np.float32)

        # Extract exposure information from filename
        filename = os.path.basename(self.train_c_path[idx])
        img_num = int(filename[-23:-20])
        img_expo = int(filename[-8:-4], 16)

        # Determine ground truth exposure
        gt_expo = 12287 if img_num < 500 else 1023
        amp = gt_expo / img_expo

        # Apply amplification
        inp_raw_image = (inp_raw_image / 255 * amp)
        gt_rgb_image = gt_rgb_image / 255

        H, W = inp_raw_image.shape

        # Data augmentation for training
        if self.training and self.patch_size:
            # Ensure even coordinates for proper Bayer pattern handling
            i = random.randint(0, (H - self.patch_size - 2) // 2) * 2
            j = random.randint(0, (W - self.patch_size - 2) // 2) * 2

            inp_raw = inp_raw_image[i:i + self.patch_size, j:j + self.patch_size]
            gt_rgb = gt_rgb_image[i:i + self.patch_size, j:j + self.patch_size, :]

            # Random flips
            if random.random() > 0.5:
                inp_raw = np.fliplr(inp_raw)
                gt_rgb = np.fliplr(gt_rgb)
            if random.random() > 0.2:  # More reasonable probability
                inp_raw = np.flipud(inp_raw)
                gt_rgb = np.flipud(gt_rgb)
        else:
            inp_raw = inp_raw_image
            gt_rgb = gt_rgb_image

        # Convert to torch tensors
        gt_tensor = torch.from_numpy(gt_rgb.transpose(2, 0, 1)).float()
        inp_tensor = torch.from_numpy(inp_raw).float().unsqueeze(0)

        return inp_tensor, gt_tensor


# Helper function for backward compatibility
def load_data_SID(short_expo_files, long_expo_files, patch_size=512, training=True, **kwargs):
    return SID_Dataset(short_expo_files, long_expo_files, patch_size, training, **kwargs)


def load_data_MCR(train_c_path, train_rgb_path, patch_size=512, training=True):
    return MCR_Dataset(train_c_path, train_rgb_path, patch_size, training)
