import numpy as np
import os
import rawpy
import torch
from torch.utils import data


class SIDSonyDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train',
                 transpose=False, h_flip=False, v_flip=False, ratio=True):
        """SID (See-in-the-Dark) Sony dataset loader.

        Args:
            data_dir: Root directory containing the images
            image_list_file: File containing image pairs and metadata
            patch_size: Size of random crops for training
            split: 'train' or 'val'
            transpose: Whether to apply random transposition
            h_flip: Whether to apply random horizontal flips
            v_flip: Whether to apply random vertical flips
            ratio: Whether to apply exposure ratio scaling
        """
        self.data_dir = data_dir
        self.image_list_file = os.path.join(data_dir, image_list_file)
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 512
        self.white_level = 16383

        # Load image pairs and metadata
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:  # Ensure we have all expected metadata
                    input_path, gt_path, iso, focus = parts[:4]
                    input_exposure = float(os.path.split(input_path)[-1][9:-5])
                    gt_exposure = float(os.path.split(gt_path)[-1][9:-5])
                    ratio = min(gt_exposure / input_exposure, 300)

                    self.img_info.append({
                        'input_path': input_path,
                        'gt_path': gt_path,
                        'ratio': np.float32(ratio),
                        'iso': float(iso[3:]),
                        'focus': focus,
                        'input_exposure': input_exposure,
                        'gt_exposure': gt_exposure
                    })

        print(f"Loaded {len(self.img_info)} images for {self.split}")

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]

        # Load and process input RAW
        input_raw = rawpy.imread(os.path.join(self.data_dir, info['input_path']))
        input_raw = self.pack_raw(input_raw)
        input_raw = (input_raw.astype(np.float32) - self.black_level) / (self.white_level - self.black_level)

        # Load and process ground truth
        gt_raw = rawpy.imread(os.path.join(self.data_dir, info['gt_path']))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1).astype(np.float32) / 65535.0
        gt_raw = self.pack_raw(gt_raw)
        gt_raw = (gt_raw.astype(np.float32) - self.black_level) / (self.white_level - self.black_level)

        # Apply exposure ratio if needed
        if self.ratio:
            input_raw = input_raw * info['ratio']

        # Clip values to [0, 1]
        input_raw = np.clip(input_raw, 0.0, 1.0)
        gt_raw = np.clip(gt_raw, 0.0, 1.0)

        # Data augmentation for training
        if self.split == 'train':
            if self.h_flip and np.random.rand() < 0.5:
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
                gt_rgb = np.flip(gt_rgb, axis=2)

            if self.v_flip and np.random.rand() < 0.5:
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
                gt_rgb = np.flip(gt_rgb, axis=1)

            if self.transpose and np.random.rand() < 0.5:
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1))
                gt_rgb = np.transpose(gt_rgb, (0, 2, 1))

            if self.patch_size:
                input_raw, gt_raw, gt_rgb = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)

        # Convert to tensors
        input_raw = torch.from_numpy(np.ascontiguousarray(input_raw)).float()
        gt_raw = torch.from_numpy(np.ascontiguousarray(gt_raw)).float()
        gt_rgb = torch.from_numpy(np.ascontiguousarray(gt_rgb)).float()

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path': info['input_path'],
            'gt_path': info['gt_path'],
            'ratio': info['ratio'],
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure']
        }

    @staticmethod
    def pack_raw(raw):
        """Pack Bayer RAW image into 4 channels (RGBG)."""
        im = raw.raw_image_visible.astype(np.uint16)
        H, W = im.shape
        out = np.stack([
            im[0:H:2, 0:W:2],  # R
            im[0:H:2, 1:W:2],  # G1
            im[1:H:2, 1:W:2],  # B
            im[1:H:2, 0:W:2]  # G2
        ], axis=0)
        return out

    @staticmethod
    def crop_random_patch(input_raw, gt_raw, gt_rgb, patch_size):
        """Random crop for training."""
        _, H, W = input_raw.shape
        yy = np.random.randint(0, H - patch_size)
        xx = np.random.randint(0, W - patch_size)

        input_patch = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_patch = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        rgb_patch = gt_rgb[:, yy * 2:(yy + patch_size) * 2, xx * 2:(xx + patch_size) * 2]

        return input_patch, gt_patch, rgb_patch


# Testing code
if __name__ == "__main__":
    # Example usage - you'll need to provide actual paths to your data
    data_dir = "/path/to/your/data"
    image_list_file = "Sony_train_list.txt"

    # Create dataset instance
    dataset = SIDSonyDataset(
        data_dir=data_dir,
        image_list_file=image_list_file,
        patch_size=512,
        split='train',
        h_flip=True,
        v_flip=True
    )

    # Test basic functionality
    print(f"Dataset contains {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample contains:")
    print(f"  - Input RAW shape: {sample['input_raw'].shape}")
    print(f"  - GT RAW shape: {sample['gt_raw'].shape}")
    print(f"  - GT RGB shape: {sample['gt_rgb'].shape}")
    print(f"  - Exposure ratio: {sample['ratio']}")
