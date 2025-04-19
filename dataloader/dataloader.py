import rawpy
import imageio
import numpy as np
import torch
import random
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import unittest
import tempfile

class BaseImageDataset(Dataset):
    """Base class for image datasets with common functionality"""
    def __init__(self, input_paths, target_paths, patch_size=512, training=True):
        self.training = training
        self.patch_size = patch_size
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.input_images, self.target_images = self._load_images()
        
    def _load_images(self):
        """Load all images into memory"""
        mode = "Train" if self.training else "Test"
        print(f"\n...... {mode} files loading\n")
        
        input_images = []
        target_images = []
        
        for inp_path, tgt_path in tqdm(zip(self.input_paths, self.target_paths), 
                                      total=len(self.input_paths)):
            input_images.append(self._read_input(inp_path))
            target_images.append(self._read_target(tgt_path))
            
        print(f"\n{mode} files loaded ......\n")
        return input_images, target_images
    
    def _read_input(self, path):
        raise NotImplementedError
        
    def _read_target(self, path):
        raise NotImplementedError
        
    def _augment(self, *images):
        """Apply random flips to images"""
        if random.random() > 0.5:  # 50% chance
            images = [np.fliplr(img).copy() for img in images]
        if random.random() < 0.2:  # 20% chance
            images = [np.flipud(img).copy() for img in images]
        return images
        
    def _crop(self, *images):
        """Random crop for training, full image for testing"""
        if not self.training:
            return images
            
        H, W = images[0].shape[:2]
        i = random.randint(0, (H - self.patch_size - 2) // 2) * 2
        j = random.randint(0, (W - self.patch_size - 2) // 2) * 2
        
        cropped = []
        for img in images:
            if img.ndim == 2:  # Raw image
                cropped.append(img[i:i+self.patch_size, j:j+self.patch_size])
            else:  # RGB image
                cropped.append(img[i:i+self.patch_size, j:j+self.patch_size, :])
                
        return cropped
        
    def __len__(self):
        return len(self.input_images)

class SIDDataset(BaseImageDataset):
    """Dataset for See-in-the-Dark (SID) task"""
    def _read_input(self, path):
        with rawpy.imread(path) as raw:
            return raw.raw_image_visible.copy()
        
    def _read_target(self, path):
        with rawpy.imread(path) as raw:
            return raw.postprocess(use_camera_wb=True, half_size=False, 
                                 no_auto_bright=True, output_bps=16)
    
    def __getitem__(self, idx):
        img_short = self.input_images[idx]
        img_long = self.target_images[idx]
        
        # Get aperture from filename
        ap = 300 if self.target_paths[idx][-7] == '3' else 100
        
        # Crop and augment if training
        img_short, img_long = self._crop(img_short, img_long)
        if self.training:
            img_short, img_long = self._augment(img_short, img_long)
        
        # Normalize
        img_short = (np.maximum(img_short.astype(np.float32) - 512, 0)) / (16383 - 512) * ap
        img_long = img_long.astype(np.float32) / 65535
        
        # Convert to tensor
        img_short = torch.from_numpy(img_short).float().unsqueeze(0)
        img_long = torch.from_numpy(np.transpose(img_long, [2, 0, 1])).float()
        
        return img_short, img_long

class MCRDataset(BaseImageDataset):
    """Dataset for Multi-Channel Reconstruction (MCR) task"""
    def _read_input(self, path):
        return imageio.imread(path)
        
    def _read_target(self, path):
        return imageio.imread(path)
    
    def __getitem__(self, idx):
        inp_raw = self.input_images[idx]
        gt_rgb = self.target_images[idx]
        
        # Get exposure info from filename
        img_num = int(self.input_paths[idx][-23:-20])
        img_expo = int(self.input_paths[idx][-8:-4], 16)
        gt_expo = 12287 if img_num < 500 else 1023
        amp = gt_expo / img_expo
        
        # Crop and augment if training
        inp_raw, gt_rgb = self._crop(inp_raw, gt_rgb)
        if self.training:
            inp_raw, gt_rgb = self._augment(inp_raw, gt_rgb)
        
        # Normalize
        inp_raw = (inp_raw / 255 * amp).astype(np.float32)
        gt_rgb = (gt_rgb / 255).astype(np.float32)
        
        # Convert to tensor
        inp = torch.from_numpy(inp_raw).float().unsqueeze(0)
        gt = torch.from_numpy(np.transpose(gt_rgb, [2, 0, 1])).float()
        
        return inp, gt

# Test Cases
class TestImageDatasets(unittest.TestCase):
    def setUp(self):
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create dummy raw files for SID
        self.sid_input_path = os.path.join(self.temp_dir.name, "test_sid_input.dng")
        self.sid_target_path = os.path.join(self.temp_dir.name, "test_sid_target3.dng")
        
        # Create dummy image files for MCR
        self.mcr_input_path = os.path.join(self.temp_dir.name, "test_mcr_input_001_1fff.png")
        self.mcr_target_path = os.path.join(self.temp_dir.name, "test_mcr_target.png")
        
        # Create actual dummy files (in reality these would be proper image files)
        with open(self.sid_input_path, 'wb') as f:
            f.write(b'dummy raw data')
        with open(self.sid_target_path, 'wb') as f:
            f.write(b'dummy raw data')
        imageio.imwrite(self.mcr_input_path, np.random.randint(0, 255, (1024, 1024), dtype=np.uint8))
        imageio.imwrite(self.mcr_target_path, np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_sid_dataset(self):
        # Test SID dataset initialization
        dataset = SIDDataset([self.sid_input_path], [self.sid_target_path], training=True)
        self.assertEqual(len(dataset), 1)
        
        # Test item access
        input_img, target_img = dataset[0]
        self.assertIsInstance(input_img, torch.Tensor)
        self.assertIsInstance(target_img, torch.Tensor)
        self.assertEqual(input_img.shape[0], 1)  # Single channel
        self.assertEqual(target_img.shape[0], 3)  # RGB channels
    
    def test_mcr_dataset(self):
        # Test MCR dataset initialization
        dataset = MCRDataset([self.mcr_input_path], [self.mcr_target_path], training=False)
        self.assertEqual(len(dataset), 1)
        
        # Test item access (no crop in test mode)
        input_img, target_img = dataset[0]
        self.assertIsInstance(input_img, torch.Tensor)
        self.assertIsInstance(target_img, torch.Tensor)
        self.assertEqual(input_img.shape[0], 1)  # Single channel
        self.assertEqual(target_img.shape[0], 3)  # RGB channels
    
    def test_augmentation(self):
        # Test augmentation (should only happen in training mode)
        dataset = MCRDataset([self.mcr_input_path], [self.mcr_target_path], training=True)
        input_img1, target_img1 = dataset[0]
        input_img2, target_img2 = dataset[0]  # Get same item again
        
        # There's a chance they'll be different due to random augmentation
        # Run multiple times to increase chance of catching differences
        different = False
        for _ in range(10):
            input_img1, target_img1 = dataset[0]
            input_img2, target_img2 = dataset[0]
            if not torch.equal(input_img1, input_img2):
                different = True
                break
        self.assertTrue(different, "Augmentation should sometimes produce different results")
    
    def test_cropping(self):
        # Test cropping (should only happen in training mode)
        dataset_train = MCRDataset([self.mcr_input_path], [self.mcr_target_path], 
                                 patch_size=256, training=True)
        input_train, _ = dataset_train[0]
        
        dataset_test = MCRDataset([self.mcr_input_path], [self.mcr_target_path], 
                                patch_size=256, training=False)
        input_test, _ = dataset_test[0]
        
        self.assertEqual(input_train.shape[-1], 256)  # Cropped size
        self.assertEqual(input_test.shape[-1], 1024)  # Original size

if __name__ == '__main__':
    unittest.main()
