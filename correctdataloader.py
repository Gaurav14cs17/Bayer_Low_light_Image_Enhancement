import rawpy
import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import os
from PIL import Image

# ------------------------------
# Enhanced Bayer pattern detection
# ------------------------------
def detect_bayer_pattern_from_metadata(raw):
    """Try to detect Bayer pattern from raw file metadata"""
    try:
        # Try to get pattern from raw pattern attribute
        if hasattr(raw, 'raw_pattern'):
            pattern_map = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
            pattern = ''
            for i in range(2):
                for j in range(2):
                    pattern += pattern_map[raw.raw_pattern[i, j]]
            return pattern
        
        # Try to get pattern from color description
        if hasattr(raw, 'color_desc'):
            pattern = raw.color_desc.decode('utf-8', errors='ignore')
            if len(pattern) >= 4 and all(c in 'RGBG' for c in pattern[:4]):
                return pattern[:4].upper()
                
    except Exception as e:
        print(f"Error detecting pattern: {e}")
    
    return None

# ------------------------------
# Pack RAW into 4 channels (RGGB) with enhanced handling
# ------------------------------
def pack_raw(raw, pattern="RGGB"):
    """
    Pack Bayer RAW into 4 channels with pattern-aware packing.
    Maintains correct channel order and alignment.
    """
    im = raw.raw_image_visible.astype(np.float32)
    
    # Get black and white levels
    black_level = np.array(raw.black_level_per_channel)
    white_level = raw.white_level
    
    # Handle different black level configurations
    if len(black_level) == 4:
        # Separate black levels for each channel
        black_level = black_level.reshape(2, 2)
    else:
        # Single black level for all channels
        black_level = np.full((2, 2), black_level[0])
    
    # Ensure even dimensions
    H, W = im.shape
    H -= H % 2
    W -= W % 2
    im = im[:H, :W]
    
    # Pattern-aware packing
    pattern = pattern.upper()
    if pattern == "RGGB":
        channels = [
            im[0:H:2, 0:W:2],  # R
            im[0:H:2, 1:W:2],  # G1
            im[1:H:2, 1:W:2],  # B
            im[1:H:2, 0:W:2]   # G2
        ]
        black_levels = [
            black_level[0, 0],  # R
            black_level[0, 1],  # G1
            black_level[1, 1],  # B
            black_level[1, 0]   # G2
        ]
    elif pattern == "BGGR":
        channels = [
            im[1:H:2, 1:W:2],  # R
            im[0:H:2, 1:W:2],  # G1
            im[0:H:2, 0:W:2],  # B
            im[1:H:2, 0:W:2]   # G2
        ]
        black_levels = [
            black_level[1, 1],  # R
            black_level[0, 1],  # G1
            black_level[0, 0],  # B
            black_level[1, 0]   # G2
        ]
    elif pattern == "GRBG":
        channels = [
            im[0:H:2, 1:W:2],  # R
            im[0:H:2, 0:W:2],  # G1
            im[1:H:2, 0:W:2],  # B
            im[1:H:2, 1:W:2]   # G2
        ]
        black_levels = [
            black_level[0, 1],  # R
            black_level[0, 0],  # G1
            black_level[1, 0],  # B
            black_level[1, 1]   # G2
        ]
    elif pattern == "GBRG":
        channels = [
            im[1:H:2, 0:W:2],  # R
            im[0:H:2, 0:W:2],  # G1
            im[0:H:2, 1:W:2],  # B
            im[1:H:2, 1:W:2]   # G2
        ]
        black_levels = [
            black_level[1, 0],  # R
            black_level[0, 0],  # G1
            black_level[0, 1],  # B
            black_level[1, 1]   # G2
        ]
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    
    # Subtract pattern-specific black levels and normalize
    out = np.stack([
        np.maximum(channels[i] - black_levels[i], 0) / max(white_level - black_levels[i], 1e-6)
        for i in range(4)
    ], axis=0)
    
    return out  # shape: (4, H/2, W/2)

# ------------------------------
# Enhanced color correction
# ------------------------------
def correct_bayer_channels(rgb, pattern="RGGB", auto_detect=True):
    """
    Enhanced Bayer channel correction with auto-detection
    """
    if auto_detect:
        # Auto-detect if pattern seems wrong based on color statistics
        r_mean = rgb[..., 0].mean()
        b_mean = rgb[..., 2].mean()
        g_mean = rgb[..., 1].mean()
        
        # If blue is significantly stronger than red, we might need to swap
        if b_mean > r_mean * 1.8 and g_mean > max(r_mean, b_mean) * 0.8:
            if pattern in ["RGGB", "GRBG"]:
                pattern = "BGGR" if pattern == "RGGB" else "GBRG"
            elif pattern in ["BGGR", "GBRG"]:
                pattern = "RGGB" if pattern == "BGGR" else "GRBG"
    
    pattern = pattern.upper()
    if pattern == "BGGR":
        rgb = rgb[..., [2, 1, 0]]  # Swap R/B
    elif pattern == "GBRG":
        rgb = rgb[..., [1, 0, 2]]  # GBRG → RGB
    elif pattern == "GRBG":
        rgb = rgb[..., [0, 2, 1]]  # GRBG → RGB
    
    return rgb

def auto_correct_color_balance(rgb):
    """
    More sophisticated auto color correction
    """
    # Simple auto white balance based on gray world assumption
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r_avg, g_avg, b_avg = r.mean(), g.mean(), b.mean()
    
    # Avoid division by zero and extreme values
    r_scale = g_avg / max(r_avg, 1e-6)
    b_scale = g_avg / max(b_avg, 1e-6)
    
    # Apply gentle correction
    rgb[..., 0] = np.clip(r * min(r_scale, 2.0), 0, 1)
    rgb[..., 2] = np.clip(b * min(b_scale, 2.0), 0, 1)
    
    return rgb

# ------------------------------
# Enhanced SID Dataset
# ------------------------------
class EnhancedLoadDataSID(Dataset):
    def __init__(self, short_expo_files, long_expo_files, patch_size=512, 
                 training=True, bayer_pattern="RGGB", use_pattern_detection=True):
        self.training = training
        self.patch_size = patch_size
        self.short_expo_files = short_expo_files
        self.long_expo_files = long_expo_files
        self.bayer_pattern = bayer_pattern
        self.use_pattern_detection = use_pattern_detection

        if self.training:
            print("\n...... Enhanced Train files loading\n")
        else:
            print("\n...... Enhanced Test files loading\n")

        # Preload all images into RAM
        self.short_list = []
        self.long_list = []
        self.pattern_list = []

        for i in tqdm.tqdm(range(len(short_expo_files))):
            short_path = short_expo_files[i]
            long_path = long_expo_files[i]
            
            # Load short exposure RAW
            raw_short = rawpy.imread(short_path)
            
            # Detect pattern if enabled
            if self.use_pattern_detection:
                detected_pattern = detect_bayer_pattern_from_metadata(raw_short)
                actual_pattern = detected_pattern if detected_pattern else bayer_pattern
            else:
                actual_pattern = bayer_pattern
                
            self.pattern_list.append(actual_pattern)
            
            # Pack RAW with correct pattern
            im_short = pack_raw(raw_short, actual_pattern)
            raw_short.close()
            self.short_list.append(im_short)

            # Load long exposure RGB GT
            raw_long = rawpy.imread(long_path)
            im_long = raw_long.postprocess(
                use_camera_wb=True, 
                half_size=False,  # Changed to False for better alignment
                no_auto_bright=True, 
                output_bps=16
            ).copy()
            raw_long.close()
            
            # Convert and correct
            im_long = im_long.astype(np.float32) / 65535.0
            im_long = correct_bayer_channels(im_long, actual_pattern, auto_detect=True)
            im_long = auto_correct_color_balance(im_long)
            self.long_list.append(im_long)

        print(f"\nFiles loaded: {len(self.short_list)} samples")
        print(f"Detected patterns: {set(self.pattern_list)}\n")

    def __len__(self):
        return len(self.short_list)

    def __getitem__(self, idx):
        im_short = self.short_list[idx].copy()
        im_long = self.long_list[idx].copy()

        # Exposure scaling using filename
        filename = str(self.long_expo_files[idx])
        ap = 300 if '100' not in filename else 100  # More robust filename parsing
        im_short = im_short * ap

        C, H, W = im_short.shape
        H_rgb, W_rgb, _ = im_long.shape

        # Ensure RGB and RAW have compatible dimensions
        if H != H_rgb or W != W_rgb:
            # Resize RGB to match RAW dimensions
            im_long = np.transpose(im_long, (2, 0, 1))  # (3, H, W)
            im_long = torch.from_numpy(im_long).unsqueeze(0)
            im_long = torch.nn.functional.interpolate(
                im_long, size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).numpy()
            im_long = np.transpose(im_long, (1, 2, 0))  # (H, W, 3)

        # Random crop & augmentation
        if self.training and H >= self.patch_size and W >= self.patch_size:
            i = random.randint(0, H - self.patch_size)
            j = random.randint(0, W - self.patch_size)
            im_short = im_short[:, i:i+self.patch_size, j:j+self.patch_size]
            im_long = im_long[i:i+self.patch_size, j:j+self.patch_size, :]

            # Random flips
            if random.random() > 0.5:
                im_short = np.flip(im_short, axis=2).copy()  # horizontal flip
                im_long = np.flip(im_long, axis=1).copy()
            if random.random() < 0.2:
                im_short = np.flip(im_short, axis=1).copy()  # vertical flip
                im_long = np.flip(im_long, axis=0).copy()

        # Convert to tensors
        im_short_tensor = torch.from_numpy(im_short).float()  # (4, H, W)
        im_long_tensor = torch.from_numpy(np.transpose(im_long, (2, 0, 1))).float()  # (3, H, W)

        return im_short_tensor, im_long_tensor

# ------------------------------
# Debug and test functions
# ------------------------------
def diagnose_image(image_path):
    """Diagnose raw image metadata"""
    try:
        raw = rawpy.imread(image_path)
        print(f"\nDiagnosing: {os.path.basename(image_path)}")
        print(f"Color description: {getattr(raw, 'color_desc', 'N/A')}")
        if hasattr(raw, 'color_desc'):
            try:
                print(f"Color desc decoded: {raw.color_desc.decode('utf-8', errors='ignore')}")
            except:
                pass
        print(f"Raw pattern: {getattr(raw, 'raw_pattern', 'N/A')}")
        print(f"Black levels: {getattr(raw, 'black_level_per_channel', 'N/A')}")
        print(f"White level: {getattr(raw, 'white_level', 'N/A')}")
        raw.close()
    except Exception as e:
        print(f"Error diagnosing {image_path}: {e}")

def test_enhanced_loader(short_files, long_files, save_dir="./debug_out"):
    """Test the enhanced loader with visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Diagnose images first
    for short_file, long_file in zip(short_files, long_files):
        print("=" * 50)
        diagnose_image(short_file)
        diagnose_image(long_file)
    
    # Load dataset
    dataset = EnhancedLoadDataSID(
        short_files, long_files, 
        patch_size=512, 
        training=True,
        use_pattern_detection=True
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (raw_pack, gt_rgb) in enumerate(loader):
        raw_np = raw_pack[0].numpy()  # (4, H, W)
        gt_np = gt_rgb[0].numpy().transpose(1, 2, 0)  # (H, W, 3)

        # Visualize all 4 RAW channels
        for ch in range(4):
            channel_img = raw_np[ch] / raw_np[ch].max() * 255
            imageio.imwrite(f"{save_dir}/raw_ch{ch}_{idx}.png", channel_img.astype(np.uint8))
        
        # Save RGB
        imageio.imwrite(f"{save_dir}/gt_{idx}.png", (gt_np * 255).astype(np.uint8))
        
        print(f"Saved debug images for sample {idx}")
        if idx >= 2:  # Just check a couple of samples
            break

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Example file paths
    short_files = ["./dataset/Sony/short_expo/0001.ARW", "./dataset/Sony/short_expo/0002.ARW"]
    long_files = ["./dataset/Sony/long_expo/0001.ARW", "./dataset/Sony/long_expo/0002.ARW"]
    
    # Filter out non-existent files
    short_files = [f for f in short_files if os.path.exists(f)]
    long_files = [f for f in long_files if os.path.exists(f)]
    
    if short_files and long_files:
        test_enhanced_loader(short_files, long_files)
    else:
        print("No valid files found. Please check your file paths.")
