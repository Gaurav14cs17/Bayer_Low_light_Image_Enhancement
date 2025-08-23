import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import numpy as np
import os
import tqdm
import imageio
import glob
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID

# ------------------------------
# Bayer CFA handling
# ------------------------------
def correct_bayer_channels(rgb, pattern="RGGB"):
    """
    Ensure correct RGB channel order based on Bayer pattern.
    """
    pattern = pattern.upper()
    if pattern == "BGGR":
        rgb = rgb[..., [2, 1, 0]]  # Swap R and B
    elif pattern == "GBRG":
        rgb = rgb[..., [1, 0, 2]]  # Adjust mapping
    elif pattern == "GRBG":
        rgb = rgb[..., [0, 2, 1]]  # Adjust mapping
    # RGGB → keep as is
    return rgb

def auto_correct_rb(rgb):
    """
    Automatically swaps R/B if the red channel is darker than blue.
    Useful for natural images where R is normally stronger than B.
    """
    r_mean = rgb[..., 0].mean()
    b_mean = rgb[..., 2].mean()
    if r_mean < b_mean:
        rgb = rgb[..., [2, 1, 0]]
    return rgb

# ------------------------------
# Main testing pipeline
# ------------------------------
if __name__ == '__main__':
    opt = {}
    opt['dataset'] = 'MCR'           # 'MCR' or 'SID'
    opt['use_gpu'] = True
    opt['gpu_id'] = '0'
    opt['model_size'] = 'S'          # S / B / L
    opt['bayer_pattern'] = "RGGB"    # Dataset Bayer pattern

    save_weights_file = os.path.join('result', opt['dataset'], 'weights')
    save_images_file = os.path.join('result', opt['dataset'], 'images')
    save_csv_file = os.path.join('result', opt['dataset'], 'csv')

    # ------------------------------
    # Load test dataset
    # ------------------------------
    if opt['dataset'] == 'SID':
        test_input_paths = glob.glob(os.path.join('Sony/short/', '*.ARW'))
        test_gt_paths = [x.replace('short', 'long') for x in test_input_paths]
        print(f'Test data: {len(test_input_paths)} pairs')
        test_data = load_data_SID(test_input_paths, test_gt_paths, training=False)

    elif opt['dataset'] == 'MCR':
        test_c_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy', allow_pickle=True)
        test_rgb_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy', allow_pickle=True)
        print(f'Test data: {len(test_c_path)} pairs')
        test_data = load_data_MCR(test_c_path.tolist(), test_rgb_path.tolist(), training=False)

    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)

    # ------------------------------
    # Device setup
    # ------------------------------
    device = torch.device("cuda" if opt['use_gpu'] and torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
        torch.cuda.empty_cache()

    # ------------------------------
    # Model setup
    # ------------------------------
    dim = {'S': 32, 'B': 48, 'L': 64}[opt['model_size']]
    model = RawFormer(dim=dim).to(device)

    checkpoint_path = os.path.join(save_weights_file, f'RawFormer_{opt["model_size"]}_{opt["dataset"]}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get('epoch', 0)
    print('Loaded model from epoch:', epoch)

    model.eval()

    # ------------------------------
    # Testing loop
    # ------------------------------
    psnr_val_rgb = []
    ssim_val_rgb = []

    os.makedirs(save_images_file, exist_ok=True)
    os.makedirs(save_csv_file, exist_ok=True)

    with torch.no_grad():
        for ii, (inp_tensor, gt_tensor) in enumerate(tqdm.tqdm(dataloader_test)):
            inp_tensor = inp_tensor.to(device)

            # Ground truth
            rgb_gt = (gt_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rgb_gt = correct_bayer_channels(rgb_gt, opt['bayer_pattern'])
            rgb_gt = auto_correct_rb(rgb_gt)

            # Model prediction
            pred_rgb = model(inp_tensor)
            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            pred_rgb = (pred_rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred_rgb = correct_bayer_channels(pred_rgb, opt['bayer_pattern'])
            pred_rgb = auto_correct_rb(pred_rgb)

            # Metrics
            psnr = PSNR(pred_rgb, rgb_gt)
            ssim = SSIM(pred_rgb, rgb_gt, channel_axis=-1)
            print(f'image:{ii}\tPSNR:{psnr:.4f}\tSSIM:{ssim:.4f}')
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            # Save images
            imageio.imwrite(os.path.join(save_images_file, f'e{epoch}_{ii}_gt.jpg'), rgb_gt)
            imageio.imwrite(os.path.join(save_images_file, f'e{epoch}_{ii}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.jpg'), pred_rgb)

    # ------------------------------
    # Average metrics
    # ------------------------------
    psnr_average = np.mean(psnr_val_rgb)
    ssim_average = np.mean(ssim_val_rgb)
    print(f"Average PSNR: {psnr_average:.4f}, Average SSIM: {ssim_average:.4f}")

    # Save CSV
    np.savetxt(os.path.join(save_csv_file, 'test_metrics.csv'),
               np.column_stack((psnr_val_rgb, ssim_val_rgb)),
               delimiter=',', fmt='%.4f')
