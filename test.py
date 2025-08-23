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
    Ensures correct RGB channel order based on Bayer pattern.
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
    tb_log_dir = os.path.join('result', opt['dataset'], 'logs')

    # Load test dataset
    if opt['dataset'] == 'SID':
        test_input_paths = glob.glob(os.path.join('Sony/short/1*_00_0.1s.ARW'))
        test_gt_paths = []
        for x in test_input_paths:
            test_gt_paths += glob.glob(os.path.join('Sony/long/*' + x[-17:-12] + '*.ARW'))
        print('test data: %d pairs' % len(test_input_paths))
        test_data = load_data_SID(test_input_paths, test_gt_paths, training=False)

    elif opt['dataset'] == 'MCR':
        test_c_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy')
        test_rgb_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy')
        print('test data: %d pairs' % len(test_c_path))
        test_data = load_data_MCR(test_c_path, test_rgb_path, training=False)

    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)

    # Device setup
    if opt['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # Model dimension
    if opt['model_size'] == 'S':
        dim = 32
    elif opt['model_size'] == 'B':
        dim = 48
    else:
        dim = 64

    # Load model
    model = RawFormer(dim=dim)
    checkpoint = torch.load(
        os.path.join(save_weights_file, f'RawFormer_{opt["model_size"]}_{opt["dataset"]}.pth'),
        map_location=device
    )

    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
        strict=True
    )
    epoch = checkpoint['epoch']
    print('Loaded model from epoch:', epoch)

    model = model.to(device)
    model.eval()
    print('Device on cuda:', next(model.parameters()).is_cuda)

    # ------------------------------
    # Test loop
    # ------------------------------
    psnr_val_rgb = []
    ssim_val_rgb = []

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm.tqdm(dataloader_test)):
            # Ground truth
            rgb_gt = (data_test[1].numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            rgb_gt = correct_bayer_channels(rgb_gt, opt['bayer_pattern'])
            rgb_gt = auto_correct_rb(rgb_gt)

            # Model prediction
            input_raw = data_test[0].to(device)
            pred_rgb = model(input_raw)
            pred_rgb = (torch.clamp(pred_rgb, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred_rgb = correct_bayer_channels(pred_rgb, opt['bayer_pattern'])
            pred_rgb = auto_correct_rb(pred_rgb)

            # Metrics
            psnr = PSNR(pred_rgb, rgb_gt)
            ssim = SSIM(pred_rgb, rgb_gt, channel_axis=-1)
            print(f'image:{ii}\tPSNR:{psnr:.4f}\tSSIM:{ssim:.4f}')
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            # Save images
            os.makedirs(save_images_file, exist_ok=True)
            imageio.imwrite(os.path.join(save_images_file, f'e{epoch}_{ii}_gt.jpg'), rgb_gt)
            imageio.imwrite(os.path.join(save_images_file, f'e{epoch}_{ii}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.jpg'), pred_rgb)

    # Average metrics
    psnr_average = sum(psnr_val_rgb) / len(dataloader_test)
    ssim_average = sum(ssim_val_rgb) / len(dataloader_test)
    print("average_PSNR: %.4f, average_SSIM: %.4f" % (psnr_average, ssim_average))

    # Save CSV
    os.makedirs(save_csv_file, exist_ok=True)
    np.savetxt(os.path.join(save_csv_file, 'test_metrics.csv'),
               [p for p in zip(psnr_val_rgb, ssim_val_rgb)], delimiter=',', fmt='%s')
