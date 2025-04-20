import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import numpy as np
import os
from tqdm import tqdm
import imageio
import glob
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID

def setup_device(use_gpu=True, gpu_id='0'):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    return device

def load_test_data(dataset):
    if dataset == 'SID':
        test_input_paths = glob.glob('Sony/short/1*_00_0.1s.ARW')
        test_gt_paths = [glob.glob(f'Sony/long/*{x[-17:-12]}*.ARW')[0] for x in test_input_paths]
        print(f'Test data: {len(test_input_paths)} pairs')
        return load_data_SID(test_input_paths, test_gt_paths, training=False)
    
    elif dataset == 'MCR':
      pass

def get_model_size(size):
    sizes = {'S': 32, 'B': 48, 'L': 64}
    return sizes.get(size, 64)

def save_results(epoch, save_dir, idx, rgb_gt, pred_rgb, psnr, ssim):
    os.makedirs(save_dir, exist_ok=True)
    imageio.imwrite(f'{save_dir}/e{epoch}_{idx}_gt.jpg', rgb_gt)
    imageio.imwrite(f'{save_dir}/e{epoch}_{idx}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.jpg', pred_rgb)

def main():
    config = {
        'dataset': 'MCR',
        'use_gpu': True,
        'gpu_id': '0',
        'model_size': 'S'  # S/B/L for small/base/large
    }
    
    # Setup paths
    base_dir = os.path.join('result', config['dataset'])
    paths = {
        'weights': os.path.join(base_dir, 'weights'),
        'images': os.path.join(base_dir, 'images'),
        'csv': os.path.join(base_dir, 'csv'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Load data and model
    test_data = load_test_data(config['dataset'])
    dataloader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    device = setup_device(config['use_gpu'], config['gpu_id'])
    
    model = RawFormer(dim=get_model_size(config['model_size']))
    model_path = f"{paths['weights']}/RawFormer_{config['model_size']}_{config['dataset']}.pth"
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    print(f'Loaded model from epoch: {checkpoint["epoch"]}')
    
    model = model.to(device)
    model.eval()
    print(f'Device on cuda: {next(model.parameters()).is_cuda}')

    # Evaluation
    psnr_values, ssim_values = [], []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            rgb_gt = (data[1].numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred_rgb = model(data[0].to(device))
            pred_rgb = (torch.clamp(pred_rgb, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            
            psnr = PSNR(pred_rgb, rgb_gt)
            ssim = SSIM(pred_rgb, rgb_gt, multichannel=True)
            
            print(f'Image: {idx}\tPSNR: {psnr:.4f}\tSSIM: {ssim:.4f}')
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            save_results(checkpoint['epoch'], paths['images'], idx, rgb_gt, pred_rgb, psnr, ssim)
    
    # Save metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f'Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}')
    
    os.makedirs(paths['csv'], exist_ok=True)
    np.savetxt(f"{paths['csv']}/test_metrics.csv", 
              np.column_stack((psnr_values, ssim_values)), 
              delimiter=',', fmt='%.4f')

if __name__ == '__main__':
    main()
