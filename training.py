import os
import time
import glob
import random
import datetime
import numpy as np
import imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from warmup_scheduler import GradualWarmupScheduler

import rawpy
from model import RawFormer  # Ensure RawFormer is implemented correctly


# -------------------------------
# Utility Classes
# -------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -------------------------------
# SID Dataset
# -------------------------------
class SID_Dataset(Dataset):
    """SID RAW Dataset with proper memory handling"""
    def __init__(self, short_files, long_files, patch_size=512, training=True, gt_png=False):
        self.short_files = short_files
        self.long_files = long_files
        self.patch_size = patch_size
        self.training = training
        self.gt_png = gt_png
        print(f"\n...... {'Train' if training else 'Test'} files initialized ({len(short_files)} pairs)\n")

    def __len__(self):
        return len(self.short_files)

    @staticmethod
    def pack_raw(raw):
        """Pack RAW into 4 channels using per-channel black level"""
        im = raw.raw_image_visible.astype(np.float32)
        black = np.array(raw.black_level_per_channel)[:, None, None]
        white = raw.white_level
        im = (im - black.min()) / (white - black.min())
        im = np.clip(im, 0, 1)
        im = np.expand_dims(im, axis=2)
        H, W, _ = im.shape
        out = np.concatenate((
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :]
        ), axis=2)
        return out

    def __getitem__(self, idx):
        # --- Load short exposure ---
        with rawpy.imread(self.short_files[idx]) as raw:
            filename = os.path.basename(self.short_files[idx])
            try:
                short_exp = float(filename.split('_')[-1].replace('s.ARW',''))
            except:
                short_exp = 0.1
            with rawpy.imread(self.long_files[idx]) as raw_gt:
                long_filename = os.path.basename(self.long_files[idx])
                try:
                    long_exp = float(long_filename.split('_')[-1].replace('s.ARW',''))
                except:
                    long_exp = 1.0
            ratio = min(long_exp / short_exp, 300)
            img_short = self.pack_raw(raw) * ratio

        # --- Load GT ---
        if self.gt_png:
            img_long = np.array(Image.open(self.long_files[idx]), dtype=np.float32) / 255.0
        else:
            with rawpy.imread(self.long_files[idx]) as raw_gt:
                img_long = raw_gt.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                img_long = np.float32(img_long / 65535.0)

        # Clamp
        img_short = np.clip(img_short, 0, 1)

        # Channel first
        img_short = img_short.transpose(2,0,1)
        img_long = img_long.transpose(2,0,1)

        H, W = img_short.shape[1:]

        # --- Augmentation ---
        if self.training and self.patch_size:
            yy = random.randint(0, (H - self.patch_size) // 2) * 2
            xx = random.randint(0, (W - self.patch_size) // 2) * 2
            img_short = img_short[:, yy:yy+self.patch_size, xx:xx+self.patch_size]

            if img_long.shape[1] == H*2:
                img_long = img_long[:, yy*2:(yy+self.patch_size)*2, xx*2:(xx+self.patch_size)*2]
            else:
                img_long = img_long[:, yy:yy+self.patch_size, xx:xx+self.patch_size]

            # Random flips
            if random.random() > 0.5:
                img_short = np.flip(img_short, axis=2)
                img_long = np.flip(img_long, axis=2)
            if random.random() > 0.5:
                img_short = np.flip(img_short, axis=1)
                img_long = np.flip(img_long, axis=1)
            if random.random() > 0.5:
                img_short = np.transpose(img_short, (0,2,1))
                img_long = np.transpose(img_long, (0,2,1))

        return torch.from_numpy(img_short).float(), torch.from_numpy(img_long).float()


# -------------------------------
# MCR Dataset
# -------------------------------
class MCR_Dataset(Dataset):
    def __init__(self, train_c_path, train_rgb_path, patch_size=512, training=True):
        self.train_c_path = train_c_path
        self.train_rgb_path = train_rgb_path
        self.patch_size = patch_size
        self.training = training
        print(f"\n...... {'Train' if training else 'Test'} files initialized ({len(train_c_path)} pairs)\n")

    def __len__(self):
        return len(self.train_c_path)

    def __getitem__(self, idx):
        inp_raw = imageio.imread(self.train_c_path[idx]).astype(np.float32)
        gt_rgb = imageio.imread(self.train_rgb_path[idx]).astype(np.float32)

        filename = os.path.basename(self.train_c_path[idx])
        img_num = int(filename.split('_')[0]) if filename.split('_')[0].isdigit() else 0
        img_expo = int(filename[-8:-4],16) if filename[-8:-4].isdigit() else 1023

        gt_expo = 12287 if img_num < 500 else 1023
        amp = gt_expo / max(img_expo,1)
        inp_raw = inp_raw / 255.0 * amp
        gt_rgb = gt_rgb / 255.0

        H, W = inp_raw.shape

        # --- Augmentation ---
        if self.training and self.patch_size:
            i = random.randint(0, (H - self.patch_size -2)//2)*2
            j = random.randint(0, (W - self.patch_size -2)//2)*2
            inp_raw = inp_raw[i:i+self.patch_size, j:j+self.patch_size]
            gt_rgb = gt_rgb[i:i+self.patch_size, j:j+self.patch_size, :]

            if random.random() > 0.5:
                inp_raw = np.fliplr(inp_raw)
                gt_rgb = np.fliplr(gt_rgb)
            if random.random() > 0.2:
                inp_raw = np.flipud(inp_raw)
                gt_rgb = np.flipud(gt_rgb)

        return torch.from_numpy(inp_raw).float().unsqueeze(0), torch.from_numpy(gt_rgb.transpose(2,0,1)).float()


# -------------------------------
# Training Script
# -------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff*diff + self.eps*self.eps))


def main(opt):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using {n_gpus} GPU(s)" if n_gpus>0 else "Using CPU")

    # Create folders
    os.makedirs(opt['save_weights_file'], exist_ok=True)
    os.makedirs(opt['save_images_file'], exist_ok=True)
    os.makedirs(opt['save_csv_file'], exist_ok=True)
    os.makedirs(os.path.dirname(opt['log_file']), exist_ok=True)
    log_f = open(opt['log_file'],'a')
    log_f.write(f"\nTraining start time: {datetime.datetime.now().isoformat()}\n")

    # Dataset
    if opt['dataset']=='SID':
        train_input_paths = glob.glob(opt['train_sid_short']) + glob.glob("Sony/short/2*_00_0.1s.ARW")
        train_gt_paths = [glob.glob(os.path.join(opt['train_sid_long'], '*' + x[-17:-12] + '*.ARW'))[0] for x in train_input_paths]
        test_input_paths = glob.glob(opt['test_sid_short'])
        test_gt_paths = [glob.glob(os.path.join(opt['test_sid_long'], '*' + x[-17:-12] + '*.ARW'))[0] for x in test_input_paths]
        train_data = SID_Dataset(train_input_paths, train_gt_paths, patch_size=opt['patch_size'], training=True)
        test_data = SID_Dataset(test_input_paths, test_gt_paths, patch_size=opt['patch_size'], training=False)
    else:
        train_c_path = np.load(opt['train_mcr_c'], allow_pickle=True).tolist()
        train_rgb_path = np.load(opt['train_mcr_rgb'], allow_pickle=True).tolist()
        test_c_path = np.load(opt['test_mcr_c'], allow_pickle=True).tolist()
        test_rgb_path = np.load(opt['test_mcr_rgb'], allow_pickle=True).tolist()
        train_data = MCR_Dataset(train_c_path, train_rgb_path, patch_size=opt['patch_size'], training=True)
        test_data = MCR_Dataset(test_c_path, test_rgb_path, patch_size=opt['patch_size'], training=False)

    dataloader_train = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    dataloader_val = DataLoader(test_data, batch_size=opt['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    # Model
    dim = {'S':32,'B':48,'L':64}[opt['model_size']]
    model = RawFormer(dim=dim)
    if n_gpus>1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt['base_lr'])
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt['epochs'], eta_min=1e-5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=scheduler_cos)
    scaler = torch.cuda.amp.GradScaler()
    loss_criterion = CharbonnierLoss()

    best_psnr = 0
    best_epoch = 0

    # Training loop
    for epoch in range(opt['epochs']+1):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(dataloader_train):
            optimizer.zero_grad()
            input_raw, gt_rgb = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast():
                pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb,0,1)
                loss = loss_criterion(pred_rgb, gt_rgb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for batch in dataloader_val:
                input_raw, gt_rgb = batch[0].to(device), batch[1].to(device)
                with torch.cuda.amp.autocast():
                    pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb,0,1)
                pred_np = pred_rgb.cpu().numpy().transpose(0,2,3,1)
                gt_np = gt_rgb.cpu().numpy().transpose(0,2,3,1)
                for p,g in zip(pred_np,gt_np):
                    psnr_list.append(PSNR(g,p,data_range=1.0))
        avg_psnr = np.mean(psnr_list)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},
                       os.path.join(opt['save_weights_file'],"model_best.pth"))

        epoch_time = time.time()-start_time
        log_f.write(f"Epoch {epoch}/{opt['epochs']} | Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f} | Avg PSNR: {avg_psnr:.4f} | Best PSNR: {best_psnr:.4f} (Epoch {best_epoch})\n")
        log_f.flush()

        if epoch%50==0 or epoch==opt['epochs']:
            torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},
                       os.path.join(opt['save_weights_file'],f"model_{epoch}.pth"))

    log_f.write(f"Training finished at: {datetime.datetime.now().isoformat()}\n")
    log_f.close()


if __name__=="__main__":
    opt = {
        'base_lr':1e-4,
        'batch_size':16,
        'dataset':'SID',           # 'SID' or 'MCR'
        'patch_size':512,
        'model_size':'S',          # S/B/L
        'epochs':3000,
        'train_sid_short': "Sony/short/0*_00_0.1s.ARW",
        'train_sid_long': "Sony/long/",
        'test_sid_short': "Sony/short/1*_00_0.1s.ARW",
        'test_sid_long': "Sony/long/",
        'train_mcr_c': "Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_c_path.npy",
        'train_mcr_rgb': "Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_rgb_path.npy",
        'test_mcr_c': "Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy",
        'test_mcr_rgb': "Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy",
        'save_weights_file': "result/SID/weights",
        'save_images_file': "result/SID/images",
        'save_csv_file': "result/SID/csv",
        'log_file': "result/SID/log.txt"
    }
    main(opt)
