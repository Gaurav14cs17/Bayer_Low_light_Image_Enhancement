import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import os
import tqdm
import glob
import time
import datetime
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

if __name__ == '__main__':
    opt = {
        'gpu_id': '0',
        'base_lr': 1e-4,
        'batch_size': 16,
        'dataset': 'SID',   # 'SID' or 'MCR'
        'patch_size': 512,
        'model_size': 'S',  # S/B/L
        'epochs': 3000,
        'train_sid_short': "Sony/short/0*_00_0.1s.ARW",  # training short exposure
        'train_sid_long': "Sony/long/",                  # training long exposure folder
        'test_sid_short': "Sony/short/1*_00_0.1s.ARW",   # testing short exposure
        'test_sid_long': "Sony/long/",                  # testing long exposure folder
        'train_mcr_c': "Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_c_path.npy",
        'train_mcr_rgb': "Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_rgb_path.npy",
        'test_mcr_c': "Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy",
        'test_mcr_rgb': "Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy",
        'save_weights_file': "result/SID/weights",
        'save_images_file': "result/SID/images",
        'save_csv_file': "result/SID/csv",
        'log_file': "result/SID/log.txt"
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folders if not exist
    os.makedirs(opt['save_weights_file'], exist_ok=True)
    os.makedirs(opt['save_images_file'], exist_ok=True)
    os.makedirs(opt['save_csv_file'], exist_ok=True)
    os.makedirs(os.path.dirname(opt['log_file']), exist_ok=True)

    log_f = open(opt['log_file'], 'a')
    log_f.write(f"\nTraining start time: {datetime.datetime.now().isoformat()}\n")

    use_pretrain = False
    pretrain_weights = os.path.join(opt['save_weights_file'], 'model_2000.pth')

    # ------------------------------
    # Dataset
    # ------------------------------
    if opt['dataset'] == 'SID':
        train_input_paths = glob.glob(opt['train_sid_short'].replace("\\", "/")) + glob.glob("Sony/short/2*_00_0.1s.ARW")
        train_gt_paths = [os.path.join(opt['train_sid_long'], os.path.basename(x).replace('short', 'long')) for x in train_input_paths]

        test_input_paths = glob.glob(opt['test_sid_short'])
        test_gt_paths = [os.path.join(opt['test_sid_long'], os.path.basename(x).replace('short', 'long')) for x in test_input_paths]

        train_data = load_data_SID(train_input_paths, train_gt_paths, patch_size=opt['patch_size'], training=True)
        test_data = load_data_SID(test_input_paths, test_gt_paths, patch_size=opt['patch_size'], training=False)

    elif opt['dataset'] == 'MCR':
        train_c_path = np.load(opt['train_mcr_c'], allow_pickle=True).tolist()
        train_rgb_path = np.load(opt['train_mcr_rgb'], allow_pickle=True).tolist()
        test_c_path = np.load(opt['test_mcr_c'], allow_pickle=True).tolist()
        test_rgb_path = np.load(opt['test_mcr_rgb'], allow_pickle=True).tolist()

        train_data = load_data_MCR(train_c_path, train_rgb_path, patch_size=opt['patch_size'], training=True)
        test_data = load_data_MCR(test_c_path, test_rgb_path, patch_size=opt['patch_size'], training=False)

    dataloader_train = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
    dataloader_val = DataLoader(test_data, batch_size=opt['batch_size'], shuffle=False, num_workers=16, pin_memory=True)

    # ------------------------------
    # Model
    # ------------------------------
    dim = {'S': 32, 'B': 48, 'L': 64}[opt['model_size']]
    model = RawFormer(dim=dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt['base_lr'])
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt['epochs'], eta_min=1e-5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=scheduler_cosine)

    scaler = torch.cuda.amp.GradScaler()
    loss_criterion = CharbonnierLoss()

    start_epoch = 0
    best_psnr = 0
    best_epoch = 0

    # ------------------------------
    # Training loop
    # ------------------------------
    for epoch in range(start_epoch, opt['epochs'] + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch in tqdm.tqdm(dataloader_train):
            optimizer.zero_grad()
            input_raw = batch[0].to(device)
            gt_rgb = batch[1].to(device)

            with torch.cuda.amp.autocast():
                pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb, 0, 1)
                loss = loss_criterion(pred_rgb, gt_rgb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        # ------------------------------
        # Validation
        # ------------------------------
        model.eval()
        psnr_val_rgb = []
        with torch.no_grad():
            for batch in dataloader_val:
                input_raw = batch[0].to(device)
                gt_rgb = batch[1].to(device)
                with torch.cuda.amp.autocast():
                    pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb, 0, 1)
                pred_np = (pred_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                gt_np = (gt_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                for p, g in zip(pred_np, gt_np):
                    psnr_val_rgb.append(PSNR(g, p))

        avg_psnr = np.mean(psnr_val_rgb)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(opt['save_weights_file'], "model_best.pth"))

        epoch_time = time.time() - start_time
        log_f.write(f"Epoch {epoch}/{opt['epochs']} | Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f} | Avg PSNR: {avg_psnr:.4f} | Best PSNR: {best_psnr:.4f} (Epoch {best_epoch})\n")
        log_f.flush()

        if epoch % 50 == 0 or epoch == opt['epochs']:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(opt['save_weights_file'], f"model_{epoch}.pth"))

    log_f.write(f"Training finished at: {datetime.datetime.now().isoformat()}\n")
    log_f.close()
