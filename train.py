import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import tqdm
import glob
import time
import datetime
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID

# Configuration
config = {
    'gpu_id': '0',
    'base_lr': 1e-4,
    'batch_size': 16,
    'dataset': 'SID',  # 'SID' or 'MCR'
    'patch_size': 512,
    'model_size': 'S',  # 'S', 'B', or 'L'
    'epochs': 3000,
    'warmup_epochs': 20,
    'min_lr': 1e-5
}

# Setup directories
result_dir = os.path.join('result', config['dataset'])
os.makedirs(result_dir, exist_ok=True)
weights_dir = os.path.join(result_dir, 'weights')
os.makedirs(weights_dir, exist_ok=True)
logs_dir = os.path.join(result_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model initialization
model_dim = {'S': 32, 'B': 48, 'L': 64}[config['model_size']]
model = RawFormer(dim=model_dim).to(device)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTrainable parameters: {trainable_params:,}\nTotal parameters: {total_params:,}\n')

# Dataset loading
def load_dataset():
    if config['dataset'] == 'SID':
        train_input = glob.glob('Sony/short/0*_00_0.1s.ARW') + glob.glob('Sony/short/2*_00_0.1s.ARW')
        train_gt = [glob.glob(f'Sony/long/*{x[-17:-12]}*.ARW')[0] for x in train_input]
        test_input = glob.glob('Sony/short/1*_00_0.1s.ARW')
        test_gt = [glob.glob(f'Sony/long/*{x[-17:-12]}*.ARW')[0] for x in test_input]
        
        train_data = load_data_SID(train_input, train_gt, config['patch_size'], True)
        test_data = load_data_SID(test_input, test_gt, config['patch_size'], False)
    else:
        pass
    
    return train_data, test_data

train_data, test_data = load_dataset()
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
print(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

# Training setup
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=config['base_lr'])

# Learning rate scheduling
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config['epochs']-config['warmup_epochs'], eta_min=config['min_lr'])
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=config['warmup_epochs'], after_scheduler=scheduler_cosine)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# TensorBoard writer
writer = SummaryWriter(log_dir=logs_dir)

# Training loop
best_psnr = 0
best_epoch = 0

print("Starting training at:", datetime.datetime.now().isoformat())
for epoch in range(config['epochs']):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0
    
    # Training phase
    for inputs, targets in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    
    scheduler.step()
    
    # Validation phase
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(test_loader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs = torch.clamp(outputs, 0, 1)
            
            # Convert to numpy and calculate PSNR
            outputs_np = (outputs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            targets_np = (targets.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            psnr_values.extend([PSNR(t, o) for t, o in zip(targets_np, outputs_np)])
    
    avg_psnr = np.mean(psnr_values)
    
    # Save best model
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(weights_dir, "model_best.pth"))
    
    # Logging
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_lr()[0]
    
    print(f"\nEpoch {epoch}:")
    print(f"Time: {epoch_time:.2f}s | Loss: {epoch_loss/len(train_loader):.4f}")
    print(f"PSNR: {avg_psnr:.4f} | Best PSNR: {best_psnr:.4f} (epoch {best_epoch})")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # TensorBoard logging
    writer.add_scalar('Loss/train', epoch_loss/len(train_loader), epoch)
    writer.add_scalar('PSNR/val', avg_psnr, epoch)
    writer.add_scalar('LR', current_lr, epoch)

# Save final model
torch.save({
    'epoch': config['epochs'],
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, os.path.join(weights_dir, f"model_final.pth"))

print("\nTraining completed at:", datetime.datetime.now().isoformat())
print(f'Best PSNR: {best_psnr:.4f} achieved at epoch {best_epoch}')
print(f'Models saved in: {weights_dir}')
