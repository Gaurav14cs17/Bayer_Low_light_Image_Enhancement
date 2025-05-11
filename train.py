import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import glob
import time
import datetime
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

# Configuration
config = {
    'gpu_ids': '0,1,2,3',          # Comma-separated GPU IDs
    'base_lr': 2e-4,                # Scaled up for multi-GPU
    'batch_size': 16,               # Per GPU batch size
    'image_size': (80, 400),        # Output image dimensions
    'font_size': 34,                # Base font size
    'epochs': 1000,
    'warmup_epochs': 20,
    'min_lr': 1e-5,
    'num_workers': 8,
    'distributed': True,            # Enable DDP
    'local_rank': 0                 # Will be set by torch.distributed.launch
}

# Initialize distributed training
if 'WORLD_SIZE' in os.environ:
    config['distributed'] = int(os.environ['WORLD_SIZE']) > 1

if config['distributed']:
    torch.cuda.set_device(config['local_rank'])
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()

# Setup directories
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)
weights_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(weights_dir, exist_ok=True)
logs_dir = os.path.join(output_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_ids']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.font_size = config['font_size']
        self.text_color = (0, 0, 0)  # Black text
        self.bg_color = (255, 255, 255)  # White background
        
    def forward(self, text_batch):
        images = []
        for text in text_batch:
            img = Image.new('RGB', config['image_size'], self.bg_color)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arialbd.ttf", self.font_size)
            except:
                font = ImageFont.load_default()
            
            # Format: 4 letters + space + 7 digits
            parts = [text[:4], ' ', text[4:11]] if len(text) >= 11 else [text]
            y_pos = 20
            
            for part in parts:
                for char in part:
                    w, h = draw.textsize(char, font=font)
                    x_pos = 5 + (config['image_size'][0] - 10 - w) // 2
                    draw.text((x_pos, y_pos), char, fill=self.text_color, font=font)
                    y_pos += h + (25 if part == text[:4] and char == part[-1] else 10)
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
            images.append(img_tensor)
        
        return torch.stack(images).to(device)

# Initialize model
model = ImageGenerator().to(device)

# Multi-GPU setup
if config['distributed']:
    model = DDP(model, device_ids=[config['local_rank']])
elif torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000):
        self.texts = []
        for _ in range(num_samples):
            letters = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4))
            digits = ''.join(random.choices("0123456789", k=7))
            self.texts.append(f"{letters}{digits}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# Training setup
dataset = TextDataset(num_samples=10000)
sampler = DistributedSampler(dataset) if config['distributed'] else None

loader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=(sampler is None),
    num_workers=config['num_workers'],
    pin_memory=True,
    sampler=sampler
)

# Loss and optimizer
criterion = nn.MSELoss()  # Using MSE for demonstration
optimizer = optim.Adam(model.parameters(), lr=config['base_lr'])

# Learning rate scheduling
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config['epochs']-config['warmup_epochs'], eta_min=config['min_lr'])
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=config['warmup_epochs'], after_scheduler=scheduler_cosine)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training loop
def train():
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, disable=config['local_rank'] != 0):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            generated = model(batch)
            # For demonstration - compare to white images
            targets = torch.ones_like(generated)
            loss = criterion(generated, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if config['distributed']:
            dist.all_reduce(loss)
            loss = loss / world_size
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Main execution
if __name__ == "__main__":
    if config['local_rank'] == 0:
        writer = SummaryWriter(log_dir=logs_dir)
        print(f"Training on {world_size if config['distributed'] else torch.cuda.device_count()} GPUs")
    
    for epoch in range(config['epochs']):
        if config['distributed']:
            sampler.set_epoch(epoch)
        
        epoch_loss = train()
        scheduler.step()
        
        if config['local_rank'] == 0:
            # Save sample images
            if epoch % 10 == 0:
                with torch.no_grad():
                    sample_texts = ["REBOD1234567", "ABCD9876543", "TEST0000000"]
                    samples = model(sample_texts)
                    for i, (text, img_tensor) in enumerate(zip(sample_texts, samples)):
                        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img).save(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png"))
            
            # Save checkpoint
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(weights_dir, f"checkpoint_{epoch}.pth"))
            
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}")
    
    if config['distributed']:
        dist.destroy_process_group()
