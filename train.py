import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Simple metric tracker
class MetricTracker:
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


# Loss function wrapper
class MultiLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, preds, targets):
        total = 0
        for loss, w, p, t in zip(self.losses, self.weights, preds, targets):
            total += loss(p, t) * w
        return total


# Image quality metrics
def psnr(pred, target, max_val=255.0):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(pred, target, max_val=255.0, window_size=11):
    # Simple SSIM implementation (for actual use, consider a proper implementation)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)

    sigma_x = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean(dim=[1, 2, 3])


# Main training class
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup directories
        os.makedirs(config['output'], exist_ok=True)
        os.makedirs(os.path.join(config['output'], 'checkpoints'), exist_ok=True)

        # Initialize components
        self.model = self._init_model().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.criterion = MultiLoss(
            [nn.MSELoss(), nn.L1Loss()],
            config['loss_weights']
        )
        self.writer = SummaryWriter(os.path.join(config['output'], 'logs'))

        print(f"Training on {self.device}")

    def _init_model(self):
        # Replace with actual model initialization
        return nn.Identity()  # Dummy model

    def train(self, train_loader, val_loader, epochs):
        best_psnr = 0

        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_loss = MetricTracker()

            for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
                inputs = batch['input'].to(self.device)
                targets = [batch['target1'].to(self.device), batch['target2'].to(self.device)]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss.update(loss.item(), inputs.size(0))

            # Validation phase
            val_psnr, val_ssim = self.validate(val_loader)

            # Save checkpoint
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                self.save_checkpoint(epoch, best=True)

            self.save_checkpoint(epoch)

            # Log metrics
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss.avg:.4f}, "
                  f"Val PSNR: {val_psnr:.2f}, "
                  f"Val SSIM: {val_ssim:.4f}")

            self.writer.add_scalar('Loss/train', train_loss.avg, epoch)
            self.writer.add_scalar('PSNR/val', val_psnr, epoch)
            self.writer.add_scalar('SSIM/val', val_ssim, epoch)

    def validate(self, loader):
        self.model.eval()
        psnr_tracker = MetricTracker()
        ssim_tracker = MetricTracker()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)[0]  # Get first output
                outputs = torch.clamp(outputs, 0, 1) * 255
                targets = torch.clamp(targets, 0, 1) * 255

                psnr_tracker.update(psnr(outputs, targets).mean().item())
                ssim_tracker.update(ssim(outputs, targets).mean().item())

        return psnr_tracker.avg, ssim_tracker.avg

    def save_checkpoint(self, epoch, best=False):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if best:
            torch.save(state, os.path.join(self.config['output'], 'checkpoints', 'best.pth'))
        torch.save(state, os.path.join(self.config['output'], 'checkpoints', 'latest.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, help='Path to config file')
    args = parser.parse_args()

    # Load config (in practice, use proper config loading)
    config = {
        'output': 'output',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'loss_weights': [1.0, 0.5],
        'epochs': 100
    }

    # Initialize trainer
    trainer = Trainer(config)

    # Dummy data loaders - replace with actual ones
    train_loader = [{'input': torch.rand(4, 3, 256, 256),
                     'target1': torch.rand(4, 3, 256, 256),
                     'target2': torch.rand(4, 3, 256, 256)}] * 100
    val_loader = [{'input': torch.rand(4, 3, 256, 256),
                   'target': torch.rand(4, 3, 256, 256)}] * 20

    # Start training
    trainer.train(train_loader, val_loader, config['epochs'])


if __name__ == '__main__':
    main()
