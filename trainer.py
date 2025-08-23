import datetime
import math
import os
import shutil
import gc
import time

import numpy as np
import torch
from PIL import Image
import tqdm

import utils  # ensure AverageMeter, get_psnr, get_ssim are implemented here


class Trainer(object):

    def __init__(self, cmd, cuda, model, criterion, optimizer,
                 train_loader, val_loader, log_file, max_iter,
                 interval_validate=None, lr_scheduler=None,
                 checkpoint_dir=None, result_dir=None, use_camera_wb=False, print_freq=1):

        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()
        self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.iteration = 0

        self.max_iter = max_iter
        self.best_psnr = 0
        self.print_freq = print_freq

        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_file = log_file
        self.use_camera_wb = use_camera_wb

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

    def print_log(self, log_str):
        print(log_str)
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')

    def validate(self):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        psnr_meter = utils.AverageMeter()
        ssim_meter = utils.AverageMeter()

        training = self.model.training
        self.model.eval()

        end = time.time()
        for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='{} iteration={} epoch={}'.format('Valid' if self.cmd == 'train' else 'Test',
                                                       self.iteration, self.epoch), ncols=80, leave=False):

            gc.collect()
            if self.cuda:
                raws, targets = raws.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            with torch.no_grad():
                output = self.model(raws)
                targets = targets[:, :, :output.size(2), :output.size(3)]
                loss = self.criterion(output, targets)
                if torch.isnan(loss):
                    raise ValueError('loss is nan while validating')
                losses.update(loss.item(), targets.size(0))

            outputs_cpu = torch.clamp(output, 0, 1).cpu()
            targets_cpu = targets.cpu()

            for output_img, target_img, img_file in zip(outputs_cpu, targets_cpu, img_files):
                output_arr = (output_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                target_arr = (target_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                if self.result_dir and self.cmd == 'test':
                    fname_compare = os.path.join(self.result_dir, f"{os.path.basename(img_file)[:-4]}_compare.jpg")
                    temp = np.concatenate((target_arr, output_arr), axis=1)
                    Image.fromarray(temp).save(fname_compare)
                    fname_single = os.path.join(self.result_dir, f"{os.path.basename(img_file)[:-4]}_single.jpg")
                    Image.fromarray(output_arr).save(fname_single)

                _psnr = utils.get_psnr(output_arr, target_arr)
                psnr_meter.update(_psnr, 1)
                if self.cmd == 'test':
                    _ssim = utils.get_ssim(output_arr, target_arr)
                    ssim_meter.update(_ssim, 1)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                log_str = '{cmd}: [{0}/{1}/{loss.count:}]\tepoch: {epoch}\titer: {iteration}\t' \
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'PSNR: {psnr.val:.2f} ({psnr.avg:.2f})\t' \
                          'SSIM: {ssim.val:.4f} ({ssim.avg:.4f})'.format(
                    batch_idx, len(self.val_loader), cmd='Valid' if self.cmd == 'train' else 'Test',
                    epoch=self.epoch, iteration=self.iteration,
                    batch_time=batch_time, loss=losses, psnr=psnr_meter, ssim=ssim_meter)
                self.print_log(log_str)

        if self.cmd == 'train':
            is_best = psnr_meter.avg > self.best_psnr
            self.best_psnr = max(psnr_meter.avg, self.best_psnr)

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_psnr': self.best_psnr,
            }, checkpoint_file)
            if is_best:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            if training:
                self.model.train()

    def train_epoch(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        psnr_meter = utils.AverageMeter()

        self.model.train()
        self.optim.zero_grad()

        end = time.time()
        for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}, iter={}'.format(self.epoch, self.iteration), ncols=80, leave=False):

            self.iteration = batch_idx + self.epoch * len(self.train_loader)
            data_time.update(time.time() - end)

            gc.collect()

            if (self.iteration + 1) % self.interval_validate == 0:
                self.validate()

            if self.cuda:
                raws, targets = raws.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = self.model(raws)
            loss = self.criterion(outputs, targets)
            if torch.isnan(loss):
                raise ValueError('loss is nan while training')
            losses.update(loss.item(), targets.size(0))

            outputs_cpu = torch.clamp(outputs, 0, 1).cpu()
            targets_cpu = targets.cpu()
            for output_img, target_img, img_file in zip(outputs_cpu, targets_cpu, img_files):
                output_arr = (output_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                target_arr = (target_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                psnr_meter.update(utils.get_psnr(output_arr, target_arr), 1)

                if self.result_dir:
                    epoch_dir = os.path.join(self.result_dir, f"{self.epoch:04d}")
                    os.makedirs(epoch_dir, exist_ok=True)
                    fname = os.path.join(epoch_dir, f"{batch_idx:04d}_{os.path.basename(img_file)[:-4]}.jpg")
                    temp = np.concatenate((target_arr, output_arr), axis=1)
                    Image.fromarray(temp).save(fname)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if self.iteration % self.print_freq == 0:
                log_str = 'Train: [{0}/{1}/{loss.count:}]\tepoch: {epoch}\titer: {iteration}\t' \
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'PSNR: {psnr.val:.1f} ({psnr.avg:.1f})\tlr {lr:.6f}'.format(
                    batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    lr=self.optim.param_groups[0]['lr'],
                    batch_time=batch_time, data_time=data_time, loss=losses, psnr=psnr_meter)
                self.print_log(log_str)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break


class Validator(Trainer):

    def __init__(self, cmd, cuda, model, criterion, val_loader, log_file, result_dir=None, use_camera_wb=False, print_freq=1):
        super(Validator, self).__init__(cmd, cuda=cuda, model=model, criterion=criterion,
                                        val_loader=val_loader, log_file=log_file, print_freq=print_freq,
                                        optimizer=None, train_loader=None, max_iter=None,
                                        interval_validate=None, lr_scheduler=None,
                                        checkpoint_dir=None, result_dir=result_dir, use_camera_wb=use_camera_wb)

    def train(self):
        raise NotImplementedError
