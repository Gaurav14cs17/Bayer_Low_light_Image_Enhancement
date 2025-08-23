import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: PyTorch model
        fname: file name of parameters converted from a Caffe model (Pickle format)
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError(
                    f'While copying the parameter named {name}, '
                    f'dimensions in the model are {own_state[name].size()} '
                    f'but in checkpoint are {param.shape}.'
                )
        else:
            raise KeyError(f'unexpected key "{name}" in state_dict')

class AverageMeter:
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_psnr(im1, im2):
    """Compute PSNR between two images"""
    return psnr(im1, im2, data_range=255)

def get_ssim(im1, im2):
    """Compute SSIM between two images"""
    return ssim(im1, im2, data_range=255, gaussian_weights=True, use_sample_covariance=False, channel_axis=-1)
