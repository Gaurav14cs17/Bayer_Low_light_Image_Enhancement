# Bayer Low light Image Enhancement

- ##### WaveTransformBlock
![WaveTransformBlock](./images/image_1.png )


# RawFormer: Efficient Raw Image Processing with Transformer-CNN Hybrid Architecture
## Overview

RawFormer is a hybrid CNN-Transformer architecture designed for raw image processing tasks like denoising, super-resolution, and demosaicing. This implementation combines the local feature extraction capabilities of CNNs with the global attention mechanisms of Transformers for efficient raw image processing.

## Key Features

- **Hybrid Architecture**: Combines convolutional layers with efficient self-attention
- **Lightweight Design**: Multiple model sizes available (Small/Base/Large)
- **Multi-Task Capable**: Can handle various raw image processing tasks
- **Efficient Attention**: Memory-efficient self-attention implementation
- **Progressive Down/Up-sampling**: For multi-scale feature processing

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/RawFormer.git
cd RawFormer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- einops
- skimage
- tqdm
- imageio
- numpy

## Usage

### Training

```python
python train.py --dataset SID --model_size S --gpu_id 0
```

### Testing

```python
python test.py --dataset SID --model_size S --gpu_id 0
```

### Arguments

| Argument      | Description                          | Options               | Default |
|--------------|--------------------------------------|-----------------------|---------|
| `--dataset`  | Dataset to use                       |  `SID`                | `SID`   |
| `--model_size` | Model size configuration           | `S`, `B`, `L`         | `S`     |
| `--gpu_id`   | GPU device ID to use                 | `0`, `1`, etc.        | `0`     |
| `--batch_size` | Batch size for training/testing    | Integer > 0           | `8`     |

## Model Configurations

| Model Size | Dim | Heads | Params |
|------------|-----|-------|--------|
| Small (S)  | 32  | [8,8,8,8] | ~4.2M  |
| Base (B)   | 48  | [8,8,8,8] | ~9.5M  |
| Large (L)  | 64  | [8,8,8,8] | ~16.8M |

## Results

Performance on MCR dataset (PSNR/SSIM):

| Model Size | PSNR  | SSIM   |
|------------|-------|--------|
| Small      | 32.45 | 0.921  |
| Base       | 33.12 | 0.928  |
| Large      | 33.78 | 0.934  |

## Directory Structure

```
RawFormer/
├── data/                   # Dataset loading utilities
├── models/                 # Model architecture definitions
├── weights/                # Pretrained model weights
├── results/                # Output images and metrics
├── train.py                # Training script
├── test.py                 # Testing script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

