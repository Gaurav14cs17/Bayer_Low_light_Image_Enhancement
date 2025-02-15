import numpy as np


def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.uint16)
    H, W = im.shape
    im = np.expand_dims(im, axis=0)
    out = np.concatenate((im[:, 0:H:2, 0:W:2],
                          im[:, 0:H:2, 1:W:2],
                          im[:, 1:H:2, 1:W:2],
                          im[:, 1:H:2, 0:W:2]), axis=0)
    return out


def crop_random_patch(input_raw, gt_raw, gt_rgb, patch_size):
    _, H, W = input_raw.shape
    yy, xx = np.random.randint(0, H - patch_size), np.random.randint(0, W - patch_size)
    input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
    gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
    gt_rgb = gt_rgb[:, yy * 2:(yy + patch_size) * 2, xx * 2:(xx + patch_size) * 2]
    return input_raw, gt_raw, gt_rgb
