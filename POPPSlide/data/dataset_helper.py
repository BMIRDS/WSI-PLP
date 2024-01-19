import numpy as np
import torch
import random


def batchsplit(img, nrows, ncols):
    return np.concatenate(
        np.split(
            np.stack(
                np.split(img, nrows, axis=0)
            ), ncols, axis=2)
    )

# ----------------------------------------------------------------------------
# Helper functions for data transformation, currently not used in the pipeline
# ----------------------------------------------------------------------------
def reverse_norm(mean, std):
    '''Reverse normalization'''
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()


def random_crop(img, height, width):
    '''random crop image'''
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img

def random_crops(img, crop_size, n_crops):
    '''Generate multiple random crops from the image (not used)'''
    h, w, c = img.shape
    if (h == crop_size) and (w == crop_size):
        return img
    nrows = h // crop_size
    ncols = w // crop_size
    if max(nrows % crop_size, ncols % crop_size):
        img = random_crop(img, nrows * crop_size, ncols * crop_size)
    splits = batchsplit(img, nrows, ncols)
    return splits[random.sample(range(splits.shape[0]), n_crops)]
