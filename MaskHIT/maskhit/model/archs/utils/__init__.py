import torch
import re
from .masking_generator import MaskingGenerator


def fill_zeros(value, masks, token):
    masked_input = value.clone()
    # masked input
    masked_input[masks] = token

    return masked_input


def fill_mask(
    value,
    mask,
    mask_token,
):
    masked_input = value.clone()
    # masked input
    masked_input[mask == 1] = mask_token

    # original input
    # do nothing

    # random input
    num_random = (mask == 3).sum()
    b, p = value.size(0), value.size(1)
    masked_input[mask == 3] = value.view(-1, value.size(-1))[torch.randperm(
        b * p)[:num_random]]
    return masked_input


def masked_average(features, masks):
    # mask: 0 - mask; 1 - no mask
    # zeros will be excluded
    masked_features = features * masks
    return masked_features.sum(1) / torch.clamp(masks.sum(1), min=1.0)


def parse_archs(arch):
    assert len(arch.split('_')) <= 2
    if '_' in arch:
        name, specs = arch.split('_')
        hh = re.findall('(?<=h)\\d+', specs)
        ll = re.findall('(?<=l)\\d+', specs)
    else:
        name = arch
        hh, ll = 1
    return {'name': name, 'h': hh, 'l': ll}
