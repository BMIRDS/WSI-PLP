import torch
from torchvision import transforms


PATH_MEAN = [0.7968, 0.6492, 0.7542]
PATH_STD = [0.1734, 0.2409, 0.1845]


def reverse_norm(mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return (-mean / std).tolist(), (1 / std).tolist()


def get_data_transforms(patch_size=224):

    data_transforms = {
        'train':
        transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.1,
                                   hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(
                PATH_MEAN, PATH_STD
            )  # mean and standard deviations for lung adenocarcinoma resection slides
        ]),
        'val':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'predict':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'normalize':
        transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(PATH_MEAN, PATH_STD)]),
        'unnormalize':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*reverse_norm(PATH_MEAN, PATH_STD))
        ]),
    }
    return data_transforms
