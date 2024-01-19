from torchvision import transforms
from data.dataset_helper import reverse_norm

def get_transformation(patch_size=224, mean=[0.5,0.5,0.5], std=[0.25, 0.25, 0.25]):
    '''
    Get data augmentation for different dataset
    patch_size: size of the crops
    mean: sample mean, default [0.5,0.5,0.5] if not defined
    std: sample std, default [0.25, 0.25, 0.25] if not defined
    '''
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
            transforms.ColorJitter(brightness=0.35,
                                   contrast=0.5,
                                   saturation=0.1,
                                   hue=0.16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),
        'val':
        transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.Normalize(mean, std)
        ]),
        'predict':
        transforms.Compose([
            transforms.Normalize(mean, std)
        ]),
        'normalize':
        transforms.Compose([
            transforms.Normalize(mean, std)
        ]),
        'unnormalize':
        transforms.Compose([
            transforms.Normalize(*reverse_norm(mean, std))
            ]),
    }
    return data_transforms

