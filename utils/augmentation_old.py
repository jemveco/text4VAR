# from datasets.transforms import *
# import torchvision
# from datasets import transforms as T
from torchvision.transforms import v2


def train_augmentation(input_size, flip=True):
    if flip:
        return v2.Compose([
            # T.GroupRandomSizedCrop(input_size),
            # T.GroupRandomHorizontalFlip(is_flow=False)
            v2.RandomResizedCrop(input_size),
            v2.RandomHorizontalFlip(is_flow=False)
        ])
    else:
        return torchvision.transforms.Compose([
            T.GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            T.GroupRandomHorizontalFlip_sth()
        ])


def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if config.dataset.input_size == 224 else config.dataset.input_size

    normalize = T.GroupNormalize(input_mean, input_std)
    if 'something' in config.dataset.name:
        groupscale = T.GroupScale((240, 320))
    else:
        groupscale = T.GroupScale(int(scale_size))

    if training:
        train_aug = train_augmentation(
            config.dataset.input_size,
            flip=False if 'something' in config.dataset.name else True
        )

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            T.GroupRandomGrayscale(p=0.2),
        ])
    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            T.GroupCenterCrop(config.dataset.input_size)
        ])

    common = torchvision.transforms.Compose([
        T.Stack(roll=False),
        T.ToTorchFormatTensor(div=True),
        normalize
    ])
    return torchvision.transforms.Compose([unique, common])

