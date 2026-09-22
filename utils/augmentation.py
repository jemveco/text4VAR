# from datasets.transforms import *
# import torchvision
# from datasets import transforms as T
import torch
from torchvision.transforms import v2
from typing import Dict, Optional


def train_augmentation(input_size) -> v2.Compose:
    """ Get train split augmentation transformations """
    return v2.Compose([
        v2.RandomResizedCrop(input_size),
        # v2.RandomCrop(input_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomGrayscale(p=0.2),
    ])


def get_augmentation(**kwargs) -> Optional[Dict[str, v2.Compose]]:
    """
    Get the data augmentation transformations for the splits provided
    in the dataset configuration

    Args:
        {split}_data: Dictionaries of splits provided in the dataset
        configuration, it'll search for the presence of {train, test and val}_data
        dictionaries
        input_size: Size to which the input should be resized.
                    Defaults to 224.

    Return:
    """
    input_size = int(kwargs.get("input_size", 224))
    scale_size = 256 if input_size == 224 else input_size

    # Mean and standard deviation values from the WIT dataset
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    resize = v2.Resize(scale_size)
    last = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(input_mean, input_std),
    ])

    compose_dict = {}
    if "train_data" in kwargs:
        train_aug = train_augmentation(input_size)

        compose_dict["train"] = v2.Compose([
            resize,
            train_aug,
            last,
        ])

    if "test_data" in kwargs:
        compose_dict["test"] = v2.Compose([
            resize,
            v2.CenterCrop(input_size),
            last,
        ])

    if "val_data" in kwargs:
        compose_dict["val"] = v2.Compose([
            resize,
            v2.CenterCrop(input_size),
            last,
        ])

    return compose_dict if any(compose_dict) else None
