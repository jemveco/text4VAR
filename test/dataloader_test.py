# from datasets import VideoDataset
# from dotmap import DotMap()
# from utils.Augmentation import get_augmentation
# import yaml
#
# with open("configs/ucf-crime/train_rgb_vitl14-f16-anomize.yaml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#
# config = DotMap(config)
#
# transform_train = get_augmentation(True, config)
#
# train_data = VideoDataset(**config.data.toDict(), transform=transform_train, list_yaml=True, split="train")

import sys
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader
from pathlib import Path

# Navigate up one directory level from the current file
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to the search path
sys.path.append(str(parent_dir))

from datasets.datasets import VideoDataset
from utils.augmentation import get_augmentation


def test_dataset():
    config_path = "configs/xd-violence/test_rgb_vitl14-f16-anomize.yaml"

    print(f"[*] Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config_dict)

    # 1. Cargar transformaciones
    print("[*] Initializing validation transformations...")
    try:
        transform_val = get_augmentation(**config.dataset.toDict())["val"]
    except Exception as e:
        raise RuntimeError(f"[!] Error loading transformations: {e}")

    # 2. Extraer solo los parámetros que VideoDataset necesita
    # Esto evita el "TypeError: __init__() got an unexpected keyword argument"
    dataset_kwargs = config.dataset.toDict()

    print("[*] Instantiating VideoDataset...")
    try:
        val_data = VideoDataset("val", transform_val, **dataset_kwargs)
        print(f"[+] Dataset instantiated correctly. Total videos: {len(val_data)}")
    except Exception as e:
        raise RuntimeError(f"[!] Error instantiating VideoDataset: {e}")

    # 3. Probar el DataLoader extrayendo el primer batch
    print("[*] Testing DataLoader...")
    val_loader = DataLoader(
        val_data,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=0  # Mantener en 0 para debugear fácilmente
    )

    try:
        # Extraer un solo batch iterando una vez
        images, paths, labels = next(iter(val_loader))
        print("[+] Succesfully batch extraction!")
        print(f" => Images tensor shape: {images.shape}")
        print(f" => Paths size:          {len(paths)}")
        print(f" => Labels tensor shape: {labels.shape}")
        print(f" => Tensor types: {images.dtype=}, {type(paths[0])=}, {labels.dtype=}")

    except Exception as e:
        raise RuntimeError(f"[!] Error in DataLoader data extraction: {e}")


if __name__ == "__main__":
    test_dataset()
