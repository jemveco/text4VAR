"""
This is a module file that delivers functions resposible for
loading centain data (specified in each function) from files
"""

from logging import Logger
from pathlib import Path
import torch
from typing import Optional, Dict

from .logger import log


def text_projector(path: Path | str, device: torch.device | str, dtype: Optional[torch.dtype] = None, logger: Optional[Logger] = None) -> torch.Tensor:
    """
    Loads text features

    Args:
        path:   The file path to load.
        device: The device to which load the features.
        logger: The object to log. If None the log won't be printed.

    Return:
        A torch Tensor with the text features.
    """
    if not isinstance(path, str):
        path = str(path)  # Ensuring compatibility with pytorch
    if not isinstance(device, torch.device):
        device = torch.device(device)

    try:
        log(msg=f" => Loading text features from: {path}...", logger=logger, log_type="info")
        text_features = torch.load(path, map_location=device)
        # breakpoint()
    except Exception as e:
        log(msg=f"Cannot open file `{path}`: {e}", logger=logger, log_type="error")
        raise RuntimeError

    text_features = torch.stack(list(text_features.values())).to(dtype=dtype)
    # text_features = text_features.to(dtype=dtype)
    log(msg=f" => Loaded text features from `{path}` with precision `{text_features.dtype}`", logger=logger, log_type="info")
    log(msg=f" => Text features shape: {text_features.shape}", logger=logger, log_type="info")
    return text_features


def reductor(path: Path | str, device: torch.device | str, dtype: Optional[torch.dtype] = None, logger: Optional[Logger] = None) -> Dict[str, torch.Tensor]:
    """
    Loads a reductor

    Args:
        path:   The file path to load.
        device: The device to which load the features.
        logger: The object to log. If None the log won't be printed.

    Return:
        A torch Tensor with the reductor.
    """
    if not isinstance(path, str):
        path = str(path)  # Ensuring compatibility with pytorch
    if not isinstance(device, torch.device):
        device = torch.device(device)

    try:
        log(msg=f" => Loading reductor from: {path}...", logger=logger, log_type="info")
        reductor = torch.load(path, map_location=device)
    except Exception as e:
        log(msg=f"Failed to load reductor: {e}", logger=logger, log_type="error")
        raise RuntimeError

    # reductor = {"reductor_type": "pca", "reductor": reductor["components"], "centerer": reductor["mean"]}
    for key in reductor.keys():
        # breakpoint()
        if key != "reductor_type" and reductor[key] is not None:
            reductor[key] = reductor[key].to(device, dtype=dtype)  # Teorically this shouldn't be necessary

    log(msg=f" => Loaded {reductor['reductor_type']} reductor from `{path}` with precision `{reductor['reductor'].dtype}`", logger=logger, log_type="info")
    return reductor


def weights(path: Path | str, device: torch.device | str, logger: Optional[Logger] = None):
    """
    Loads checkpoint's weights

    Args:
        path:   The file path to load.
        device: The device to which load the features.
        logger: The object to log. If None the log won't be printed.

    Return:
        A tuple with the state_dict and the model epoch
    """
    if not isinstance(path, str):
        path = str(path)  # Ensuring compatibility with pytorch
    if not isinstance(device, torch.device):
        device = torch.device(device)

    def update_dict(state_dict):
        """ Removes "module." prefix from state dict keys """
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k.replace("module.", "")] = v
        return new_dict

    try:
        log(msg=f" => Loading checkpoint's weights from: {path}...", logger=logger, log_type="info")
        checkpoint = torch.load(path, map_location=device)
        # breakpoint()
        if "model_state_dict" in checkpoint:
            state_dict = update_dict(checkpoint["model_state_dict"])
        else:
            state_dict = update_dict(checkpoint)

        if "epoch" in checkpoint:
            log(msg=f" => Checkpoint's weights from epoch `{checkpoint['epoch']}` succesfully loaded", logger=logger, log_type="info")
            return state_dict, checkpoint["epoch"]

        log(msg=" => Checkpoint's weights succesfully loaded", logger=logger, log_type="info")
        return state_dict, -1
    except Exception as e:
        log(msg=f"Failed to load reductor: {e}", logger=logger, log_type="error")
        raise RuntimeError

