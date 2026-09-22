from pathlib import Path
import sys
import copy

import clip
from dotmap import DotMap
import torch
from torch.backends import cudnn
from torch import optim
import yaml

# Navigate up one directory level from the current file
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to the search path
sys.path.append(str(parent_dir))

from modules.video_clip import VideoHead, VideoCLIP
from utils import load
from utils import solver

CONFIG_FILE = "configs/xd-violence/train_rgb_vitb16-amp.yaml"
DEVICE = 0
PRECISION = "amp"
DEFAULT_CONFIG = {
    "seed": 1,
    "projector": None,
    "reductor": None,
    "dataset": {},
    "dataloader": {
        "batch_size": 8,
        "num_workers": 0,
    },
    "network": {
        "arch": "ViT-B/16",
        "init": True,
        "drop_out": 0.0,
        "emb_dropout": 0.0,
        "type": "clip_ucf",
        "sim_header": "Transf",
        "drop": 0,
    },
    "solver": {
        "type": "cosine",
        "epochs": 30,
        "optimizer": "adamw",
        "lr": 5.e-5,
        "lr_warmup_step": 5,
        "weight_decay": 0.2,
        "loss_type": "CE",
        "evaluate": False,
        "clip_ratio": 0.1,
        "grad_accumulation_steps": 2,
    },
    "logging": {
        "print_freq": 10,
        "eval_freq": 1,
    },
}


class Config:
    def __init__(self, config: dict, default_config: dict = None, _level="root", _father=None):
        self._config_dict = copy.deepcopy(config)
        self._level = _level
        self._father = _father
        # Default values of the configuration.
        # They will be used if there's a missing
        # value in the "_config_dict"
        if isinstance(default_config, dict):
            self._default_config = copy.deepcopy(default_config)
        else:
            self._default_config = DEFAULT_CONFIG

        for k, v in self._default_config.items():
            if isinstance(v, dict):
                cfg_value = self._config_dict.get(k, {})
                if cfg_value is not None and not isinstance(cfg_value, dict):
                    raise ValueError
                self.__setattr__(k, Config(
                    cfg_value,
                    default_config=v,
                    _level="child",
                    _father=k
                ))

    def __getattr__(self, name):
        if self._config_dict is None:
            return None
        if name not in self._config_dict:
            # If there's a missing value in
            # "_default_values", then there's
            # no default value for that config
            # and "None" will be returned
            return self._default_config.get(name)
        else:
            return self._config_dict.get(name)

    def to_dict(self):
        if self._config_dict is None:
            return {self._father: None}

        def merge_dicts(d1, d2) -> dict:
            """ Merge dicts keeping the values of d1 if a key is present in both dictionaries """
            data = {}
            for k, v in d2.items():
                if k in d1:
                    if isinstance(v, dict):
                        if d1[k] is None:
                            data[k] = None
                        else:
                            data[k] = merge_dicts(d1[k], d2[k])
                    else:
                        data[k] = d1[k]
                else:
                    data[k] = v
            return data

        return merge_dicts(self._config_dict, self._default_config)

    # def __getattribute__(self, name):
    #     if name in object.__getattribute__(self, "__dict__"):
    #         cfg = super().__getattribute__(name)
    #         if isinstance(cfg, Config):
    #             if cfg._config_dict or cfg._config_dict is None:
    #                 return cfg._config_dict
    #             else:
    #                 return cfg._default_config
    #         else:
    #             return cfg
    #     else:
    #         return self.__getattr__(name)


def check_model_params(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            breakpoint()
        if "module." in name:
            breakpoint()
        print(f"Parámetro: {name} | Tipo de dato: {param.dtype}")
    precision_modelo = next(model.parameters()).dtype
    print(f"La precisión del modelo es: {precision_modelo}")


def test_model_load():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    # Device setup
    if DEVICE == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{DEVICE}")
        cudnn.benchmark = True
    torch.set_default_device(device)

    a = config.network.arch
    # ---------------------------- Building full model ----------------------------
    # Load CLIP model
    model, _ = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        # internal_modeling=config.network.tm,
        # T=config.dataset.num_segments,
        # dropout=config.network.drop_out,
        # emb_dropout=config.network.emb_dropout,
        # pretrain=config.network.init,
        # joint_st=config.network.joint_st
    )
    clip_state_dict = model.state_dict()
    check_model_params(model)
    # Set precision
    if PRECISION == "fp32":
        model = model.float()

    breakpoint()
    # Video header
    video_head = VideoHead(config.network.sim_header, clip_state_dict).to(device)
    # Load text features
    try:
        text_features = load.text_features(path=config.projector, device=device, logger=None)
    except Exception:
        text_features = None
    # Load reductor
    try:
        reductor = load.reductor(path=config.reductor, device=device, logger=None)
    except Exception:
        reductor = None
    # Build full model
    full_model = VideoCLIP(model, video_head, config.dataset.num_segments, reductor)

    clip_params = []
    other_params = []
    breakpoint()
    for name, param in full_model.named_parameters():
        if not param.requires_grad:
            breakpoint()
            continue

        if "visual" in name and "control_point" not in name:
            clip_params.append(param)
            breakpoint()
        elif "logit_scale" in name:
            clip_params.append(param)
            breakpoint()
        else:
            other_params.append(param)

    if config.solver.optim == "adamw":
        optimizer = optim.AdamW(
            [
                {'params': clip_params, 'lr': config.solver.lr * config.solver.clip_ratio},
                {'params': other_params, 'lr': config.solver.lr}
            ],
            betas=(0.9, 0.999),
            lr=config.solver.lr,
            eps=1e-8,
            weight_decay=config.solver.weight_decay
        )

    lr_scheduler = solver._lr_scheduler(config, optimizer)
    breakpoint()


if __name__ == "__main__":
    test_model_load()
