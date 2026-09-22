# import copy
from dotmap import DotMap

# _DEFAULT_CONFIG = {
#     "seed": {"default": 1, "allowed_values": any, "allowed_types": any},  # Template
#     "dataset": {"default": {
#         "name": {"default": None, "allowed_types": str},  # Template specific data type, but still be able to use None
#         "num_classes": {"default": 0},  # Template data type inferred from default
#         "modality": {"default": "RGB", "allowed_values": ["RGB", "video"]},
#         "frame_template": "frame_{}.jpg",  # Template only default value, default type also inferred from default
#         "frame_idx_bias": 0,
#         "input_size": 224,
#         "num_segments": 8,
#         "seg_length": 1,
#         "loop": True,
#         "sample_type": {"allowed_values": ["random_shift", "dense_sampling"]},
#         "#####################
#         "# Sample range in the case of dense sampling
#         "# sample_range: 64
#         "#####################
#         "#
#         "# test_clips: 3
#         "# min_frames: 8
#         "#####################
#         "root_path: {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},/mnt/hdd/datasets/xd-violence/frames/single-label
#         "val_data:{"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},
#             path: {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},test
#             list: {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},lists/xd-violence/test.yaml
#         label_list:{"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str}, lists/xd-violence/xd_violence_labels.csv
# , "allowed_values": any, "allowed_types": dict},
#     "reductor": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},None,
#     "dataloader": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},{
#         "batch_size": 8,
#         "num_workers": 0,
#     },
#     "network": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},{
#         "arch": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"ViT-B/16",
#         "init": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},True,
#         "drop_out": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},0.0,
#         "emb_dropout": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},0.0,
#         "type": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"clip_ucf",
#         "sim_header": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"Transf",
#         "drop": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},0,
#     },
#     "solver": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},{
#         "type": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"cosine",
#         "epochs": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},30,
#         "optimizer": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"adamw",
#         "lr": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},5.e-5,
#         "lr_warmup_step": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},5,
#         "weight_decay": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},0.2,
#         "loss_type": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},"CE",
#         "evaluate": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},False,
#         "clip_ratio": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},0.1,
#         "grad_accumulation_steps": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},2,
#     },
#     "logging": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},{
#         "print_freq": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},10,
#         "eval_freq": {"default": "RGB", "allowed_values": ["RGB", "video"], "allowed_types": str},1,
#     },
# }


class TrackDict(dict):
    """
    Wrapper around the dictionary class to track the keys that are not
    present in the dictionary. It stores the keys and default values
    in a tracker dictionary variable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = dict()  # Tracks the keys that are not in the dictionary

    def __str__(self):
        formatted_output = "{\n"
        for key, value in (self | self.tracker).items():
            formatted_output += f"{' ' * 4}{key}: {value}\n"

        formatted_output += "}"
        return formatted_output  # Removes last "\n"

    def __getitem__(self, key_value):
        """
        Get a value indexed by a key or a default value.

        Args:
            key_value: Index key to return a value. In the case it is
                       a tuple, uses the first value as the index key
                       and returns the second value in the case the
                       key is missing.

        Return:
            The value of a key or a default value in the case
            {key_value} is a tuple.
        """
        if isinstance(key_value, tuple) and len(key_value) > 1:
            return self.get(key_value[0], key_value[1])

        return super().__getitem__(key_value)

    def get(self, key, default_value=None):
        """
        Get a value in the dictionary given a key. If the key doesn't
        exist get a default value and copy the missing key and default
        value to the tracker dict.

        Args:
            key:           Index key to return a value or to store
                           {default_value} into tracker dict.
            default_value: Value to return in the case the key is
                           missing.

        Return:
            A value indexed by a key if it exists, "default_value"
            otherwise.
        """
        if key not in self:
            self.tracker[key] = default_value
            return default_value

        return super().__getitem__(key)

    def get_tracker(self):
        """ Get the tracker """
        return self.tracker


# class Config:
#     def __init__(self, config: dict, default_config: dict = None, allow_any: bool = False, _level="root", _father=None):
#         self._config_dict = TrackDict(copy.deepcopy(config))
#         # Default values of the configuration.
#         # They will be used if there's a missing
#         # value in the "_config_dict"
#         if isinstance(default_config, dict):
#             self._default_config = copy.deepcopy(default_config)
#         else:
#             self._default_config = _DEFAULT_CONFIG
#         self.allow_any = allow_any
#         self._level = _level
#         self._father = _father
#
#         for k, v in self._default_config.items():
#             if isinstance(v, dict) and v not in self._config_dict:
#                 cfg_value = self._config_dict[k, v]
#                 # if cfg_value is not None and not isinstance(cfg_value, dict):
#                 if allow_any and not isinstance(cfg_value, dict):
#                     raise ValueError
#                 self.__setattr__(k, config(
#                     cfg_value,
#                     default_config=v,
#                     _level="child",
#                     _father=k
#                 ))
#
#         for k, v in self._config_dict.items():
#             if isinstance(v, dict):
#                 self.__setattr__(k, config(
#                     cfg_value,
#                     default_config=v,
#                     _level="child",
#                     _father=k
#                 ))
#
#     def __getattr__(self, name):
#         if self._config_dict is None:
#             return None
#         if name not in self._config_dict:
#             # If there's a missing value in
#             # "_default_values", then there's
#             # no default value for that config
#             # and "None" will be returned
#             return self._default_config.get(name)
#         else:
#             return self._config_dict.get(name)
#
#     def to_dict(self):
#         if self._config_dict is None:
#             return {self._father: None}
#
#         def merge_dicts(d1, d2) -> dict:
#             """ Merge dicts keeping the values of d1 if a key is present in both dictionaries """
#             data = {}
#             for k, v in d2.items():
#                 if k in d1:
#                     if isinstance(v, dict):
#                         if d1[k] is None:
#                             data[k] = None
#                         else:
#                             data[k] = merge_dicts(d1[k], d2[k])
#                     else:
#                         data[k] = d1[k]
#                 else:
#                     data[k] = v
#             return data
#
#         return merge_dicts(self._config_dict, self._default_config)
#
#     # def __getattribute__(self, name):
#     #     if name in object.__getattribute__(self, "__dict__"):
#     #         cfg = super().__getattribute__(name)
#     #         if isinstance(cfg, Config):
#     #             if cfg._config_dict or cfg._config_dict is None:
#     #                 return cfg._config_dict
#     #             else:
#     #                 return cfg._default_config
#     #         else:
#     #             return cfg
#     #     else:
#     #         return self.__getattr__(name)


class Config(DotMap):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)

    def __getattr__(self, name):
        # value = super().__getattribute__(name)
        if name in self:
            return self[name]

        return None

    # def toDict(self, seen):
    #     d = super().toDict()
    #     breakpoint()
    #     for k, v in d:
    #         if isinstance(v, Config):
    #             d[k] = d[k].toDict()

    #     breakpoint()
    #     return d
