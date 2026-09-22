import os
from dataclasses import dataclass

import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from numpy.random import randint
# from PIL import Image
#  import decord
from torchcodec.decoders import VideoDecoder
# import torchvision.transforms.v2.transforms.functional.to_image
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision import tv_tensors

# Data types for type annotations
from typing import List, Optional, Tuple
from logging import Logger
from numpy.typing import NDArray


_VALID_MODALITIES = ["RGB", "video"]
_VALID_SPLITS = ["train", "test", "val"]
_VALID_SAMPLE_TYPES = ["frames", "random_shift", "dense_sample"]


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


@dataclass(frozen=True)
class Video:
    """ Dataclass representing "metadata" of a single video in a dataset """
    path: str        # The path to the video
    num_frames: int  # Number of frames of the video
    label: str       # Video class
    id: int          # Video class
    frame_range: Tuple[int, int]  # Range within frames will be taken

    def fmt(self, root_path: str) -> str:
        return f"Video metadata: {{path: {os.path.join(root_path, self.path)}, range: {self.frame_range[0]}, label: {self.label}}}"


class VideoDataset(data.Dataset):
    """
    PyTorch Dataset for loading video data from either raw video files
    or extracted frames.

    Supports multiple sampling strategies including dense sampling and
    uniform sampling, with options for training, validation, and test
    modes.
    """

    def __init__(self, split: str, transform: v2.Compose, logger: Optional[Logger] = None, **kwargs) -> dict:
        """
        Args:
            split:          Dataset split name (train, test, or val).
            transform:      torchvision v2 transforms to apply to loaded frames.
            looger:         Logger to which print messages, this is
                            an optional variable, in case it is not
                            provided, the messages will be printed to
                            the standard output with "print".
            name:           Dataset's name.
            num_classes:    Dataset's number of classes.
            modality:       Data modality. Valid options are: "RGB"
                            or "video". Defaults to "RGB".
            frame_template: Template for frame filenames. Only used
                            during RGB modality. Defaults to
                            "frame_{:05d}.jpg".
            frame_idx_bias: Offset to add to frame indices (0 or 1
                            based indexing). Defaults to 0.
            input_size:     Size of the each individual frame in the
                            video. Defaults to 224.
            num_segments:   Number of segments to sample from each
                            video. Defaults to 8.
            seg_length:     Number of consecutive frames per segment.
                            Defaults to 1.
            loop:           Whether to loop frames when video is
                            shorter than required. Defaults to False.
            sample_type:    Sample type. Valid options are: "frame",
                            "random_shift" or "dense_sample".
                            Defaults to "random_shift".
            sample_range:   Range for dense sampling. Defaults to 64.
            test_clips:     Number of clips to sample during testing.
                            Defaults to 3.
            min_frames:     Minimum number of frames required for a
                            valid video. Defaults to 64.
            root_path:      Root directory containing video files or
                            frames. Defaults to "".
            {split}_data:   Dictionary containing "path" and "list"
                            keys with the path to videos/frames and
                            path to the list file listing the split
                            samples respectively.
            label_list:     Path to CSV file with class labels.

        Return:
            A dictionary containing the keys that weren't present in
            the "kwargs" argument (useful for logging info). The
            values of these keys are the default values.
        """
        self.split = split
        self.transform = transform

        if split not in _VALID_SPLITS:
            self._log(f"Unknown dataset split `{split}`, valid splits are 'train', 'test' or 'val'", "error")
            raise

        self._parameters = TrackDict(**kwargs)

        self.dataset_name = self._parameters.get("name", "Unknown")
        self.num_classes = self._parameters.get("num_classes", "Unknown")
        # First we retrieve and initialize the variables with mandatory values
        split_data = self._parameters.get(f"{split}_data")
        self.label_list = self._parameters.get("label_list")

        # - Check for "None" values the variables without default value -
        if split_data is None:
            self._log(f"`{split}_data` can't be undefined or None", "error")
            raise

        if type(split_data) is dict:
            if not split_data.get("path") or not split_data.get("list"):
                self._log(f"`{split}` split data should have `path` and `list` fields", "error")
                raise
        else:
            self._log(f"`{split}` split data should be a dictionary containing a `path` and a `list` fields", "error")
            raise

        if self.label_list is None:
            self._log("`label_list` can't be undefined or None", "error")
            raise
        # ---------------------------------------------------------------

        # - Default values for possible "None" values -
        self.modality = str(self._parameters.get("modality", "RGB"))

        if self.modality not in _VALID_MODALITIES:
            self._log(f"Unknown modality `{self.modality}`, valid modalities are {_VALID_MODALITIES}", "error")
            raise

        self.frame_template = str(self._parameters.get("frame_template", "img_{:05d}.jpg"))
        self.frame_idx_bias = int(self._parameters.get("frame_idx_bias", 0))
        self.input_size = int(self._parameters.get("input_size", 224))
        self.num_segments = int(self._parameters.get("num_segments", 8))
        self.seg_length = int(self._parameters.get("seg_length", 1))
        self.loop = bool(self._parameters.get("loop", False))
        self.sample_type = self._parameters.get("sample_type", "random_shift")

        if self.sample_type not in _VALID_SAMPLE_TYPES:
            self._log(f"Unknown sample type `{self.sample_type}`, valid samples are 'frame', 'random_shift' or 'val'", "error")
            raise

        if self.sample_type == "dense_sample":
            # Get or set default value of "sample_range" for "dense_sample" sample type
            self.sample_range = int(self._parameters.get("sample_range", 64))
        if split == "test":
            # Get or set default value of "test_clips" for the test split
            self.test_clips = int(self._parameters.get("test_clips", 3))

        self.min_frames = int(self._parameters.get("min_frames", 64))

        root_path = str(self._parameters.get("root_path", ""))
        # ---------------------------------------------

        self.root_path = os.path.join(root_path, str(split_data["path"]))
        self.list_file = str(split_data["list"])

        # Storage for video records and captions
        self.video_list: List[Video] = list()

        # Dataset classes in a dictionary form
        self.classes = pd.read_csv(self.label_list, index_col="label").to_dict()["id"]

        self._parse_list_yaml()

        if isinstance(logger, Logger):
            self._logger = logger
        else:
            self._logger = None
        self._log(f"Dataset metadata:\n{'=' * 125}", "info")
        self._log(f" => Dataset name: {self.dataset_name}", "info")
        self._log(f" => Number of classes: {self.num_classes}", "info")
        self._log(f" => Loaded split: {split}", "info")
        self._log(f" => Transformations to apply: {transform.transforms}", "info")
        self._log(f" => {self.split} split initialized with values: {self._parameters}", "info")
        self._log(f" => Video records loaded (dataset {self.split} split size): {len(self)}\n{'=' * 125}", "info")

    def __len__(self) -> int:
        """ Return the total number of video records in the dataset """
        return len(self.video_list)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, int]:
        """
        Get a single sample from the dataset.

        Args:
            index: Index of the sample.

        Return:
            Tuple of processed frames, video path, video label.
        """
        max_retries = 10  # If a retry fails it will try with a new random index

        for attempt in range(max_retries):
            # if self.split == "val":
            #     breakpoint()
            record = self.video_list[index]
            if attempt > 0:
                self._log(f"Retrying with idx {index} ({record}, frame_idx_bias: {self.frame_idx_bias})...", "info")

            # Check for empty video
            if record.num_frames == 0:
                self._log(f"Empty video ({record.fmt(self.root_path)}, frame_idx_bias: {self.frame_idx_bias})", "error")
                index = randint(0, len(self.video_list) - 1)
                continue

            # Sample frame indices
            if self.sample_type == "dense_sample":
                indices = self._dense_sampling(record.num_frames)
            elif self.sample_type == "random_shift":
                indices = self._random_shift_sampling(record.num_frames)
            elif self.sample_type == "full_segment":
                # Get all the frames (it could be used to process each frame independently)
                indices = np.arange(record.num_frames)
            else:  # This should never be reached since we check first in __init__()
                self._log(f"Unknown option `{self.sample_type}`", "error")
                raise

            # Load and process frames
            processed_frames = self.load_frames(record, indices)
            return processed_frames, record.path, record.id

        # If all retries failed, raise error
        self._log(
            f"Failed to load video after {max_retries} attempts. Last attempted index: {index} "
            f"({record.fmt(self.root_path)}, frame_idx_bias: {self.frame_idx_bias})"
        )
        raise

    def _log(self, msg: str, log_type: str):
        if isinstance(self._logger, Logger):
            if log_type == "info":
                self._logger.info(msg)
            elif log_type == "warn":
                self._logger.warn(msg)
            elif log_type == "error":
                self._logger.error(msg)
        else:
            if log_type == "info":
                print(f"[INFO] {msg}")
            elif log_type == "warn":
                print(f"[WARN] {msg}")
            elif log_type == "error":
                print(f"[ERROR] {msg}")

    def _parse_list_yaml(self):
        """ Parse video list from YAML file with temporal segments and captions """
        with open(self.list_file) as file:
            data = yaml.safe_load(file)

        for cls, videos in data.items():
            for video in videos:
                # Video filename
                path = os.path.join(cls, video["name"])
                # path = video["name"]  # For XD-Violence multiclass
                # breakpoint()
                # TODO: Change the yaml format and separate the captions from his video segment
                # if video["name"] == "Shoplifting026_x264":
                #     breakpoint()
                if video.get("video-segments"):
                    for segment in video.get("video-segments", []):
                        start_frame = segment["timestamp-frames"]["start"]
                        end_frame = segment["timestamp-frames"]["end"]

                        total_frames = end_frame - start_frame + 1  # We add 1 because we are taking the "end_frame" into account
                        # TODO: Check the range that annotator app writes to the videos to
                        # see if it adds the frame_index_bias and remove if is the case
                        frame_range = tuple([self.frame_idx_bias + start_frame, self.frame_idx_bias + end_frame + 1])  # Inclusive end
                        self.video_list.append(Video(
                            path=path,
                            num_frames=total_frames,  # We add 1 because we are taking the "end_frame" into account
                            label=cls,             # Get class label
                            id=self.classes[cls],  # Get class id
                            frame_range=frame_range  # Inclusive end
                        ))
                else:
                    continue
                    if self.modality == "RGB":
                        # In the case the video is "RGB" (pre-extracted frames), open the "meta_inf.json"
                        # file to get the amount of frames of the video
                        # with open(os.path.join(path, "meta_info.json"), "r", encoding="utf-8") as f:
                        with open(os.path.join(self.root_path, path, "info.json"), "r", encoding="utf-8") as f:
                            total_frames = json.load(f)["frames"]
                    elif self.modality == "video":
                        total_frames = len(VideoDecoder(os.path.join(self.root_path, path)))
                        # TODO: Make the video case (should open the video and get the number of frames)
                        raise NotImplementedError(f"Modality `{self.modality}` not implemented yet")
                    else:  # This should never be reached
                        self._log(f"Unknown modality `{self.modality}`", "error")
                        raise

                    frame_range = tuple([self.frame_idx_bias, self.frame_idx_bias + total_frames + 1])  # Full video leght range
                    self.video_list.append(Video(
                        path=path,
                        num_frames=total_frames,  # We add 1 because we are taking the "end_frame" into account
                        label=cls,             # Get class label
                        id=self.classes[cls],  # Get class id
                        frame_range=frame_range  # Inclusive end
                    ))

    def get_sample_weights(self) -> torch.Tensor:
        targets = {}

        for record in self.video_list:
            if record.id not in targets:
                targets[record.id] = list()
            targets[record.id].append(record.id)

        # breakpoint()
        targets = torch.cat(list(torch.tensor(i) for i in targets.values()))
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()

        return class_weights[targets]

    def get_default_params(self):
        """ Get dictionary parameters passed during the initialization """
        return self._parameters.get_tracker()

    def load_frames(self, record: Video, indices: List[int] | NDArray | dict | tuple) -> torch.Tensor:
        """
        Load frames based on indices. Also applies the transformation
        provided in the __init__()

        Args:
            record:  Video object.
            indices: Array of frame indices to load.

        Return:
            Tuple of (processed frames tensor, label id).
        """
        start_offset = record.frame_range[0]
        # if record.name == "Shoplifting026_x264":
        #     breakpoint()
        if isinstance(indices, list):
            indices = list(int(v + start_offset) for v in indices)
        elif isinstance(indices, np.ndarray):
            # Converts a numpy array to a list
            indices = (indices + start_offset).astype(int).tolist()
        elif isinstance(indices, dict):
            # Converts the values of a dictionary to a list
            indices = list(int(v + start_offset) for v in indices.values())
        elif isinstance(indices, tuple):
            # Converts a tuple to a list
            indices = list(int(v + start_offset) for v in indices)
        else:
            self._log(f"Unsupported type `{type(indices)}`", "error")
            raise

        video_path = os.path.join(self.root_path, record.path)
        if self.modality == "RGB":
            frames = list()

            try:
                # Fallback to first frame
                fallback_frame = decode_image(os.path.join(video_path, self.frame_template.format(indices[0])), mode="RGB")
                fallback_type = "first frame"
            except Exception:
                fallback_frame = torch.zeros(size=(3, self.input_size, self.input_size), dtype=torch.uint8)
                fallback_type = "black image"

            for idx in indices:
                file = os.path.join(video_path, self.frame_template.format(idx))
                # if file == "/mnt/data/Emilio/XD-Violence/test/frames/v=bhZs3ALdL7Y__#1_label_G-0-0/frame_0.jpg":
                #     breakpoint()
                try:
                    frame = decode_image(file, mode="RGB")
                except Exception as e:
                    self._log(f"Error loading frame `{file}`: {e}", "error")
                    self._log(f"Using {fallback_type} as fallback frame", "warn")
                    frame = fallback_frame.clone().detach()

                frames.append(frame)
            frames = torch.stack(frames)
        elif self.modality == "video":
            # Load from video file
            decoder = VideoDecoder(video_path)
            frames = decoder.get_frames_at(indices).data
        else:  # This should never be reached since we check first in __init__()
            self._log(f"Unknown modality `{self.modality}`", "error")
            raise

        # breakpoint()
        # Apply transformations
        return self.transform(tv_tensors.Video(frames))

    def _dense_sampling(self, num_frames: int) -> List[int]:
        """
        Dense sampling strategies for train, test and validation splits

        Args:
            num_frames: Number of available frames in the video.

        Return:
            Array of frame indices to sample.
        """
        sample_pos = max(1, 1 + num_frames - self.sample_range)
        interval = max(1, self.sample_range // self.num_segments)
        base_offsets = np.arange(self.num_segments) * interval

        if self.split == "train":
            start_idx = 0 if sample_pos == 1 else randint(0, sample_pos)
            offsets = (base_offsets + start_idx) % num_frames
        elif self.split == "test":
            # Dense sampling with multiple clips
            start_list = np.linspace(0, sample_pos - 1, num=self.test_clips, dtype=int)
            offsets = list()
            for start_idx in start_list:
                clip_offsets = (base_offsets + start_idx) % num_frames
                offsets.extend(clip_offsets.astype(int).tolist())
        elif self.split == "val":
            start_idx = 0 if sample_pos == 1 else sample_pos // 2
            # offsets = [
            #     (idx * interval + start_idx) % num_frames
            #     for idx in range(self.num_segments)
            # ]
            offsets = (base_offsets + start_idx) % num_frames

        if isinstance(offsets, np.ndarray):
            offsets = offsets.astype(int).tolist()

        return offsets  # If we use "+ self.frame_idx_bias" here, we can break the indexing

    def _random_shift_sampling(self, num_frames: int) -> List[int]:
        """
        Sample frame indices for train, test and validation split with random shifts.

        Args:
            num_frames: Number of available frames in the video.

        Return:
            Array of frame indices to sample.
        """
        interval = max(1, num_frames // self.num_segments)

        if self.split == "train":
            total_needed = self.num_segments * self.seg_length  # Total number of frames to sample per video
            # Standard sampling strategy
            if num_frames < total_needed:
                # Video is shorter than required length
                # Adds indices until the "total_needed"
                # is satisfied
                if self.loop:
                    # Loop through frames with random offset, basically
                    # it repeats the first indices
                    half = max(1, num_frames // 2)
                    offsets = np.mod(
                        np.arange(total_needed) + randint(0, half),
                        num_frames
                    )
                    # return offsets  # + self.frame_idx_bias
                else:
                    # Adds extra random indices
                    extra = total_needed - num_frames
                    rand_indices = randint(0, num_frames, size=extra)
                    offsets = np.concatenate([np.arange(num_frames), rand_indices])
                    offsets = np.sort(offsets)
                    # return np.sort(offsets)  # + self.frame_idx_bias
            else:
                # Video is longer than required - sample segments
                offsets = list()
                base_offsets = np.arange(self.num_segments) * interval
                max_offset = interval - self.seg_length
                for segment_start in base_offsets:
                    # Randomly select starting position within segment
                    segment_start += randint(0, max_offset + 1)

                    # Sample consecutive frames
                    segment_end = segment_start + self.seg_length
                    offsets.extend(np.arange(segment_start, segment_end, dtype=int).tolist())
                # return np.array(offsets)  # + self.frame_idx_bias
        elif self.split == "test":
            # Uniform sampling with multiple clips
            start_list = np.linspace(0, interval - 1, num=self.test_clips, dtype=int)

            offsets = list()
            for start_idx in start_list:
                for seg_idx in range(self.num_segments):
                    frame_idx = (start_idx + interval * seg_idx) % num_frames
                    offsets.append(frame_idx)

            # return np.array(offsets) + self.frame_idx_bias
        elif self.split == "val":
            # Uniform sampling
            interval = max(1, num_frames // self.num_segments)
            offsets = np.mod(np.arange(self.num_segments) * interval, num_frames)
            # return offsets  # + self.frame_idx_bias
        else:
            self._log(f"Unknown dataset split `{self.split}`", "error")
            raise

        if isinstance(offsets, np.ndarray):
            offsets = offsets.astype(int).tolist()

        return offsets

