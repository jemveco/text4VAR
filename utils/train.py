import argparse
import datetime
from logging import Logger  # For type annotations
import os
from pathlib import Path
import shutil
import sys
import time
from typing import List, Optional, Tuple
from contextlib import suppress

import clip
from dotmap import DotMap
import numpy as np
import torch
from torch.amp import GradScaler
from torch.amp import autocast as torch_autocast
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchcodec
import torchvision
import yaml

from datasets.datasets import VideoDataset
from modules.video_clip import VideoCLIP, VideoHead
from utils import load
from utils.augmentation import get_augmentation
from utils.clip_utils import AverageMeter, accuracy
from utils.config import Config
from utils.focal_loss import FocalLoss
from utils.logger import setup_logger, log

import random

_CLIP_ARCHS = ["ViT-B/16", "ViT-B/32", "ViT-L/14"]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default="clip.yaml",
                        help="global config file")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--dist_url", default="env://",
                        help="url used to set up distributed training")
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local rank for DistributedDataParallel")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id to use (for single GPU training)")
    parser.add_argument("--custom_name", type=str, default="",
                        help="custom post fix to save the model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debugpy for debugging")
    return parser.parse_args()


def init_distributed_mode(args, logger: Optional[Logger] = None):
    """ Initialize distributed training if enabled """
    if not args.distributed or not torch.distributed.is_available():
        log('"torch.distributed" not available or "--distributed" set to false, training on single GPU mode', logger, "info")
        args.rank = 0
        args.world_size = 1
        return

    # Init for distribute mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        log('Neither "RANK" nor "WORLD_SIZE" nor "SLURM_PROCID" defined, ignoring using distributed mode', logger, "info")
        args.distributed = False
        return

    # torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    """
    This is commented due to the stupid icoding pylint checking.
    print("distributed init rank {}: {}".format(args.rank, args.dist_url), flush=True)
    """
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()


def reduce_tensor(tensor, args, n=None):
    """ Reduce tensor across processes (only in distributed mode) """
    if not args.distributed or not torch.distributed.is_available():
        return tensor

    if n is None:
        n = torch.distributed.get_world_size()

    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)

    rt = rt / n
    return rt


def update_dict(state_dict):
    """ Remove "module." prefix from state dict keys """
    new_dict = {}
    for k, v in state_dict.items():
        new_dict[k.replace("module.", "")] = v
    return new_dict


def epoch_saving(epoch: int, model, optimizer, filename: str):
    """ Save checkpoint for a specific epoch """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filename)


def set_deterministic_seed(seed=1):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # Core PyTorch seeds
    torch.manual_seed(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Strict determinism flag for PyTorch operations
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_debugger(args):
    """ Setup debugpy only if explicitly requested """
    if args.debug and args.rank == 0:
        try:
            import debugpy
            debugpy.listen(5678)
            print("Debugger listening on port 5678. Waiting for client...")
            debugpy.wait_for_client()
            print("Debugger connected!")
        except ImportError:
            print("Warning: debugpy not installed. Skipping debug mode.")
        except Exception as e:
            print(f"Error: Could not start debugger: {e}")


def setup_few_shot_sampling(data: VideoDataset, cfg: DotMap, logger: Optional[Logger] = None):
    """ Setup few-shot sampling if configured """
    try:
        K = cfg.dataset.shot
    except Exception:
        return

    cls_dict = {}
    for video in data.video_list:
        if video.label in cls_dict:
            cls_dict[video.label].append(video)
        else:
            cls_dict[video.label] = [video]

    selected_videos = []
    for category, videos in cls_dict.items():
        if len(videos) < K:
            log(f"Category `{category}` has only {len(videos)} samples, less than K={K}, keeping all the videos", logger, "warn")
            selected_videos.extend(videos)
        else:
            slice_videos = np.random.choice(videos, size=K, replace=False)
            selected_videos.extend(slice_videos)

    n_repeat = max(1, len(data) // len(selected_videos))
    data.video_list = selected_videos * n_repeat
    log(f"Few-shot sampling ({len(selected_videos)} samples x {n_repeat} repeats): {data.video_list}", logger, "warn")


def plot_loss_curve(log_file, output_path="loss_curve.png", logger: Optional[Logger] = None):
    """ Plot training loss curve from log file """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        data = pd.read_csv(log_file, header=None, names=["epoch", "loss"])
        plt.figure(figsize=(10, 6))
        plt.plot(data["epoch"], data["loss"], marker="o", linewidth=2)
        plt.title("Training Loss Curve", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        log(f" => Loss curve saved to {output_path}", logger, "info")
    except Exception as e:
        log(f"Warning: Could not plot loss curve: {e}", logger, "warn")


def copy_files(rank: int, dest: Path | str, files: dict, logger: Logger | None = None):
    """ Copy the given files to a {dest} directory (only for rank 0) """
    if rank == 0:
        if not isinstance(dest, Path):
            dest = Path(dest)

        dest.mkdir(parents=True, exist_ok=True)
        for file in files:
            filename = file.get("file")
            if filename is None:
                continue
            dest_name = dest / file.get("dest_name", "")
            log(f' => Copying "{filename}" to "{dest_name}"...', logger, "info")
            shutil.copy(file.get("file"), dest_name)

        log("Files succesfully copied", logger, "info")


def get_sampler(data: VideoDataset, distributed: bool = False, shuffle: bool = False):
    if distributed:
        return DistributedSampler(data, shuffle=shuffle)
    else:
        return None


def get_criterion(cfg: Config, logger: Optional[Logger] = None):
    if not isinstance(cfg, Config):
        params = {"type": cfg}
    else:
        params = cfg.toDict()
    loss_type = params.pop("type", None)
    if loss_type == "CE":
        return CrossEntropyLoss(**params)
    elif loss_type == "FL":
        return FocalLoss(**params)
        # raise NotImplementedError
    else:
        log(msg=f"Unrecognized loss type `{loss_type}`", logger=logger, log_type="error")
        raise RuntimeError


def get_scheduler(schedulers_cfg: List | str, optimizer, total_epochs: int = 0, steps_per_epoch: int = 1, logger: Optional[Logger] = None):
    schedulers = list()
    if not isinstance(schedulers_cfg, List):
        schedulers_cfg = [Config({"type": schedulers_cfg, "epochs": total_epochs})]

    milestones = list()
    epochs_remainder = total_epochs
    accumulative_steps = 0
    for idx, cfg in enumerate(schedulers_cfg):
        if epochs_remainder <= 0:
            log(msg="There are no remaining epochs to set for the next schedulers, skipping next configurations...",
                logger=logger,
                log_type="warn")
            break

        cfg = cfg.toDict()
        scheduler_type = cfg.pop("type", None)
        epochs = cfg.pop("epochs", 0)

        available_epochs = epochs_remainder
        epochs_remainder = epochs_remainder - epochs
        if epochs_remainder > 0 and idx == len(schedulers_cfg) - 1:
            log(msg=f"Attempt to set {epochs} epochs to the last scheduler but there are still {available_epochs} "
                f"remaining epochs, setting up {available_epochs} epochs to the last scheduler",
                logger=logger,
                log_type="warn")
            epochs = available_epochs
        elif epochs_remainder < 0:
            log(msg=f"Attempt to set {epochs} epochs to the scheduler but there are only {available_epochs} "
                f"remaining epochs, setting up {available_epochs} epochs for the scheduler",
                logger=logger,
                log_type="warn")
            epochs = available_epochs

        total_iters = epochs * steps_per_epoch
        if scheduler_type == "LinearLR":
            schedulers.append(lr_scheduler.LinearLR(optimizer, total_iters=total_iters, **cfg))
        elif scheduler_type == "CosineAnnealingLR":
            schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, **cfg))
        else:
            log(msg=f"Unrecongnized scheduler type `{scheduler_type}`", logger=logger, log_type="error")
            raise RuntimeError

        accumulative_steps = accumulative_steps + total_iters
        if idx < len(schedulers_cfg) - 1:
            milestones.append(accumulative_steps)

    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones
    )


def get_solver(model: VideoCLIP, cfg: Config, steps_per_epoch: int = 1, logger: Optional[Logger] = None):
    """
    Setup the optimizer and the learning rate scheduler

    Args:
        model: VideoCLIP model
        cfg:   Solver configuration
        steps_per_epoch: How many steps are in the training per each epoch
        logger: Logger object

    Return:
        A tuple with the optimizer, learning rate scheduler and criterion
    """
    # clip_params = []  # clip visual model parameters
    fusor_params = []  # fusion/temporal modeling model parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # if "visual" in name and "control_point" not in name:
        if "visual" in name or "logit_scale" in name:
            # clip_params.append(param)
            param.requires_grad = False
        # elif "logit_scale" in name:
        #     clip_params.append(param)
        elif "fusion_model" in name:
            fusor_params.append(param)

    optimizer = optim.AdamW(
        # [
        #     {"params": clip_params, "lr": cfg.lr * cfg.clip_ratio},
        #     {"params": fusor_params, "lr": cfg.lr}
        # ],
        fusor_params,
        betas=(0.9, 0.999),
        lr=cfg.lr,
        eps=1e-08,
        weight_decay=cfg.weight_decay
    )

    lr_scheduler = get_scheduler(cfg.lr_scheduler, optimizer, cfg.epochs, steps_per_epoch, logger)

    # - Loss function -
    criterion = get_criterion(cfg.criterion)

    return optimizer, lr_scheduler, criterion


def get_working_dir(cfg: dict, custom_name: str = "") -> Path:
    """ Setup working directory """
    dataset = cfg.get("dataset", {})
    network = cfg.get("network", {})

    exp_dir = "./exp_fewshot" if dataset.get("shot") else "./exp"

    dataset_name = dataset.get("name", "")
    network_arch = network.get("arch", "")
    if network_arch in _CLIP_ARCHS:
        network_arch = "clip_" + network_arch
    network_arch = network_arch.replace("/", "").lower()

    if custom_name:
        custom_name = custom_name + "-"

    return Path(
        exp_dir,
        dataset_name,
        network_arch,
        f"{custom_name}{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
    )


def create_new_dataset_config(old_cfg: dict, working_dir: Path, *args):
    """
    Creates a new config file from the provided config and the
    default values, and store this config to a "train_cfg.yaml" file
    under working_dir
    """
    new_dataset = dict()  # New dataset config
    new_cfg = old_cfg.copy()
    for d in args:
        new_cfg |= args

    new_dataset = old_cfg.get("dataset", {}) | new_dataset
    new_cfg["dataset"] = new_dataset

    with open(working_dir / "train_cfg.yaml", "w") as f:
        yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False, indent=4)


def get_model(cfg: Config, num_segments: int = 1, device: torch.device | str = "cpu", logger: Optional[Logger] = None):
    """
    Building full model

    If network precision is AMP the function will return the GradScaler and autocast
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    model, _ = clip.load(cfg.arch, device=device, jit=False)
    clip_state_dict = model.state_dict()

    # Video header
    video_head = VideoHead(cfg.video_head, clip_state_dict, cfg.dropout).to(device)
    # Load text features
    # TODO: check function name
    dtype = torch.float32 if cfg.precision in ["AMP", "FP32"] else torch.float16
    projector = load.text_projector(path=cfg.projector, dtype=dtype, device=device, logger=logger)
    # Load reductor
    try:
        reductor = load.reductor(path=cfg.reductor, dtype=dtype, device=device, logger=None)
    except Exception:
        reductor = None

    # Build full model
    full_model = VideoCLIP(model, video_head, num_segments, projector, reductor)
    # full_model = VideoCLIPConv(model, video_head, cfg.data.num_segments)

    # Set precision
    if cfg.precision in ["AMP", "FP32"]:
        full_model = full_model.float()
        full_model.text_projector = full_model.text_projector.to(dtype=dtype)
    elif cfg.precision == "FP16":
        full_model = full_model.half()
        full_model.text_projector = full_model.text_projector.to(dtype=dtype)

    log(msg=f" => Loaded VideoCLIP model with precision {dtype}", logger=logger, log_type="info")

    # breakpoint()
    ret = {"model": full_model, "scaler": None, "autocast": suppress()}
    if cfg.precision == "AMP":
        ret["scaler"] = GradScaler(device.type, enabled=True)  # Gradient scaler for automatic mixed precision (AMP)
        ret["autocast"] = torch_autocast(device_type=device.type)
        log(msg=" => GradScaler and autocast was set for AMP", logger=logger, log_type="info")

    return ret


def get_data_loaders(cfg_dataset: Config, cfg_dataloader: Config, seed: int, distributed: bool = False,
                     device: torch.device | str = "cpu", logger: Optional[Logger] = None) -> Tuple[DataLoader, DataLoader]:
    if not isinstance(device, torch.device):
        device = torch.device(device)
    # - Data augmentation -
    # TODO: Change parameter datatype
    transformations = get_augmentation(**cfg_dataset.toDict())

    default_params = dict()
    dataloaders = dict()
    # breakpoint()
    for key, value in transformations.items():
        dataset = VideoDataset(split=key, transform=value, logger=logger, **cfg_dataset.toDict())
        if key == "train":
            setup_few_shot_sampling(dataset, cfg_dataset, logger)
            sampler = get_sampler(dataset, distributed, True)
        else:
            sampler = get_sampler(dataset, distributed, False)
        default_params |= dataset.get_default_params()

        # For reproductibility
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        dataloaders[key] = DataLoader(
            dataset,
            **cfg_dataloader.toDict(),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=(key == "train"),  # Changed to True for stable batch sizes
            pin_memory=device.type == "cpu",
            worker_init_fn=seed_worker,
            generator=g
        )

    return dataloaders


def main(args):
    # Load config with safe loader
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = get_working_dir(cfg, args.custom_name)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    if args.distributed:
        logger = setup_logger(dist_rank=args.rank, name="Text4Var", output=working_dir)
    else:
        logger = setup_logger(dist_rank=0, name="Text4Var", output=working_dir)

    logger.info(f"Environment Versions:\n{'=' * 125}")
    logger.info(f" => Python: {sys.version}")
    logger.info(f" => PyTorch: {torch.__version__}")
    logger.info(f" => TorchVision: {torchvision.__version__}")
    logger.info(f" => TorchCodec: {torchcodec.__version__}\n{'=' * 125}")
    logger.info(" => CLI arguments: {}".format(*[f"--{v} {k}" for v, k in vars(args).items()]))
    logger.info(f"YAML config:\n{'=' * 125}\n{yaml.dump(
        cfg,
        default_flow_style=False,
        indent=4,
        allow_unicode=True,
        sort_keys=False
    )}{'=' * 125}")
    logger.info(f" => Working directory: {working_dir}")

    cfg = Config(cfg)
    # breakpoint()

    # Initialize distributed mode (or single GPU mode)
    # clip_utils.init_distributed_mode(args, logger)
    init_distributed_mode(args, logger)

    # Fix seed for reproducibility
    seed = cfg.seed + args.rank
    set_deterministic_seed(seed)

    # Setup debugger only if requested
    setup_debugger(args)

    # Device setup
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    # torch.set_default_device(device)
    logger.info(f" => Using device: {device}")

    # Get data loaders for each split in cfg
    d = get_data_loaders(cfg.dataset, cfg.dataloader, seed, args.distributed, logger=logger)
    train_loader = d["train"]
    val_loader = d["val"]
    # train_loader = None
    # val_loader = None

    # Load VideoCLIP model
    d = get_model(cfg.network, cfg.dataset.num_segments, device, logger)
    model = d["model"]
    scaler = d["scaler"]
    autocast = d["autocast"]

    # - Solver setup -
    optimizer, lr_scheduler, criterion = get_solver(model, cfg.solver)

    copy_files(args.rank, working_dir, [
        {"file": __file__},
        {"file": cfg.network.projector, "dest_name": "projector.pt"},
        {"file": cfg.network.reductor, "dest_name": "reductor.pt"}
        {"file": args.config, "dest_name": "train_config.yaml"}
    ], logger)

    # Load pretrained weights
    start_epoch = 0
    if cfg.network.pretrain:
        try:
            state_dict, epoch = load.weights(cfg.network.pretrain, device, logger)
            model.load_state_dict(state_dict)
            start_epoch = epoch + 1
        except Exception:
            raise RuntimeError

    # Wrap model for distributed training
    if args.distributed and device.type != "cpu":
        model = DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False  # Set to True if needed
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    best_prec1 = 0.0

    # Training loop
    logger.info(f"Starting training...\n{'=' * 125}")
    for epoch in range(start_epoch, cfg.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss = train(
            model=model,
            train_loader=train_loader,
            scaler=scaler,
            autocast=autocast,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            epoch=epoch,
            device=device,
            cfg=cfg,
            logger=logger
        )

        # Log loss
        if args.rank == 0:
            with open(f"{working_dir}/loss_log.txt", "a") as f:
                f.write(f"{epoch},{train_loss:.6f}\n")

        # Validation
        if (epoch + 1) % cfg.logging.eval_freq == 0:
            prec1 = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                cfg=cfg,
                logger=logger
            )

            if args.rank == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info(f"Testing: Current={prec1:.3f}, Best={best_prec1:.3f}")
                logger.info("Saving checkpoint...")
                # clip_utils.epoch_saving(epoch, model_without_ddp, optimizer, f"{working_dir}/last_model.pt")
                epoch_saving(epoch, model_without_ddp, optimizer, f"{working_dir}/last_model.pt")

                if is_best:
                    # Save best model checkpoint
                    # clip_utils.epoch_saving(epoch, model_without_ddp, optimizer, f"{working_dir}/best_model.pt")
                    epoch_saving(epoch, model_without_ddp, optimizer, f"{working_dir}/best_model.pt")
                    logger.info("*** New best model saved ***")
    logger.info("=" * 125)

    # Plot loss curve at the end
    if args.rank == 0:
        plot_loss_curve(f"{working_dir}/loss_log.txt", f"{working_dir}/loss_curve.png")


def train(model: VideoCLIP, train_loader: DataLoader, scaler: GradScaler, autocast, optimizer, criterion,
          epoch: int, lr_scheduler, cfg: Config, device: torch.device | str = "cpu", logger: Optional[Logger] = None):
    """ Train for one epoch """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    dtype = torch.float32 if cfg.network.precision in ["AMP", "FP32"] else torch.float16

    model.train()
    # breakpoint()
    for i, (images, _, list_id) in enumerate(train_loader):
        # Update learning rate
        # if cfg.solver.type != "monitor":
        #     if (i + 1) == 1 or (i + 1) % 10 == 0:
        #         lr_scheduler.step(epoch + i / len(train_loader))

        data_time.update(time.time() - end)

        # Move data to device
        # breakpoint()
        images = images.to(device, dtype=dtype, non_blocking=True)
        list_id = list_id.to(device, non_blocking=True)

        # Reshape images
        # images = images.view((-1, cfg.dataset.num_segments, 3) + images.size()[-2:])
        b, t, c, h, w = images.size()
        images = images.view(-1, c, h, w)

        # Forward pass with autocast for AMP if set
        with autocast:
            logits = model(images)
            loss = criterion(logits, list_id)

            loss_value = loss.item()
            loss = loss / cfg.solver.grad_accumulation_steps

            # Backward pass for AMP
            if scaler is not None:
                # AMP training
                scaler.scale(loss).backward()
                if (i + 1) % cfg.solver.grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    current_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if current_scale == scaler.get_scale():
                        lr_scheduler.step()
            else:
                # Standard training
                loss.backward()
                if (i + 1) % cfg.solver.grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

        # Update metrics
        losses.update(loss_value, logits.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if i % cfg.logging.print_freq == 0:
            cur_iter = epoch * len(train_loader) + i
            max_iter = cfg.solver.epochs * len(train_loader)
            eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

            log(
                msg=f"Epoch: [{' ' * (len(str(cfg.solver.epochs)) - len(str(epoch)))}{epoch}][{' ' * (len(str(len(train_loader))) - len(str(i)))}{i}/{len(train_loader)}], "
                f"lr: {optimizer.param_groups[-1]["lr"]:.2e}, "
                f"eta: {eta_str}\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})",
                logger=logger,
                log_type="info"
            )

    return losses.avg


def validate(model: VideoCLIP, val_loader: DataLoader, device: torch.device, cfg: Config, logger: Optional[Logger] = None):
    """Validate the model"""
    top1 = AverageMeter()
    top3 = AverageMeter()

    dtype = torch.float32 if cfg.network.precision in ["AMP", "FP32"] else torch.float16

    model.eval()
    with torch.no_grad():
        # breakpoint()
        for i, (images, _, class_id) in enumerate(val_loader):
            # Move data to device
            images = images.to(device, dtype=dtype, non_blocking=True)
            # images = images.to(device, non_blocking=True)
            class_id = class_id.to(device, non_blocking=True)

            # Reshape images
            # image = image.view((-1, cfg.dataset.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            logits = model(images)

            # Compute accuracy
            prec = accuracy(logits, class_id, topk=(1, 3))
            prec1 = reduce_tensor(prec[0], args)
            prec3 = reduce_tensor(prec[1], args)

            top1.update(prec1.item(), class_id.size(0))
            top3.update(prec3.item(), class_id.size(0))

            if i % cfg.logging.print_freq == 0:
                log(
                    msg=f"Test: [{' ' * (len(str(len(val_loader))) - len(str(i)))}{i}/{len(val_loader)}]\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@3 {top3.val:.3f} ({top3.avg:.3f})",
                    logger=logger,
                    log_type="info"
                )

    log(msg=f"Testing Results: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}", logger=logger, log_type="info")
    return top1.avg


if __name__ == "__main__":
    args = get_parser()
    main(args)

