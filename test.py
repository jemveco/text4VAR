import argparse
from logging import Logger
from pathlib import Path
import sys
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
import torchcodec
import torchvision
import yaml

from modules.video_clip import VideoCLIP
import train
from utils import load
from utils.clip_utils import AverageMeter, accuracy
from utils.config import Config
from utils.logger import setup_logger, log

matplotlib.use("Agg")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Global config file")
    parser.add_argument("--custom-name", type=str, default="", help="Custom name for the model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    args = parser.parse_args()
    return args


def validate(model: VideoCLIP, val_loader: DataLoader, cfg: Dict | Config, device: torch.device | str = "cpu", logger: Logger | None = None):
    """ Validate the model """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    top1 = AverageMeter()
    top3 = AverageMeter()

    dtype = torch.float32 if cfg.network.precision in ["AMP", "FP32"] else torch.float16

    all_probs = dict()
    # To compute Precision, Recall and F1-Score
    all_preds = list()
    all_labels = list()

    model.eval()
    with torch.no_grad():
        for i, (images, paths, list_ids) in enumerate(val_loader):
            # Move data to device
            images = images.to(device, dtype=dtype, non_blocking=True)
            list_ids = list_ids.to(device, non_blocking=True)

            # Reshape images
            b, t, c, h, w = images.size()
            images = images.view(-1, c, h, w)

            # Forward pass
            logits = model(images)

            # Get probs (softmax values)
            softmax_values = F.softmax(logits, dim=-1).cpu()
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu()

            for path, label_id, sfmax_v in zip(paths, list_ids, softmax_values):
                if path not in all_probs:
                    all_probs[path] = {"softmax": [], "label": label_id}

                all_probs[path]["softmax"].append(sfmax_v)

            all_preds.extend(preds.numpy())
            all_labels.extend(list_ids.cpu().numpy())

            # Compute accuracy
            acc = accuracy(logits, list_ids, topk=(1, 3))
            # breakpoint()
            top1.update(acc[0].item(), list_ids.size(0))
            top3.update(acc[1].item(), list_ids.size(0))

            if (i + 1) % cfg.logging.print_freq == 0 or (i + 1) == len(val_loader):
                log(msg=f"Test [{' ' * (len(str(len(val_loader))) - len(str(i + 1)))}{i + 1}/{len(val_loader)}]: "
                    f"Top@1 {top1.val:.2f} ({top1.avg:.2f}), "
                    f"Top@3 {top3.val:.2f} ({top3.avg:.2f})",
                    logger=logger,
                    log_type="info")

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    log(msg=f"Final Results: "
        f"Top@1: {top1.avg:.2f}, "
        f"Top@3: {top3.avg:.2f}, "
        f"Precision: {precision:.2f}, "
        f"Recall: {recall:.2f}, "
        f"F1-score: {f1:.2f}",
        logger=logger,
        log_type="info")

    top1_sv = AverageMeter()  # Top 1 soft voting
    top3_sv = AverageMeter()  # Top 3 soft voting
    for path, values in all_probs.items():
        sv_values = torch.stack(values["softmax"]).mean(dim=0)
        sv_logits = sv_values.unsqueeze(0)
        sv_target = torch.tensor([values["label"]])
        acc = accuracy(sv_logits, sv_target, topk=(1, 3))
        top1_sv.update(acc[0].item(), 1)
        top3_sv.update(acc[1].item(), 1)

    log(msg="Soft Voting Test Results: "
        f"Top@1: {top1_sv.avg:.2f}, "
        f"Top@3: {top3_sv.avg:.2f}",
        logger=logger,
        log_type="info")

    top1_hv = AverageMeter()  # Top 1 hard voting
    top3_hv = AverageMeter()  # Top 3 hard voting
    for path, values in all_probs.items():
        hv_values = torch.stack(values["softmax"])
        num_classes = hv_values.shape[1]
        segment_votes = hv_values.argmax(dim=1)
        vote_counts = torch.bincount(segment_votes, minlength=num_classes).float()

        hv_logits = vote_counts.unsqueeze(0)
        hv_target = torch.tensor([values["label"]])
        acc = accuracy(hv_logits, hv_target, topk=(1, 3))
        top1_hv.update(acc[0].item(), 1)
        top3_hv.update(acc[1].item(), 1)

    log("Hard Voting Test Results: "
        f"Top@1: {top1_hv.avg:.2f},"
        f"Top@3: {top3_hv.avg:.2f}",
        logger=logger,
        log_type="info")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    similitud = cosine_similarity(cm)
    kmeans = KMeans(n_clusters=3)  # Group clusters by KMeans
    kmeans.fit(similitud)
    clusters = kmeans.labels_

    labels = pd.read_csv(cfg.dataset.label_list, index_col="label").index.tolist()
    # Creates a list of grouped labels
    grouped_labels = {}
    for idx, label in enumerate(labels):
        cluster = clusters[idx]
        if cluster not in grouped_labels:
            grouped_labels[cluster] = []

        grouped_labels[cluster].append(label)

    # Sort the labels according with the groups found by KMeans
    new_order = []
    for cluster in sorted(grouped_labels.keys()):
        new_order.extend(grouped_labels[cluster])

    # Sort the confusion matrix according with the new order
    sorted_cm = cm[np.ix_([labels.index(label) for label in new_order],
                          [labels.index(label) for label in new_order])]
    disp = ConfusionMatrixDisplay(confusion_matrix=sorted_cm, display_labels=new_order)

    return top1.avg, top3.avg, all_preds, all_labels, disp


def main(args):
    # Load cfg
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Output directory
    working_dir = Path(cfg["network"]["weights"]).resolve().parent / "results" / args.custom_name
    # Create output directory
    working_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(dist_rank=0, name="Text4Var(Test)", output=working_dir)

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

    # Device setup
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    # torch.set_default_device(device)
    logger.info(f" => Using device: {device}")

    # Get data loaders for each split in cfg
    d = train.get_data_loaders(cfg.dataset, cfg.dataloader, cfg.seed, logger=logger)
    val_loader = d["val"]

    # Load VideoCLIP model
    d = train.get_model(cfg.network, cfg.dataset.num_segments, device, logger)
    model = d["model"]

    try:
        state_dict, epoch = load.weights(cfg.network.weights, device, logger)
        # for name, param in model.named_parameters(): print(name)
        # breakpoint()
        model.load_state_dict(state_dict)
    except Exception:
        raise RuntimeError

    log(f"Starting validation...\n{'=' * 125}", logger, "info")

    dummy_input = torch.randn(1 * cfg.dataset.num_segments, 3, 224, 224).to(device=device)
    flops_counter = FlopCounterMode(display=False)

    with flops_counter:
        model(dummy_input)

    log(f"Model FLOPs: {flops_counter.get_total_flops() / 1e9:.2f} GFLOPs", logger, "info")
    log(f"FLOPs during inference (theorically): {(3 * flops_counter.get_total_flops()) /  1e9:.2f} GFLOPs", logger, "info")

    # Paramater number
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params}", logger, "info")
    log(f"Trainable parameters: {trainable_params}", logger, "info")

    # Run validation
    top1, top3, preds, labels, cm = validate(model, val_loader, cfg, device, logger)

    plt.figure(figsize=(12, 10))
    cm.plot(cmap="viridis", xticks_rotation="vertical", include_values=False)
    plt.title("Confusion Matrix - Validation")
    plt.tight_layout()

    # Save confusion matrix
    working_dir.mkdir(exist_ok=True)
    plt.savefig(working_dir / "confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Save predictions
    np.savez(working_dir / "validation_predictions.npz",
             predictions=preds,
             labels=labels,
             top1_accuracy=top1,
             top3_accuracy=top3)

    log("Validation completed", logger, "info")
    log(f"Results saved to: {working_dir}", logger, "info")


if __name__ == "__main__":
    args = get_parser()
    main(args)

