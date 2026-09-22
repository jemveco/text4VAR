import os
import yaml
import csv
import clip
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from typing import Optional, Dict

from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.cluster import KMeans
# import pickle

import matplotlib.pyplot as plt
import umap

import datetime

import logging


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="embeddings")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--model-type", choices=["bert", "clip", "mpnet"], default="clip")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--reduction-method", choices=["pca", "nca", "lda", "tsne", "no_reduce"], default="no_reduce")  # Not use TSNE. TODO: Remove everything about TSNE
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--truncate-input", type=int, default=0)

    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def save_matrix(data, name: Optional[str] = None, dir="embeddings/"):
    save_dir = os.path.abspath(dir)
    os.makedirs(save_dir, exist_ok=True)

    name = "centroids_" + name if name is not None else "centroids"
    logging.info(f"Saving projection matrix and full data ({{ class: embedding, ... }} dictionary) to {save_dir}/projector_{name}.pt")

    #torch_e = {str(k): torch.from_numpy(v) for k, v in data.items()}
    torch.save(data, f"{save_dir}/{name}.pt")


def reduce(embeddings, labels, reduction_method, save_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    logging.info(f"Reducing embeddings dimensions from {embeddings[0].shape[0]} to {args.dimension} with {args.reduction_method}")
    if reduction_method == "tsne":
        reductor = TSNE(n_components=args.dimension, method="exact", max_iter=500, n_iter_without_progress=150, n_jobs=2, random_state=0, perplexity=25)
    elif reduction_method == "pca":
        reductor = PCA(n_components=args.dimension, whiten=True, random_state=0)
    elif reduction_method == "nca":
        reductor = NCA(n_components=args.dimension, init="pca", random_state=0)
    elif reduction_method == "lda":
        # LDA can have at most n_classes - 1 components
        n_classes = len(labels)
        max_components = n_classes - 1
        if args.dimension > max_components:
            # print(f"[WARN] LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
            logging.warning(f"LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
            args.dimension = max_components
        reductor = LDA(n_components=args.dimension)

    # embeddings = reductor.fit_transform(embeddings, labels)

    # Only for PCA
    num_repeats = 50
    augmented_embeddings = np.repeat(original_embeddings, num_repeats, axis=0)
    noise_level = 0.01
    noise = np.random.normal(0, noise_level, augmented_embeddings.shape)
    augmented_embeddings = augmented_embeddings + noise # Shape: (650, 768)
    augmented_embeddings = reductor.fit(augmented_embeddings, labels)
    embeddings = reductor.transform(augmented_embeddings)

    if reduction_method == "pca":
        model = {"reductor": torch.from_numpy(reductor.components_), "centerer": torch.from_numpy(reductor.mean_), "reductor_type": "pca"}
    elif reduction_method == "nca":
        model = {"reductor": torch.from_numpy(reductor.components_), "centerer": None, "reductor_type": "nca"}
    elif reduction_method == "lda":
        model = {"reductor": torch.from_numpy(reductor.scalings_), "centerer": torch.from_numpy(reductor.xbar_), "reductor_type": "lda"}

    if save_dir:
        torch.save(model, f"{save_dir}/{args.dimension}.pt")

    data = {}
    for idx, label in enumerate(labels):
        data[str(label)] = torch.from_numpy(embeddings[idx])
    return data


def main(args):
    save_dir = os.path.abspath(args.save_dir)
    save_dir = f"{save_dir}/single_description/{args.model_type}/{args.model.replace('/', '')}_{args.reduction_method}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Configure the root logger to write to a file and set the level to DEBUG
    logging.basicConfig(
        filename=f'{save_dir}/log.txt', level=logging.DEBUG,
        format='[%(name)s - %(asctime)s] %(levelname)s: %(message)s'
    )
    logging.getLogger(os.path.basename(__file__).replace(os.path.splitext(__file__)[1], ""))
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info(f"Running from file '{__file__}'")
    logging.info(f"Saving logs to '{save_dir}/log.txt'")

    projector_dir = f"{save_dir}/projector"
    os.makedirs(projector_dir, exist_ok=True)

    reductor_dir = f"{save_dir}/reductor"
    os.makedirs(reductor_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == "bert":
        model = BertModel.from_pretrained(args.model, output_hidden_states=False)
        tokenizer = BertTokenizer.from_pretrained(args.model)
        if args.truncate_input:
            MAXIMUM_CONTEXT_SIZE = 512
    elif args.model_type == "clip":
        model, _ = clip.load(args.model, device=device)
        tokenizer = clip.tokenize
        if args.truncate_input:
            MAXIMUM_CONTEXT_SIZE = 77
    elif args.model_type == "mpnet":
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.truncate_input:
            MAXIMUM_CONTEXT_SIZE = 512

    with open(args.yaml) as f:
        data = yaml.safe_load(f)

    no_fit = {}
    embeddings = []
    labels = []

    model = model.to(device)
    model.eval()
    # print(f"[INFO] Extracting embeddings with {args.model_type}")
    logging.info(f"Extracting embeddings with {args.model_type}")

    with torch.no_grad():
        for cls_name, cls_data in data.items():
            no_fit[cls_name] = ""

            if args.augment:
                text = f"{cls_name}: {cls_data}"
            else:
                text = cls_data

            try:
                if args.truncate_input:
                    if args.model_type == "bert":
                        tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE], return_tensors="pt").to(device)
                    elif args.model_type == "clip":
                        tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE]).to(device)
                    elif args.model_type == "mpnet":
                        tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE], padding=True, return_tensors="pt").to(device)

                    if len(text) > MAXIMUM_CONTEXT_SIZE:
                        no_fit[cls_name] = cls_data
                else:
                    if args.model_type == "bert":
                        tokens = tokenizer(text, return_tensors="pt").to(device)
                    elif args.model_type == "clip":
                        tokens = tokenizer(text).to(device)
                    elif args.model_type == "mpnet":
                        tokens = tokenizer(text, padding=True, return_tensors="pt").to(device)

                # Encoding descriptions
                if args.model_type == "bert":
                    embeddings.append(model(**tokens)[-1][0].to("cpu").numpy())
                elif args.model_type == "clip":
                    embeddings.append(model.encode_text(tokens).to("cpu").numpy()[0])
                elif args.model_type == "mpnet":
                    model_output = model(**tokens)
                    # Perform pooling
                    sentence_embeddings = mean_pooling(model_output, tokens["attention_mask"])
                    # Normalize embeddings
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                    embeddings.append(sentence_embeddings[0].to("cpu").numpy())

                labels.append(cls_name)
            except Exception as e:
                # print(f"[WARN] Error processing sentence in {video_name}: {e}")
                logging.warning(f"Ignoring sentence in '{cls_name}': {e}")
                no_fit[cls_name] = cls_data

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    data = {}
    for idx, label in enumerate(labels):
        data[str(label)] = torch.from_numpy(embeddings[idx])
    torch.save(data, f"{save_dir}/feats_per_class_without_reduction.pt")

    if args.reduction_method != "no_reduce" and args.dimension < embeddings[0].shape[0]:
        # print(f"[INFO] Reducing embeddings dimensions from {embeddings[0].shape[0]} to {args.dimension} with {args.reduction_method}")
        logging.info(f"Reducing embeddings dimensions from {embeddings[0].shape[0]} to {args.dimension} with {args.reduction_method}")
        if args.reduction_method == "tsne":
            reductor = TSNE(n_components=args.dimension, method="exact", max_iter=500, n_iter_without_progress=150, n_jobs=2, random_state=0, perplexity=25)
        elif args.reduction_method == "pca":
            reductor = PCA(n_components=args.dimension, whiten=True, random_state=0)
        elif args.reduction_method == "nca":
            reductor = NCA(n_components=args.dimension, init="pca", random_state=0)
        elif args.reduction_method == "lda":
            # LDA can have at most n_classes - 1 components
            n_classes = len(labels)
            max_components = n_classes - 1
            if args.dimension > max_components:
                # print(f"[WARN] LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
                logging.warning(f"LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
                args.dimension = max_components
            reductor = LDA(n_components=args.dimension)

        # embeddings = reductor.fit_transform(embeddings, labels)
        
        # Only for PCA
        num_repeats = 50
        augmented_embeddings = np.repeat(embeddings, num_repeats, axis=0)
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, augmented_embeddings.shape)
        augmented_embeddings = augmented_embeddings + noise # Shape: (650, 768)
        augmented_embeddings = reductor.fit(augmented_embeddings, labels)
        embeddings = reductor.transform(embeddings)
        # with open(f"{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}-model_{args.dimension}.pt", "wb") as f:
        #     pickle.dump(reductor, f)
        # model = {"components": torch.from_numpy(reductor.components_), "mean": torch.from_numpy(reductor.mean_)}
        if args.reduction_method == "pca":
            model = {"reductor": torch.from_numpy(reductor.components_), "centerer": torch.from_numpy(reductor.mean_), "reductor_type": "pca"}
        elif args.reduction_method == "nca":
            model = {"reductor": torch.from_numpy(reductor.components_), "centerer": None, "reductor_type": "nca"}
        elif args.reduction_method == "lda":
            model = {"reductor": torch.from_numpy(reductor.scalings_), "centerer": torch.from_numpy(reductor.xbar_), "reductor_type": "lda"}
        torch.save(model, f"{reductor_dir}/{args.dimension}.pt")

        # Only for PCA
        data = {}
        for idx, label in enumerate(labels):
            data[str(label)] = torch.from_numpy(embeddings[idx])
        # torch.save(data, f"{save_dir}/feats_per_class_without_reduction.pt")
        save_matrix(data, f"raw_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}", projector_dir)
        del reductor
    else:
        save_matrix(data, f"raw_{args.model_type}_{args.model.replace('/', '')}_nr", projector_dir)
    # plot_embeddings(centroids, embeddings, labels, distance="cosine", save_name="raw", save_dir=f"{save_dir}/figures")


if __name__ == "__main__":
    args = get_parser()
    main(args)
