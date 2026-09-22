import os
import yaml
import csv
import clip
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from typing import Optional

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
    parser.add_argument("--pre-extracted-feats", default=None, help="If set new features won't be extracted but '--model-type' and '--model' will be used to save the model")
    parser.add_argument("--model-type", choices=["bert", "clip", "mpnet"], default="clip")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--reduction-method", choices=["pca", "nca", "lda", "tsne", "no_reduce"], default="no_reduce")  # Not use TSNE. TODO: Remove everything about TSNE
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--centroid-method", choices=["mean", "lda", "kmeans"], default="mean")
    parser.add_argument("--truncate-input", type=int, default=0)

    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mad_outlier_detection(X, y, centroids, distance="euclidean"):
    X = np.asarray(X)
    y = np.asarray(y)

    outlier_mask = np.zeros(len(X), dtype=bool)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        vecs = X[idx]
        centroid = centroids[cls]

        if distance == "euclidean":
            distances = np.linalg.norm(vecs - centroid, axis=1)
        elif distance == "cosine":
            # use 1 - cosine_similarity
            sims = cosine_similarity(vecs, centroid.reshape(1, -1)).reshape(-1,)
            distances = 1.0 - sims
        elif distance == "angular":
            dots = np.clip(vecs.dot(centroid), -1.0, 1.0)
            distances = np.arccos(dots)
        else:
            logging.error(f"Distance type '{distance}' unknown")
            raise ValueError("Distance type unknown")

        median = np.median(distances)
        mad = np.median(np.abs(distances - median)) + 1e-12

        # Standard rule: |x - median| / MAD > 3.5
        z_scores = np.abs(distances - median) / mad
        outliers = z_scores > 3.5

        outlier_mask[idx[outliers]] = True

    return outlier_mask


def get_centroids(embeddings, labels, method="mean", normalize_centroids=False):
    assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels should be the same size"

    # print(f"[INFO] Computing centroids with {method}")
    logging.info(f"Computing centroids with '{method}'")

    centroids = {}
    u_labels = np.unique(labels)
    if method == "mean":
        for cls in u_labels:
            cls_points = embeddings[labels == cls]
            centroid = cls_points.mean(axis=0)
            if normalize_centroids:
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroids[cls] = centroid

        return centroids
    elif method == "lda":
        lda = LDA()
        lda.fit(embeddings, labels)
        x = lda.coef_
        # for cls, centroid in zip(u_labels, lda.coef_):
        #     centroids[cls] = centroid
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=len(u_labels), random_state=0, n_init="auto")
        x = kmeans.fit(embeddings).cluster_centers_
        # for cls, centroid in zip(u_labels, lda.coef_):
        #     ce
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")

    for cls, centroid in zip(u_labels, x):
        centroids[cls] = centroid

    return centroids


def analysis(centroids, embeddings, labels, save_dir, distance="euclidean", save_name=None):
    assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels should be the same size"
    os.makedirs(save_dir, exist_ok=True)

    intra = {}
    inter = {}

    # print(f"[INFO] Starting {distance} distance analysis")
    logging.info(f"Starting '{distance}' distance analysis")
    if distance == "euclidean":
        for cls, centroid in centroids.items():
            cluster = embeddings[labels == cls]

            # Intra-class (mean euclidean distance to centroid)
            distances = np.linalg.norm(centroid - cluster, axis=1, keepdims=True)
            intra[cls] = np.mean(distances)
            # Inter class distances
            inter[cls] = {k: np.linalg.norm(centroid - v) for k, v in centroids.items() if k != cls}
            print(f" => {cls} intraclass value: {intra[cls]}")
            print(f" => {cls} interclass values: {inter[cls]}")
    elif distance == "cosine":
        for cls, centroid in centroids.items():
            cluster = embeddings[labels == cls]
            sims = cosine_similarity(cluster, centroid.reshape(1, -1)).reshape(-1,)
            distances = 1.0 - sims
            intra[cls] = distances.mean()
            inter[cls] = {k: 1.0 - float(np.dot(centroid, v)) for k, v in centroids.items() if k != cls}
    elif distance == "angular":
        for cls, centroid in centroids.items():
            cluster = embeddings[labels == cls]
            dots = np.clip(cluster.dot(centroid), -1.0, 1.0)
            distances = np.arccos(dots)
            intra[cls] = distances.mean()
            inter[cls] = {k: float(np.arccos(np.clip(np.dot(centroid, v), -1.0, 1.0))) for k, v in centroids.items() if k != cls}
    else:
        raise ValueError("Distance type unknown")

    metric = "euclidean" if distance == "euclidean" else "cosine"
    try:
        silhouette = silhouette_score(embeddings, labels, metric=metric)
        if save_name is not None:
            with open(f"{save_dir}/silhouette.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([save_name, silhouette])

        # print(f"[INFO] silhouette (using {metric} distance as metric):", silhouette)
        logging.info(f"silhouette (using '{metric}' distance as metric): {silhouette}")

    except Exception as e:
        silhouette = None
        # print("[WARN] silhouette score failed:", e)
        logging.warning(f"silhouette score failed: {e}")

    if save_name is not None:
        # os.makedirs("embeddings/info", exist_ok=True)
        with open(f"{save_dir}/{save_name}_intra.csv", "w", newline="") as file:
            writer = csv.writer(file)
            for key, value in intra.items():
                writer.writerow([key, value])

        for cls, dists in inter.items():
            with open(f"{save_dir}/{save_name}_{cls}_inter.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for key, value in dists.items():
                    writer.writerow([key, value])


def save_centroids(centroids, embeddings, labels, name: Optional[str] = None, dir="embeddings/"):
    assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels should be the same size"

    centroids_dir = os.path.abspath(dir)
    os.makedirs(centroids_dir, exist_ok=True)

    _name = "centroids_" + name if name is not None else "centroids"
    logging.info(f"Saving centroids and full data ({{ class: centroid, ... }} dictionary) to {centroids_dir}/{_name}.pt")

    torch_e = {str(k): torch.from_numpy(v) for k, v in centroids.items()}
    torch.save(torch_e, f"{centroids_dir}/{_name}.pt")

    data = {}
    for label in np.unique(labels):
        data[str(label)] = torch.from_numpy(embeddings[labels == label])

    _name = "full_embeddings_" + name if name is not None else "full_embeddings"
    logging.info(f"Saving all extracted embeddings ({{ class: centroid, ... }} dictionary) to {centroids_dir}/{_name}.pt")
    torch.save(data, f"{centroids_dir}/{_name}.pt")


def plot_embeddings(centroids, embeddings, labels, distance="euclidean", save_name=None, save_dir="embeddings/figures"):
    save_dir = os.path.abspath(save_dir)
    # Embeddings + centroids (for visualization)
    all_embeddings = np.vstack([embeddings] + [c.reshape(1, -1) for c in centroids.values()])
    dimension = all_embeddings[0].shape[0]
    centroid_names = [f"centroid_{cls}" for cls in centroids.keys()]
    all_labels = np.concatenate([labels, np.array(centroid_names)])

    umap_proj = umap.UMAP(n_neighbors=20, min_dist=0.1, metric=distance, random_state=0).fit_transform(all_embeddings)
    if distance == "cosine":
        D = pairwise_distances(all_embeddings, metric="cosine")  # NxN
        tsne_proj = TSNE(n_components=2, init="random", metric="precomputed", perplexity=30, random_state=0).fit_transform(D)
    else:
        tsne_proj = TSNE(n_components=2, metric="euclidean", perplexity=30, random_state=0).fit_transform(all_embeddings)

    if save_name is not None:
        titles = [f"UMAP {save_name} ({distance}) - CLIP space ({dimension})", f"T-SNE {save_name} ({distance}) - CLIP space ({dimension})"]
    else:
        titles = [f"UMAP ({distance}) - CLIP space ({dimension})", f"T-SNE ({distance}) - CLIP space ({dimension})"]

    colors = np.array(["red", "green", "blue", "cyan", "magenta", "yellow", "black", "orange", "purple", "brown", "gray", "pink", "beige"])
    _, ax = plt.subplots(1, 2, figsize=(16, 8))
    for proj_idx, proj in enumerate([umap_proj, tsne_proj]):
        for c_idx, cls in enumerate(np.unique(labels)):
            idx = np.where(all_labels == cls)[0]
            ax[proj_idx].scatter(proj[idx, 0], proj[idx, 1], s=10, label=cls, alpha=0.6, c=colors[c_idx])

        # Centroids con etiquetas
        for i, (cls_key, cls_name) in enumerate(zip(centroids.keys(), centroid_names)):
            c_idx = len(labels) + i
            ax[proj_idx].scatter(proj[c_idx, 0], proj[c_idx, 1], s=200, marker="X",
                                 edgecolor="black", linewidths=2, zorder=5, c=colors[i])
            # Agregar etiqueta de texto al lado del centroide
            ax[proj_idx].annotate(cls_key,
                                  xy=(proj[c_idx, 0], proj[c_idx, 1]),
                                  xytext=(5, 5), textcoords="offset points",
                                  fontsize=10, fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="black"))

        ax[proj_idx].legend(loc="best", framealpha=0.9)
        ax[proj_idx].set_title(f"{titles[proj_idx]} ", fontsize=12, fontweight="bold")
        ax[proj_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{save_name}_{distance}_space.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main(args):
    save_dir = os.path.abspath(args.save_dir)
    save_dir = f"{save_dir}/{args.model_type}/{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Configure the root logger to write to a file and set the level to DEBUG
    logging.basicConfig(
        filename=f'{save_dir}/log.txt', level=logging.DEBUG,
        format='[%(name)s - %(asctime)s] %(levelname)s: %(message)s'
    )
    logging.getLogger(__name__)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info(f"Saving logs to '{save_dir}/log.txt'")

    projector_dir = f"{save_dir}/projector"
    os.makedirs(projector_dir, exist_ok=True)

    reductor_dir = f"{save_dir}/reductor"
    os.makedirs(reductor_dir, exist_ok=True)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if args.pre_extracted_feats is None:
        # if args.pre_extracted_feats != None:
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
                no_fit[cls_name] = []

                for video in cls_data:
                    video_name = video["name"]
                    no_fit_sentences = []

                    for sentence in video["sentences"]:
                        if sentence["classification"] != "abnormal":
                            continue
                        text = f"{cls_name}: {sentence['text']}"

                        try:
                            if args.truncate_input:
                                if args.model_type == "bert":
                                    tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE], return_tensors="pt").to(device)
                                elif args.model_type == "clip":
                                    tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE]).to(device)
                                elif args.model_type == "mpnet":
                                    tokens = tokenizer(text[:MAXIMUM_CONTEXT_SIZE], padding=True, return_tensors="pt").to(device)

                                if len(text) > MAXIMUM_CONTEXT_SIZE:
                                    no_fit_sentences.append({"sentence": text, "id": sentence["id"]})
                            else:
                                if args.model_type == "bert":
                                    tokens = tokenizer(text, return_tensors="pt").to(device)
                                elif args.model_type == "clip":
                                    tokens = tokenizer(text).to(device)
                                elif args.model_type == "mpnet":
                                    tokens = tokenizer(text, padding=True, return_tensors="pt").to(device)

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
                            logging.warning(f"Ignoring sentence in '{video_name}': {e}")
                            no_fit_sentences.append({"sentence": text, "id": sentence["id"]})
                    no_fit[cls_name].append({"name": video_name, "sentences": no_fit_sentences})

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        data = {}
        for ul in np.unique(labels):
            data[str(ul)] = torch.from_numpy(embeddings[labels == ul])
        # torch.save({k: v for ul in np.unique(labels) v = embeddings[ul == labels]}, f"{save_dir}/feats_per_class.pt")
        torch.save(data, f"{save_dir}/feats_per_class.pt")
        del data
    else:
        # print(f"Pre extracted features: {args.pre_extracted_feats}")
        logging.info(f"Loading pre-extracted features from: {args.pre_extracted_feats}")
        # data = np.load(args.pre_extracted_feats)
        data = torch.load(args.pre_extracted_feats, map_location="cpu")
        embeddings = []
        labels = []
        for label, embedding in data.items():
            embeddings.append(embedding.numpy())
            labels += [label] * embedding.size()[0]

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)

    # save_dir = f"embeddings/{args.model_type}/{args.reduction_method}/{args.centroid_method}"
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
            n_classes = len(np.unique(labels))
            max_components = n_classes - 1
            if args.dimension > max_components:
                # print(f"[WARN] LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
                logging.warning(f"LDA can have at most {max_components} components (n_classes - 1). Adjusting dimension from {args.dimension} to {max_components}")
                args.dimension = max_components
            reductor = LDA(n_components=args.dimension)

        embeddings = reductor.fit_transform(embeddings, labels)
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

        centroids = get_centroids(embeddings, labels, args.centroid_method)
        save_centroids(centroids, embeddings, labels, f"raw_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}", projector_dir)
        del reductor
    else:
        centroids = get_centroids(embeddings, labels, args.centroid_method)
        save_centroids(centroids, embeddings, labels, f"raw_{args.model_type}_{args.model.replace('/', '')}_nr_{args.centroid_method}", projector_dir)
    plot_embeddings(centroids, embeddings, labels, distance="cosine", save_name="raw", save_dir=f"{save_dir}/figures")

    standardized_embeddings = StandardScaler().fit_transform(embeddings)  # Standardized embeddings
    normalized_embeddings = l2_normalize(embeddings)  # L2 normalized embeddings

    standardized_centroids = get_centroids(standardized_embeddings, labels, args.centroid_method)
    # print("[INFO] Standardized embeddings analysis")
    logging.info("Standardized embeddings analysis")
    analysis(standardized_centroids, standardized_embeddings, labels, f"{save_dir}/info", distance="euclidean", save_name="standardized")
    plot_embeddings(standardized_centroids, standardized_embeddings, labels, distance="euclidean", save_name="standardized", save_dir=f"{save_dir}/figures")

    # if args.reduction_method != "no_reduce" and args.dimension < embeddings[0].shape[0]:
    if args.reduction_method != "no_reduce":
        save_centroids(standardized_centroids, standardized_embeddings, labels, f"standardized_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}", projector_dir)
    else:
        save_centroids(standardized_centroids, standardized_embeddings, labels, f"standardized_{args.model_type}_{args.model.replace('/', '')}_nr_{args.centroid_method}", projector_dir)

    normalized_centroids = get_centroids(normalized_embeddings, labels, args.centroid_method, normalize_centroids=True)
    # print("[INFO] Normalized embeddings analysis")
    logging.info("Normalized embeddings analysis")
    analysis(normalized_centroids, normalized_embeddings, labels, f"{save_dir}/info", distance="cosine", save_name="normalized")
    plot_embeddings(normalized_centroids, normalized_embeddings, labels, distance="cosine", save_name="normalized", save_dir=f"{save_dir}/figures")
    # if args.reduction_method != "no_reduce" and args.dimension < embeddings[0].shape[0]:
    if args.reduction_method != "no_reduce":
        save_centroids(normalized_centroids, normalized_embeddings, labels, f"normalized_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}", projector_dir)
    else:
        save_centroids(normalized_centroids, normalized_embeddings, labels, f"normalized_{args.model_type}_{args.model.replace('/', '')}_nr_{args.centroid_method}", projector_dir)

    # print("[INFO] Starting standardized embeddings MAD outlier detection")
    logging.info("Starting standardized embeddings MAD outlier detection")
    standardized_outlier_mask = mad_outlier_detection(standardized_embeddings, labels, standardized_centroids)
    standardized_embeddings = standardized_embeddings[~standardized_outlier_mask]
    mad_labels = labels[~standardized_outlier_mask]
    del standardized_outlier_mask
    standardized_centroids = get_centroids(standardized_embeddings, mad_labels, args.centroid_method)
    # print("[INFO] Standardized embeddings without outliers analysis")
    logging.info("Standardized embeddings without outliers analysis")
    analysis(standardized_centroids, standardized_embeddings, mad_labels, f"{save_dir}/info", distance="euclidean", save_name="standardized_no_outliers")
    plot_embeddings(standardized_centroids, standardized_embeddings, mad_labels, distance="euclidean", save_name="standardized_no_outliers", save_dir=f"{save_dir}/figures")
    # if args.reduction_method != "no_reduce" and args.dimension < embeddings[0].shape[0]:
    if args.reduction_method != "no_reduce":
        save_centroids(standardized_centroids, standardized_embeddings, mad_labels, f"standardized_no-outlier_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}", projector_dir)
    else:
        save_centroids(standardized_centroids, standardized_embeddings, mad_labels, f"standardized_no-outlier_{args.model_type}_{args.model.replace('/', '')}_nr_{args.centroid_method}", projector_dir)

    # print("[INFO] Starting normalized embeddings MAD outlier detection")
    logging.info("Starting normalized embeddings MAD outlier detection")
    normalized_outlier_mask = mad_outlier_detection(normalized_embeddings, labels, normalized_centroids, "cosine")
    normalized_embeddings = normalized_embeddings[~normalized_outlier_mask]
    mad_labels = labels[~normalized_outlier_mask]
    del normalized_outlier_mask
    normalized_centroids = get_centroids(normalized_embeddings, mad_labels, args.centroid_method, normalize_centroids=True)
    # print("[INFO] Normalized embeddings without outliers analysis")
    logging.info("Normalized embeddings without outliers analysis")
    analysis(normalized_centroids, normalized_embeddings, mad_labels, f"{save_dir}/info", distance="cosine", save_name="normalized_no_outliers")
    plot_embeddings(normalized_centroids, normalized_embeddings, mad_labels, distance="cosine", save_name="normalized_no_outliers", save_dir=f"{save_dir}/figures")
    # if args.reduction_method != "no_reduce" and args.dimension < embeddings[0].shape[0]:
    if args.reduction_method != "no_reduce":
        save_centroids(normalized_centroids, normalized_embeddings, mad_labels, f"normalized_no-outlier_{args.model_type}_{args.model.replace('/', '')}_{args.reduction_method}_{args.centroid_method}", projector_dir)
    else:
        save_centroids(normalized_centroids, normalized_embeddings, mad_labels, f"normalized_no-outlier_{args.model_type}_{args.model.replace('/', '')}_nr_{args.centroid_method}", projector_dir)


if __name__ == "__main__":
    args = get_parser()
    main(args)
