import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# 1. Main Clustering Function

def cluster_embeddings(
    embeddings,
    method="dbscan",
    normalize_emb=True,
    use_pca=False,
    pca_dim=50,
    merge_threshold=0.7,
    max_clusters=10,
    avg_images_per_person=4,
    adaptive_eps=True,
    **kwargs
):
    """
    Cluster embeddings into groups (persons).

    Args:
        embeddings (np.ndarray): Shape (N, D) array of embeddings.
        method (str): ["dbscan", "kmeans", "agglomerative"].
        normalize_emb (bool): Whether to L2-normalize embeddings.
        use_pca (bool): Whether to apply PCA before clustering.
        pca_dim (int): Dimension for PCA reduction.
        merge_threshold (float): Cosine similarity threshold for merging small clusters.
        adaptive_eps (bool): For DBSCAN, automatically adjust eps from median distance.
        kwargs: Extra clustering params.

    Returns:
        labels (np.ndarray): Cluster labels for each embedding.
    """

    X = embeddings

    # Step 1: Normalize (L2)
    if normalize_emb:
        X = normalize(X)

    # Step 2: Optional PCA (improves clustering stability)
    if use_pca and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)

    # Step 3: Clustering
    if method == "dbscan":
        eps = kwargs.get("eps", 0.3)
        min_samples = kwargs.get("min_samples", 2)

        if adaptive_eps:
            eps = estimate_optimal_eps(X)
            print(f"[DBSCAN] Adaptive eps chosen: {eps:.3f}")

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
        labels = clustering.labels_

    elif method in ["kmeans", "agglomerative"]:
        estimated_clusters = max(2, len(embeddings) // avg_images_per_person)
        n_clusters = min(estimated_clusters, max_clusters)
        if method == "kmeans":
            clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        else:
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        labels = clustering.labels_

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Step 4: Merge similar clusters (optional)
    labels = merge_small_clusters(X, labels, threshold=merge_threshold)

    return labels


# 2. Adaptive eps estimator (DBSCAN only)

def estimate_optimal_eps(X, sample_size=200):
    """
    Estimate DBSCAN eps automatically using median pairwise distance.
    This approximates the old face_recognition behavior.
    """
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    dists = np.linalg.norm(X_sample[:, None, :] - X_sample[None, :, :], axis=2)
    median_dist = np.median(dists[dists > 0])
    return round(median_dist * 1.15, 3)  # Slightly relaxed


# 3. Merge Small / Similar Clusters

def merge_small_clusters(X, labels, threshold=0.7):
    """
    Merge very small clusters into nearest larger cluster
    if their centroid similarity > threshold.
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) < 2:
        return labels

    new_labels = labels.copy()

    # Compute centroids
    centroids = {
        lbl: np.mean(X[labels == lbl], axis=0)
        for lbl in unique_labels
        if np.sum(labels == lbl) > 0
    }

    merged = {}
    for i, ci in centroids.items():
        for j, cj in centroids.items():
            if i >= j:
                continue
            sim = np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-9)
            if sim >= threshold:
                merged[j] = i

    for old, new in merged.items():
        new_labels[new_labels == old] = new

    return new_labels


# 4. Display Clusters with Duplicate Filtering

def display_clusters(embeddings, image_paths, labels, method_name="clustering", max_clusters=5, max_images=3, dedup_thresh=0.95):
    """
    Display clustered face images with visual deduplication.
    Duplicate detection is based on cosine similarity of embeddings.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        print("No clusters found.")
        return

    noise_label = -1 if -1 in unique_labels else None
    non_noise_labels = [l for l in unique_labels if l != noise_label]

    print(f"\n--- Displaying Clusters for {method_name.upper()} ---")

    for label in non_noise_labels[:max_clusters]:
        print(f"\n--- Cluster {label} ---")
        cluster_indices = np.where(labels == label)[0]
        cluster_paths = [image_paths[i] for i in cluster_indices]

        cluster_embs = embeddings[cluster_indices]

        # ✅ Deduplicate based on cosine similarity
        keep_indices = []
        for i in range(len(cluster_embs)):
            if not any(
                cosine_similarity(cluster_embs[i].reshape(1, -1), cluster_embs[j].reshape(1, -1))[0][0] > dedup_thresh
                for j in keep_indices
            ):
                keep_indices.append(i)

        deduped_paths = [cluster_paths[i] for i in keep_indices[:max_images]]

        if not deduped_paths:
            continue

        fig, axes = plt.subplots(1, len(deduped_paths), figsize=(len(deduped_paths) * 2.5, 2.5))
        if len(deduped_paths) == 1:
            axes = [axes]

        for ax, img_path in zip(axes, deduped_paths):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(os.path.basename(img_path), fontsize=8)
                ax.axis("off")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    if noise_label is not None:
        print(f"\n--- Noise (Label {noise_label}) ---")
        noise_indices = np.where(labels == noise_label)[0]
        for img_path in np.array(image_paths)[noise_indices[:max_images]]:
            print(f"Noise image: {os.path.basename(img_path)}")
