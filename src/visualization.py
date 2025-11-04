import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import cv2


def tsne_embeddings(embeddings, n_components=2, random_state=42, perplexity=30, **kwargs):
    n_samples = embeddings.shape[0]
    # t-SNE requires perplexity < n_samples
    if n_samples - 1 <= perplexity:
        perplexity = max(1, (n_samples - 1) // 3)
        print(f"[INFO] Adjusted perplexity to {perplexity} (too few samples).")

    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, **kwargs)
    return tsne.fit_transform(embeddings)



def plot_scatter(emb2d, labels=None, title=None, cmap='tab20', s=10):
    """
    Plot a 2D scatter of t-SNE embeddings.
    Works inside Streamlit using st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap=cmap, s=s)
    if title:
        ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    st.pyplot(fig)  # ✅ display inside Streamlit


def imscatter_with_thumbs(emb2d, image_paths, N=50, zoom=0.5, random_seed=42):
    """
    Overlay small image thumbnails on the t-SNE scatter.
    """
    np.random.seed(random_seed)
    n = len(image_paths)
    if n == 0:
        return

    # Randomly pick N images
    indices = np.random.choice(n, min(N, n), replace=False)
    subset_coords = emb2d[indices]
    subset_paths = [image_paths[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (x, y) in enumerate(subset_coords):
        img = cv2.imread(subset_paths[i])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2RGB)
        im = OffsetImage(img_rgb, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.update_datalim(subset_coords)
    ax.autoscale()
    ax.axis('off')
    st.pyplot(fig)  # ✅ display inside Streamlit
