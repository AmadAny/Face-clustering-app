import streamlit as st
import torch
import os
import shutil
import numpy as np
from PIL import Image
import random

# Import project functions
from src.face_detection import crop_faces_from_folder
from src.embedding import load_model, generate_embeddings_from_folder
from src.clustering import cluster_embeddings
from src.evaluation import evaluate_clustering

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Face Clustering App", layout="wide")
st.title("🧠 Face Clustering and Evaluation Dashboard")

PROJECT_PATH = os.getcwd()
TEMP_UPLOAD = os.path.join(PROJECT_PATH, "temp_upload")
TEMP_FACES = os.path.join(PROJECT_PATH, "temp_faces")

# Ensure directories exist
os.makedirs(TEMP_UPLOAD, exist_ok=True)
os.makedirs(TEMP_FACES, exist_ok=True)

# Initialize session state for uploads
if 'uploaded_file_contents' not in st.session_state:
    st.session_state.uploaded_file_contents = {}
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("Settings")

# --- Model selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_choice = st.sidebar.selectbox(
    "Select model to use:",
    [
        "Pre-trained Facenet (VGGFace2)",
        "Trained model: facenet_lfw_best_state.pth"
    ],
    index=0
)

# Clustering method control
selected_methods = st.sidebar.multiselect(
    "Select clustering methods:",
    ["dbscan", "kmeans", "agglomerative"],
    default=["dbscan", "kmeans", "agglomerative"]
)

# Determine if only Agglomerative is selected and dynamic
only_agglo_selected = selected_methods == ["agglomerative"]

agglo_dynamic = st.sidebar.checkbox(
    "Agglomerative: Adaptive clustering (similarity-based merging)",
    value=False
) if "agglomerative" in selected_methods else False

is_only_agglo_dynamic = only_agglo_selected and agglo_dynamic

# Conditional sliders
if is_only_agglo_dynamic:
    st.sidebar.info("Agglomerative (dynamic) mode: Max clusters/images sliders are disabled.")
    max_clusters, max_images = 999999, 999999
else:
    max_clusters = st.sidebar.slider("Max clusters to display", 1, 100, 5)
    max_images = st.sidebar.slider("Max images per cluster", 1, 100, 5)

# DBSCAN parameters
dbscan_params = {}
if "dbscan" in selected_methods:
    st.sidebar.subheader("DBSCAN Parameters")
    dbscan_params["eps"] = st.sidebar.slider("EPS (Neighborhood radius)", 0.1, 2.0, 0.5, 0.1)
    dbscan_params["min_samples"] = st.sidebar.slider("Min Samples per cluster", 1, 20, 2, 1)

# Agglomerative options
agglo_threshold, agglo_n_clusters = None, None
if "agglomerative" in selected_methods:
    st.sidebar.subheader("Agglomerative Settings (Post-clustering merging)")
    if agglo_dynamic:
        agglo_threshold = st.sidebar.slider(
            "Cluster merging threshold (cosine similarity)",
            0.1, 5.0, 1.0, 0.1
        )
    else:
        agglo_n_clusters = st.sidebar.slider("Number of clusters (manual mode)", 2, 20, 5, 1)

# -----------------------------
# Step 1: Upload Images
# -----------------------------
st.subheader("📸 Upload Images")

newly_uploaded_files = st.file_uploader(
    "Upload face images (multiple allowed)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if newly_uploaded_files:
    st.session_state.uploaded_file_names = []
    st.session_state.uploaded_file_contents = {}
    for file in newly_uploaded_files:
        st.session_state.uploaded_file_names.append(file.name)
        st.session_state.uploaded_file_contents[file.name] = file.read()
    st.success(f"✅ Uploaded {len(newly_uploaded_files)} files.")

if st.session_state.uploaded_file_names:
    st.write(f"**Displaying {len(st.session_state.uploaded_file_names)} uploaded files:**")
    cols = st.columns(5)
    for i, file_name in enumerate(st.session_state.uploaded_file_names):
        try:
            import io
            image_bytes = st.session_state.uploaded_file_contents[file_name]
            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream)
            cols[i % 5].image(image, caption=file_name, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying {file_name}: {e}")
else:
    st.info("No files uploaded yet. Please use the uploader above.")

# -----------------------------
# Step 2: Run Clustering
# -----------------------------
if st.button("🚀 Run Face Clustering"):
    if not st.session_state.uploaded_file_names:
        st.error("Please upload some images first.")
    else:
        st.subheader("🔍 Detecting Faces and Generating Embeddings...")

        # Clear temp_upload folder
        for filename in os.listdir(TEMP_UPLOAD):
            file_path = os.path.join(TEMP_UPLOAD, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save uploaded files to temp_upload
        for file_name, file_content in st.session_state.uploaded_file_contents.items():
            file_path = os.path.join(TEMP_UPLOAD, file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)

        cropped_faces = crop_faces_from_folder(TEMP_UPLOAD, TEMP_FACES)
        st.write(f"✅ Detected and cropped {len(cropped_faces)} faces.")

        # --- Load selected model ---
        if model_choice == "Pre-trained Facenet (VGGFace2)":
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            st.info("✅ Using pre-trained Facenet (VGGFace2).")
        else:
            model_path = "/content/drive/MyDrive/Project alt version/models/facenet_lfw_best_state.pth"
            model, device = load_model(model_path=model_path, device=device)
            st.info("✅ Using trained model: facenet_lfw_best_state.pth.")

        embeddings, embedding_paths = generate_embeddings_from_folder(model, TEMP_FACES, device=device)

        if len(embeddings) == 0:
            st.error("No faces detected. Try clearer photos.")
        else:
            st.success(f"✅ Generated {embeddings.shape[0]} embeddings.")

            all_labels = {}
            from collections import defaultdict

            st.write(f"📘 **Results using model:** `{model_choice}`")

            for method in selected_methods:
                st.markdown(f"### 🔸 {method.upper()} Clustering")

                if method == "dbscan":
                    labels = cluster_embeddings(
                        embeddings,
                        method=method,
                        eps=dbscan_params["eps"],
                        min_samples=dbscan_params["min_samples"],
                        adaptive_eps=False
                    )

                elif method == "agglomerative":
                    if agglo_dynamic:
                        st.info(f"🔄 Adaptive clustering (merge threshold = {agglo_threshold})...")
                        labels = cluster_embeddings(
                            embeddings,
                            method=method,
                            n_clusters=min(50, len(embeddings)),
                            merge_threshold=agglo_threshold
                        )
                    else:
                        st.info(f"📊 Fixed number of clusters = {agglo_n_clusters}")
                        labels = cluster_embeddings(
                            embeddings,
                            method=method,
                            n_clusters=agglo_n_clusters
                        )

                else:
                    labels = cluster_embeddings(embeddings, method=method)

                all_labels[method] = labels
                st.write("🖼️ Clusters:")

                cluster_dict = defaultdict(list)
                for img_path, label in zip(embedding_paths, labels):
                    cluster_dict[label].append(img_path)

                cluster_items = sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)

                for cluster_id, imgs in cluster_items[:int(max_clusters)]:
                    st.markdown(f"#### 👥 Cluster {cluster_id} ({len(imgs)} images)")
                    num_to_select = min(len(imgs), int(max_images))
                    display_imgs = random.sample(imgs, num_to_select) if num_to_select > 0 else []
                    cols = st.columns(len(display_imgs))
                    for i, img_path in enumerate(display_imgs):
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            cols[i].image(image, use_container_width=True)

                st.write("📊 **Evaluation Metrics:**")
                metrics = evaluate_clustering(embeddings, labels)
                for key, value in metrics.items():
                    st.write(f"- **{key}**: {value:.4f}" if isinstance(value, (int, float)) else f"- **{key}**: {value}")

            # Visualization
            st.subheader("🌀 t-SNE Visualization of Embeddings")
            from src.visualization import tsne_embeddings, plot_scatter, imscatter_with_thumbs

            emb2d = tsne_embeddings(embeddings, n_components=2, perplexity=30, random_state=42)
            for method in selected_methods:
                labels = all_labels[method]
                title = f"t-SNE Visualization — {method.upper()} Clustering"
                plot_scatter(emb2d, labels=labels, title=title)
                st.write(f"📸 Showing image thumbnails for {method.upper()}")
                imscatter_with_thumbs(emb2d, embedding_paths, N=70, zoom=0.6)

            # Cleanup
            try:
                for folder in [TEMP_UPLOAD, TEMP_FACES]:
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                st.success("🧹 Temporary image files cleaned up! Folders remain for next run.")
            except Exception as e:
                st.warning(f"⚠️ Could not fully clean temporary folders: {e}")

# Clear uploads
if st.button("🔄 Clear Uploaded Files"):
    st.session_state.uploaded_file_names = []
    st.session_state.uploaded_file_contents = {}
    st.success("Uploaded files cleared from session.")
    st.rerun()
