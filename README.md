# Face-clustering-app
Unsupervised face clustering by individuals
A deep learning–powered **Face Clustering Web App** built with **Streamlit**, **PyTorch**, and **Facenet**.  
The app detects faces, generates embeddings, and clusters them by **individual identity**, providing both **visual** and **quantitative** insights.

---

## Features
- **Face Detection & Cropping** using MTCNN  
- **Facial Embeddings** generated via FaceNet 
- **Multiple Clustering Algorithms:**
  - DBSCAN 
  - K-Means
  - Agglomerative (supports dynamic threshold mode)
- Silhouette score evaluation
- **Interactive Web Interface** (built with Streamlit)
- **Dynamic Visualization** using **t-SNE**

---

## Project Structure
project-folder/
│  
├── app.py # Main Streamlit application  
├── run_in_colab.ipynb  # Notebook to launch Streamlit + Ngrok  
├── requirements.txt # Python dependencies  
│  
├── src/ # Core project modules  
│ ├── face_detection.py # Face cropping using MTCNN  
│ ├── embedding.py # Embedding generation utilities  
│ ├── clustering.py # Clustering algorithms  
│ ├── evaluation.py # Clustering metrics & evaluation  
│ └── visualization.py # t-SNE and image scatter visualization  
└── README.md # Project documentation  

---

## Installation
You can run everything directly in colab, no local setup required.
The notebook run_in_colab.ipynb will automatically install dependencies and start the app.

However, if you want to run it locally, you can install the same dependencies manually:
pip install --upgrade setuptools wheel  
pip install facenet-pytorch==2.5.3  
pip install opencv-python-headless  
pip install mtcnn==0.1.1  
pip install pandas>=2.1.2  
pip install scikit-learn matplotlib numpy Pillow  
pip install streamlit pyngrok  

# Model
The app supports two models:
  1. Pre-trained FaceNet (VGGFace2); automatically downloaded via facenet-pytorch.
  2.Fine-tuned FaceNet (ArcFace Head); trained on LFW and saved as: /content/drive/MyDrive/Project alt version/models/facenet_lfw_best_state.pth
This model is not included in the repository due to file size, to use it, mount your Google Drive in Colab before running the app.

---

## Running the app
  1. Open run_in_colab.ipynb in Google Colab.
  2. Run all cells, this will:
     . Install the required libraries
     . Launch Streamlit
     . Create a public Ngrok link to access your app
  3. Click the printed link to open the app in your browser.
