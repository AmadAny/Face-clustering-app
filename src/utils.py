import os
import numpy as np

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def save_embeddings(embeddings, paths, out_dir="results/embeddings", prefix="embeddings"):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, f"{prefix}.npy"), embeddings)
    np.save(os.path.join(out_dir, f"{prefix}_paths.npy"), np.array(paths))
    return os.path.join(out_dir, f"{prefix}.npy")