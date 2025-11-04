import os
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from .data_augmentation import get_val_transform


def load_model(model_path=None, device=None, pretrained='vggface2', search_dir="checkpoints", use_pretrained_only=False):
    """
    Loads a FaceNet model checkpoint or the original pretrained InceptionResnetV1 model.

    Args:
        model_path (str): Path to the model checkpoint (optional).
        device (torch.device): Device for model loading.
        pretrained (str): One of ['vggface2', 'casia-webface'].
        search_dir (str): Directory to search for checkpoints if model_path is None.
        use_pretrained_only (bool): If True → always load the original pretrained model.

    Returns:
        (model, device)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 If user explicitly wants the original FaceNet
    if use_pretrained_only:
        print("[INFO] Using original pretrained InceptionResnetV1 model.")
        model = InceptionResnetV1(pretrained=pretrained, classify=False).to(device).eval()
        return model, device

    # 🔹 Otherwise, try to load checkpoint
    if model_path is None:
        if not os.path.isdir(search_dir):
            print(f"[WARN] No checkpoint dir found at {search_dir}. Using pretrained model.")
            return InceptionResnetV1(pretrained=pretrained, classify=False).to(device).eval(), device

        state_candidates = [f for f in os.listdir(search_dir) if f.endswith("_state.pth")]
        full_candidates = [f for f in os.listdir(search_dir) if f.endswith("_full.pth")]

        if state_candidates:
            model_path = os.path.join(search_dir, sorted(state_candidates)[-1])
            print(f"[INFO] Auto-selected state_dict checkpoint: {model_path}")
        elif full_candidates:
            model_path = os.path.join(search_dir, sorted(full_candidates)[-1])
            print(f"[INFO] Auto-selected full model checkpoint: {model_path}")
        else:
            print(f"[WARN] No checkpoints found in {search_dir}. Using pretrained model.")
            return InceptionResnetV1(pretrained=pretrained, classify=False).to(device).eval(), device

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model = InceptionResnetV1(pretrained=pretrained, classify=False)
        model.load_state_dict(state)
        print(f"[INFO] Loaded model weights from: {model_path}")
    except Exception as e:
        print(f"[WARN] Could not load checkpoint ({e}). Using pretrained weights instead.")
        model = InceptionResnetV1(pretrained=pretrained, classify=False)

    return model.to(device).eval(), device


def _prepare_image(img_bgr, transform=None, device=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if transform is None:
        transform = get_val_transform()
    img_t = transform(image=img_rgb)["image"].unsqueeze(0)
    if device is not None:
        img_t = img_t.to(device).float()
    return img_t


def generate_embeddings_from_folder(model, image_folder, device=None, transform=None, batch_size=64):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()

    paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    embeddings, image_paths = [], []
    batch, batch_paths = [], []

    with torch.no_grad():
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            tensor = _prepare_image(img, transform=transform, device=device)
            batch.append(tensor)
            batch_paths.append(p)

            if len(batch) == batch_size:
                emb = model(torch.cat(batch, dim=0)).cpu().numpy()
                embeddings.append(emb)
                image_paths.extend(batch_paths)
                batch, batch_paths = [], []

        if batch:
            emb = model(torch.cat(batch, dim=0)).cpu().numpy()
            embeddings.append(emb)
            image_paths.extend(batch_paths)

    if not embeddings:
        return np.zeros((0, 512)), []

    return np.vstack(embeddings), image_paths
