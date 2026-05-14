import os
import cv2
from PIL import Image
from mtcnn import MTCNN

# Create one global MTCNN detector instance
GLOBAL_DETECTOR = MTCNN()

def detect_and_align_faces(image_path, output_size=(160,160), padding=0.2, detector=GLOBAL_DETECTOR):
    """
    Detect faces in an image using MTCNN and return aligned, resized RGB numpy arrays.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        boxes = detector.detect_faces(img_rgb)
    except Exception:
        boxes = []

    faces = []
    for face in boxes:
        x, y, w, h = face['box']
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_rgb.shape[1], x + w + pad_w)
        y2 = min(img_rgb.shape[0], y + h + pad_h)

        crop = img_rgb[y1:y2, x1:x2]
        try:
            resized = cv2.resize(crop, output_size)
            faces.append(resized)
        except Exception:
            continue
    return faces


def crop_faces_from_folder(input_folder, output_folder, detector=GLOBAL_DETECTOR, output_size=(160,160)):
    """
    Detect and crop faces from all images in input_folder and save under output_folder.
    Returns list of saved face paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(input_folder, fname)
        faces = detect_and_align_faces(path, output_size=output_size, detector=detector)

        basename = os.path.splitext(fname)[0]
        for i, face in enumerate(faces):
            out_name = f"{basename}_face{i}.jpg"
            out_path = os.path.join(output_folder, out_name)
            img_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, img_bgr)
            saved_paths.append(out_path)

    return saved_paths
