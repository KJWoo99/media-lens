"""Image preprocessing and resolution utilities."""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Register HEIC/HEIF support via pillow-heif if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff',
    '.heic', '.heif', '.avif',
}

RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def collect_images(folder_path, recursive=False):
    """Collect image file paths from a folder."""
    folder_path = str(folder_path)
    images = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                if is_image_file(full):
                    images.append(full)
    else:
        try:
            for f in os.listdir(folder_path):
                full = os.path.join(folder_path, f)
                if os.path.isfile(full) and is_image_file(full):
                    images.append(full)
        except Exception:
            pass
    return sorted(images)


_PIL_ONLY_EXTS = {'.heic', '.heif', '.avif'}


def load_image_cv2(path):
    """Load image via OpenCV, handling unicode paths.
    Falls back to PIL for formats OpenCV cannot decode (HEIC, HEIF, AVIF)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _PIL_ONLY_EXTS:
        try:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            return np.array(img)
        except Exception:
            return None
    try:
        with open(path, 'rb') as f:
            data = np.fromfile(f, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def get_resolution(path):
    """Return (width, height) or None. Uses PIL header-only read (fast, no full decode)."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def preprocess_for_resnet(path):
    """Load and preprocess image for DINOv2/ResNet (224x224 tensor, ImageNet normalization)."""
    img = load_image_cv2(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return RESNET_TRANSFORM(img)


def collect_resolutions(paths):
    """Collect resolution info for a list of image paths."""
    result = {}
    for p in paths:
        result[p] = get_resolution(p)
    return result
