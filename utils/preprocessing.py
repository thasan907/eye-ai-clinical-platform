"""
utils/preprocessing.py
Handles all image loading, cleaning, and augmentation.
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from pathlib import Path
from config import IMAGE_SIZE


# ── Transforms ───────────────────────────────────────────────────

def get_train_transforms():
    """Aggressive augmentation for training — makes model robust."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Minimal transforms for validation / inference — no augmentation."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Image loading ────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """Load image from disk and convert to RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load image from uploaded bytes (FastAPI endpoint use)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_for_inference(image: np.ndarray) -> torch.Tensor:
    """
    Prepare a single image for model inference.
    Returns a tensor of shape (1, 3, H, W).
    """
    transforms = get_val_transforms()
    augmented = transforms(image=image)
    tensor = augmented["image"]          # shape: (3, H, W)
    return tensor.unsqueeze(0)           # shape: (1, 3, H, W)


# ── Quality check ────────────────────────────────────────────────

def check_image_quality(image: np.ndarray) -> dict:
    """
    Basic quality check on the retinal image.
    Returns a dict with a pass/fail flag and reason.
    """
    h, w = image.shape[:2]

    # Too small
    if h < 100 or w < 100:
        return {"ok": False, "reason": "Image too small (min 100×100 px)"}

    # Too dark
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 20:
        return {"ok": False, "reason": "Image too dark"}

    # Too blurry (Laplacian variance method)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50:
        return {"ok": False, "reason": f"Image too blurry (score: {blur_score:.1f})"}

    return {"ok": True, "reason": "Quality check passed",
            "brightness": round(float(mean_brightness), 1),
            "sharpness": round(float(blur_score), 1)}
