"""
models/train.py
Fine-tune EfficientNet on your retinal image dataset.

Usage:
    python models/train.py --data_dir data/eyepacs --epochs 20

Dataset folder structure expected:
    data/eyepacs/
        train/
            0/   (No DR images)
            1/   (Mild DR)
            2/   (Moderate DR)
            3/   (Severe DR)
            4/   (Proliferative DR)
        val/
            0/ 1/ 2/ 3/ 4/
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from models.eye_model import EyeAIModel, DISEASE_LABELS
from utils.preprocessing import get_train_transforms, get_val_transforms
from config import MODEL_PATH, IMAGE_SIZE


def get_dataloaders(data_dir: str, batch_size: int = 32):
    """Build train and validation DataLoaders."""

    # Wrap albumentations transforms for torchvision ImageFolder
    class AlbumentationsWrapper:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, pil_img):
            import numpy as np
            img = np.array(pil_img.convert("RGB"))
            return self.transform(image=img)["image"]

    train_ds = ImageFolder(
        root=f"{data_dir}/train",
        transform=AlbumentationsWrapper(get_train_transforms())
    )
    val_ds = ImageFolder(
        root=f"{data_dir}/val",
        transform=AlbumentationsWrapper(get_val_transforms())
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


def train(data_dir: str, epochs: int = 20, batch_size: int = 32, lr: float = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    model     = EyeAIModel(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Training phase ──────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validation phase ────────────────────────────────────
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # AUC (one-vs-rest, macro average)
        y_bin = label_binarize(all_labels, classes=list(range(5)))
        auc   = roc_auc_score(y_bin, np.array(all_probs), multi_class="ovr", average="macro")
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"AUC: {auc:.4f}"
        )

        scheduler.step()

        # Save best model
        if auc > best_auc:
            best_auc = auc
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            logger.success(f"New best model saved — AUC: {best_auc:.4f}")

    logger.success(f"Training complete. Best AUC: {best_auc:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=DISEASE_LABELS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/eyepacs")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr)
