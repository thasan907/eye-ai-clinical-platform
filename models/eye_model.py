"""
models/eye_model.py
EfficientNet-B3 fine-tuned for diabetic retinopathy grading.
Uses transfer learning — starts from ImageNet weights.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from loguru import logger
from config import MODEL_PATH, DISEASE_LABELS


class EyeAIModel(nn.Module):
    """
    EfficientNet-B3 backbone with a custom classification head.
    Classifies retinal images into 5 DR severity grades.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B3 backbone (pretrained on ImageNet)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,          # Remove original classifier
            global_pool="avg",
        )
        in_features = self.backbone.num_features   # 1536 for B3

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ── Model loader ─────────────────────────────────────────────────

_model_instance: EyeAIModel | None = None


def load_model(device: str = "cpu") -> EyeAIModel:
    """
    Load the model once and cache it.
    Uses saved weights if available, otherwise uses pretrained backbone
    (good for development / testing before you train).
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model = EyeAIModel(num_classes=len(DISEASE_LABELS), pretrained=True)

    if Path(MODEL_PATH).exists():
        logger.info(f"Loading trained weights from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    else:
        logger.warning(
            "No trained weights found. Using ImageNet pretrained backbone only. "
            "Run training before deploying to production."
        )

    model.to(device)
    model.eval()
    _model_instance = model
    return model


# ── Inference ────────────────────────────────────────────────────

def predict(tensor: torch.Tensor, device: str = "cpu") -> dict:
    """
    Run inference on a preprocessed image tensor.

    Args:
        tensor: shape (1, 3, H, W) — output of preprocess_for_inference()
        device: "cpu" or "cuda"

    Returns:
        {
          "label":       "Moderate DR",
          "class_index": 2,
          "confidence":  0.87,
          "all_scores":  {"No DR": 0.03, "Mild DR": 0.06, ...}
        }
    """
    model = load_model(device)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                         # (1, 5)
        probs  = torch.softmax(logits, dim=1)[0]       # (5,)

    class_index = int(probs.argmax())
    confidence  = float(probs[class_index])
    label       = DISEASE_LABELS[class_index]

    all_scores = {
        DISEASE_LABELS[i]: round(float(probs[i]), 4)
        for i in range(len(DISEASE_LABELS))
    }

    return {
        "label":       label,
        "class_index": class_index,
        "confidence":  round(confidence, 4),
        "all_scores":  all_scores,
    }
