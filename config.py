"""
config.py — Central configuration for the Eye AI system.
All settings are read from environment variables (or .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ── App ──────────────────────────────────────────────────────────
APP_ENV        = os.getenv("APP_ENV", "development")
APP_HOST       = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT       = int(os.getenv("APP_PORT", 8000))
SECRET_KEY     = os.getenv("SECRET_KEY", "change-me")

# ── AI / Model ───────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
MODEL_PATH             = BASE_DIR / os.getenv("MODEL_PATH", "models/eye_ai_model.pth")
CONFIDENCE_THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
IMAGE_SIZE             = int(os.getenv("IMAGE_SIZE", 224))
MAX_UPLOAD_BYTES       = int(os.getenv("MAX_UPLOAD_MB", 10)) * 1024 * 1024

# ── Database ─────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./eye_ai.db")

# ── Disease labels (matches training dataset order) ──────────────
DISEASE_LABELS = [
    "No DR",           # 0 — healthy
    "Mild DR",         # 1
    "Moderate DR",     # 2
    "Severe DR",       # 3
    "Proliferative DR" # 4
]

SEVERITY_MAP = {
    "No DR":           {"level": 0, "color": "green",  "action": "Routine annual check"},
    "Mild DR":         {"level": 1, "color": "yellow", "action": "Follow up in 12 months"},
    "Moderate DR":     {"level": 2, "color": "orange", "action": "Follow up in 3–6 months"},
    "Severe DR":       {"level": 3, "color": "red",    "action": "Urgent referral within 1 month"},
    "Proliferative DR":{"level": 4, "color": "red",    "action": "URGENT — refer immediately"},
}
