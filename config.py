"""
config.py — Central configuration for Smart Vision App.
Edit this file to tune detection behaviour, paths, and TTS cooldowns.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_float32.tflite")
IMG_SIZE = 640  # model input resolution (pixels)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.30   # minimum class confidence to keep a detection
NMS_THRESHOLD  = 0.45   # IoU threshold for Non-Max Suppression

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
TARGET_FPS = 30

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
# Seconds that must pass before the same class is announced again
TTS_COOLDOWN = 3.0

# High-priority classes are re-announced on a shorter cooldown
PRIORITY_COOLDOWN = 1.5
PRIORITY_CLASSES = {
    "Person Front",
    "Crossroad",
    "Pedestrian Light Red",
    "Traffic Light Red",
}
