"""Centralized model storage paths."""

import os

# Project root → models/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
