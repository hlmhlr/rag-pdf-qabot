"""Folders the app and scripts use, found relative to the repository."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/qabot/paths.py -> repository root
SAMPLES_DIR = ROOT / "data" / "samples"
