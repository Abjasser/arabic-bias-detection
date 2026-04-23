"""
utils.py — Shared utilities for Arabic Bias Detection experiments.

Provides config loading, reproducibility helpers, class weight computation,
and the HuggingFace Trainer metric callback.
"""

import random
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML experiment config and return as a plain dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels) -> list:
    """
    Compute balanced class weights from a list/array of integer labels.

    Returns a list [w0, w1] where w_i = n_samples / (n_classes * count_i).
    Compatible with torch.FloatTensor and HuggingFace Trainer.
    """
    labels = np.array(labels)
    classes = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(classes)

    weights = []
    for c in sorted(classes):
        count = np.sum(labels == c)
        w = n_samples / (n_classes * count)
        weights.append(float(w))

    return weights  # e.g. [0.72, 1.48] for typical 2:1 imbalance


# ---------------------------------------------------------------------------
# HuggingFace Trainer metric callback
# ---------------------------------------------------------------------------

def make_compute_metrics(label_names: list = None):
    """
    Factory that returns a compute_metrics function for HuggingFace Trainer.

    The returned function is called with an EvalPrediction object and returns
    a dict with accuracy, f1_macro, f1_weighted, precision_macro, recall_macro.

    Args:
        label_names: Optional list like ['Non-biased', 'Biased'].
                     Not used in computation but kept for reference.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        return {
            "accuracy":        float(accuracy_score(labels, preds)),
            "f1_macro":        float(f1_score(labels, preds, average="macro",    zero_division=0)),
            "f1_weighted":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(labels, preds, average="macro",    zero_division=0)),
            "recall_macro":    float(recall_score(labels, preds,    average="macro",    zero_division=0)),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    """Return the repository root (parent of the src/ directory)."""
    return Path(__file__).resolve().parent.parent
