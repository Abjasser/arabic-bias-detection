"""
evaluate.py — Full evaluation pipeline for Arabic Bias Detection experiments.

Loads the best saved model checkpoint and runs:
  - Accuracy, Precision, Recall, F1 (macro + weighted) per class
  - Confusion matrix (saved as PNG)
  - Full classification report (saved as TXT)
  - Per-source breakdown (if cross_source_eval: true in config)
  - Predictions CSV with gold labels, predicted labels, and confidence

Usage:
    python src/evaluate.py --config configs/baseline_gpt_only.yaml
    python src/evaluate.py --config configs/main_gpt_claude_mixed.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on Colab & servers
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, set_seed, ensure_dir


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_batch(texts: list, model, tokenizer, max_length: int,
                  batch_size: int, device) -> tuple:
    """
    Run inference on a list of strings.

    Returns:
        preds   (np.ndarray of int)  — argmax predictions
        probs   (np.ndarray of float) — softmax probability of class 1
    """
    model.eval()
    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = np.argmax(probs, axis=-1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs[:, 1].tolist())   # prob of class 1 (Biased)

    return np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, label_names: list,
                          save_path: str, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Row-normalised"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(subtitle)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [evaluate] Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_full_metrics(y_true, y_pred, label_names: list) -> dict:
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred,    average="macro",    zero_division=0)),
        "f1_per_class": {
            name: float(f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0))
            for i, name in enumerate(label_names)
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    print(f"\n{'='*60}")
    print(f"[evaluate] Experiment: {cfg['experiment_name']}")
    print(f"{'='*60}")

    # ---- Paths ----------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent

    def abs_path(rel):
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    test_path    = abs_path(cfg["data"]["test_path"])
    output_dir   = abs_path(cfg["output_dir"])
    best_model   = output_dir / "best_model"
    ensure_dir(str(output_dir))

    if not best_model.exists():
        raise FileNotFoundError(
            f"No best_model found at {best_model}. Run train.py first."
        )

    # ---- Load model -----------------------------------------------------------
    print(f"\n[evaluate] Loading model from: {best_model}")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(best_model))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(best_model)
    ).to(device)

    id2label = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    label_names = [id2label[i] for i in sorted(id2label)]

    # ---- Load test data -------------------------------------------------------
    test_df = pd.read_csv(str(test_path))
    test_df = test_df.dropna(subset=[cfg["data"]["text_column"],
                                     cfg["data"]["label_column"]])
    texts  = test_df[cfg["data"]["text_column"]].astype(str).tolist()
    labels = test_df[cfg["data"]["label_column"]].astype(int).tolist()

    print(f"\n[evaluate] Test set: {len(texts):,} samples")

    # ---- Inference ------------------------------------------------------------
    max_len    = cfg["training"]["max_length"]
    batch_size = cfg["training"]["eval_batch_size"]

    print(f"[evaluate] Running inference (batch_size={batch_size})...")
    preds, probs = predict_batch(texts, model, tokenizer, max_len, batch_size, device)

    y_true = np.array(labels)
    y_pred = preds

    # ---- Metrics --------------------------------------------------------------
    metrics = compute_full_metrics(y_true, y_pred, label_names)
    print(f"\n[evaluate] Overall metrics:")
    for k, v in metrics.items():
        if k != "f1_per_class":
            print(f"  {k}: {v:.4f}")
    for cls, score in metrics["f1_per_class"].items():
        print(f"  F1 ({cls}): {score:.4f}")

    # ---- Classification report ------------------------------------------------
    report_str = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n{report_str}")

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {cfg['experiment_name']}\n")
        f.write(f"Model:      {cfg['model']['name']}\n")
        f.write(f"Test rows:  {len(y_true)}\n\n")
        f.write(report_str)
    print(f"  [evaluate] Report saved: {report_path}")

    # ---- Confusion matrix -----------------------------------------------------
    cm_path = str(output_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        y_true, y_pred,
        label_names=label_names,
        save_path=cm_path,
        title=f"Confusion Matrix — {cfg['experiment_name']}",
    )

    # ---- Save metrics.json ----------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    payload = {
        "experiment": cfg["experiment_name"],
        "model":      cfg["model"]["name"],
        "test_rows":  int(len(y_true)),
        **{k: v for k, v in metrics.items() if k != "f1_per_class"},
        "f1_non_biased": metrics["f1_per_class"].get("Non-biased", 0.0),
        "f1_biased":     metrics["f1_per_class"].get("Biased", 0.0),
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [evaluate] Metrics saved: {metrics_path}")

    # ---- Predictions CSV ------------------------------------------------------
    preds_df = test_df[[
        cfg["data"]["text_column"],
        cfg["data"]["label_column"],
    ]].copy()
    preds_df.columns = ["arabic_text", "true_label"]
    preds_df["true_label_name"] = preds_df["true_label"].map(id2label)
    preds_df["predicted_label"] = y_pred
    preds_df["predicted_label_name"] = pd.Series(y_pred).map(id2label).values
    preds_df["confidence_biased"] = np.round(probs, 4)
    preds_df["correct"] = (y_true == y_pred).astype(int)

    if "source" in test_df.columns:
        preds_df["source"] = test_df["source"].values
    if "sentence_id" in test_df.columns:
        preds_df.insert(0, "sentence_id", test_df["sentence_id"].values)

    preds_path = output_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"  [evaluate] Predictions saved: {preds_path}")

    # ---- Cross-source evaluation (main experiment only) -----------------------
    if cfg["evaluation"].get("cross_source_eval", False) and "source" in test_df.columns:
        print(f"\n[evaluate] Cross-source breakdown:")
        sources = test_df["source"].unique()
        cross_metrics = {}

        for src in sorted(sources):
            mask = (test_df["source"] == src).values
            n = mask.sum()
            if n == 0:
                continue
            m = compute_full_metrics(y_true[mask], y_pred[mask], label_names)
            cross_metrics[src] = {"n": int(n), **m}
            print(f"  [{src}] n={n} | acc={m['accuracy']:.4f} | "
                  f"f1_macro={m['f1_macro']:.4f}")

        cross_path = output_dir / "cross_source_metrics.json"
        with open(cross_path, "w") as f:
            json.dump(cross_metrics, f, indent=2)
        print(f"  [evaluate] Cross-source metrics saved: {cross_path}")

    print(f"\n[evaluate] Done — {cfg['experiment_name']}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bias detection model.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
