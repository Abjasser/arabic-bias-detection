"""
train.py — HuggingFace Trainer-based fine-tuning pipeline.

Fine-tunes UBC-NLP/ARBERTv2 (or any BERT-style model) on the prepared
Arabic bias detection dataset. Handles class-weighted loss, group-aware
train/test split, and saves the best checkpoint by F1-macro.

Usage:
    python src/train.py --config configs/baseline_gpt_only.yaml
    python src/train.py --config configs/main_gpt_claude_mixed.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_config,
    set_seed,
    compute_class_weights,
    make_compute_metrics,
    ensure_dir,
)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class BiasDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted cross-entropy
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the loss function."""

    def __init__(self, class_weights: list = None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float).to(device)
            )
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def load_split(path: str, cfg: dict) -> tuple:
    """Load a processed CSV and return (texts, labels) lists."""
    df = pd.read_csv(path)
    text_col  = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    # Drop rows with missing text
    df = df.dropna(subset=[text_col, label_col])
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels, df


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    print(f"\n{'='*60}")
    print(f"[train] Experiment: {cfg['experiment_name']}")
    print(f"{'='*60}")

    # ---- Paths ----------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent

    def abs_path(rel):
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    train_path  = abs_path(cfg["data"]["train_path"])
    test_path   = abs_path(cfg["data"]["test_path"])
    output_dir  = abs_path(cfg["output_dir"])
    ensure_dir(str(output_dir))

    # ---- Load data ------------------------------------------------------------
    print(f"\n[train] Loading data...")
    train_texts, train_labels, train_df = load_split(str(train_path), cfg)
    test_texts,  test_labels,  test_df  = load_split(str(test_path),  cfg)

    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,}  samples")

    # ---- Tokenizer & Model ----------------------------------------------------
    model_name = cfg["model"]["name"]
    num_labels = cfg["model"]["num_labels"]
    id2label   = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    label2id   = {v: k for k, v in id2label.items()}

    print(f"\n[train] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ---- Datasets -------------------------------------------------------------
    max_len = cfg["training"]["max_length"]
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer, max_len)
    eval_dataset  = BiasDataset(test_texts,  test_labels,  tokenizer, max_len)

    # ---- Class weights --------------------------------------------------------
    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        class_weights = compute_class_weights(train_labels)
        print(f"\n[train] Class weights: Non-biased={class_weights[0]:.4f}, "
              f"Biased={class_weights[1]:.4f}")

    # ---- Training arguments ---------------------------------------------------
    t = cfg["training"]
    ev = cfg["evaluation"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        fp16=t.get("fp16", False) and torch.cuda.is_available(),
        seed=t["seed"],
        logging_steps=t.get("logging_steps", 20),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=ev.get("metric_for_best_model", "f1_macro"),
        greater_is_better=True,
        report_to="none",                   # disable W&B / TensorBoard
        save_total_limit=2,                 # keep only 2 checkpoints on disk
    )

    # ---- Trainer --------------------------------------------------------------
    compute_metrics = make_compute_metrics(list(id2label.values()))

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----------------------------------------------------------------
    print(f"\n[train] Starting training ({t['epochs']} epochs)...\n")
    train_result = trainer.train()

    # ---- Save best model & tokenizer ------------------------------------------
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"\n[train] Best model saved to: {best_model_dir}")

    # ---- Final evaluation on test set -----------------------------------------
    print(f"\n[train] Final evaluation on test set...")
    eval_results = trainer.evaluate(eval_dataset)
    print(f"  Results: {json.dumps(eval_results, indent=2)}")

    # Save eval results
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        payload = {
            "experiment":   cfg["experiment_name"],
            "model":        model_name,
            "train_samples": len(train_texts),
            "test_samples":  len(test_texts),
            "training_loss": train_result.training_loss,
            **{k.replace("eval_", ""): v for k, v in eval_results.items()},
        }
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    print(f"\n[train] Done — {cfg['experiment_name']}\n")
    return trainer, tokenizer, test_df


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ARBERT for bias detection.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
