"""
predict.py — CLI inference script for Arabic Bias Detection.

Loads a saved best_model checkpoint and classifies new Arabic sentences.

Usage (stdin / file):
    echo "هذا نص عربي" | python src/predict.py --config configs/baseline_gpt_only.yaml
    python src/predict.py --config configs/baseline_gpt_only.yaml --text "هذا نص عربي"
    python src/predict.py --config configs/baseline_gpt_only.yaml --input new_texts.txt
    python src/predict.py --config configs/main_gpt_claude_mixed.yaml \
                          --input new_texts.csv --text_col arabic_text \
                          --output predictions_new.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, set_seed, ensure_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(best_model_dir: str, device):
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir
    ).to(device)
    model.eval()
    return model, tokenizer


def predict(texts: list, model, tokenizer, max_length: int,
            batch_size: int, device) -> tuple:
    """
    Returns:
        labels (list of int)   — predicted class indices
        names  (list of str)   — predicted class names
        probs  (list of float) — probability of class 1 (Biased)
    """
    id2label = model.config.id2label
    all_labels, all_names, all_probs = [], [], []

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

        for pred, prob_row in zip(preds, probs):
            all_labels.append(int(pred))
            all_names.append(id2label[int(pred)])
            all_probs.append(float(prob_row[1]))   # prob of class 1

    return all_labels, all_names, all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify Arabic sentences for media bias.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",   required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--text",     default=None,
                        help="Single Arabic sentence to classify")
    parser.add_argument("--input",    default=None,
                        help="Path to a .txt (one sentence/line) or .csv file")
    parser.add_argument("--text_col", default="arabic_text",
                        help="Column name if --input is a CSV (default: arabic_text)")
    parser.add_argument("--output",   default=None,
                        help="Save results to this CSV path (optional)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # ---- Config & model path --------------------------------------------------
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    repo_root = Path(__file__).resolve().parent.parent
    output_dir  = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    best_model_dir = str(output_dir / "best_model")

    if not Path(best_model_dir).exists():
        print(f"[predict] ERROR: No model found at {best_model_dir}")
        print("[predict]  → Run train.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Loading model from: {best_model_dir} (device: {device})")
    model, tokenizer = load_model(best_model_dir, device)
    max_length = cfg["training"]["max_length"]

    # ---- Gather input texts ---------------------------------------------------
    texts = []
    source_df = None

    if args.text:
        texts = [args.text.strip()]

    elif args.input:
        inp = Path(args.input)
        if not inp.exists():
            print(f"[predict] ERROR: Input file not found: {inp}")
            sys.exit(1)

        if inp.suffix.lower() == ".csv":
            source_df = pd.read_csv(inp)
            if args.text_col not in source_df.columns:
                print(f"[predict] ERROR: Column '{args.text_col}' not in {inp}")
                print(f"  Available: {source_df.columns.tolist()}")
                sys.exit(1)
            texts = source_df[args.text_col].astype(str).tolist()
        else:
            with open(inp, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

    elif not sys.stdin.isatty():
        # Read from stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]

    else:
        parser.print_help()
        sys.exit(0)

    if not texts:
        print("[predict] No input texts provided.")
        sys.exit(1)

    print(f"[predict] Classifying {len(texts):,} sentence(s)...")

    # ---- Inference ------------------------------------------------------------
    labels, names, probs = predict(
        texts, model, tokenizer, max_length, args.batch_size, device
    )

    # ---- Output ---------------------------------------------------------------
    results = pd.DataFrame({
        "arabic_text":           texts,
        "predicted_label":       labels,
        "predicted_label_name":  names,
        "confidence_biased":     [round(p, 4) for p in probs],
    })

    if source_df is not None:
        # Merge extra columns from source CSV
        for col in source_df.columns:
            if col != args.text_col and col not in results.columns:
                results.insert(0, col, source_df[col].values)

    if args.output:
        out_path = Path(args.output)
        ensure_dir(str(out_path.parent))
        results.to_csv(out_path, index=False)
        print(f"[predict] Results saved to: {out_path}")
    else:
        # Print to stdout
        print(f"\n{'─'*60}")
        for _, row in results.iterrows():
            conf = row["confidence_biased"]
            indicator = "🔴 BIASED" if row["predicted_label"] == 1 else "🟢 NON-BIASED"
            print(f"{indicator}  (conf={conf:.1%})")
            print(f"  {row['arabic_text'][:120]}")
            print()


if __name__ == "__main__":
    main()
