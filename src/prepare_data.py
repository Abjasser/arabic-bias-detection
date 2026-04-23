"""
prepare_data.py — Data preparation pipeline for Arabic Bias Detection.

Reads consensus_dataset.csv, extracts the correct Arabic translation per
translator source, maps labels, performs a group-aware (sentence_id) train/test
split to prevent data leakage, and saves processed CSVs + dataset_summary.json.

Usage:
    python src/prepare_data.py --config configs/baseline_gpt_only.yaml
    python src/prepare_data.py --config configs/main_gpt_claude_mixed.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, set_seed, ensure_dir


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "Non-biased": 0,
    "Biased":     1,
}

# Maps config source name → column in consensus_dataset.csv
SOURCE_TEXT_COL = {
    "gpt":    "gpt54mini_translation",
    "claude": "claude_translation",
    "gemini": "gemini_translation",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_raw(raw_path: str) -> pd.DataFrame:
    """Load the consensus CSV and validate required columns."""
    df = pd.read_csv(raw_path)
    required = {"sentence_id", "original_label", "translator",
                "gpt54mini_translation", "claude_translation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw CSV: {missing}")
    print(f"[prepare_data] Loaded {len(df):,} rows from {raw_path}")
    return df


def extract_source_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Return rows for a single source (e.g. 'gpt') with a unified arabic_text
    column populated from the appropriate translation column.
    """
    text_col = SOURCE_TEXT_COL[source]
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found for source '{source}'")

    # Filter to rows produced by this translator
    mask = df["translator"].str.lower() == source
    subset = df[mask].copy()

    subset["arabic_text"] = subset[text_col].astype(str).str.strip()
    subset["source"] = source

    # Drop rows where translation is missing or was a failure marker
    bad = subset["arabic_text"].str.lower().str.contains("translation failed", na=True)
    n_bad = bad.sum()
    if n_bad:
        print(f"  [prepare_data] Dropping {n_bad} failed translations for source '{source}'")
        subset = subset[~bad]

    return subset[["sentence_id", "arabic_text", "original_label", "source"]]


def build_combined(df: pd.DataFrame, sources: list) -> pd.DataFrame:
    """Combine rows for the requested sources into a single DataFrame."""
    parts = []
    for src in sources:
        part = extract_source_rows(df, src)
        print(f"  [prepare_data] {src}: {len(part):,} rows")
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    return combined


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map original_label strings to integer 0/1 and drop unmappable rows."""
    df = df.copy()
    df["label"] = df["original_label"].map(LABEL_MAP)
    n_missing = df["label"].isna().sum()
    if n_missing:
        unique_labels = df.loc[df["label"].isna(), "original_label"].unique().tolist()
        print(f"  [prepare_data] WARNING: {n_missing} rows have unmapped labels: {unique_labels}. Dropping.")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def group_aware_split(df: pd.DataFrame, group_col: str, test_size: float,
                      seed: int) -> tuple:
    """
    Split df into train/test ensuring no sentence_id appears in both sets.

    Uses sklearn's GroupShuffleSplit so that the same English sentence (even
    translated by different LLMs) stays in the same partition.
    """
    groups = df[group_col].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cfg: dict) -> dict:
    """Build a JSON-serialisable summary dict for record-keeping."""
    def label_dist(df):
        counts = df["label"].value_counts().to_dict()
        return {
            "Non-biased": int(counts.get(0, 0)),
            "Biased":     int(counts.get(1, 0)),
        }

    def source_dist(df):
        if "source" in df.columns:
            return df["source"].value_counts().to_dict()
        return {}

    return {
        "experiment": cfg.get("experiment_name"),
        "sources_included": cfg["data"].get("sources_to_include", []),
        "total_rows":  int(len(train_df) + len(test_df)),
        "train_rows":  int(len(train_df)),
        "test_rows":   int(len(test_df)),
        "train_labels": label_dist(train_df),
        "test_labels":  label_dist(test_df),
        "train_sources": source_dist(train_df),
        "test_sources":  source_dist(test_df),
        "unique_sentence_ids_train": int(train_df["sentence_id"].nunique()),
        "unique_sentence_ids_test":  int(test_df["sentence_id"].nunique()),
        "label_map": LABEL_MAP,
        "test_size": cfg["training"]["test_size"],
        "seed":      cfg["training"]["seed"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    print(f"\n{'='*60}")
    print(f"[prepare_data] Experiment: {cfg['experiment_name']}")
    print(f"{'='*60}")

    # Load raw data
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.is_absolute():
        raw_path = Path(__file__).resolve().parent.parent / raw_path
    raw_df = load_raw(str(raw_path))

    # Build combined Arabic text + label table
    sources = cfg["data"]["sources_to_include"]
    combined = build_combined(raw_df, sources)
    combined = map_labels(combined)

    # Stats before split
    total = len(combined)
    label_counts = combined["label"].value_counts().sort_index()
    print(f"\n[prepare_data] Total usable rows: {total:,}")
    for lbl, count in label_counts.items():
        name = {0: "Non-biased", 1: "Biased"}.get(lbl, str(lbl))
        print(f"  {name}: {count:,} ({100*count/total:.1f}%)")

    # Group-aware train/test split
    group_col  = cfg["data"]["group_column"]
    test_size  = cfg["training"]["test_size"]
    seed       = cfg["training"]["seed"]

    train_df, test_df = group_aware_split(combined, group_col, test_size, seed)
    print(f"\n[prepare_data] Train: {len(train_df):,} rows | "
          f"Test: {len(test_df):,} rows")
    print(f"  Train sentence_ids: {train_df['sentence_id'].nunique()}")
    print(f"  Test  sentence_ids: {test_df['sentence_id'].nunique()}")

    # Verify no leakage
    overlap = set(train_df["sentence_id"]) & set(test_df["sentence_id"])
    assert len(overlap) == 0, f"Leakage detected! {len(overlap)} shared sentence_ids."
    print(f"  [OK] Zero sentence_id overlap between train and test.")

    # Save outputs
    train_path   = Path(cfg["data"]["train_path"])
    test_path    = Path(cfg["data"]["test_path"])
    summary_path = Path(cfg["data"]["summary_path"])

    # Resolve relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    for p in [train_path, test_path, summary_path]:
        if not p.is_absolute():
            p = repo_root / p
        ensure_dir(str(p.parent))

    # Write absolute paths
    abs_train   = repo_root / train_path   if not train_path.is_absolute()   else train_path
    abs_test    = repo_root / test_path    if not test_path.is_absolute()    else test_path
    abs_summary = repo_root / summary_path if not summary_path.is_absolute() else summary_path

    # Select and order output columns
    cols_out = ["sentence_id", "arabic_text", "label", "source"]
    train_df[cols_out].to_csv(abs_train,   index=False)
    test_df[cols_out].to_csv(abs_test,    index=False)

    summary = dataset_summary(train_df, test_df, cfg)
    with open(abs_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[prepare_data] Saved:")
    print(f"  {abs_train}")
    print(f"  {abs_test}")
    print(f"  {abs_summary}")
    print(f"\n[prepare_data] Done — {cfg['experiment_name']}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for an experiment.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/baseline_gpt_only.yaml)")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
