# Arabic Media Bias Detection — ARBERT Fine-tuning

Fine-tunes **UBC-NLP/ARBERTv2** on Arabic translations of the MBIC dataset for binary sentence-level bias classification (Non-biased vs Biased).

---

## Repository layout

```
arabic-bias-detection/
├── configs/
│   ├── baseline_gpt_only.yaml        # Experiment A: GPT-translated sentences only
│   └── main_gpt_claude_mixed.yaml    # Experiment B: GPT + Claude mixed
├── data/
│   └── raw/
│       └── consensus_dataset.csv     # Source data (2 321 rows)
├── notebooks/
│   └── colab_train.ipynb             # One-click Colab notebook
├── outputs/                          # Generated at runtime (git-ignored)
│   └── <experiment_name>/
│       ├── best_model/               # Saved HuggingFace model + tokenizer
│       ├── metrics.json
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       └── predictions.csv
├── src/
│   ├── utils.py                      # Shared helpers
│   ├── prepare_data.py               # Data preparation
│   ├── train.py                      # Fine-tuning
│   ├── evaluate.py                   # Evaluation + plots
│   └── predict.py                    # CLI inference
├── requirements.txt
└── .gitignore
```

---

## Quick start — Google Colab (recommended)

1. Open `notebooks/colab_train.ipynb` in Colab  
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type`)  
3. Update the `REPO_URL` in Cell 2 to your GitHub URL  
4. Select an experiment in Cell 3  
5. `Runtime → Run all`

All outputs (metrics, confusion matrix, predictions) are downloadable from Cell 7.

---

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/arabic-bias-detection.git
cd arabic-bias-detection
pip install -r requirements.txt
```

---

## Running experiments

### Step 1 — Prepare data

```bash
# Experiment A: GPT-only baseline
python src/prepare_data.py --config configs/baseline_gpt_only.yaml

# Experiment B: GPT + Claude mixed
python src/prepare_data.py --config configs/main_gpt_claude_mixed.yaml
```

Outputs: `data/processed/<experiment>/train.csv`, `test.csv`, `dataset_summary.json`

### Step 2 — Train

```bash
python src/train.py --config configs/baseline_gpt_only.yaml
python src/train.py --config configs/main_gpt_claude_mixed.yaml
```

Saves best checkpoint (by F1-macro) to `outputs/<experiment>/best_model/`.

### Step 3 — Evaluate

```bash
python src/evaluate.py --config configs/baseline_gpt_only.yaml
python src/evaluate.py --config configs/main_gpt_claude_mixed.yaml
```

Outputs per experiment:

| File | Contents |
|------|----------|
| `metrics.json` | Accuracy, F1-macro, F1-weighted, Precision, Recall |
| `classification_report.txt` | Per-class precision / recall / F1 |
| `confusion_matrix.png` | Counts + row-normalised heatmaps |
| `predictions.csv` | Row-level predictions with confidence scores |
| `cross_source_metrics.json` | Per-source GPT vs Claude breakdown (main exp only) |

### Step 4 — Predict new sentences

```bash
# Single sentence
python src/predict.py --config configs/main_gpt_claude_mixed.yaml \
    --text "هذا النص يتحيز بشكل واضح ضد الحكومة."

# Plain text file (one sentence per line)
python src/predict.py --config configs/main_gpt_claude_mixed.yaml \
    --input new_sentences.txt

# CSV file
python src/predict.py --config configs/main_gpt_claude_mixed.yaml \
    --input new_sentences.csv --text_col arabic_text \
    --output outputs/new_predictions.csv
```

---

## Experiments

### A — `baseline_gpt_only`

| Setting | Value |
|---------|-------|
| Source | GPT-5.4 mini translations only |
| Rows | 725 |
| Purpose | Single-source ceiling; no cross-source generalisation challenge |
| Cross-source eval | ✗ |

### B — `main_gpt_claude_mixed` *(primary)*

| Setting | Value |
|---------|-------|
| Sources | GPT-5.4 mini + Claude Sonnet 4.6 |
| Rows | 1 468 (725 GPT + 743 Claude) |
| Purpose | Reduce translation-style overfitting; test cross-source robustness |
| Cross-source eval | ✓ |

Both experiments use a **group-aware train/test split** (`sentence_id` as group key) to prevent data leakage — sentences from the same source English sentence never appear in both train and test sets.

---

## Model configuration

Default: `UBC-NLP/ARBERTv2`  
To switch models, edit the `model.name` field in any YAML config:

```yaml
model:
  name: aubmindlab/bert-base-arabertv2   # alternative: AraBERT v2
```

---

## Data notes

| Column | Description |
|--------|-------------|
| `arabic_text` | Arabic translation (source-specific column extracted at prep time) |
| `label` | `0` = Non-biased, `1` = Biased |
| `source` | `gpt` or `claude` |
| `sentence_id` | Original MBIC sentence ID (group key for leakage-safe split) |

**Label source**: `original_label` from MBIC human annotations (NOT the LLM consensus label).  
**Class imbalance**: ~2:1 Non-biased:Biased ratio — handled via class-weighted cross-entropy loss.

---

## Extending to a third experiment (Gemini)

1. Duplicate a config file and set:
   ```yaml
   sources_to_include: [gpt, claude, gemini]
   ```
2. The `gemini_translation` column is already present in `consensus_dataset.csv`  
3. Run `prepare_data.py → train.py → evaluate.py` with the new config

---

## Requirements

| Package | Version |
|---------|---------|
| transformers | 4.44.2 |
| datasets | 2.21.0 |
| torch | ≥ 2.0.0 |
| scikit-learn | ≥ 1.3.0 |
| accelerate | ≥ 0.30.0 |
| evaluate | ≥ 0.4.0 |

GPU recommended (T4 on Colab is sufficient). CPU training is possible but slow (~30 min/epoch on 1 400 samples).
