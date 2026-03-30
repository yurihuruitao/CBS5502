# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Word-in-Context (WIC) disambiguation research project. Given a target word appearing in two different sentences, the task is binary classification: do they share the same meaning? Built on SemCor corpus data converted to WIC format, with 6 deep learning models compared and statistically validated via 5-fold CV, Bootstrap CI, and McNemar tests.

Primary language of documentation and comments is Chinese (Simplified).

## Documentation

- **README.md** — Project overview, data description, model summaries, quick start guide
- **METHODOLOGY.md** — Detailed experimental methodology (data pipeline, model architectures, training strategies, statistical tests)
- **RESULTS.md** — Training logs, single-split results, 5-fold CV results, Bootstrap CI, McNemar pairwise tests, cross-model comparisons
- **ANALYSIS.md** — Linguistic dimension analysis (POS, polysemy, frequency) and BERT contextualized embedding analysis with visualizations

## Commands

All scripts run from `src/` directory:

```bash
cd src/

# Data pipeline
python data_clean.py              # Clean raw data → data/split/{train,dev,test}.jsonl

# Train individual models (single split)
python model_bilstm.py
python model_bert.py
python model_bert_frozen.py
python model_roberta.py
python model_deberta.py
python model_sbert.py

# Train with k-fold (from project root)
bash run_kfold.sh                 # All models × 5 folds, logs to logs/kfold/
bash run_kfold.sh bert            # Single model × 5 folds
python model_bert.py --fold 0     # Single model × single fold (from src/)

# Evaluation & analysis
python evaluate.py                # Unified evaluation + linguistic/error analysis → results/
python statistical_tests.py       # Bootstrap CI + McNemar tests on k-fold predictions → results/statistical_tests.json
python analyze_bert_embeddings.py # BERT embedding analysis + visualization → plots/
python eval_official_wic.py       # Official WiC benchmark evaluation
```

Install dependencies: `pip install -r requirements.txt`. Also needs `pip install protobuf tiktoken` for DeBERTa-v3 tokenizer. NLTK data needed: `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`.

## Architecture

- **`src/utils.py`** — Shared constants and helpers. `ROOT_DIR` = parent of `src/`. Path constants: `SPLIT_DIR`, `MODEL_DIR`, `PRED_DIR`. Key functions: `load_split(name)`, `load_kfold(fold, k=5)` (lemma-grouped k-fold), `evaluate(y_true, y_pred)`, `save_predictions()`, `set_seed()`.

- **Model files** (`src/model_*.py`) — Each is self-contained with its own Dataset/model class, training loop, and `__main__` entry point. Hyperparameters are module-level constants at the top. All support `--fold N` for k-fold. Models save weights to `models/`, predictions to `results/predictions/`. Two exceptions to the standard pattern:
  - `model_bert_frozen.py` — No Dataset class. Uses `extract_embeddings()` to pre-compute features from frozen BERT, then trains only an `MLP` classifier.
  - `model_sbert.py` — No classes at all. Purely function-based: encodes pairs via `encode_pairs()`, searches optimal cosine similarity threshold on dev set.

- **Model class names** (for imports): `BertWICClassifier`, `RoBERTaWICClassifier`, `DeBERTaWICClassifier`, `BiLSTMClassifier`, `MLP` (frozen BERT).

- **`src/evaluate.py`** — Loads all trained models, runs inference on test set, computes metrics, performs linguistic analysis (by POS, polysemy, frequency), and error analysis. Outputs to `results/`.

- **`src/statistical_tests.py`** — Loads per-fold predictions from `results/predictions/`, computes 5-fold mean±std, Bootstrap 95% CI (1000 resamples), and McNemar pairwise tests (Edwards correction). Outputs to `results/statistical_tests.json`.

- **`src/analyze_bert_embeddings.py`** — Extracts contextualized embeddings from fine-tuned BERT, performs statistical tests (Welch's t, Mann-Whitney U, Cohen's d) and generates visualizations (t-SNE, cosine distributions, activation heatmaps). Outputs to `plots/`.

- **`run_kfold.sh`** — Runs all (or specified) models across 5 folds, logs to `logs/kfold/`, then runs `statistical_tests.py`.

- **Data format** — JSONL files in `data/split/` with fields: `word`, `sentence1`, `sentence2`, `index1`, `index2`, `surface1`, `surface2`, `sense1`, `sense2`, `pos1`, `pos2`, `label`.

## Key Design Decisions

- All transformer models use `MAX_LEN=256`. Data cleaning removes samples exceeding this to prevent target word truncation.
- BERT/RoBERTa classification concatenates `[CLS_emb; target_word1_emb; target_word2_emb]` (2304-dim). DeBERTa-v3 uses 5-way concatenation `[CLS; t1; t2; t1-t2; t1*t2]` (3840-dim) with subword average pooling.
- Train/dev/test split is by lemma (70/15/15) — all samples for a given lemma appear in only one split to prevent data leakage. K-fold also splits by lemma.
- Class imbalance (~37% positive) handled via weighted `CrossEntropyLoss`. DeBERTa uses sqrt-smoothed weights.
- DeBERTa-v3 must use BF16 (not FP16) to avoid NaN. It has known training instability — 2/5 folds collapsed in k-fold CV.

## Data & Model Storage

Large files are on Google Drive (`gdrive:CBS5502/`), publicly shared:

- `models/` (~11 GB): trained weights for all 6 models × (single split + 5 folds)
- `data/` (~400 MB): GloVe embeddings (`glove.6B.100d.txt`), raw dataset, split data, official WiC benchmark

Download via rclone (recommended) or gdown — see README.md §5.2. **Note:** `gdown --folder ... -O data/` creates a nested `data/data/` directory; move contents up one level after download.
