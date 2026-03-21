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
python data_clean.py              # Clean raw data and split into train/dev/test

# Train individual models
python model_bilstm.py
python model_bert.py
python model_bert_frozen.py
python model_roberta.py
python model_deberta.py
python model_sbert.py

# Train with k-fold (from project root)
bash run_kfold.sh                 # All models × 5 folds
bash run_kfold.sh bert            # Single model × 5 folds
python model_bert.py --fold 0     # Single model × single fold

# Evaluation
python evaluate.py                # Unified evaluation of all models + linguistic/error analysis
python eval_official_wic.py       # Official WiC benchmark evaluation
python analyze_bert_embeddings.py # BERT embedding analysis + visualization plots
python statistical_tests.py       # Bootstrap CI + McNemar tests on k-fold predictions
```

Key dependencies: PyTorch 2.0+ (CUDA), transformers, scikit-learn, matplotlib, scipy, nltk, sentence-transformers, tqdm. Install via `pip install -r requirements.txt`.

## Architecture

- **`src/utils.py`** — Shared constants and helpers. Defines `ROOT_DIR`, `SPLIT_DIR`, `MODEL_DIR`, `PRED_DIR` paths (ROOT_DIR = parent of `src/`). Provides `load_split(name)` for JSONL data, `load_kfold(fold, k=5)` for k-fold splits by lemma, `evaluate(y_true, y_pred)` for metrics, `save_predictions()` for storing per-fold predictions, and `set_seed()` for reproducibility.

- **Model files** (`src/model_*.py`) — Each is self-contained: defines its own Dataset class, model class, training loop, and `if __name__ == "__main__"` entry point. Hyperparameters are module-level constants at the top of each file. All support `--fold N` argument for k-fold training. Models save weights to `models/`, predictions to `results/predictions/`.

- **`src/evaluate.py`** — Loads all trained models, runs inference on test set, computes metrics, performs linguistic analysis (by POS, polysemy, frequency), and error analysis. Outputs to `results/`.

- **`src/statistical_tests.py`** — Loads per-fold predictions from `results/predictions/`, computes 5-fold mean±std, Bootstrap 95% CI (1000 resamples), and McNemar pairwise tests (Edwards correction) for all model pairs. Outputs to `results/statistical_tests.json`.

- **`src/analyze_bert_embeddings.py`** — Extracts contextualized embeddings from fine-tuned BERT, performs statistical tests (Welch's t, Mann-Whitney U, Cohen's d) and generates visualizations (t-SNE, cosine distributions, activation heatmaps). Outputs to `plots/`.

- **`run_kfold.sh`** — Shell script that runs all (or specified) models across 5 folds, logs to `logs/kfold/`, then runs `statistical_tests.py`.

- **Data format** — JSONL files in `data/split/` with fields: `word`, `sentence1`, `sentence2`, `index1`, `index2`, `surface1`, `surface2`, `sense1`, `sense2`, `pos1`, `pos2`, `label`.

## Key Design Decisions

- All transformer models use `MAX_LEN=256`. Data cleaning removes samples exceeding this to prevent target word truncation.
- BERT/RoBERTa classification uses concatenation of `[CLS_emb; target_word1_emb; target_word2_emb]` (2304-dim). DeBERTa-v3 uses 5-way concatenation `[CLS; t1; t2; t1-t2; t1*t2]` (3840-dim) with subword average pooling.
- Train/dev/test split is by lemma (70/15/15) — all samples for a given lemma appear in only one split to prevent data leakage. K-fold also splits by lemma.
- Class imbalance (~37% positive) handled via weighted `CrossEntropyLoss`. DeBERTa uses sqrt-smoothed weights.
- DeBERTa-v3 must use BF16 (not FP16) to avoid NaN. It has known training instability — 2/5 folds collapsed in k-fold CV.
- GloVe embeddings file (`data/glove.6B.100d.txt`, ~347MB) and model weights (~11GB) are stored on Google Drive. Download instructions in README.md §5.2.

## Data & Model Storage

Large files are synced to Google Drive (`gdrive:CBS5502/`), publicly shared:

- `models/` (~11 GB): trained weights for all 6 models × (single split + 5 folds)
- `data/` (~400 MB): GloVe embeddings, raw dataset, split data, official WiC benchmark

Download via rclone or gdown — see README.md §5.2.
