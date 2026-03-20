# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Word-in-Context (WIC) disambiguation research project. Given a target word appearing in two different sentences, the task is binary classification: do they share the same meaning? Built on SemCor corpus data converted to WIC format, with multiple deep learning models compared.

Primary language of documentation and comments is Chinese (Simplified).

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
python model_sbert.py

# Evaluation
python evaluate.py                # Unified evaluation of all models + linguistic/error analysis
python eval_official_wic.py       # Official WiC benchmark evaluation
python analyze_bert_embeddings.py # BERT embedding analysis + visualization plots
```

No `requirements.txt` exists. Key dependencies: PyTorch 2.0+ (CUDA), transformers, scikit-learn, matplotlib, scipy, nltk, sentence-transformers, tqdm.

## Architecture

- **`src/utils.py`** — Shared constants and helpers. Defines `ROOT_DIR`, `SPLIT_DIR`, `MODEL_DIR` paths (ROOT_DIR = parent of `src/`). Provides `load_split(name)` for loading JSONL data and `evaluate(y_true, y_pred)` for metrics.

- **Model files** (`src/model_*.py`) — Each is self-contained: defines its own Dataset class, model class, training loop, and `if __name__ == "__main__"` entry point. Hyperparameters are module-level constants at the top of each file. Models save weights to `models/`.

- **`src/evaluate.py`** — Loads all trained models, runs inference on test set, computes metrics, performs linguistic analysis (by POS, polysemy, frequency), and error analysis. Outputs to `results/`.

- **`src/analyze_bert_embeddings.py`** — Extracts contextualized embeddings from fine-tuned BERT, performs statistical tests and generates visualizations (t-SNE, cosine distributions, activation heatmaps). Outputs to `plots/`.

- **Data format** — JSONL files in `data/split/` with fields: target word, two sentences, word indices, POS tags, synset IDs, and binary label.

## Key Design Decisions

- All transformer models use `MAX_LEN=256`. Data cleaning removes samples exceeding this to prevent target word truncation.
- BERT/RoBERTa classification uses concatenation of `[CLS_emb; target_word1_emb; target_word2_emb]` (2304-dim vector).
- Train/dev/test split is by lemma (70/15/15) — all samples for a given lemma appear in only one split to prevent data leakage.
- Class imbalance (~37% positive) handled via weighted `CrossEntropyLoss`.
- GloVe embeddings file (`data/glove.6B.100d.txt`, ~347MB) is stored locally for BiLSTM.
