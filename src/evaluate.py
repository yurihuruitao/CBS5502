"""
评估：所有深度学习模型在 test set 上的表现 + 语言学维度分析 + 错误分析。
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from transformers import AutoTokenizer, AutoModel

from utils import ROOT_DIR

SPLIT_DIR = ROOT_DIR / "data" / "split"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name):
    path = SPLIT_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ═══════════════════════════════════════════════
# 预测函数
# ═══════════════════════════════════════════════

def predict_baseline_majority(samples):
    return [0] * len(samples)


def predict_baseline_random(samples, seed=42):
    rng = np.random.RandomState(seed)
    train = load_split("train")
    pos_ratio = sum(1 for s in train if s["label"]) / len(train)
    return [int(rng.random() < pos_ratio) for _ in samples]


def predict_bilstm(samples, batch_size=64):
    from model_bilstm import load_glove, WICDataset, collate_fn, BiLSTMClassifier
    word2idx, emb_matrix = load_glove()
    ds = WICDataset(samples, word2idx)
    dl = torch.utils.data.DataLoader(ds, batch_size, collate_fn=collate_fn)

    model = BiLSTMClassifier(emb_matrix).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "bilstm.pt", map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for ids1, ids2, idx1, idx2, _ in dl:
            ids1, ids2, idx1, idx2 = [x.to(DEVICE) for x in [ids1, ids2, idx1, idx2]]
            logits = model(ids1, ids2, idx1, idx2)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def predict_bert(samples, batch_size=32):
    from model_bert import BertWICClassifier, WICDataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = WICDataset(samples, tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size)

    model = BertWICClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "bert.pt", map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            target_pos1 = batch["target_pos1"].to(DEVICE)
            target_pos2 = batch["target_pos2"].to(DEVICE)
            logits = model(input_ids, attention_mask, token_type_ids,
                           target_pos1, target_pos2)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def predict_roberta(samples, batch_size=32):
    from model_roberta import RoBERTaWICClassifier, WICDataset
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    ds = WICDataset(samples, tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size)

    model = RoBERTaWICClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "roberta.pt", map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_pos1 = batch["target_pos1"].to(DEVICE)
            target_pos2 = batch["target_pos2"].to(DEVICE)
            logits = model(input_ids, attention_mask,
                           target_pos1, target_pos2)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def predict_sbert(samples, batch_size=64):
    with open(MODEL_DIR / "sbert_threshold.json") as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]
    model_name = cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    preds = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        enc1 = tokenizer([s["sentence1"] for s in batch], truncation=True,
                         max_length=256, padding=True, return_tensors="pt").to(DEVICE)
        enc2 = tokenizer([s["sentence2"] for s in batch], truncation=True,
                         max_length=256, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out1 = model(**enc1).last_hidden_state.mean(dim=1)
            out2 = model(**enc2).last_hidden_state.mean(dim=1)
        cos = nn.functional.cosine_similarity(out1, out2).cpu().numpy()
        preds.extend([1 if c >= threshold else 0 for c in cos])
    return preds


def predict_bert_frozen(samples, batch_size=64):
    from model_bert_frozen import MLP, extract_embeddings
    from transformers import BertModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    X, y = extract_embeddings(samples, tokenizer, bert, batch_size)
    del bert
    torch.cuda.empty_cache()

    model = MLP(X.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "bert_frozen_mlp.pt", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(X.to(DEVICE)).argmax(1).cpu().tolist()
    return preds


# ═══════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_all_models(test_data):
    y_true = [int(s["label"]) for s in test_data]

    predictors = {
        "Majority": lambda d: predict_baseline_majority(d),
        "Random": lambda d: predict_baseline_random(d),
        "BiLSTM": lambda d: predict_bilstm(d),
        "BERT-Frozen+MLP": lambda d: predict_bert_frozen(d),
        "BERT": lambda d: predict_bert(d),
        "RoBERTa": lambda d: predict_roberta(d),
        "SentenceBERT": lambda d: predict_sbert(d),
    }

    all_preds = {}
    all_metrics = {}

    for name, pred_fn in predictors.items():
        print(f"\n评估 {name}...")
        try:
            preds = pred_fn(test_data)
            metrics = compute_metrics(y_true, preds)
            all_preds[name] = preds
            all_metrics[name] = metrics
            print(f"  acc={metrics['accuracy']:.4f}  macro-F1={metrics['f1_macro']:.4f}")
        except Exception as e:
            print(f"  跳过（错误: {e}）")

    return all_preds, all_metrics


# ═══════════════════════════════════════════════
# 语言学维度分析
# ═══════════════════════════════════════════════

def linguistic_analysis(test_data, all_preds):
    y_true = [int(s["label"]) for s in test_data]

    all_data = load_split("train") + load_split("dev") + test_data
    lemma_senses = defaultdict(set)
    for s in all_data:
        lemma_senses[s["word"]].add(s["sense1"])
        lemma_senses[s["word"]].add(s["sense2"])
    lemma_freq = Counter(s["word"] for s in all_data)

    dimensions = {}

    pos_groups = defaultdict(list)
    for i, s in enumerate(test_data):
        pos_groups[s["pos"]].append(i)
    dimensions["POS"] = dict(pos_groups)

    poly_groups = {"≤3": [], "4-6": [], "≥7": []}
    for i, s in enumerate(test_data):
        n = len(lemma_senses.get(s["word"], set()))
        if n <= 3:
            poly_groups["≤3"].append(i)
        elif n <= 6:
            poly_groups["4-6"].append(i)
        else:
            poly_groups["≥7"].append(i)
    dimensions["Polysemy"] = poly_groups

    freqs = sorted(lemma_freq.values())
    low_thr = freqs[len(freqs) // 3]
    high_thr = freqs[2 * len(freqs) // 3]
    freq_groups = {"low": [], "mid": [], "high": []}
    for i, s in enumerate(test_data):
        f = lemma_freq[s["word"]]
        if f <= low_thr:
            freq_groups["low"].append(i)
        elif f <= high_thr:
            freq_groups["mid"].append(i)
        else:
            freq_groups["high"].append(i)
    dimensions["Frequency"] = freq_groups

    results = {}
    for dim_name, groups in dimensions.items():
        results[dim_name] = {}
        for group_name, indices in groups.items():
            if len(indices) < 10:
                continue
            group_true = [y_true[i] for i in indices]
            results[dim_name][group_name] = {"n": len(indices)}
            for model_name, preds in all_preds.items():
                group_pred = [preds[i] for i in indices]
                f1 = f1_score(group_true, group_pred, average="macro", zero_division=0)
                results[dim_name][group_name][model_name] = f1

    return results


# ═══════════════════════════════════════════════
# 错误分析
# ═══════════════════════════════════════════════

def error_analysis(test_data, all_preds):
    y_true = [int(s["label"]) for s in test_data]
    analysis = {}

    for model_name, preds in all_preds.items():
        errors = []
        for i, (s, pred, true) in enumerate(zip(test_data, preds, y_true)):
            if pred != true:
                errors.append({
                    "index": i,
                    "word": s["word"],
                    "pos": s["pos"],
                    "sense1": s["sense1"],
                    "sense2": s["sense2"],
                    "true_label": true,
                    "pred_label": pred,
                    "sentence1_preview": s["sentence1"][:80],
                    "sentence2_preview": s["sentence2"][:80],
                })

        sense_pair_errors = Counter()
        for e in errors:
            pair = tuple(sorted([e["sense1"], e["sense2"]]))
            sense_pair_errors[pair] += 1

        analysis[model_name] = {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(y_true),
            "top_confused_pairs": sense_pair_errors.most_common(10),
            "sample_errors": errors[:5],
        }

    if "BERT" in all_preds and "BiLSTM" in all_preds:
        bert_preds = all_preds["BERT"]
        bilstm_preds = all_preds["BiLSTM"]
        analysis["cross_model"] = {
            "BERT_wrong_BiLSTM_right": sum(
                1 for i in range(len(y_true))
                if bert_preds[i] != y_true[i] and bilstm_preds[i] == y_true[i]
            ),
            "BiLSTM_wrong_BERT_right": sum(
                1 for i in range(len(y_true))
                if bilstm_preds[i] != y_true[i] and bert_preds[i] == y_true[i]
            ),
        }

    return analysis


# ═══════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════

def save_visualizations(all_metrics, ling_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
    })

    RESULT_DIR.mkdir(exist_ok=True)

    models = list(all_metrics.keys())
    f1_scores = [all_metrics[m]["f1_macro"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, f1_scores, color="steelblue")
    ax.set_ylabel("Macro F1")
    ax.set_title("Model Comparison (Macro F1)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    for dim_name, groups in ling_results.items():
        group_names = list(groups.keys())
        model_names = [m for m in models if m in list(groups.values())[0]]
        if not model_names:
            continue

        x = np.arange(len(group_names))
        width = 0.8 / max(len(model_names), 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, m in enumerate(model_names):
            vals = [groups[g].get(m, 0) for g in group_names]
            ax.bar(x + j * width, vals, width, label=m)
        ax.set_xticks(x + width * len(model_names) / 2)
        ax.set_xticklabels(group_names)
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Linguistic Analysis: {dim_name}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"ling_{dim_name}.png", dpi=150)
        plt.close()

    print(f"图表已保存到 {RESULT_DIR}/")


# ═══════════════════════════════════════════════

def save_all(all_metrics, ling_results, error_results):
    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "linguistic_analysis.json", "w", encoding="utf-8") as f:
        json.dump(ling_results, f, ensure_ascii=False, indent=2, default=str)
    with open(RESULT_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_results, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "="*70)
    print(f"{'Model':<22s} {'Acc':>7s} {'P(macro)':>9s} {'R(macro)':>9s} {'F1(macro)':>10s}")
    print("-"*70)
    for name, m in all_metrics.items():
        print(f"{name:<22s} {m['accuracy']:>7.4f} {m['precision_macro']:>9.4f} "
              f"{m['recall_macro']:>9.4f} {m['f1_macro']:>10.4f}")
    print("="*70)


if __name__ == "__main__":
    test_data = load_split("test")
    print(f"Test set: {len(test_data)} 条")

    all_preds, all_metrics = evaluate_all_models(test_data)
    ling_results = linguistic_analysis(test_data, all_preds)
    error_results = error_analysis(test_data, all_preds)

    save_all(all_metrics, ling_results, error_results)
    save_visualizations(all_metrics, ling_results)
