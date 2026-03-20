"""
在官方 WiC 数据集上评估所有深度学习模型。
官方数据格式：
  data.txt: target_word \t POS \t idx1-idx2 \t sentence1 \t sentence2
  gold.txt: T/F
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, BertModel

from utils import ROOT_DIR, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIC_DIR = ROOT_DIR / "data" / "wic_official"


def load_wic_split(split_name):
    """加载官方 WiC 数据集的一个 split。"""
    data_path = WIC_DIR / split_name / f"{split_name}.data.txt"
    gold_path = WIC_DIR / split_name / f"{split_name}.gold.txt"

    samples = []
    with open(data_path, encoding="utf-8") as f_data, \
         open(gold_path, encoding="utf-8") as f_gold:
        for data_line, gold_line in zip(f_data, f_gold):
            parts = data_line.strip().split("\t")
            word = parts[0]
            pos = parts[1]
            idx1, idx2 = parts[2].split("-")
            sentence1 = parts[3]
            sentence2 = parts[4]
            label = 1 if gold_line.strip() == "T" else 0

            samples.append({
                "word": word,
                "pos": pos.lower(),
                "index1": int(idx1),
                "index2": int(idx2),
                "sentence1": sentence1,
                "sentence2": sentence2,
                "surface1": word,
                "surface2": word,
                "label": label,
            })
    return samples


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
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
                batch["target_pos1"].to(DEVICE),
                batch["target_pos2"].to(DEVICE),
            )
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
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["target_pos1"].to(DEVICE),
                batch["target_pos2"].to(DEVICE),
            )
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


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


def predict_bert_frozen(samples, batch_size=64):
    from model_bert_frozen import MLP, extract_embeddings
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


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("Official WiC Dataset Evaluation")
    print("=" * 70)

    # 加载官方 test set
    test_samples = load_wic_split("test")
    y_true = [s["label"] for s in test_samples]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    print(f"\nTest set: {len(test_samples)} samples (pos={n_pos}, neg={n_neg})")

    # 也加载 dev set 做参考
    dev_samples = load_wic_split("dev")
    dev_true = [s["label"] for s in dev_samples]
    print(f"Dev set:  {len(dev_samples)} samples")

    predictors = {
        "BiLSTM": predict_bilstm,
        "BERT-Frozen+MLP": predict_bert_frozen,
        "BERT": predict_bert,
        "RoBERTa": predict_roberta,
        "SentenceBERT": predict_sbert,
    }

    all_results = {}

    for name, pred_fn in predictors.items():
        print(f"\n评估 {name}...")
        try:
            # Test
            test_preds = pred_fn(test_samples)
            test_metrics = compute_metrics(y_true, test_preds)
            # Dev
            dev_preds = pred_fn(dev_samples)
            dev_metrics = compute_metrics(dev_true, dev_preds)

            all_results[name] = {
                "test": test_metrics,
                "dev": dev_metrics,
            }
            print(f"  Dev:  acc={dev_metrics['accuracy']:.4f}  F1={dev_metrics['f1_macro']:.4f}")
            print(f"  Test: acc={test_metrics['accuracy']:.4f}  F1={test_metrics['f1_macro']:.4f}")
        except Exception as e:
            print(f"  跳过（错误: {e}）")

    # 汇总表
    print("\n" + "=" * 70)
    print(f"{'Model':<20s}  {'Dev Acc':>8s}  {'Dev F1':>8s}  {'Test Acc':>9s}  {'Test F1':>8s}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<20s}  {r['dev']['accuracy']:>8.4f}  {r['dev']['f1_macro']:>8.4f}  "
              f"{r['test']['accuracy']:>9.4f}  {r['test']['f1_macro']:>8.4f}")
    print("=" * 70)

    # 保存结果
    out_path = ROOT_DIR / "results" / "official_wic_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")
