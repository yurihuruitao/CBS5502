"""
Sentence-BERT 余弦相似度 + 阈值搜索
"""

import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score as sklearn_f1
from tqdm import tqdm
from utils import load_split, evaluate, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════
# 超参数（在这里调）
# ══════════════════════════════════
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 可换: "all-mpnet-base-v2" 等
MAX_LEN = 256
BATCH_SIZE = 32
THRESHOLD_RANGE = (0.5, 1.0, 0.01)  # (start, stop, step) 阈值搜索范围
# ══════════════════════════════════


def encode_pairs(model, tokenizer, samples):
    """编码句子对，返回余弦相似度列表。"""
    all_sims = []
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Encoding"):
        batch = samples[i:i+BATCH_SIZE]
        enc1 = tokenizer([s["sentence1"] for s in batch], truncation=True,
                         max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        enc2 = tokenizer([s["sentence2"] for s in batch], truncation=True,
                         max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out1 = model(**enc1).last_hidden_state.mean(dim=1)
            out2 = model(**enc2).last_hidden_state.mean(dim=1)
        cos = nn.functional.cosine_similarity(out1, out2)
        all_sims.extend(cos.cpu().numpy().tolist())
    return all_sims


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"超参数: model={MODEL_NAME}, threshold_range={THRESHOLD_RANGE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # 在 dev 上搜索最优阈值
    dev_data = load_split("dev")
    dev_sims = encode_pairs(model, tokenizer, dev_data)
    dev_labels = [int(s["label"]) for s in dev_data]

    best_thr, best_f1 = 0, 0
    start, stop, step = THRESHOLD_RANGE
    for thr in np.arange(start, stop, step):
        preds = [1 if s >= thr else 0 for s in dev_sims]
        f1 = sklearn_f1(dev_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"  最优阈值: {best_thr:.2f}  dev macro-F1: {best_f1:.4f}")

    # 保存配置
    with open(MODEL_DIR / "sbert_threshold.json", "w") as f:
        json.dump({"threshold": float(best_thr), "model_name": MODEL_NAME}, f)

    # 评估 test
    test_data = load_split("test")
    test_sims = encode_pairs(model, tokenizer, test_data)
    test_labels = [int(s["label"]) for s in test_data]
    test_preds = [1 if s >= best_thr else 0 for s in test_sims]
    evaluate(test_labels, test_preds, "SentenceBERT")
