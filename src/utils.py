"""
共用工具：路径定义、数据加载、评估函数、随机种子设定。
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# 项目根目录（src/ 的上一级）
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT_DIR / "data" / "split"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "results" / "predictions"

MODEL_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """设定全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_predictions(model_name, fold, y_true, y_pred):
    """保存预测结果用于统计检验。"""
    path = PRED_DIR / f"{model_name}_fold{fold}.json"
    with open(path, "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f)


def load_kfold(fold, k=5, seed=42):
    """按 lemma 分组的 k 折交叉验证。fold i 做测试，fold (i+1)%k 做验证，其余做训练。"""
    from collections import defaultdict

    all_samples = load_split("train") + load_split("dev") + load_split("test")

    lemma_to_samples = defaultdict(list)
    for s in all_samples:
        lemma_to_samples[s["word"]].append(s)

    lemmas = sorted(lemma_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(lemmas)

    # 将 lemma 均匀分成 k 组
    groups = [[] for _ in range(k)]
    for i, lemma in enumerate(lemmas):
        groups[i % k].append(lemma)

    test_lemmas = set(groups[fold])
    dev_lemmas = set(groups[(fold + 1) % k])
    train_lemmas = set()
    for j in range(k):
        if j != fold and j != (fold + 1) % k:
            train_lemmas.update(groups[j])

    train = [s for l in sorted(train_lemmas) for s in lemma_to_samples[l]]
    dev = [s for l in sorted(dev_lemmas) for s in lemma_to_samples[l]]
    test = [s for l in sorted(test_lemmas) for s in lemma_to_samples[l]]

    rng2 = random.Random(seed + fold)
    rng2.shuffle(train)
    rng2.shuffle(dev)
    rng2.shuffle(test)

    return train, dev, test


def load_split(name):
    path = SPLIT_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(y_true, y_pred, model_name=""):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    print(f"\n[{model_name}] 评估结果:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision_macro']:.4f}")
    print(f"  Recall:      {metrics['recall_macro']:.4f}")
    print(f"  Macro-F1:    {metrics['f1_macro']:.4f}")
    return metrics
