"""
共用工具：路径定义、数据加载、评估函数。
"""

import json
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# 项目根目录（src/ 的上一级）
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT_DIR / "data" / "split"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


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
