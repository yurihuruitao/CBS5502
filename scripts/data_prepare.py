"""
数据准备：加载 WIC 数据集，统计分布，按 lemma 分层划分 train/dev/test。
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "wic_dataset.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "split"


def load_data(path=DATA_PATH):
    """加载 JSONL 数据集。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def print_stats(samples):
    """打印数据集统计信息。"""
    print(f"\n{'='*50}")
    print(f"总样本数: {len(samples)}")

    # 正负例
    labels = Counter(s["label"] for s in samples)
    print(f"正例 (同义): {labels[True]}  ({labels[True]/len(samples)*100:.1f}%)")
    print(f"负例 (异义): {labels[False]} ({labels[False]/len(samples)*100:.1f}%)")

    # 词性分布
    pos_counts = Counter(s["pos"] for s in samples)
    print(f"\n词性分布:")
    for pos, cnt in pos_counts.most_common():
        print(f"  {pos:6s}: {cnt:6d} ({cnt/len(samples)*100:.1f}%)")

    # 多义程度
    lemma_senses = defaultdict(set)
    for s in samples:
        lemma_senses[s["word"]].add(s["sense1"])
        lemma_senses[s["word"]].add(s["sense2"])
    n_senses = [len(v) for v in lemma_senses.values()]
    print(f"\n覆盖词元数: {len(lemma_senses)}")
    print(f"每词平均义项数: {sum(n_senses)/len(n_senses):.2f}")

    # 多义程度分段
    bins = {"≤3": 0, "4-6": 0, "≥7": 0}
    for n in n_senses:
        if n <= 3:
            bins["≤3"] += 1
        elif n <= 6:
            bins["4-6"] += 1
        else:
            bins["≥7"] += 1
    print(f"多义程度分布 (按词元):")
    for k, v in bins.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")


def split_by_lemma(samples, train_ratio=0.7, dev_ratio=0.15, seed=42):
    """按 lemma 分层划分，同一 lemma 的所有样本只出现在一个集合中。"""
    rng = random.Random(seed)

    # 按 lemma 分组
    lemma_to_samples = defaultdict(list)
    for s in samples:
        lemma_to_samples[s["word"]].append(s)

    lemmas = list(lemma_to_samples.keys())
    rng.shuffle(lemmas)

    # 按样本量累计划分
    total = len(samples)
    train_target = int(total * train_ratio)
    dev_target = int(total * dev_ratio)

    train, dev, test = [], [], []
    count = 0
    for lemma in lemmas:
        group = lemma_to_samples[lemma]
        if count < train_target:
            train.extend(group)
        elif count < train_target + dev_target:
            dev.extend(group)
        else:
            test.extend(group)
        count += len(group)

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def save_split(train, dev, test, output_dir=OUTPUT_DIR):
    """保存划分后的数据。"""
    output_dir.mkdir(exist_ok=True)
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} 条 → {path}")


if __name__ == "__main__":
    samples = load_data()
    print_stats(samples)

    train, dev, test = split_by_lemma(samples)
    save_split(train, dev, test)

    print("\n各集合标签分布:")
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        pos = sum(1 for s in data if s["label"])
        neg = len(data) - pos
        print(f"  {name:5s}: {len(data):6d} (正 {pos}, 负 {neg}, 正例比 {pos/len(data)*100:.1f}%)")
