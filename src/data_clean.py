"""
数据清洗：过滤噪声样本，重新按 lemma 划分。

过滤规则：
  1. 去掉负例中跨词性的样本（pos1 != pos2）—— 太简单，不考查语义消歧
  2. 去掉正例中 sentence1 == sentence2 的样本 —— 无消歧意义
  3. 去掉目标词索引与 surface 不匹配的样本
  4. 去掉重复样本（同句对 + 同词元）
  5. 去掉极短句样本（任一句子 < 5 词）—— 上下文不足
  6. 去掉超过 256 token 的样本（统一 MAX_LEN=256）
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "wic_dataset.jsonl"
OUTPUT_DIR = ROOT_DIR / "data" / "split"


def load_raw():
    with open(RAW_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean(samples):
    """过滤噪声样本。"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cleaned = []
    removed = Counter()
    seen = set()

    for s in samples:
        # 规则1: 负例跨词性
        if not s["label"] and s["pos1"] != s["pos2"]:
            removed["neg_cross_pos"] += 1
            continue

        # 规则2: 正例同句
        if s["label"] and s["sentence1"] == s["sentence2"]:
            removed["pos_same_sent"] += 1
            continue

        # 规则3: 目标词索引不匹配
        tokens1 = s["sentence1"].split()
        tokens2 = s["sentence2"].split()
        strip = lambda x: x.strip(".,;:!?'\"()-").lower()
        if s["index1"] >= len(tokens1) or s["index2"] >= len(tokens2):
            removed["index_oob"] += 1
            continue
        if strip(tokens1[s["index1"]]) != strip(s["surface1"]) or \
           strip(tokens2[s["index2"]]) != strip(s["surface2"]):
            removed["surface_mismatch"] += 1
            continue

        # 规则4: 重复样本
        key = (s["sentence1"], s["sentence2"], s["word"])
        if key in seen:
            removed["duplicate"] += 1
            continue
        seen.add(key)

        # 规则5: 极短句（上下文不足）
        if len(tokens1) < 5 or len(tokens2) < 5:
            removed["short_sent"] += 1
            continue

        # 规则6: 超过 256 token（统一 MAX_LEN=256）
        enc = tokenizer(s["sentence1"], s["sentence2"], truncation=False)
        if len(enc["input_ids"]) > 256:
            removed["over_256_tokens"] += 1
            continue

        cleaned.append(s)

    print(f"原始: {len(samples)} → 清洗后: {len(cleaned)}")
    print(f"移除明细: {dict(removed)}")
    return cleaned


def balance_split(data, rng):
    """对单个 split 做下采样，使正负例数量一致。"""
    pos = [s for s in data if s["label"]]
    neg = [s for s in data if not s["label"]]
    n_min = min(len(pos), len(neg))
    if len(pos) > n_min:
        pos = rng.sample(pos, n_min)
    if len(neg) > n_min:
        neg = rng.sample(neg, n_min)
    balanced = pos + neg
    rng.shuffle(balanced)
    return balanced


def split_by_lemma(samples, train_ratio=0.7, dev_ratio=0.15, seed=42):
    """按 lemma 划分，同一 lemma 只出现在一个集合。"""
    rng = random.Random(seed)
    lemma_to_samples = defaultdict(list)
    for s in samples:
        lemma_to_samples[s["word"]].append(s)

    lemmas = list(lemma_to_samples.keys())
    rng.shuffle(lemmas)

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

    # 下采样使正负例平衡
    train = balance_split(train, rng)
    dev = balance_split(dev, rng)
    test = balance_split(test, rng)

    return train, dev, test


def save_split(train, dev, test):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        pos = sum(1 for s in data if s["label"])
        neg = len(data) - pos
        print(f"  {name:5s}: {len(data)} 条 (正 {pos}, 负 {neg}, 正例比 {pos/len(data)*100:.1f}%)")


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)

    train, dev, test = split_by_lemma(cleaned)
    print(f"\n划分结果:")
    save_split(train, dev, test)
