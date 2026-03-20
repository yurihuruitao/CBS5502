"""
将 SemCor 语料库转化为 WIC (Word-in-Context) 格式。

每条样本包含：
  - word:       目标词的 lemma（如 "bank"）
  - pos:        WordNet 词性缩写（n/v/a/r）
  - sentence1, sentence2:  两个包含该词的句子
  - sense1,  sense2:       各自的 WordNet synset（如 "bank.n.01"）
  - pos1,    pos2:         各自的词性全称（noun/verb/adj/adv）
  - index1,  index2:       目标词在句子 token 列表中的起始位置（从 0 开始）
  - label:   True = 同义，False = 异义
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from nltk.corpus import semcor
from nltk.tree import Tree
import nltk

nltk.data.path.append("/workspace/projects/CBS5502/nltk_data")

# WordNet 词性缩写 → 全称
POS_MAP = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}


# ── 第1步：从 SemCor 提取 (词, 义项, 句子, 位置) ──

def parse_sense(sense_label):
    """解析 synset label，如 'take_place.v.01' → ('take place', 'v', 'take_place.v.01')"""
    parts = sense_label.rsplit(".", 2)
    if len(parts) != 3:
        return None
    lemma, pos, _ = parts
    if pos not in POS_MAP:
        return None
    return lemma.replace("_", " "), pos, sense_label


def get_sentence_and_annotations(tagged_sent):
    """返回 (sentence_text, token_list, annotations)。
    annotations[i] = (lemma, pos, synset) 或 None。"""
    tokens = []
    annotations = []

    for chunk in tagged_sent:
        if isinstance(chunk, Tree):
            sense_label = chunk.label()
            leaves = chunk.leaves()
            # 跳过命名实体
            if leaves and leaves[0] == "NE":
                leaves = leaves[1:]
                sense_label = None

            parsed = parse_sense(sense_label) if sense_label else None

            for i, token in enumerate(leaves):
                tokens.append(token)
                # 只在短语首词标注义项
                annotations.append(parsed if i == 0 else None)
        else:
            for token in chunk:
                tokens.append(token)
                annotations.append(None)

    sentence = " ".join(tokens)
    return sentence, tokens, annotations


def extract_sense_examples():
    """遍历 SemCor，收集每个 synset 的所有出现。
    返回:
      sense_to_examples: { "bank.n.01": [ {sentence, lemma, pos, synset, index}, ... ] }
      lemma_senses:      { "bank": {"bank.n.01", "bank.n.02"} }
    """
    sense_to_examples = defaultdict(list)
    lemma_senses = defaultdict(set)

    tagged_sents = semcor.tagged_sents(tag="sem")
    print(f"正在处理 {len(tagged_sents)} 个句子...")

    for tagged_sent in tagged_sents:
        sentence, tokens, annotations = get_sentence_and_annotations(tagged_sent)

        for idx, ann in enumerate(annotations):
            if ann is None:
                continue
            lemma, pos, synset = ann

            example = {
                "sentence": sentence,
                "lemma": lemma,
                "pos": pos,
                "synset": synset,
                "index": idx,           # 目标词在 token 列表中的位置
                "surface": tokens[idx],  # 句中实际出现的词形（可能是屈折形式）
            }
            sense_to_examples[synset].append(example)
            lemma_senses[lemma].add(synset)

    return sense_to_examples, lemma_senses


# ── 第2步：生成 WIC 样本对 ──

def make_pair(ex1, ex2, label):
    """从两条 example 构造一条 WIC 样本。"""
    return {
        "word": ex1["lemma"],
        "pos": POS_MAP[ex1["pos"]],
        # 句子 1
        "sentence1": ex1["sentence"],
        "sense1": ex1["synset"],
        "pos1": POS_MAP[ex1["pos"]],
        "surface1": ex1["surface"],
        "index1": ex1["index"],
        # 句子 2
        "sentence2": ex2["sentence"],
        "sense2": ex2["synset"],
        "pos2": POS_MAP[ex2["pos"]],
        "surface2": ex2["surface"],
        "index2": ex2["index"],
        # 标签
        "label": label,
    }


def generate_wic_pairs(sense_to_examples, lemma_senses, seed=42):
    """
    遍历 SemCor 中所有词，穷举生成 WIC 样本对：
      正例：每个有 ≥2 句的 synset，随机取一对 → label=True
      负例：每个多义词的每一对不同 synset，各取一句 → label=False
    """
    rng = random.Random(seed)
    pairs = []
    pos_count = neg_count = 0

    # ── 正例：同一 synset 的不同句子 ──
    for sense, examples in sense_to_examples.items():
        if len(examples) < 2:
            continue
        ex1, ex2 = rng.sample(examples, 2)
        pairs.append(make_pair(ex1, ex2, label=True))
        pos_count += 1

    # ── 负例：同一 lemma 的不同 synset ──
    for lemma, senses in lemma_senses.items():
        if len(senses) < 2:
            continue
        sense_list = sorted(senses)  # 排序保证可复现
        # 遍历所有 sense 两两组合
        for i in range(len(sense_list)):
            for j in range(i + 1, len(sense_list)):
                s_a, s_b = sense_list[i], sense_list[j]
                if not sense_to_examples[s_a] or not sense_to_examples[s_b]:
                    continue
                ex1 = rng.choice(sense_to_examples[s_a])
                ex2 = rng.choice(sense_to_examples[s_b])
                pairs.append(make_pair(ex1, ex2, label=False))
                neg_count += 1

    rng.shuffle(pairs)
    print(f"生成完毕：正例 {pos_count}，负例 {neg_count}，共 {len(pairs)} 条")
    return pairs


# ── 第3步：保存 ──

def save_wic(pairs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"已保存到 {output_path}")


if __name__ == "__main__":
    sense_to_examples, lemma_senses = extract_sense_examples()
    print(f"共 {len(sense_to_examples)} 个义项，{len(lemma_senses)} 个词元")

    pairs = generate_wic_pairs(sense_to_examples, lemma_senses)
    output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "wic_dataset.jsonl"
    save_wic(pairs, output_path)

    # 打印几条示例
    print("\n══ 示例 ══")
    for p in pairs[:5]:
        tag = "同义 ✓" if p["label"] else "异义 ✗"
        print(f"\n[{tag}]  目标词: {p['word']}  词性: {p['pos']}")
        print(f"  句1: ...{p['sentence1'][:90]}...")
        print(f"       词形=\"{p['surface1']}\"  义项={p['sense1']}  位置={p['index1']}")
        print(f"  句2: ...{p['sentence2'][:90]}...")
        print(f"       词形=\"{p['surface2']}\"  义项={p['sense2']}  位置={p['index2']}")
