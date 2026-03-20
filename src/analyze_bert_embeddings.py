"""
Fine-tuned BERT 上下文向量分析与可视化

生成内容：
  1. 正例 vs 负例 目标词 cosine similarity 分布直方图
  2. 目标词 embedding t-SNE 降维散点图（按 label 着色）
  3. 按词性(POS) 分组的 cosine similarity 箱线图
  4. 高置信正例 / 负例的 cosine similarity 示例
  5. embedding 各维度激活值热力图（采样）
  6. 统计摘要打印
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, BertModel
from utils import load_split, MODEL_DIR, ROOT_DIR
from model_bert import BertWICClassifier, MAX_LEN, MODEL_NAME

# Global font: Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT_DIR / "plots"


@torch.no_grad()
def extract_target_embeddings(samples, tokenizer, model, batch_size=64):
    """Extract target word embeddings from the fine-tuned BERT."""
    model.eval()
    emb1_list, emb2_list, labels, words, pos_list = [], [], [], [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        encs = tokenizer(
            [s["sentence1"] for s in batch],
            [s["sentence2"] for s in batch],
            truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt",
        ).to(DEVICE)

        hidden = model.bert(**encs).last_hidden_state

        for j, s in enumerate(batch):
            word_ids = tokenizer(
                s["sentence1"], s["sentence2"],
                truncation=True, max_length=MAX_LEN,
            ).word_ids()
            seq_ids = tokenizer(
                s["sentence1"], s["sentence2"],
                truncation=True, max_length=MAX_LEN,
            ).sequence_ids()

            pos1, pos2 = 0, 0
            for k, (si, wi) in enumerate(zip(seq_ids, word_ids)):
                if si == 0 and wi == s["index1"]:
                    pos1 = k
                    break
            for k, (si, wi) in enumerate(zip(seq_ids, word_ids)):
                if si == 1 and wi == s["index2"]:
                    pos2 = k
                    break

            emb1_list.append(hidden[j, pos1].cpu().numpy())
            emb2_list.append(hidden[j, pos2].cpu().numpy())
            labels.append(int(s["label"]))
            words.append(s["word"])
            pos_list.append(s.get("pos1", "unk"))

        if (i // batch_size) % 20 == 0:
            print(f"  Progress: {min(i + batch_size, len(samples))}/{len(samples)}")

    return (np.array(emb1_list), np.array(emb2_list),
            np.array(labels), words, pos_list)


def cosine_sim(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1) + 1e-8
    norm_b = np.linalg.norm(b, axis=1) + 1e-8
    return dot / (norm_a * norm_b)


def plot_cosine_distribution(cos_pos, cos_neg):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cos_pos, bins=60, alpha=0.6,
            label=f"Positive (n={len(cos_pos)}, $\\mu$={cos_pos.mean():.3f})",
            color="steelblue")
    ax.hist(cos_neg, bins=60, alpha=0.6,
            label=f"Negative (n={len(cos_neg)}, $\\mu$={cos_neg.mean():.3f})",
            color="salmon")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Fine-tuned BERT: Target Word Cosine Similarity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cosine_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved cosine_distribution.png")


def plot_tsne(emb1, emb2, labels):
    diff = emb1 - emb2
    n = len(diff)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        diff, lab = diff[idx], labels[idx]
    else:
        lab = labels

    print("  Running t-SNE (diff vectors)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    proj = tsne.fit_transform(diff)

    fig, ax = plt.subplots(figsize=(8, 8))
    for lbl, color, name in [(1, "steelblue", "Positive (same sense)"),
                              (0, "salmon", "Negative (diff sense)")]:
        mask = lab == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, s=8, alpha=0.4, label=name)
    ax.set_title("t-SNE of Target Word Embedding Differences (emb1 $-$ emb2)")
    ax.legend(markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/tsne_diff.png", dpi=150)
    plt.close(fig)
    print("  Saved tsne_diff.png")

    # Joint t-SNE of emb1 and emb2
    all_emb = np.concatenate([emb1, emb2], axis=0)
    if len(all_emb) > 6000:
        idx = np.random.RandomState(42).choice(len(emb1), 3000, replace=False)
        all_emb = np.concatenate([emb1[idx], emb2[idx]], axis=0)
        lab2 = labels[idx]
    else:
        idx = np.arange(len(emb1))
        lab2 = labels

    print("  Running t-SNE (emb1 + emb2 joint)...")
    proj2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_emb)
    n_half = len(idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, p, l in [
        (axes[0], "Sentence 1 Target Embeddings", proj2[:n_half], lab2),
        (axes[1], "Sentence 2 Target Embeddings", proj2[n_half:], lab2),
    ]:
        for lbl, color, name in [(1, "steelblue", "Positive"), (0, "salmon", "Negative")]:
            mask = l == lbl
            ax.scatter(p[mask, 0], p[mask, 1], c=color, s=8, alpha=0.4, label=name)
        ax.set_title(title)
        ax.legend(markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/tsne_emb12.png", dpi=150)
    plt.close(fig)
    print("  Saved tsne_emb12.png")


def plot_pos_boxplot(cos_sims, labels, pos_list):
    pos_types = sorted(set(pos_list))
    data_pos = {p: [] for p in pos_types}
    data_neg = {p: [] for p in pos_types}
    for c, l, p in zip(cos_sims, labels, pos_list):
        if l == 1:
            data_pos[p].append(c)
        else:
            data_neg[p].append(c)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    tick_labels = []
    bp_data_pos, bp_data_neg = [], []
    for i, p in enumerate(pos_types):
        bp_data_pos.append(data_pos[p])
        bp_data_neg.append(data_neg[p])
        positions.append(i)
        tick_labels.append(p.capitalize())

    width = 0.35
    pos_positions = [p - width / 2 for p in positions]
    neg_positions = [p + width / 2 for p in positions]

    bp1 = ax.boxplot(bp_data_pos, positions=pos_positions, widths=width,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(bp_data_neg, positions=neg_positions, widths=width,
                     patch_artist=True, showfliers=False)
    for box in bp1["boxes"]:
        box.set_facecolor("steelblue")
        box.set_alpha(0.7)
    for box in bp2["boxes"]:
        box.set_facecolor("salmon")
        box.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Part of Speech")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity by Part of Speech")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Positive", "Negative"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cosine_by_pos.png", dpi=150)
    plt.close(fig)
    print("  Saved cosine_by_pos.png")


def plot_activation_heatmap(emb1, emb2, labels):
    rng = np.random.RandomState(42)
    pos_idx = rng.choice(np.where(labels == 1)[0], min(50, (labels == 1).sum()), replace=False)
    neg_idx = rng.choice(np.where(labels == 0)[0], min(50, (labels == 0).sum()), replace=False)

    diff_pos = emb1[pos_idx] - emb2[pos_idx]
    diff_neg = emb1[neg_idx] - emb2[neg_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].imshow(diff_pos[:, :200], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[0].set_title("Positive Pairs (emb1 $-$ emb2), Dims 0\u2013199")
    axes[0].set_ylabel("Sample")
    axes[1].imshow(diff_neg[:, :200], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[1].set_title("Negative Pairs (emb1 $-$ emb2), Dims 0\u2013199")
    axes[1].set_ylabel("Sample")
    axes[1].set_xlabel("Dimension")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/activation_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved activation_heatmap.png")


def plot_norm_distribution(emb1, emb2, labels):
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    norm_diff = np.abs(norm1 - norm2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(norm1, bins=50, alpha=0.6, label="emb1", color="steelblue")
    axes[0].hist(norm2, bins=50, alpha=0.6, label="emb2", color="salmon")
    axes[0].set_title("Target Word Embedding L2 Norm")
    axes[0].set_xlabel("L2 Norm")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(norm_diff[labels == 1], bins=50, alpha=0.6, label="Positive", color="steelblue")
    axes[1].hist(norm_diff[labels == 0], bins=50, alpha=0.6, label="Negative", color="salmon")
    axes[1].set_title("|norm(emb1) $-$ norm(emb2)| by Label")
    axes[1].set_xlabel("Norm Difference")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/norm_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved norm_distribution.png")


def cohens_d(a, b):
    """计算 Cohen's d 效应量。"""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std


def bootstrap_mean_diff_ci(a, b, n_boot=10000, alpha=0.05):
    """Bootstrap 计算两组均值差的置信区间。"""
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        sa = a[rng.randint(0, len(a), len(a))]
        sb = b[rng.randint(0, len(b), len(b))]
        diffs.append(sa.mean() - sb.mean())
    diffs = np.sort(diffs)
    lo = diffs[int(n_boot * alpha / 2)]
    hi = diffs[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def print_statistics(cos_sims, labels, words, pos_list, emb1, emb2):
    from scipy import stats
    from sklearn.metrics import f1_score

    cos_pos = cos_sims[labels == 1]
    cos_neg = cos_sims[labels == 0]

    print("\n" + "=" * 70)
    print("Statistics Summary")
    print("=" * 70)
    print(f"  Samples:  Positive {(labels==1).sum()},  Negative {(labels==0).sum()}")

    # ── 1. 余弦相似度: 描述统计 + 假设检验 ──
    print(f"\n  [1] Cosine Similarity — Descriptive Statistics")
    print(f"    Positive  mean={cos_pos.mean():.4f}  std={cos_pos.std():.4f}  "
          f"median={np.median(cos_pos):.4f}  min={cos_pos.min():.4f}  max={cos_pos.max():.4f}")
    print(f"    Negative  mean={cos_neg.mean():.4f}  std={cos_neg.std():.4f}  "
          f"median={np.median(cos_neg):.4f}  min={cos_neg.min():.4f}  max={cos_neg.max():.4f}")

    print(f"\n  [2] Cosine Similarity — Hypothesis Tests")
    # Welch's t-test (不假设方差相等)
    t_stat, p_val = stats.ttest_ind(cos_pos, cos_neg, equal_var=False)
    d = cohens_d(cos_pos, cos_neg)
    ci_lo, ci_hi = bootstrap_mean_diff_ci(cos_pos, cos_neg)
    print(f"    Welch's t-test:  t={t_stat:.4f},  p={p_val:.2e}")
    print(f"    Cohen's d:       {d:.4f}  ", end="")
    if abs(d) < 0.2:
        print("(negligible)")
    elif abs(d) < 0.5:
        print("(small)")
    elif abs(d) < 0.8:
        print("(medium)")
    else:
        print("(large)")
    print(f"    Mean diff:       {cos_pos.mean() - cos_neg.mean():.4f}")
    print(f"    95% Bootstrap CI of mean diff: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Mann-Whitney U (非参数检验)
    u_stat, u_p = stats.mannwhitneyu(cos_pos, cos_neg, alternative="two-sided")
    print(f"    Mann-Whitney U:  U={u_stat:.0f},  p={u_p:.2e}")

    # ── 2. 按 POS 分组检验 ──
    print(f"\n  [3] Cosine Similarity by POS — Hypothesis Tests")
    print(f"    {'POS':5s}  {'Pos_μ':>7s}  {'Neg_μ':>7s}  {'Diff':>7s}  "
          f"{'t':>8s}  {'p':>10s}  {'Cohen_d':>8s}  {'95% CI':>20s}")
    print(f"    {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*20}")
    for p in sorted(set(pos_list)):
        mask = np.array([pp == p for pp in pos_list])
        cp = cos_sims[mask & (labels == 1)]
        cn = cos_sims[mask & (labels == 0)]
        t_s, p_v = stats.ttest_ind(cp, cn, equal_var=False)
        d_v = cohens_d(cp, cn)
        ci_l, ci_h = bootstrap_mean_diff_ci(cp, cn)
        print(f"    {p:5s}  {cp.mean():7.4f}  {cn.mean():7.4f}  {cp.mean()-cn.mean():7.4f}  "
              f"{t_s:8.3f}  {p_v:10.2e}  {d_v:8.4f}  [{ci_l:.4f}, {ci_h:.4f}]")

    # ── 3. 范数差异检验 ──
    print(f"\n  [4] Norm Difference |norm(emb1) - norm(emb2)| — Hypothesis Tests")
    norm_diff = np.abs(np.linalg.norm(emb1, axis=1) - np.linalg.norm(emb2, axis=1))
    nd_pos = norm_diff[labels == 1]
    nd_neg = norm_diff[labels == 0]
    print(f"    Positive  mean={nd_pos.mean():.4f}  std={nd_pos.std():.4f}")
    print(f"    Negative  mean={nd_neg.mean():.4f}  std={nd_neg.std():.4f}")
    t_n, p_n = stats.ttest_ind(nd_pos, nd_neg, equal_var=False)
    d_n = cohens_d(nd_neg, nd_pos)  # 负例范数差更大，方向: neg - pos
    ci_n_lo, ci_n_hi = bootstrap_mean_diff_ci(nd_neg, nd_pos)
    print(f"    Welch's t-test:  t={t_n:.4f},  p={p_n:.2e}")
    print(f"    Cohen's d (neg-pos): {d_n:.4f}")
    print(f"    95% Bootstrap CI of mean diff (neg-pos): [{ci_n_lo:.4f}, {ci_n_hi:.4f}]")
    u_n, u_pn = stats.mannwhitneyu(nd_neg, nd_pos, alternative="two-sided")
    print(f"    Mann-Whitney U:  U={u_n:.0f},  p={u_pn:.2e}")

    # ── 4. 欧氏距离检验 ──
    print(f"\n  [5] Euclidean Distance ||emb1 - emb2||₂ — Hypothesis Tests")
    euc_dist = np.linalg.norm(emb1 - emb2, axis=1)
    ed_pos = euc_dist[labels == 1]
    ed_neg = euc_dist[labels == 0]
    print(f"    Positive  mean={ed_pos.mean():.4f}  std={ed_pos.std():.4f}")
    print(f"    Negative  mean={ed_neg.mean():.4f}  std={ed_neg.std():.4f}")
    t_e, p_e = stats.ttest_ind(ed_pos, ed_neg, equal_var=False)
    d_e = cohens_d(ed_neg, ed_pos)
    ci_e_lo, ci_e_hi = bootstrap_mean_diff_ci(ed_neg, ed_pos)
    print(f"    Welch's t-test:  t={t_e:.4f},  p={p_e:.2e}")
    print(f"    Cohen's d (neg-pos): {d_e:.4f}")
    print(f"    95% Bootstrap CI of mean diff (neg-pos): [{ci_e_lo:.4f}, {ci_e_hi:.4f}]")

    # ── 5. 余弦阈值分类 ──
    best_thr, best_f1 = 0, 0
    for thr in np.arange(0.0, 1.0, 0.005):
        preds = (cos_sims >= thr).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"\n  [6] Cosine Threshold Classification:")
    print(f"    Best threshold={best_thr:.3f}  macro-F1={best_f1:.4f}")

    # ── 6. 难例 ──
    print(f"\n  [7] Hardest positive pairs (lowest cosine, same sense):")
    pos_indices = np.where(labels == 1)[0]
    sorted_pos = pos_indices[np.argsort(cos_sims[pos_indices])]
    for idx in sorted_pos[:5]:
        print(f"    cos={cos_sims[idx]:.4f}  word=\"{words[idx]}\"")

    print(f"\n  [8] Hardest negative pairs (highest cosine, diff sense):")
    neg_indices = np.where(labels == 0)[0]
    sorted_neg = neg_indices[np.argsort(-cos_sims[neg_indices])]
    for idx in sorted_neg[:5]:
        print(f"    cos={cos_sims[idx]:.4f}  word=\"{words[idx]}\"")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertWICClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "bert.pt", map_location=DEVICE))
    model.eval()
    print("Fine-tuned BERT model loaded.\n")

    test_samples = load_split("test")
    print("Extracting test set target word embeddings...")
    emb1, emb2, labels, words, pos_list = extract_target_embeddings(
        test_samples, tokenizer, model
    )

    cos_sims = cosine_sim(emb1, emb2)
    cos_pos = cos_sims[labels == 1]
    cos_neg = cos_sims[labels == 0]

    print_statistics(cos_sims, labels, words, pos_list, emb1, emb2)

    print("\nGenerating plots...")
    plot_cosine_distribution(cos_pos, cos_neg)
    plot_tsne(emb1, emb2, labels)
    plot_pos_boxplot(cos_sims, labels, pos_list)
    plot_activation_heatmap(emb1, emb2, labels)
    plot_norm_distribution(emb1, emb2, labels)

    print(f"\nAll plots saved to {OUT_DIR}/")
