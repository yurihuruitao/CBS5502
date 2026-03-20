"""
统计检验：基于 5 折交叉验证的 Bootstrap CI + McNemar 检验。
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import chi2

PRED_DIR = Path(__file__).resolve().parent.parent / "results" / "predictions"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = ["bilstm", "bert_frozen", "bert", "roberta", "deberta", "sbert"]
MODEL_NAMES = {
    "bilstm": "BiLSTM",
    "bert_frozen": "BERT-Frozen+MLP",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "deberta": "DeBERTa-v3",
    "sbert": "SentenceBERT",
}
K = 5
N_BOOTSTRAP = 1000
SEED = 42


def load_predictions(model, fold):
    path = PRED_DIR / f"{model}_fold{fold}.json"
    with open(path) as f:
        d = json.load(f)
    return d["y_true"], d["y_pred"]


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap 置信区间。"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return lo, hi


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar 检验（带 Edwards 连续性校正）。返回 chi2 统计量和 p 值。"""
    y_true = np.array(y_true)
    correct_a = (np.array(y_pred_a) == y_true)
    correct_b = (np.array(y_pred_b) == y_true)

    # b: A 对 B 错;  c: A 错 B 对
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return 0.0, 1.0

    # Edwards 连续性校正
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main():
    # 检查哪些模型有预测文件
    available = []
    for m in MODELS:
        folds_found = [f for f in range(K) if (PRED_DIR / f"{m}_fold{f}.json").exists()]
        if len(folds_found) == K:
            available.append(m)
        elif len(folds_found) > 0:
            print(f"警告: {m} 只有 {len(folds_found)}/{K} 折结果，跳过")

    if not available:
        print("没有找到任何完整的 5 折预测结果。请先运行 run_kfold.sh")
        return

    print(f"找到 {len(available)} 个模型的完整 5 折结果: {[MODEL_NAMES[m] for m in available]}\n")

    # ═══════════════════════════════════
    # 1. 每折指标 + 均值±标准差
    # ═══════════════════════════════════
    print("=" * 70)
    print("1. 5 折交叉验证结果 (mean ± std)")
    print("=" * 70)

    all_fold_metrics = {}  # model -> {metric -> [fold0, fold1, ...]}

    for m in available:
        fold_f1s = []
        fold_accs = []
        for fold in range(K):
            y_true, y_pred = load_predictions(m, fold)
            fold_f1s.append(macro_f1(y_true, y_pred))
            fold_accs.append(accuracy_score(y_true, y_pred))
        all_fold_metrics[m] = {"f1": fold_f1s, "acc": fold_accs}

        print(f"\n{MODEL_NAMES[m]}:")
        for fold in range(K):
            print(f"  Fold {fold}: Acc={fold_accs[fold]:.4f}  Macro-F1={fold_f1s[fold]:.4f}")
        print(f"  ────────────────────────────────")
        print(f"  Mean:  Acc={np.mean(fold_accs):.4f}±{np.std(fold_accs):.4f}  "
              f"Macro-F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}")

    # 汇总表
    print(f"\n{'模型':<20s} {'Accuracy':>18s} {'Macro-F1':>18s}")
    print("-" * 58)
    for m in available:
        acc = all_fold_metrics[m]["acc"]
        f1s = all_fold_metrics[m]["f1"]
        print(f"{MODEL_NAMES[m]:<20s} {np.mean(acc):.4f}±{np.std(acc):.4f}      "
              f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    # ═══════════════════════════════════
    # 2. Bootstrap 95% CI（合并全部 5 折预测）
    # ═══════════════════════════════════
    print(f"\n{'=' * 70}")
    print("2. Bootstrap 95% 置信区间 (合并 5 折预测)")
    print("=" * 70)

    merged = {}  # model -> (all_true, all_pred)
    for m in available:
        all_true, all_pred = [], []
        for fold in range(K):
            y_true, y_pred = load_predictions(m, fold)
            all_true.extend(y_true)
            all_pred.extend(y_pred)
        merged[m] = (all_true, all_pred)

    print(f"\n{'模型':<20s} {'Macro-F1':>10s} {'95% CI':>20s}")
    print("-" * 52)
    for m in available:
        y_true, y_pred = merged[m]
        point = macro_f1(y_true, y_pred)
        lo, hi = bootstrap_ci(y_true, y_pred, macro_f1)
        print(f"{MODEL_NAMES[m]:<20s} {point:>10.4f} [{lo:.4f}, {hi:.4f}]")

    # ═══════════════════════════════════
    # 3. McNemar 检验（逐折 + 汇总）
    # ═══════════════════════════════════
    print(f"\n{'=' * 70}")
    print("3. McNemar 配对检验")
    print("=" * 70)

    pairs = list(combinations(available, 2))
    mcnemar_results = []

    for m_a, m_b in pairs:
        fold_pvals = []
        for fold in range(K):
            y_true_a, y_pred_a = load_predictions(m_a, fold)
            y_true_b, y_pred_b = load_predictions(m_b, fold)
            assert y_true_a == y_true_b, f"fold {fold}: {m_a} 和 {m_b} 的 y_true 不一致"
            _, p = mcnemar_test(y_true_a, y_pred_a, y_pred_b)
            fold_pvals.append(p)

        # 合并 5 折做一次总体 McNemar
        all_true_a, all_pred_a = merged[m_a]
        all_true_b, all_pred_b = merged[m_b]
        chi2_all, p_all = mcnemar_test(all_true_a, all_pred_a, all_pred_b)

        sig = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else "n.s."
        mcnemar_results.append((m_a, m_b, p_all, sig, fold_pvals))

    print(f"\n{'模型 A':<20s} {'模型 B':<20s} {'p-value':>10s} {'显著性':>6s}")
    print("-" * 58)
    for m_a, m_b, p_all, sig, _ in mcnemar_results:
        print(f"{MODEL_NAMES[m_a]:<20s} {MODEL_NAMES[m_b]:<20s} {p_all:>10.4f} {sig:>6s}")

    print("\n显著性水平: *** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著")

    # ═══════════════════════════════════
    # 保存结果到 JSON
    # ═══════════════════════════════════
    results = {
        "kfold_metrics": {
            MODEL_NAMES[m]: {
                "accuracy_mean": float(np.mean(all_fold_metrics[m]["acc"])),
                "accuracy_std": float(np.std(all_fold_metrics[m]["acc"])),
                "f1_mean": float(np.mean(all_fold_metrics[m]["f1"])),
                "f1_std": float(np.std(all_fold_metrics[m]["f1"])),
                "per_fold_f1": [float(x) for x in all_fold_metrics[m]["f1"]],
                "per_fold_acc": [float(x) for x in all_fold_metrics[m]["acc"]],
            }
            for m in available
        },
        "bootstrap_ci": {
            MODEL_NAMES[m]: {
                "f1": float(macro_f1(*merged[m])),
                "ci_lower": float(bootstrap_ci(*merged[m], macro_f1)[0]),
                "ci_upper": float(bootstrap_ci(*merged[m], macro_f1)[1]),
            }
            for m in available
        },
        "mcnemar": [
            {
                "model_a": MODEL_NAMES[m_a],
                "model_b": MODEL_NAMES[m_b],
                "p_value": p_all,
                "significant": sig,
                "per_fold_p": [float(p) for p in fold_pvals],
            }
            for m_a, m_b, p_all, sig, fold_pvals in mcnemar_results
        ],
    }

    out_path = OUT_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
