"""
DeBERTa-v3-base fine-tune — 使用 [CLS] + 目标词上下文表示（subword 平均池化 + 交互特征）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score as sklearn_f1
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import load_split, evaluate, MODEL_DIR, set_seed, save_predictions, load_kfold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════
# 超参数（在这里调）
# ══════════════════════════════════
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
EPOCHS = 10
LR = 2e-5
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 0
MAX_GRAD_NORM = 1.0
PATIENCE = 5
NUM_WORKERS = 4
USE_BF16 = True              # DeBERTa-v3 不兼容 FP16，改用 BF16
# ══════════════════════════════════


class WICDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["sentence1"], s["sentence2"],
            truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )

        word_ids = enc.word_ids()
        seq_ids = enc.sequence_ids()

        # 收集 sentence1 中目标词的所有 subword token 位置
        positions1 = []
        for i, (si, wi) in enumerate(zip(seq_ids, word_ids)):
            if si == 0 and wi == s["index1"]:
                positions1.append(i)

        # 收集 sentence2 中目标词的所有 subword token 位置
        positions2 = []
        for i, (si, wi) in enumerate(zip(seq_ids, word_ids)):
            if si == 1 and wi == s["index2"]:
                positions2.append(i)

        # 构建 mask 向量用于平均池化
        seq_len = MAX_LEN
        mask1 = torch.zeros(seq_len)
        mask2 = torch.zeros(seq_len)
        for p in positions1:
            mask1[p] = 1.0
        for p in positions2:
            mask2[p] = 1.0

        # fallback: 如果没找到目标词，用 [CLS] 位置
        if mask1.sum() == 0:
            mask1[0] = 1.0
        if mask2.sum() == 0:
            mask2[0] = 1.0

        result = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "target_mask1": mask1,
            "target_mask2": mask2,
            "label": int(s["label"]),
        }
        # DeBERTa-v3 不使用 token_type_ids
        if "token_type_ids" in enc:
            result["token_type_ids"] = enc["token_type_ids"].squeeze()
        return result


class DeBERTaWICClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        if FREEZE_LAYERS > 0:
            for layer in self.encoder.encoder.layer[:FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden = self.encoder.config.hidden_size
        # [CLS; t1; t2; t1-t2; t1*t2] → 两层分类头
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden * 5, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, target_mask1, target_mask2,
                token_type_ids=None):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state  # (B, L, H)

        cls = hidden_states[:, 0]  # (B, H)

        # subword 平均池化
        m1 = target_mask1.unsqueeze(-1)  # (B, L, 1)
        t1 = (hidden_states * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1)  # (B, H)

        m2 = target_mask2.unsqueeze(-1)
        t2 = (hidden_states * m2).sum(dim=1) / m2.sum(dim=1).clamp(min=1)

        # 交互特征
        diff = t1 - t2
        prod = t1 * t2

        features = torch.cat([cls, t1, t2, diff, prod], dim=-1)
        return self.fc(features)


def predict_dl(model, dl):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            target_mask1 = batch["target_mask1"].to(DEVICE, non_blocking=True)
            target_mask2 = batch["target_mask2"].to(DEVICE, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE, non_blocking=True)
            labels = batch["label"]

            preds = model(input_ids, attention_mask, target_mask1, target_mask2,
                          token_type_ids=token_type_ids).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


if __name__ == "__main__":
    import sys, logging, math, os, argparse
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    fold = args.fold
    set_seed(42)

    # 抑制第三方库日志，只保留自己的输出
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "urllib3", "transformers", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    log = logging.getLogger("deberta_train")
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.propagate = False

    log.info(f"Device: {DEVICE}  Fold: {fold}")
    log.info(f"超参数: model={MODEL_NAME}, lr={LR}, epochs={EPOCHS}, "
             f"batch={BATCH_SIZE}, freeze={FREEZE_LAYERS}, patience={PATIENCE}, "
             f"warmup={WARMUP_RATIO}, weight_decay={WEIGHT_DECAY}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if fold is not None:
        train_samples, dev_samples, test_samples = load_kfold(fold)
    else:
        train_samples, dev_samples, test_samples = load_split("train"), load_split("dev"), load_split("test")
    train_ds = WICDataset(train_samples, tokenizer)
    dev_ds = WICDataset(dev_samples, tokenizer)
    test_ds = WICDataset(test_samples, tokenizer)
    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, **dl_kwargs)
    dev_dl = DataLoader(dev_ds, BATCH_SIZE, **dl_kwargs)
    test_dl = DataLoader(test_ds, BATCH_SIZE, **dl_kwargs)

    log.info(f"数据: train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}")

    model = DeBERTaWICClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + linear decay scheduler
    total_steps = len(train_dl) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 加权损失（缓和补偿）
    labels_all = [int(s["label"]) for s in train_ds.samples]
    n_pos, n_neg = sum(labels_all), len(labels_all) - sum(labels_all)
    weight = torch.FloatTensor([1.0, math.sqrt(n_neg / max(n_pos, 1))]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    log.info(f"类别权重: neg={weight[0]:.4f}, pos={weight[1]:.4f}")

    best_f1 = 0
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}",
                          leave=False, file=sys.stderr):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            target_mask1 = batch["target_mask1"].to(DEVICE, non_blocking=True)
            target_mask2 = batch["target_mask2"].to(DEVICE, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(input_ids, attention_mask, target_mask1, target_mask2,
                               token_type_ids=token_type_ids)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        y_true, y_pred = predict_dl(model, dev_dl)
        f1 = sklearn_f1(y_true, y_pred, average="macro")
        lr_now = scheduler.get_last_lr()[0]
        log.info(f"Epoch {epoch+1}/{EPOCHS}  train_loss={total_loss/len(train_dl):.4f}  "
                 f"dev_macro-F1={f1:.4f}  lr={lr_now:.2e}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            suffix = f"_fold{fold}" if fold is not None else ""
            torch.save(model.state_dict(), MODEL_DIR / f"deberta{suffix}.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    log.info(f"最优 dev macro-F1: {best_f1:.4f}")

    # 加载最优模型评估 test
    suffix = f"_fold{fold}" if fold is not None else ""
    model.load_state_dict(torch.load(MODEL_DIR / f"deberta{suffix}.pt", map_location=DEVICE))
    y_true, y_pred = predict_dl(model, test_dl)
    evaluate(y_true, y_pred, f"DeBERTa (fold={fold})")
    if fold is not None:
        save_predictions("deberta", fold, y_true, y_pred)
