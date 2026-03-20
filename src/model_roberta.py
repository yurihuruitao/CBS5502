"""
RoBERTa fine-tune — 使用 <s> + 目标词上下文表示
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score as sklearn_f1
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import load_split, evaluate, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════
# 超参数（在这里调）
# ══════════════════════════════════
MODEL_NAME = "roberta-base"
MAX_LEN = 256
EPOCHS = 8
LR = 2e-5
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 0
MAX_GRAD_NORM = 1.0
PATIENCE = 3
NUM_WORKERS = 4
USE_AMP = True
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

        target_pos1 = 0
        for i, (si, wi) in enumerate(zip(seq_ids, word_ids)):
            if si == 0 and wi == s["index1"]:
                target_pos1 = i
                break

        target_pos2 = 0
        for i, (si, wi) in enumerate(zip(seq_ids, word_ids)):
            if si == 1 and wi == s["index2"]:
                target_pos2 = i
                break

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "target_pos1": target_pos1,
            "target_pos2": target_pos2,
            "label": int(s["label"]),
        }


class RoBERTaWICClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        if FREEZE_LAYERS > 0:
            for layer in self.encoder.encoder.layer[:FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden = self.encoder.config.hidden_size
        # <s> + target_word1 + target_word2 → 分类
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden * 3, 2),
        )

    def forward(self, input_ids, attention_mask, target_pos1, target_pos2):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        cls = hidden_states[:, 0]
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        t1 = hidden_states[batch_idx, target_pos1]
        t2 = hidden_states[batch_idx, target_pos2]

        return self.fc(torch.cat([cls, t1, t2], dim=-1))


def predict_dl(model, dl):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            target_pos1 = batch["target_pos1"].to(DEVICE, non_blocking=True)
            target_pos2 = batch["target_pos2"].to(DEVICE, non_blocking=True)
            preds = model(input_ids, attention_mask,
                          target_pos1, target_pos2).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    return all_labels, all_preds


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"超参数: model={MODEL_NAME}, lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, freeze={FREEZE_LAYERS}, patience={PATIENCE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = WICDataset(load_split("train"), tokenizer)
    dev_ds = WICDataset(load_split("dev"), tokenizer)
    test_ds = WICDataset(load_split("test"), tokenizer)
    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, **dl_kwargs)
    dev_dl = DataLoader(dev_ds, BATCH_SIZE, **dl_kwargs)
    test_dl = DataLoader(test_ds, BATCH_SIZE, **dl_kwargs)

    model = RoBERTaWICClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_dl) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    labels_all = [int(s["label"]) for s in train_ds.samples]
    n_pos, n_neg = sum(labels_all), len(labels_all) - sum(labels_all)
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1 = 0
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            target_pos1 = batch["target_pos1"].to(DEVICE, non_blocking=True)
            target_pos2 = batch["target_pos2"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(input_ids, attention_mask,
                               target_pos1, target_pos2)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        y_true, y_pred = predict_dl(model, dev_dl)
        f1 = sklearn_f1(y_true, y_pred, average="macro")
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}  "
              f"dev macro-F1={f1:.4f}  lr={lr_now:.2e}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "roberta.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n最优 dev macro-F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(MODEL_DIR / "roberta.pt", map_location=DEVICE))
    y_true, y_pred = predict_dl(model, test_dl)
    evaluate(y_true, y_pred, "RoBERTa")
