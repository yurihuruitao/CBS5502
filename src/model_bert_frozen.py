"""
BERT Feature Extraction — 冻结 BERT，提取目标词 contextualized embedding，训练 MLP 分类
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score as sklearn_f1
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
from utils import load_split, evaluate, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════
# 超参数
# ══════════════════════════════════
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
EPOCHS = 8
LR = 1e-3
BATCH_SIZE = 32
PATIENCE = 3
# ══════════════════════════════════


@torch.no_grad()
def extract_embeddings(samples, tokenizer, bert, batch_size=64):
    """用冻结的 BERT 提取每个样本的 [CLS] + 目标词1 + 目标词2 embedding。"""
    bert.eval()
    all_feats, all_labels = [], []

    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting embeddings"):
        batch = samples[i:i + batch_size]
        encs = tokenizer(
            [s["sentence1"] for s in batch],
            [s["sentence2"] for s in batch],
            truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt",
        ).to(DEVICE)

        hidden = bert(**encs).last_hidden_state  # (B, seq_len, 768)

        for j, s in enumerate(batch):
            word_ids = tokenizer(
                s["sentence1"], s["sentence2"],
                truncation=True, max_length=MAX_LEN,
            ).word_ids()
            seq_ids = tokenizer(
                s["sentence1"], s["sentence2"],
                truncation=True, max_length=MAX_LEN,
            ).sequence_ids()

            # 找目标词位置
            pos1, pos2 = 0, 0
            for k, (si, wi) in enumerate(zip(seq_ids, word_ids)):
                if si == 0 and wi == s["index1"]:
                    pos1 = k
                    break
            for k, (si, wi) in enumerate(zip(seq_ids, word_ids)):
                if si == 1 and wi == s["index2"]:
                    pos2 = k
                    break

            cls = hidden[j, 0]
            t1 = hidden[j, pos1]
            t2 = hidden[j, pos2]
            all_feats.append(torch.cat([cls, t1, t2]).cpu())
            all_labels.append(int(s["label"]))


    return torch.stack(all_feats), torch.tensor(all_labels)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ── 1. 加载 BERT（冻结） ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    # ── 2. 提取 embedding ──
    print("提取 train embeddings...")
    train_X, train_y = extract_embeddings(load_split("train"), tokenizer, bert)
    print("提取 dev embeddings...")
    dev_X, dev_y = extract_embeddings(load_split("dev"), tokenizer, bert)
    print("提取 test embeddings...")
    test_X, test_y = extract_embeddings(load_split("test"), tokenizer, bert)

    # 释放 BERT 显存
    del bert
    torch.cuda.empty_cache()

    # ── 3. 训练 MLP ──
    train_dl = DataLoader(TensorDataset(train_X, train_y), BATCH_SIZE, shuffle=True)
    dev_dl = DataLoader(TensorDataset(dev_X, dev_y), BATCH_SIZE)
    test_dl = DataLoader(TensorDataset(test_X, test_y), BATCH_SIZE)

    hidden_size = train_X.shape[1]  # 768 * 3
    model = MLP(hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 加权损失
    n_pos, n_neg = train_y.sum().item(), (train_y == 0).sum().item()
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    print(f"\nMLP 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练 MLP (input_dim={hidden_size}, epochs={EPOCHS}, lr={LR})\n")

    best_f1 = 0
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # dev 评估
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in dev_dl:
                preds.extend(model(X_batch.to(DEVICE)).argmax(1).cpu().tolist())
                labels.extend(y_batch.tolist())
        f1 = sklearn_f1(labels, preds, average="macro")
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}  dev macro-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "bert_frozen_mlp.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n最优 dev macro-F1: {best_f1:.4f}")

    # ── 4. Test 评估 ──
    model.load_state_dict(torch.load(MODEL_DIR / "bert_frozen_mlp.pt", map_location=DEVICE))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            preds.extend(model(X_batch.to(DEVICE)).argmax(1).cpu().tolist())
            labels.extend(y_batch.tolist())
    evaluate(labels, preds, "BERT-Frozen + MLP")
