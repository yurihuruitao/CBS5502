"""
BiLSTM
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score as sklearn_f1
from tqdm import tqdm
from utils import load_split, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent.parent
GLOVE_PATH = ROOT_DIR / "data" / "glove.6B.100d.txt"

# ══════════════════════════════════
# 超参数（在这里调）
# ══════════════════════════════════
HIDDEN_SIZE = 128       # LSTM 隐藏层维度
DROPOUT = 0.3           # Dropout 比例
FC_DIM = 64             # 全连接层维度
EPOCHS = 8
LR = 1e-3               # 学习率
PATIENCE = 3             # early stopping 耐心值
BATCH_SIZE = 32
MAX_LEN = 256           # 句子最大 token 数
FREEZE_EMB = False      # 是否冻结词向量
# ══════════════════════════════════


def load_glove(path=GLOVE_PATH, dim=100):
    word2idx = {"<pad>": 0, "<unk>": 1}
    vectors = [np.zeros(dim), np.random.normal(0, 0.1, dim)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word2idx[parts[0]] = len(word2idx)
            vectors.append(np.array(parts[1:], dtype=np.float32))
    return word2idx, np.array(vectors, dtype=np.float32)


class WICDataset(Dataset):
    def __init__(self, samples, word2idx):
        self.samples = samples
        self.word2idx = word2idx

    def __len__(self):
        return len(self.samples)

    def _encode(self, sentence):
        tokens = sentence.lower().split()[:MAX_LEN]
        return [self.word2idx.get(t, 1) for t in tokens]

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids1 = self._encode(s["sentence1"])
        ids2 = self._encode(s["sentence2"])
        return {
            "ids1": ids1, "ids2": ids2,
            "idx1": min(s["index1"], len(ids1) - 1),
            "idx2": min(s["index2"], len(ids2) - 1),
            "label": int(s["label"]),
        }


def collate_fn(batch):
    max1 = max(len(b["ids1"]) for b in batch)
    max2 = max(len(b["ids2"]) for b in batch)
    ids1 = torch.zeros(len(batch), max1, dtype=torch.long)
    ids2 = torch.zeros(len(batch), max2, dtype=torch.long)
    idx1, idx2, labels = [], [], []
    for i, b in enumerate(batch):
        ids1[i, :len(b["ids1"])] = torch.tensor(b["ids1"])
        ids2[i, :len(b["ids2"])] = torch.tensor(b["ids2"])
        idx1.append(b["idx1"])
        idx2.append(b["idx2"])
        labels.append(b["label"])
    return ids1, ids2, torch.tensor(idx1), torch.tensor(idx2), torch.tensor(labels)


class BiLSTMClassifier(nn.Module):
    def __init__(self, pretrained_emb):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb.shape
        self.emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_emb), freeze=FREEZE_EMB
        )
        self.lstm = nn.LSTM(emb_dim, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 4, FC_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_DIM, 2),
        )

    def forward(self, ids1, ids2, idx1, idx2):
        out1, _ = self.lstm(self.emb(ids1))
        out2, _ = self.lstm(self.emb(ids2))
        h1 = out1[torch.arange(out1.size(0)), idx1]
        h2 = out2[torch.arange(out2.size(0)), idx2]
        return self.fc(torch.cat([h1, h2], dim=-1))


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"超参数: hidden={HIDDEN_SIZE}, dropout={DROPOUT}, fc={FC_DIM}, "
          f"lr={LR}, epochs={EPOCHS}, batch={BATCH_SIZE}")

    word2idx, emb_matrix = load_glove()
    train_ds = WICDataset(load_split("train"), word2idx)
    dev_ds = WICDataset(load_split("dev"), word2idx)
    test_ds = WICDataset(load_split("test"), word2idx)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, BATCH_SIZE, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, BATCH_SIZE, collate_fn=collate_fn)

    model = BiLSTMClassifier(emb_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 加权损失
    labels_all = [int(s["label"]) for s in train_ds.samples]
    n_pos, n_neg = sum(labels_all), len(labels_all) - sum(labels_all)
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    def predict_dl(dl):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for ids1, ids2, idx1, idx2, labels in dl:
                ids1, ids2, idx1, idx2, labels = [x.to(DEVICE) for x in [ids1, ids2, idx1, idx2, labels]]
                preds = model(ids1, ids2, idx1, idx2).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        return all_labels, all_preds

    best_f1 = 0
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for ids1, ids2, idx1, idx2, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            ids1, ids2, idx1, idx2, labels = [x.to(DEVICE) for x in [ids1, ids2, idx1, idx2, labels]]
            logits = model(ids1, ids2, idx1, idx2)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        y_true, y_pred = predict_dl(dev_dl)
        f1 = sklearn_f1(y_true, y_pred, average="macro")
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}  dev macro-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "bilstm.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n最优 dev macro-F1: {best_f1:.4f}")

    # 加载最优模型评估 test
    model.load_state_dict(torch.load(MODEL_DIR / "bilstm.pt", map_location=DEVICE))
    y_true, y_pred = predict_dl(test_dl)
    from utils import evaluate
    evaluate(y_true, y_pred, "BiLSTM")
