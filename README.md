# 基于深度学习的 Word-in-Context 词义消歧研究

## 1. 研究概述

本项目基于 SemCor 语料库构建 WIC（Word-in-Context）数据集，训练并对比多种深度学习模型在词义消歧二分类任务上的表现。核心研究问题是：**给定同一个词在两个不同句子中的出现，判断它们是否表达相同的含义**。

在模型对比之外，本项目重点利用微调 BERT 的上下文向量（contextualized embedding）进行深入的可解释性分析，通过统计检验和可视化揭示模型如何在向量空间中编码词义信息。

### 任务示例

| 句子 1 | 句子 2 | 目标词 | 标签 |
|--------|--------|--------|------|
| I deposited money in the **bank**. | The river **bank** was muddy. | bank | 异义（0） |
| He **broke** the window. | She **broke** the vase. | broke | 同义（1） |

### 文档导航

- [RESULTS.md](RESULTS.md) — 模型训练过程与分类结果对比
- [ANALYSIS.md](ANALYSIS.md) — 语言学维度分析与 BERT 上下文向量分析

---

## 2. 项目结构

```
CBS5502/
├── README.md                          # 本文档（项目概述）
├── RESULTS.md                         # 模型训练与分类结果
├── ANALYSIS.md                        # 语言学分析与向量分析
├── requirements.txt                   # Python 依赖
├── data/
│   ├── glove.6B.100d.txt              # GloVe 词向量（Google Drive）
│   ├── raw/
│   │   └── wic_dataset.jsonl          # 原始 WIC 数据集（Google Drive）
│   └── split/
│       ├── train.jsonl                # 训练集
│       ├── dev.jsonl                  # 验证集
│       └── test.jsonl                 # 测试集
├── run_kfold.sh                       # 一键运行 5 折交叉验证
├── src/
│   ├── utils.py                       # 共用工具（路径、数据加载、评估、k-fold 划分）
│   ├── data_clean.py                  # 数据清洗与划分
│   ├── model_bilstm.py               # BiLSTM 模型
│   ├── model_bert.py                  # BERT 微调模型
│   ├── model_bert_frozen.py           # BERT 冻结 + MLP 模型
│   ├── model_roberta.py               # RoBERTa 微调模型
│   ├── model_deberta.py               # DeBERTa-v3-base 微调模型
│   ├── model_sbert.py                 # Sentence-BERT 余弦相似度模型
│   ├── evaluate.py                    # 统一评估 + 语言学分析 + 错误分析
│   ├── statistical_tests.py           # Bootstrap CI + McNemar 统计检验
│   └── analyze_bert_embeddings.py     # BERT 上下文向量分析与可视化
├── models/                            # 训练好的模型权重（Google Drive）
│   ├── bert.pt / bert_fold{0-4}.pt
│   ├── bert_frozen_mlp.pt / ..._fold{0-4}.pt
│   ├── bilstm.pt / ..._fold{0-4}.pt
│   ├── roberta.pt / ..._fold{0-4}.pt
│   ├── deberta.pt / ..._fold{0-4}.pt
│   └── sbert_threshold.json
├── plots/                             # BERT 向量分析图表
│   ├── README.md                      # 图表详细解读
│   ├── cosine_distribution.png
│   ├── tsne_diff.png
│   ├── tsne_emb12.png
│   ├── cosine_by_pos.png
│   ├── activation_heatmap.png
│   └── norm_distribution.png
├── logs/                              # 训练日志
│   └── kfold/                         # 5 折交叉验证日志
├── results/                           # 评估结果
│   ├── metrics.json
│   ├── linguistic_analysis.json
│   ├── error_analysis.json
│   ├── statistical_tests.json         # 统计检验结果
│   └── predictions/                   # 各模型各折的预测结果（用于统计检验）
└── scripts/                           # 数据构建脚本
    ├── semcor_to_wic.py               # SemCor → WIC 转换
    └── data_prepare.py                # 数据准备
```

---

## 3. 数据集

### 3.1 数据来源与构建

- **来源：** SemCor 3.0（Brown Corpus 子集，标注了 WordNet 义项）
- **构建方式：** 将 SemCor 转换为 WIC 格式，每条样本包含一个目标词、两个句子及是否同义的标签
  - **正例：** 同一 synset（义项）下取两个不同句子（同义）
  - **负例：** 同一 lemma（词元）、不同 synset 下各取一句（异义）

### 3.2 数据清洗

清洗脚本 `src/data_clean.py` 执行以下过滤规则：

1. **去掉负例中跨词性的样本**（pos1 ≠ pos2）—— 仅靠词性即可判断，无需语义消歧
2. **去掉正例中两句完全相同的样本** —— 无消歧意义
3. **去掉目标词索引与 surface form 不匹配的样本** —— 数据对齐错误
4. **去掉重复样本** —— 去重
5. **去掉极短句样本**（任一句子 < 5 词）—— 上下文不足
6. **去掉超过 256 token 的样本** —— 与统一 MAX_LEN=256 一致，避免截断导致目标词丢失

### 3.3 数据规模与划分

| 集合 | 样本数 | 正例 | 负例 | 正例比 |
|------|--------|------|------|--------|
| 训练集 | 35,547 | 13,083 | 22,464 | 36.8% |
| 验证集 | 7,746 | 3,013 | 4,733 | 38.9% |
| 测试集 | 7,476 | 2,907 | 4,569 | 38.9% |

- **划分策略：** 按 lemma 分组划分（70/15/15），同一 lemma 的所有样本只出现在一个集合中，防止数据泄露
- **5 折交叉验证：** 全部数据按 lemma 均匀分为 5 组，轮流用 1 组做测试、1 组做验证、3 组做训练，5 折测试集互不重叠，覆盖全部样本。用于报告 mean±std 指标并进行统计显著性检验
- **类别不平衡处理：** 训练时使用加权损失函数（`CrossEntropyLoss(weight=...)`），权重按正负例比例反比设定

---

## 4. 模型

### 4.1 Baseline

| 模型 | 策略 |
|------|------|
| Majority | 全部预测为多数类（负例） |
| Random | 按训练集类别比例随机猜测 |

### 4.2 BiLSTM（`src/model_bilstm.py`）

- **输入：** 使用 GloVe 100d 静态词向量，两句分别过双向 LSTM
- **分类：** 取两句中目标词位置的 BiLSTM 隐状态拼接，经全连接层分类
- **特点：** 使用静态词向量，无法区分多义词的不同含义；通过 LSTM 的上下文建模部分弥补

**关键超参数：** hidden_size=128, dropout=0.3, lr=1e-3, epochs=8, batch=32, MAX_LEN=256, patience=3

### 4.3 BERT Fine-tune（`src/model_bert.py`）

- **输入：** 两句拼接为 `[CLS] sentence1 [SEP] sentence2 [SEP]`，经 BERT 编码
- **分类：** 提取三个向量的拼接 `[CLS_emb; target_word1_emb; target_word2_emb]`（2304 维），经线性层分类
- **特点：** 全参数微调，BERT 的 self-attention 让每个 token 都能看到两句的全部信息

**关键超参数：** model=bert-base-uncased, MAX_LEN=256, lr=3e-5, epochs=8, batch=32, warmup=10%, AMP(FP16), patience=3

### 4.4 BERT Frozen + MLP（`src/model_bert_frozen.py`）

- **输入：** 与 BERT Fine-tune 相同
- **分类：** 冻结 BERT 参数不更新，提取目标词 embedding 后训练一个独立的 MLP 分类器
- **特点：** BERT 只作为特征提取器，embedding 是通用预训练表示，未针对 WIC 任务优化
- **用途：** 与 Fine-tune 对比，验证微调对 embedding 质量的提升

**MLP 结构：** Linear(2304, 512) → ReLU → Dropout(0.3) → Linear(512, 256) → ReLU → Dropout(0.2) → Linear(256, 2)

### 4.5 RoBERTa Fine-tune（`src/model_roberta.py`）

- **输入：** 两句拼接为 `<s> sentence1 </s></s> sentence2 </s>`，经 RoBERTa 编码
- **分类：** 与 BERT 相同的 `[CLS + target1 + target2]` 拼接策略
- **特点：** RoBERTa 使用更大预训练数据、动态 masking、去掉 NSP 任务，通常优于 BERT

**关键超参数：** model=roberta-base, MAX_LEN=256, lr=2e-5, epochs=8, batch=32, warmup=10%, AMP(FP16), patience=3

### 4.6 DeBERTa-v3-base Fine-tune（`src/model_deberta.py`）

- **输入：** 两句拼接经 DeBERTa-v3-base 编码（disentangled attention 机制）
- **分类：** 提取五路特征拼接 `[CLS; t1; t2; t1-t2; t1*t2]`（3840 维），经两层 MLP 分类。目标词使用 subword 平均池化（而非仅取第一个 subword）
- **特点：** DeBERTa-v3 的解耦注意力分别建模内容和位置信息，理论上更适合需要精确位置信息的 WIC 任务；使用 BF16 混合精度（DeBERTa-v3 不兼容 FP16）

**关键超参数：** model=microsoft/deberta-v3-base, MAX_LEN=256, lr=2e-5, epochs=10, batch=32, warmup=10%, BF16, patience=5

### 4.7 Sentence-BERT（`src/model_sbert.py`）

- **输入：** 两句分别独立编码（不拼接），取 mean pooling 的句向量
- **分类：** 计算两句向量的余弦相似度，在验证集上搜索最优阈值进行分类
- **特点：** 不需要微调，利用预训练 Sentence-BERT 的语义表示；但句向量是整句级别的，无法精确捕捉目标词级别的语义差异

**使用模型：** sentence-transformers/all-MiniLM-L6-v2

---

## 5. 快速开始

### 5.1 环境配置

```bash
# 克隆仓库
git clone https://github.com/yurihuruitao/CBS5502.git
cd CBS5502

# 安装依赖（需要 Python 3.8+，建议 CUDA 环境）
pip install -r requirements.txt
```

### 5.2 下载大文件

模型权重和语料库文件存储在 Google Drive，需手动下载或使用 gdown：

| 文件 | 放置路径 | Google Drive |
|------|---------|--------------|
| 模型权重（`*.pt`） | `models/` | [下载](https://drive.google.com/open?id=1CkFDaNVM5LDHtWulIc588xBvtSTpjGjM) |
| 语料库（GloVe + 原始数据集） | `data/` | [下载](https://drive.google.com/open?id=17Wzb4FGMvM5Gr9od9b0u7mJA0wjbY7wH) |

下载后将文件放到对应位置：

```
models/
├── bert.pt
├── bert_frozen_mlp.pt
├── bilstm.pt
├── roberta.pt
└── deberta.pt

data/
├── glove.6B.100d.txt
└── raw/
    └── wic_dataset.jsonl
```

或使用 gdown 批量下载：

```bash
# 下载模型权重
gdown --folder "https://drive.google.com/open?id=1CkFDaNVM5LDHtWulIc588xBvtSTpjGjM" -O models/

# 下载语料库
gdown --folder "https://drive.google.com/open?id=17Wzb4FGMvM5Gr9od9b0u7mJA0wjbY7wH" -O data/raw/
mv data/raw/glove.6B.100d.txt data/
```

### 5.3 运行

所有脚本从 `src/` 目录运行：

```bash
cd src/

# 1. 数据清洗与划分（生成 data/split/）
python data_clean.py

# 2a. 训练单个模型（使用默认 train/dev/test 划分）
python model_bilstm.py
python model_bert.py
python model_bert_frozen.py
python model_roberta.py
python model_deberta.py
python model_sbert.py

# 2b. 5 折交叉验证（推荐，从项目根目录运行）
cd ..
bash run_kfold.sh              # 全部模型 × 5 折
bash run_kfold.sh bert         # 单个模型 × 5 折

# 3. 统计检验（需要先完成 5 折训练）
cd src/
python statistical_tests.py    # Bootstrap CI + McNemar 配对检验

# 4. 统一评估所有模型
python evaluate.py

# 5. BERT 上下文向量分析
python analyze_bert_embeddings.py

# 6. 官方 WiC 基准测试
python eval_official_wic.py
```

> **注意：** 如果只想运行评估（跳过训练），只需下载模型权重，然后直接执行步骤 4-6。5 折交叉验证需要重新训练所有模型。

---

## 6. 统计验证

为确保实验结论的统计可靠性，本项目采用以下方法：

- **5 折交叉验证：** 按 lemma 分组将全部数据均匀分为 5 折，每折轮流作为测试集（互不重叠），报告 5 折 Macro-F1 的 mean ± std
- **Bootstrap 95% 置信区间：** 合并 5 折预测结果，进行 1000 次有放回抽样计算 Macro-F1 的 95% CI
- **McNemar 配对检验：** 对所有模型两两比较，基于逐样本对错构建 2×2 列联表，使用 Edwards 连续性校正的 McNemar 检验判断模型差异是否统计显著（p < 0.05）

统计检验脚本：`src/statistical_tests.py`，结果输出至 `results/statistical_tests.json`。

---

## 7. 参考文献

- Pilehvar, M.T. & Camacho-Collados, J. (2019). *WiC: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations.* NAACL-HLT.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv.
- He, P. et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* ICLR.
- He, P. et al. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* ICLR.
- Miller, G.A. et al. (1993). *A Semantic Concordance.* HLT.
