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
├── data/
│   ├── raw/
│   │   └── wic_dataset.jsonl          # 原始 WIC 数据集（SemCor 转换而来）
│   └── split/
│       ├── train.jsonl                # 训练集
│       ├── dev.jsonl                  # 验证集
│       └── test.jsonl                 # 测试集
├── src/
│   ├── utils.py                       # 共用工具（路径、数据加载、评估函数）
│   ├── data_clean.py                  # 数据清洗与划分
│   ├── model_bilstm.py               # BiLSTM 模型
│   ├── model_bert.py                  # BERT 微调模型
│   ├── model_bert_frozen.py           # BERT 冻结 + MLP 模型
│   ├── model_roberta.py               # RoBERTa 微调模型
│   ├── model_sbert.py                 # Sentence-BERT 余弦相似度模型
│   ├── evaluate.py                    # 统一评估 + 语言学分析 + 错误分析
│   └── analyze_bert_embeddings.py     # BERT 上下文向量分析与可视化
├── models/                            # 训练好的模型权重
│   ├── bert.pt
│   ├── bert_frozen_mlp.pt
│   ├── bilstm.pt
│   ├── roberta.pt
│   └── sbert_threshold.json
├── plots/                             # BERT 向量分析图表
│   ├── README.md                      # 图表详细解读
│   ├── cosine_distribution.png
│   ├── tsne_diff.png
│   ├── tsne_emb12.png
│   ├── cosine_by_pos.png
│   ├── activation_heatmap.png
│   └── norm_distribution.png
├── results/                           # 评估结果
│   ├── metrics.json
│   ├── linguistic_analysis.json
│   └── error_analysis.json
├── scripts/                           # 数据构建脚本
│   ├── semcor_to_wic.py               # SemCor → WIC 转换
│   └── data_prepare.py                # 数据准备
└── nltk_data/                         # NLTK 资源（SemCor 语料库）
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

### 4.6 Sentence-BERT（`src/model_sbert.py`）

- **输入：** 两句分别独立编码（不拼接），取 mean pooling 的句向量
- **分类：** 计算两句向量的余弦相似度，在验证集上搜索最优阈值进行分类
- **特点：** 不需要微调，利用预训练 Sentence-BERT 的语义表示；但句向量是整句级别的，无法精确捕捉目标词级别的语义差异

**使用模型：** sentence-transformers/all-MiniLM-L6-v2

---

## 5. 使用方法

所有脚本从 `src/` 目录运行：

```bash
cd src/

# 1. 数据清洗与划分
python data_clean.py

# 2. 训练模型
python model_bilstm.py
python model_bert.py
python model_bert_frozen.py
python model_roberta.py
python model_sbert.py

# 3. 统一评估所有模型
python evaluate.py

# 4. BERT 上下文向量分析
python analyze_bert_embeddings.py

# 5. 官方 WiC 基准测试
python eval_official_wic.py
```

### 环境依赖

- Python 3.8+
- PyTorch 2.0+（CUDA 支持）
- transformers
- scikit-learn
- matplotlib
- scipy（统计检验）
- nltk（SemCor 语料库）

---

## 6. 参考文献

- Pilehvar, M.T. & Camacho-Collados, J. (2019). *WiC: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations.* NAACL-HLT.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv.
- Miller, G.A. et al. (1993). *A Semantic Concordance.* HLT.
