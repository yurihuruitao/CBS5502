# 实验方法论

本文档详细描述本项目的实验设计、数据处理流程、模型架构、训练策略和统计验证方法。

---

## 1. 任务定义

**Word-in-Context (WIC)** 是一个词义消歧二分类任务：给定目标词 *w* 在两个不同句子 *s₁*、*s₂* 中的出现，判断 *w* 在两句中是否表达相同含义。

- **输入：** 目标词 *w*、句子 *s₁*、句子 *s₂*、*w* 在 *s₁* 中的位置索引 *i₁*、*w* 在 *s₂* 中的位置索引 *i₂*
- **输出：** 二分类标签（1 = 同义，0 = 异义）
- **评估指标：** Macro-F1（主要指标）、Accuracy、Precision、Recall

选用 Macro-F1 而非 Accuracy 作为主要指标，因为数据集正负样本不平衡（正例约 37%），Accuracy 倾向于偏好多数类预测。

---

## 2. 数据集构建

### 2.1 数据来源

数据来自 **SemCor 3.0**（Miller et al., 1993），这是 Brown Corpus 的一个子集，其中的词汇标注了 WordNet 义项（synset）。SemCor 包含约 226,000 个标注词例，覆盖多种词性。

### 2.2 WIC 格式转换

使用 `scripts/semcor_to_wic.py` 将 SemCor 转换为 WIC 格式：

1. **解析 SemCor**：遍历所有标注句子，提取每个带义项标注的词的 (lemma, POS, synset, 句子文本, 词位置索引)
2. **按 lemma 分组**：将同一 lemma 的所有出现按 synset 分组
3. **构造正例**：同一 synset（义项）下随机取两个不同句子中的出现，标签为「同义」
4. **构造负例**：同一 lemma、不同 synset 下各取一个出现，标签为「异义」

每条样本的 JSONL 字段：

```json
{
  "word": "bank",
  "sentence1": "I deposited money in the bank.",
  "sentence2": "The river bank was muddy.",
  "index1": 5,
  "index2": 2,
  "surface1": "bank",
  "surface2": "bank",
  "sense1": "bank.n.01",
  "sense2": "bank.n.02",
  "pos1": "noun",
  "pos2": "noun",
  "label": 0
}
```

### 2.3 数据清洗

清洗脚本 `src/data_clean.py` 执行 6 项过滤规则，消除对模型不公平或无意义的样本：

| 规则 | 描述 | 理由 |
|------|------|------|
| 1 | 去掉负例中跨词性样本（pos1 ≠ pos2） | 仅靠词性即可判断异义，无需语义消歧 |
| 2 | 去掉正例中两句完全相同的样本 | 无消歧意义，模型无需理解语义即可判对 |
| 3 | 去掉目标词索引与 surface form 不匹配的样本 | 数据对齐错误，索引指向错误位置 |
| 4 | 去掉重复样本（同句对 + 同词元） | 避免数据泄露和评估偏差 |
| 5 | 去掉极短句样本（任一句子 < 5 词） | 上下文不足，不能有效考查语义理解 |
| 6 | 去掉超过 256 token 的样本 | 与统一 MAX_LEN=256 一致，避免截断导致目标词丢失 |

规则 6 使用 `bert-base-uncased` tokenizer 进行 token 化检查。

### 2.4 数据划分

- **策略：** 按 lemma 分组划分（70% train / 15% dev / 15% test），同一 lemma 的所有样本只出现在一个集合中，**防止数据泄露**
- **重要性：** 如果同一 lemma 出现在训练集和测试集中，模型可能记忆了该词的义项模式而非学会泛化的语义理解

最终数据规模：

| 集合 | 样本数 | 正例 | 负例 | 正例比 |
|------|--------|------|------|--------|
| 训练集 | 35,547 | 13,083 | 22,464 | 36.8% |
| 验证集 | 7,746 | 3,013 | 4,733 | 38.9% |
| 测试集 | 7,476 | 2,907 | 4,569 | 38.9% |
| **合计** | **50,769** | **19,003** | **31,766** | **37.4%** |

---

## 3. 模型架构

### 3.1 基线模型

| 基线 | 策略 | 用途 |
|------|------|------|
| Random | 按训练集正负比例随机预测 | 最低合理基线 |
| Majority | 全部预测为多数类（负例） | 展示类别不平衡的影响 |

### 3.2 BiLSTM

- **词向量：** GloVe 6B 100d 预训练静态词向量（400K 词表），训练中允许微调
- **编码器：** 单层双向 LSTM（hidden_size=128），两句分别独立编码
- **目标词表示：** 取 BiLSTM 在目标词位置的隐状态（256 维 = 128 × 2 方向）
- **分类头：** 将两句目标词隐状态拼接（512 维），经 `Linear(512, 64) → ReLU → Dropout(0.3) → Linear(64, 2)`
- **设计意图：** 验证静态词向量 + 上下文建模能在多大程度上进行词义消歧

```
sentence1 → GloVe → BiLSTM → hidden[target_pos1] → ─┐
                                                      ├→ concat(512d) → FC → 2
sentence2 → GloVe → BiLSTM → hidden[target_pos2] → ─┘
```

### 3.3 BERT Fine-tune

- **预训练模型：** `bert-base-uncased`（12 层，768 维，110M 参数）
- **输入格式：** `[CLS] sentence1 [SEP] sentence2 [SEP]`，两句拼接为一个序列
- **目标词定位：** 通过 `word_ids()` 和 `sequence_ids()` 精确定位 sentence1 和 sentence2 中目标词的**第一个 subword token** 位置
- **特征提取：** 从最后一层隐状态中取出三个向量：`[CLS]` 表示、target_word1 表示、target_word2 表示
- **分类头：** 三向量拼接（2304 维 = 768 × 3），经 `Dropout(0.1) → Linear(2304, 2)`
- **训练：** 全参数微调，线性学习率 warmup（前 10% 步数），FP16 混合精度

```
[CLS] s1_tokens [SEP] s2_tokens [SEP]
  ↓       ↓              ↓
  cls     t1             t2
  ↓       ↓              ↓
  concat(cls, t1, t2) → 2304d → Linear → 2
```

### 3.4 BERT Frozen + MLP

- **预训练模型：** 同上 `bert-base-uncased`，但**完全冻结**所有参数
- **特征提取：** 与 BERT Fine-tune 相同的三向量提取策略（2304 维）
- **分类器：** 独立 MLP，结构为 `Linear(2304, 512) → ReLU → Dropout(0.3) → Linear(512, 256) → ReLU → Dropout(0.2) → Linear(256, 2)`
- **参数量：** 仅 1,312,002（约 1.3M），远小于全量微调的 ~110M
- **设计意图：** 验证预训练 BERT 的通用表示（不针对 WIC 任务优化）配合 MLP 能达到怎样的效果，作为微调的消融对照

### 3.5 RoBERTa Fine-tune

- **预训练模型：** `roberta-base`（12 层，768 维，125M 参数）
- **输入格式：** `<s> sentence1 </s></s> sentence2 </s>`
- **与 BERT 的区别：** RoBERTa 使用更大预训练数据（160GB vs 16GB）、动态 masking、去掉 NSP 任务、BPE tokenizer
- **分类策略：** 与 BERT 相同的 `[CLS + target1 + target2]` 三向量拼接（2304 维）
- **其余设置：** 全参数微调，lr=2e-5（略低于 BERT），FP16 混合精度

### 3.6 DeBERTa-v3-base Fine-tune

- **预训练模型：** `microsoft/deberta-v3-base`（12 层，768 维，184M 参数）
- **核心创新：** 解耦注意力（Disentangled Attention）分别建模内容和位置信息，理论上更适合需要精确位置信息的 WIC 任务
- **目标词定位：** 收集目标词的**所有 subword token** 位置，通过 mask 平均池化得到目标词表示（而非仅取第一个 subword）

  ```
  target_emb = sum(hidden * mask) / sum(mask)
  ```

- **特征提取：** 五路拼接 `[CLS; t1; t2; t1−t2; t1×t2]`（3840 维 = 768 × 5），其中 `t1−t2` 和 `t1×t2` 分别捕捉差异和交互信息
- **分类头：** 两层 MLP `Dropout(0.1) → Linear(3840, 256) → GELU → Dropout(0.1) → Linear(256, 2)`
- **混合精度：** 使用 **BF16**（DeBERTa-v3 不兼容 FP16，FP16 会导致 NaN）
- **类别权重：** 使用 sqrt 平滑的权重（~1.31），而非线性反比

### 3.7 Sentence-BERT

- **预训练模型：** `sentence-transformers/all-MiniLM-L6-v2`（6 层，384 维）
- **编码方式：** 两句**分别独立编码**（不拼接），对最后一层隐状态做 mean pooling 得到句向量
- **分类方式：** 计算两个句向量的余弦相似度，在 dev 集上网格搜索最优阈值（范围 0.50–1.00，步长 0.01）
- **无训练：** 直接使用预训练权重，不进行任何微调
- **设计意图：** 验证整句级预训练表示（无词级位置信息）在 WIC 任务上的效果

---

## 4. 训练策略

### 4.1 共享设置

| 设置 | 值 | 说明 |
|------|-----|------|
| MAX_LEN | 256 | 经过数据清洗，无超长样本 |
| BATCH_SIZE | 32 | GPU 显存允许范围内的最大值 |
| 随机种子 | 42 | `set_seed(42)` 确保可复现 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸（仅 Transformer 模型） |

### 4.2 类别不平衡处理

训练集正例比例约 36.8%，存在类别不平衡。所有模型使用加权交叉熵损失：

```python
CrossEntropyLoss(weight=[w_neg, w_pos])
```

- BERT/RoBERTa/BiLSTM/BERT-Frozen：`w_pos = n_neg / n_pos`（线性反比）
- DeBERTa-v3：`w_pos = sqrt(n_neg / n_pos)`（sqrt 平滑，约 1.31）

### 4.3 学习率调度

Transformer 模型使用 **linear warmup + linear decay** 调度：

- Warmup 阶段：前 10% 的训练步数，学习率从 0 线性增长到目标 LR
- Decay 阶段：剩余步数，学习率线性衰减到 0
- Weight Decay：0.01（AdamW 优化器）

BiLSTM 和 BERT-Frozen MLP 使用固定学习率（Adam 优化器，lr=1e-3）。

### 4.4 Early Stopping

所有模型使用 **dev Macro-F1** 作为 early stopping 的监控指标：

- 每个 epoch 结束后评估 dev 集
- 如果连续 PATIENCE 个 epoch 未超过历史最优 dev F1，停止训练
- 恢复历史最优 dev F1 对应的模型权重
- PATIENCE：BERT/RoBERTa/BiLSTM/BERT-Frozen = 3，DeBERTa-v3 = 5

### 4.5 混合精度训练

- **BERT / RoBERTa：** FP16 混合精度（`torch.cuda.amp`）
- **DeBERTa-v3：** BF16 混合精度（DeBERTa-v3 的 LayerNorm 在 FP16 下产生 NaN）
- **BiLSTM / BERT-Frozen MLP：** FP32

---

## 5. 评估方法

### 5.1 单次划分评估

默认按 lemma 分组的 70/15/15 train/dev/test 划分。在 test 集上报告 Accuracy、Precision (macro)、Recall (macro)、Macro-F1。

### 5.2 5 折交叉验证

为获得更可靠的性能估计，使用**按 lemma 分组的 5 折交叉验证**：

1. 将全部 50,769 个样本按 lemma 分组，所有 lemma 按字母排序后随机打乱（seed=42）
2. 将 lemma 均匀轮流分配到 5 个组（`groups[i % 5]`）
3. 对于 fold *i*：
   - 测试集 = 第 *i* 组的所有样本
   - 验证集 = 第 *(i+1) mod 5* 组的所有样本
   - 训练集 = 其余 3 组的所有样本
4. 5 折的测试集互不重叠，合并覆盖全部样本

```
Fold 0: Train=[2,3,4] Dev=[1] Test=[0]
Fold 1: Train=[3,4,0] Dev=[2] Test=[1]
Fold 2: Train=[4,0,1] Dev=[3] Test=[2]
Fold 3: Train=[0,1,2] Dev=[4] Test=[3]
Fold 4: Train=[1,2,3] Dev=[0] Test=[4]
```

报告 5 折 Macro-F1 的 mean ± std。

### 5.3 Bootstrap 95% 置信区间

合并 5 折的全部预测结果（50,769 个样本），进行非参数 Bootstrap：

1. 从合并预测中**有放回抽样** N 个样本（N = 样本总数）
2. 计算该 Bootstrap 样本的 Macro-F1
3. 重复 1000 次
4. 取第 2.5 百分位和第 97.5 百分位作为 95% 置信区间

若两模型的 95% CI 不重叠，可认为差异有统计学意义。

### 5.4 McNemar 配对检验

McNemar 检验比较两个模型在**同一组样本**上的逐样本对错差异：

1. 合并 5 折预测，对每个样本标记 模型A 是否预测正确、模型B 是否预测正确
2. 构建 2×2 列联表：

   |  | B 对 | B 错 |
   |--|------|------|
   | **A 对** | a | b |
   | **A 错** | c | d |

3. 使用 **Edwards 连续性校正** 的 McNemar 检验统计量：

   ```
   χ² = (|b - c| - 1)² / (b + c)
   ```

4. 在 χ²(df=1) 分布下计算 p 值

同时对每一折单独运行 McNemar 检验，以观察显著性在各折间的稳定性。

### 5.5 官方 WiC 基准评估

使用 [官方 WiC 数据集](https://pilehvar.github.io/wic/)（Pilehvar & Camacho-Collados, 2019）进行跨域评估：

- Dev 集 638 个样本，Test 集 1400 个样本
- 正负样本完全平衡（各 50%）
- 仅包含名词（N）和动词（V）
- 数据来源：WordNet 例句、VerbNet、Wiktionary（与 SemCor 不同）

该评估旨在衡量模型的跨域泛化能力。

---

## 6. 语言学维度分析

评估脚本（`src/evaluate.py`）按三个语言学维度拆分测试集，分析各模型的 Macro-F1 表现：

### 6.1 按词性（POS）分析

将测试集按目标词词性分为四组（动词、名词、形容词、副词），分别评估各模型。各词性子集的类别分布差异较大，因此同时计算各子集的随机猜测基线作为参照。

### 6.2 按多义程度（Polysemy）分析

根据目标词在 WordNet 中的义项数量分为三组：
- 低多义（≤3 个义项）
- 中多义（4-6 个义项）
- 高多义（≥7 个义项）

### 6.3 按词频（Frequency）分析

根据目标词在训练集中的出现频次分为三组（低频、中频、高频），观察训练数据充分度对模型表现的影响。

---

## 7. BERT 上下文向量分析

使用微调后的 BERT 提取目标词的 768 维上下文向量，通过统计检验和可视化揭示模型如何在向量空间中编码词义信息。

### 7.1 余弦相似度分析

对每个测试样本，提取 sentence1 和 sentence2 中目标词的向量，计算余弦相似度。比较正例（同义）和负例（异义）的分布差异：

- **参数检验：** Welch's t-test（不假设等方差）
- **非参数检验：** Mann-Whitney U test
- **效应量：** Cohen's d
- **置信区间：** 95% Bootstrap CI（1000 次重采样）

### 7.2 分析维度

- **按词性分组**：观察模型对不同词性的义项区分能力
- **余弦阈值 vs 完整分类头**：对比单一余弦相似度分类与三向量拼接 + Linear 分类的效果差异
- **Fine-tune vs Frozen**：对比微调和冻结 BERT 的 embedding 质量

### 7.3 可视化方法

| 可视化 | 输入 | 方法 | 观察目的 |
|--------|------|------|---------|
| 余弦相似度分布 | cos(emb1, emb2) per sample | 直方图 + KDE | 正负例分布重叠程度 |
| 差向量 t-SNE | emb1 − emb2 | t-SNE 降维到 2D | 正负例在差异空间中的聚类 |
| 联合 t-SNE | emb1 和 emb2 联合 | t-SNE 降维到 2D | 验证共享向量空间假设 |
| 词性箱线图 | cos sim grouped by POS | 箱线图 | 各词性的义项区分度 |
| 激活热力图 | emb1 − emb2 per dim | 热力图 | 差异信号的维度分布 |
| L2 范数分布 | ‖emb1‖, ‖emb2‖, ‖emb1−emb2‖ | 直方图 | 方向 vs 长度信号 |

---

## 8. 可复现性

### 8.1 随机种子控制

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 8.2 环境依赖

- Python 3.8+
- PyTorch 2.0+（CUDA）
- transformers（HuggingFace）
- scikit-learn
- scipy
- nltk
- sentence-transformers
- matplotlib, tqdm

### 8.3 硬件

所有实验在单 GPU 环境下运行。Transformer 模型使用混合精度训练以节省显存。

### 8.4 数据与模型权重

所有训练好的模型权重和语料库文件存储在 Google Drive，可通过 rclone 或 gdown 一键下载（详见 [README.md](README.md) 第 5.2 节）。
