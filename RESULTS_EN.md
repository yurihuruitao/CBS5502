# 4. Results

This section presents the experimental findings in four parts: overall model performance on the primary evaluation (Section 4.1), statistical validation of observed differences (Section 4.2), linguistic dimension analyses that reveal where and why models succeed or fail (Section 4.3), and a probing analysis of BERT's internal representations that sheds light on how contextualized embeddings encode word sense (Section 4.4).

## 4.1 Model Performance

### 4.1.1 Single-Split Evaluation

Table 3 reports the performance of all six models and two baselines on the held-out test set (*N* = 7,476). The models are ordered by Macro-F1, the primary metric.

**Table 3**

*Single-Split Test Set Performance*

| Model | Accuracy | Precision | Recall | Macro-F1 |
|-------|----------|-----------|--------|----------|
| Majority Baseline | .611 | .306 | .500 | .379 |
| Random Baseline | .523 | .494 | .494 | .494 |
| Sentence-BERT | .575 | .566 | .569 | .565 |
| BiLSTM | .628 | .653 | .655 | .627 |
| BERT-Frozen + MLP | .739 | .726 | .726 | .726 |
| RoBERTa | .737 | .729 | .740 | .731 |
| DeBERTa-v3 | .755 | .744 | .734 | .738 |
| **BERT** | **.751** | **.741** | **.750** | **.744** |

Several patterns are immediately apparent. First, all models that explicitly extract the target word's representation — that is, all models except Sentence-BERT — substantially outperform the baselines, confirming that lexical disambiguation requires more than holistic sentence comparison. Second, the three fine-tuned transformer models (BERT, RoBERTa, DeBERTa-v3) cluster together at the top, separated by fewer than 1.3 percentage points, while a clear gap of approximately 10 points separates them from the non-contextualized BiLSTM. Third, BERT fine-tuning achieves the highest Macro-F1 (.744), followed closely by DeBERTa-v3 (.738) and RoBERTa (.731).

### 4.1.2 Five-Fold Cross-Validation

To obtain more robust estimates, all models were evaluated via lemma-grouped 5-fold cross-validation over the full dataset (*N* = 50,769). Table 4 summarizes the results.

**Table 4**

*Five-Fold Cross-Validation Results (Macro-F1)*

| Model | Mean | *SD* | Range |
|-------|------|------|-------|
| Sentence-BERT | .563 | .009 | .549 – .574 |
| BiLSTM | .573 | .040 | .506 – .621 |
| BERT-Frozen + MLP | .714 | .016 | .691 – .738 |
| RoBERTa | .726 | .016 | .707 – .756 |
| DeBERTa-v3 | .642 | .144 | .375 – .762 |
| **BERT** | **.735** | **.010** | **.724 – .752** |

BERT fine-tuning emerged as the best-performing model overall, combining the highest mean Macro-F1 (.735) with the lowest standard deviation (.010). This stability is noteworthy: across five independent data partitions, BERT's performance varied by fewer than 3 percentage points, indicating that its learned representations generalize reliably across different subsets of the lexicon.

DeBERTa-v3 presents the most striking case. Although it achieved the highest single-fold score of any model (F1 = .762 on Fold 4), two of its five folds suffered training collapse — the model degenerated into majority-class prediction, producing F1 scores of .375 and .606. When only the three successful folds are considered, DeBERTa-v3's mean F1 (.743) matches BERT's overall average, suggesting that its architectural innovations (disentangled attention, richer feature concatenation) hold genuine promise but are undermined by training instability under mixed-precision conditions.

The BiLSTM model exhibited the second-highest variance (*SD* = .040), with performance ranging from near-random (F1 = .506) to moderate (F1 = .621). This instability reflects the fundamental limitation of static word embeddings: because GloVe vectors assign a single representation to each word form regardless of context, the model's success depends heavily on whether the particular lemmas in each fold happen to have senses that are distinguishable from distributional context alone.

### 4.1.3 Cross-Domain Generalization

To test whether models trained on SemCor-derived data can generalize to out-of-distribution inputs, all models were evaluated on the official WiC benchmark (Pilehvar & Camacho-Collados, 2019), which draws its instances from WordNet example sentences, VerbNet, and Wiktionary rather than running text. Table 5 reports the results.

**Table 5**

*Performance on the Official WiC Benchmark*

| Model | Dev Acc | Dev F1 | Test Acc | Test F1 |
|-------|---------|--------|----------|---------|
| Random Baseline | .500 | .500 | .500 | .500 |
| BiLSTM | .536 | .521 | .531 | .517 |
| Sentence-BERT | .585 | .585 | .523 | .522 |
| BERT-Frozen + MLP | .638 | .628 | .611 | .603 |
| RoBERTa | .666 | .660 | .655 | .650 |
| **BERT** | **.669** | **.659** | **.666** | **.658** |

All models showed reduced performance compared to the in-domain evaluation, confirming the expected domain shift. However, the performance ranking remained largely consistent: BERT and RoBERTa led, followed by BERT-Frozen, with BiLSTM and Sentence-BERT near the random baseline. This consistency suggests that the relative advantages of fine-tuned contextualized models reflect genuine differences in representational quality rather than artifacts of the training distribution.

The BiLSTM's near-random performance on the official benchmark (Test F1 = .517 vs. random .500) is particularly revealing. It indicates that static word embeddings, which lack the ability to modulate word representations based on context, fail almost entirely when applied to unfamiliar lexical items drawn from a different text genre.

## 4.2 Statistical Validation

### 4.2.1 Bootstrap Confidence Intervals

To quantify the uncertainty around each model's performance estimate, 95% bootstrap confidence intervals were constructed from the pooled five-fold predictions (*N* = 50,769; 1,000 resamples). Table 6 presents the results.

**Table 6**

*Bootstrap 95% Confidence Intervals for Macro-F1*

| Model | Macro-F1 | 95% CI | Width |
|-------|----------|--------|-------|
| Sentence-BERT | .565 | [.561, .569] | .009 |
| BiLSTM | .576 | [.572, .581] | .009 |
| BERT-Frozen + MLP | .716 | [.712, .720] | .008 |
| RoBERTa | .729 | [.725, .733] | .008 |
| **BERT** | **.736** | **[.732, .740]** | **.008** |
| DeBERTa-v3 | .680 | [.677, .685] | .008 |

The narrow confidence intervals (all under .01 in width) reflect the large sample size and indicate that the performance estimates are precise. Critically, BERT's confidence interval [.732, .740] does not overlap with that of any other model, providing strong evidence that its superiority is not attributable to sampling variability. The gap between BERT and RoBERTa — whose intervals are [.732, .740] and [.725, .733], respectively — is small but statistically reliable.

DeBERTa-v3's interval [.677, .685] falls below that of BERT-Frozen + MLP [.712, .720], a direct consequence of the two collapsed folds that substantially reduced its pooled performance. This result underscores an important practical consideration: a model's theoretical capability matters little if it cannot be trained reliably.

### 4.2.2 McNemar's Pairwise Tests

McNemar's test with Edwards' continuity correction was applied to all 15 pairwise model comparisons using pooled five-fold predictions. All 15 pairs yielded *p* < .001, confirming that every observed performance difference is statistically significant. The complete pairwise results establish a clear and robust ranking:

> BERT > RoBERTa > BERT-Frozen + MLP > DeBERTa-v3 > BiLSTM > Sentence-BERT

To assess whether these differences are consistent across data partitions, McNemar's test was additionally conducted within each fold. Table 7 reports fold-level *p*-values for three key comparisons.

**Table 7**

*Fold-Level McNemar's Test p-Values for Selected Model Pairs*

| Model Pair | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|------------|--------|--------|--------|--------|--------|
| BERT vs. RoBERTa | .054 | .129 | < .001 | < .001 | < .001 |
| BERT vs. DeBERTa-v3 | < .001 | < .001 | .489 | < .001 | < .001 |
| BERT-Frozen vs. BERT | < .001 | < .001 | < .001 | .204 | .851 |

The fold-level analysis reveals important nuances. The BERT–RoBERTa difference does not reach significance in Folds 0 and 1 individually, suggesting that BERT's advantage is modest in magnitude and only becomes statistically detectable when aggregated across the full dataset. The BERT–DeBERTa-v3 comparison in Fold 2 (*p* = .489) — one of the three folds where DeBERTa-v3 trained successfully — confirms that the two models perform comparably when DeBERTa-v3's training proceeds normally. Similarly, the BERT-Frozen–BERT comparison is non-significant in Folds 3 and 4, where the frozen model happened to perform particularly well, narrowing the gap with its fine-tuned counterpart.

## 4.3 Linguistic Dimension Analysis

To move beyond aggregate metrics and understand which linguistic properties make disambiguation easier or harder, the test set was stratified along three dimensions: part of speech, polysemy level, and word frequency.

### 4.3.1 Part of Speech

Table 8 reports Macro-F1 by the part of speech of the target word. Because class distributions vary substantially across POS categories — for instance, 73% of adverb instances are positive (same-sense) compared with only 25.5% for verbs — a per-subgroup random baseline is included for calibrated comparison.

**Table 8**

*Macro-F1 by Part of Speech*

| POS | *N* | Random | BiLSTM | BERT-Frozen | BERT | RoBERTa | SBERT |
|-----|-----|--------|--------|-------------|------|---------|-------|
| Adjective | 993 | .499 | .463 | .663 | **.676** | .667 | .562 |
| Adverb | 237 | .472 | .532 | .555 | .530 | **.559** | .494 |
| Noun | 2,734 | .500 | .609 | .737 | **.753** | .728 | .604 |
| Verb | 3,512 | .468 | .605 | .665 | **.705** | .697 | .514 |

**Figure 2**

*Cosine Similarity Distributions of Target Word Embeddings by Part of Speech*

![Figure 2. Cosine similarity boxplots by POS](plots/cosine_by_pos.png)

*Note.* For each POS category, the left box (blue) shows the cosine similarity distribution between target word embeddings in same-sense pairs, and the right box (red) shows different-sense pairs. The gap between boxes reflects the model's ability to encode sense distinctions in its embedding space.

The most striking finding is the sharp contrast between nouns and adverbs. Nouns are the easiest category to disambiguate: BERT achieves F1 = .753, the highest score for any POS–model combination, and even the frozen BERT baseline reaches .737. This aligns with the linguistic intuition that noun senses tend to have relatively clear semantic boundaries — *bank* as a financial institution versus a river bank involves fundamentally different conceptual domains. The embedding analysis in Figure 2 corroborates this: nouns show the largest separation between same-sense and different-sense cosine similarity distributions (Cohen's *d* = 0.593).

Adverbs, by contrast, represent a shared weakness across all models. BERT's F1 on adverbs (.530) barely exceeds the random baseline (.472) and is actually lower than the BiLSTM's (.532) — the only POS category where a static embedding model matches a fine-tuned transformer. As shown in Figure 2, the positive and negative cosine distributions for adverbs overlap almost completely (Cohen's *d* = 0.031, *p* = .832), indicating that BERT's embedding space does not meaningfully distinguish adverb senses. This difficulty likely stems from the nature of adverbial polysemy: adverb senses (e.g., *still* meaning "motionless" vs. "nevertheless") often operate at a pragmatic or discourse-functional level rather than a referential level, making them harder to capture through distributional semantics alone.

Verbs and adjectives occupy intermediate positions. Verbs benefit most from fine-tuning: BERT's advantage over the frozen variant is largest for verbs (+4.0 points), suggesting that verb sense distinctions require the model to learn task-specific contextual patterns beyond what general pre-training provides. Adjectives show the smallest gap between BERT and RoBERTa (< 1 point), indicating that adjective disambiguation is relatively insensitive to the choice of pre-trained model.

### 4.3.2 Polysemy Level

Target words were stratified by the number of senses listed in WordNet: low polysemy (1–3 senses), medium polysemy (4–6 senses), and high polysemy (7 or more senses). Table 9 reports the results.

**Table 9**

*Macro-F1 by Polysemy Level*

| Polysemy | *N* | Random | BiLSTM | BERT-Frozen | BERT | RoBERTa | SBERT |
|----------|-----|--------|--------|-------------|------|---------|-------|
| Low (1–3) | 2,587 | .478 | .478 | .606 | **.657** | .638 | .545 |
| Medium (4–6) | 1,648 | .489 | .448 | .627 | **.626** | .597 | .532 |
| High (7+) | 3,241 | .433 | .539 | .661 | **.674** | .666 | .497 |

A counterintuitive pattern emerges: highly polysemous words (7+ senses) are actually *easier* to disambiguate than words with moderate polysemy (4–6 senses). BERT achieves F1 = .674 on the high-polysemy group but only .626 on the medium group. This finding can be explained by the semantic distance hypothesis: words with many senses tend to have senses that span widely separated semantic domains (e.g., *run* as physical movement, managing an organization, a sequence of events, a flow of liquid), making any given pair of senses relatively easy to distinguish. Words with moderate polysemy, by contrast, often have senses that are semantically adjacent — subtle distinctions within the same general domain — which are precisely the cases that challenge distributional models.

The low-polysemy group presents a different kind of challenge. Although these words have few senses, the random baseline is relatively high (.478) because of extreme class imbalance (most pairs of a word with few senses share the same sense). The BiLSTM scores exactly at the random baseline (.478) for low-polysemy words, meaning it learns nothing useful for this category. This underscores a fundamental limitation: without contextual representations, models cannot distinguish between occurrences that happen to share a sense and those that genuinely differ, particularly when the number of possible senses is small and most usages are of the same sense.

### 4.3.3 Word Frequency

Target words were categorized by occurrence count in the training set into low-, medium-, and high-frequency groups. Table 10 reports the results.

**Table 10**

*Macro-F1 by Word Frequency*

| Frequency | *N* | Random | BiLSTM | BERT-Frozen | BERT | RoBERTa | SBERT |
|-----------|-----|--------|--------|-------------|------|---------|-------|
| Low | 1,135 | .395 | .495 | .537 | **.597** | .570 | .453 |
| Medium | 416 | .497 | .392 | .549 | **.612** | .598 | .579 |
| High | 5,925 | .473 | .571 | .678 | **.689** | .677 | .531 |

All models show a clear performance gradient from high-frequency to low-frequency words, confirming the expected relationship between data availability and disambiguation quality. BERT maintains its lead across all frequency bands, but its advantage over RoBERTa is most pronounced on low-frequency words (+2.7 points) — the category where generalization matters most because the model has seen fewest training examples.

The BiLSTM's behavior on medium-frequency words is notable: its F1 (.392) falls below the random baseline (.497), suggesting that the model has learned misleading associations that actively hurt performance. This may reflect the vulnerability of static embeddings to spurious distributional patterns that arise from limited training data.

## 4.4 Contextualized Embedding Analysis

To understand *how* BERT encodes word sense information, a series of probing analyses were conducted on the 768-dimensional target word embeddings extracted from the final layer of the fine-tuned model.

### 4.4.1 Sense Distinctions in Embedding Space

The central question is whether BERT's embedding space geometrically separates same-sense word pairs from different-sense pairs. Figure 3 provides the answer.

**Figure 3**

*Distribution of Cosine Similarities Between Target Word Embeddings for Same-Sense and Different-Sense Pairs*

![Figure 3. Cosine similarity distribution](plots/cosine_distribution.png)

*Note.* Blue: same-sense pairs (mean = 0.500); Red: different-sense pairs (mean = 0.405). Welch's *t* = 23.30, *p* < .001; Cohen's *d* = 0.56 (medium effect). Dashed lines indicate group means.

The fine-tuned BERT model encodes word sense information in a measurable and statistically robust way. Same-sense word pairs have significantly higher cosine similarity (mean = 0.500) than different-sense pairs (mean = 0.405), with a medium effect size (Cohen's *d* = 0.56). This confirms that fine-tuning reshapes BERT's representation space so that identical senses in different contexts are pulled closer together while distinct senses are pushed apart.

However, Figure 3 also reveals substantial overlap between the two distributions. This overlap explains why a simple cosine-threshold classifier achieves only Macro-F1 = .625 — far below the full model's .744. The scalar cosine similarity, while informative, discards much of the discriminative information that is distributed across the 768 embedding dimensions.

### 4.4.2 Visualizing the Geometry of Sense

Two complementary t-SNE visualizations illustrate how sense information is organized in the embedding space.

**Figure 4**

*t-SNE Projection of Difference Vectors (t₁ − t₂) for Same-Sense and Different-Sense Pairs*

![Figure 4. t-SNE of difference vectors](plots/tsne_diff.png)

*Note.* Each point represents one test instance. Blue: same-sense pairs; Red: different-sense pairs. Same-sense pairs tend to cluster near the center (small difference vectors), while different-sense pairs spread toward the periphery.

The difference vectors reveal a clear geometric pattern: same-sense pairs cluster near the origin of the difference space (small, concentrated vectors), while different-sense pairs are scattered toward the periphery (larger, more variable vectors). This is consistent with the cosine similarity analysis — when two occurrences share a sense, their embeddings are similar and their difference is small. The tight peripheral clusters likely correspond to specific lemmas for which the model has learned particularly distinct sense representations.

**Figure 5**

*Joint t-SNE Projection of Target Word Embeddings from Both Sentences*

![Figure 5. Joint t-SNE of embeddings from both sentences](plots/tsne_emb12.png)

*Note.* Left panel: embeddings from sentence 1; Right panel: embeddings from sentence 2. Both panels share the same t-SNE coordinate space. The highly similar spatial structure across panels confirms that BERT maps target words from both sentence positions into a unified representational space.

Figure 5 addresses a more fundamental question: does BERT represent target words from the two sentences in the same embedding space, or does the sentence position introduce systematic distortions? The answer is clear — the two panels show nearly identical spatial structure, confirming that the model constructs a position-invariant representation of word meaning. This is a prerequisite for meaningful cross-sentence comparison and validates the architectural choice of extracting and comparing target word embeddings from the two sentence positions.

### 4.4.3 What the Classifier Sees Beyond Cosine Similarity

The gap between cosine-threshold classification (F1 = .625) and the full classification head (F1 = .744) raises a natural question: what additional information does the classifier exploit? Figure 6 provides an answer.

**Figure 6**

*Dimension-Level Activation Differences Between Target Word Pairs*

![Figure 6. Activation difference heatmap](plots/activation_heatmap.png)

*Note.* Each row represents one test instance; each column represents one of the first 200 embedding dimensions. Top panel: 50 same-sense pairs; Bottom panel: 50 different-sense pairs. Blue indicates negative differences; red indicates positive differences; white indicates near-zero difference. Same-sense pairs show predominantly white (small differences), while different-sense pairs display pronounced colored bands (large differences) distributed across many dimensions.

The heatmap reveals that sense-discriminative information is broadly distributed across many embedding dimensions rather than concentrated in a few. Same-sense pairs (top panel) show predominantly white bands — small differences across nearly all dimensions. Different-sense pairs (bottom panel), by contrast, display conspicuous colored streaks across numerous dimensions. This distributed pattern explains why the linear classification head, which can assign a learned weight to each of the 2,304 concatenated features, substantially outperforms a single cosine similarity score: the classifier effectively combines weak signals from many dimensions into a strong aggregate prediction.

### 4.4.4 Direction vs. Magnitude

**Figure 7**

*L2 Norm Distributions of Target Word Embeddings*

![Figure 7. L2 norm distributions](plots/norm_distribution.png)

*Note.* Left: L2 norms of embeddings from sentence 1 (blue) and sentence 2 (orange), showing nearly identical distributions. Right: absolute norm difference, with same-sense pairs (blue) concentrated near zero and different-sense pairs (red) showing a heavier tail.

A final analysis disentangles directional similarity (cosine) from magnitude-based signals (L2 norm). The left panel of Figure 7 confirms that BERT does not systematically produce embeddings of different magnitude depending on sentence position — the norm distributions for sentence 1 and sentence 2 are virtually identical. The right panel shows that same-sense pairs have more similar norms than different-sense pairs (Welch's *t* = −6.95, *p* < .001), but the effect is small (Cohen's *d* = 0.16). In BERT's fine-tuned embedding space, therefore, the primary signal for sense discrimination is *directional* (cosine similarity, *d* = 0.56), with vector magnitude providing only a weak supplementary cue.

## 4.5 Summary of Key Findings

The experimental results converge on five principal findings:

1. **BERT fine-tuning is the most reliable approach.** It achieves the highest Macro-F1 across all evaluation settings — single-split (.744), five-fold cross-validation (.735), bootstrap estimation (.736), and the official WiC benchmark (.658) — while maintaining the lowest variance (*SD* = .010). All pairwise differences are statistically significant (*p* < .001).

2. **Fine-tuning confers a consistent but modest advantage over frozen features.** The performance gap between BERT fine-tuning and the frozen BERT + MLP baseline is approximately 2 percentage points across all evaluation settings, confirming that task-specific adaptation of the encoder's representation space contributes meaningful — if not dramatic — improvements.

3. **A clear representational hierarchy emerges: fine-tuned contextualized > frozen contextualized > static > sentence-level.** BERT fine-tuned (.735) > BERT-Frozen (.714) > BiLSTM (.573) > Sentence-BERT (.563). Each level represents a statistically significant improvement, and the largest jump (~14 points) occurs at the transition from static to contextualized representations, highlighting the transformative impact of context-dependent word meaning.

4. **Target word localization is essential.** Sentence-BERT, despite employing a pre-trained transformer, performs worse than the BiLSTM because it lacks access to target word position information. This demonstrates that for word-level disambiguation, the granularity of the representation matters as much as the power of the encoder.

5. **Linguistic properties systematically modulate disambiguation difficulty.** Nouns are easiest, adverbs are hardest, and even fine-tuned BERT fails to encode adverb sense distinctions in its embedding space. Highly polysemous words are easier to distinguish than moderately polysemous ones, likely because their senses span greater semantic distances. Low-frequency words remain challenging for all models, reflecting the fundamental dependence of distributional methods on sufficient data.
