# BEDA-FND Overall Experimental Results and Per-Run Results

This document summarizes the BEDA-FND results and per-run results reported in the current manuscript. Overall and per-domain metrics retain the three-decimal precision used in the manuscript’s main tables; for conciseness, structural ablation and per-run results are reported as accuracy percentages rounded to one decimal place.

## Overall Results

| Dataset | Macro-F1 | Accuracy |   AUC |
| ------- | -------: | -------: | ----: |
| Weibo   |    0.952 |    0.952 | 0.987 |
| Weibo21 |    0.960 |    0.960 | 0.989 |

## Per-Domain Results

The values below represent the Macro-F1 scores for individual domains. The last column corresponds to the International domain in Weibo and the Disasters and Accidents domain in Weibo21.

| Dataset | Science/Technology | Military | Education | Society | Politics | Health | Finance | Entertainment | International/Disasters |
| ------- | -----------------: | -------: | --------: | ------: | -------: | -----: | ------: | ------------: | ----------------------: |
| Weibo   |              0.912 |    0.934 |     0.946 |   0.950 |    0.869 |  0.967 |   0.923 |         0.930 |                   0.972 |
| Weibo21 |              0.978 |    0.953 |     0.953 |   0.942 |    0.983 |  0.952 |   0.927 |         0.980 |                   0.983 |

## Structural Ablations

DRCM stands for Domain Representation and Context Modeling, BEM stands for Branch-specific Evidence Modeling, and DCEA stands for Domain-Conditioned Evidence Arbitration. ✓ indicates that the corresponding component is retained, while ✗ indicates that it is removed or bypassed.

| Variant     | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | Weibo Acc. (%) | Weibo21 Acc. (%) |
| ----------- | :--: | :-: | :------: | :----------------: | -------------: | ---------------: |
| Full Model  |   ✓  |  ✓  |     ✓    |          ✓         |           95.2 |             96.0 |
| w/o DRCM    |   ✗  |  ✓  |     ✓    |          ✓         |           94.1 |             94.5 |
| w/o BEM     |   ✓  |  ✗  |     ✓    |          ✓         |           93.7 |             93.9 |
| w/o Refine. |   ✓  |  ✓  |     ✓    |          ✗         |           94.4 |             95.0 |
| w/o DCEA    |   ✓  |  ✓  |     ✗    |          ✗         |           93.9 |             94.1 |
| w/o All     |   ✗  |  ✗  |     ✗    |          ✗         |           93.1 |             92.6 |

* **w/o DRCM**: Retain semantic extraction and allocation, disable the domain inputs and domain supervision for BEM, and set the domain context in DCEA to zero.
* **w/o BEM**: Share evidence across branches and disable matching enhancement.
* **w/o Refine.**: Retain the reference decision p₀ and remove p_w and the final fusion stage.
* **w/o DCEA**: Retain branch evidence and domain processing, and output the mean of the probabilities from the three branches.
* **w/o All**: Bypass all three components and use standard fusion of encoded features with an FFN classification head.

## Per-Run Accuracy for the Main Results

The table below lists the accuracies of all runs included in the current main-result summaries. Values are expressed as percentages; means are calculated with equal weighting using the original, unrounded metrics.

| Dataset | Value 1 | Value 2 | Value 3 | Mean Accuracy (%) | Main Table Accuracy |
| ------- | ------: | ------: | ------: | ----------------: | ------------------: |
| Weibo   | 95.2901 | 95.2218 | 94.8123 |           95.1775 |               0.952 |
| Weibo21 | 96.2602 | 95.9350 | 95.7724 |           95.9892 |               0.960 |

## Per-Run Accuracy for the Ablation Experiments

The table below lists all accuracies included in the current ablation-table statistics. Values are expressed as percentages, and means are calculated using the original precision.

| Dataset | Variant     | Runs | Value 1 | Value 2 | Value 3 | Mean Accuracy (%) | Value Reported in the Paper (%) |
| ------- | ----------- | ---: | ------: | ------: | ------: | ----------------: | ------------------------------: |
| Weibo   | w/o DRCM    |    3 | 94.6075 | 93.9249 | 93.8567 |           94.1297 |                            94.1 |
| Weibo   | w/o BEM     |    3 | 93.4471 | 94.0614 | 93.5836 |           93.6974 |                            93.7 |
| Weibo   | w/o Refine. |    3 | 94.3345 | 94.1297 | 94.8123 |           94.4255 |                            94.4 |
| Weibo   | w/o DCEA    |    3 | 93.1741 | 94.1980 | 94.2662 |           93.8794 |                            93.9 |
| Weibo   | w/o All     |    3 | 92.9693 | 93.1058 | 93.2423 |           93.1058 |                            93.1 |
| Weibo21 | w/o DRCM    |    3 | 93.9837 | 94.6341 | 94.7967 |           94.4715 |                            94.5 |
| Weibo21 | w/o BEM     |    3 | 94.3089 | 93.6585 | 93.8211 |           93.9295 |                            93.9 |
| Weibo21 | w/o Refine. |    3 | 94.4715 | 95.1220 | 95.2846 |           94.9593 |                            95.0 |
| Weibo21 | w/o DCEA    |    3 | 93.3333 | 94.7967 | 94.1463 |           94.0921 |                            94.1 |
| Weibo21 | w/o All     |    3 | 92.5203 | 92.3577 | 92.8455 |           92.5745 |                            92.6 |




# BEDA-FND 实验整体结果及逐次结果

本文档汇总当前论文报告的 BEDA-FND 结果及逐次结果。整体与分领域指标沿用论文主表的三位小数；为了保证文档的简洁性，结构消融以及逐次结果使用准确率，以百分比表示，保留一位小数。

## 整体结果

| Dataset | Macro-F1 | Accuracy | AUC |
|---|---:|---:|---:|
| Weibo | 0.952 | 0.952 | 0.987 |
| Weibo21 | 0.960 | 0.960 | 0.989 |

## 分领域结果

以下数值为各领域的 Macro-F1。最后一列在 Weibo 中对应国际领域，在 Weibo21 中对应灾难事故领域。

| Dataset | 科学/科技 | 军事 | 教育 | 社会 | 政治 | 健康 | 财经 | 娱乐 | 国际/灾难 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Weibo | 0.912 | 0.934 | 0.946 | 0.950 | 0.869 | 0.967 | 0.923 | 0.930 | 0.972 |
| Weibo21 | 0.978 | 0.953 | 0.953 | 0.942 | 0.983 | 0.952 | 0.927 | 0.980 | 0.983 |

## 结构消融

DRCM 为 Domain Representation and Context Modeling，BEM 为 Branch-specific Evidence Modeling，DCEA 为 Domain-Conditioned Evidence Arbitration。✓ 表示保留相应组件，✗ 表示移除或旁路相应组件。

| Variant | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | Weibo Acc. (%) | Weibo21 Acc. (%) |
|---|:---:|:---:|:---:|:---:|---:|---:|
| Full Model | ✓ | ✓ | ✓ | ✓ | 95.2 | 96.0 |
| w/o DRCM | ✗ | ✓ | ✓ | ✓ | 94.1 | 94.5 |
| w/o BEM | ✓ | ✗ | ✓ | ✓ | 93.7 | 93.9 |
| w/o Refine. | ✓ | ✓ | ✓ | ✗ | 94.4 | 95.0 |
| w/o DCEA | ✓ | ✓ | ✗ | ✗ | 93.9 | 94.1 |
| w/o All | ✗ | ✗ | ✗ | ✗ | 93.1 | 92.6 |

- **w/o DRCM**：保留语义提取与分配，关闭 BEM 的领域输入和领域监督，并将 DCEA 的领域上下文置零。
- **w/o BEM**：共享分支证据并关闭匹配增强。
- **w/o Refine.**：保留参考决策 p₀，移除 p_w 与最终融合部分。
- **w/o DCEA**：保留分支证据与领域处理，以三个分支概率的均值输出。
- **w/o All**：旁路三个组件，使用普通编码特征融合与 FFN 分类头。


## 主结果的逐次准确率

下表列出参与当前主结果汇总的全部运行准确率。单位为百分比；均值按未四舍五入的原始指标等权计算。

| Dataset | 值1 | 值2 | 值3 | 平均 Accuracy (%) | 主表 Accuracy |
|---|---:|---:|---:|---:|---:|
| Weibo | 95.2901 | 95.2218 | 94.8123 | 95.1775 | 0.952 |
| Weibo21 | 96.2602 | 95.9350 | 95.7724 | 95.9892 | 0.960 |

Weibo 、Weibo21 汇总三个结果，均包含保留的历史正式检查点与后续选取的运行。

## 消融实验的逐次准确率

以下展示参与当前消融表统计的全部准确率。“—”表示该行没有对应次数的结果。单位为百分比，均值使用原始精度计算。

| Dataset | Variant | 次数 | 值1 | 值2 | 值3 | 平均 Accuracy (%) | 论文表值 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Weibo | w/o DRCM | 3 | 94.6075 | 93.9249 | 93.8567 | 94.1297 | 94.1 |
| Weibo | w/o BEM | 3 | 93.4471 | 94.0614 | 93.5836 | 93.6974 | 93.7 |
| Weibo | w/o Refine. | 3 | 94.3345 | 94.1297 | 94.8123 | 94.4255 | 94.4 |
| Weibo | w/o DCEA | 3 | 93.1741 | 94.1980 | 94.2662 | 93.8794 | 93.9 |
| Weibo | w/o All | 3 | 92.9693 | 93.1058 | 93.2423 | 93.1058 | 93.1 |
| Weibo21 | w/o DRCM | 3 | 93.9837 | 94.6341 | 94.7967 | 94.4715 | 94.5 |
| Weibo21 | w/o BEM | 3 | 94.3089 | 93.6585 | 93.8211 | 93.9295 | 93.9 |
| Weibo21 | w/o Refine. | 3 | 94.4715 | 95.1220 | 95.2846 | 94.9593 | 95.0 |
| Weibo21 | w/o DCEA | 3 | 93.3333 | 94.7967 | 94.1463 | 94.0921 | 94.1 |
| Weibo21 | w/o All | 3 | 92.5203 | 92.3577 | 92.8455 | 92.5745 | 92.6 |
