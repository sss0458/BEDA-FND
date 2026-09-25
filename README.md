# BEDA-FND

Official research code for **“From Multimodal Signals to Reliable Evidence: Domain-Aware Fake News Detection.”**

**BEDA-FND** stands for **Branch Evidence and Domain Arbitration for Multimodal Fake News Detection**. It follows a two-stage principle: first convert raw multimodal signals into branch-specific evidence, then let the observed domain and sample interaction state arbitrate among that evidence.Individual accuracies, aggregate values, and their statistical scope are available in the [experiment report](EXPERIMENT_REPORT_DUAL_STATE_ZH.md).

## Model

- **Domain Representation and Context Modeling (DRCM)** forms branch-level domain representations before BEM and supplies domain context to DCEA.
- **Branch-specific Evidence Modeling (BEM)** preserves text, visual, and cross-modal relation evidence instead of collapsing them into one representation too early.
- **Domain-Conditioned Evidence Arbitration (DCEA)** separately estimates image-text feature compatibility and three-branch decision agreement, combines them into four continuous interaction states, and lets domain-conditioned experts arbitrate among the evidence.


## Repository layout

```text
BEDA-FND/
├── src/
│   ├── main.py                 # training entry point
│   ├── config.py               # Full and structural ablation definitions
│   ├── run.py                  # data and training orchestration
│   ├── model/beda_fnd.py       # model and trainer
│   ├── model/                  # shared layers
│   ├── utils/                  # dataset loaders and metrics
│   └── util/                   # visual encoder utilities
├── scripts/
│   ├── run_ablations.sh        # Full and four structural ablations
│   ├── smoke_state_arbitration.py   # four-state numerical invariant test
│   ├── evaluate_ablations.sh   # fixed-test evaluation
│   ├── evaluate.py             # Accuracy, F1, AUC and calibration metrics
│   ├── export_all_features.sh  # export decision representations
│   ├── export_features.py
│   ├── plot_tsne.py            # model-level t-SNE comparison
│   ├── plot_ablation.py        # ablation accuracy figure
│   └── profile_complexity.py   # FLOPs, parameters and latency
├── EXPERIMENT_REPORT_DUAL_STATE_ZH.md  # paper results and per-run accuracy
└── requirements.txt
```


## Environment

The reference environment uses Python 3.10, PyTorch with CUDA, and an NVIDIA RTX 4090 GPU.

```bash
pip install -r requirements.txt
```

Place pretrained assets under `src/`:

```text
src/
├── mae_pretrain_vit_base.pth
├── clip_cn_vit-b-16.pt
└── pretrained_model/
    └── chinese_roberta_wwm_base_ext_pytorch/
        ├── config.json
        ├── pytorch_model.bin
        └── vocab.txt
```

## Data

The Weibo and Weibo21 data are **not redistributed in this repository**. They were obtained under authorization from the data provider, and that authorization does not grant us redistribution rights. Users must obtain permission and the original data directly from the corresponding authors or rights holders.

Prepare each dataset in a separate directory. The loaders expect the following files:

```text
data/
├── weibo/
│   ├── train_2_domain.csv
│   ├── val_2_domain.csv
│   ├── test_2_domain.csv
│   ├── train_loader.pkl
│   ├── val_loader.pkl
│   ├── test_loader.pkl
│   └── *_clip_loader.pkl
└── weibo21/
    ├── train_2_domain.xlsx
    ├── val_2_domain.xlsx
    ├── test_2_domain.xlsx
    ├── train_loader.pkl
    ├── val_loader.pkl
    ├── test_loader.pkl
    └── *_clip_loader.pkl
```

The preprocessing entry points are `src/data_pre.py`, `src/clip_data_pre.py`, `src/weibo21_data_pre.py`, and `src/weibo21_clip_data_pre.py`.

## Training

Choose an integer random seed locally for your run:

```bash
read -r -p "Random seed for this run: " RUN_SEED
export RUN_SEED
```

Train the Full model on Weibo:

```bash
python src/main.py \
  --dataset weibo \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo \
  --output-dir outputs/weibo/full
```

Weibo21 uses the same model definition:

```bash
python src/main.py \
  --dataset weibo21 \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo21 \
  --output-dir outputs/weibo21/full
```

Use `--init-checkpoint PATH` when reproducing the warm-start training protocol. 

## Structural ablations

The paper reports the Full model and five structural ablations:

| Variant | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | Structural setting |
|---|:---:|:---:|:---:|:---:|---|
| Full | ✓ | ✓ | ✓ | ✓ | Preserve separate branch evidence and use domain-conditioned arbitration. |
| w/o DRCM | ✗ | ✓ | ✓ | ✓ | Retain semantic extraction and allocation; disable BEM's domain inputs and domain supervision, and set DCEA's domain context to zero. |
| w/o BEM | ✓ | ✗ | ✓ | ✓ | Share semantic evidence across branches and base attention; disable matching reinforcement. |
| w/o Refine | ✓ | ✓ | ✓ | ✗ | Retain the reference decision p₀; remove p_w and the final fusion step. |
| w/o DCEA | ✓ | ✓ | ✗ | ✗ | Average the three branch probabilities, retaining evidence modeling and domain processing. |
| w/o All | ✗ | ✗ | ✗ | ✗ | Classify ordinary pooled/projected encoder features with the plain FFN head; bypass all three components and their auxiliary losses. |

Checkmarks denote retained components; crosses denote removed components. For w/o DRCM, zero domain context also nulls DCEA's domain-conditioned adjustment.

These commands expose the available structural settings. 
Run the available variants:

```bash
bash scripts/run_ablations.sh
bash scripts/evaluate_ablations.sh
```

Dataset paths can be overridden without editing the scripts:

```bash
WEIBO_ROOT=/path/to/weibo \
WEIBO21_ROOT=/path/to/weibo21 \
bash scripts/run_ablations.sh
```

Batch scripts require `RUN_SEED`; `REPORT_SEED` is accepted as a compatibility alias. The Python entry points require `--seed` and have no built-in training seed.

## Evaluation and figures

Evaluate one checkpoint:

```bash
python scripts/evaluate.py \
  --dataset weibo \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo \
  --checkpoint outputs/weibo/full/checkpoints/parameter_beda_fnd_best_accuracy.pkl \
  --output-dir outputs/evaluation/weibo/full
```

Generate the ablation figure:

```bash
python scripts/plot_ablation.py \
  --results-root outputs/evaluation \
  --output-dir outputs/figures
```

The current plotting script does not yet include w/o Refine; the complete paper table is shown below.

Export decision representations and create the model-level t-SNE comparison:

```bash
bash scripts/export_all_features.sh
python scripts/plot_tsne.py \
  --feature-dir outputs/features \
  --output-dir outputs/figures
```

## Reference accuracy results

| Variant | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | Weibo Acc. (%) | Weibo21 Acc. (%) |
|---|:---:|:---:|:---:|:---:|---:|---:|
| Full | ✓ | ✓ | ✓ | ✓ | **95.2** | **96.0** |
| w/o DRCM | ✗ | ✓ | ✓ | ✓ | 94.1 | 94.5 |
| w/o BEM | ✓ | ✗ | ✓ | ✓ | 93.7 | 93.9 |
| w/o Refine | ✓ | ✓ | ✓ | ✗ | 94.4 | 95.0 |
| w/o DCEA | ✓ | ✓ | ✗ | ✗ | 93.9 | 94.1 |
| w/o All | ✗ | ✗ | ✗ | ✗ | 93.1 | 92.6 |

Accuracy is displayed to one decimal place as in the paper. 

Check marks denote retained computational components. The w/o DRCM intervention retains semantic extraction and allocation, disables BEM's domain inputs and supervision, and zeros DCEA's domain context, also nulling its domain-conditioned adjustment.

## Acknowledgements

This implementation builds on the public codebases of MMDFND and DAMMFND. We thank their authors for releasing the datasets, model components, and training pipeline. Please also cite the original works when using this repository.

```bibtex
@inproceedings{lu2025dammfnd,
  title={DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection},
  author={Lu, Weihai and Tong, Yu and Ye, Zhiqiu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={559--567},
  year={2025}
}
```


# BEDA-FND

论文 **《从多模态信号到可靠证据：领域感知的虚假新闻检测》** 的官方研究代码。

**BEDA-FND** 是 **Branch Evidence and Domain Arbitration for Multimodal Fake News Detection（用于多模态虚假新闻检测的分支证据与领域仲裁）** 的缩写。该方法遵循两阶段原则：首先将原始多模态信号转化为分支特定证据，然后根据观测到的领域信息和样本交互状态，对这些证据进行仲裁。各次运行的准确率、汇总数值及其统计范围详见[实验报告](EXPERIMENT_REPORT_DUAL_STATE_ZH.md)。

## 模型

* **领域表示与上下文建模（DRCM）** 在 BEM 之前构建分支级领域表示，并向 DCEA 提供领域上下文。
* **分支特定证据建模（BEM）** 保留文本、视觉和跨模态关系证据，而不是过早地将它们压缩为单一表示。
* **领域条件化证据仲裁（DCEA）** 分别估计图文特征兼容性和三分支决策一致性，将二者组合为四种连续交互状态，并由领域条件化专家对证据进行仲裁。


## 仓库结构

```text
BEDA-FND/
├── src/
│   ├── main.py                 # 训练入口
│   ├── config.py               # 完整模型及结构消融定义
│   ├── run.py                  # 数据与训练流程调度
│   ├── model/beda_fnd.py       # 模型与训练器
│   ├── model/                  # 共享层
│   ├── utils/                  # 数据集加载器与评估指标
│   └── util/                   # 视觉编码器工具
├── scripts/
│   ├── run_ablations.sh        # 完整模型及四种结构消融
│   ├── smoke_state_arbitration.py   # 四状态数值不变量测试
│   ├── evaluate_ablations.sh   # 固定测试集评估
│   ├── evaluate.py             # 准确率、F1、AUC 及校准指标
│   ├── export_all_features.sh  # 导出决策表示
│   ├── export_features.py
│   ├── plot_tsne.py            # 模型级 t-SNE 对比
│   ├── plot_ablation.py        # 消融实验准确率图
│   └── profile_complexity.py   # FLOPs、参数量与延迟
├── EXPERIMENT_REPORT_DUAL_STATE_ZH.md  # 论文结果与逐次运行准确率
└── requirements.txt
```

## 环境

参考环境使用 Python 3.10、支持 CUDA 的 PyTorch，以及 NVIDIA RTX 4090 GPU。

```bash
pip install -r requirements.txt
```

将预训练资源放置在 `src/` 目录下：

```text
src/
├── mae_pretrain_vit_base.pth
├── clip_cn_vit-b-16.pt
└── pretrained_model/
    └── chinese_roberta_wwm_base_ext_pytorch/
        ├── config.json
        ├── pytorch_model.bin
        └── vocab.txt
```

## 数据

本仓库**不提供 Weibo 和 Weibo21 数据的再分发**。这些数据是在获得数据提供方授权的情况下取得的，该授权并未赋予我们再分发权。使用者必须直接向相应作者或权利持有人获取许可及原始数据。

请将各数据集分别放置在独立目录中。数据加载器需要以下文件：

```text
data/
├── weibo/
│   ├── train_2_domain.csv
│   ├── val_2_domain.csv
│   ├── test_2_domain.csv
│   ├── train_loader.pkl
│   ├── val_loader.pkl
│   ├── test_loader.pkl
│   └── *_clip_loader.pkl
└── weibo21/
    ├── train_2_domain.xlsx
    ├── val_2_domain.xlsx
    ├── test_2_domain.xlsx
    ├── train_loader.pkl
    ├── val_loader.pkl
    ├── test_loader.pkl
    └── *_clip_loader.pkl
```

预处理入口为 `src/data_pre.py`、`src/clip_data_pre.py`、`src/weibo21_data_pre.py` 和 `src/weibo21_clip_data_pre.py`。

## 训练

在本地为本次运行选择一个整数随机种子：

```bash
read -r -p "Random seed for this run: " RUN_SEED
export RUN_SEED
```

在 Weibo 上训练完整模型：

```bash
python src/main.py \
  --dataset weibo \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo \
  --output-dir outputs/weibo/full
```

Weibo21 使用相同的模型定义：

```bash
python src/main.py \
  --dataset weibo21 \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo21 \
  --output-dir outputs/weibo21/full
```

复现热启动训练方案时，请使用 `--init-checkpoint PATH`。

## 结构消融

论文报告了完整模型及五种结构消融：

| 变体         | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | 结构设置                                             |
| ---------- | :--: | :-: | :------: | :----------------: | ------------------------------------------------ |
| Full       |   ✓  |  ✓  |     ✓    |          ✓         | 保留独立的分支证据，并使用领域条件化仲裁。                            |
| w/o DRCM   |   ✗  |  ✓  |     ✓    |          ✓         | 保留语义提取与分配；关闭 BEM 的领域输入和领域监督，并将 DCEA 的领域上下文置零。    |
| w/o BEM    |   ✓  |  ✗  |     ✓    |          ✓         | 在各分支及基础注意力中共享语义证据；关闭匹配增强。                        |
| w/o Refine |   ✓  |  ✓  |     ✓    |          ✗         | 保留参考决策 p₀；移除 p_w 与最终融合步骤。                        |
| w/o DCEA   |   ✓  |  ✓  |     ✗    |          ✗         | 对三个分支的概率取均值，同时保留证据建模与领域处理。                       |
| w/o All    |   ✗  |  ✗  |     ✗    |          ✗         | 使用普通 FFN 分类头对经过常规池化/投影的编码器特征进行分类；旁路全部三个组件及其辅助损失。 |

对号表示保留相应组件；叉号表示移除相应组件。对于 w/o DRCM，将领域上下文置零也会使 DCEA 的领域条件化调整失效。

以下命令可运行当前可用的结构设置。

运行当前可用的变体：

```bash
bash scripts/run_ablations.sh
bash scripts/evaluate_ablations.sh
```

无需修改脚本即可覆盖数据集路径：

```bash
WEIBO_ROOT=/path/to/weibo \
WEIBO21_ROOT=/path/to/weibo21 \
bash scripts/run_ablations.sh
```

批处理脚本要求设置 `RUN_SEED`；同时接受 `REPORT_SEED` 作为兼容别名。Python 入口要求提供 `--seed`，且未内置训练随机种子。

## 评估与图表

评估单个检查点：

```bash
python scripts/evaluate.py \
  --dataset weibo \
  --variant full \
  --seed "${RUN_SEED}" \
  --data-root data/weibo \
  --checkpoint outputs/weibo/full/checkpoints/parameter_beda_fnd_best_accuracy.pkl \
  --output-dir outputs/evaluation/weibo/full
```

生成消融实验图：

```bash
python scripts/plot_ablation.py \
  --results-root outputs/evaluation \
  --output-dir outputs/figures
```

当前绘图脚本尚未包含 w/o Refine；完整的论文表格如下所示。

导出决策表示并生成模型级 t-SNE 对比图：

```bash
bash scripts/export_all_features.sh
python scripts/plot_tsne.py \
  --feature-dir outputs/features \
  --output-dir outputs/figures
```

## 参考准确率结果

| 变体         | DRCM | BEM | DCEA: p₀ | DCEA: p_w + Fusion | Weibo 准确率 (%) | Weibo21 准确率 (%) |
| ---------- | :--: | :-: | :------: | :----------------: | ------------: | --------------: |
| Full       |   ✓  |  ✓  |     ✓    |          ✓         |      **95.2** |        **96.0** |
| w/o DRCM   |   ✗  |  ✓  |     ✓    |          ✓         |          94.1 |            94.5 |
| w/o BEM    |   ✓  |  ✗  |     ✓    |          ✓         |          93.7 |            93.9 |
| w/o Refine |   ✓  |  ✓  |     ✓    |          ✗         |          94.4 |            95.0 |
| w/o DCEA   |   ✓  |  ✓  |     ✗    |          ✗         |          93.9 |            94.1 |
| w/o All    |   ✗  |  ✗  |     ✗    |          ✗         |          93.1 |            92.6 |

准确率与论文一致，保留一位小数。

勾号表示保留相应的计算组件。w/o DRCM 保留语义提取与分配，关闭 BEM 的领域输入和领域监督，并将 DCEA 的领域上下文置零，同时使其领域条件化调整失效。

## 致谢

本实现基于 MMDFND 和 DAMMFND 的公开代码库。感谢原作者公开数据集、模型组件及训练流程。使用本仓库时，也请引用相关原始工作。

```bibtex
@inproceedings{lu2025dammfnd,
  title={DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection},
  author={Lu, Weihai and Tong, Yu and Ye, Zhiqiu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={559--567},
  year={2025}
}
```
