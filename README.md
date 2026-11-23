# ctr_frappe_fm

## 背景

本项目做点击率（CTR）预估，使用独立的官方紧凑基准 **Frappe_x1**。Frappe 是 Baltrunas 等人发布的上下文感知应用使用日志（用户、应用及时间、天气、国家等上下文），标签表示该上下文下是否使用了应用。reczoo/BARS 提供已发布的完整 7:2:1 划分（训练 202,027、验证 57,722、测试 28,860 条），数据来源 https://huggingface.co/datasets/reczoo/Frappe_x1 ，可核对官方 md5。本套件中 FM / DeepFM / DCN 各自使用不同的官方数据集，本项目只使用 Frappe。评价指标为官方测试集上的 **AUC** 与 **LogLoss**。该集仅约 2.8MB，可在 CPU 上数分钟内跑完，远低于 3 小时上限。

## 模型结构

模型为经典 Factorization Machine（`model/fm.py`，并内联在 `scripts/train.py` 中以便 `kaggle kernels push -p .` 只上传入口文件时仍能运行）。每个特征域经哈希后查表得到 16 维 embedding。输出为：全局偏置 + 一阶线性项（Embedding→1 再求和）+ 二阶交互项，交互用平方和公式 0.5*((Σ v_i)^2 − Σ v_i^2) 计算。训练损失为 BCE with logits。

## Python 环境

历史基线使用 Python 3.13.5，依赖版本见 `requirements.txt`。在本项目根目录执行 `bash setup_env.sh` 创建本地环境，再运行下方命令。

## 训练与评估入口

在项目根目录执行。

本机训练与评估（设计为在本地 CPU 上 3 小时内完成）：

```bash
.venv/bin/python scripts/train.py
.venv/bin/python scripts/eval.py
```

提交到 Kaggle 并启动训练（可直接执行的提交入口，不是口头说明）。项目根目录已包含 `kernel-metadata.json`，`code_file` 为自包含的 `scripts/train.py`：

```bash
kaggle kernels push -p .
```

默认超参数：6 epoch，batch 1024，Adam 1e-3，embed_dim=16，seed=42，设备优先 CUDA 否则 CPU。Kernel 默认关闭 GPU（表格 CTR 在 CPU 上已足够快）。

数据目录：`./data`。检查点：`./model/checkpoints/best.pt`。指标：`./results.json`。

## 基线

本机结果（CPU，`results.json`，Asia/Shanghai 2026-11-18 11:51 完成）：官方 test **AUC 0.9330**、**LogLoss 0.2883**，6 epoch，脚本记录的经过时间 **22.38s**。命令：`python scripts/train.py --epochs 6 --batch-size 1024 --lr 0.001 --seed 42`。划分是已发布的 Frappe_x1 官方 test（28,860 条）。`metrics.auc` 与 `best_auc` 均为 0.9330。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
