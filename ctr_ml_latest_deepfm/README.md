# ctr_ml_latest_deepfm

## 背景

本项目做 CTR 预估，使用独立的官方紧凑基准 **MovielensLatest_x1**（BARS 发布的完整 7:2:1 划分，约 140 万 / 40 万 / 20 万条）。它把 GroupLens MovieLens Latest 的打标记录做成个性化标签推荐/CTR 任务，特征为 user_id、item_id、tag_id，标签表示用户是否给该电影打了该标签。来源 https://huggingface.co/datasets/reczoo/MovielensLatest_x1 ，可核对官方 md5。这与 Frappe、iPinYou 以及本套件中的 MovieLens 100K 序列推荐都不是同一份数据。评价指标为官方测试集 **AUC** 与 **LogLoss**。规模中等，CPU 上可在 3 小时内完成。

## 模型结构

模型为 DeepFM（`model/deepfm.py`，并内联在 `scripts/train.py`）。共享的域 embedding（16 维）同时供给：一阶线性项、FM 二阶交互（平方和公式），以及两层 MLP（128→64，ReLU 与 Dropout 0.2）对拼接后的 embedding 建模高阶交互。三部分 logit 相加，BCE with logits 训练。

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

默认超参数：5 epoch，batch 2048，Adam 1e-3，embed_dim=16，MLP 128-64，seed=42，CPU/CUDA 自适应。Kaggle Kernel 默认关闭 GPU。

数据目录：`./data`。检查点：`./model/checkpoints/best.pt`。指标：`./results.json`。

## 基线

本机结果（CPU，`results.json`，Asia/Shanghai 2026-11-18 11:53 完成）：官方 test **AUC 0.9298**、**LogLoss 0.3277**，5 epoch，脚本记录的经过时间 **71.71s**。命令：`python scripts/train.py --epochs 5 --batch-size 2048 --lr 0.001 --seed 42`。划分是已发布的 MovielensLatest_x1 官方 test（200,686 条）。`metrics.auc` 与 `best_auc` 均为 0.9298。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
