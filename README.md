# cv_fashion_mnist_cnn

## 背景

本项目做服饰灰度图分类，使用官方完整 Fashion-MNIST（Zalando Research）：60,000 张训练图、10,000 张测试图，10 类（T-shirt、Trouser、Sneaker 等）。评价指标是官方测试集 top-1 准确率。它与 MNIST 同尺寸，但比数字分类更难，适合作为下一档视觉起步基线。

## 模型结构

与 MNIST 项目相同的 `SmallCNN`（`model/cnn.py`）：五层 3×3 卷积（32-32-64-64-128）+ BN + ReLU，三次 2×2 最大池化，AdaptiveAvgPool2d(1)，再接 Linear(128→256) + ReLU + Dropout(0.3) + Linear(256→10)。输入为 1×28×28，从零训练。

## Python 环境

历史基线使用 Python 3.13.5，依赖版本见 `requirements.txt`。在本项目根目录执行 `bash setup_env.sh` 创建本地环境，再运行下方命令。

## 训练与评估入口

在项目根目录执行（不要在 `scripts/` 里直接跑，以便本地正确解析 `data/` 与 `model/` 路径）。

本机训练与评估：

```bash
.venv/bin/python scripts/train.py --epochs 3 --batch-size 128 --lr 1e-3 --seed 42
.venv/bin/python scripts/eval.py --ckpt model/checkpoints/best.pt
```

Kaggle 的提交配置在 `kernel-metadata.json`。用自己的账号提交前，请更新其中的项目 ID。

```bash
kaggle kernels push -p .
```

视觉模型从 `model/` 导入。在 Kaggle 上运行时，需要让训练入口能导入这个目录。

关键超参数：epochs=3，batch size=128，优化器 Adam，学习率 1e-3，seed=42。设备优先使用 CUDA，否则 CPU。

数据路径：项目内 `./data`（官方 Fashion-MNIST；本机可复用缓存符号链接，否则下载到 `./data`）。随包权重在 `model/checkpoints/best.pt`。本地训练也会生成 `model/checkpoints/last.pt`。指标输出：`./results.json`。

## 基线

历史指标来自 `results.json`。

- 指标：官方测试集 top-1 准确率 **0.9053**（best 同为 0.9053）
- 划分：official test (10,000 images)
- 训练：3 epoch，seed 42，设备 `cpu`
- 耗时：127.43 秒（Asia/Shanghai 完成时间 2026-11-18T01:43:34.476643+08:00）
- 命令：`python scripts/train.py --epochs 3 --batch-size 128 --lr 0.001 --seed 42`
- 模型检查点：`model/checkpoints/best.pt`

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
