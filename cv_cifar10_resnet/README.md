# cv_cifar10_resnet

## 背景

本项目做自然图像 10 分类，使用官方完整 CIFAR-10（University of Toronto，Krizhevsky）：50,000 张 32×32 训练图、10,000 张官方测试图。评价指标是官方测试集 top-1 准确率。相比 MNIST，它需要颜色与更强的特征提取，是常用的起步 CNN/ResNet 基准。脚本优先从 torchvision 下载官方 tar；若源站过慢，则改用 Hugging Face 上的官方镜像 `uoft-cs/cifar10`（同样是完整 50k/10k 划分），并在结果里记录实际数据来源。

## 模型结构

模型为面向 32×32 输入改过 stem 的 ResNet-18（`model/resnet.py` 中 `cifar_resnet18`）：把 ImageNet 版 7×7 stride-2 conv1 换成 3×3 stride-1、padding-1、无 bias 的卷积，并去掉第一个 3×3 stride-2 max-pool，以免过早下采样。之后仍是标准 4 个 stage、每个 stage 两个 BasicBlock，通道 64-128-256-512，全局平均池化后接 Linear(512→10)。从零训练，不加载 ImageNet 预训练。训练时使用 RandomCrop(32, padding=4) 与随机水平翻转。

## Python 环境

历史基线使用 Python 3.13.5，依赖版本见 `requirements.txt`。在本项目根目录执行 `bash setup_env.sh` 创建本地环境，再运行下方命令。

## 训练与评估入口

在项目根目录执行（不要在 `scripts/` 里直接跑，以便本地正确解析 `data/` 与 `model/` 路径）。

本机训练与评估：

```bash
export HF_HOME="$PWD/data/hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
.venv/bin/python scripts/train.py --epochs 5 --batch-size 128 --lr 1e-3 --seed 42
.venv/bin/python scripts/eval.py --ckpt model/checkpoints/best.pt
```

Kaggle 的提交配置在 `kernel-metadata.json`。用自己的账号提交前，请更新其中的项目 ID。

```bash
kaggle kernels push -p .
```

视觉模型从 `model/` 导入。在 Kaggle 上运行时，需要让训练入口能导入这个目录。

关键超参数：epochs=5，batch size=128，优化器 Adam，学习率 1e-3，seed=42。设备优先使用 CUDA，否则 CPU。

数据路径：项目内 `./data`（官方完整 CIFAR-10，下载到 `./data`；失败时使用 HF `uoft-cs/cifar10` 镜像）。随包权重在 `model/checkpoints/best.pt`。本地训练也会生成 `model/checkpoints/last.pt`。指标输出：`./results.json`。

## 基线

本机结果（CPU，`results.json`，Asia/Shanghai 2026-11-18 02:10 完成）：官方 test top-1 **0.7451**，3 epoch，耗时 1106.25s。命令：`python scripts/train.py --epochs 3 --batch-size 128 --lr 0.001 --seed 42`。CIFAR stem ResNet-18（3×3 stride-1、无首层 max-pool），短 epoch 起步设定，不是论文完整配方复现。

可引用的完整训练参考（设定不同，**不可直接对比**）：OpenMMLab MMClassification 文档中 CIFAR-10 上 ResNet-18-b16x8 报告 top-1 **94.82%**（https://mmclassification.readthedocs.io/en/stable/papers/resnet.html）。He et al. CVPR 2016 原文在 CIFAR-10 上报告的是更深的 CIFAR ResNet（如 ResNet-110 test error 6.43%，即约 93.57%），**未单独给出本仓库这种 ResNet-18 CIFAR stem + 3 epoch Adam 的官方数字**。因此本机 0.7451 只作为 starter 基线，不宣称对齐上述完整训练结果。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
