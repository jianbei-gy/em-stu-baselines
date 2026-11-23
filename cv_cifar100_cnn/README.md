# cv_cifar100_cnn

## 背景

本项目做 100 类细粒度自然图像分类，使用官方完整 CIFAR-100：50,000 张训练图、10,000 张官方测试图，标签为 100 个 fine label（脚本不使用 coarse label 做分类）。评价指标是官方测试集 top-1 准确率。它与 CIFAR-10 同分辨率但类别更细，用作更难的视觉起步基线。数据优先 torchvision 官方包，备选 HF `uoft-cs/cifar100`。

## 模型结构

使用 `SmallCNN`（`model/cnn.py`），输入 3×32×32，最后一层改为 Linear(256→100)。特征部分仍为 32-32-64-64-128 的 3×3 卷积 + BN + ReLU 与三次 MaxPool，AdaptiveAvgPool2d(1) 后接 256 维隐层与 Dropout。从零训练，不用预训练 ResNet，以便在 CPU/短时 GPU 上尽快跑完完整官方集。训练增强为 RandomCrop(32, padding=4) 与随机水平翻转。

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

数据路径：项目内 `./data`（官方完整 CIFAR-100，下载到 `./data`）。随包权重在 `model/checkpoints/best.pt`。本地训练也会生成 `model/checkpoints/last.pt`。指标输出：`./results.json`。

## 基线

本机结果（CPU，`results.json`，Asia/Shanghai 2026-11-18 01:51 完成）：官方 test top-1 **0.3338**，5 epoch，耗时 208.33s。命令：`python scripts/train.py --epochs 5 --batch-size 128 --lr 0.001 --seed 42`。模型为从零训练的 **SmallCNN**，短 epoch 起步设定。

**没有与本仓库 SmallCNN 结构/epoch/优化器直接可比的公开发表官方数字。** 仅作不同架构、完整训练的量级参考：He et al. ECCV 2016 *Identity Mappings in Deep Residual Networks* Table 3，CIFAR-100 上 ResNet-164（原 residual unit）test error **25.16%**（约 top-1 74.84%），与本机 SmallCNN 5 epoch 设定**不可直接对比**。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
