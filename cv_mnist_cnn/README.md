# cv_mnist_cnn

## 背景

本项目做手写数字图像分类，使用官方完整 MNIST（LeCun et al.）：60,000 张训练图、10,000 张测试图，10 类。评价指标是官方测试集 top-1 准确率。作为新手起步套件，MNIST 体积小、协议清晰，适合验证训练/评估入口和目录约定，而不是追求最新精度。

## 模型结构

模型为自研小型 CNN（`model/cnn.py` 中的 `SmallCNN`），输入 1×28×28。特征堆叠为：32 通道 3×3 卷积 + BN + ReLU 两层，MaxPool(2)；64 通道同样两层卷积后 MaxPool(2)；128 通道一层卷积后再 MaxPool(2)；AdaptiveAvgPool2d(1) 将空间维压成 128 维向量；分类头为 Linear(128→256) + ReLU + Dropout(0.3) + Linear(256→10)。不使用预训练权重。

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

数据路径：项目内 `./data`（官方 MNIST；本机可复用缓存符号链接，缺失时由脚本下载到 `./data`）。权重路径：`./model/checkpoints/best.pt` 与 `last.pt`。指标输出：`./results.json`。

## 基线

历史指标来自 `results.json`。

- 指标：官方测试集 top-1 准确率 **0.9911**（best 同为 0.9911）
- 划分：official test (10,000 images)
- 训练：3 epoch，seed 42，设备 `cpu`
- 耗时：71.59 秒（Asia/Shanghai 完成时间 2026-11-18T01:33:49.833883+08:00）
- 命令：`python train.py --epochs 3 --batch-size 128 --lr 0.001 --seed 42`
- 训练完成后生成 `model/checkpoints/best.pt`。此项目仅随附历史指标。

## 本地交付包说明

注意：源项目中没有 `model/checkpoints/`（训练只写入了 `results.json`，未在仓库中留下 best/last 权重）。本包因此不含检查点，无法直接用已有权重跑 `scripts/eval.py`。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
