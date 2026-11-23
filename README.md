# gr_ml100k_sasrec

## 背景

本项目做序列推荐，使用官方 **GroupLens MovieLens 100K**（`u.data`，100,000 条评分）。来源 https://files.grouplens.org/datasets/movielens/ml-100k.zip 。按约定，本项目只使用这一份数据，不与 Amazon Beauty 共用。所有评分视为隐式反馈，按时间排序后做 leave-one-out：每个用户最后一条作测试、倒数第二条作验证，至少 3 条交互的用户才保留。评价指标为测试集上的全排序 **HR@10** 与 **NDCG@10**（排除该用户训练+验证中出现过的物品）。

## 模型结构

模型为因果 SASRec（`model/sasrec.py`，并内联在 `scripts/train.py` 中，便于 `kaggle kernels push -p .`）。默认 hidden 64、max len 50、2 层、2 头。训练用因果自注意力，对每个时间步做下一物品的全词表交叉熵。

## Python 环境

历史基线使用 Python 3.13.5，依赖版本见 `requirements.txt`。在本项目根目录执行 `bash setup_env.sh` 创建本地环境，再运行下方命令。

## 训练与评估入口

在项目根目录执行。

本机训练与评估（设计为在本地 CPU 上 3 小时内完成）：

```bash
.venv/bin/python scripts/train.py
.venv/bin/python scripts/eval.py
```

提交到 Kaggle：

```bash
kaggle kernels push -p .
```

默认超参数：30 epoch（验证 NDCG@10 早停 patience=5，且 `--max-seconds 10000`），batch 128，Adam 1e-3，hidden 64，max len 50，seed=42，设备优先 CUDA 否则 CPU。为避免抢占同机其他任务，默认 `torch.set_num_threads(2)`。

数据目录：`./data`。检查点：`./model/checkpoints/best.pt`。指标：`./results.json`。

## 基线

2026-11-20 用修正后的注意力掩码复评随包的 `best.pt`，在 943 个测试用户上得到 **HR@10 0.1368、NDCG@10 0.0713**。下方与 `results.json` 保留历史记录。旧评估代码可能将非有限分数计为第 1 名，比较新实验时请使用复评值。

历史指标来自 `results.json`。

- 指标：leave-one-out 全排序 **HR@10 0.4740**，**NDCG@10 0.4410**（精确值 0.47401908909149415 / 0.44096842986528595）
- 划分：GroupLens MovieLens 100K `u.data`，全部评分视为隐式反馈，按时间 leave-one-out（最后一条测试、倒数第二条验证），全目录排序并排除该用户训练+验证中出现过的物品；943 用户、1682 物品
- 训练：30 epoch（验证集最后一轮 NDCG@10 最好，未触发早停），hidden 64，max len 50，2 层 2 头，batch 128，Adam 1e-3，seed 42，设备 `cpu`
- 耗时：45.97 秒（Asia/Shanghai 完成时间 2026-11-18T02:03:46.022145+08:00）
- 命令：`python scripts/train.py --epochs 30 --batch-size 128 --lr 0.001 --hidden 64 --max-len 50 --seed 42`
- 模型检查点：`model/checkpoints/best.pt`

### 与公开论文数字的对比

**没有直接可比的官方值。** Kang & McAuley, *Self-Attentive Sequential Recommendation*（ICDM 2018，[arXiv:1808.09781](https://arxiv.org/abs/1808.09781)，[会议 PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)）没有报告 MovieLens 100K。原文 Table III 中 SASRec 的数字是：

- MovieLens-1M：Hit@10 **0.8245**，NDCG@10 **0.5905**
- Amazon Beauty：Hit@10 **0.4854**，NDCG@10 **0.3219**

这与本项目设定对不上，不能当成同设定对照，只作为出处备查。主要差异：

1. 数据不是同一份：原文用的是 MovieLens-1M（6,040 用户 / 3,416 物品）和另一份 Amazon Beauty 预处理结果（52,024 用户 / 57,289 物品），不是本项目的 MovieLens 100K。
2. 评估协议不同：原文对每个用户将目标物品与 **100 个随机负样本** 一起排序；本项目是 **全目录排序**，全排序数字通常会明显更低。
3. 模型规模不同：本项目固定 hidden 64、max len 50、2 层，属于起步小配置，不是原文调好的规模。

未找到与本项目相同数据（ML-100K）且同为时间 leave-one-out 全排序的 SASRec 官方或广泛引用复现数字，故不填写“官方基线”一列。

## 本地交付包说明

源项目检查点目录只有 `best.pt`，没有 `last.pt`。

本匿名版本仅提供源码和指标。请先训练生成本地权重，再运行评估。
