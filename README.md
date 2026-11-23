# 机器学习基线项目

7 个可在本地运行的示例，涵盖图像分类、点击率预测和序列推荐。每个分支提供训练脚本、评估脚本和基线指标。

| 数据集 | 项目分支 |
| --- | --- |
| MNIST | [cv_mnist_cnn](./cv_mnist_cnn/README.md) |
| Fashion-MNIST | [cv_fashion_mnist_cnn](./cv_fashion_mnist_cnn/README.md) |
| CIFAR-10 | [cv_cifar10_resnet](./cv_cifar10_resnet/README.md) |
| CIFAR-100 | [cv_cifar100_cnn](./cv_cifar100_cnn/README.md) |
| Frappe_x1 | [ctr_frappe_fm](./ctr_frappe_fm/README.md) |
| MovieLens Latest (x1) | [ctr_ml_latest_deepfm](./ctr_ml_latest_deepfm/README.md) |
| MovieLens 100K | [gr_ml100k_sasrec](./gr_ml100k_sasrec/README.md) |

## 快速开始

匿名评审页面可直接打开表中的项目目录。在 Git 仓库中，在仓库根目录运行 `git switch cv_mnist_cnn` 选择 MNIST，或换成表中的其他分支。按所选分支的 `DATA.md` 下载并解压数据，再按分支说明运行训练。此匿名源码包仅提供代码和指标；请先训练生成本地权重，再运行评估。

[所有数据集的下载地址与校验值](./downloads.json)
