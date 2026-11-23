# CIFAR-10 数据下载

PaddlePaddle 北京镜像。以下文件均已免登录下载并校验。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| [cifar-10-python.tar.gz](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz) | 170,498,071 | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |

将 tar.gz 解压到 `data/`，保留 `data/cifar-10-batches-py/`。使用本地数据训练时设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`，训练入口可通过 torchvision 读取这份 Python 版数据。
