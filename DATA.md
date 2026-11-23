# CIFAR-100 数据下载

PaddlePaddle 北京镜像。以下文件均已免登录下载并校验。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| [cifar-100-python.tar.gz](https://dataset.bj.bcebos.com/cifar/cifar-100-python.tar.gz) | 169,001,437 | `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` |

将 tar.gz 解压到 `data/`，保留 `data/cifar-100-python/`。使用本地数据训练时设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`，训练入口可通过 torchvision 读取这份 Python 版数据。
