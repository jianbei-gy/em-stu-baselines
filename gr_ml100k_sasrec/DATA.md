# MovieLens 100K 数据下载

清华 TUNA 的 RecBole 发布包。以下文件均已免登录下载并校验。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| [recbole-1.2.1-py3-none-any.whl](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/ab/fe/7d606cb7cd2b166a36b100cb9435d21014ceee16c192d972deb0976967a8/recbole-1.2.1-py3-none-any.whl) | 2,130,151 | `9c9948202011f37eb0a7c6768129313f00d6403ad221ec940d5e2d5d5f33a407` |

把 wheel 当作 ZIP 读取。取出 `recbole/dataset_example/ml-100k/ml-100k.inter`，去掉第一行表头，写入 `data/u.data`。该文件与 GroupLens 原始文件的 SHA-256 一致。首次训练或评估时，脚本会从 `data/u.data` 在项目内生成 `data/processed/`。

在项目根目录可执行

```sh
mkdir -p data
unzip -p recbole-1.2.1-py3-none-any.whl recbole/dataset_example/ml-100k/ml-100k.inter | sed '1d' > data/u.data
shasum -a 256 data/u.data
```

`data/u.data` 的预期 SHA-256 为 `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`。
