#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
# 为本项目创建本地 .venv 并安装 requirements.txt。
# 用法：在项目根目录执行 bash setup_env.sh（本脚本不会自动运行）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "creating $ROOT/.venv"
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

echo "Python env ready: $ROOT/.venv/bin/python"
.venv/bin/python -c "import sys; print(sys.version)"
