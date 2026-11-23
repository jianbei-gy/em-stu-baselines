#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and LICENSING.md in the project root.
# 为本项目创建本地 .venv 并安装 requirements.txt 中锁定的依赖。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3.13}"
VENV="$ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "creating $VENV with $PY"
  "$PY" -m venv "$VENV"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements.txt"
else
  if ! "$VENV/bin/python" -c "import pip" >/dev/null 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade
  fi
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

echo "Python env ready: $VENV/bin/python"
"$VENV/bin/python" -c "import sys; print(sys.version)"
