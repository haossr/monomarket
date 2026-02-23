#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -x ".venv/bin/python" ]]; then
  current_v="$(.venv/bin/python - <<'PY'
import sys
print(sys.version_info.major * 100 + sys.version_info.minor)
PY
)"
  if [[ "$current_v" -lt 311 ]]; then
    echo "[ci_local] recreate .venv with ${PYTHON_BIN} (current too old)"
    rm -rf .venv
  fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ci_local] create .venv (using ${PYTHON_BIN})"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[ci_local] install editable package + dev deps"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "[ci_local] ruff"
ruff check src tests

echo "[ci_local] black --check"
black --check src tests

echo "[ci_local] mypy"
mypy src/monomarket

echo "[ci_local] pytest"
pytest

if [[ "${1:-}" == "--with-security" ]]; then
  echo "[ci_local] pip-audit"
  pip-audit
  echo "[ci_local] bandit"
  bandit -q -r src/monomarket
fi

echo "[ci_local] done"
