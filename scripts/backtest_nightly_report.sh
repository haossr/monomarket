#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOOKBACK_HOURS="24"
MARKET_LIMIT="2000"
INGEST_LIMIT="300"
CONFIG_PATH="configs/config.yaml"
NIGHTLY_ROOT="artifacts/backtest/nightly"
NIGHTLY_DATE=""

usage() {
  cat <<'USAGE'
Usage: bash scripts/backtest_nightly_report.sh [options]

Options:
  --lookback-hours <float>   Backtest lookback in hours (default: 24)
  --market-limit <int>       Market limit for signal generation (default: 2000)
  --ingest-limit <int>       Ingest limit for gamma source (default: 300)
  --config <path>            Config path (default: configs/config.yaml)
  --nightly-root <path>      Nightly root dir (default: artifacts/backtest/nightly)
  --date <YYYY-MM-DD>        Override nightly date (default: today local date)
  -h, --help                 Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lookback-hours)
      LOOKBACK_HOURS="$2"
      shift 2
      ;;
    --market-limit)
      MARKET_LIMIT="$2"
      shift 2
      ;;
    --ingest-limit)
      INGEST_LIMIT="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --nightly-root)
      NIGHTLY_ROOT="$2"
      shift 2
      ;;
    --date)
      NIGHTLY_DATE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[nightly] unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -x ".venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[nightly] python interpreter not found (python/python3)" >&2
    exit 1
  fi
fi

if [[ -z "$NIGHTLY_DATE" ]]; then
  NIGHTLY_DATE="$(date +%Y-%m-%d)"
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
NIGHTLY_DIR="${NIGHTLY_ROOT}/${NIGHTLY_DATE}"
RUN_DIR="${NIGHTLY_DIR}/run-${RUN_TS}"
PDF_PATH="${NIGHTLY_DIR}/report.pdf"
SUMMARY_TXT="${NIGHTLY_DIR}/summary.txt"

mkdir -p "$NIGHTLY_DIR"

echo "[nightly] running cycle"
bash scripts/backtest_cycle.sh \
  --lookback-hours "$LOOKBACK_HOURS" \
  --market-limit "$MARKET_LIMIT" \
  --ingest-limit "$INGEST_LIMIT" \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_DIR"

if command -v uv >/dev/null 2>&1; then
  echo "[nightly] render PDF via uv + reportlab"
  uv run --with reportlab "$PYTHON_BIN" scripts/backtest_pdf_report.py \
    --backtest-json "$RUN_DIR/latest.json" \
    --strategy-csv "$RUN_DIR/strategy.csv" \
    --event-csv "$RUN_DIR/event.csv" \
    --output "$PDF_PATH" \
    --title "Monomarket Nightly Backtest Report (${NIGHTLY_DATE})"
else
  echo "[nightly] uv not found, fallback to current python env"
  "$PYTHON_BIN" scripts/backtest_pdf_report.py \
    --backtest-json "$RUN_DIR/latest.json" \
    --strategy-csv "$RUN_DIR/strategy.csv" \
    --event-csv "$RUN_DIR/event.csv" \
    --output "$PDF_PATH" \
    --title "Monomarket Nightly Backtest Report (${NIGHTLY_DATE})"
fi

"$PYTHON_BIN" - "$RUN_DIR/latest.json" "$PDF_PATH" "$SUMMARY_TXT" "$NIGHTLY_DATE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


def _f(raw: object) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


payload = json.loads(Path(sys.argv[1]).read_text())
pdf_path = Path(sys.argv[2]).resolve()
summary_path = Path(sys.argv[3])
nightly_date = sys.argv[4]

rows = payload.get("results") or []
best = None
if isinstance(rows, list) and rows:
    best = max((r for r in rows if isinstance(r, dict)), key=lambda r: _f(r.get("pnl")), default=None)

best_text = "best_strategy=n/a"
if isinstance(best, dict):
    best_text = f"best_strategy={best.get('strategy', '')} pnl={_f(best.get('pnl')):.4f}"

line = (
    f"Nightly {nightly_date} | window={payload.get('from_ts', '')} -> {payload.get('to_ts', '')} "
    f"| signals total={payload.get('total_signals', 0)} executed={payload.get('executed_signals', 0)} "
    f"rejected={payload.get('rejected_signals', 0)} | {best_text} | pdf={pdf_path}"
)
summary_path.write_text(line + "\n")
print(line)
PY

echo "[nightly] done"
echo "- run_dir: $RUN_DIR"
echo "- pdf: $PDF_PATH"
echo "- summary: $SUMMARY_TXT"
