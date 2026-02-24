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
ROLLING_WINDOW_HOURS="24"
ROLLING_STEP_HOURS="12"
ROLLING_REJECT_TOP_K="2"

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
  --rolling-window-hours <float>  Rolling window size in hours (default: 24)
  --rolling-step-hours <float>    Rolling step size in hours (default: 12)
  --rolling-reject-top-k <int>    Number of top rolling reject reasons in summary (default: 2)
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
    --rolling-window-hours)
      ROLLING_WINDOW_HOURS="$2"
      shift 2
      ;;
    --rolling-step-hours)
      ROLLING_STEP_HOURS="$2"
      shift 2
      ;;
    --rolling-reject-top-k)
      ROLLING_REJECT_TOP_K="$2"
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
ROLLING_JSON="${NIGHTLY_DIR}/rolling-summary.json"

mkdir -p "$NIGHTLY_DIR"

echo "[nightly] running cycle"
bash scripts/backtest_cycle.sh \
  --lookback-hours "$LOOKBACK_HOURS" \
  --market-limit "$MARKET_LIMIT" \
  --ingest-limit "$INGEST_LIMIT" \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_DIR"

ROLLING_FROM_TO="$($PYTHON_BIN - "$LOOKBACK_HOURS" <<'PY'
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys

lookback_hours = float(sys.argv[1])
now = datetime.now(UTC).replace(microsecond=0)
from_ts = now - timedelta(hours=lookback_hours)
print(from_ts.isoformat().replace("+00:00", "Z"))
print(now.isoformat().replace("+00:00", "Z"))
PY
)"
ROLLING_FROM_TS="$(echo "$ROLLING_FROM_TO" | sed -n '1p')"
ROLLING_TO_TS="$(echo "$ROLLING_FROM_TO" | sed -n '2p')"

echo "[nightly] rolling backtest window=${ROLLING_FROM_TS} -> ${ROLLING_TO_TS}"
monomarket backtest-rolling \
  --strategies "s1,s2,s4,s8" \
  --from "$ROLLING_FROM_TS" \
  --to "$ROLLING_TO_TS" \
  --window-hours "$ROLLING_WINDOW_HOURS" \
  --step-hours "$ROLLING_STEP_HOURS" \
  --out-json "$ROLLING_JSON" \
  --config "$CONFIG_PATH"

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

"$PYTHON_BIN" - "$RUN_DIR/latest.json" "$PDF_PATH" "$ROLLING_JSON" "$SUMMARY_TXT" "$NIGHTLY_DATE" "$ROLLING_REJECT_TOP_K" <<'PY'
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
rolling_path = Path(sys.argv[3]).resolve()
summary_path = Path(sys.argv[4])
nightly_date = sys.argv[5]
rolling_reject_top_k = max(0, int(_f(sys.argv[6])))

rows = payload.get("results") or []
best = None
if isinstance(rows, list) and rows:
    best = max((r for r in rows if isinstance(r, dict)), key=lambda r: _f(r.get("pnl")), default=None)

best_text = "best_strategy=n/a"
if isinstance(best, dict):
    best_text = f"best_strategy={best.get('strategy', '')} pnl={_f(best.get('pnl')):.4f}"

rolling_runs = 0
rolling_exec_rate = 0.0
rolling_range_hours = 0.0
rolling_coverage_ratio = 0.0
rolling_overlap_ratio = 0.0
rolling_coverage_label = "unknown"
rolling_positive_window_rate = 0.0
rolling_empty_window_count = 0
rolling_reject_top = "none"
if rolling_path.exists():
    rolling_payload = json.loads(rolling_path.read_text())
    rolling_summary = rolling_payload.get("summary")
    if isinstance(rolling_summary, dict):
        rolling_runs = int(_f(rolling_summary.get("run_count")))
        rolling_exec_rate = _f(rolling_summary.get("execution_rate"))
        rolling_range_hours = _f(rolling_summary.get("range_hours"))
        rolling_coverage_ratio = _f(rolling_summary.get("coverage_ratio"))
        rolling_overlap_ratio = _f(rolling_summary.get("overlap_ratio"))
        rolling_positive_window_rate = _f(rolling_summary.get("positive_window_rate"))
        rolling_empty_window_count = int(_f(rolling_summary.get("empty_window_count")))
        coverage_label_raw = rolling_summary.get("coverage_label")
        if isinstance(coverage_label_raw, str) and coverage_label_raw.strip():
            rolling_coverage_label = coverage_label_raw.strip()

        raw_reasons = rolling_summary.get("risk_rejection_reasons")
        if isinstance(raw_reasons, dict):
            reason_items: list[tuple[str, int]] = []
            for key, value in raw_reasons.items():
                reason = str(key).strip() or "unknown"
                count = int(_f(value))
                if count <= 0:
                    continue
                reason_items.append((reason, count))
            if reason_items and rolling_reject_top_k > 0:
                reason_items.sort(key=lambda x: (-x[1], x[0]))
                rolling_reject_top = ",".join(
                    f"{reason}:{count}" for reason, count in reason_items[:rolling_reject_top_k]
                )

line = (
    f"Nightly {nightly_date} | window={payload.get('from_ts', '')} -> {payload.get('to_ts', '')} "
    f"| signals total={payload.get('total_signals', 0)} executed={payload.get('executed_signals', 0)} "
    f"rejected={payload.get('rejected_signals', 0)} | {best_text} "
    f"| rolling runs={rolling_runs} exec_rate={rolling_exec_rate:.2%} "
    f"pos_win_rate={rolling_positive_window_rate:.2%} empty_windows={rolling_empty_window_count} "
    f"range_h={rolling_range_hours:.2f} coverage={rolling_coverage_ratio:.2%} "
    f"overlap={rolling_overlap_ratio:.2%} coverage_label={rolling_coverage_label} "
    f"rolling_reject_top={rolling_reject_top} "
    f"| pdf={pdf_path} | rolling_json={rolling_path}"
)
summary_path.write_text(line + "\n")
print(line)
PY

echo "[nightly] done"
echo "- run_dir: $RUN_DIR"
echo "- rolling_json: $ROLLING_JSON"
echo "- pdf: $PDF_PATH"
echo "- summary: $SUMMARY_TXT"
