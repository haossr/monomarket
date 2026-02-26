#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOOKBACK_HOURS="4380"
MARKET_LIMIT="2000"
INGEST_LIMIT="300"
CONFIG_PATH="configs/config.yaml"
NIGHTLY_ROOT="artifacts/backtest/nightly"
NIGHTLY_DATE=""
ROLLING_WINDOW_HOURS="24"
ROLLING_STEP_HOURS="12"
ROLLING_REJECT_TOP_K="2"
NIGHTLY_SUMMARY_CHECKSUM="1"
FROM_TS=""
TO_TS=""
CLEAR_SIGNALS_WINDOW="0"
REBUILD_SIGNALS_WINDOW="0"
REBUILD_STEP_HOURS="12"
REQUIRE_INTERPRETABLE="0"

usage() {
  cat <<'USAGE'
Usage: bash scripts/backtest_nightly_report.sh [options]

Options:
  --lookback-hours <float>   Backtest lookback in hours (default: 4380)
  --market-limit <int>       Market limit for signal generation (default: 2000)
  --ingest-limit <int>       Ingest limit for gamma source (default: 300)
  --config <path>            Config path (default: configs/config.yaml)
  --nightly-root <path>      Nightly root dir (default: artifacts/backtest/nightly)
  --date <YYYY-MM-DD>        Override nightly date (default: today local date)
  --rolling-window-hours <float>  Rolling window size in hours (default: 24)
  --rolling-step-hours <float>    Rolling step size in hours (default: 12)
  --rolling-reject-top-k <int>    Number of top rolling reject reasons in summary (default: 2; 0=disabled)
  --from-ts <ISO8601>             Optional fixed backtest window start (requires --to-ts)
  --to-ts <ISO8601>               Optional fixed backtest window end (requires --from-ts)
  --clear-signals-window          Delete existing signals in [from_ts,to_ts] before generate-signals
                                  (safety: fixed-window mode only)
  --rebuild-signals-window        Rebuild signals across window from market_snapshots
                                  (requires --clear-signals-window)
  --rebuild-step-hours <f>        Step hours for rebuild-signals-window (default: 12)
  --require-interpretable         Fail if summary marks experiment_interpretable=false
  --no-checksum              Disable checksum fields in nightly summary.json sidecar
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
    --from-ts)
      FROM_TS="$2"
      shift 2
      ;;
    --to-ts)
      TO_TS="$2"
      shift 2
      ;;
    --clear-signals-window)
      CLEAR_SIGNALS_WINDOW="1"
      shift 1
      ;;
    --rebuild-signals-window)
      REBUILD_SIGNALS_WINDOW="1"
      shift 1
      ;;
    --rebuild-step-hours)
      REBUILD_STEP_HOURS="$2"
      shift 2
      ;;
    --require-interpretable)
      REQUIRE_INTERPRETABLE="1"
      shift 1
      ;;
    --no-checksum)
      NIGHTLY_SUMMARY_CHECKSUM="0"
      shift 1
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

if [[ -n "$FROM_TS" || -n "$TO_TS" ]]; then
  if [[ -z "$FROM_TS" || -z "$TO_TS" ]]; then
    echo "[nightly] --from-ts and --to-ts must be provided together" >&2
    exit 1
  fi
fi

if [[ "$CLEAR_SIGNALS_WINDOW" == "1" && ( -z "$FROM_TS" || -z "$TO_TS" ) ]]; then
  echo "[nightly] --clear-signals-window requires --from-ts and --to-ts" >&2
  exit 1
fi

if [[ "$REBUILD_SIGNALS_WINDOW" == "1" && "$CLEAR_SIGNALS_WINDOW" != "1" ]]; then
  echo "[nightly] --rebuild-signals-window requires --clear-signals-window" >&2
  exit 1
fi

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
SUMMARY_JSON="${NIGHTLY_DIR}/summary.json"
ROLLING_JSON="${NIGHTLY_DIR}/rolling-summary.json"

mkdir -p "$NIGHTLY_DIR"

CYCLE_WINDOW_ARGS=()
if [[ -n "$FROM_TS" && -n "$TO_TS" ]]; then
  CYCLE_WINDOW_ARGS=(--from-ts "$FROM_TS" --to-ts "$TO_TS")
fi

CYCLE_CLEAR_ARGS=()
if [[ "$CLEAR_SIGNALS_WINDOW" == "1" ]]; then
  CYCLE_CLEAR_ARGS=(--clear-signals-window)
fi

CYCLE_REBUILD_ARGS=()
if [[ "$REBUILD_SIGNALS_WINDOW" == "1" ]]; then
  CYCLE_REBUILD_ARGS=(--rebuild-signals-window --rebuild-step-hours "$REBUILD_STEP_HOURS")
fi

echo "[nightly] running cycle"
bash scripts/backtest_cycle.sh \
  --lookback-hours "$LOOKBACK_HOURS" \
  --market-limit "$MARKET_LIMIT" \
  --ingest-limit "$INGEST_LIMIT" \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_DIR" \
  ${CYCLE_WINDOW_ARGS[@]-} \
  ${CYCLE_CLEAR_ARGS[@]-} \
  ${CYCLE_REBUILD_ARGS[@]-}

ROLLING_FROM_TO="$($PYTHON_BIN - "$RUN_DIR/latest.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(str(payload.get("from_ts", "")))
print(str(payload.get("to_ts", "")))
PY
)"
ROLLING_FROM_TS="$(echo "$ROLLING_FROM_TO" | sed -n '1p')"
ROLLING_TO_TS="$(echo "$ROLLING_FROM_TO" | sed -n '2p')"

if [[ -z "$ROLLING_FROM_TS" || -z "$ROLLING_TO_TS" ]]; then
  echo "[nightly] failed to derive rolling window from $RUN_DIR/latest.json" >&2
  exit 1
fi

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
    --rolling-json "$ROLLING_JSON" \
    --output "$PDF_PATH" \
    --title "Monomarket Nightly Backtest Report (${NIGHTLY_DATE})"
else
  echo "[nightly] uv not found, fallback to current python env"
  "$PYTHON_BIN" scripts/backtest_pdf_report.py \
    --backtest-json "$RUN_DIR/latest.json" \
    --strategy-csv "$RUN_DIR/strategy.csv" \
    --event-csv "$RUN_DIR/event.csv" \
    --rolling-json "$ROLLING_JSON" \
    --output "$PDF_PATH" \
    --title "Monomarket Nightly Backtest Report (${NIGHTLY_DATE})"
fi

SUMMARY_CHECKSUM_ARGS=()
if [[ "$NIGHTLY_SUMMARY_CHECKSUM" == "1" ]]; then
  SUMMARY_CHECKSUM_ARGS=(--with-checksum)
fi

"$PYTHON_BIN" scripts/nightly_summary_line.py \
  --backtest-json "$RUN_DIR/latest.json" \
  --pdf-path "$PDF_PATH" \
  --rolling-json "$ROLLING_JSON" \
  --cycle-meta-json "$RUN_DIR/cycle-meta.json" \
  --summary-path "$SUMMARY_TXT" \
  --summary-json-path "$SUMMARY_JSON" \
  --nightly-date "$NIGHTLY_DATE" \
  --rolling-reject-top-k "$ROLLING_REJECT_TOP_K" \
  "${SUMMARY_CHECKSUM_ARGS[@]}"

if [[ "$REQUIRE_INTERPRETABLE" == "1" ]]; then
  "$PYTHON_BIN" - "$SUMMARY_JSON" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
cycle_meta = payload.get("cycle_meta")
signal_generation = cycle_meta.get("signal_generation") if isinstance(cycle_meta, dict) else None
if not isinstance(signal_generation, dict):
    raise SystemExit("[nightly] require-interpretable failed: missing cycle_meta.signal_generation")

ok = bool(signal_generation.get("experiment_interpretable", False))
reason = str(signal_generation.get("experiment_reason", "unknown"))
if not ok:
    raise SystemExit(
        "[nightly] require-interpretable failed: "
        f"experiment_interpretable=false (reason={reason})"
    )
print(f"[nightly] require-interpretable passed (reason={reason})")
PY
fi

echo "[nightly] done"
echo "- run_dir: $RUN_DIR"
echo "- rolling_json: $ROLLING_JSON"
echo "- pdf: $PDF_PATH"
echo "- summary: $SUMMARY_TXT"
echo "- summary_json: $SUMMARY_JSON"
