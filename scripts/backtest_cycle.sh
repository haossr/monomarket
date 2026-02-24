#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOOKBACK_HOURS="24"
MARKET_LIMIT="2000"
INGEST_LIMIT="300"
CONFIG_PATH="configs/config.yaml"
OUTPUT_DIR=""
STRATEGIES="s1,s2,s4,s8"

usage() {
  cat <<'USAGE'
Usage: bash scripts/backtest_cycle.sh [options]

Run one reusable backtest cycle:
  init-db -> ingest(gamma, incremental) -> generate-signals(S1,S2,S4,S8) -> backtest

Options:
  --lookback-hours <float>   Lookback window in hours (default: 24)
  --market-limit <int>       Market limit for generate-signals (default: 2000)
  --ingest-limit <int>       Ingest limit for gamma source (default: 300)
  --config <path>            Config path (default: configs/config.yaml)
  --output-dir <path>        Output run directory (default: artifacts/backtest/runs/<timestamp>)
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[backtest-cycle] unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -x ".venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v monomarket >/dev/null 2>&1; then
  echo "[backtest-cycle] monomarket command not found. Run: pip install -e '.[dev]'" >&2
  exit 1
fi

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[backtest-cycle] python interpreter not found (python/python3)" >&2
    exit 1
  fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="artifacts/backtest/runs/${RUN_TS}"
else
  RUN_DIR="$OUTPUT_DIR"
fi

mkdir -p "$RUN_DIR"
mkdir -p artifacts/backtest

_compute_window() {
  "$PYTHON_BIN" - "$LOOKBACK_HOURS" <<'PY'
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys

lookback_hours = float(sys.argv[1])
now = datetime.now(UTC).replace(microsecond=0)
from_ts = now - timedelta(hours=lookback_hours)
print(from_ts.isoformat().replace("+00:00", "Z"))
print(now.isoformat().replace("+00:00", "Z"))
PY
}

_now_iso() {
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from datetime import UTC, datetime

print(datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
PY
}

_max_iso() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
from __future__ import annotations

from datetime import UTC, datetime
import sys


def parse(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


a = parse(sys.argv[1])
b = parse(sys.argv[2])
print((a if a >= b else b).isoformat().replace("+00:00", "Z"))
PY
}

_add_seconds_iso() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys

text = sys.argv[1].strip()
if text.endswith("Z"):
    text = text[:-1] + "+00:00"
dt = datetime.fromisoformat(text)
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=UTC)
seconds = int(float(sys.argv[2]))
print((dt.astimezone(UTC) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"))
PY
}

export ENABLE_LIVE_TRADING=false
export MONOMARKET_MODE=paper

echo "[backtest-cycle] run_dir=$RUN_DIR"

echo "[backtest-cycle] init-db"
monomarket init-db --config "$CONFIG_PATH"

echo "[backtest-cycle] ingest gamma incremental"
monomarket ingest --source gamma --limit "$INGEST_LIMIT" --incremental --config "$CONFIG_PATH"

# Isolate backtest to signals generated in THIS cycle to avoid historical signal contamination.
CYCLE_START_TS="$(_now_iso)"

echo "[backtest-cycle] generate signals: $STRATEGIES"
monomarket generate-signals \
  --strategies "$STRATEGIES" \
  --market-limit "$MARKET_LIMIT" \
  --config "$CONFIG_PATH"

FROM_TO="$(_compute_window)"
LOOKBACK_FROM_TS="$(echo "$FROM_TO" | sed -n '1p')"
FROM_TS="$(_max_iso "$LOOKBACK_FROM_TS" "$CYCLE_START_TS")"
TO_TS="$(_now_iso)"
if [[ "$TO_TS" == "$FROM_TS" ]]; then
  TO_TS="$(_add_seconds_iso "$FROM_TS" 60)"
fi

echo "[backtest-cycle] window=${FROM_TS} -> ${TO_TS} (cycle_start=${CYCLE_START_TS})"

echo "[backtest-cycle] backtest"
monomarket backtest \
  --strategies "$STRATEGIES" \
  --from "$FROM_TS" \
  --to "$TO_TS" \
  --replay-limit 0 \
  --out-json "$RUN_DIR/latest.json" \
  --out-replay-csv "$RUN_DIR/replay.csv" \
  --out-strategy-csv "$RUN_DIR/strategy.csv" \
  --out-event-csv "$RUN_DIR/event.csv" \
  --config "$CONFIG_PATH"

"$PYTHON_BIN" - "$RUN_DIR/latest.json" "$RUN_DIR/summary.md" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


def _f(raw: object) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


in_json = Path(sys.argv[1])
out_md = Path(sys.argv[2])
payload = json.loads(in_json.read_text())
rows = payload.get("results") or []

lines = [
    "# Backtest Cycle Summary",
    "",
    f"- generated_at: {payload.get('generated_at', '')}",
    f"- window: {payload.get('from_ts', '')} -> {payload.get('to_ts', '')}",
    (
        "- signals: "
        f"total={payload.get('total_signals', 0)} "
        f"executed={payload.get('executed_signals', 0)} "
        f"rejected={payload.get('rejected_signals', 0)}"
    ),
    f"- replay_rows: {len(payload.get('replay') or [])}",
    "",
    "## Strategy Metrics",
    "",
    "| strategy | pnl | winrate | max_drawdown | trades | wins | losses |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    winrate = _f(row.get("winrate")) * 100.0
    lines.append(
        "| "
        + f"{row.get('strategy', '')} "
        + f"| {_f(row.get('pnl')):.4f} "
        + f"| {winrate:.2f}% "
        + f"| {_f(row.get('max_drawdown')):.4f} "
        + f"| {int(_f(row.get('trade_count')))} "
        + f"| {int(_f(row.get('wins')))} "
        + f"| {int(_f(row.get('losses')))} "
        + "|"
    )

out_md.write_text("\n".join(lines) + "\n")
PY

"$PYTHON_BIN" - "$RUN_DIR" "$FROM_TS" "$TO_TS" <<'PY'
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sys

run_dir = Path(sys.argv[1]).resolve()
from_ts = sys.argv[2]
to_ts = sys.argv[3]

pointer = {
    "updated_at": datetime.now(UTC).isoformat(),
    "run_dir": str(run_dir),
    "from_ts": from_ts,
    "to_ts": to_ts,
    "artifact": str((run_dir / "latest.json").resolve()),
}

pointer_path = Path("artifacts/backtest/latest-run.json")
pointer_path.parent.mkdir(parents=True, exist_ok=True)
pointer_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n")
PY

ln -sfn "$RUN_DIR" artifacts/backtest/latest || true

echo "[backtest-cycle] artifacts"
echo "- $RUN_DIR/latest.json"
echo "- $RUN_DIR/replay.csv"
echo "- $RUN_DIR/strategy.csv"
echo "- $RUN_DIR/event.csv"
echo "- $RUN_DIR/summary.md"
echo "- artifacts/backtest/latest-run.json"
echo "- artifacts/backtest/latest (symlink)"
