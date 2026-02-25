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
FROM_TS=""
TO_TS=""

usage() {
  cat <<'USAGE'
Usage: bash scripts/backtest_cycle.sh [options]

Run one reusable backtest cycle:
  init-db -> ingest(gamma, incremental) -> generate-signals(S1,S2,S4,S8) -> backtest

Options:
  --lookback-hours <float>   Lookback window in hours (default: 24)
  --from-ts <ISO8601>        Optional fixed backtest window start (requires --to-ts)
  --to-ts <ISO8601>          Optional fixed backtest window end (requires --from-ts)
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
    --from-ts)
      FROM_TS="$2"
      shift 2
      ;;
    --to-ts)
      TO_TS="$2"
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

FIXED_WINDOW_MODE="false"
if [[ -n "$FROM_TS" || -n "$TO_TS" ]]; then
  if [[ -z "$FROM_TS" || -z "$TO_TS" ]]; then
    echo "[backtest-cycle] --from-ts and --to-ts must be provided together" >&2
    exit 1
  fi
  FIXED_WINDOW_MODE="true"
fi

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

_signal_generation_overlap_stats() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$1" "$2" "$3" "$STRATEGIES" <<'PY'
from __future__ import annotations

import sys

from monomarket.config import load_settings
from monomarket.db import Storage

config_path, run_start_ts, from_ts, to_ts, strategies_csv = sys.argv[1:]
settings = load_settings(config_path)
storage = Storage(settings.app.db_path)
strategies = [s.strip() for s in strategies_csv.split(",") if s.strip()]

where_strat = ""
params_tail: list[str] = []
if strategies:
    where_strat = " AND strategy IN (" + ",".join(["?"] * len(strategies)) + ")"
    params_tail = strategies

with storage.conn() as conn:
    row_new = conn.execute(
        "SELECT COUNT(*) AS c FROM signals "
        "WHERE datetime(created_at) >= datetime(?)" + where_strat,
        [run_start_ts, *params_tail],
    ).fetchone()
    row_in_window = conn.execute(
        "SELECT COUNT(*) AS c FROM signals "
        "WHERE datetime(created_at) >= datetime(?) "
        "AND datetime(created_at) >= datetime(?) "
        "AND datetime(created_at) <= datetime(?)" + where_strat,
        [run_start_ts, from_ts, to_ts, *params_tail],
    ).fetchone()

new_count = int(row_new["c"]) if row_new else 0
in_window_count = int(row_in_window["c"]) if row_in_window else 0
print(new_count)
print(in_window_count)
PY
}

export ENABLE_LIVE_TRADING=false
export MONOMARKET_MODE=paper

echo "[backtest-cycle] run_dir=$RUN_DIR"

echo "[backtest-cycle] init-db"
monomarket init-db --config "$CONFIG_PATH"

echo "[backtest-cycle] ingest gamma incremental"
monomarket ingest --source gamma --limit "$INGEST_LIMIT" --incremental --config "$CONFIG_PATH"

RUN_START_TS="$(_now_iso)"

echo "[backtest-cycle] generate signals: $STRATEGIES"
monomarket generate-signals \
  --strategies "$STRATEGIES" \
  --market-limit "$MARKET_LIMIT" \
  --config "$CONFIG_PATH"

if [[ -z "$FROM_TS" || -z "$TO_TS" ]]; then
  FROM_TO="$(_compute_window)"
  LOOKBACK_FROM_TS="$(echo "$FROM_TO" | sed -n '1p')"
  FROM_TS="$LOOKBACK_FROM_TS"
  TO_TS="$(_now_iso)"
fi

if [[ "$TO_TS" == "$FROM_TS" ]]; then
  TO_TS="$(_add_seconds_iso "$FROM_TS" 60)"
fi

echo "[backtest-cycle] window=${FROM_TS} -> ${TO_TS} (requested_lookback_hours=${LOOKBACK_HOURS})"

OVERLAP_STATS="$(_signal_generation_overlap_stats "$RUN_START_TS" "$FROM_TS" "$TO_TS")"
NEW_SIGNALS_TOTAL="$(echo "$OVERLAP_STATS" | sed -n '1p')"
NEW_SIGNALS_IN_WINDOW="$(echo "$OVERLAP_STATS" | sed -n '2p')"

echo "[backtest-cycle] signal_generation new_total=${NEW_SIGNALS_TOTAL} new_in_window=${NEW_SIGNALS_IN_WINDOW}"
if [[ "$NEW_SIGNALS_TOTAL" -gt 0 && "$NEW_SIGNALS_IN_WINDOW" -eq 0 ]]; then
  echo "[backtest-cycle] warning: generated signals are outside replay window; fixed-window runs may be replaying historical signals only" >&2
fi

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
  --with-csv-digest-sidecar \
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
    "| strategy | pnl | closed_winrate | mtm_winrate | max_drawdown | trades | closed_samples | mtm_samples |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    closed_samples = int(_f(row.get("closed_sample_count")))
    mtm_samples = int(_f(row.get("mtm_sample_count")))
    closed_winrate = "n/a" if closed_samples <= 0 else f"{(_f(row.get('closed_winrate', row.get('winrate'))) * 100.0):.2f}%"
    mtm_winrate = "n/a" if mtm_samples <= 0 else f"{(_f(row.get('mtm_winrate')) * 100.0):.2f}%"
    lines.append(
        "| "
        + f"{row.get('strategy', '')} "
        + f"| {_f(row.get('pnl')):.4f} "
        + f"| {closed_winrate} "
        + f"| {mtm_winrate} "
        + f"| {_f(row.get('max_drawdown')):.4f} "
        + f"| {int(_f(row.get('trade_count')))} "
        + f"| {closed_samples} "
        + f"| {mtm_samples} "
        + "|"
    )

out_md.write_text("\n".join(lines) + "\n")
PY

"$PYTHON_BIN" - "$RUN_DIR" "$FROM_TS" "$TO_TS" "$RUN_START_TS" "$NEW_SIGNALS_TOTAL" "$NEW_SIGNALS_IN_WINDOW" "$FIXED_WINDOW_MODE" <<'PY'
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sys

run_dir = Path(sys.argv[1]).resolve()
from_ts = sys.argv[2]
to_ts = sys.argv[3]
run_start_ts = sys.argv[4]
new_signals_total = int(float(sys.argv[5]))
new_signals_in_window = int(float(sys.argv[6]))
fixed_window_mode = str(sys.argv[7]).strip().lower() == "true"

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

cycle_meta = {
    "updated_at": pointer["updated_at"],
    "run_start_ts": run_start_ts,
    "fixed_window_mode": fixed_window_mode,
    "signal_generation": {
        "new_signals_total": new_signals_total,
        "new_signals_in_window": new_signals_in_window,
        "historical_replay_only": new_signals_total > 0 and new_signals_in_window == 0,
    },
}
(run_dir / "cycle-meta.json").write_text(json.dumps(cycle_meta, ensure_ascii=False, indent=2) + "\n")
PY

ln -sfn "$RUN_DIR" artifacts/backtest/latest || true

echo "[backtest-cycle] artifacts"
echo "- $RUN_DIR/latest.json"
echo "- $RUN_DIR/replay.csv"
echo "- $RUN_DIR/replay.csv.sha256"
echo "- $RUN_DIR/strategy.csv"
echo "- $RUN_DIR/strategy.csv.sha256"
echo "- $RUN_DIR/event.csv"
echo "- $RUN_DIR/event.csv.sha256"
echo "- $RUN_DIR/summary.md"
echo "- $RUN_DIR/cycle-meta.json"
echo "- artifacts/backtest/latest-run.json"
echo "- artifacts/backtest/latest (symlink)"
