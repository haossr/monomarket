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
CLEAR_SIGNALS_WINDOW="0"
REBUILD_SIGNALS_WINDOW="0"
REBUILD_STEP_HOURS="12"

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
  --clear-signals-window     Delete existing signals in [from_ts,to_ts] before generate-signals
                             (safety: fixed-window mode only)
  --rebuild-signals-window   Rebuild signals across fixed window from market_snapshots
                             (requires --clear-signals-window; experimental)
  --rebuild-step-hours <f>   Step hours for rebuild-signals-window sampling (default: 12)
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

if [[ "$CLEAR_SIGNALS_WINDOW" == "1" && "$FIXED_WINDOW_MODE" != "true" ]]; then
  echo "[backtest-cycle] --clear-signals-window requires fixed window (--from-ts/--to-ts)" >&2
  exit 1
fi

if [[ "$REBUILD_SIGNALS_WINDOW" == "1" && "$FIXED_WINDOW_MODE" != "true" ]]; then
  echo "[backtest-cycle] --rebuild-signals-window requires fixed window (--from-ts/--to-ts)" >&2
  exit 1
fi

if [[ "$REBUILD_SIGNALS_WINDOW" == "1" && "$CLEAR_SIGNALS_WINDOW" != "1" ]]; then
  echo "[backtest-cycle] --rebuild-signals-window requires --clear-signals-window" >&2
  exit 1
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
        "SELECT COUNT(*) AS c, MIN(created_at) AS first_ts, MAX(created_at) AS last_ts FROM signals "
        "WHERE datetime(created_at) >= datetime(?) "
        "AND datetime(created_at) >= datetime(?) "
        "AND datetime(created_at) <= datetime(?)" + where_strat,
        [run_start_ts, from_ts, to_ts, *params_tail],
    ).fetchone()

new_count = int(row_new["c"]) if row_new else 0
in_window_count = int(row_in_window["c"]) if row_in_window else 0
first_ts = str(row_in_window["first_ts"] or "") if row_in_window else ""
last_ts = str(row_in_window["last_ts"] or "") if row_in_window else ""
print(new_count)
print(in_window_count)
print(first_ts)
print(last_ts)
PY
}

_latest_signal_generation_run_json() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$1" <<'PY'
from __future__ import annotations

import json
import sys

from monomarket.config import load_settings
from monomarket.db import Storage

config_path, since_ts = sys.argv[1:]
settings = load_settings(config_path)
storage = Storage(settings.app.db_path)
row = storage.latest_signal_generation_run(since_ts=since_ts)
print(json.dumps(row or {}, ensure_ascii=False))
PY
}

_clear_signals_in_window() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$1" "$2" "$STRATEGIES" <<'PY'
from __future__ import annotations

import sys

from monomarket.config import load_settings
from monomarket.db import Storage

config_path, from_ts, to_ts, strategies_csv = sys.argv[1:]
settings = load_settings(config_path)
storage = Storage(settings.app.db_path)
strategies = [s.strip() for s in strategies_csv.split(",") if s.strip()]

where_strat = ""
params_tail: list[str] = []
if strategies:
    where_strat = " AND strategy IN (" + ",".join(["?"] * len(strategies)) + ")"
    params_tail = strategies

with storage.conn() as conn:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM signals WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)"
        + where_strat,
        [from_ts, to_ts, *params_tail],
    ).fetchone()
    count = int(row["c"]) if row else 0
    conn.execute(
        "DELETE FROM signals WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)"
        + where_strat,
        [from_ts, to_ts, *params_tail],
    )
print(count)
PY
}

_rebuild_signals_from_snapshots() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$1" "$2" "$3" "$4" "$5" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

from monomarket.config import load_settings
from monomarket.db import Storage
from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.s1_cross_venue import S1CrossVenueScanner
from monomarket.signals.strategies.s2_negrisk_rebalance import S2NegRiskRebalance
from monomarket.signals.strategies.s4_low_prob_yes import S4LowProbYesBasket
from monomarket.signals.strategies.s8_no_carry_tailhedge import S8NoCarryTailHedge

config_path, from_ts, to_ts, strategies_csv, market_limit_raw, step_hours_raw = sys.argv[1:]
settings = load_settings(config_path)
storage = Storage(settings.app.db_path)
strategies = [s.strip().lower() for s in strategies_csv.split(",") if s.strip()]
market_limit = max(1, int(float(market_limit_raw)))
step_hours = max(0.25, float(step_hours_raw))


def _parse_iso(ts: str) -> datetime:
    txt = ts.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


registry = {
    "s1": S1CrossVenueScanner(),
    "s2": S2NegRiskRebalance(),
    "s4": S4LowProbYesBasket(),
    "s8": S8NoCarryTailHedge(),
}

from_dt = _parse_iso(from_ts)
to_dt = _parse_iso(to_ts)
step_delta = timedelta(hours=step_hours)

with storage.conn() as conn:
    ts_rows = conn.execute(
        """
        SELECT DISTINCT captured_at
        FROM market_snapshots
        WHERE datetime(captured_at) >= datetime(?) AND datetime(captured_at) <= datetime(?)
        ORDER BY captured_at ASC
        """,
        (from_dt.isoformat(), to_dt.isoformat()),
    ).fetchall()

timestamps = [_parse_iso(str(row["captured_at"])) for row in ts_rows if row["captured_at"]]
sampled: list[datetime] = []
last_keep: datetime | None = None
for ts in timestamps:
    if last_keep is None or ts - last_keep >= step_delta:
        sampled.append(ts)
        last_keep = ts

if sampled and sampled[-1] < to_dt:
    sampled.append(to_dt)
elif not sampled:
    sampled = [to_dt]

with storage.conn() as conn:
    meta_rows = conn.execute(
        """
        SELECT market_id, canonical_id, event_id, question, neg_risk
        FROM markets
        """
    ).fetchall()
meta = {
    str(r["market_id"]): {
        "canonical_id": str(r["canonical_id"] or r["market_id"]),
        "event_id": str(r["event_id"] or ""),
        "question": str(r["question"] or r["market_id"]),
        "neg_risk": bool(r["neg_risk"]),
    }
    for r in meta_rows
}

inserted_total = 0
generated_total = 0
first_ts = ""
last_ts = ""

for ts in sampled:
    ts_iso = _iso(ts)
    with storage.conn() as conn:
        snap_rows = conn.execute(
            """
            SELECT source, market_id, event_id, yes_price, no_price, mid_price, liquidity, volume
            FROM market_snapshots
            WHERE datetime(captured_at) = datetime(?)
            ORDER BY liquidity DESC, market_id ASC
            LIMIT ?
            """,
            (ts_iso, market_limit),
        ).fetchall()

    views: list[MarketView] = []
    for row in snap_rows:
        market_id = str(row["market_id"])
        meta_row = meta.get(market_id)
        if not meta_row:
            continue
        yes_price = row["yes_price"]
        no_price = row["no_price"]
        mid_price = row["mid_price"]
        if mid_price is None:
            mid_price = yes_price if yes_price is not None else None
        views.append(
            MarketView(
                source=str(row["source"] or "gamma"),
                market_id=market_id,
                canonical_id=meta_row["canonical_id"],
                event_id=str(row["event_id"] or meta_row["event_id"]),
                question=meta_row["question"],
                status="open",
                neg_risk=bool(meta_row["neg_risk"]),
                liquidity=float(row["liquidity"] or 0.0),
                volume=float(row["volume"] or 0.0),
                yes_price=yes_price,
                no_price=no_price,
                best_bid=None,
                best_ask=None,
                mid_price=mid_price,
            )
        )

    generated: list[Signal] = []
    for strategy_name in strategies:
        impl = registry.get(strategy_name)
        if impl is None:
            continue
        cfg = settings.strategies.get(strategy_name, {})
        generated.extend(impl.generate(views, cfg))

    dedup: dict[tuple[str, str, str, str], Signal] = {}
    for signal in generated:
        key = (
            signal.strategy.lower(),
            signal.market_id,
            signal.event_id,
            signal.side.lower(),
        )
        prev = dedup.get(key)
        if prev is None or signal.score > prev.score:
            dedup[key] = signal

    dedup_signals = list(dedup.values())
    generated_total += len(dedup_signals)

    placeholders = ",".join("?" for _ in strategies)
    with storage.conn() as conn:
        rows = conn.execute(
            "SELECT strategy, market_id, event_id, side FROM signals "
            f"WHERE created_at = ? AND strategy IN ({placeholders})",
            [ts_iso, *strategies],
        ).fetchall()
    existing = {
        (str(r["strategy"]).lower(), str(r["market_id"]), str(r["event_id"]), str(r["side"]).lower())
        for r in rows
    }

    to_insert: list[Signal] = []
    for signal in dedup_signals:
        key = (
            signal.strategy.lower(),
            signal.market_id,
            signal.event_id,
            signal.side.lower(),
        )
        if key in existing:
            continue
        payload = dict(signal.payload or {})
        payload["rebuilt_from_snapshots"] = True
        payload["rebuilt_ts"] = ts_iso
        to_insert.append(
            Signal(
                strategy=signal.strategy,
                market_id=signal.market_id,
                event_id=signal.event_id,
                side=signal.side,
                score=signal.score,
                confidence=signal.confidence,
                target_price=signal.target_price,
                size_hint=signal.size_hint,
                rationale=signal.rationale,
                payload=payload,
            )
        )

    inserted_now = storage.insert_signals(to_insert, created_at=ts_iso)
    inserted_total += inserted_now
    if inserted_now > 0:
        if not first_ts:
            first_ts = ts_iso
        last_ts = ts_iso

print(generated_total)
print(inserted_total)
print(first_ts)
print(last_ts)
print(len(sampled))
PY
}

export ENABLE_LIVE_TRADING=false
export MONOMARKET_MODE=paper

echo "[backtest-cycle] run_dir=$RUN_DIR"

echo "[backtest-cycle] init-db"
monomarket init-db --config "$CONFIG_PATH"

echo "[backtest-cycle] ingest gamma incremental"
monomarket ingest --source gamma --limit "$INGEST_LIMIT" --incremental --config "$CONFIG_PATH"

CLEARED_SIGNALS_IN_WINDOW="0"
if [[ "$CLEAR_SIGNALS_WINDOW" == "1" ]]; then
  CLEARED_SIGNALS_IN_WINDOW="$(_clear_signals_in_window "$FROM_TS" "$TO_TS")"
  echo "[backtest-cycle] cleared_signals_in_window=${CLEARED_SIGNALS_IN_WINDOW}"
fi

RUN_START_TS="$(_now_iso)"

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

NEW_SIGNALS_TOTAL="0"
NEW_SIGNALS_IN_WINDOW="0"
NEW_SIGNALS_FIRST_TS=""
NEW_SIGNALS_LAST_TS=""
REBUILD_SAMPLED_STEPS="0"
EDGE_GATE_RUN_JSON="{}"

if [[ "$REBUILD_SIGNALS_WINDOW" == "1" ]]; then
  echo "[backtest-cycle] rebuild signals from snapshots: $STRATEGIES (step_h=${REBUILD_STEP_HOURS})"
  REBUILD_STATS="$(_rebuild_signals_from_snapshots "$FROM_TS" "$TO_TS" "$STRATEGIES" "$MARKET_LIMIT" "$REBUILD_STEP_HOURS")"
  REBUILD_GENERATED_TOTAL="$(echo "$REBUILD_STATS" | sed -n '1p')"
  REBUILD_INSERTED_TOTAL="$(echo "$REBUILD_STATS" | sed -n '2p')"
  NEW_SIGNALS_FIRST_TS="$(echo "$REBUILD_STATS" | sed -n '3p')"
  NEW_SIGNALS_LAST_TS="$(echo "$REBUILD_STATS" | sed -n '4p')"
  REBUILD_SAMPLED_STEPS="$(echo "$REBUILD_STATS" | sed -n '5p')"

  NEW_SIGNALS_TOTAL="$REBUILD_INSERTED_TOTAL"
  NEW_SIGNALS_IN_WINDOW="$REBUILD_INSERTED_TOTAL"
  echo "[backtest-cycle] rebuilt_signals generated=${REBUILD_GENERATED_TOTAL} inserted=${REBUILD_INSERTED_TOTAL} sampled_steps=${REBUILD_SAMPLED_STEPS}"
else
  echo "[backtest-cycle] generate signals: $STRATEGIES"
  monomarket generate-signals \
    --strategies "$STRATEGIES" \
    --market-limit "$MARKET_LIMIT" \
    --config "$CONFIG_PATH"

  OVERLAP_STATS="$(_signal_generation_overlap_stats "$RUN_START_TS" "$FROM_TS" "$TO_TS")"
  NEW_SIGNALS_TOTAL="$(echo "$OVERLAP_STATS" | sed -n '1p')"
  NEW_SIGNALS_IN_WINDOW="$(echo "$OVERLAP_STATS" | sed -n '2p')"
  NEW_SIGNALS_FIRST_TS="$(echo "$OVERLAP_STATS" | sed -n '3p')"
  NEW_SIGNALS_LAST_TS="$(echo "$OVERLAP_STATS" | sed -n '4p')"
  EDGE_GATE_RUN_JSON="$(_latest_signal_generation_run_json "$RUN_START_TS")"
fi

echo "[backtest-cycle] signal_generation new_total=${NEW_SIGNALS_TOTAL} new_in_window=${NEW_SIGNALS_IN_WINDOW} first_ts=${NEW_SIGNALS_FIRST_TS:-n/a} last_ts=${NEW_SIGNALS_LAST_TS:-n/a}"
if [[ "$EDGE_GATE_RUN_JSON" != "{}" ]]; then
  EDGE_GATE_LINE="$($PYTHON_BIN - "$EDGE_GATE_RUN_JSON" <<'PY'
from __future__ import annotations
import json, sys
obj = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
diag = obj.get("diagnostics", {}).get("edge_gate", {}) if isinstance(obj, dict) else {}
if isinstance(diag, dict):
    print(
        "[backtest-cycle] edge_gate "
        f"raw={int(float(diag.get('total_raw', 0)))} "
        f"pass={int(float(diag.get('total_pass', 0)))} "
        f"fail={int(float(diag.get('total_fail', 0)))} "
        f"pass_rate={float(diag.get('pass_rate', 0.0) or 0.0):.2%}"
    )
PY
)"
  echo "$EDGE_GATE_LINE"
fi
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

"$PYTHON_BIN" - "$RUN_DIR" "$FROM_TS" "$TO_TS" "$RUN_START_TS" "$NEW_SIGNALS_TOTAL" "$NEW_SIGNALS_IN_WINDOW" "$NEW_SIGNALS_FIRST_TS" "$NEW_SIGNALS_LAST_TS" "$FIXED_WINDOW_MODE" "$CLEAR_SIGNALS_WINDOW" "$CLEARED_SIGNALS_IN_WINDOW" "$REBUILD_SIGNALS_WINDOW" "$REBUILD_STEP_HOURS" "$REBUILD_SAMPLED_STEPS" "$EDGE_GATE_RUN_JSON" <<'PY'
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
new_signals_first_ts = str(sys.argv[7] or "")
new_signals_last_ts = str(sys.argv[8] or "")
fixed_window_mode = str(sys.argv[9]).strip().lower() == "true"
clear_signals_window = str(sys.argv[10]).strip() == "1"
cleared_signals_in_window = int(float(sys.argv[11]))
rebuild_signals_window = str(sys.argv[12]).strip() == "1"
rebuild_step_hours = float(sys.argv[13])
rebuild_sampled_steps = int(float(sys.argv[14]))
edge_gate_run_raw = str(sys.argv[15] or "{}")

edge_gate_run: dict[str, object] = {}
try:
    parsed = json.loads(edge_gate_run_raw)
    if isinstance(parsed, dict):
        edge_gate_run = parsed
except json.JSONDecodeError:
    edge_gate_run = {}

edge_gate_diag = {}
raw_diag = edge_gate_run.get("diagnostics") if isinstance(edge_gate_run, dict) else None
if isinstance(raw_diag, dict):
    eg = raw_diag.get("edge_gate")
    if isinstance(eg, dict):
        edge_gate_diag = eg

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
        "new_signals_first_ts": new_signals_first_ts,
        "new_signals_last_ts": new_signals_last_ts,
        "historical_replay_only": new_signals_total > 0 and new_signals_in_window == 0,
        "clear_signals_window": clear_signals_window,
        "cleared_signals_in_window": cleared_signals_in_window,
        "rebuild_signals_window": rebuild_signals_window,
        "rebuild_step_hours": rebuild_step_hours,
        "rebuild_sampled_steps": rebuild_sampled_steps,
        "edge_gate": edge_gate_diag,
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
