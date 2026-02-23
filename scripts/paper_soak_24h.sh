#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOURS="24"
INTERVAL_SEC="300"
MAX_SIGNALS_PER_CYCLE="10"
INGEST_LIMIT="300"
RETRY_MAX="3"
RETRY_BACKOFF_SEC="5"
CONFIG_PATH="configs/config.yaml"
OUT_DIR=""

usage() {
  cat <<'USAGE'
Usage: bash scripts/paper_soak_24h.sh [options]

Options:
  --hours <float>                 Soak duration in hours (default: 24)
  --interval-sec <int>            Loop interval in seconds (default: 300)
  --max-signals-per-cycle <int>   Max new signals executed each cycle (default: 10)
  --ingest-limit <int>            Ingest limit per source (default: 300)
  --retry-max <int>               Retry attempts per stage (default: 3)
  --retry-backoff-sec <int>       Backoff base for retries (default: 5)
  --config <path>                 Config path (default: configs/config.yaml)
  --out-dir <path>                Output directory (default: artifacts/soak/paper-<ts>)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)
      HOURS="$2"
      shift 2
      ;;
    --interval-sec)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --max-signals-per-cycle)
      MAX_SIGNALS_PER_CYCLE="$2"
      shift 2
      ;;
    --ingest-limit)
      INGEST_LIMIT="$2"
      shift 2
      ;;
    --retry-max)
      RETRY_MAX="$2"
      shift 2
      ;;
    --retry-backoff-sec)
      RETRY_BACKOFF_SEC="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[soak] unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="artifacts/soak/paper-$(date -u +%Y%m%dT%H%M%SZ)"
fi

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/status"
LOG_FILE="$OUT_DIR/soak.log"
STATUS_LATEST="$OUT_DIR/status/latest.json"
STATUS_HISTORY="$OUT_DIR/status/history.jsonl"

echo "[soak] root=$ROOT" | tee -a "$LOG_FILE"
echo "[soak] out_dir=$OUT_DIR" | tee -a "$LOG_FILE"

# Hard safety defaults: always paper; never live by default.
export ENABLE_LIVE_TRADING=false
export MONOMARKET_MODE=paper

if [[ -x ".venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v monomarket >/dev/null 2>&1; then
  echo "[soak] monomarket command not found. Run: pip install -e '.[dev]'" | tee -a "$LOG_FILE"
  exit 1
fi

DB_PATH="$(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = Path(sys.argv[1])
raw = {}
if cfg.exists():
    raw = yaml.safe_load(cfg.read_text()) or {}
print(str((raw.get("app") or {}).get("db_path", "data/monomarket.db")))
PY
)"

run_with_retry() {
  local stage="$1"
  shift

  local attempt=1
  while [[ "$attempt" -le "$RETRY_MAX" ]]; do
    echo "[soak][stage=$stage] attempt=$attempt cmd=$*" | tee -a "$LOG_FILE"
    if "$@" >>"$LOG_FILE" 2>&1; then
      echo "[soak][stage=$stage] ok" | tee -a "$LOG_FILE"
      return 0
    fi

    if [[ "$attempt" -ge "$RETRY_MAX" ]]; then
      echo "[soak][stage=$stage] failed after $attempt attempts" | tee -a "$LOG_FILE"
      return 1
    fi

    local sleep_sec=$(( RETRY_BACKOFF_SEC * attempt ))
    echo "[soak][stage=$stage] retry in ${sleep_sec}s" | tee -a "$LOG_FILE"
    sleep "$sleep_sec"
    attempt=$(( attempt + 1 ))
  done

  return 1
}

write_status() {
  local status_json="$1"
  echo "$status_json" > "$STATUS_LATEST"
  echo "$status_json" >> "$STATUS_HISTORY"
}

new_signal_ids() {
  python - "$DB_PATH" "$MAX_SIGNALS_PER_CYCLE" <<'PY'
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
cur = conn.execute(
    "SELECT id FROM signals WHERE status = 'new' ORDER BY id ASC LIMIT ?",
    (int(sys.argv[2]),),
)
for row in cur.fetchall():
    print(row[0])
conn.close()
PY
}

run_with_retry "init-db" monomarket init-db --config "$CONFIG_PATH"

END_EPOCH="$(python - "$HOURS" <<'PY'
import sys, time
print(int(time.time() + float(sys.argv[1]) * 3600))
PY
)"

echo "[soak] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) end_epoch=$END_EPOCH" | tee -a "$LOG_FILE"

CYCLE=0
while [[ "$(date +%s)" -lt "$END_EPOCH" ]]; do
  CYCLE=$(( CYCLE + 1 ))
  CYCLE_START_EPOCH="$(date +%s)"
  CYCLE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  cycle_ok=true
  ingest_ok=true
  signals_ok=true
  execute_ok=true

  if ! run_with_retry "ingest" monomarket ingest --source all --limit "$INGEST_LIMIT" --incremental --config "$CONFIG_PATH"; then
    cycle_ok=false
    ingest_ok=false
  fi

  if ! run_with_retry "generate-signals" monomarket generate-signals --strategies s1,s2,s4,s8 --market-limit 2000 --config "$CONFIG_PATH"; then
    cycle_ok=false
    signals_ok=false
  fi

  executed_count=0
  execute_failures=0

  while IFS= read -r signal_id; do
    [[ -z "$signal_id" ]] && continue
    if run_with_retry "execute-signal:$signal_id" monomarket execute-signal "$signal_id" --mode paper --config "$CONFIG_PATH"; then
      executed_count=$(( executed_count + 1 ))
    else
      cycle_ok=false
      execute_ok=false
      execute_failures=$(( execute_failures + 1 ))
    fi
  done < <(new_signal_ids)

  run_with_retry "pnl-report" monomarket pnl-report --config "$CONFIG_PATH" || true
  run_with_retry "metrics-report" monomarket metrics-report --config "$CONFIG_PATH" || true

  CYCLE_END_EPOCH="$(date +%s)"
  CYCLE_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CYCLE_SEC=$(( CYCLE_END_EPOCH - CYCLE_START_EPOCH ))

  STATUS="ok"
  if [[ "$cycle_ok" != true ]]; then
    STATUS="partial"
  fi

  ingest_ok_py="False"
  signals_ok_py="False"
  execute_ok_py="False"
  [[ "$ingest_ok" == true ]] && ingest_ok_py="True"
  [[ "$signals_ok" == true ]] && signals_ok_py="True"
  [[ "$execute_ok" == true ]] && execute_ok_py="True"

  STATUS_JSON="$(python - <<PY
import json
print(json.dumps({
  "ts": "$CYCLE_END_ISO",
  "cycle": $CYCLE,
  "status": "$STATUS",
  "ingest_ok": $ingest_ok_py,
  "signals_ok": $signals_ok_py,
  "execute_ok": $execute_ok_py,
  "executed_count": $executed_count,
  "execute_failures": $execute_failures,
  "cycle_sec": $CYCLE_SEC,
  "db_path": "$DB_PATH",
  "config": "$CONFIG_PATH"
}, ensure_ascii=False))
PY
)"

  echo "[soak] cycle=$CYCLE status=$STATUS executed=$executed_count failures=$execute_failures sec=$CYCLE_SEC" | tee -a "$LOG_FILE"
  write_status "$STATUS_JSON"

  NOW_EPOCH="$(date +%s)"
  if [[ "$NOW_EPOCH" -ge "$END_EPOCH" ]]; then
    break
  fi

  sleep_for=$(( INTERVAL_SEC - CYCLE_SEC ))
  if [[ "$sleep_for" -gt 0 ]]; then
    echo "[soak] sleeping ${sleep_for}s" | tee -a "$LOG_FILE"
    sleep "$sleep_for"
  fi

done

FINAL_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FINAL_JSON="$(python - <<PY
import json
print(json.dumps({
  "ts": "$FINAL_ISO",
  "status": "finished",
  "cycles": $CYCLE,
  "out_dir": "$OUT_DIR",
  "db_path": "$DB_PATH"
}, ensure_ascii=False))
PY
)"

write_status "$FINAL_JSON"
echo "[soak] finished cycles=$CYCLE" | tee -a "$LOG_FILE"
