#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOAK_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      SOAK_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/paper_soak_status.sh [--dir artifacts/soak/paper-...]
USAGE
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOAK_DIR" ]]; then
  latest="$(ls -1dt artifacts/soak/paper-* 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest" ]]; then
    echo "no soak run found under artifacts/soak"
    exit 1
  fi
  SOAK_DIR="$latest"
fi

LATEST_JSON="$SOAK_DIR/status/latest.json"
HISTORY_JSONL="$SOAK_DIR/status/history.jsonl"
LOG_FILE="$SOAK_DIR/soak.log"

echo "soak_dir=$SOAK_DIR"
if [[ -f "$LATEST_JSON" ]]; then
  echo "--- latest status ---"
  cat "$LATEST_JSON"
else
  echo "latest status not found: $LATEST_JSON"
fi

if [[ -f "$HISTORY_JSONL" ]]; then
  echo "--- last 5 history rows ---"
  tail -n 5 "$HISTORY_JSONL"
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "--- log tail ---"
  tail -n 20 "$LOG_FILE"
fi
