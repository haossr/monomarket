#!/usr/bin/env python3
"""Import PMXT Polymarket parquet snapshots into monomarket market_snapshots.

This bridges local PMXT archive files with monomarket backtests by writing denser
`market_snapshots` rows (source = 'pmxt_archive').
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb

FILENAME_RE = re.compile(r"^polymarket_orderbook_(\d{4}-\d{2}-\d{2}T\d{2})\.parquet$")
IMPORT_SOURCE = "pmxt_archive"


@dataclass(frozen=True)
class FileHour:
    name: str
    path: Path
    hour_start_utc: datetime
    hour_end_utc: datetime


class SyncLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "SyncLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another importer process is running (lock={self.path})") from exc
        os.ftruncate(self.fd, 0)
        os.write(self.fd, str(os.getpid()).encode("utf-8"))
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def parse_hour_from_filename(path: Path) -> FileHour | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    hour_start = datetime.strptime(m.group(1), "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    return FileHour(
        name=path.name,
        path=path,
        hour_start_utc=hour_start,
        hour_end_utc=hour_start + timedelta(hours=1),
    )


def list_parquet_hours(input_dir: Path) -> list[FileHour]:
    out: list[FileHour] = []
    for path in sorted(input_dir.glob("polymarket_orderbook_*.parquet")):
        fh = parse_hour_from_filename(path)
        if fh is not None:
            out.append(fh)
    return out


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"bucket_minutes": None, "processed": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"bucket_minutes": None, "processed": {}}
    if not isinstance(payload, dict):
        return {"bucket_minutes": None, "processed": {}}
    processed = payload.get("processed")
    if not isinstance(processed, dict):
        processed = {}
    return {
        "bucket_minutes": payload.get("bucket_minutes"),
        "processed": processed,
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_event_map(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT market_id, event_id FROM markets").fetchall()
    event_map: dict[str, str] = {}
    for market_id, event_id in rows:
        if market_id not in event_map and event_id:
            event_map[str(market_id)] = str(event_id)
    return event_map


def to_iso_utc(dt_obj: datetime) -> str:
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    else:
        dt_obj = dt_obj.astimezone(timezone.utc)
    return dt_obj.isoformat()


def aggregate_hour_rows(con: duckdb.DuckDBPyConnection, file_hour: FileHour, bucket_minutes: int) -> list[tuple[Any, ...]]:
    bucket_sec = bucket_minutes * 60
    start_ts = file_hour.hour_start_utc.timestamp()
    end_ts = file_hour.hour_end_utc.timestamp()
    parquet_path = str(file_hour.path).replace("'", "''")

    sql = f"""
    WITH raw AS (
      SELECT
        CAST(json_extract_string(data, '$.timestamp') AS DOUBLE) AS ts_epoch,
        market_id,
        upper(json_extract_string(data, '$.side')) AS side,
        TRY_CAST(json_extract_string(data, '$.best_bid') AS DOUBLE) AS best_bid,
        TRY_CAST(json_extract_string(data, '$.best_ask') AS DOUBLE) AS best_ask,
        COALESCE(timestamp_created_at, timestamp_received) AS row_ts
      FROM read_parquet('{parquet_path}')
      WHERE update_type='price_change'
    ),
    bucketed AS (
      SELECT
        to_timestamp(floor(ts_epoch / {bucket_sec}) * {bucket_sec}) AS bucket_ts,
        market_id,
        side,
        CASE
          WHEN best_bid IS NOT NULL AND best_ask IS NOT NULL THEN (best_bid + best_ask) / 2.0
          WHEN best_bid IS NOT NULL THEN best_bid
          ELSE best_ask
        END AS side_mid,
        row_ts
      FROM raw
      WHERE ts_epoch >= {start_ts}
        AND ts_epoch < {end_ts}
        AND side IN ('YES', 'NO')
        AND (best_bid IS NOT NULL OR best_ask IS NOT NULL)
    ),
    ranked AS (
      SELECT
        bucket_ts,
        market_id,
        side,
        side_mid,
        ROW_NUMBER() OVER (
          PARTITION BY bucket_ts, market_id, side
          ORDER BY row_ts DESC
        ) AS rn
      FROM bucketed
    )
    SELECT
      bucket_ts,
      market_id,
      MAX(CASE WHEN side='YES' AND rn=1 THEN side_mid END) AS yes_price,
      MAX(CASE WHEN side='NO' AND rn=1 THEN side_mid END) AS no_price
    FROM ranked
    WHERE rn=1
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    return con.execute(sql).fetchall()


def chunked(iterable: list[tuple[Any, ...]], size: int) -> Iterable[list[tuple[Any, ...]]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def insert_hour(
    conn: sqlite3.Connection,
    file_hour: FileHour,
    rows: list[tuple[Any, ...]],
    event_map: dict[str, str],
) -> tuple[int, int]:
    start_iso = to_iso_utc(file_hour.hour_start_utc)
    end_iso = to_iso_utc(file_hour.hour_end_utc)

    conn.execute("BEGIN")
    try:
        deleted = conn.execute(
            """
            DELETE FROM market_snapshots
            WHERE source = ? AND captured_at >= ? AND captured_at < ?
            """,
            (IMPORT_SOURCE, start_iso, end_iso),
        ).rowcount

        to_insert: list[tuple[Any, ...]] = []
        for bucket_ts, market_id, yes_price, no_price in rows:
            yes_px = float(yes_price) if yes_price is not None else None
            no_px = float(no_price) if no_price is not None else None
            if yes_px is not None and no_px is not None:
                mid_px = (yes_px + no_px) / 2.0
            else:
                mid_px = yes_px if yes_px is not None else no_px

            market = str(market_id)
            event_id = event_map.get(market, market)
            captured_at = to_iso_utc(bucket_ts)
            to_insert.append(
                (
                    IMPORT_SOURCE,
                    market,
                    event_id,
                    yes_px,
                    no_px,
                    mid_px,
                    0.0,
                    0.0,
                    captured_at,
                )
            )

        inserted = 0
        for batch in chunked(to_insert, 5000):
            conn.executemany(
                """
                INSERT INTO market_snapshots(
                    source, market_id, event_id,
                    yes_price, no_price, mid_price,
                    liquidity, volume, captured_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            inserted += len(batch)

        conn.execute("COMMIT")
        return inserted, deleted
    except Exception:
        conn.execute("ROLLBACK")
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import PMXT parquet snapshots into monomarket backtest DB")
    p.add_argument(
        "--input-dir",
        default="/Users/hao/.openclaw/workspace/data/pmxt/Polymarket",
        help="Directory with polymarket_orderbook_*.parquet files",
    )
    p.add_argument(
        "--db-path",
        default="/Users/hao/.openclaw/workspace/projects/monomarket/data/monomarket.db",
        help="Path to monomarket sqlite DB",
    )
    p.add_argument(
        "--state-file",
        default="",
        help="Path to state JSON (default: <input-dir>/.ingest_state.json)",
    )
    p.add_argument("--bucket-minutes", type=int, default=15, help="Snapshot bucket size in minutes")
    p.add_argument("--max-files", type=int, default=0, help="Max files per run (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="List pending files without importing")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.bucket_minutes <= 0:
        raise SystemExit("--bucket-minutes must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    state_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else (input_dir / ".ingest_state.json")
    )
    lock_path = input_dir / ".ingest.lock"

    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")
    if not db_path.exists():
        raise SystemExit(f"db path not found: {db_path}")

    started = time.time()

    with SyncLock(lock_path):
        state = load_state(state_path)
        processed = state.get("processed") or {}
        if not isinstance(processed, dict):
            processed = {}

        old_bucket = state.get("bucket_minutes")
        if old_bucket is not None and int(old_bucket) != int(args.bucket_minutes):
            print(
                f"[import] bucket changed {old_bucket} -> {args.bucket_minutes}; "
                "reprocessing all files"
            )
            processed = {}

        files = list_parquet_hours(input_dir)
        todo = [f for f in files if f.name not in processed]

        if args.max_files and args.max_files > 0:
            todo = todo[: args.max_files]

        print(f"[import] discovered={len(files)} pending={len(todo)} bucket={args.bucket_minutes}m")

        if args.dry_run:
            for f in todo[:20]:
                print(f"[dry-run] {f.name}")
            if len(todo) > 20:
                print(f"[dry-run] ... and {len(todo) - 20} more")
            return 0

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            event_map = load_event_map(conn)

            con = duckdb.connect()
            imported_files = 0
            imported_rows = 0
            deleted_rows = 0

            for idx, file_hour in enumerate(todo, start=1):
                print(f"[import] ({idx}/{len(todo)}) {file_hour.name}")
                rows = aggregate_hour_rows(con, file_hour, bucket_minutes=args.bucket_minutes)
                inserted, deleted = insert_hour(conn, file_hour, rows, event_map)

                now_iso = datetime.now(timezone.utc).isoformat()
                processed[file_hour.name] = {
                    "hour_start": to_iso_utc(file_hour.hour_start_utc),
                    "rows": inserted,
                    "deleted": deleted,
                    "processed_at": now_iso,
                }
                state["bucket_minutes"] = int(args.bucket_minutes)
                state["processed"] = processed
                save_state(state_path, state)

                imported_files += 1
                imported_rows += inserted
                deleted_rows += max(0, deleted)
                print(f"[import] done {file_hour.name}: inserted={inserted} deleted={deleted}")

        elapsed = time.time() - started
        print(
            f"[import] finished files={imported_files} inserted={imported_rows} "
            f"deleted={deleted_rows} elapsed={elapsed:.1f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
