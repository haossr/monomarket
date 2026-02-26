from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monomarket.models import MarketView, OrderRequest, Signal

OUTCOME_TOKEN_YES = "YES"  # nosec B105

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    rows_ingested INTEGER NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_buckets_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ingestion_state (
    source TEXT PRIMARY KEY,
    checkpoint_ts TEXT,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_failures INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_error_buckets (
    source TEXT NOT NULL,
    error_bucket TEXT NOT NULL,
    total_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY(source, error_bucket)
);

CREATE TABLE IF NOT EXISTS ingestion_error_bucket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    error_bucket TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ingestion_error_bucket_events_src_bucket_id
    ON ingestion_error_bucket_events(source, error_bucket, id DESC);

CREATE TABLE IF NOT EXISTS ingestion_breakers (
    source TEXT PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    open_until_ts TEXT,
    last_error_bucket TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_breaker_transitions (
    source TEXT NOT NULL,
    state TEXT NOT NULL,
    transition_count INTEGER NOT NULL DEFAULT 0,
    last_transition_at TEXT NOT NULL,
    PRIMARY KEY(source, state)
);

CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    market_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    neg_risk INTEGER NOT NULL DEFAULT 0,
    yes_price REAL,
    no_price REAL,
    best_bid REAL,
    best_ask REAL,
    mid_price REAL,
    liquidity REAL NOT NULL DEFAULT 0,
    volume REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(source, market_id)
);
CREATE INDEX IF NOT EXISTS idx_markets_canonical ON markets(canonical_id);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_id);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    yes_price REAL,
    no_price REAL,
    mid_price REAL,
    liquidity REAL NOT NULL DEFAULT 0,
    volume REAL NOT NULL DEFAULT 0,
    captured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_market_time
    ON market_snapshots(market_id, captured_at);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    side TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    target_price REAL NOT NULL,
    size_hint REAL NOT NULL,
    rationale TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at);

CREATE TABLE IF NOT EXISTS signal_generation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    strategies_csv TEXT NOT NULL,
    market_limit INTEGER NOT NULL,
    total_raw INTEGER NOT NULL DEFAULT 0,
    total_pass INTEGER NOT NULL DEFAULT 0,
    total_fail INTEGER NOT NULL DEFAULT 0,
    diagnostics_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_signal_generation_runs_finished_at
    ON signal_generation_runs(finished_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS switches (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    mode TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    status TEXT NOT NULL,
    external_id TEXT,
    message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy);
CREATE INDEX IF NOT EXISTS idx_orders_event ON orders(event_id);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    external_fill_id TEXT,
    raw_report_json TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(id)
);
CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_external_fill_id ON fills(external_fill_id);

CREATE TABLE IF NOT EXISTS positions (
    strategy TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    net_qty REAL NOT NULL,
    avg_price REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(strategy, market_id, token_id)
);
"""


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.conn() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_ingestion_runs_columns(conn)
            self._ensure_fills_columns(conn)

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_ingestion_runs_columns(self, conn: sqlite3.Connection) -> None:
        cols = self._table_columns(conn, "ingestion_runs")

        if "request_count" not in cols:
            conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN request_count INTEGER NOT NULL DEFAULT 0"
            )
        if "failure_count" not in cols:
            conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0"
            )
        if "retry_count" not in cols:
            conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
            )
        if "error_buckets_json" not in cols:
            conn.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN error_buckets_json TEXT NOT NULL DEFAULT '{}'"
            )

    def _ensure_fills_columns(self, conn: sqlite3.Connection) -> None:
        cols = self._table_columns(conn, "fills")

        if "external_fill_id" not in cols:
            conn.execute("ALTER TABLE fills ADD COLUMN external_fill_id TEXT")
        if "raw_report_json" not in cols:
            conn.execute("ALTER TABLE fills ADD COLUMN raw_report_json TEXT NOT NULL DEFAULT ''")

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_external_fill_id ON fills(external_fill_id)"
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def record_ingestion(
        self,
        source: str,
        status: str,
        started_at: str,
        finished_at: str,
        rows_ingested: int,
        error: str,
        request_count: int = 0,
        failure_count: int = 0,
        retry_count: int = 0,
        error_buckets: dict[str, int] | None = None,
    ) -> None:
        bucket_payload: dict[str, int] = {}
        if error_buckets:
            for key, value in error_buckets.items():
                if value <= 0:
                    continue
                bucket_payload[str(key)] = int(value)

        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs(
                    source, status, started_at, finished_at, rows_ingested,
                    request_count, failure_count, retry_count, error_buckets_json, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    status,
                    started_at,
                    finished_at,
                    rows_ingested,
                    max(0, request_count),
                    max(0, failure_count),
                    max(0, retry_count),
                    json.dumps(bucket_payload, sort_keys=True, separators=(",", ":")),
                    error,
                ),
            )

    def get_ingestion_checkpoint(self, source: str) -> str | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT checkpoint_ts FROM ingestion_state WHERE source = ?",
                (source.lower(),),
            ).fetchone()
        if row is None:
            return None
        checkpoint = row["checkpoint_ts"]
        return str(checkpoint) if checkpoint else None

    def update_ingestion_checkpoint(self, source: str, checkpoint_ts: str) -> None:
        now = self._now()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_state(source, checkpoint_ts, total_requests, total_failures, updated_at)
                VALUES (?, ?, 0, 0, ?)
                ON CONFLICT(source) DO UPDATE SET
                    checkpoint_ts = excluded.checkpoint_ts,
                    updated_at = excluded.updated_at
                """,
                (source.lower(), checkpoint_ts, now),
            )

    def update_ingestion_counters(
        self, source: str, request_count: int, failure_count: int
    ) -> None:
        now = self._now()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_state(source, checkpoint_ts, total_requests, total_failures, updated_at)
                VALUES (?, NULL, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    total_requests = ingestion_state.total_requests + excluded.total_requests,
                    total_failures = ingestion_state.total_failures + excluded.total_failures,
                    updated_at = excluded.updated_at
                """,
                (source.lower(), request_count, failure_count, now),
            )

    def update_ingestion_error_bucket(
        self,
        source: str,
        error_bucket: str,
        count: int,
        last_error: str,
    ) -> None:
        if count <= 0:
            return
        now = self._now()
        source_norm = source.lower()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_error_buckets(source, error_bucket, total_count, last_error, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, error_bucket) DO UPDATE SET
                    total_count = ingestion_error_buckets.total_count + excluded.total_count,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (source_norm, error_bucket, count, last_error, now),
            )
            conn.execute(
                """
                INSERT INTO ingestion_error_bucket_events(
                    source, error_bucket, count, last_error, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_norm, error_bucket, count, last_error, now),
            )

    def get_ingestion_breaker(self, source: str) -> dict[str, Any] | None:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT source, consecutive_failures, open_until_ts, last_error_bucket, updated_at
                FROM ingestion_breakers
                WHERE source = ?
                """,
                (source.lower(),),
            ).fetchone()
        return None if row is None else dict(row)

    def set_ingestion_breaker(
        self,
        source: str,
        consecutive_failures: int,
        open_until_ts: str | None,
        last_error_bucket: str,
    ) -> None:
        now = self._now()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_breakers(
                    source, consecutive_failures, open_until_ts, last_error_bucket, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    consecutive_failures = excluded.consecutive_failures,
                    open_until_ts = excluded.open_until_ts,
                    last_error_bucket = excluded.last_error_bucket,
                    updated_at = excluded.updated_at
                """,
                (
                    source.lower(),
                    max(0, consecutive_failures),
                    open_until_ts,
                    last_error_bucket,
                    now,
                ),
            )

    def list_ingestion_error_buckets(
        self,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            if source:
                rows = conn.execute(
                    """
                    SELECT source, error_bucket, total_count, last_error, updated_at
                    FROM ingestion_error_buckets
                    WHERE source = ?
                    ORDER BY total_count DESC, source ASC, error_bucket ASC
                    LIMIT ?
                    """,
                    (source.lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source, error_bucket, total_count, last_error, updated_at
                    FROM ingestion_error_buckets
                    ORDER BY total_count DESC, source ASC, error_bucket ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_ingestion_error_bucket_trends(
        self,
        source: str | None = None,
        window: int = 20,
        limit: int = 200,
        sort_by_abs_delta: bool = False,
    ) -> list[dict[str, Any]]:
        source_norm = source.lower() if source else None
        bucket_window = max(1, window)
        with self.conn() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        source,
                        error_bucket,
                        count,
                        created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY source, error_bucket
                            ORDER BY id DESC
                        ) AS rn
                    FROM ingestion_error_bucket_events
                    WHERE (? IS NULL OR source = ?)
                )
                SELECT
                    source,
                    error_bucket,
                    SUM(CASE WHEN rn <= ? THEN count ELSE 0 END) AS recent_count,
                    SUM(CASE WHEN rn > ? AND rn <= ? THEN count ELSE 0 END) AS prev_count,
                    MAX(CASE WHEN rn <= ? THEN created_at ELSE NULL END) AS recent_last_at,
                    MAX(CASE WHEN rn > ? AND rn <= ? THEN created_at ELSE NULL END) AS prev_last_at
                FROM ranked
                WHERE rn <= ?
                GROUP BY source, error_bucket
                ORDER BY source ASC, error_bucket ASC
                """,
                (
                    source_norm,
                    source_norm,
                    bucket_window,
                    bucket_window,
                    bucket_window * 2,
                    bucket_window,
                    bucket_window,
                    bucket_window * 2,
                    bucket_window * 2,
                ),
            ).fetchall()

        out = [dict(row) for row in rows]
        if sort_by_abs_delta:
            out.sort(
                key=lambda row: (
                    -abs(int(row["recent_count"] or 0) - int(row["prev_count"] or 0)),
                    str(row["source"]),
                    str(row["error_bucket"]),
                )
            )

        return out[: max(1, limit)]

    def list_ingestion_breakers(
        self,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            if source:
                rows = conn.execute(
                    """
                    SELECT source, consecutive_failures, open_until_ts, last_error_bucket, updated_at
                    FROM ingestion_breakers
                    WHERE source = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (source.lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source, consecutive_failures, open_until_ts, last_error_bucket, updated_at
                    FROM ingestion_breakers
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_ingestion_run_summary_by_source(
        self,
        source: str | None = None,
        run_window: int = 20,
    ) -> list[dict[str, Any]]:
        source_norm = source.lower() if source else None
        with self.conn() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        source,
                        status,
                        rows_ingested,
                        request_count,
                        failure_count,
                        retry_count,
                        finished_at,
                        ROW_NUMBER() OVER (PARTITION BY source ORDER BY id DESC) AS rn
                    FROM ingestion_runs
                    WHERE (? IS NULL OR source = ?)
                )
                SELECT
                    source,
                    COUNT(1) AS total_runs,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_runs,
                    SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) AS partial_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_runs,
                    SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS non_ok_runs,
                    AVG(rows_ingested) AS avg_rows,
                    AVG(request_count) AS avg_requests,
                    AVG(failure_count) AS avg_failures,
                    AVG(retry_count) AS avg_retries,
                    SUM(failure_count) AS total_failures,
                    SUM(request_count) AS total_requests,
                    MAX(finished_at) AS last_finished_at
                FROM ranked
                WHERE rn <= ?
                GROUP BY source
                ORDER BY source ASC
                """,
                (source_norm, source_norm, max(1, run_window)),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_ingestion_error_bucket_share_by_source(
        self,
        source: str | None = None,
        run_window: int = 20,
        limit: int = 200,
        top_k_per_source: int = 0,
        min_share: float = 0.0,
        min_bucket_count: int = 0,
        min_runs_with_error: int = 0,
        min_total_runs: int = 0,
        min_source_bucket_total: int = 0,
    ) -> list[dict[str, Any]]:
        source_norm = source.lower() if source else None
        with self.conn() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        source,
                        error_buckets_json,
                        ROW_NUMBER() OVER (PARTITION BY source ORDER BY id DESC) AS rn
                    FROM ingestion_runs
                    WHERE (? IS NULL OR source = ?)
                )
                SELECT source, error_buckets_json
                FROM ranked
                WHERE rn <= ?
                ORDER BY source ASC, rn ASC
                """,
                (source_norm, source_norm, max(1, run_window)),
            ).fetchall()

        total_runs: dict[str, int] = {}
        runs_with_error: dict[str, int] = {}
        counts_by_bucket: dict[tuple[str, str], int] = {}
        source_bucket_totals: dict[str, int] = {}

        for row in rows:
            row_source = str(row["source"])
            total_runs[row_source] = total_runs.get(row_source, 0) + 1

            raw_json = str(row["error_buckets_json"] or "{}")
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                parsed = {}

            if not isinstance(parsed, dict):
                parsed = {}

            normalized: dict[str, int] = {}
            for key, value in parsed.items():
                try:
                    bucket_count = int(value)
                except (TypeError, ValueError):
                    continue
                if bucket_count <= 0:
                    continue
                normalized[str(key)] = normalized.get(str(key), 0) + bucket_count

            if normalized:
                runs_with_error[row_source] = runs_with_error.get(row_source, 0) + 1

            for bucket, bucket_count in normalized.items():
                pair = (row_source, bucket)
                counts_by_bucket[pair] = counts_by_bucket.get(pair, 0) + bucket_count
                source_bucket_totals[row_source] = (
                    source_bucket_totals.get(row_source, 0) + bucket_count
                )

        out: list[dict[str, Any]] = []
        min_share_norm = max(0.0, min(1.0, float(min_share)))
        min_count_norm = max(0, int(min_bucket_count))
        min_runs_norm = max(0, int(min_runs_with_error))
        min_total_runs_norm = max(0, int(min_total_runs))
        min_source_total_norm = max(0, int(min_source_bucket_total))
        for (row_source, bucket), count in counts_by_bucket.items():
            source_total = source_bucket_totals.get(row_source, 0)
            share = (count / source_total) if source_total else 0.0
            source_runs_with_error = runs_with_error.get(row_source, 0)
            source_total_runs = total_runs.get(row_source, 0)
            if share < min_share_norm:
                continue
            if count < min_count_norm:
                continue
            if source_runs_with_error < min_runs_norm:
                continue
            if source_total_runs < min_total_runs_norm:
                continue
            if source_total < min_source_total_norm:
                continue
            out.append(
                {
                    "source": row_source,
                    "error_bucket": bucket,
                    "bucket_count": count,
                    "bucket_share": share,
                    "bucket_total": source_total,
                    "runs_with_error": source_runs_with_error,
                    "total_runs": source_total_runs,
                }
            )

        out.sort(
            key=lambda x: (
                str(x["source"]),
                -int(x["bucket_count"]),
                str(x["error_bucket"]),
            )
        )

        source_top_k = max(0, int(top_k_per_source))
        if source_top_k > 0:
            per_source_seen: dict[str, int] = {}
            capped: list[dict[str, Any]] = []
            for row in out:
                row_source = str(row["source"])
                seen = per_source_seen.get(row_source, 0)
                if seen >= source_top_k:
                    continue
                per_source_seen[row_source] = seen + 1
                capped.append(row)
            out = capped

        return out[: max(1, limit)]

    def list_ingestion_recent_errors(
        self,
        source: str | None = None,
        per_source_limit: int = 5,
    ) -> list[dict[str, Any]]:
        source_norm = source.lower() if source else None
        with self.conn() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        source,
                        status,
                        error,
                        finished_at,
                        ROW_NUMBER() OVER (PARTITION BY source ORDER BY id DESC) AS rn
                    FROM ingestion_runs
                    WHERE error != '' AND (? IS NULL OR source = ?)
                )
                SELECT source, status, error, finished_at
                FROM ranked
                WHERE rn <= ?
                ORDER BY source ASC, finished_at DESC
                """,
                (source_norm, source_norm, max(1, per_source_limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_ingestion_breaker_transition(self, source: str, state: str) -> None:
        now = self._now()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_breaker_transitions(
                    source, state, transition_count, last_transition_at
                )
                VALUES (?, ?, 1, ?)
                ON CONFLICT(source, state) DO UPDATE SET
                    transition_count = ingestion_breaker_transitions.transition_count + 1,
                    last_transition_at = excluded.last_transition_at
                """,
                (source.lower(), state, now),
            )

    def list_ingestion_breaker_transitions(
        self,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            if source:
                rows = conn.execute(
                    """
                    SELECT source, state, transition_count, last_transition_at
                    FROM ingestion_breaker_transitions
                    WHERE source = ?
                    ORDER BY state ASC
                    LIMIT ?
                    """,
                    (source.lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source, state, transition_count, last_transition_at
                    FROM ingestion_breaker_transitions
                    ORDER BY source ASC, state ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def upsert_markets(self, markets: list[MarketView], snapshot_at: str | None = None) -> int:
        if not markets:
            return 0
        now = self._now()
        captured_at = snapshot_at or now
        with self.conn() as conn:
            for m in markets:
                conn.execute(
                    """
                    INSERT INTO markets(
                        source, market_id, canonical_id, event_id, question, status, neg_risk,
                        yes_price, no_price, best_bid, best_ask, mid_price, liquidity, volume, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, market_id) DO UPDATE SET
                        canonical_id=excluded.canonical_id,
                        event_id=excluded.event_id,
                        question=excluded.question,
                        status=excluded.status,
                        neg_risk=excluded.neg_risk,
                        yes_price=excluded.yes_price,
                        no_price=excluded.no_price,
                        best_bid=excluded.best_bid,
                        best_ask=excluded.best_ask,
                        mid_price=excluded.mid_price,
                        liquidity=excluded.liquidity,
                        volume=excluded.volume,
                        updated_at=excluded.updated_at
                    """,
                    (
                        m.source,
                        m.market_id,
                        m.canonical_id,
                        m.event_id,
                        m.question,
                        m.status,
                        1 if m.neg_risk else 0,
                        m.yes_price,
                        m.no_price,
                        m.best_bid,
                        m.best_ask,
                        m.mid_price,
                        m.liquidity,
                        m.volume,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO market_snapshots(
                        source, market_id, event_id, yes_price, no_price, mid_price, liquidity, volume, captured_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.source,
                        m.market_id,
                        m.event_id,
                        m.yes_price,
                        m.no_price,
                        m.mid_price,
                        m.liquidity,
                        m.volume,
                        captured_at,
                    ),
                )
        return len(markets)

    def fetch_markets(self, limit: int = 1000, status: str | None = "open") -> list[MarketView]:
        with self.conn() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT source, market_id, canonical_id, event_id, question, status, neg_risk,
                           liquidity, volume, yes_price, no_price, best_bid, best_ask, mid_price
                    FROM markets
                    WHERE status = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source, market_id, canonical_id, event_id, question, status, neg_risk,
                           liquidity, volume, yes_price, no_price, best_bid, best_ask, mid_price
                    FROM markets
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [
            MarketView(
                source=row["source"],
                market_id=row["market_id"],
                canonical_id=row["canonical_id"],
                event_id=row["event_id"],
                question=row["question"],
                status=row["status"],
                neg_risk=bool(row["neg_risk"]),
                liquidity=float(row["liquidity"] or 0.0),
                volume=float(row["volume"] or 0.0),
                yes_price=row["yes_price"],
                no_price=row["no_price"],
                best_bid=row["best_bid"],
                best_ask=row["best_ask"],
                mid_price=row["mid_price"],
            )
            for row in rows
        ]

    def insert_signals(self, signals: list[Signal], created_at: str | None = None) -> int:
        if not signals:
            return 0
        now = created_at or self._now()
        with self.conn() as conn:
            conn.executemany(
                """
                INSERT INTO signals(
                    strategy, market_id, event_id, side, score, confidence,
                    target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                """,
                [
                    (
                        s.strategy,
                        s.market_id,
                        s.event_id,
                        s.side,
                        s.score,
                        s.confidence,
                        s.target_price,
                        s.size_hint,
                        s.rationale,
                        json.dumps(s.payload, ensure_ascii=False),
                        now,
                        now,
                    )
                    for s in signals
                ],
            )
        return len(signals)

    def insert_signal_generation_run(
        self,
        *,
        started_at: str,
        finished_at: str,
        strategies: list[str],
        market_limit: int,
        total_raw: int,
        total_pass: int,
        total_fail: int,
        diagnostics: dict[str, Any],
    ) -> int:
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO signal_generation_runs(
                    started_at, finished_at, strategies_csv, market_limit,
                    total_raw, total_pass, total_fail, diagnostics_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    finished_at,
                    ",".join(str(s).strip().lower() for s in strategies if str(s).strip()),
                    int(market_limit),
                    int(total_raw),
                    int(total_pass),
                    int(total_fail),
                    json.dumps(diagnostics, ensure_ascii=False),
                ),
            )
            row_id = cur.lastrowid
            return int(row_id) if row_id is not None else 0

    def latest_signal_generation_run(self, since_ts: str | None = None) -> dict[str, Any] | None:
        with self.conn() as conn:
            row = None
            if since_ts:
                row = conn.execute(
                    """
                    SELECT id, started_at, finished_at, strategies_csv, market_limit,
                           total_raw, total_pass, total_fail, diagnostics_json
                    FROM signal_generation_runs
                    WHERE datetime(finished_at) >= datetime(?)
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (since_ts,),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT id, started_at, finished_at, strategies_csv, market_limit,
                           total_raw, total_pass, total_fail, diagnostics_json
                    FROM signal_generation_runs
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()

        if row is None:
            return None

        diagnostics: dict[str, Any] = {}
        raw_diag = row["diagnostics_json"]
        if raw_diag:
            try:
                diagnostics = json.loads(str(raw_diag))
            except json.JSONDecodeError:
                diagnostics = {}

        return {
            "id": int(row["id"]),
            "started_at": str(row["started_at"]),
            "finished_at": str(row["finished_at"]),
            "strategies": [x for x in str(row["strategies_csv"] or "").split(",") if x],
            "market_limit": int(row["market_limit"]),
            "total_raw": int(row["total_raw"]),
            "total_pass": int(row["total_pass"]),
            "total_fail": int(row["total_fail"]),
            "diagnostics": diagnostics,
        }

    def list_signals(
        self,
        limit: int = 20,
        status: str | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            if status and strategy:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, side, score, confidence,
                           target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                    FROM signals
                    WHERE status = ? AND strategy = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, strategy, limit),
                ).fetchall()
            elif status:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, side, score, confidence,
                           target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                    FROM signals
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            elif strategy:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, side, score, confidence,
                           target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                    FROM signals
                    WHERE strategy = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (strategy, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, side, score, confidence,
                           target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                    FROM signals
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            payload = {}
            if row["payload_json"]:
                try:
                    payload = json.loads(row["payload_json"])
                except json.JSONDecodeError:
                    payload = {}
            d = dict(row)
            d["payload"] = payload
            d.pop("payload_json", None)
            out.append(d)
        return out

    def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT id, strategy, market_id, event_id, side, score, confidence,
                       target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                FROM signals WHERE id = ?
                """,
                (signal_id,),
            ).fetchone()

        if row is None:
            return None
        payload = {}
        if row["payload_json"]:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
        d = dict(row)
        d["payload"] = payload
        d.pop("payload_json", None)
        return d

    def list_signals_in_window(
        self,
        from_ts: str,
        to_ts: str,
        strategies: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT id, strategy, market_id, event_id, side, score, confidence,
                       target_price, size_hint, rationale, payload_json, status, created_at, updated_at
                FROM signals
                WHERE created_at >= ? AND created_at <= ?
                ORDER BY created_at ASC, id ASC
                """,
                (from_ts, to_ts),
            ).fetchall()

        wanted = {s.lower() for s in strategies} if strategies else None
        out: list[dict[str, Any]] = []
        for row in rows:
            if wanted is not None and str(row["strategy"]).lower() not in wanted:
                continue

            payload = {}
            if row["payload_json"]:
                try:
                    payload = json.loads(row["payload_json"])
                except json.JSONDecodeError:
                    payload = {}
            d = dict(row)
            d["payload"] = payload
            d.pop("payload_json", None)
            out.append(d)
        return out

    def get_snapshot_price_at(self, market_id: str, token_id: str, ts: str) -> float | None:
        token = token_id.upper()
        with self.conn() as conn:
            if token == OUTCOME_TOKEN_YES:
                row = conn.execute(
                    """
                    SELECT yes_price AS px
                    FROM market_snapshots
                    WHERE market_id = ? AND captured_at <= ? AND yes_price IS NOT NULL
                    ORDER BY captured_at DESC
                    LIMIT 1
                    """,
                    (market_id, ts),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """
                        SELECT yes_price AS px
                        FROM market_snapshots
                        WHERE market_id = ? AND yes_price IS NOT NULL
                        ORDER BY captured_at ASC
                        LIMIT 1
                        """,
                        (market_id,),
                    ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT no_price AS px
                    FROM market_snapshots
                    WHERE market_id = ? AND captured_at <= ? AND no_price IS NOT NULL
                    ORDER BY captured_at DESC
                    LIMIT 1
                    """,
                    (market_id, ts),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """
                        SELECT no_price AS px
                        FROM market_snapshots
                        WHERE market_id = ? AND no_price IS NOT NULL
                        ORDER BY captured_at ASC
                        LIMIT 1
                        """,
                        (market_id,),
                    ).fetchone()
        if row is None:
            return None
        return float(row["px"])

    def get_snapshot_liquidity_at(self, market_id: str, ts: str) -> float | None:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT liquidity AS v
                FROM market_snapshots
                WHERE market_id = ? AND captured_at <= ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                (market_id, ts),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT liquidity AS v
                    FROM market_snapshots
                    WHERE market_id = ?
                    ORDER BY captured_at ASC
                    LIMIT 1
                    """,
                    (market_id,),
                ).fetchone()
        if row is None:
            return None
        return float(row["v"])

    def get_snapshot_yes_no_at(self, market_id: str, ts: str) -> tuple[float | None, float | None]:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT yes_price, no_price
                FROM market_snapshots
                WHERE market_id = ? AND captured_at <= ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                (market_id, ts),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT yes_price, no_price
                    FROM market_snapshots
                    WHERE market_id = ?
                    ORDER BY captured_at ASC
                    LIMIT 1
                    """,
                    (market_id,),
                ).fetchone()

        if row is None:
            return None, None

        yes_px = float(row["yes_price"]) if row["yes_price"] is not None else None
        no_px = float(row["no_price"]) if row["no_price"] is not None else None
        return yes_px, no_px

    def update_signal_status(self, signal_id: int, status: str) -> None:
        with self.conn() as conn:
            conn.execute(
                "UPDATE signals SET status = ?, updated_at = ? WHERE id = ?",
                (status, self._now(), signal_id),
            )

    def set_switch(self, name: str, value: str) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO switches(name, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (name.upper(), value, self._now()),
            )

    def get_switch(self, name: str) -> str | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT value FROM switches WHERE name = ?", (name.upper(),)
            ).fetchone()
        return None if row is None else str(row["value"])

    def list_switches(self) -> list[dict[str, str]]:
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT name, value, updated_at FROM switches ORDER BY name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_order(self, order_id: int) -> dict[str, Any] | None:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT id, strategy, market_id, event_id, token_id, side, action, mode,
                       price, qty, status, external_id, message, created_at
                FROM orders
                WHERE id = ?
                """,
                (order_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def list_orders(
        self,
        mode: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.conn() as conn:
            if mode:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, token_id, side, action, mode,
                           price, qty, status, external_id, message, created_at
                    FROM orders
                    WHERE mode = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (mode, max(1, limit * 5)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, strategy, market_id, event_id, token_id, side, action, mode,
                           price, qty, status, external_id, message, created_at
                    FROM orders
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (max(1, limit * 5),),
                ).fetchall()

        out = [dict(row) for row in rows]
        if statuses:
            allowed = {x.lower() for x in statuses}
            out = [row for row in out if str(row["status"]).lower() in allowed]

        return out[: max(1, limit)]

    def insert_order(
        self,
        req: OrderRequest,
        status: str,
        message: str = "",
        external_id: str | None = None,
    ) -> int:
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO orders(
                    strategy, market_id, event_id, token_id, side, action, mode,
                    price, qty, status, external_id, message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req.strategy,
                    req.market_id,
                    req.event_id,
                    req.token_id,
                    req.side,
                    req.action,
                    req.mode,
                    req.price,
                    req.qty,
                    status,
                    external_id,
                    message,
                    self._now(),
                ),
            )
            row_id = cur.lastrowid
            if row_id is None:
                raise RuntimeError("failed to persist order")
            return int(row_id)

    def update_order_status(
        self,
        order_id: int,
        status: str,
        message: str = "",
        external_id: str | None = None,
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = ?, message = ?, external_id = COALESCE(?, external_id)
                WHERE id = ?
                """,
                (status, message, external_id, order_id),
            )

    def record_fill(
        self,
        order_id: int,
        strategy: str,
        market_id: str,
        event_id: str,
        token_id: str,
        side: str,
        price: float,
        qty: float,
        fee: float = 0.0,
        external_fill_id: str | None = None,
        raw_report_json: str = "",
    ) -> None:
        with self.conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO fills(
                        order_id, strategy, market_id, event_id, token_id, side,
                        price, qty, fee, external_fill_id, raw_report_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_id,
                        strategy,
                        market_id,
                        event_id,
                        token_id,
                        side,
                        price,
                        qty,
                        fee,
                        external_fill_id,
                        raw_report_json,
                        self._now(),
                    ),
                )
            except sqlite3.IntegrityError:
                if external_fill_id:
                    return
                raise

            self._apply_fill_to_position(
                conn,
                strategy=strategy,
                market_id=market_id,
                event_id=event_id,
                token_id=token_id,
                side=side,
                price=price,
                qty=qty,
            )

    def _apply_fill_to_position(
        self,
        conn: sqlite3.Connection,
        strategy: str,
        market_id: str,
        event_id: str,
        token_id: str,
        side: str,
        price: float,
        qty: float,
    ) -> None:
        row = conn.execute(
            """
            SELECT net_qty, avg_price, realized_pnl
            FROM positions
            WHERE strategy = ? AND market_id = ? AND token_id = ?
            """,
            (strategy, market_id, token_id),
        ).fetchone()

        net_qty = float(row["net_qty"]) if row else 0.0
        avg_price = float(row["avg_price"]) if row else 0.0
        realized = float(row["realized_pnl"]) if row else 0.0

        signed = qty if side.lower() == "buy" else -qty

        if net_qty == 0 or net_qty * signed > 0:
            new_qty = net_qty + signed
            if abs(new_qty) < 1e-9:
                new_avg = 0.0
            else:
                new_avg = ((avg_price * abs(net_qty)) + (price * abs(signed))) / abs(new_qty)
            new_realized = realized
        else:
            close_qty = min(abs(net_qty), abs(signed))
            direction = 1.0 if net_qty > 0 else -1.0
            pnl_delta = close_qty * (price - avg_price) * direction
            new_realized = realized + pnl_delta
            remainder = net_qty + signed
            if abs(remainder) < 1e-9:
                new_qty = 0.0
                new_avg = 0.0
            elif net_qty * remainder > 0:
                new_qty = remainder
                new_avg = avg_price
            else:
                new_qty = remainder
                new_avg = price

        conn.execute(
            """
            INSERT INTO positions(strategy, market_id, event_id, token_id, net_qty, avg_price, realized_pnl, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy, market_id, token_id)
            DO UPDATE SET
                event_id = excluded.event_id,
                net_qty = excluded.net_qty,
                avg_price = excluded.avg_price,
                realized_pnl = excluded.realized_pnl,
                updated_at = excluded.updated_at
            """,
            (strategy, market_id, event_id, token_id, new_qty, new_avg, new_realized, self._now()),
        )

    def get_market_prices(self) -> dict[tuple[str, str], float]:
        with self.conn() as conn:
            rows = conn.execute("SELECT market_id, yes_price, no_price FROM markets").fetchall()
        out: dict[tuple[str, str], float] = {}
        for row in rows:
            if row["yes_price"] is not None:
                out[(row["market_id"], "YES")] = float(row["yes_price"])
            if row["no_price"] is not None:
                out[(row["market_id"], "NO")] = float(row["no_price"])
        return out

    def get_positions(self) -> list[dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT strategy, market_id, event_id, token_id, net_qty, avg_price, realized_pnl, updated_at
                FROM positions
                ORDER BY strategy, market_id, token_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def order_filled_qty(self, order_id: int) -> float:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(qty), 0) AS v FROM fills WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        return float(row["v"] if row else 0.0)

    def total_realized_pnl(self) -> float:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) AS v FROM positions"
            ).fetchone()
        return float(row["v"] if row else 0.0)

    def strategy_notional(self, strategy: str) -> float:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(ABS(net_qty * avg_price)), 0) AS v
                FROM positions
                WHERE strategy = ?
                """,
                (strategy,),
            ).fetchone()
        return float(row["v"] if row else 0.0)

    def event_notional(self, event_id: str) -> float:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(ABS(net_qty * avg_price)), 0) AS v
                FROM positions
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return float(row["v"] if row else 0.0)

    def rejection_count(self) -> int:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM orders WHERE status = 'rejected'"
            ).fetchone()
        return int(row["c"] if row else 0)

    def order_stats(self) -> dict[str, float]:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(1) AS total,
                    SUM(CASE WHEN status IN ('filled', 'accepted') THEN 1 ELSE 0 END) AS filled,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected
                FROM orders
                """
            ).fetchone()
        total = int(row["total"] or 0)
        filled = int(row["filled"] or 0)
        rejected = int(row["rejected"] or 0)
        return {
            "total": total,
            "filled": filled,
            "rejected": rejected,
            "fill_rate": (filled / total) if total else 0.0,
            "rejection_rate": (rejected / total) if total else 0.0,
        }
