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
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ingestion_state (
    source TEXT PRIMARY KEY,
    checkpoint_ts TEXT,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_failures INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
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
    created_at TEXT NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(id)
);
CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy);

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
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs(source, status, started_at, finished_at, rows_ingested, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, status, started_at, finished_at, rows_ingested, error),
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
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO fills(order_id, strategy, market_id, event_id, token_id, side, price, qty, fee, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    self._now(),
                ),
            )
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
            rows = conn.execute("""
                SELECT strategy, market_id, event_id, token_id, net_qty, avg_price, realized_pnl, updated_at
                FROM positions
                ORDER BY strategy, market_id, token_id
                """).fetchall()
        return [dict(row) for row in rows]

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
            row = conn.execute("""
                SELECT
                    COUNT(1) AS total,
                    SUM(CASE WHEN status IN ('filled', 'accepted') THEN 1 ELSE 0 END) AS filled,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected
                FROM orders
                """).fetchone()
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
