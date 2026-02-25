#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from monomarket.backtest import BacktestEngine, BacktestExecutionConfig, BacktestRiskConfig
from monomarket.config import Settings, load_settings
from monomarket.db import Storage
from monomarket.models import MarketView, Signal
from monomarket.signals.strategies import (
    S1CrossVenueScanner,
    S2NegRiskRebalance,
    S4LowProbYesBasket,
    S8NoCarryTailHedge,
)


@dataclass(slots=True)
class GammaMarketMeta:
    market_id: str
    event_id: str
    canonical_id: str
    question: str
    outcomes: list[str]
    clob_token_ids: list[str]
    yes_token_id: str
    no_token_id: str
    neg_risk: bool
    liquidity: float
    volume: float
    end_ts: int | None


@dataclass(slots=True)
class MarketSeries:
    meta: GammaMarketMeta
    epochs: list[int]
    yes_prices: list[float | None]
    no_prices: list[float | None]


def _parse_iso_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extract_event_id(item: dict[str, Any]) -> str:
    for key in ("event_id", "eventId", "questionID", "conditionId"):
        value = item.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text

    events = item.get("events")
    if isinstance(events, list) and events:
        first = events[0]
        if isinstance(first, dict):
            for key in ("id", "event_id", "eventId", "ticker", "slug"):
                value = first.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
    return "unknown"


def _extract_yes_no_token_ids(item: dict[str, Any]) -> tuple[str | None, str | None, list[str], list[str]]:
    outcomes_raw = _parse_json_list(item.get("outcomes"))
    outcomes = [str(x) for x in outcomes_raw]
    token_ids_raw = _parse_json_list(item.get("clobTokenIds"))
    token_ids = [str(x) for x in token_ids_raw]

    yes_token: str | None = None
    no_token: str | None = None

    if len(outcomes) == len(token_ids):
        for idx, outcome in enumerate(outcomes):
            key = outcome.strip().lower()
            if key == "yes":
                yes_token = token_ids[idx]
            elif key == "no":
                no_token = token_ids[idx]

    if (yes_token is None or no_token is None) and len(token_ids) >= 2:
        yes_token = yes_token or token_ids[0]
        no_token = no_token or token_ids[1]

    return yes_token, no_token, outcomes, token_ids


def _request_json_with_retries(
    *,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_base_sec: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_sec)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            backoff = max(0.0, backoff_base_sec) * (2**attempt)
            if backoff > 0:
                time.sleep(backoff)
    if last_exc is None:
        raise RuntimeError(f"request failed without exception: {url}")
    raise RuntimeError(f"request failed: {url} params={params} err={last_exc}") from last_exc


def fetch_gamma_market_meta(
    *,
    session: requests.Session,
    settings: Settings,
    market_limit: int,
    from_dt: datetime,
    page_limit: int = 200,
    max_pages: int = 200,
) -> list[GammaMarketMeta]:
    out: list[GammaMarketMeta] = []
    offset = 0
    page_no = 0
    from_epoch = int(from_dt.timestamp())

    while len(out) < market_limit and page_no < max_pages:
        limit = min(page_limit, market_limit - len(out))
        params = {
            "limit": limit,
            "offset": offset,
            "archived": "false",
            "active": "true",
            "closed": "false",
        }
        payload = _request_json_with_retries(
            session=session,
            url=f"{settings.data.gamma_base_url.rstrip('/')}/markets",
            params=params,
            timeout_sec=settings.data.timeout_sec,
            max_retries=settings.data.max_retries,
            backoff_base_sec=settings.data.backoff_base_sec,
        )
        if not isinstance(payload, list) or not payload:
            break

        for item in payload:
            if not isinstance(item, dict):
                continue
            market_id = str(item.get("id") or item.get("market_id") or "").strip()
            if not market_id:
                continue

            yes_token, no_token, outcomes, token_ids = _extract_yes_no_token_ids(item)
            if not yes_token or not no_token:
                continue

            question = str(item.get("question") or item.get("title") or market_id)
            event_id = _extract_event_id(item)
            canonical_id = str(
                item.get("canonical_id")
                or item.get("groupItemTitle")
                or item.get("slug")
                or f"{event_id}:{question[:60]}"
            )

            end_ts: int | None = None
            end_dt = _parse_iso_ts(item.get("endDate") or item.get("endDateIso"))
            if end_dt is None:
                events = item.get("events")
                if isinstance(events, list) and events and isinstance(events[0], dict):
                    end_dt = _parse_iso_ts(events[0].get("endDate"))
            if end_dt is not None:
                end_ts = int(end_dt.timestamp())

            # Keep markets that were still alive during the requested backfill window.
            if end_ts is not None and end_ts < from_epoch:
                continue

            out.append(
                GammaMarketMeta(
                    market_id=market_id,
                    event_id=event_id,
                    canonical_id=canonical_id,
                    question=question,
                    outcomes=outcomes,
                    clob_token_ids=token_ids,
                    yes_token_id=yes_token,
                    no_token_id=no_token,
                    neg_risk=_as_bool(item.get("negRisk") or item.get("neg_risk"), False),
                    liquidity=_as_float(item.get("liquidity") or item.get("liquidityNum"), 0.0),
                    volume=_as_float(item.get("volume") or item.get("volumeNum"), 0.0),
                    end_ts=end_ts,
                )
            )
            if len(out) >= market_limit:
                break

        if len(payload) < limit:
            break
        offset += limit
        page_no += 1

    return out[:market_limit]


def _fetch_token_history(
    *,
    session: requests.Session,
    settings: Settings,
    token_id: str,
    fidelity: int,
) -> dict[int, float]:
    payload = _request_json_with_retries(
        session=session,
        url=f"{settings.data.clob_base_url.rstrip('/')}/prices-history",
        params={
            "market": token_id,
            "interval": "max",
            "fidelity": fidelity,
        },
        timeout_sec=settings.data.timeout_sec,
        max_retries=settings.data.max_retries,
        backoff_base_sec=settings.data.backoff_base_sec,
    )

    if not isinstance(payload, dict):
        return {}

    history = payload.get("history")
    if not isinstance(history, list):
        return {}

    out: dict[int, float] = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        ts_raw = row.get("t")
        p_raw = row.get("p")
        try:
            ts = int(ts_raw)
            price = float(p_raw)
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        out[ts] = max(0.0, min(1.0, price))
    return out


def backfill_snapshots_from_history(
    *,
    storage: Storage,
    settings: Settings,
    metas: list[GammaMarketMeta],
    from_dt: datetime,
    to_dt: datetime,
    fidelity: int,
    request_sleep_sec: float,
) -> tuple[list[MarketSeries], dict[str, Any]]:
    from_iso = from_dt.isoformat()
    to_iso = to_dt.isoformat()
    from_epoch = int(from_dt.timestamp())
    to_epoch = int(to_dt.timestamp())

    series_out: list[MarketSeries] = []
    inserted_rows = 0
    markets_with_history = 0
    request_count = 0
    failed_markets: list[str] = []

    with requests.Session() as session:
        for idx, meta in enumerate(metas, start=1):
            try:
                yes_hist = _fetch_token_history(
                    session=session,
                    settings=settings,
                    token_id=meta.yes_token_id,
                    fidelity=fidelity,
                )
                request_count += 1
                if request_sleep_sec > 0:
                    time.sleep(request_sleep_sec)

                no_hist = _fetch_token_history(
                    session=session,
                    settings=settings,
                    token_id=meta.no_token_id,
                    fidelity=fidelity,
                )
                request_count += 1
            except Exception:
                failed_markets.append(meta.market_id)
                if request_sleep_sec > 0:
                    time.sleep(request_sleep_sec)
                continue

            timestamps = sorted(set(yes_hist.keys()) | set(no_hist.keys()))
            if not timestamps:
                if request_sleep_sec > 0:
                    time.sleep(request_sleep_sec)
                continue

            epochs: list[int] = []
            yes_prices: list[float | None] = []
            no_prices: list[float | None] = []
            rows_for_db: list[tuple[Any, ...]] = []

            with storage.conn() as conn:
                existing = {
                    str(r["captured_at"])
                    for r in conn.execute(
                        """
                        SELECT captured_at
                        FROM market_snapshots
                        WHERE source = ? AND market_id = ? AND captured_at >= ? AND captured_at <= ?
                        """,
                        ("gamma", meta.market_id, from_iso, to_iso),
                    ).fetchall()
                }

                for ts in timestamps:
                    if ts < from_epoch or ts > to_epoch:
                        continue

                    y = yes_hist.get(ts)
                    n = no_hist.get(ts)

                    if y is None and n is not None:
                        y = max(0.0, min(1.0, 1.0 - n))
                    if n is None and y is not None:
                        n = max(0.0, min(1.0, 1.0 - y))

                    if y is None and n is None:
                        continue

                    captured_at = datetime.fromtimestamp(ts, tz=UTC).isoformat()
                    epochs.append(ts)
                    yes_prices.append(y)
                    no_prices.append(n)

                    if captured_at in existing:
                        continue

                    mid = y if y is not None else (1.0 - float(n)) if n is not None else None
                    rows_for_db.append(
                        (
                            "gamma",
                            meta.market_id,
                            meta.event_id,
                            y,
                            n,
                            mid,
                            meta.liquidity,
                            meta.volume,
                            captured_at,
                        )
                    )

                if rows_for_db:
                    conn.executemany(
                        """
                        INSERT INTO market_snapshots(
                            source, market_id, event_id, yes_price, no_price, mid_price,
                            liquidity, volume, captured_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows_for_db,
                    )

            if epochs:
                markets_with_history += 1
                series_out.append(
                    MarketSeries(
                        meta=meta,
                        epochs=epochs,
                        yes_prices=yes_prices,
                        no_prices=no_prices,
                    )
                )
                inserted_rows += len(rows_for_db)

            if request_sleep_sec > 0:
                time.sleep(request_sleep_sec)

            if idx % 20 == 0:
                print(
                    "[price-backfill] "
                    f"processed={idx}/{len(metas)} with_history={markets_with_history} "
                    f"inserted_rows={inserted_rows}"
                )

    stats = {
        "markets_total": len(metas),
        "markets_with_history": markets_with_history,
        "snapshot_rows_inserted": inserted_rows,
        "history_requests": request_count,
        "failed_market_count": len(failed_markets),
        "failed_market_ids": failed_markets[:30],
    }
    return series_out, stats


def _strategy_registry() -> dict[str, Any]:
    return {
        "s1": S1CrossVenueScanner(),
        "s2": S2NegRiskRebalance(),
        "s4": S4LowProbYesBasket(),
        "s8": S8NoCarryTailHedge(),
    }


def _existing_signal_keys(storage: Storage, created_at: str, strategies: list[str]) -> set[tuple[str, str, str, str]]:
    placeholders = ",".join("?" for _ in strategies)
    query = (
        "SELECT strategy, market_id, event_id, side FROM signals "
        f"WHERE created_at = ? AND strategy IN ({placeholders})"
    )
    params: list[Any] = [created_at, *strategies]
    with storage.conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return {
        (
            str(row["strategy"]).lower(),
            str(row["market_id"]),
            str(row["event_id"]),
            str(row["side"]).lower(),
        )
        for row in rows
    }


def _dedupe_signals(signals: list[Signal]) -> list[Signal]:
    by_key: dict[tuple[str, str, str, str], Signal] = {}
    for signal in signals:
        key = (
            signal.strategy.lower(),
            signal.market_id,
            signal.event_id,
            signal.side.lower(),
        )
        prev = by_key.get(key)
        if prev is None or signal.score > prev.score:
            by_key[key] = signal
    return list(by_key.values())


def _build_market_views_at(
    *,
    series_rows: list[MarketSeries],
    ts_epoch: int,
) -> list[MarketView]:
    out: list[MarketView] = []
    for series in series_rows:
        idx = bisect_right(series.epochs, ts_epoch) - 1
        if idx < 0:
            continue

        status = "open"
        if series.meta.end_ts is not None and ts_epoch > series.meta.end_ts:
            status = "closed"
        if status != "open":
            continue

        yes_px = series.yes_prices[idx]
        no_px = series.no_prices[idx]
        mid_px = yes_px if yes_px is not None else None

        out.append(
            MarketView(
                source="gamma",
                market_id=series.meta.market_id,
                canonical_id=series.meta.canonical_id,
                event_id=series.meta.event_id,
                question=series.meta.question,
                status=status,
                neg_risk=series.meta.neg_risk,
                liquidity=series.meta.liquidity,
                volume=series.meta.volume,
                yes_price=yes_px,
                no_price=no_px,
                best_bid=None,
                best_ask=None,
                mid_price=mid_px,
            )
        )
    return out


def backfill_signals_from_series(
    *,
    storage: Storage,
    settings: Settings,
    series_rows: list[MarketSeries],
    from_dt: datetime,
    to_dt: datetime,
    step_hours: float,
    strategies: list[str],
    dataset_id: str,
    replace_existing_backfill: bool,
) -> dict[str, Any]:
    if replace_existing_backfill:
        placeholders = ",".join("?" for _ in strategies)
        delete_query = (
            "DELETE FROM signals WHERE created_at >= ? AND created_at <= ? "
            f"AND strategy IN ({placeholders}) "
            "AND payload_json LIKE ?"
        )
        delete_params: list[Any] = [
            from_dt.isoformat(),
            to_dt.isoformat(),
            *strategies,
            '%"backfill_source": "history"%',
        ]
        with storage.conn() as conn:
            conn.execute(delete_query, delete_params)

    registry = _strategy_registry()

    cursor = from_dt
    step = timedelta(hours=step_hours)
    if step <= timedelta(0):
        raise ValueError("step_hours must be > 0")

    inserted_total = 0
    generated_total = 0
    by_strategy_inserted: dict[str, int] = {s: 0 for s in strategies}
    by_strategy_generated: dict[str, int] = {s: 0 for s in strategies}

    step_index = 0
    while cursor <= to_dt:
        ts_iso = cursor.isoformat()
        ts_epoch = int(cursor.timestamp())
        market_views = _build_market_views_at(series_rows=series_rows, ts_epoch=ts_epoch)

        generated: list[Signal] = []
        for strategy_name in strategies:
            strategy_impl = registry.get(strategy_name)
            if strategy_impl is None:
                continue
            cfg = settings.strategies.get(strategy_name, {})
            strategy_signals = strategy_impl.generate(market_views, cfg)
            by_strategy_generated[strategy_name] = by_strategy_generated.get(strategy_name, 0) + len(
                strategy_signals
            )
            generated.extend(strategy_signals)

        generated = _dedupe_signals(generated)
        generated_total += len(generated)

        existing_keys = _existing_signal_keys(storage, ts_iso, strategies)
        to_insert: list[Signal] = []
        for signal in generated:
            key = (
                signal.strategy.lower(),
                signal.market_id,
                signal.event_id,
                signal.side.lower(),
            )
            if key in existing_keys:
                continue

            payload = dict(signal.payload or {})
            payload["backfill_source"] = "history"
            payload["dataset_id"] = dataset_id
            payload["backfill_ts"] = ts_iso

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
        for signal in to_insert:
            by_strategy_inserted[signal.strategy] = by_strategy_inserted.get(signal.strategy, 0) + 1

        step_index += 1
        if step_index % 25 == 0:
            print(
                "[signal-backfill] "
                f"step={step_index} ts={ts_iso} markets={len(market_views)} "
                f"inserted_total={inserted_total}"
            )

        cursor += step

    with storage.conn() as conn:
        row = conn.execute(
            """
            SELECT MIN(created_at) AS min_created_at, MAX(created_at) AS max_created_at
            FROM signals
            WHERE strategy IN ({})
            """.format(
                ",".join("?" for _ in strategies)
            ),
            strategies,
        ).fetchone()

    return {
        "signal_rows_generated": generated_total,
        "signal_rows_inserted": inserted_total,
        "signal_rows_generated_by_strategy": by_strategy_generated,
        "signal_rows_inserted_by_strategy": by_strategy_inserted,
        "signal_min_created_at": str(row["min_created_at"]) if row and row["min_created_at"] else None,
        "signal_max_created_at": str(row["max_created_at"]) if row and row["max_created_at"] else None,
    }


def rolling_summary(
    *,
    storage: Storage,
    settings: Settings,
    strategies: list[str],
    from_dt: datetime,
    to_dt: datetime,
    window_hours: float,
    step_hours: float,
) -> dict[str, Any]:
    window_delta = timedelta(hours=window_hours)
    step_delta = timedelta(hours=step_hours)
    if window_delta <= timedelta(0):
        raise ValueError("window_hours must be > 0")
    if step_delta <= timedelta(0):
        raise ValueError("step_hours must be > 0")

    engine = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(),
        risk=BacktestRiskConfig(
            max_daily_loss=settings.risk.max_daily_loss,
            max_strategy_notional=settings.risk.max_strategy_notional,
            max_event_notional=settings.risk.max_event_notional,
            circuit_breaker_rejections=settings.risk.circuit_breaker_rejections,
        ),
    )

    cursor = from_dt
    run_count = 0
    empty_window_count = 0
    total_signals = 0
    executed_signals = 0

    while cursor < to_dt:
        run_to = min(to_dt, cursor + window_delta)
        report = engine.run(strategies, from_ts=cursor.isoformat(), to_ts=run_to.isoformat())

        run_count += 1
        total_signals += report.total_signals
        executed_signals += report.executed_signals
        if report.total_signals <= 0:
            empty_window_count += 1

        cursor += step_delta

    non_empty = max(0, run_count - empty_window_count)
    return {
        "run_count": run_count,
        "empty_window_count": empty_window_count,
        "non_empty_window_count": non_empty,
        "non_empty_window_ratio": (non_empty / run_count) if run_count > 0 else 0.0,
        "total_signals": total_signals,
        "executed_signals": executed_signals,
        "execution_rate": (executed_signals / total_signals) if total_signals > 0 else 0.0,
    }


def _build_dataset_id(
    *,
    prefix: str,
    generated_at: datetime,
    params_for_hash: dict[str, Any],
) -> str:
    canonical = json.dumps(params_for_hash, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    ts = generated_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}-{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill historical market_snapshots from CLOB price history and "
            "rebuild historical strategy signals."
        )
    )
    parser.add_argument("--config", default=None, help="config yaml path")
    parser.add_argument("--window-days", type=int, default=180, help="history window days")
    parser.add_argument("--step-hours", type=float, default=12.0, help="signal rebuild step hours")
    parser.add_argument(
        "--fidelity",
        type=int,
        default=720,
        help="CLOB prices-history fidelity (minutes, e.g. 60/360/720/1440)",
    )
    parser.add_argument("--market-limit", type=int, default=220, help="max gamma markets to backfill")
    parser.add_argument(
        "--strategies",
        default="s2,s4,s8",
        help="comma-separated strategies for historical signal rebuild",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="hist6m",
        help="dataset id prefix",
    )
    parser.add_argument(
        "--rolling-window-hours",
        type=float,
        default=24.0,
        help="rolling backtest window hours (before/after compare)",
    )
    parser.add_argument(
        "--rolling-step-hours",
        type=float,
        default=12.0,
        help="rolling backtest step hours (before/after compare)",
    )
    parser.add_argument(
        "--request-sleep-sec",
        type=float,
        default=0.05,
        help="sleep between history requests to reduce API pressure",
    )
    parser.add_argument(
        "--replace-existing-backfill",
        action="store_true",
        default=True,
        help="remove old history-backfill signals in window before insert (default: on)",
    )
    parser.add_argument(
        "--no-replace-existing-backfill",
        dest="replace_existing_backfill",
        action="store_false",
        help="do not delete old history-backfill signals before insert",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    storage = Storage(settings.app.db_path)
    storage.init_db()

    now = datetime.now(UTC)
    from_dt = now - timedelta(days=max(1, args.window_days))
    to_dt = now

    strategies = [s.strip().lower() for s in str(args.strategies).split(",") if s.strip()]
    if not strategies:
        raise SystemExit("strategies must not be empty")

    params_for_hash = {
        "window_days": args.window_days,
        "step_hours": args.step_hours,
        "fidelity": args.fidelity,
        "market_limit": args.market_limit,
        "strategies": strategies,
        "rolling_window_hours": args.rolling_window_hours,
        "rolling_step_hours": args.rolling_step_hours,
    }
    dataset_id = _build_dataset_id(
        prefix=args.dataset_prefix,
        generated_at=now,
        params_for_hash=params_for_hash,
    )

    print(
        "[start] "
        f"dataset_id={dataset_id} from={from_dt.isoformat()} to={to_dt.isoformat()} "
        f"strategies={','.join(strategies)}"
    )

    before = rolling_summary(
        storage=storage,
        settings=settings,
        strategies=strategies,
        from_dt=from_dt,
        to_dt=to_dt,
        window_hours=args.rolling_window_hours,
        step_hours=args.rolling_step_hours,
    )
    print(
        "[rolling-before] "
        f"non_empty={before['non_empty_window_count']}/{before['run_count']} "
        f"({before['non_empty_window_ratio']:.2%}) "
        f"execution_rate={before['execution_rate']:.2%}"
    )

    with requests.Session() as session:
        metas = fetch_gamma_market_meta(
            session=session,
            settings=settings,
            market_limit=max(1, args.market_limit),
            from_dt=from_dt,
        )

    print(f"[meta] fetched={len(metas)} markets with yes/no tokens")

    series_rows, price_stats = backfill_snapshots_from_history(
        storage=storage,
        settings=settings,
        metas=metas,
        from_dt=from_dt,
        to_dt=to_dt,
        fidelity=max(1, args.fidelity),
        request_sleep_sec=max(0.0, args.request_sleep_sec),
    )

    print(
        "[price-backfill-done] "
        f"markets_with_history={price_stats['markets_with_history']}/{price_stats['markets_total']} "
        f"inserted_rows={price_stats['snapshot_rows_inserted']} "
        f"requests={price_stats['history_requests']}"
    )

    signal_stats = backfill_signals_from_series(
        storage=storage,
        settings=settings,
        series_rows=series_rows,
        from_dt=from_dt,
        to_dt=to_dt,
        step_hours=args.step_hours,
        strategies=strategies,
        dataset_id=dataset_id,
        replace_existing_backfill=bool(args.replace_existing_backfill),
    )

    print(
        "[signal-backfill-done] "
        f"inserted={signal_stats['signal_rows_inserted']} "
        f"generated={signal_stats['signal_rows_generated']} "
        f"min={signal_stats['signal_min_created_at']} "
        f"max={signal_stats['signal_max_created_at']}"
    )

    after = rolling_summary(
        storage=storage,
        settings=settings,
        strategies=strategies,
        from_dt=from_dt,
        to_dt=to_dt,
        window_hours=args.rolling_window_hours,
        step_hours=args.rolling_step_hours,
    )
    print(
        "[rolling-after] "
        f"non_empty={after['non_empty_window_count']}/{after['run_count']} "
        f"({after['non_empty_window_ratio']:.2%}) "
        f"execution_rate={after['execution_rate']:.2%}"
    )

    coverage_ok = False
    with storage.conn() as conn:
        row = conn.execute(
            "SELECT MIN(created_at) AS min_ts, MAX(created_at) AS max_ts FROM signals"
        ).fetchone()
        min_ts = str(row["min_ts"]) if row and row["min_ts"] else None
        max_ts = str(row["max_ts"]) if row and row["max_ts"] else None

    if min_ts:
        min_dt = _parse_iso_ts(min_ts)
        if min_dt is not None and min_dt <= (now - timedelta(days=170)):
            coverage_ok = True

    dataset_payload = {
        "dataset_id": dataset_id,
        "generated_at": now.isoformat(),
        "window": {
            "from": from_dt.isoformat(),
            "to": to_dt.isoformat(),
            "days": int(args.window_days),
        },
        "params": params_for_hash,
        "strategies": strategies,
        "price_backfill": price_stats,
        "signal_backfill": signal_stats,
        "signal_time_coverage": {
            "min_created_at": min_ts,
            "max_created_at": max_ts,
            "meets_ge_170d_requirement": coverage_ok,
        },
        "rolling_compare": {
            "before": before,
            "after": after,
            "delta": {
                "non_empty_window_ratio": after["non_empty_window_ratio"]
                - before["non_empty_window_ratio"],
                "execution_rate": after["execution_rate"] - before["execution_rate"],
            },
        },
        "notes": {
            "history_data_source": "gamma metadata + clob /prices-history",
            "price_forgery": "none",
            "s1_note": "s1 requires multi-source same-canonical snapshots; this run focuses on selected strategies.",
        },
    }

    out_dir = Path("artifacts/backtest/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}.json"
    out_path.write_text(json.dumps(dataset_payload, ensure_ascii=False, indent=2) + "\n")

    print(f"[dataset] {out_path}")
    print(
        "[done] "
        f"dataset_id={dataset_id} coverage_ok={str(coverage_ok).lower()} "
        f"signals_min={min_ts} signals_max={max_ts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
