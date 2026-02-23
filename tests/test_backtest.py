from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from monomarket.backtest import BacktestEngine, BacktestExecutionConfig, BacktestRiskConfig
from monomarket.cli import app
from monomarket.db.storage import Storage
from monomarket.models import MarketView, Signal


def _seed_market_snapshots(storage: Storage) -> None:
    t1 = "2026-02-20T00:00:00+00:00"
    t2 = "2026-02-20T01:00:00+00:00"

    storage.upsert_markets(
        [
            MarketView(
                market_id="m1",
                canonical_id="c1",
                source="gamma",
                event_id="e1",
                question="Q1",
                status="open",
                neg_risk=False,
                liquidity=1000,
                volume=100,
                yes_price=0.40,
                no_price=0.60,
                mid_price=0.40,
            ),
            MarketView(
                market_id="m2",
                canonical_id="c2",
                source="gamma",
                event_id="e2",
                question="Q2",
                status="open",
                neg_risk=False,
                liquidity=900,
                volume=90,
                yes_price=0.30,
                no_price=0.70,
                mid_price=0.30,
            ),
        ],
        snapshot_at=t1,
    )
    storage.upsert_markets(
        [
            MarketView(
                market_id="m1",
                canonical_id="c1",
                source="gamma",
                event_id="e1",
                question="Q1",
                status="open",
                neg_risk=False,
                liquidity=1000,
                volume=100,
                yes_price=0.60,
                no_price=0.40,
                mid_price=0.60,
            ),
            MarketView(
                market_id="m2",
                canonical_id="c2",
                source="gamma",
                event_id="e2",
                question="Q2",
                status="open",
                neg_risk=False,
                liquidity=900,
                volume=90,
                yes_price=0.20,
                no_price=0.80,
                mid_price=0.20,
            ),
        ],
        snapshot_at=t2,
    )


def _seed_signals(storage: Storage) -> None:
    storage.insert_signals(
        [
            Signal(
                strategy="s1",
                market_id="m1",
                event_id="e1",
                side="buy",
                score=1.0,
                confidence=0.8,
                target_price=0.40,
                size_hint=10.0,
                rationale="open-e1",
            )
        ],
        created_at="2026-02-20T00:10:00+00:00",
    )
    storage.insert_signals(
        [
            Signal(
                strategy="s1",
                market_id="m1",
                event_id="e1",
                side="sell",
                score=1.0,
                confidence=0.8,
                target_price=0.60,
                size_hint=10.0,
                rationale="close-e1",
            )
        ],
        created_at="2026-02-20T00:50:00+00:00",
    )
    storage.insert_signals(
        [
            Signal(
                strategy="s1",
                market_id="m2",
                event_id="e2",
                side="buy",
                score=1.0,
                confidence=0.8,
                target_price=0.30,
                size_hint=5.0,
                rationale="open-e2",
            )
        ],
        created_at="2026-02-20T00:20:00+00:00",
    )
    storage.insert_signals(
        [
            Signal(
                strategy="s1",
                market_id="m2",
                event_id="e2",
                side="sell",
                score=1.0,
                confidence=0.8,
                target_price=0.20,
                size_hint=5.0,
                rationale="close-e2",
            )
        ],
        created_at="2026-02-20T00:55:00+00:00",
    )


def test_backtest_engine_attribution(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(slippage_bps=0.0, fee_bps=0.0),
    ).run(["s1"], from_ts="2026-02-20T00:00:00Z", to_ts="2026-02-20T02:00:00Z")

    assert report.total_signals == 4
    assert report.executed_signals == 4
    assert report.rejected_signals == 0
    assert len(report.results) == 1
    r = report.results[0]
    assert r.strategy == "s1"
    assert abs(r.pnl - 1.5) < 1e-9
    assert r.trade_count == 4
    assert r.winrate == 0.5
    assert r.max_drawdown >= 0.0

    assert len(report.event_results) == 2
    event_map = {(x.strategy, x.event_id): x for x in report.event_results}
    assert abs(event_map[("s1", "e1")].pnl - 2.0) < 1e-9
    assert abs(event_map[("s1", "e2")].pnl + 0.5) < 1e-9
    assert event_map[("s1", "e1")].trade_count == 2
    assert event_map[("s1", "e2")].trade_count == 2

    assert len(report.replay) == 4
    assert report.replay[0].event_id == "e1"
    assert report.replay[-1].event_id == "e2"
    assert abs(report.replay[-1].realized_change + 0.5) < 1e-9
    assert all(x.risk_allowed for x in report.replay)
    assert all(x.risk_reason == "ok" for x in report.replay)


def test_backtest_risk_replay_rejection(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(slippage_bps=0.0, fee_bps=0.0),
        risk=BacktestRiskConfig(
            max_daily_loss=1000.0,
            max_strategy_notional=1000.0,
            max_event_notional=5.0,
            circuit_breaker_rejections=5,
        ),
    ).run(["s1"], from_ts="2026-02-20T00:00:00Z", to_ts="2026-02-20T02:00:00Z")

    assert report.total_signals == 4
    assert report.executed_signals == 3
    assert report.rejected_signals == 1

    rejected_rows = [x for x in report.replay if not x.risk_allowed]
    assert len(rejected_rows) == 1
    assert "event notional limit exceeded" in rejected_rows[0].risk_reason
    assert abs(rejected_rows[0].risk_max_event_notional - 5.0) < 1e-9
    assert rejected_rows[0].risk_rejections_before == 0
    assert rejected_rows[0].risk_allowed is False


def test_cli_backtest_command(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                f"  db_path: {db}",
                "trading:",
                "  mode: paper",
            ]
        )
    )

    json_out = tmp_path / "artifacts" / "backtest.json"
    csv_out = tmp_path / "artifacts" / "replay.csv"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "backtest",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--slippage-bps",
            "0",
            "--out-json",
            str(json_out),
            "--out-replay-csv",
            str(csv_out),
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Backtest attribution" in res.output
    assert "Backtest event attribution" in res.output
    assert "Backtest replay ledger" in res.output
    assert "json exported" in res.output
    assert "replay csv exported" in res.output
    assert "s1" in res.output

    payload = json.loads(json_out.read_text())
    assert payload["total_signals"] == 4
    assert payload["executed_signals"] == 4
    assert payload["rejected_signals"] == 0
    assert len(payload["results"]) == 1
    assert len(payload["event_results"]) == 2
    assert len(payload["replay"]) == 4

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert rows[0]["strategy"] == "s1"
    assert rows[0]["event_id"] == "e1"
    assert rows[0]["risk_allowed"] == "True"
    assert rows[0]["risk_reason"] == "ok"
