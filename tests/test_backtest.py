from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from monomarket.backtest import BacktestEngine, BacktestExecutionConfig
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
            )
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
            )
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
                rationale="open",
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
                rationale="close",
            )
        ],
        created_at="2026-02-20T00:50:00+00:00",
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

    assert report.total_signals == 2
    assert len(report.results) == 1
    r = report.results[0]
    assert r.strategy == "s1"
    assert abs(r.pnl - 2.0) < 1e-9
    assert r.trade_count == 2
    assert r.winrate == 1.0
    assert r.max_drawdown >= 0.0


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
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Backtest attribution" in res.output
    assert "s1" in res.output
