from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from monomarket.backtest import (
    BACKTEST_ARTIFACT_CHECKSUM_ALGO,
    BACKTEST_ARTIFACT_SCHEMA_VERSION,
    BacktestEngine,
    BacktestExecutionConfig,
    BacktestRiskConfig,
    validate_backtest_json_artifact,
    verify_backtest_json_artifact_checksum,
)
from monomarket.cli import app
from monomarket.config import AppSettings, DataSettings, RiskSettings, Settings, TradingSettings
from monomarket.db.storage import Storage
from monomarket.models import MarketView, Signal
from monomarket.signals import SignalEngine


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


def _seed_extreme_s8_markets(storage: Storage) -> None:
    rows = [
        MarketView(
            market_id=f"m{i}",
            canonical_id=f"c{i}",
            source="gamma",
            event_id=f"e{i}",
            question=f"Q{i}",
            status="open",
            neg_risk=False,
            liquidity=100000.0,
            volume=1000.0,
            yes_price=0.01 + (i * 0.005),
            no_price=0.99 - (i * 0.005),
            mid_price=0.01 + (i * 0.005),
        )
        for i in range(1, 7)
    ]
    storage.upsert_markets(rows, snapshot_at="2026-02-24T01:39:20+00:00")


def _default_settings_for_test(db_path: str) -> Settings:
    return Settings(
        app=AppSettings(db_path=db_path, log_level="INFO"),
        trading=TradingSettings(
            mode="paper",
            enable_live_trading=False,
            require_manual_confirm=True,
            kill_switch=False,
        ),
        risk=RiskSettings(
            max_daily_loss=250.0,
            max_strategy_notional=1000.0,
            max_event_notional=1500.0,
            circuit_breaker_rejections=5,
        ),
        data=DataSettings(),
        strategies={
            "s8": {
                "yes_price_max_for_no": 0.25,
                "hedge_budget_ratio": 0.15,
            }
        },
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

    assert report.schema_version == BACKTEST_ARTIFACT_SCHEMA_VERSION
    assert report.total_signals == 4
    assert report.executed_signals == 4
    assert report.rejected_signals == 0
    assert report.execution_config["enable_partial_fill"] is False
    assert report.execution_config["enable_fill_probability"] is False
    assert report.risk_config["max_event_notional"] == 1e18
    assert len(report.results) == 1
    r = report.results[0]
    assert r.strategy == "s1"
    assert abs(r.pnl - 1.5) < 1e-9
    assert r.trade_count == 4
    assert r.winrate == 0.5
    assert r.closed_winrate == 0.5
    assert r.closed_sample_count == 2
    assert r.mtm_winrate == 0.5
    assert r.mtm_wins == 1
    assert r.mtm_losses == 1
    assert r.mtm_sample_count == 2
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
    assert all(abs(x.fill_ratio - 1.0) < 1e-9 for x in report.replay)
    assert all(abs(x.fill_probability - 1.0) < 1e-9 for x in report.replay)


def test_backtest_partial_fill_model(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(
            slippage_bps=0.0,
            fee_bps=0.0,
            enable_partial_fill=True,
            liquidity_full_fill=1000.0,
            min_fill_ratio=0.0,
        ),
    ).run(["s1"], from_ts="2026-02-20T00:00:00Z", to_ts="2026-02-20T02:00:00Z")

    assert report.total_signals == 4
    assert report.executed_signals == 4
    assert report.rejected_signals == 0

    r = report.results[0]
    assert abs(r.pnl - 1.55) < 1e-9

    # m2 liquidity=900 => fill ratio 90%
    m2_rows = [x for x in report.replay if x.market_id == "m2"]
    assert len(m2_rows) == 2
    assert all(abs(x.fill_ratio - 0.9) < 1e-9 for x in m2_rows)
    assert all(abs(x.executed_qty - 4.5) < 1e-9 for x in m2_rows)


def test_backtest_fill_probability_model(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(
            slippage_bps=0.0,
            fee_bps=0.0,
            enable_partial_fill=True,
            liquidity_full_fill=1000.0,
            min_fill_ratio=0.0,
            enable_fill_probability=True,
            min_fill_probability=0.05,
        ),
    ).run(["s1"], from_ts="2026-02-20T00:00:00Z", to_ts="2026-02-20T02:00:00Z")

    assert report.total_signals == 4
    assert report.executed_signals == 4
    assert report.rejected_signals == 0
    assert report.execution_config["enable_fill_probability"] is True
    assert report.execution_config["enable_partial_fill"] is True

    # m2 liquidity=900 => partial ratio 0.9; buy prob=0.855; sell prob=0.9
    m2_buy = [x for x in report.replay if x.market_id == "m2" and x.side == "buy"][0]
    m2_sell = [x for x in report.replay if x.market_id == "m2" and x.side == "sell"][0]
    assert abs(m2_buy.fill_probability - 0.855) < 1e-9
    assert abs(m2_sell.fill_probability - 0.9) < 1e-9
    assert abs(m2_buy.fill_ratio - (0.9 * 0.855)) < 1e-9
    assert abs(m2_sell.fill_ratio - (0.9 * 0.9)) < 1e-9

    # deterministic pnl under current toy execution model
    assert abs(report.results[0].pnl - 1.51525) < 1e-9


def test_backtest_dynamic_slippage_model(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_market_snapshots(storage)
    _seed_signals(storage)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(
            slippage_bps=0.0,
            fee_bps=0.0,
            enable_dynamic_slippage=True,
            spread_slippage_weight_bps=50.0,
            liquidity_slippage_weight_bps=100.0,
            liquidity_reference=1000.0,
        ),
    ).run(["s1"], from_ts="2026-02-20T00:00:00Z", to_ts="2026-02-20T02:00:00Z")

    assert report.total_signals == 4
    assert report.executed_signals == 4
    assert report.rejected_signals == 0

    # m1 liquidity=1000 => no dynamic addition; m2 liquidity=900 => +10 bps
    m1_rows = [x for x in report.replay if x.market_id == "m1"]
    m2_rows = [x for x in report.replay if x.market_id == "m2"]
    assert all(abs(x.slippage_bps_applied - 0.0) < 1e-9 for x in m1_rows)
    assert all(abs(x.slippage_bps_applied - 10.0) < 1e-9 for x in m2_rows)

    # deterministic pnl under current toy execution model
    assert abs(report.results[0].pnl - 1.4975) < 1e-9


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


def test_backtest_default_sizing_avoids_zero_execution_rate(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_extreme_s8_markets(storage)

    settings = _default_settings_for_test(str(db))
    generated = SignalEngine(storage, settings).generate(["s8"])
    assert len(generated) == 6

    created = [row["created_at"] for row in storage.list_signals(limit=100, strategy="s8")]
    assert created
    from_ts = min(created)
    to_ts = max(created)

    report = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(slippage_bps=0.0, fee_bps=0.0),
        risk=BacktestRiskConfig(
            max_daily_loss=settings.risk.max_daily_loss,
            max_strategy_notional=settings.risk.max_strategy_notional,
            max_event_notional=settings.risk.max_event_notional,
            circuit_breaker_rejections=settings.risk.circuit_breaker_rejections,
        ),
    ).run(["s8"], from_ts=from_ts, to_ts=to_ts)

    assert report.total_signals == 6
    assert report.executed_signals == 6
    assert report.rejected_signals == 0
    assert all(x.risk_reason == "ok" for x in report.replay)
    assert all(x.risk_notional <= 25.0 + 1e-9 for x in report.replay)


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
    strategy_csv_out = tmp_path / "artifacts" / "strategy.csv"
    event_csv_out = tmp_path / "artifacts" / "event.csv"

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
            "--out-strategy-csv",
            str(strategy_csv_out),
            "--out-event-csv",
            str(event_csv_out),
            "--with-csv-digest-sidecar",
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert f"schema={BACKTEST_ARTIFACT_SCHEMA_VERSION}" in res.output
    assert "Backtest attribution" in res.output
    assert "Backtest event attribution" in res.output
    assert "Backtest replay ledger" in res.output
    assert "json exported" in res.output
    assert "replay csv exported" in res.output
    assert "strategy csv exported" in res.output
    assert "event csv exported" in res.output
    assert "replay csv digest exported" in res.output
    assert "strategy csv digest exported" in res.output
    assert "event csv digest exported" in res.output
    assert "s1" in res.output

    payload = json.loads(json_out.read_text())
    assert payload["schema_version"] == BACKTEST_ARTIFACT_SCHEMA_VERSION
    validate_backtest_json_artifact(payload)
    assert "checksum_sha256" not in payload
    assert payload["total_signals"] == 4
    assert payload["executed_signals"] == 4
    assert payload["rejected_signals"] == 0
    assert payload["execution_config"]["slippage_bps"] == 0.0
    assert payload["risk_config"]["max_event_notional"] == 1500.0
    assert len(payload["results"]) == 1
    assert len(payload["event_results"]) == 2
    assert len(payload["replay"]) == 4
    assert abs(float(payload["results"][0]["closed_winrate"]) - 0.5) < 1e-9
    assert abs(float(payload["results"][0]["mtm_winrate"]) - 0.5) < 1e-9
    assert int(payload["results"][0]["closed_sample_count"]) == 2
    assert int(payload["results"][0]["mtm_sample_count"]) == 2

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert rows[0]["schema_version"] == BACKTEST_ARTIFACT_SCHEMA_VERSION
    assert rows[0]["strategy"] == "s1"
    assert rows[0]["event_id"] == "e1"
    assert rows[0]["risk_allowed"] == "True"
    assert rows[0]["risk_reason"] == "ok"
    assert abs(float(rows[0]["executed_qty"]) - float(rows[0]["qty"])) < 1e-9
    assert abs(float(rows[0]["fill_ratio"]) - 1.0) < 1e-9
    assert abs(float(rows[0]["fill_probability"]) - 1.0) < 1e-9
    assert abs(float(rows[0]["slippage_bps_applied"]) - 0.0) < 1e-9

    with strategy_csv_out.open() as f:
        strategy_rows = list(csv.DictReader(f))
    assert len(strategy_rows) == 1
    assert strategy_rows[0]["schema_version"] == BACKTEST_ARTIFACT_SCHEMA_VERSION
    assert strategy_rows[0]["strategy"] == "s1"
    assert abs(float(strategy_rows[0]["pnl"]) - 1.5) < 1e-9
    assert abs(float(strategy_rows[0]["closed_winrate"]) - 0.5) < 1e-9
    assert abs(float(strategy_rows[0]["mtm_winrate"]) - 0.5) < 1e-9
    assert int(strategy_rows[0]["closed_sample_count"]) == 2
    assert int(strategy_rows[0]["mtm_sample_count"]) == 2

    with event_csv_out.open() as f:
        event_rows = list(csv.DictReader(f))
    assert len(event_rows) == 2
    assert all(x["schema_version"] == BACKTEST_ARTIFACT_SCHEMA_VERSION for x in event_rows)
    assert {x["event_id"] for x in event_rows} == {"e1", "e2"}
    assert all("closed_winrate" in x for x in event_rows)
    assert all("mtm_winrate" in x for x in event_rows)

    for csv_path in [csv_out, strategy_csv_out, event_csv_out]:
        sidecar_path = csv_path.with_name(csv_path.name + ".sha256")
        assert sidecar_path.exists()
        line = sidecar_path.read_text().strip()
        digest, filename = line.split("  ", 1)
        assert filename == csv_path.name
        assert digest == hashlib.sha256(csv_path.read_bytes()).hexdigest()


def test_cli_backtest_rolling_command(tmp_path: Path) -> None:
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

    out_json = tmp_path / "artifacts" / "rolling.json"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "backtest-rolling",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--window-hours",
            "1",
            "--step-hours",
            "1",
            "--slippage-bps",
            "0",
            "--out-json",
            str(out_json),
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "backtest rolling runs=2" in res.output
    assert "Backtest rolling windows" in res.output
    assert "Backtest rolling strategy aggregate" in res.output
    assert "overlap_mode=tiled" in res.output
    assert "coverage_label=full" in res.output
    assert "range_hours=" in res.output
    assert "rejection reasons" in res.output.lower()
    assert "rolling json exported" in res.output

    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == "rolling-1.0"
    assert payload["kind"] == "backtest_rolling_summary"
    assert payload["overlap_mode"] == "tiled"
    assert payload["execution_config"]["slippage_bps"] == 0.0
    assert payload["execution_config"]["fee_bps"] == 0.0
    assert payload["risk_config"]["max_event_notional"] == 1500.0

    assert payload["summary"]["run_count"] == 2
    assert payload["summary"]["total_signals"] == 4
    assert payload["summary"]["executed_signals"] == 4
    assert payload["summary"]["rejected_signals"] == 0
    assert payload["summary"]["empty_window_count"] == 1
    assert payload["summary"]["positive_window_count"] == 1
    assert abs(float(payload["summary"]["positive_window_rate"]) - 0.5) < 1e-9
    assert abs(float(payload["summary"]["pnl_sum"]) - 1.5) < 1e-9
    assert abs(float(payload["summary"]["pnl_avg"]) - 0.75) < 1e-9
    assert abs(float(payload["summary"]["range_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["sampled_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["covered_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["overlap_hours"]) - 0.0) < 1e-9
    assert abs(float(payload["summary"]["coverage_ratio"]) - 1.0) < 1e-9
    assert abs(float(payload["summary"]["overlap_ratio"]) - 0.0) < 1e-9
    assert payload["summary"]["coverage_label"] == "full"
    assert "covered_hours/range_hours" in str(payload["summary"]["coverage_basis"])
    assert payload["summary"]["risk_rejection_reasons"] == {}
    assert len(payload["windows"]) == 2
    assert payload["windows"][0]["total_signals"] == 4
    assert payload["windows"][1]["total_signals"] == 0
    assert payload["windows"][0]["risk_rejection_reasons"] == {}
    assert payload["windows"][1]["risk_rejection_reasons"] == {}
    assert any(x["strategy"] == "s1" for x in payload["strategy_aggregate"])
    s1_row = next(x for x in payload["strategy_aggregate"] if x["strategy"] == "s1")
    assert "avg_closed_winrate" in s1_row
    assert "avg_mtm_winrate" in s1_row
    assert "closed_sample_count" in s1_row
    assert "mtm_sample_count" in s1_row


def test_cli_backtest_rolling_risk_reason_histogram(tmp_path: Path) -> None:
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
                "risk:",
                "  max_daily_loss: 1000",
                "  max_strategy_notional: 1000",
                "  max_event_notional: 5",
                "  circuit_breaker_rejections: 5",
            ]
        )
    )

    out_json = tmp_path / "artifacts" / "rolling-risk.json"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "backtest-rolling",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--window-hours",
            "2",
            "--step-hours",
            "2",
            "--slippage-bps",
            "0",
            "--out-json",
            str(out_json),
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "rejection reasons" in res.output.lower()
    assert "event notional limit exceeded" in res.output

    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == "rolling-1.0"
    assert payload["overlap_mode"] == "tiled"
    assert payload["execution_config"]["slippage_bps"] == 0.0
    assert payload["risk_config"]["max_event_notional"] == 5.0

    assert payload["summary"]["run_count"] == 1
    assert payload["summary"]["rejected_signals"] == 1
    assert payload["summary"]["empty_window_count"] == 0
    assert abs(float(payload["summary"]["pnl_avg"]) - float(payload["summary"]["pnl_sum"])) < 1e-9
    assert abs(float(payload["summary"]["range_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["sampled_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["covered_hours"]) - 2.0) < 1e-9
    assert abs(float(payload["summary"]["overlap_hours"]) - 0.0) < 1e-9
    assert abs(float(payload["summary"]["coverage_ratio"]) - 1.0) < 1e-9
    assert abs(float(payload["summary"]["overlap_ratio"]) - 0.0) < 1e-9
    assert payload["summary"]["coverage_label"] == "full"
    assert "covered_hours/range_hours" in str(payload["summary"]["coverage_basis"])
    assert (
        abs(
            float(payload["summary"]["positive_window_rate"])
            - (
                float(payload["summary"]["positive_window_count"])
                / float(payload["summary"]["run_count"])
            )
        )
        < 1e-9
    )

    reason_map = payload["summary"]["risk_rejection_reasons"]
    assert isinstance(reason_map, dict)
    assert sum(int(v) for v in reason_map.values()) == 1
    assert any("event notional limit exceeded" in str(k) for k in reason_map.keys())

    win_reasons = payload["windows"][0]["risk_rejection_reasons"]
    assert isinstance(win_reasons, dict)
    assert sum(int(v) for v in win_reasons.values()) == 1
    assert any("event notional limit exceeded" in str(k) for k in win_reasons.keys())


def test_cli_backtest_rolling_overlap_mode_flags(tmp_path: Path) -> None:
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

    overlap_out = tmp_path / "artifacts" / "rolling-overlap.json"
    overlap_res = runner.invoke(
        app,
        [
            "backtest-rolling",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--window-hours",
            "1",
            "--step-hours",
            "0.5",
            "--slippage-bps",
            "0",
            "--out-json",
            str(overlap_out),
            "--config",
            str(config_path),
        ],
    )
    assert overlap_res.exit_code == 0, overlap_res.output
    assert "overlap_mode=overlap" in overlap_res.output
    assert "coverage_label=full" in overlap_res.output
    overlap_payload = json.loads(overlap_out.read_text())
    assert overlap_payload["overlap_mode"] == "overlap"
    assert overlap_payload["summary"]["coverage_label"] == "full"
    assert abs(float(overlap_payload["summary"]["range_hours"]) - 2.0) < 1e-9
    assert abs(float(overlap_payload["summary"]["coverage_ratio"]) - 1.0) < 1e-9
    assert abs(float(overlap_payload["summary"]["overlap_ratio"]) - 0.75) < 1e-9
    assert abs(float(overlap_payload["summary"]["overlap_hours"]) - 1.5) < 1e-9

    gapped_out = tmp_path / "artifacts" / "rolling-gapped.json"
    gapped_res = runner.invoke(
        app,
        [
            "backtest-rolling",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--window-hours",
            "1",
            "--step-hours",
            "2",
            "--slippage-bps",
            "0",
            "--out-json",
            str(gapped_out),
            "--config",
            str(config_path),
        ],
    )
    assert gapped_res.exit_code == 0, gapped_res.output
    assert "overlap_mode=gapped" in gapped_res.output
    assert "coverage_label=partial" in gapped_res.output
    gapped_payload = json.loads(gapped_out.read_text())
    assert gapped_payload["overlap_mode"] == "gapped"
    assert gapped_payload["summary"]["coverage_label"] == "partial"
    assert abs(float(gapped_payload["summary"]["range_hours"]) - 2.0) < 1e-9
    assert abs(float(gapped_payload["summary"]["coverage_ratio"]) - 0.5) < 1e-9
    assert abs(float(gapped_payload["summary"]["overlap_ratio"]) - 0.0) < 1e-9
    assert abs(float(gapped_payload["summary"]["overlap_hours"]) - 0.0) < 1e-9

    sparse_out = tmp_path / "artifacts" / "rolling-sparse.json"
    sparse_res = runner.invoke(
        app,
        [
            "backtest-rolling",
            "--strategies",
            "s1",
            "--from",
            "2026-02-20T00:00:00Z",
            "--to",
            "2026-02-20T02:00:00Z",
            "--window-hours",
            "0.4",
            "--step-hours",
            "3",
            "--slippage-bps",
            "0",
            "--out-json",
            str(sparse_out),
            "--config",
            str(config_path),
        ],
    )
    assert sparse_res.exit_code == 0, sparse_res.output
    assert "coverage_label=sparse" in sparse_res.output
    sparse_payload = json.loads(sparse_out.read_text())
    assert sparse_payload["overlap_mode"] == "gapped"
    assert sparse_payload["summary"]["coverage_label"] == "sparse"
    assert abs(float(sparse_payload["summary"]["coverage_ratio"]) - 0.2) < 1e-9


def test_cli_backtest_json_with_checksum(tmp_path: Path) -> None:
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

    json_out = tmp_path / "artifacts" / "backtest.with-checksum.json"

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
            "--with-checksum",
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "json exported" in res.output
    assert "checksum" in res.output

    payload = json.loads(json_out.read_text())
    validate_backtest_json_artifact(payload)
    assert payload["checksum_algo"] == BACKTEST_ARTIFACT_CHECKSUM_ALGO
    assert isinstance(payload["checksum_sha256"], str)
    assert len(payload["checksum_sha256"]) == 64
    assert verify_backtest_json_artifact_checksum(payload)


def test_cli_backtest_migrate_v1_to_v2(tmp_path: Path) -> None:
    v1 = {
        "schema_version": "1.0",
        "generated_at": "2026-02-20T00:00:00+00:00",
        "from_ts": "2026-02-20T00:00:00+00:00",
        "to_ts": "2026-02-20T01:00:00+00:00",
        "total_signals": 1,
        "executed_signals": 1,
        "rejected_signals": 0,
        "execution_config": {},
        "risk_config": {},
        "results": [],
        "event_results": [],
        "replay": [],
    }
    in_path = tmp_path / "v1.json"
    out_path = tmp_path / "v2.json"
    in_path.write_text(json.dumps(v1))

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "backtest-migrate-v1-to-v2",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "migrated" in res.output

    payload_v2 = json.loads(out_path.read_text())
    assert payload_v2["schema_version"] == "2.0"
    assert payload_v2["meta"]["migration"] == "v1_to_v2"
    assert validate_backtest_json_artifact(payload_v2, supported_major=None) == (2, 0)


def test_cli_backtest_migration_map_table_and_json(tmp_path: Path) -> None:
    runner = CliRunner()

    map_out = tmp_path / "migration_map.json"
    table_res = runner.invoke(
        app,
        [
            "backtest-migration-map",
            "--with-checksum",
            "--out-json",
            str(map_out),
        ],
    )
    assert table_res.exit_code == 0, table_res.output
    assert "Backtest v1 -> v2 field mapping" in table_res.output
    assert "schema_version" in table_res.output
    assert "migration map exported" in table_res.output
    assert "checksum=" in table_res.output

    payload = json.loads(map_out.read_text())
    assert payload["kind"] == "backtest_migration_map"
    assert payload["from_schema_major"] == 1
    assert payload["to_schema_major"] == 2
    assert isinstance(payload["mappings"], list)
    assert isinstance(payload["checksum_sha256"], str)
    assert len(payload["checksum_sha256"]) == 64

    json_res = runner.invoke(app, ["backtest-migration-map", "--format", "json"])
    assert json_res.exit_code == 0, json_res.output
    assert '"v1_path"' in json_res.output
    assert '"v2_path"' in json_res.output
