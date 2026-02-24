from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from monomarket.backtest import (
    BacktestEngine,
    BacktestExecutionConfig,
    BacktestReport,
    BacktestRiskConfig,
    backtest_migration_v1_to_v2_field_map,
    build_backtest_migration_map_artifact,
    migrate_backtest_artifact_v1_to_v2,
    validate_backtest_json_artifact,
)
from monomarket.config import Settings, load_settings
from monomarket.data import IngestionService, MarketDataClients
from monomarket.db import Storage
from monomarket.execution import ExecutionRouter
from monomarket.models import OrderRequest
from monomarket.pnl import MetricsReporter, PnlTracker
from monomarket.signals import SignalEngine

app = typer.Typer(add_completion=False, help="Monomarket CLI")
console = Console()

# Outcome token label used in trade requests (not a credential).
DEFAULT_OUTCOME_TOKEN = "YES"  # nosec B105


def _ctx(config_path: str | None = None) -> tuple[Settings, Storage]:
    settings = load_settings(config_path)
    storage = Storage(settings.app.db_path)
    return settings, storage


def _write_backtest_json(report: BacktestReport, output_path: str) -> None:
    payload = {
        "schema_version": report.schema_version,
        "generated_at": report.generated_at.isoformat(),
        "from_ts": report.from_ts,
        "to_ts": report.to_ts,
        "total_signals": report.total_signals,
        "executed_signals": report.executed_signals,
        "rejected_signals": report.rejected_signals,
        "execution_config": report.execution_config,
        "risk_config": report.risk_config,
        "results": [asdict(x) for x in report.results],
        "event_results": [asdict(x) for x in report.event_results],
        "replay": [asdict(x) for x in report.replay],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_backtest_replay_csv(report: BacktestReport, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "schema_version",
                "ts",
                "strategy",
                "event_id",
                "market_id",
                "token_id",
                "side",
                "qty",
                "executed_qty",
                "fill_ratio",
                "fill_probability",
                "slippage_bps_applied",
                "target_price",
                "fill_price",
                "realized_change",
                "strategy_equity",
                "event_equity",
                "risk_allowed",
                "risk_reason",
                "risk_notional",
                "risk_realized_pnl_before",
                "risk_strategy_notional_before",
                "risk_event_notional_before",
                "risk_rejections_before",
                "risk_max_daily_loss",
                "risk_max_strategy_notional",
                "risk_max_event_notional",
                "risk_circuit_breaker_rejections",
            ],
        )
        writer.writeheader()
        for replay_row in report.replay:
            row = asdict(replay_row)
            row["schema_version"] = report.schema_version
            writer.writerow(row)


def _write_backtest_strategy_csv(report: BacktestReport, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "schema_version",
                "strategy",
                "pnl",
                "winrate",
                "max_drawdown",
                "trade_count",
                "wins",
                "losses",
            ],
        )
        writer.writeheader()
        for row in report.results:
            csv_row = asdict(row)
            csv_row["schema_version"] = report.schema_version
            writer.writerow(csv_row)


def _write_backtest_event_csv(report: BacktestReport, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "schema_version",
                "strategy",
                "event_id",
                "pnl",
                "winrate",
                "max_drawdown",
                "trade_count",
                "wins",
                "losses",
            ],
        )
        writer.writeheader()
        for row in report.event_results:
            csv_row = asdict(row)
            csv_row["schema_version"] = report.schema_version
            writer.writerow(csv_row)


@app.command("init-db")
def init_db(config: str | None = typer.Option(None, help="Config yaml path")) -> None:
    _, storage = _ctx(config)
    storage.init_db()
    console.print(f"[green]DB initialized:[/green] {storage.db_path}")


@app.command("ingest")
def ingest(
    source: str = typer.Option("gamma", help="gamma|data|clob|all"),
    limit: int = typer.Option(200, min=1, max=5000),
    incremental: bool = typer.Option(True, "--incremental/--full"),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    svc = IngestionService(
        MarketDataClients(settings.data),
        storage,
        breaker_failure_threshold=settings.data.breaker_failure_threshold,
        breaker_cooldown_sec=settings.data.breaker_cooldown_sec,
    )
    res = svc.ingest(source, limit, incremental=incremental)

    if res.status == "ok":
        color = "green"
    elif res.status == "partial":
        color = "yellow"
    else:
        color = "red"

    console.print(
        f"[{color}]ingest {res.status}[/{color}] source={res.source} rows={res.rows} "
        f"requests={res.request_count} retries={res.retry_count} failures={res.failure_count}"
    )
    if res.error_buckets:
        parts = [f"{k}:{v}" for k, v in sorted(res.error_buckets.items())]
        console.print("error_buckets=" + ", ".join(parts))
    if res.error:
        console.print(f"[red]error[/red] {res.error}")


@app.command("ingest-health")
def ingest_health(
    source: str | None = typer.Option(None, help="gamma|data|clob (default: all)"),
    limit: int = typer.Option(20, min=1, max=200),
    run_window: int = typer.Option(20, min=1, max=500, help="Recent runs per source"),
    error_trend_window: int = typer.Option(
        20,
        min=1,
        max=500,
        help="Recent error-bucket events per source/bucket to compare against previous window",
    ),
    error_sample_limit: int = typer.Option(
        5,
        min=1,
        max=50,
        help="Recent non-empty errors per source",
    ),
    config: str | None = typer.Option(None),
) -> None:
    _, storage = _ctx(config)
    storage.init_db()

    buckets = storage.list_ingestion_error_buckets(source=source, limit=limit)
    trends = storage.list_ingestion_error_bucket_trends(
        source=source,
        window=error_trend_window,
        limit=limit,
    )
    breakers = storage.list_ingestion_breakers(source=source, limit=limit)
    run_summary = storage.list_ingestion_run_summary_by_source(
        source=source,
        run_window=run_window,
    )
    error_share = storage.list_ingestion_error_bucket_share_by_source(
        source=source,
        run_window=run_window,
        limit=limit,
    )
    transitions = storage.list_ingestion_breaker_transitions(source=source, limit=limit)
    recent_errors = storage.list_ingestion_recent_errors(
        source=source,
        per_source_limit=error_sample_limit,
    )

    tb1 = Table(title=f"Ingestion error buckets ({len(buckets)})")
    tb1.add_column("source")
    tb1.add_column("bucket")
    tb1.add_column("count")
    tb1.add_column("last_error")
    tb1.add_column("updated_at")
    for row in buckets:
        tb1.add_row(
            str(row["source"]),
            str(row["error_bucket"]),
            str(row["total_count"]),
            str(row["last_error"]),
            str(row["updated_at"]),
        )
    console.print(tb1)

    tb1b = Table(
        title=(
            "Ingestion error bucket trends "
            f"(recent={error_trend_window} vs prev={error_trend_window})"
        )
    )
    tb1b.add_column("source")
    tb1b.add_column("bucket")
    tb1b.add_column("recent")
    tb1b.add_column("prev")
    tb1b.add_column("delta")
    tb1b.add_column("change")
    tb1b.add_column("recent_last_at")
    for row in trends:
        recent_count = int(row["recent_count"] or 0)
        prev_count = int(row["prev_count"] or 0)
        delta = recent_count - prev_count
        if prev_count <= 0:
            change = "new" if recent_count > 0 else "0.00%"
        else:
            change = f"{(delta / prev_count):.2%}"
        tb1b.add_row(
            str(row["source"]),
            str(row["error_bucket"]),
            str(recent_count),
            str(prev_count),
            f"{delta:+d}",
            change,
            str(row["recent_last_at"] or ""),
        )
    console.print(tb1b)

    tb2 = Table(title=f"Ingestion breakers ({len(breakers)})")
    tb2.add_column("source")
    tb2.add_column("consecutive_failures")
    tb2.add_column("open_until")
    tb2.add_column("last_bucket")
    tb2.add_column("updated_at")
    for row in breakers:
        tb2.add_row(
            str(row["source"]),
            str(row["consecutive_failures"]),
            str(row["open_until_ts"] or ""),
            str(row["last_error_bucket"]),
            str(row["updated_at"]),
        )
    console.print(tb2)

    tb3 = Table(title=f"Breaker transitions ({len(transitions)})")
    tb3.add_column("source")
    tb3.add_column("state")
    tb3.add_column("count")
    tb3.add_column("last_transition_at")
    for row in transitions:
        tb3.add_row(
            str(row["source"]),
            str(row["state"]),
            str(row["transition_count"]),
            str(row["last_transition_at"]),
        )
    console.print(tb3)

    tb4 = Table(title=f"Ingestion run summary by source (window={run_window})")
    tb4.add_column("source")
    tb4.add_column("total")
    tb4.add_column("ok")
    tb4.add_column("partial")
    tb4.add_column("error")
    tb4.add_column("non_ok_rate")
    tb4.add_column("avg_rows")
    tb4.add_column("avg_failures")
    tb4.add_column("avg_retries")
    tb4.add_column("failure_per_req")
    tb4.add_column("last_finished_at")
    for row in run_summary:
        total_runs = int(row["total_runs"] or 0)
        non_ok_runs = int(row["non_ok_runs"] or 0)
        non_ok_rate = (non_ok_runs / total_runs) if total_runs else 0.0
        avg_rows = float(row["avg_rows"] or 0.0)
        avg_failures = float(row["avg_failures"] or 0.0)
        avg_retries = float(row["avg_retries"] or 0.0)
        total_failures = float(row["total_failures"] or 0.0)
        total_requests = float(row["total_requests"] or 0.0)
        failure_per_req = (total_failures / total_requests) if total_requests else 0.0
        tb4.add_row(
            str(row["source"]),
            str(total_runs),
            str(int(row["ok_runs"] or 0)),
            str(int(row["partial_runs"] or 0)),
            str(int(row["error_runs"] or 0)),
            f"{non_ok_rate:.2%}",
            f"{avg_rows:.2f}",
            f"{avg_failures:.2f}",
            f"{avg_retries:.2f}",
            f"{failure_per_req:.2%}",
            str(row["last_finished_at"] or ""),
        )
    console.print(tb4)

    tb5 = Table(title=f"Ingestion error bucket share by source (runs window={run_window})")
    tb5.add_column("source")
    tb5.add_column("bucket")
    tb5.add_column("count")
    tb5.add_column("share")
    tb5.add_column("bucket_total")
    tb5.add_column("runs_with_error")
    tb5.add_column("total_runs")
    for row in error_share:
        tb5.add_row(
            str(row["source"]),
            str(row["error_bucket"]),
            str(int(row["bucket_count"] or 0)),
            f"{float(row['bucket_share'] or 0.0):.2%}",
            str(int(row["bucket_total"] or 0)),
            str(int(row["runs_with_error"] or 0)),
            str(int(row["total_runs"] or 0)),
        )
    console.print(tb5)

    tb6 = Table(title=f"Recent ingestion errors (per source <= {error_sample_limit})")
    tb6.add_column("source")
    tb6.add_column("finished_at")
    tb6.add_column("status")
    tb6.add_column("error")
    for row in recent_errors:
        tb6.add_row(
            str(row["source"]),
            str(row["finished_at"]),
            str(row["status"]),
            str(row["error"]),
        )
    console.print(tb6)


@app.command("list-markets")
def list_markets(
    limit: int = typer.Option(20, min=1, max=1000),
    status: str = typer.Option("open", help="open|closed|all"),
    config: str | None = typer.Option(None),
) -> None:
    _, storage = _ctx(config)
    filter_status = None if status.lower() == "all" else status.lower()
    rows = storage.fetch_markets(limit=limit, status=filter_status)
    tb = Table(title=f"Markets ({len(rows)})")
    tb.add_column("source")
    tb.add_column("market_id")
    tb.add_column("event")
    tb.add_column("yes")
    tb.add_column("no")
    tb.add_column("liq")
    for r in rows:
        tb.add_row(
            r.source,
            r.market_id,
            r.event_id,
            f"{r.yes_price:.3f}" if r.yes_price is not None else "-",
            f"{r.no_price:.3f}" if r.no_price is not None else "-",
            f"{r.liquidity:.1f}",
        )
    console.print(tb)


@app.command("generate-signals")
def generate_signals(
    strategies: str = typer.Option("s1,s2,s4,s8", help="Comma-separated strategy ids"),
    market_limit: int = typer.Option(2000, min=10, max=20000),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    engine = SignalEngine(storage, settings)
    selected = [s.strip().lower() for s in strategies.split(",") if s.strip()]
    signals = engine.generate(selected, market_limit=market_limit)

    by_strategy: dict[str, int] = {}
    for s in signals:
        by_strategy[s.strategy] = by_strategy.get(s.strategy, 0) + 1

    console.print(f"[green]generated[/green] {len(signals)} signals")
    for k, v in sorted(by_strategy.items()):
        console.print(f"- {k}: {v}")


@app.command("list-signals")
def list_signals(
    limit: int = typer.Option(20, min=1, max=200),
    status: str | None = typer.Option(None, help="new|executed|rejected"),
    strategy: str | None = typer.Option(None),
    config: str | None = typer.Option(None),
) -> None:
    _, storage = _ctx(config)
    rows = storage.list_signals(limit=limit, status=status, strategy=strategy)
    tb = Table(title=f"Signals ({len(rows)})")
    tb.add_column("id")
    tb.add_column("strategy")
    tb.add_column("market")
    tb.add_column("side")
    tb.add_column("score")
    tb.add_column("target")
    tb.add_column("size")
    tb.add_column("status")

    for r in rows:
        tb.add_row(
            str(r["id"]),
            str(r["strategy"]),
            str(r["market_id"]),
            str(r["side"]),
            f"{float(r['score']):.3f}",
            f"{float(r['target_price']):.3f}",
            f"{float(r['size_hint']):.2f}",
            str(r["status"]),
        )

    console.print(tb)


@app.command("backtest")
def backtest(
    strategies: str = typer.Option("s1,s2,s4,s8", help="Comma-separated strategy ids"),
    from_ts: str = typer.Option(..., "--from", help="Inclusive ISO timestamp"),
    to_ts: str = typer.Option(..., "--to", help="Inclusive ISO timestamp"),
    slippage_bps: float = typer.Option(5.0, min=0.0),
    fee_bps: float = typer.Option(0.0, min=0.0),
    partial_fill: bool = typer.Option(
        False,
        "--partial-fill/--no-partial-fill",
        help="Enable liquidity-based partial fills in replay",
    ),
    liquidity_full_fill: float = typer.Option(
        1000.0,
        min=1.0,
        help="Liquidity threshold where fills become 100%",
    ),
    min_fill_ratio: float = typer.Option(
        0.1,
        min=0.0,
        max=1.0,
        help="Minimum fill ratio when partial fill is enabled",
    ),
    fill_probability: bool = typer.Option(
        False,
        "--fill-probability/--no-fill-probability",
        help="Enable liquidity-bucket fill probability model",
    ),
    min_fill_probability: float = typer.Option(
        0.05,
        min=0.0,
        max=1.0,
        help="Minimum fill probability when probability model is enabled",
    ),
    dynamic_slippage: bool = typer.Option(
        False,
        "--dynamic-slippage/--no-dynamic-slippage",
        help="Enable spread/liquidity layered slippage",
    ),
    spread_slippage_weight_bps: float = typer.Option(
        50.0,
        min=0.0,
        help="Spread proxy weight for dynamic slippage (bps)",
    ),
    liquidity_slippage_weight_bps: float = typer.Option(
        25.0,
        min=0.0,
        help="Liquidity penalty weight for dynamic slippage (bps)",
    ),
    slippage_liquidity_reference: float = typer.Option(
        1000.0,
        min=1.0,
        help="Liquidity reference used by dynamic slippage",
    ),
    replay_limit: int = typer.Option(20, min=0, help="Rows to print from replay ledger (0=skip)"),
    out_json: str | None = typer.Option(None, help="Write full backtest report as JSON"),
    out_replay_csv: str | None = typer.Option(None, help="Write replay ledger as CSV"),
    out_strategy_csv: str | None = typer.Option(
        None,
        help="Write strategy attribution as CSV",
    ),
    out_event_csv: str | None = typer.Option(
        None,
        help="Write event attribution as CSV",
    ),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()

    selected = [s.strip().lower() for s in strategies.split(",") if s.strip()]
    engine = BacktestEngine(
        storage,
        execution=BacktestExecutionConfig(
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            enable_partial_fill=partial_fill,
            liquidity_full_fill=liquidity_full_fill,
            min_fill_ratio=min_fill_ratio,
            enable_fill_probability=fill_probability,
            min_fill_probability=min_fill_probability,
            enable_dynamic_slippage=dynamic_slippage,
            spread_slippage_weight_bps=spread_slippage_weight_bps,
            liquidity_slippage_weight_bps=liquidity_slippage_weight_bps,
            liquidity_reference=slippage_liquidity_reference,
        ),
        risk=BacktestRiskConfig(
            max_daily_loss=settings.risk.max_daily_loss,
            max_strategy_notional=settings.risk.max_strategy_notional,
            max_event_notional=settings.risk.max_event_notional,
            circuit_breaker_rejections=settings.risk.circuit_breaker_rejections,
        ),
    )
    report = engine.run(selected, from_ts=from_ts, to_ts=to_ts)

    console.print(
        f"backtest schema={report.schema_version} signals={report.total_signals} "
        f"executed={report.executed_signals} rejected={report.rejected_signals} "
        f"replay_rows={len(report.replay)} from={report.from_ts} to={report.to_ts}"
    )

    tb = Table(title="Backtest attribution")
    tb.add_column("strategy")
    tb.add_column("pnl")
    tb.add_column("winrate")
    tb.add_column("max_drawdown")
    tb.add_column("trades")

    for result in report.results:
        tb.add_row(
            result.strategy,
            f"{result.pnl:.4f}",
            f"{result.winrate:.2%}",
            f"{result.max_drawdown:.4f}",
            str(result.trade_count),
        )
    console.print(tb)

    event_tb = Table(title="Backtest event attribution")
    event_tb.add_column("strategy")
    event_tb.add_column("event")
    event_tb.add_column("pnl")
    event_tb.add_column("winrate")
    event_tb.add_column("max_drawdown")
    event_tb.add_column("trades")

    for event_result in report.event_results:
        event_tb.add_row(
            event_result.strategy,
            event_result.event_id,
            f"{event_result.pnl:.4f}",
            f"{event_result.winrate:.2%}",
            f"{event_result.max_drawdown:.4f}",
            str(event_result.trade_count),
        )

    console.print(event_tb)

    if replay_limit > 0 and report.replay:
        replay_tb = Table(title=f"Backtest replay ledger (first {replay_limit})")
        replay_tb.add_column("ts")
        replay_tb.add_column("strategy")
        replay_tb.add_column("event")
        replay_tb.add_column("market")
        replay_tb.add_column("token")
        replay_tb.add_column("side")
        replay_tb.add_column("qty")
        replay_tb.add_column("filled")
        replay_tb.add_column("fill_ratio")
        replay_tb.add_column("fill_prob")
        replay_tb.add_column("slip_bps")
        replay_tb.add_column("target")
        replay_tb.add_column("fill")
        replay_tb.add_column("realized")
        replay_tb.add_column("strat_eq")
        replay_tb.add_column("event_eq")
        replay_tb.add_column("risk_ok")
        replay_tb.add_column("risk_reason")

        for replay_row in report.replay[:replay_limit]:
            replay_tb.add_row(
                replay_row.ts,
                replay_row.strategy,
                replay_row.event_id,
                replay_row.market_id,
                replay_row.token_id,
                replay_row.side,
                f"{replay_row.qty:.4f}",
                f"{replay_row.executed_qty:.4f}",
                f"{replay_row.fill_ratio:.2%}",
                f"{replay_row.fill_probability:.2%}",
                f"{replay_row.slippage_bps_applied:.2f}",
                f"{replay_row.target_price:.4f}",
                f"{replay_row.fill_price:.4f}",
                f"{replay_row.realized_change:.4f}",
                f"{replay_row.strategy_equity:.4f}",
                f"{replay_row.event_equity:.4f}",
                "yes" if replay_row.risk_allowed else "no",
                replay_row.risk_reason,
            )
        console.print(replay_tb)

    if out_json:
        _write_backtest_json(report, out_json)
        console.print(f"[green]json exported[/green] {out_json}")

    if out_replay_csv:
        _write_backtest_replay_csv(report, out_replay_csv)
        console.print(f"[green]replay csv exported[/green] {out_replay_csv}")

    if out_strategy_csv:
        _write_backtest_strategy_csv(report, out_strategy_csv)
        console.print(f"[green]strategy csv exported[/green] {out_strategy_csv}")

    if out_event_csv:
        _write_backtest_event_csv(report, out_event_csv)
        console.print(f"[green]event csv exported[/green] {out_event_csv}")


@app.command("backtest-migrate-v1-to-v2")
def backtest_migrate_v1_to_v2(
    in_json: str = typer.Option(..., "--in", help="Input v1 backtest JSON artifact"),
    out_json: str = typer.Option(..., "--out", help="Output v2 backtest JSON artifact"),
) -> None:
    in_path = Path(in_json)
    payload = json.loads(in_path.read_text())
    migrated = migrate_backtest_artifact_v1_to_v2(payload)

    # sanity check using dual-stack validator path
    validate_backtest_json_artifact(migrated, supported_major=None)

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n")

    console.print(
        "[green]migrated[/green] " f"{in_path} -> {out_path} schema={migrated['schema_version']}"
    )


@app.command("backtest-migration-map")
def backtest_migration_map(
    format: str = typer.Option("table", help="table|json"),
    out_json: str | None = typer.Option(None, help="Optional path to write mapping artifact JSON"),
    with_checksum: bool = typer.Option(
        False,
        "--with-checksum/--no-checksum",
        help="Attach checksum_sha256 into exported mapping artifact",
    ),
) -> None:
    rows = backtest_migration_v1_to_v2_field_map()
    artifact = build_backtest_migration_map_artifact(with_checksum=with_checksum)
    fmt = format.strip().lower()

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
        checksum_msg = f" checksum={artifact['checksum_sha256']}" if with_checksum else ""
        console.print(f"[green]migration map exported[/green] {out_path}{checksum_msg}")

    if fmt == "json":
        console.print_json(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if fmt != "table":
        raise typer.BadParameter("format must be table or json")

    tb = Table(title="Backtest v1 -> v2 field mapping")
    tb.add_column("v1_path")
    tb.add_column("v2_path")
    tb.add_column("transform")
    tb.add_column("reversible")
    tb.add_column("note")

    for row in rows:
        tb.add_row(
            row["v1_path"],
            row["v2_path"],
            row["transform"],
            "yes" if row["reversible"] else "no",
            row["note"],
        )

    console.print(tb)


@app.command("execute-signal")
def execute_signal(
    signal_id: int,
    qty: float | None = typer.Option(None, help="Override quantity"),
    mode: str | None = typer.Option(None, help="paper|live"),
    confirm_live: bool = typer.Option(False, help="Required when REQUIRE_MANUAL_CONFIRM=true"),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    row = storage.get_signal(signal_id)
    if row is None:
        raise typer.BadParameter(f"Signal {signal_id} not found")

    payload = row.get("payload") or {}
    token_id = DEFAULT_OUTCOME_TOKEN
    price = float(row["target_price"])

    primary_leg = payload.get("primary_leg") if isinstance(payload, dict) else None
    if isinstance(primary_leg, dict):
        token_id = str(primary_leg.get("token", token_id)).upper()
        price = float(primary_leg.get("price", price))

    req = OrderRequest(
        strategy=str(row["strategy"]),
        market_id=str(row["market_id"]),
        event_id=str(row["event_id"]),
        token_id=token_id,
        side=str(row["side"]),
        action="open",
        price=price,
        qty=float(qty if qty is not None else row["size_hint"]),
        mode=(mode or settings.trading.mode),
        reason=f"signal:{signal_id}",
    )

    router = ExecutionRouter(storage, settings)
    res = router.execute(req, requested_mode=mode, manual_confirm=confirm_live)
    new_status = "executed" if res.accepted else "rejected"
    storage.update_signal_status(signal_id, new_status)
    console.print(
        f"status={res.status} order_id={res.order_id} accepted={res.accepted} message={res.message}"
    )


@app.command("place-order")
def place_order(
    strategy: str,
    market_id: str,
    event_id: str,
    token_id: str = typer.Option(DEFAULT_OUTCOME_TOKEN, help="YES|NO"),
    side: str = typer.Option("buy", help="buy|sell"),
    action: str = typer.Option("open", help="open|close"),
    price: float = typer.Option(...),
    qty: float = typer.Option(...),
    mode: str = typer.Option("paper", help="paper|live"),
    confirm_live: bool = typer.Option(False),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    req = OrderRequest(
        strategy=strategy,
        market_id=market_id,
        event_id=event_id,
        token_id=token_id.upper(),
        side=side.lower(),
        action=action.lower(),
        price=price,
        qty=qty,
        mode=mode.lower(),
        reason="manual",
    )

    res = ExecutionRouter(storage, settings).execute(
        req, requested_mode=mode, manual_confirm=confirm_live
    )
    console.print(
        f"status={res.status} accepted={res.accepted} order_id={res.order_id} message={res.message}"
    )


@app.command("live-cancel")
def live_cancel(
    order_id: int = typer.Argument(..., help="Local order id"),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    res = ExecutionRouter(storage, settings).cancel_live_order(order_id)
    console.print(
        f"status={res.status} accepted={res.accepted} order_id={res.order_id} message={res.message}"
    )


@app.command("live-sync")
def live_sync(
    limit: int = typer.Option(100, min=1, max=1000),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    summary = ExecutionRouter(storage, settings).sync_live_orders(limit=limit)
    console.print(
        "live-sync "
        f"scanned={summary['orders_scanned']} "
        f"updated={summary['orders_updated']} "
        f"errors={summary['errors']} "
        f"filled_delta_qty={summary['filled_delta_qty']}"
    )


@app.command("switches")
def switches(config: str | None = typer.Option(None)) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    router = ExecutionRouter(storage, settings)
    sw = router.resolve_switches()

    tb = Table(title="Trading Switches")
    tb.add_column("name")
    tb.add_column("value")
    tb.add_row("ENABLE_LIVE_TRADING", str(sw.enable_live_trading).lower())
    tb.add_row("REQUIRE_MANUAL_CONFIRM", str(sw.require_manual_confirm).lower())
    tb.add_row("KILL_SWITCH", str(sw.kill_switch).lower())
    console.print(tb)


@app.command("set-switch")
def set_switch(
    name: str,
    value: str,
    config: str | None = typer.Option(None),
) -> None:
    _, storage = _ctx(config)
    storage.init_db()
    k = name.upper().strip()
    if k not in {"ENABLE_LIVE_TRADING", "REQUIRE_MANUAL_CONFIRM", "KILL_SWITCH"}:
        raise typer.BadParameter("allowed: ENABLE_LIVE_TRADING|REQUIRE_MANUAL_CONFIRM|KILL_SWITCH")
    storage.set_switch(k, value)
    console.print(f"[green]set[/green] {k}={value}")


@app.command("pnl-report")
def pnl_report(config: str | None = typer.Option(None)) -> None:
    _, storage = _ctx(config)
    report = PnlTracker(storage).report()

    console.print(
        f"realized={report.realized_total:.4f} unrealized={report.unrealized_total:.4f} total={(report.realized_total + report.unrealized_total):.4f}"
    )

    tb = Table(title="PnL by strategy")
    tb.add_column("strategy")
    tb.add_column("pnl")
    for k, v in sorted(report.by_strategy.items()):
        tb.add_row(k, f"{v:.4f}")
    console.print(tb)


@app.command("metrics-report")
def metrics_report(config: str | None = typer.Option(None)) -> None:
    _, storage = _ctx(config)
    m = MetricsReporter(storage).report()

    tb = Table(title="Metrics")
    tb.add_column("metric")
    tb.add_column("value")
    tb.add_row("total_orders", str(m.total_orders))
    tb.add_row("filled_orders", str(m.filled_orders))
    tb.add_row("rejected_orders", str(m.rejected_orders))
    tb.add_row("fill_rate", f"{m.fill_rate:.4f}")
    tb.add_row("rejection_rate", f"{m.rejection_rate:.4f}")
    tb.add_row("realized_pnl", f"{m.realized_pnl:.4f}")
    tb.add_row("unrealized_pnl", f"{m.unrealized_pnl:.4f}")
    tb.add_row("max_drawdown", f"{m.max_drawdown:.4f}")
    console.print(tb)


if __name__ == "__main__":
    app()
