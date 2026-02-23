from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

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


@app.command("init-db")
def init_db(config: str | None = typer.Option(None, help="Config yaml path")) -> None:
    _, storage = _ctx(config)
    storage.init_db()
    console.print(f"[green]DB initialized:[/green] {storage.db_path}")


@app.command("ingest")
def ingest(
    source: str = typer.Option("gamma", help="gamma|data|clob|all"),
    limit: int = typer.Option(200, min=1, max=5000),
    config: str | None = typer.Option(None),
) -> None:
    settings, storage = _ctx(config)
    storage.init_db()
    svc = IngestionService(MarketDataClients(settings.data), storage)
    res = svc.ingest(source, limit)
    if res.status == "ok":
        console.print(f"[green]ingest ok[/green] source={res.source} rows={res.rows}")
    else:
        console.print(f"[red]ingest error[/red] source={res.source} error={res.error}")


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
