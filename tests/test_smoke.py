from __future__ import annotations

from pathlib import Path

from monomarket.config import AppSettings, DataSettings, RiskSettings, Settings, TradingSettings
from monomarket.db.storage import Storage
from monomarket.execution.router import ExecutionRouter
from monomarket.models import MarketView, OrderRequest
from monomarket.pnl.tracker import PnlTracker
from monomarket.signals.engine import SignalEngine


def _settings(db_path: str) -> Settings:
    return Settings(
        app=AppSettings(db_path=db_path, log_level="INFO"),
        trading=TradingSettings(
            mode="paper", enable_live_trading=False, require_manual_confirm=True, kill_switch=False
        ),
        risk=RiskSettings(
            max_daily_loss=1000.0,
            max_strategy_notional=100000.0,
            max_event_notional=100000.0,
            circuit_breaker_rejections=5,
        ),
        data=DataSettings(),
        strategies={
            "s1": {"min_spread": 0.02},
            "s2": {"prob_sum_tolerance": 0.01},
            "s4": {"yes_price_min": 0.01, "yes_price_max": 0.2},
            "s8": {"yes_price_max_for_no": 0.25, "hedge_budget_ratio": 0.1},
        },
    )


def _seed_markets(storage: Storage) -> None:
    storage.upsert_markets(
        [
            MarketView(
                market_id="m1",
                canonical_id="c1",
                source="gamma",
                event_id="e1",
                question="Q1",
                status="open",
                neg_risk=True,
                liquidity=1000,
                volume=200,
                yes_price=0.40,
                no_price=0.60,
                best_bid=0.39,
                best_ask=0.41,
                mid_price=0.40,
            ),
            MarketView(
                market_id="m1-clob",
                canonical_id="c1",
                source="clob",
                event_id="e1",
                question="Q1",
                status="open",
                neg_risk=True,
                liquidity=900,
                volume=180,
                yes_price=0.47,
                no_price=0.53,
                best_bid=0.46,
                best_ask=0.48,
                mid_price=0.47,
            ),
            MarketView(
                market_id="m2",
                canonical_id="c2",
                source="gamma",
                event_id="e2",
                question="Longshot",
                status="open",
                neg_risk=False,
                liquidity=700,
                volume=120,
                yes_price=0.08,
                no_price=0.92,
                best_bid=0.07,
                best_ask=0.09,
                mid_price=0.08,
            ),
            MarketView(
                market_id="m3",
                canonical_id="c3",
                source="data",
                event_id="e3",
                question="Tail",
                status="open",
                neg_risk=False,
                liquidity=500,
                volume=100,
                yes_price=0.03,
                no_price=0.97,
                best_bid=0.02,
                best_ask=0.04,
                mid_price=0.03,
            ),
        ]
    )


def test_end_to_end_smoke(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    _seed_markets(storage)
    settings = _settings(str(db))

    engine = SignalEngine(storage, settings)
    signals = engine.generate(["s1", "s2", "s4", "s8"])
    assert signals, "should generate at least one signal"

    top = storage.list_signals(limit=1)[0]
    req = OrderRequest(
        strategy=top["strategy"],
        market_id=top["market_id"],
        event_id=top["event_id"],
        token_id="YES",
        side="buy",
        action="open",
        price=float(top["target_price"]),
        qty=2.0,
        mode="paper",
    )
    res = ExecutionRouter(storage, settings).execute(
        req, requested_mode="paper", manual_confirm=False
    )
    assert res.accepted

    pnl = PnlTracker(storage).report()
    assert pnl.generated_at is not None
