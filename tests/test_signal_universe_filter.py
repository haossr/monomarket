from __future__ import annotations

from pathlib import Path
from typing import Any

from monomarket.config import (
    AppSettings,
    DataSettings,
    EdgeGateSettings,
    RiskSettings,
    Settings,
    TradingSettings,
    UniverseSettings,
)
from monomarket.db.storage import Storage
from monomarket.models import MarketView
from monomarket.signals.engine import SignalEngine


class _SpyStrategy:
    def __init__(self) -> None:
        self.seen_market_ids: list[str] = []

    def generate(self, markets: list[MarketView], _config: dict[str, Any]) -> list[Any]:
        self.seen_market_ids = [m.market_id for m in markets]
        return []


def _market(i: int, liquidity: float) -> MarketView:
    return MarketView(
        source="gamma",
        market_id=f"m{i}",
        canonical_id=f"c{i}",
        event_id=f"e{i}",
        question=f"Q{i}",
        status="open",
        neg_risk=False,
        liquidity=liquidity,
        volume=10.0,
        yes_price=0.4,
        no_price=0.6,
        mid_price=0.5,
    )


def test_signal_engine_applies_top_liquidity_universe_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "mono.db"
    storage = Storage(str(db_path))
    storage.init_db()

    markets = [_market(i, liquidity=float(i)) for i in range(1, 11)]
    storage.upsert_markets(markets)

    settings = Settings(
        app=AppSettings(db_path=str(db_path)),
        trading=TradingSettings(),
        risk=RiskSettings(),
        data=DataSettings(),
        strategies={"s1": {}},
        universe=UniverseSettings(liquidity_top_fraction=0.30),
        edge_gate=EdgeGateSettings(),
    )

    engine = SignalEngine(storage, settings)
    spy = _SpyStrategy()
    engine.registry = {"s1": spy}

    generated = engine.generate(["s1"], market_limit=2000)

    assert generated == []
    assert set(spy.seen_market_ids) == {"m8", "m9", "m10"}

    universe = engine.last_generation_stats.get("universe")
    assert isinstance(universe, dict)
    assert int(float(universe.get("selected_markets", 0))) == 3
    assert int(float(universe.get("total_markets", 0))) == 10
    assert abs(float(universe.get("selected_market_share", 0.0)) - 0.3) < 1e-9
    assert abs(float(universe.get("liquidity_cutoff", 0.0)) - 8.0) < 1e-9
