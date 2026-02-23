from __future__ import annotations

from pathlib import Path

from monomarket.config import AppSettings, DataSettings, RiskSettings, Settings, TradingSettings
from monomarket.db.storage import Storage
from monomarket.execution.router import ExecutionRouter
from monomarket.models import OrderRequest


def _settings(db_path: str) -> Settings:
    return Settings(
        app=AppSettings(db_path=db_path, log_level="INFO"),
        trading=TradingSettings(
            mode="paper", enable_live_trading=False, require_manual_confirm=True, kill_switch=False
        ),
        risk=RiskSettings(),
        data=DataSettings(),
        strategies={},
    )


def _req(mode: str = "paper") -> OrderRequest:
    return OrderRequest(
        strategy="manual",
        market_id="m1",
        event_id="e1",
        token_id="YES",
        side="buy",
        action="open",
        price=0.2,
        qty=10,
        mode=mode,
    )


def test_kill_switch_blocks_orders(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    settings = _settings(str(db))
    storage.set_switch("KILL_SWITCH", "true")

    router = ExecutionRouter(storage, settings)
    res = router.execute(_req("paper"))
    assert not res.accepted
    assert "KILL_SWITCH" in res.message


def test_live_requires_enable_switch(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()
    settings = _settings(str(db))
    router = ExecutionRouter(storage, settings)

    res = router.execute(_req("live"), requested_mode="live", manual_confirm=True)
    assert not res.accepted
    assert "ENABLE_LIVE_TRADING" in res.message
