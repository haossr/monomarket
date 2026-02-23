from __future__ import annotations

import os
from dataclasses import dataclass

from monomarket.config import Settings
from monomarket.db.storage import Storage
from monomarket.execution.live import LiveExecutor
from monomarket.execution.paper import PaperExecutor
from monomarket.execution.risk import UnifiedRiskGuard
from monomarket.models import OrderRequest, OrderResult


@dataclass(slots=True)
class Switches:
    enable_live_trading: bool
    require_manual_confirm: bool
    kill_switch: bool


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    t = value.strip().lower()
    if t in {"1", "true", "yes", "on", "y"}:
        return True
    if t in {"0", "false", "no", "off", "n"}:
        return False
    return default


class ExecutionRouter:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.risk = UnifiedRiskGuard(storage, settings)
        self.paper = PaperExecutor(storage)
        self.live = LiveExecutor(storage)

    def resolve_switches(self) -> Switches:
        # config defaults
        enable_live = self.settings.trading.enable_live_trading
        require_confirm = self.settings.trading.require_manual_confirm
        kill_switch = self.settings.trading.kill_switch

        # DB overrides
        db_live = self.storage.get_switch("ENABLE_LIVE_TRADING")
        db_confirm = self.storage.get_switch("REQUIRE_MANUAL_CONFIRM")
        db_kill = self.storage.get_switch("KILL_SWITCH")

        enable_live = _as_bool(db_live, enable_live)
        require_confirm = _as_bool(db_confirm, require_confirm)
        kill_switch = _as_bool(db_kill, kill_switch)

        # ENV highest priority (emergency)
        enable_live = _as_bool(os.getenv("ENABLE_LIVE_TRADING"), enable_live)
        require_confirm = _as_bool(os.getenv("REQUIRE_MANUAL_CONFIRM"), require_confirm)
        kill_switch = _as_bool(os.getenv("KILL_SWITCH"), kill_switch)

        return Switches(
            enable_live_trading=enable_live,
            require_manual_confirm=require_confirm,
            kill_switch=kill_switch,
        )

    def execute(
        self,
        req: OrderRequest,
        requested_mode: str | None = None,
        manual_confirm: bool = False,
    ) -> OrderResult:
        switches = self.resolve_switches()
        mode = (requested_mode or req.mode or self.settings.trading.mode).lower().strip()
        req.mode = mode

        if switches.kill_switch:
            order_id = self.storage.insert_order(
                req, status="rejected", message="KILL_SWITCH enabled"
            )
            return OrderResult(False, "rejected", order_id, "KILL_SWITCH enabled")

        if mode == "live":
            if not switches.enable_live_trading:
                order_id = self.storage.insert_order(
                    req,
                    status="rejected",
                    message="ENABLE_LIVE_TRADING is false",
                )
                return OrderResult(False, "rejected", order_id, "ENABLE_LIVE_TRADING is false")
            if switches.require_manual_confirm and not manual_confirm:
                order_id = self.storage.insert_order(
                    req,
                    status="rejected",
                    message="REQUIRE_MANUAL_CONFIRM=true and --confirm-live missing",
                )
                return OrderResult(
                    False,
                    "rejected",
                    order_id,
                    "REQUIRE_MANUAL_CONFIRM=true and --confirm-live missing",
                )

        risk_decision = self.risk.check(req)
        if not risk_decision.ok:
            order_id = self.storage.insert_order(
                req, status="rejected", message=risk_decision.reason
            )
            return OrderResult(False, "rejected", order_id, risk_decision.reason)

        if mode == "live":
            return self.live.execute(req)
        return self.paper.execute(req)
