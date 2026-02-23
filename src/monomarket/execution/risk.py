from __future__ import annotations

from dataclasses import dataclass

from monomarket.config import Settings
from monomarket.db.storage import Storage
from monomarket.models import OrderRequest


@dataclass(slots=True)
class RiskDecision:
    ok: bool
    reason: str = ""


class UnifiedRiskGuard:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings

    def check(self, req: OrderRequest) -> RiskDecision:
        if req.qty <= 0 or req.price <= 0:
            return RiskDecision(False, "invalid order qty/price")

        notional = abs(req.qty * req.price)
        strategy_now = self.storage.strategy_notional(req.strategy)
        event_now = self.storage.event_notional(req.event_id)
        realized = self.storage.total_realized_pnl()
        rej_count = self.storage.rejection_count()

        if realized <= -abs(self.settings.risk.max_daily_loss):
            return RiskDecision(False, f"global stop-loss triggered: realized={realized:.2f}")

        if strategy_now + notional > self.settings.risk.max_strategy_notional:
            return RiskDecision(
                False,
                (
                    f"strategy notional limit exceeded: {strategy_now + notional:.2f} "
                    f"> {self.settings.risk.max_strategy_notional:.2f}"
                ),
            )

        if event_now + notional > self.settings.risk.max_event_notional:
            return RiskDecision(
                False,
                (
                    f"event notional limit exceeded: {event_now + notional:.2f} "
                    f"> {self.settings.risk.max_event_notional:.2f}"
                ),
            )

        if rej_count >= self.settings.risk.circuit_breaker_rejections:
            return RiskDecision(
                False,
                f"circuit breaker open: rejected={rej_count}, threshold={self.settings.risk.circuit_breaker_rejections}",
            )

        return RiskDecision(True, "ok")
