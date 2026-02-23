from __future__ import annotations

from collections.abc import Iterable

from monomarket.config import Settings
from monomarket.db.storage import Storage
from monomarket.models import Signal
from monomarket.signals.strategies import (
    S1CrossVenueScanner,
    S2NegRiskRebalance,
    S4LowProbYesBasket,
    S8NoCarryTailHedge,
    Strategy,
)


class SignalEngine:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.registry: dict[str, Strategy] = {
            "s1": S1CrossVenueScanner(),
            "s2": S2NegRiskRebalance(),
            "s4": S4LowProbYesBasket(),
            "s8": S8NoCarryTailHedge(),
        }

    def generate(
        self, strategies: Iterable[str] | None = None, market_limit: int = 2000
    ) -> list[Signal]:
        selected = [s.lower() for s in strategies] if strategies else list(self.registry.keys())
        markets = self.storage.fetch_markets(limit=market_limit, status="open")

        out: list[Signal] = []
        for name in selected:
            impl = self.registry.get(name)
            if impl is None:
                continue
            cfg = self.settings.strategies.get(name, {})
            out.extend(impl.generate(markets, cfg))

        out.sort(key=lambda x: x.score, reverse=True)
        self.storage.insert_signals(out)
        return out
