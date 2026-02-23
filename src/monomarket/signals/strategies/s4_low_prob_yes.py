from __future__ import annotations

from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy


class S4LowProbYesBasket(Strategy):
    name = "s4"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}
        p_min = float(cfg.get("yes_price_min", 0.01))
        p_max = float(cfg.get("yes_price_max", 0.15))

        candidates = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and p_min <= float(m.yes_price) <= p_max
            and m.liquidity > 50
        ]
        candidates.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))

        signals: list[Signal] = []
        for idx, m in enumerate(candidates[:40], start=1):
            tier = "A" if idx <= 10 else "B" if idx <= 25 else "C"
            tier_mult = {"A": 1.2, "B": 1.0, "C": 0.7}[tier]
            base_qty = max(2.0, m.liquidity * 0.01)
            size = base_qty * tier_mult
            px = float(m.yes_price or 0.05)
            ladder = [max(0.01, px), max(0.01, px - 0.01), max(0.01, px - 0.02)]
            score = (p_max - px + 0.01) * (1 + m.liquidity / 2000)

            signals.append(
                Signal(
                    strategy=self.name,
                    market_id=m.market_id,
                    event_id=m.event_id,
                    side="buy",
                    score=score,
                    confidence=min(0.9, 0.45 + (p_max - px)),
                    target_price=ladder[0],
                    size_hint=size,
                    rationale=f"低概率YES篮子候选 tier={tier}, yes={px:.3f}, liq={m.liquidity:.1f}",
                    payload={
                        "tier": tier,
                        "ladder_prices": ladder,
                        "allocation_mult": tier_mult,
                        "basket_family": "low_prob_yes",
                    },
                )
            )

        return signals
