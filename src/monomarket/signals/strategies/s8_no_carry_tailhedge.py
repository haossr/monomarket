from __future__ import annotations

from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy


class S8NoCarryTailHedge(Strategy):
    name = "s8"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}
        yes_cap = float(cfg.get("yes_price_max_for_no", 0.25))
        hedge_ratio = float(cfg.get("hedge_budget_ratio", 0.15))

        mains = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and float(m.yes_price) <= yes_cap
            and m.no_price is not None
            and m.liquidity > 100
        ]
        tails = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and float(m.yes_price) <= 0.08
            and m.liquidity > 30
        ]

        mains.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))
        tails.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))

        signals: list[Signal] = []
        for m in mains[:30]:
            no_px = float(m.no_price or 1.0 - float(m.yes_price or 0.0))
            base_qty = max(3.0, m.liquidity * 0.012)
            hedge_budget = base_qty * no_px * hedge_ratio
            hedge = tails[0] if tails else None
            hedge_qty = 0.0
            if hedge and hedge.yes_price and hedge.yes_price > 0:
                hedge_qty = hedge_budget / float(hedge.yes_price)

            score = (1.0 - float(m.yes_price or 0.0)) * (1 + m.liquidity / 2500)
            signals.append(
                Signal(
                    strategy=self.name,
                    market_id=m.market_id,
                    event_id=m.event_id,
                    side="buy",
                    score=score,
                    confidence=min(0.97, 0.6 + (1 - float(m.yes_price or 0.0)) * 0.4),
                    target_price=min(0.99, no_px),
                    size_hint=base_qty,
                    rationale=(
                        f"高胜率NO carry: yes={float(m.yes_price or 0):.3f}, no={no_px:.3f}; "
                        f"tail hedge ratio={hedge_ratio:.2f}"
                    ),
                    payload={
                        "primary_leg": {
                            "token": "NO",
                            "market_id": m.market_id,
                            "qty": base_qty,
                            "price": no_px,
                        },
                        "tail_hedge": (
                            {
                                "token": "YES",
                                "market_id": hedge.market_id,
                                "qty": hedge_qty,
                                "price": float(hedge.yes_price or 0.0),
                            }
                            if hedge
                            else None
                        ),
                        "hedge_budget_ratio": hedge_ratio,
                    },
                )
            )

        return signals
