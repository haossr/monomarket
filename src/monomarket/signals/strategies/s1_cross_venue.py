from __future__ import annotations

from collections import defaultdict
from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy


class S1CrossVenueScanner(Strategy):
    name = "s1"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}
        min_spread = float(cfg.get("min_spread", 0.03))
        max_order_notional = float(cfg.get("max_order_notional", 25.0))

        by_canonical: dict[str, list[MarketView]] = defaultdict(list)
        for m in markets:
            if m.status != "open":
                continue
            if m.mid_price is None:
                continue
            by_canonical[m.canonical_id].append(m)

        signals: list[Signal] = []
        for canonical_id, rows in by_canonical.items():
            if len(rows) < 2:
                continue
            low = min(rows, key=lambda x: float(x.mid_price or 0.0))
            high = max(rows, key=lambda x: float(x.mid_price or 0.0))
            spread = float((high.mid_price or 0.0) - (low.mid_price or 0.0))
            if spread < min_spread:
                continue
            score = spread * (1 + min(low.liquidity, high.liquidity) / 2000)
            qty = max(5.0, min(low.liquidity, high.liquidity) * 0.02)
            target_price = min(0.99, float(low.mid_price or 0.5) + 0.005)
            if max_order_notional > 0 and target_price > 0:
                qty = min(qty, max_order_notional / target_price)
            signals.append(
                Signal(
                    strategy=self.name,
                    market_id=low.market_id,
                    event_id=low.event_id,
                    side="buy",
                    score=score,
                    confidence=min(0.95, 0.5 + spread),
                    target_price=target_price,
                    size_hint=qty,
                    rationale=(
                        f"跨平台价差 {spread:.4f} (buy {low.source}/{low.market_id}, "
                        f"hedge on {high.source}/{high.market_id})"
                    ),
                    payload={
                        "scanner": "cross_venue",
                        "canonical_id": canonical_id,
                        "spread": spread,
                        "buy_leg": {
                            "source": low.source,
                            "market_id": low.market_id,
                            "mid": low.mid_price,
                        },
                        "sell_leg": {
                            "source": high.source,
                            "market_id": high.market_id,
                            "mid": high.mid_price,
                        },
                        "execution_mode": "semi_auto",
                    },
                )
            )

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
