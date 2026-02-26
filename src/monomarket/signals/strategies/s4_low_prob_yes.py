from __future__ import annotations

from collections import defaultdict
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
        max_candidates = int(cfg.get("max_candidates", 40))
        min_edge_buffer = float(cfg.get("min_edge_buffer", 0.015))
        min_no_price = float(cfg.get("min_no_price", 0.82))
        min_liquidity = float(cfg.get("min_liquidity", 80.0))
        max_order_notional = float(cfg.get("max_order_notional", 20.0))
        max_signals_per_event = max(1, int(float(cfg.get("max_signals_per_event", 3))))
        event_notional_cap = float(cfg.get("event_notional_cap", 30.0))
        inventory_decay = max(0.0, min(0.95, float(cfg.get("inventory_decay", 0.18))))

        candidates = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and m.no_price is not None
            and p_min <= float(m.yes_price) <= p_max
            and float(m.no_price) >= min_no_price
            and (p_max - float(m.yes_price)) >= min_edge_buffer
            and float(m.liquidity) >= min_liquidity
        ]
        candidates.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))

        event_count: dict[str, int] = defaultdict(int)
        event_notional: dict[str, float] = defaultdict(float)

        signals: list[Signal] = []
        for idx, m in enumerate(candidates[:max_candidates], start=1):
            event_id = str(m.event_id)
            slot = event_count[event_id]
            if slot >= max_signals_per_event:
                continue

            tier = "A" if idx <= 12 else "B" if idx <= 30 else "C"
            tier_mult = {"A": 1.2, "B": 1.0, "C": 0.7}[tier]
            base_qty = max(2.0, float(m.liquidity) * 0.01)
            size = base_qty * tier_mult
            decay = max(0.25, 1.0 - (inventory_decay * slot))
            size *= decay

            px = float(m.yes_price or 0.05)
            ladder = [max(0.01, px), max(0.01, px - 0.01), max(0.01, px - 0.02)]
            target_price = ladder[0]
            if max_order_notional > 0 and target_price > 0:
                size = min(size, max_order_notional / target_price)
            if event_notional_cap > 0 and target_price > 0:
                remain = event_notional_cap - event_notional[event_id]
                if remain <= 1e-12:
                    continue
                size = min(size, remain / target_price)
            if size <= 1e-12:
                continue

            score = (p_max - px + 0.01) * (1 + m.liquidity / 2000)
            edge_hint_bps = max(0.0, (p_max - px) * 10000.0)
            signals.append(
                Signal(
                    strategy=self.name,
                    market_id=m.market_id,
                    event_id=m.event_id,
                    side="buy",
                    score=score,
                    confidence=min(0.9, 0.45 + (p_max - px)),
                    target_price=target_price,
                    size_hint=size,
                    rationale=(
                        f"低概率YES篮子候选 tier={tier}, yes={px:.3f}, no={float(m.no_price or 0):.3f}, "
                        f"liq={m.liquidity:.1f}, slot={slot + 1}"
                    ),
                    payload={
                        "signal_source": "low_prob_yes_mm",
                        "tier": tier,
                        "ladder_prices": ladder,
                        "allocation_mult": tier_mult,
                        "basket_family": "low_prob_yes",
                        "edge_hint_bps": edge_hint_bps,
                        "inventory_decay_applied": decay,
                        "event_slot": slot + 1,
                        "event_notional_cap": event_notional_cap,
                    },
                )
            )
            event_count[event_id] += 1
            event_notional[event_id] += size * target_price

        return signals
