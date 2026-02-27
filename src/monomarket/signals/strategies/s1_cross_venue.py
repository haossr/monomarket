from __future__ import annotations

from collections import defaultdict
from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy


def _as_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class S1CrossVenueScanner(Strategy):
    name = "s1"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}
        min_spread = float(cfg.get("min_spread", 0.03))
        min_event_spread = float(cfg.get("min_event_spread", 0.05))
        min_liquidity = float(cfg.get("min_liquidity", 80.0))
        max_order_notional = float(cfg.get("max_order_notional", 25.0))
        max_signals_per_event = max(1, int(float(cfg.get("max_signals_per_event", 3))))
        if max_order_notional <= 0:
            return []
        event_notional_cap = float(cfg.get("event_notional_cap", 24.0))
        inventory_decay = max(0.0, min(0.95, float(cfg.get("inventory_decay", 0.20))))
        enable_cross_market_arb = _as_bool(cfg.get("enable_cross_market_arb"), True)

        tradable = [
            m
            for m in markets
            if m.status == "open"
            and m.mid_price is not None
            and float(m.liquidity) >= min_liquidity
        ]

        by_canonical: dict[str, list[MarketView]] = defaultdict(list)
        by_event: dict[str, list[MarketView]] = defaultdict(list)
        for m in tradable:
            by_canonical[m.canonical_id].append(m)
            by_event[m.event_id].append(m)

        candidates: list[Signal] = []

        for _canonical_id, rows in by_canonical.items():
            if len(rows) < 2:
                continue
            low = min(rows, key=lambda x: float(x.mid_price or 0.0))
            high = max(rows, key=lambda x: float(x.mid_price or 0.0))
            spread = float((high.mid_price or 0.0) - (low.mid_price or 0.0))
            if spread < min_spread:
                continue
            score = spread * (1.0 + min(low.liquidity, high.liquidity) / 2000.0)
            qty = max(4.0, min(low.liquidity, high.liquidity) * 0.02)
            target_price = min(0.99, float(low.mid_price or 0.5) + 0.005)
            if max_order_notional > 0 and target_price > 0:
                qty = min(qty, max_order_notional / target_price)
            edge_hint_bps = max(0.0, spread / max(1e-6, target_price) * 10000.0)
            candidates.append(
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
                        f"跨平台价差 {spread:.4f} (buy {low.source}/{low.market_id}, hedge {high.source}/{high.market_id})"
                    ),
                    payload={
                        "signal_source": "cross_venue_arb",
                        "spread": spread,
                        "edge_hint_bps": edge_hint_bps,
                    },
                )
            )

        if enable_cross_market_arb:
            for event_id, rows in by_event.items():
                if len(rows) < 2:
                    continue
                low = min(rows, key=lambda x: float(x.mid_price or 0.0))
                high = max(rows, key=lambda x: float(x.mid_price or 0.0))
                spread = float((high.mid_price or 0.0) - (low.mid_price or 0.0))
                if spread < min_event_spread:
                    continue
                if low.market_id == high.market_id:
                    continue
                score = spread * (1.0 + min(low.liquidity, high.liquidity) / 2500.0) * 0.85
                qty = max(3.0, min(low.liquidity, high.liquidity) * 0.015)
                target_price = min(0.99, float(low.mid_price or 0.5) + 0.003)
                if max_order_notional > 0 and target_price > 0:
                    qty = min(qty, max_order_notional / target_price)
                edge_hint_bps = max(0.0, spread / max(1e-6, target_price) * 10000.0)
                candidates.append(
                    Signal(
                        strategy=self.name,
                        market_id=low.market_id,
                        event_id=event_id,
                        side="buy",
                        score=score,
                        confidence=min(0.9, 0.45 + spread),
                        target_price=target_price,
                        size_hint=qty,
                        rationale=f"跨市场事件价差 {spread:.4f} (buy {low.market_id}, ref {high.market_id})",
                        payload={
                            "signal_source": "cross_market_arb",
                            "spread": spread,
                            "edge_hint_bps": edge_hint_bps,
                        },
                    )
                )

        candidates.sort(key=lambda s: s.score, reverse=True)
        out: list[Signal] = []
        event_count: dict[str, int] = defaultdict(int)
        event_notional: dict[str, float] = defaultdict(float)

        for signal in candidates:
            event_id = signal.event_id
            slot = event_count[event_id]
            if slot >= max_signals_per_event:
                continue
            decay = max(0.2, 1.0 - (inventory_decay * slot))
            qty = float(signal.size_hint) * decay
            target_price = max(0.01, float(signal.target_price))
            if max_order_notional > 0 and target_price > 0:
                qty = min(qty, max_order_notional / target_price)
            if event_notional_cap > 0 and target_price > 0:
                remain = event_notional_cap - event_notional[event_id]
                if remain <= 1e-12:
                    continue
                qty = min(qty, remain / target_price)
            if qty <= 1e-12:
                continue
            payload = dict(signal.payload or {})
            payload["inventory_decay_applied"] = decay
            payload["event_slot"] = slot + 1
            out.append(
                Signal(
                    strategy=signal.strategy,
                    market_id=signal.market_id,
                    event_id=signal.event_id,
                    side=signal.side,
                    score=signal.score,
                    confidence=signal.confidence,
                    target_price=signal.target_price,
                    size_hint=qty,
                    rationale=signal.rationale,
                    payload=payload,
                )
            )
            event_count[event_id] += 1
            event_notional[event_id] += qty * target_price

        return out
