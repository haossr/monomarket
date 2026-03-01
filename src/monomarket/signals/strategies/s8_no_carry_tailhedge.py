from __future__ import annotations

from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy

# Outcome token labels used in strategy payloads (not credentials).
OUTCOME_TOKEN_NO = "NO"  # nosec B105
OUTCOME_TOKEN_YES = "YES"  # nosec B105


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
        max_order_notional = float(cfg.get("max_order_notional", 25.0))
        max_candidates = int(cfg.get("max_candidates", 30))
        max_signals_per_event = int(cfg.get("max_signals_per_event", 0))
        if max_candidates < 1:
            return []
        if max_signals_per_event < 0:
            max_signals_per_event = 0

        edge_gate_min_bps = float(cfg.get("edge_gate_min_bps", 0.0))
        edge_gate_budget_penalty_bps = max(
            0.0,
            float(cfg.get("edge_gate_budget_penalty_bps", 0.0)),
        )
        edge_gate_budget_cap_notional = float(
            cfg.get("edge_gate_budget_cap_notional", max_order_notional)
        )
        if edge_gate_budget_cap_notional <= 0:
            edge_gate_budget_cap_notional = max(0.0, max_order_notional)

        exclude_event_ids_raw = cfg.get("exclude_event_ids", [])
        exclude_event_ids: set[str] = set()
        if isinstance(exclude_event_ids_raw, str):
            exclude_event_ids = {
                item.strip() for item in exclude_event_ids_raw.split(",") if item.strip()
            }
        elif isinstance(exclude_event_ids_raw, list):
            exclude_event_ids = {
                str(item).strip() for item in exclude_event_ids_raw if str(item).strip()
            }

        mains = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and float(m.yes_price) <= yes_cap
            and m.no_price is not None
            and m.liquidity > 100
            and str(m.event_id) not in exclude_event_ids
        ]
        tails = [
            m
            for m in markets
            if m.status == "open"
            and m.yes_price is not None
            and float(m.yes_price) <= 0.08
            and m.liquidity > 30
            and str(m.event_id) not in exclude_event_ids
        ]

        mains.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))
        tails.sort(key=lambda m: (float(m.yes_price or 1.0), -m.liquidity))

        signals: list[Signal] = []
        event_signal_counts: dict[str, int] = {}
        for m in mains:
            event_id = str(m.event_id)
            if (
                max_signals_per_event > 0
                and event_signal_counts.get(event_id, 0) >= max_signals_per_event
            ):
                continue
            yes_px = float(m.yes_price or 0.0)
            no_px = float(m.no_price or 1.0 - yes_px)
            base_qty = max(3.0, m.liquidity * 0.012)
            if max_order_notional > 0 and no_px > 0:
                base_qty = min(base_qty, max_order_notional / no_px)

            order_notional = base_qty * no_px
            edge_hint_bps = max(0.0, (1.0 - yes_px) * 1000.0)
            budget_utilization = (
                min(1.0, order_notional / edge_gate_budget_cap_notional)
                if edge_gate_budget_cap_notional > 0
                else 0.0
            )
            budget_penalty_bps = budget_utilization * edge_gate_budget_penalty_bps
            effective_edge_bps = edge_hint_bps - budget_penalty_bps
            if effective_edge_bps < edge_gate_min_bps:
                continue

            hedge_budget = order_notional * hedge_ratio
            hedge = tails[0] if tails else None
            hedge_qty = 0.0
            if hedge and hedge.yes_price and hedge.yes_price > 0:
                hedge_qty = hedge_budget / float(hedge.yes_price)

            score = (1.0 - yes_px) * (1 + m.liquidity / 2500)
            signals.append(
                Signal(
                    strategy=self.name,
                    market_id=m.market_id,
                    event_id=m.event_id,
                    side="buy",
                    score=score,
                    confidence=min(0.97, 0.6 + (1 - yes_px) * 0.4),
                    target_price=min(0.99, no_px),
                    size_hint=base_qty,
                    rationale=(
                        f"高胜率NO carry: yes={yes_px:.3f}, no={no_px:.3f}; "
                        f"tail hedge ratio={hedge_ratio:.2f}"
                    ),
                    payload={
                        "primary_leg": {
                            "token": OUTCOME_TOKEN_NO,
                            "market_id": m.market_id,
                            "qty": base_qty,
                            "price": no_px,
                        },
                        "tail_hedge": (
                            {
                                "token": OUTCOME_TOKEN_YES,
                                "market_id": hedge.market_id,
                                "qty": hedge_qty,
                                "price": float(hedge.yes_price or 0.0),
                            }
                            if hedge
                            else None
                        ),
                        "hedge_budget_ratio": hedge_ratio,
                        "edge_hint_bps": edge_hint_bps,
                        "edge_gate_local": {
                            "min_edge_bps": edge_gate_min_bps,
                            "budget_cap_notional": edge_gate_budget_cap_notional,
                            "budget_utilization": budget_utilization,
                            "budget_penalty_bps": budget_penalty_bps,
                            "effective_edge_bps": effective_edge_bps,
                        },
                    },
                )
            )
            event_signal_counts[event_id] = event_signal_counts.get(event_id, 0) + 1
            if len(signals) >= max_candidates:
                break

        return signals
