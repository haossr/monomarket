from __future__ import annotations

from collections import defaultdict
from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy


class S2NegRiskRebalance(Strategy):
    name = "s2"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}
        tol = float(cfg.get("prob_sum_tolerance", 0.04))
        max_order_notional = float(cfg.get("max_order_notional", 12.0))
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

        by_event: dict[str, list[MarketView]] = defaultdict(list)
        for m in markets:
            if m.status != "open" or not m.neg_risk:
                continue
            if m.yes_price is None:
                continue
            by_event[m.event_id].append(m)

        signals: list[Signal] = []
        for event_id, rows in by_event.items():
            if len(rows) < 2:
                continue
            prob_sum = sum(float(r.yes_price or 0.0) for r in rows)
            deviation = prob_sum - 1.0
            if abs(deviation) < tol:
                continue

            direction = "buy" if deviation < 0 else "sell"
            confidence = min(0.95, 0.55 + abs(deviation))
            score = abs(deviation) * len(rows)
            per_leg_qty = max(2.0, 15.0 * abs(deviation))

            for r in rows:
                px = float(r.yes_price or 0.5)
                target = max(0.01, min(0.99, px + (0.005 if direction == "buy" else -0.005)))
                leg_notional = per_leg_qty * target

                # Local per-strategy edge gate: penalize extreme basket dislocation
                # and budget saturation before emitting replay signals.
                edge_hint_bps = max(0.0, (1.0 - min(1.0, abs(deviation))) * 1000.0)
                budget_utilization = (
                    min(1.0, leg_notional / edge_gate_budget_cap_notional)
                    if edge_gate_budget_cap_notional > 0
                    else 0.0
                )
                budget_penalty_bps = budget_utilization * edge_gate_budget_penalty_bps
                effective_edge_bps = edge_hint_bps - budget_penalty_bps
                if effective_edge_bps < edge_gate_min_bps:
                    continue

                signals.append(
                    Signal(
                        strategy=self.name,
                        market_id=r.market_id,
                        event_id=event_id,
                        side=direction,
                        score=score,
                        confidence=confidence,
                        target_price=target,
                        size_hint=per_leg_qty,
                        rationale=(
                            f"NegRisk组合偏离: sum_yes={prob_sum:.4f}, deviation={deviation:+.4f}, "
                            f"rebalance={direction}"
                        ),
                        payload={
                            "event_id": event_id,
                            "prob_sum": prob_sum,
                            "deviation": deviation,
                            "basket_markets": [x.market_id for x in rows],
                            "leg_weight": 1 / len(rows),
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

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
