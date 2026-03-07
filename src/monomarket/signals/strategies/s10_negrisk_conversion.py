from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy

OUTCOME_TOKEN_YES = "YES"  # nosec B105


@dataclass(slots=True)
class _LegCost:
    total_bps: float
    depth_penalty_bps: float


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


class S10NegRiskConversionArb(Strategy):
    """
    S10: NegRisk conversion/rebalance arbitrage.

    Event-level logic on negRisk events:
      - buy conversion: buy YES basket when sum(yes) < 1 - tolerance
      - sell conversion: sell YES basket when sum(yes) > 1 + tolerance (optional)

    Effective edge uses depth/fee/slippage-aware per-leg costs.
    """

    name = "s10"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}

        prob_sum_tolerance = max(0.0, float(cfg.get("prob_sum_tolerance", 0.02)))
        max_abs_deviation = max(
            prob_sum_tolerance,
            float(cfg.get("max_abs_deviation", 0.20)),
        )
        min_leg_liquidity = float(cfg.get("min_leg_liquidity", 100.0))
        max_order_notional = float(cfg.get("max_order_notional", 10.0))
        min_effective_edge_bps = float(cfg.get("min_effective_edge_bps", 30.0))
        fee_bps = float(cfg.get("fee_bps", 0.0))
        slippage_bps = float(cfg.get("slippage_bps", 6.0))
        depth_reference_liquidity = max(1.0, float(cfg.get("depth_reference_liquidity", 1000.0)))
        depth_penalty_max_bps = max(0.0, float(cfg.get("depth_penalty_max_bps", 35.0)))
        liquidity_fraction = max(0.0, float(cfg.get("liquidity_fraction", 0.01)))
        min_qty = max(0.1, float(cfg.get("min_qty", 1.0)))
        max_signals = max(0, int(float(cfg.get("max_signals", 180))))
        max_legs_per_event = max(0, int(float(cfg.get("max_legs_per_event", 32))))
        quote_improve = max(0.0, float(cfg.get("quote_improve", 0.0)))
        allow_sell_conversion = _as_bool(cfg.get("allow_sell_conversion"), False)

        exclude_event_ids_raw = cfg.get("exclude_event_ids", [])
        if isinstance(exclude_event_ids_raw, str | bytes):
            exclude_event_ids_raw = [exclude_event_ids_raw]
        exclude_event_ids = {
            str(event_id).strip() for event_id in exclude_event_ids_raw if str(event_id).strip()
        }

        if max_order_notional <= 0 or max_signals <= 0:
            return []

        by_event: dict[str, list[MarketView]] = defaultdict(list)
        for m in markets:
            if m.status != "open" or not m.neg_risk:
                continue
            if m.yes_price is None:
                continue
            if float(m.liquidity) < min_leg_liquidity:
                continue
            by_event[m.event_id].append(m)

        signals: list[Signal] = []
        for event_id, rows in by_event.items():
            if event_id in exclude_event_ids:
                continue
            if len(rows) < 2:
                continue

            if max_legs_per_event > 0 and len(rows) > max_legs_per_event:
                rows = sorted(rows, key=lambda m: float(m.liquidity), reverse=True)[:max_legs_per_event]

            sum_yes = sum(float(r.yes_price or 0.0) for r in rows)
            if sum_yes <= 1e-9:
                continue

            deviation = sum_yes - 1.0
            abs_deviation = abs(deviation)
            if abs_deviation > max_abs_deviation:
                continue

            direction: str | None = None
            gross_edge = 0.0
            if deviation < -prob_sum_tolerance:
                direction = "buy_conversion"
                gross_edge = 1.0 - sum_yes
            elif allow_sell_conversion and deviation > prob_sum_tolerance:
                direction = "sell_conversion"
                gross_edge = sum_yes - 1.0

            if direction is None or gross_edge <= 0:
                continue

            legs = self._emit_conversion_signals(
                event_id=event_id,
                rows=rows,
                sum_yes=sum_yes,
                direction=direction,
                gross_edge=gross_edge,
                quote_improve=quote_improve,
                min_effective_edge_bps=min_effective_edge_bps,
                max_order_notional=max_order_notional,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                depth_reference_liquidity=depth_reference_liquidity,
                depth_penalty_max_bps=depth_penalty_max_bps,
                liquidity_fraction=liquidity_fraction,
                min_qty=min_qty,
            )
            signals.extend(legs)

            if len(signals) >= max_signals:
                break

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals[:max_signals]

    @staticmethod
    def _leg_cost(
        *,
        liquidity: float,
        fee_bps: float,
        slippage_bps: float,
        depth_reference_liquidity: float,
        depth_penalty_max_bps: float,
    ) -> _LegCost:
        liq = max(0.0, float(liquidity))
        liq_ratio = min(1.0, liq / max(1e-9, depth_reference_liquidity))
        depth_penalty = (1.0 - liq_ratio) * max(0.0, depth_penalty_max_bps)
        return _LegCost(
            total_bps=max(0.0, fee_bps) + max(0.0, slippage_bps) + depth_penalty,
            depth_penalty_bps=depth_penalty,
        )

    def _emit_conversion_signals(
        self,
        *,
        event_id: str,
        rows: list[MarketView],
        sum_yes: float,
        direction: str,
        gross_edge: float,
        quote_improve: float,
        min_effective_edge_bps: float,
        max_order_notional: float,
        fee_bps: float,
        slippage_bps: float,
        depth_reference_liquidity: float,
        depth_penalty_max_bps: float,
        liquidity_fraction: float,
        min_qty: float,
    ) -> list[Signal]:
        gross_edge_bps = (gross_edge / max(sum_yes, 1e-9)) * 10000.0

        weighted_cost_numerator = 0.0
        depth_penalties: dict[str, float] = {}
        leg_cost_bps: dict[str, float] = {}
        for row in rows:
            yes_px = float(row.yes_price or 0.0)
            cost = self._leg_cost(
                liquidity=row.liquidity,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                depth_reference_liquidity=depth_reference_liquidity,
                depth_penalty_max_bps=depth_penalty_max_bps,
            )
            weighted_cost_numerator += yes_px * cost.total_bps
            depth_penalties[row.market_id] = cost.depth_penalty_bps
            leg_cost_bps[row.market_id] = cost.total_bps

        weighted_total_cost_bps = weighted_cost_numerator / max(sum_yes, 1e-9)
        effective_edge_bps = gross_edge_bps - weighted_total_cost_bps
        if effective_edge_bps < min_effective_edge_bps:
            return []

        min_leg_liq = min(float(r.liquidity) for r in rows)
        qty = max(min_qty, min_leg_liq * liquidity_fraction)
        if max_order_notional > 0:
            qty = min(qty, max_order_notional / max(sum_yes, 1e-9))
        if qty <= 1e-12:
            return []

        basket_notional = qty * sum_yes
        side = "buy" if direction == "buy_conversion" else "sell"
        confidence = min(0.99, 0.50 + max(0.0, effective_edge_bps) / 700.0)
        score = (effective_edge_bps / 10000.0) * (1.0 + min_leg_liq / depth_reference_liquidity)
        sorted_markets = sorted(str(r.market_id) for r in rows)
        basket_id = f"{event_id}:{direction}:{'-'.join(sorted_markets)}"

        if side == "buy":
            rationale = (
                f"NegRisk转换套利({direction}): sum_yes={sum_yes:.4f}<1, "
                f"gross={gross_edge_bps:.1f}bps, eff={effective_edge_bps:.1f}bps"
            )
        else:
            rationale = (
                f"NegRisk转换套利({direction}): sum_yes={sum_yes:.4f}>1, "
                f"gross={gross_edge_bps:.1f}bps, eff={effective_edge_bps:.1f}bps"
            )

        basket_markets = [
            {
                "market_id": r.market_id,
                "yes_price": float(r.yes_price or 0.0),
                "liquidity": float(r.liquidity),
            }
            for r in rows
        ]

        out: list[Signal] = []
        total_legs = len(rows)
        for idx, row in enumerate(rows):
            yes_px = float(row.yes_price or 0.0)
            if yes_px <= 0:
                continue

            target = (
                min(0.99, yes_px + quote_improve)
                if side == "buy"
                else max(0.01, yes_px - quote_improve)
            )

            payload = {
                "signal_source": "negrisk_conversion_arb",
                "event_id": event_id,
                "basket_id": basket_id,
                "direction": direction,
                "sum_yes": sum_yes,
                "deviation": sum_yes - 1.0,
                "gross_edge": gross_edge,
                "gross_edge_bps": gross_edge_bps,
                "effective_edge_bps": effective_edge_bps,
                "basket_qty": qty,
                "basket_notional": basket_notional,
                "convert_value": 1.0,
                "legs_count": total_legs,
                "leg_index": idx,
                "basket_markets": basket_markets,
                "cost_model": {
                    "fee_bps_per_leg": fee_bps,
                    "slippage_bps_per_leg": slippage_bps,
                    "depth_reference_liquidity": depth_reference_liquidity,
                    "depth_penalty_max_bps_per_leg": depth_penalty_max_bps,
                    "weighted_total_cost_bps": weighted_total_cost_bps,
                    "leg_total_cost_bps": leg_cost_bps.get(row.market_id, 0.0),
                    "leg_depth_penalty_bps": depth_penalties.get(row.market_id, 0.0),
                },
                "primary_leg": {
                    "token": OUTCOME_TOKEN_YES,
                    "market_id": row.market_id,
                    "qty": qty,
                    "price": target,
                },
            }

            out.append(
                Signal(
                    strategy=self.name,
                    market_id=row.market_id,
                    event_id=event_id,
                    side=side,
                    score=score,
                    confidence=confidence,
                    target_price=target,
                    size_hint=qty,
                    rationale=rationale,
                    payload=payload,
                )
            )

        return out
