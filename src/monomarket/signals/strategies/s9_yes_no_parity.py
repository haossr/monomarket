from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from monomarket.models import MarketView, Signal
from monomarket.signals.strategies.base import Strategy

OUTCOME_TOKEN_NO = "NO"  # nosec B105
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


class S9YesNoParityArb(Strategy):
    """
    S9: same-condition YES/NO parity arbitrage.

    Signal construction:
      - buy carry: buy cheapest YES + cheapest NO when yes+no < 1
      - sell overround: sell richest YES + richest NO when yes+no > 1

    Effective edge is local and depth/fee/slippage aware.
    """

    name = "s9"

    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        cfg = strategy_config or {}

        min_leg_liquidity = float(cfg.get("min_leg_liquidity", 80.0))
        max_order_notional = float(cfg.get("max_order_notional", 12.0))
        min_effective_edge_bps = float(cfg.get("min_effective_edge_bps", 20.0))
        fee_bps = float(cfg.get("fee_bps", 0.0))
        slippage_bps = float(cfg.get("slippage_bps", 6.0))
        depth_reference_liquidity = max(1.0, float(cfg.get("depth_reference_liquidity", 1000.0)))
        depth_penalty_max_bps = max(0.0, float(cfg.get("depth_penalty_max_bps", 30.0)))
        liquidity_fraction = max(0.0, float(cfg.get("liquidity_fraction", 0.01)))
        min_qty = max(0.1, float(cfg.get("min_qty", 1.0)))
        max_signals = max(0, int(float(cfg.get("max_signals", 120))))
        quote_improve = max(0.0, float(cfg.get("quote_improve", 0.0)))
        allow_sell_parity = _as_bool(cfg.get("allow_sell_parity"), False)

        if max_order_notional <= 0 or max_signals <= 0:
            return []

        by_canonical: dict[str, list[MarketView]] = defaultdict(list)
        for m in markets:
            if m.status != "open":
                continue
            if m.yes_price is None or m.no_price is None:
                continue
            if float(m.liquidity) < min_leg_liquidity:
                continue
            by_canonical[m.canonical_id].append(m)

        signals: list[Signal] = []
        for canonical_id, rows in by_canonical.items():
            if len(rows) < 2:
                continue

            buy_pair = self._best_pair(rows, buy_mode=True)
            if buy_pair is not None:
                buy_signals = self._emit_pair_signals(
                    canonical_id=canonical_id,
                    yes_leg=buy_pair[0],
                    no_leg=buy_pair[1],
                    side="buy",
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
                signals.extend(buy_signals)

            if allow_sell_parity:
                sell_pair = self._best_pair(rows, buy_mode=False)
                if sell_pair is not None:
                    sell_signals = self._emit_pair_signals(
                        canonical_id=canonical_id,
                        yes_leg=sell_pair[0],
                        no_leg=sell_pair[1],
                        side="sell",
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
                    signals.extend(sell_signals)

            if len(signals) >= max_signals:
                break

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals[:max_signals]

    @staticmethod
    def _best_pair(
        rows: list[MarketView],
        *,
        buy_mode: bool,
    ) -> tuple[MarketView, MarketView] | None:
        if buy_mode:
            yes_sorted = sorted(rows, key=lambda m: float(m.yes_price or 1.0))
            no_sorted = sorted(rows, key=lambda m: float(m.no_price or 1.0))
        else:
            yes_sorted = sorted(rows, key=lambda m: float(m.yes_price or 0.0), reverse=True)
            no_sorted = sorted(rows, key=lambda m: float(m.no_price or 0.0), reverse=True)

        if not yes_sorted or not no_sorted:
            return None

        # Prefer cross-market pair to avoid same-row token coupling where possible.
        for yes_leg in yes_sorted:
            for no_leg in no_sorted:
                if yes_leg.market_id != no_leg.market_id:
                    return yes_leg, no_leg

        return yes_sorted[0], no_sorted[0]

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

    def _emit_pair_signals(
        self,
        *,
        canonical_id: str,
        yes_leg: MarketView,
        no_leg: MarketView,
        side: str,
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
        yes_px = float(yes_leg.yes_price or 0.0)
        no_px = float(no_leg.no_price or 0.0)
        if yes_px <= 0 or no_px <= 0:
            return []

        parity_sum = yes_px + no_px
        if side == "buy":
            gross_edge = 1.0 - parity_sum
            direction = "buy_carry"
        else:
            gross_edge = parity_sum - 1.0
            direction = "sell_overround"

        if gross_edge <= 0:
            return []

        notional_ref = max(1e-9, parity_sum)
        gross_edge_bps = (gross_edge / notional_ref) * 10000.0

        yes_cost = self._leg_cost(
            liquidity=yes_leg.liquidity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            depth_reference_liquidity=depth_reference_liquidity,
            depth_penalty_max_bps=depth_penalty_max_bps,
        )
        no_cost = self._leg_cost(
            liquidity=no_leg.liquidity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            depth_reference_liquidity=depth_reference_liquidity,
            depth_penalty_max_bps=depth_penalty_max_bps,
        )

        total_cost_bps = yes_cost.total_bps + no_cost.total_bps
        effective_edge_bps = gross_edge_bps - total_cost_bps
        if effective_edge_bps < min_effective_edge_bps:
            return []

        min_leg_liq = min(float(yes_leg.liquidity), float(no_leg.liquidity))
        pair_qty = max(min_qty, min_leg_liq * liquidity_fraction)
        if max_order_notional > 0:
            pair_qty = min(pair_qty, max_order_notional / max(1e-9, parity_sum))
        if pair_qty <= 1e-12:
            return []

        pair_notional = pair_qty * parity_sum
        confidence = min(0.98, 0.50 + max(0.0, effective_edge_bps) / 600.0)
        score = (effective_edge_bps / 10000.0) * (1.0 + min_leg_liq / depth_reference_liquidity)
        pair_id = (
            f"{canonical_id}:{direction}:{yes_leg.market_id}:{no_leg.market_id}:"
            f"{yes_leg.source}:{no_leg.source}"
        )

        if side == "buy":
            yes_target = min(0.99, yes_px + quote_improve)
            no_target = min(0.99, no_px + quote_improve)
            rationale = (
                f"同条件平价套利({direction}): yes+no={parity_sum:.4f}<1, "
                f"gross={gross_edge_bps:.1f}bps, eff={effective_edge_bps:.1f}bps"
            )
        else:
            yes_target = max(0.01, yes_px - quote_improve)
            no_target = max(0.01, no_px - quote_improve)
            rationale = (
                f"同条件平价套利({direction}): yes+no={parity_sum:.4f}>1, "
                f"gross={gross_edge_bps:.1f}bps, eff={effective_edge_bps:.1f}bps"
            )

        common_payload = {
            "signal_source": "yes_no_parity_arb",
            "canonical_id": canonical_id,
            "pair_id": pair_id,
            "direction": direction,
            "parity_sum": parity_sum,
            "gross_edge": gross_edge,
            "gross_edge_bps": gross_edge_bps,
            "effective_edge_bps": effective_edge_bps,
            "pair_qty": pair_qty,
            "pair_notional": pair_notional,
            "cost_model": {
                "fee_bps_per_leg": fee_bps,
                "slippage_bps_per_leg": slippage_bps,
                "depth_reference_liquidity": depth_reference_liquidity,
                "depth_penalty_max_bps_per_leg": depth_penalty_max_bps,
                "yes_depth_penalty_bps": yes_cost.depth_penalty_bps,
                "no_depth_penalty_bps": no_cost.depth_penalty_bps,
                "yes_total_cost_bps": yes_cost.total_bps,
                "no_total_cost_bps": no_cost.total_bps,
                "total_cost_bps": total_cost_bps,
            },
            "legs": [
                {
                    "market_id": yes_leg.market_id,
                    "token": OUTCOME_TOKEN_YES,
                    "price": yes_target,
                    "side": side,
                },
                {
                    "market_id": no_leg.market_id,
                    "token": OUTCOME_TOKEN_NO,
                    "price": no_target,
                    "side": side,
                },
            ],
        }

        yes_payload = dict(common_payload)
        yes_payload.update(
            {
                "leg_role": "yes_leg",
                "partner_market_id": no_leg.market_id,
                "partner_token": OUTCOME_TOKEN_NO,
                "primary_leg": {
                    "token": OUTCOME_TOKEN_YES,
                    "market_id": yes_leg.market_id,
                    "qty": pair_qty,
                    "price": yes_target,
                },
            }
        )

        no_payload = dict(common_payload)
        no_payload.update(
            {
                "leg_role": "no_leg",
                "partner_market_id": yes_leg.market_id,
                "partner_token": OUTCOME_TOKEN_YES,
                "primary_leg": {
                    "token": OUTCOME_TOKEN_NO,
                    "market_id": no_leg.market_id,
                    "qty": pair_qty,
                    "price": no_target,
                },
            }
        )

        return [
            Signal(
                strategy=self.name,
                market_id=yes_leg.market_id,
                event_id=yes_leg.event_id,
                side=side,
                score=score,
                confidence=confidence,
                target_price=yes_target,
                size_hint=pair_qty,
                rationale=rationale,
                payload=yes_payload,
            ),
            Signal(
                strategy=self.name,
                market_id=no_leg.market_id,
                event_id=no_leg.event_id,
                side=side,
                score=score,
                confidence=confidence,
                target_price=no_target,
                size_hint=pair_qty,
                rationale=rationale,
                payload=no_payload,
            ),
        ]
