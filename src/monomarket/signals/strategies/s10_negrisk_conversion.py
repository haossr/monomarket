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


@dataclass(slots=True)
class _ConversionCandidate:
    direction: str
    rows: list[MarketView]
    sum_yes: float
    gross_edge: float
    deviation: float
    max_leg_weight: float
    raw_leg_count: int
    selection_mode: str


@dataclass(slots=True)
class _ConversionPricingDiagnostics:
    pre_quote_sum_yes: float
    post_quote_sum_yes: float
    post_slippage_sum_yes: float
    pre_quote_gross_edge_bps: float
    post_quote_gross_edge_bps: float
    post_slippage_gross_edge_bps: float
    post_quote_effective_edge_bps: float
    post_slippage_effective_edge_bps: float
    floor_adjusted_legs: int
    tiny_price_legs: int


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


def _bump_counter(bucket: dict[str, int], key: str) -> None:
    bucket[key] = int(bucket.get(key, 0)) + 1


def _record_reject_reason(
    *,
    reject_reasons: dict[str, int],
    reject_reasons_by_event: dict[str, dict[str, int]],
    event_id: str,
    reason: str,
) -> None:
    _bump_counter(reject_reasons, reason)
    event_rejects = reject_reasons_by_event.setdefault(event_id, {})
    _bump_counter(event_rejects, reason)


def _normalize_reject_reasons_by_event(
    reject_reasons_by_event: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        event_id: {reason: reasons[reason] for reason in sorted(reasons)}
        for event_id, reasons in sorted(reject_reasons_by_event.items())
        if reasons
    }


def _top_k_reject_reasons_by_event(
    reject_reasons_by_event: dict[str, dict[str, int]],
    *,
    top_k: int,
) -> dict[str, dict[str, int]]:
    normalized = _normalize_reject_reasons_by_event(reject_reasons_by_event)
    if top_k <= 0:
        return normalized

    ranked = sorted(
        normalized.items(),
        key=lambda item: (-sum(int(v) for v in item[1].values()), item[0]),
    )
    return {event_id: reasons for event_id, reasons in ranked[:top_k]}


class S10NegRiskConversionArb(Strategy):
    """
    S10: NegRisk conversion/rebalance arbitrage.

    Event-level logic on negRisk events:
      - buy conversion: buy YES basket when sum(yes) < 1 - tolerance
      - sell conversion: sell YES basket when sum(yes) > 1 + tolerance (optional)

    Effective edge uses depth/fee/slippage-aware per-leg costs.
    """

    name = "s10"

    def __init__(self) -> None:
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return float(total) / float(count)

    @staticmethod
    def _clamp_target_price(price: float, *, price_floor: float) -> float:
        return max(price_floor, min(0.99, float(price)))

    @classmethod
    def _expected_fill_price(
        cls,
        *,
        side: str,
        target_price: float,
        slippage_bps: float,
        price_floor: float,
    ) -> float:
        px = cls._clamp_target_price(target_price, price_floor=price_floor)
        slip = px * max(0.0, slippage_bps) / 10000.0
        if side == "buy":
            return min(0.99, px + slip)
        return max(price_floor, px - slip)

    @staticmethod
    def _gross_edge_from_sum_yes(*, direction: str, sum_yes: float) -> float:
        if direction == "buy_conversion":
            return 1.0 - sum_yes
        return sum_yes - 1.0

    @classmethod
    def _gross_edge_bps_from_sum_yes(cls, *, direction: str, sum_yes: float) -> float:
        gross_edge = cls._gross_edge_from_sum_yes(direction=direction, sum_yes=sum_yes)
        return (gross_edge / max(sum_yes, 1e-9)) * 10000.0

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
        depth_reference_liquidity = max(
            1.0,
            float(cfg.get("depth_reference_liquidity", 1000.0)),
        )
        depth_penalty_max_bps = max(0.0, float(cfg.get("depth_penalty_max_bps", 35.0)))
        liquidity_fraction = max(0.0, float(cfg.get("liquidity_fraction", 0.01)))
        min_qty = max(0.1, float(cfg.get("min_qty", 1.0)))
        max_signals = max(0, int(float(cfg.get("max_signals", 180))))
        max_legs_per_event = max(0, int(float(cfg.get("max_legs_per_event", 32))))
        min_unique_canonicals = max(2, int(float(cfg.get("min_unique_canonicals", 3))))
        max_leg_weight = min(1.0, max(0.0, float(cfg.get("max_leg_weight", 0.70))))
        max_weighted_total_cost_bps = max(
            0.0,
            float(cfg.get("max_weighted_total_cost_bps", 120.0)),
        )
        max_leg_total_cost_bps = max(0.0, float(cfg.get("max_leg_total_cost_bps", 95.0)))
        quote_improve = max(0.0, float(cfg.get("quote_improve", 0.0)))
        executable_price_floor = max(0.01, float(cfg.get("executable_price_floor", 0.01)))
        max_tiny_price_leg_share = min(
            1.0,
            max(0.0, float(cfg.get("max_tiny_price_leg_share", 1.0))),
        )
        max_floor_adjusted_leg_share = min(
            1.0,
            max(0.0, float(cfg.get("max_floor_adjusted_leg_share", 1.0))),
        )
        allow_sell_conversion = _as_bool(cfg.get("allow_sell_conversion"), False)
        diagnostics_event_top_k = max(0, int(float(cfg.get("diagnostics_event_top_k", 20))))

        exclude_event_ids_raw = cfg.get("exclude_event_ids", [])
        if isinstance(exclude_event_ids_raw, str | bytes):
            exclude_event_ids_raw = [exclude_event_ids_raw]
        exclude_event_ids = {
            str(event_id).strip() for event_id in exclude_event_ids_raw if str(event_id).strip()
        }

        if max_order_notional <= 0 or max_signals <= 0:
            self.last_diagnostics = {
                "events_total": 0,
                "events_with_candidates": 0,
                "signals_emitted": 0,
                "direction_attempts": {"buy_conversion": 0, "sell_conversion": 0},
                "direction_pass": {"buy_conversion": 0, "sell_conversion": 0},
                "candidate_reject_reasons": {"strategy_disabled": 1},
                "candidate_reject_reasons_by_event": {},
                "candidate_reject_reasons_by_event_top_k": diagnostics_event_top_k,
                "candidate_reject_reasons_by_event_top": {},
                "pricing_consistency": {
                    "price_floor": executable_price_floor,
                    "max_tiny_price_leg_share": max_tiny_price_leg_share,
                    "max_floor_adjusted_leg_share": max_floor_adjusted_leg_share,
                    "pair_candidates_priced": 0,
                    "pairs_with_floor_adjustment": 0,
                    "tiny_price_pairs": 0,
                    "avg_pre_floor_gross_edge_bps": 0.0,
                    "avg_post_floor_gross_edge_bps": 0.0,
                    "avg_post_slippage_gross_edge_bps": 0.0,
                    "avg_post_slippage_effective_edge_bps": 0.0,
                    "filtered_post_floor_non_positive": 0,
                    "filtered_post_slippage_non_positive": 0,
                    "filtered_post_slippage_effective_edge_below_min": 0,
                    "filtered_tiny_price_leg_share_exceeded": 0,
                    "filtered_floor_adjusted_leg_share_exceeded": 0,
                },
            }
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

        reject_reasons: dict[str, int] = {}
        reject_reasons_by_event: dict[str, dict[str, int]] = {}
        direction_attempts: dict[str, int] = {"buy_conversion": 0, "sell_conversion": 0}
        direction_pass: dict[str, int] = {"buy_conversion": 0, "sell_conversion": 0}

        signals: list[Signal] = []
        events_with_candidates = 0

        pricing_diag_count = 0
        pricing_diag_sum_pre_quote_sum_yes = 0.0
        pricing_diag_sum_post_quote_sum_yes = 0.0
        pricing_diag_sum_post_slippage_sum_yes = 0.0
        pricing_diag_sum_pre_quote_gross_bps = 0.0
        pricing_diag_sum_post_quote_gross_bps = 0.0
        pricing_diag_sum_post_slippage_gross_bps = 0.0
        pricing_diag_sum_post_quote_effective_bps = 0.0
        pricing_diag_sum_post_slippage_effective_bps = 0.0
        pricing_diag_floor_adjusted_baskets = 0
        pricing_diag_tiny_price_legs = 0
        pricing_diag_tiny_price_baskets = 0
        pricing_diag_filtered_post_quote_non_positive = 0
        pricing_diag_filtered_post_slippage_non_positive = 0
        pricing_diag_filtered_effective_below_min = 0
        pricing_diag_filtered_tiny_price_leg_share_exceeded = 0
        pricing_diag_filtered_floor_adjusted_leg_share_exceeded = 0

        for event_id, event_rows in by_event.items():
            if event_id in exclude_event_ids:
                _record_reject_reason(
                    reject_reasons=reject_reasons,
                    reject_reasons_by_event=reject_reasons_by_event,
                    event_id=event_id,
                    reason="event_excluded",
                )
                continue
            if len(event_rows) < min_unique_canonicals:
                _record_reject_reason(
                    reject_reasons=reject_reasons,
                    reject_reasons_by_event=reject_reasons_by_event,
                    event_id=event_id,
                    reason="event_under_min_unique",
                )
                continue

            candidates: list[list[Signal]] = []

            buy_rows = self._select_rows_for_direction(
                event_rows,
                buy_mode=True,
                max_legs_per_event=max_legs_per_event,
            )
            direction_attempts["buy_conversion"] += 1
            buy_candidate, buy_candidate_reject_reason = self._build_candidate(
                rows=buy_rows,
                direction="buy_conversion",
                prob_sum_tolerance=prob_sum_tolerance,
                max_abs_deviation=max_abs_deviation,
                max_leg_weight=max_leg_weight,
                min_unique_canonicals=min_unique_canonicals,
                raw_leg_count=len(event_rows),
            )
            if buy_candidate is not None:
                (
                    buy_signals,
                    buy_emit_reject_reason,
                    buy_pricing_diag,
                ) = self._emit_conversion_signals(
                    event_id=event_id,
                    candidate=buy_candidate,
                    quote_improve=quote_improve,
                    min_effective_edge_bps=min_effective_edge_bps,
                    max_order_notional=max_order_notional,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    depth_reference_liquidity=depth_reference_liquidity,
                    depth_penalty_max_bps=depth_penalty_max_bps,
                    max_weighted_total_cost_bps=max_weighted_total_cost_bps,
                    max_leg_total_cost_bps=max_leg_total_cost_bps,
                    liquidity_fraction=liquidity_fraction,
                    min_qty=min_qty,
                    executable_price_floor=executable_price_floor,
                    max_tiny_price_leg_share=max_tiny_price_leg_share,
                    max_floor_adjusted_leg_share=max_floor_adjusted_leg_share,
                )
                if buy_pricing_diag is not None:
                    pricing_diag_count += 1
                    pricing_diag_sum_pre_quote_sum_yes += buy_pricing_diag.pre_quote_sum_yes
                    pricing_diag_sum_post_quote_sum_yes += buy_pricing_diag.post_quote_sum_yes
                    pricing_diag_sum_post_slippage_sum_yes += buy_pricing_diag.post_slippage_sum_yes
                    pricing_diag_sum_pre_quote_gross_bps += (
                        buy_pricing_diag.pre_quote_gross_edge_bps
                    )
                    pricing_diag_sum_post_quote_gross_bps += (
                        buy_pricing_diag.post_quote_gross_edge_bps
                    )
                    pricing_diag_sum_post_slippage_gross_bps += (
                        buy_pricing_diag.post_slippage_gross_edge_bps
                    )
                    pricing_diag_sum_post_quote_effective_bps += (
                        buy_pricing_diag.post_quote_effective_edge_bps
                    )
                    pricing_diag_sum_post_slippage_effective_bps += (
                        buy_pricing_diag.post_slippage_effective_edge_bps
                    )
                    pricing_diag_tiny_price_legs += int(buy_pricing_diag.tiny_price_legs)
                    if buy_pricing_diag.floor_adjusted_legs > 0:
                        pricing_diag_floor_adjusted_baskets += 1
                    if buy_pricing_diag.tiny_price_legs > 0:
                        pricing_diag_tiny_price_baskets += 1

                if buy_emit_reject_reason == "post_quote_non_positive_gross_edge":
                    pricing_diag_filtered_post_quote_non_positive += 1
                elif buy_emit_reject_reason == "post_slippage_non_positive_gross_edge":
                    pricing_diag_filtered_post_slippage_non_positive += 1
                elif buy_emit_reject_reason == "effective_edge_below_min":
                    pricing_diag_filtered_effective_below_min += 1
                elif buy_emit_reject_reason == "tiny_price_leg_share_exceeded":
                    pricing_diag_filtered_tiny_price_leg_share_exceeded += 1
                elif buy_emit_reject_reason == "floor_adjusted_leg_share_exceeded":
                    pricing_diag_filtered_floor_adjusted_leg_share_exceeded += 1

                if buy_signals:
                    direction_pass["buy_conversion"] += 1
                    candidates.append(buy_signals)
                elif buy_emit_reject_reason:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_id,
                        reason=f"buy_conversion:{buy_emit_reject_reason}",
                    )
            elif buy_candidate_reject_reason:
                _record_reject_reason(
                    reject_reasons=reject_reasons,
                    reject_reasons_by_event=reject_reasons_by_event,
                    event_id=event_id,
                    reason=f"buy_conversion:{buy_candidate_reject_reason}",
                )

            if allow_sell_conversion:
                sell_rows = self._select_rows_for_direction(
                    event_rows,
                    buy_mode=False,
                    max_legs_per_event=max_legs_per_event,
                )
                direction_attempts["sell_conversion"] += 1
                sell_candidate, sell_candidate_reject_reason = self._build_candidate(
                    rows=sell_rows,
                    direction="sell_conversion",
                    prob_sum_tolerance=prob_sum_tolerance,
                    max_abs_deviation=max_abs_deviation,
                    max_leg_weight=max_leg_weight,
                    min_unique_canonicals=min_unique_canonicals,
                    raw_leg_count=len(event_rows),
                )
                if sell_candidate is not None:
                    (
                        sell_signals,
                        sell_emit_reject_reason,
                        sell_pricing_diag,
                    ) = self._emit_conversion_signals(
                        event_id=event_id,
                        candidate=sell_candidate,
                        quote_improve=quote_improve,
                        min_effective_edge_bps=min_effective_edge_bps,
                        max_order_notional=max_order_notional,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        depth_reference_liquidity=depth_reference_liquidity,
                        depth_penalty_max_bps=depth_penalty_max_bps,
                        max_weighted_total_cost_bps=max_weighted_total_cost_bps,
                        max_leg_total_cost_bps=max_leg_total_cost_bps,
                        liquidity_fraction=liquidity_fraction,
                        min_qty=min_qty,
                        executable_price_floor=executable_price_floor,
                        max_tiny_price_leg_share=max_tiny_price_leg_share,
                        max_floor_adjusted_leg_share=max_floor_adjusted_leg_share,
                    )
                    if sell_pricing_diag is not None:
                        pricing_diag_count += 1
                        pricing_diag_sum_pre_quote_sum_yes += sell_pricing_diag.pre_quote_sum_yes
                        pricing_diag_sum_post_quote_sum_yes += sell_pricing_diag.post_quote_sum_yes
                        pricing_diag_sum_post_slippage_sum_yes += (
                            sell_pricing_diag.post_slippage_sum_yes
                        )
                        pricing_diag_sum_pre_quote_gross_bps += (
                            sell_pricing_diag.pre_quote_gross_edge_bps
                        )
                        pricing_diag_sum_post_quote_gross_bps += (
                            sell_pricing_diag.post_quote_gross_edge_bps
                        )
                        pricing_diag_sum_post_slippage_gross_bps += (
                            sell_pricing_diag.post_slippage_gross_edge_bps
                        )
                        pricing_diag_sum_post_quote_effective_bps += (
                            sell_pricing_diag.post_quote_effective_edge_bps
                        )
                        pricing_diag_sum_post_slippage_effective_bps += (
                            sell_pricing_diag.post_slippage_effective_edge_bps
                        )
                        pricing_diag_tiny_price_legs += int(sell_pricing_diag.tiny_price_legs)
                        if sell_pricing_diag.floor_adjusted_legs > 0:
                            pricing_diag_floor_adjusted_baskets += 1
                        if sell_pricing_diag.tiny_price_legs > 0:
                            pricing_diag_tiny_price_baskets += 1

                    if sell_emit_reject_reason == "post_quote_non_positive_gross_edge":
                        pricing_diag_filtered_post_quote_non_positive += 1
                    elif sell_emit_reject_reason == "post_slippage_non_positive_gross_edge":
                        pricing_diag_filtered_post_slippage_non_positive += 1
                    elif sell_emit_reject_reason == "effective_edge_below_min":
                        pricing_diag_filtered_effective_below_min += 1
                    elif sell_emit_reject_reason == "tiny_price_leg_share_exceeded":
                        pricing_diag_filtered_tiny_price_leg_share_exceeded += 1
                    elif sell_emit_reject_reason == "floor_adjusted_leg_share_exceeded":
                        pricing_diag_filtered_floor_adjusted_leg_share_exceeded += 1

                    if sell_signals:
                        direction_pass["sell_conversion"] += 1
                        candidates.append(sell_signals)
                    elif sell_emit_reject_reason:
                        _record_reject_reason(
                            reject_reasons=reject_reasons,
                            reject_reasons_by_event=reject_reasons_by_event,
                            event_id=event_id,
                            reason=f"sell_conversion:{sell_emit_reject_reason}",
                        )
                elif sell_candidate_reject_reason:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_id,
                        reason=f"sell_conversion:{sell_candidate_reject_reason}",
                    )

            if not candidates:
                _record_reject_reason(
                    reject_reasons=reject_reasons,
                    reject_reasons_by_event=reject_reasons_by_event,
                    event_id=event_id,
                    reason="event_no_actionable_candidate",
                )
                continue

            events_with_candidates += 1
            best_candidate = max(candidates, key=lambda legs: float(legs[0].score))
            signals.extend(best_candidate)

            if len(signals) >= max_signals:
                break

        signals.sort(key=lambda s: s.score, reverse=True)
        selected_signals = signals[:max_signals]
        reject_by_event = _normalize_reject_reasons_by_event(reject_reasons_by_event)
        self.last_diagnostics = {
            "events_total": len(by_event),
            "events_with_candidates": events_with_candidates,
            "signals_emitted": len(selected_signals),
            "direction_attempts": direction_attempts,
            "direction_pass": direction_pass,
            "candidate_reject_reasons": {
                key: reject_reasons[key] for key in sorted(reject_reasons)
            },
            "candidate_reject_reasons_by_event": reject_by_event,
            "candidate_reject_reasons_by_event_top_k": diagnostics_event_top_k,
            "candidate_reject_reasons_by_event_top": _top_k_reject_reasons_by_event(
                reject_by_event,
                top_k=diagnostics_event_top_k,
            ),
            "pricing_consistency": {
                "price_floor": executable_price_floor,
                "max_tiny_price_leg_share": max_tiny_price_leg_share,
                "max_floor_adjusted_leg_share": max_floor_adjusted_leg_share,
                "pair_candidates_priced": pricing_diag_count,
                "pairs_with_floor_adjustment": pricing_diag_floor_adjusted_baskets,
                "tiny_price_pairs": pricing_diag_tiny_price_baskets,
                "tiny_price_legs": pricing_diag_tiny_price_legs,
                "avg_pre_quote_sum_yes": self._safe_avg(
                    pricing_diag_sum_pre_quote_sum_yes,
                    pricing_diag_count,
                ),
                "avg_post_quote_sum_yes": self._safe_avg(
                    pricing_diag_sum_post_quote_sum_yes,
                    pricing_diag_count,
                ),
                "avg_post_slippage_sum_yes": self._safe_avg(
                    pricing_diag_sum_post_slippage_sum_yes,
                    pricing_diag_count,
                ),
                "avg_pre_floor_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_pre_quote_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_floor_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_quote_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_slippage_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_slippage_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_quote_effective_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_quote_effective_bps,
                    pricing_diag_count,
                ),
                "avg_post_slippage_effective_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_slippage_effective_bps,
                    pricing_diag_count,
                ),
                "filtered_post_floor_non_positive": pricing_diag_filtered_post_quote_non_positive,
                "filtered_post_slippage_non_positive": pricing_diag_filtered_post_slippage_non_positive,
                "filtered_post_slippage_effective_edge_below_min": (
                    pricing_diag_filtered_effective_below_min
                ),
                "filtered_tiny_price_leg_share_exceeded": (
                    pricing_diag_filtered_tiny_price_leg_share_exceeded
                ),
                "filtered_floor_adjusted_leg_share_exceeded": (
                    pricing_diag_filtered_floor_adjusted_leg_share_exceeded
                ),
            },
        }
        return selected_signals

    @staticmethod
    def _select_rows_for_direction(
        rows: list[MarketView],
        *,
        buy_mode: bool,
        max_legs_per_event: int,
    ) -> list[MarketView]:
        by_canonical: dict[str, list[MarketView]] = defaultdict(list)
        for row in rows:
            by_canonical[row.canonical_id].append(row)

        selected: list[MarketView] = []
        for canonical_rows in by_canonical.values():
            if buy_mode:
                canonical_rows_sorted = sorted(
                    canonical_rows,
                    key=lambda m: (
                        float(m.yes_price or 1.0),
                        -float(m.liquidity),
                        str(m.market_id),
                    ),
                )
            else:
                canonical_rows_sorted = sorted(
                    canonical_rows,
                    key=lambda m: (
                        -float(m.yes_price or 0.0),
                        -float(m.liquidity),
                        str(m.market_id),
                    ),
                )
            selected.append(canonical_rows_sorted[0])

        if max_legs_per_event > 0 and len(selected) > max_legs_per_event:
            if buy_mode:
                selected = sorted(
                    selected,
                    key=lambda m: (
                        -float(m.liquidity),
                        float(m.yes_price or 1.0),
                        str(m.market_id),
                    ),
                )[:max_legs_per_event]
            else:
                selected = sorted(
                    selected,
                    key=lambda m: (
                        -float(m.liquidity),
                        -float(m.yes_price or 0.0),
                        str(m.market_id),
                    ),
                )[:max_legs_per_event]

        selected.sort(key=lambda m: str(m.market_id))
        return selected

    @staticmethod
    def _build_candidate(
        *,
        rows: list[MarketView],
        direction: str,
        prob_sum_tolerance: float,
        max_abs_deviation: float,
        max_leg_weight: float,
        min_unique_canonicals: int,
        raw_leg_count: int,
    ) -> tuple[_ConversionCandidate | None, str | None]:
        if len(rows) < min_unique_canonicals:
            return None, "under_min_unique_canonicals"

        sum_yes = sum(float(r.yes_price or 0.0) for r in rows)
        if sum_yes <= 1e-9:
            return None, "non_positive_sum_yes"

        deviation = sum_yes - 1.0
        if abs(deviation) > max_abs_deviation:
            return None, "deviation_above_max_abs"

        if direction == "buy_conversion":
            if deviation >= -prob_sum_tolerance:
                return None, "deviation_not_below_tolerance"
            gross_edge = 1.0 - sum_yes
            selection_mode = "min_yes_per_canonical"
        else:
            if deviation <= prob_sum_tolerance:
                return None, "deviation_not_above_tolerance"
            gross_edge = sum_yes - 1.0
            selection_mode = "max_yes_per_canonical"

        if gross_edge <= 0:
            return None, "non_positive_gross_edge"

        max_leg_weight_ratio = max(float(r.yes_price or 0.0) for r in rows) / max(sum_yes, 1e-9)
        if max_leg_weight_ratio > max_leg_weight:
            return None, "max_leg_weight_exceeded"

        return (
            _ConversionCandidate(
                direction=direction,
                rows=rows,
                sum_yes=sum_yes,
                gross_edge=gross_edge,
                deviation=deviation,
                max_leg_weight=max_leg_weight_ratio,
                raw_leg_count=raw_leg_count,
                selection_mode=selection_mode,
            ),
            None,
        )

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
        candidate: _ConversionCandidate,
        quote_improve: float,
        min_effective_edge_bps: float,
        max_order_notional: float,
        fee_bps: float,
        slippage_bps: float,
        depth_reference_liquidity: float,
        depth_penalty_max_bps: float,
        max_weighted_total_cost_bps: float,
        max_leg_total_cost_bps: float,
        liquidity_fraction: float,
        min_qty: float,
        executable_price_floor: float,
        max_tiny_price_leg_share: float,
        max_floor_adjusted_leg_share: float,
    ) -> tuple[list[Signal], str | None, _ConversionPricingDiagnostics | None]:
        rows = candidate.rows
        side = "buy" if candidate.direction == "buy_conversion" else "sell"

        pre_quote_sum_yes = candidate.sum_yes
        pre_quote_gross_edge = self._gross_edge_from_sum_yes(
            direction=candidate.direction,
            sum_yes=pre_quote_sum_yes,
        )
        pre_quote_gross_edge_bps = self._gross_edge_bps_from_sum_yes(
            direction=candidate.direction,
            sum_yes=pre_quote_sum_yes,
        )

        target_prices: dict[str, float] = {}
        estimated_fill_prices: dict[str, float] = {}
        floor_adjusted_legs = 0
        tiny_price_legs = 0

        for row in rows:
            yes_px = float(row.yes_price or 0.0)
            if yes_px <= 0:
                return [], "non_positive_leg_price", None

            target_raw = yes_px + quote_improve if side == "buy" else yes_px - quote_improve
            target = self._clamp_target_price(target_raw, price_floor=executable_price_floor)
            estimated_fill_price = self._expected_fill_price(
                side=side,
                target_price=target,
                slippage_bps=slippage_bps,
                price_floor=executable_price_floor,
            )

            if target_raw < executable_price_floor:
                tiny_price_legs += 1
            if abs(target - target_raw) > 1e-12:
                floor_adjusted_legs += 1

            target_prices[row.market_id] = target
            estimated_fill_prices[row.market_id] = estimated_fill_price

        post_quote_sum_yes = sum(target_prices.values())
        post_slippage_sum_yes = sum(estimated_fill_prices.values())

        post_quote_gross_edge = self._gross_edge_from_sum_yes(
            direction=candidate.direction,
            sum_yes=post_quote_sum_yes,
        )
        post_slippage_gross_edge = self._gross_edge_from_sum_yes(
            direction=candidate.direction,
            sum_yes=post_slippage_sum_yes,
        )

        post_quote_gross_edge_bps = self._gross_edge_bps_from_sum_yes(
            direction=candidate.direction,
            sum_yes=post_quote_sum_yes,
        )
        post_slippage_gross_edge_bps = self._gross_edge_bps_from_sum_yes(
            direction=candidate.direction,
            sum_yes=post_slippage_sum_yes,
        )

        weighted_cost_numerator = 0.0
        weighted_non_slippage_cost_numerator = 0.0
        depth_penalties: dict[str, float] = {}
        leg_cost_bps: dict[str, float] = {}

        for row in rows:
            target_px = target_prices.get(row.market_id, 0.0)
            cost = self._leg_cost(
                liquidity=row.liquidity,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                depth_reference_liquidity=depth_reference_liquidity,
                depth_penalty_max_bps=depth_penalty_max_bps,
            )
            weighted_cost_numerator += target_px * cost.total_bps

            non_slippage_leg_cost = max(0.0, fee_bps) + cost.depth_penalty_bps
            weighted_non_slippage_cost_numerator += target_px * non_slippage_leg_cost

            depth_penalties[row.market_id] = cost.depth_penalty_bps
            leg_cost_bps[row.market_id] = cost.total_bps

        weighted_total_cost_bps = weighted_cost_numerator / max(post_quote_sum_yes, 1e-9)
        weighted_non_slippage_cost_bps = weighted_non_slippage_cost_numerator / max(
            post_quote_sum_yes, 1e-9
        )

        post_quote_effective_edge_bps = post_quote_gross_edge_bps - weighted_total_cost_bps
        post_slippage_effective_edge_bps = (
            post_slippage_gross_edge_bps - weighted_non_slippage_cost_bps
        )

        pricing_diag = _ConversionPricingDiagnostics(
            pre_quote_sum_yes=pre_quote_sum_yes,
            post_quote_sum_yes=post_quote_sum_yes,
            post_slippage_sum_yes=post_slippage_sum_yes,
            pre_quote_gross_edge_bps=pre_quote_gross_edge_bps,
            post_quote_gross_edge_bps=post_quote_gross_edge_bps,
            post_slippage_gross_edge_bps=post_slippage_gross_edge_bps,
            post_quote_effective_edge_bps=post_quote_effective_edge_bps,
            post_slippage_effective_edge_bps=post_slippage_effective_edge_bps,
            floor_adjusted_legs=floor_adjusted_legs,
            tiny_price_legs=tiny_price_legs,
        )

        total_legs = len(rows)
        tiny_price_leg_share = tiny_price_legs / max(1, total_legs)
        floor_adjusted_leg_share = floor_adjusted_legs / max(1, total_legs)

        if tiny_price_leg_share > max_tiny_price_leg_share:
            return [], "tiny_price_leg_share_exceeded", pricing_diag

        if floor_adjusted_leg_share > max_floor_adjusted_leg_share:
            return [], "floor_adjusted_leg_share_exceeded", pricing_diag

        if pre_quote_gross_edge <= 0:
            return [], "non_positive_gross_edge", pricing_diag

        if post_quote_gross_edge <= 0:
            return [], "post_quote_non_positive_gross_edge", pricing_diag

        if post_slippage_gross_edge <= 0:
            return [], "post_slippage_non_positive_gross_edge", pricing_diag

        if (
            max_weighted_total_cost_bps > 0
            and weighted_total_cost_bps > max_weighted_total_cost_bps
        ):
            return [], "weighted_cost_cap_exceeded", pricing_diag

        if max_leg_total_cost_bps > 0 and any(
            leg_cost > max_leg_total_cost_bps for leg_cost in leg_cost_bps.values()
        ):
            return [], "leg_cost_cap_exceeded", pricing_diag

        effective_edge_bps = post_slippage_effective_edge_bps
        if effective_edge_bps < min_effective_edge_bps:
            return [], "effective_edge_below_min", pricing_diag

        min_leg_liq = min(float(r.liquidity) for r in rows)
        qty = max(min_qty, min_leg_liq * liquidity_fraction)
        if max_order_notional > 0:
            qty = min(qty, max_order_notional / max(post_quote_sum_yes, 1e-9))
        if qty <= 1e-12:
            return [], "qty_non_positive", pricing_diag

        basket_notional = qty * post_quote_sum_yes
        confidence = min(0.99, 0.50 + max(0.0, effective_edge_bps) / 700.0)
        score = (effective_edge_bps / 10000.0) * (1.0 + min_leg_liq / depth_reference_liquidity)
        sorted_markets = sorted(str(r.market_id) for r in rows)
        basket_id = f"{event_id}:{candidate.direction}:{'-'.join(sorted_markets)}"

        if side == "buy":
            rationale = (
                f"NegRisk转换套利({candidate.direction}): post_quote_sum_yes={post_quote_sum_yes:.4f}<1, "
                f"gross_exec={post_slippage_gross_edge_bps:.1f}bps, "
                f"eff_exec={effective_edge_bps:.1f}bps"
            )
        else:
            rationale = (
                f"NegRisk转换套利({candidate.direction}): post_quote_sum_yes={post_quote_sum_yes:.4f}>1, "
                f"gross_exec={post_slippage_gross_edge_bps:.1f}bps, "
                f"eff_exec={effective_edge_bps:.1f}bps"
            )

        basket_markets = [
            {
                "market_id": r.market_id,
                "canonical_id": r.canonical_id,
                "yes_price": float(r.yes_price or 0.0),
                "target_price": target_prices.get(r.market_id, 0.0),
                "estimated_fill_price": estimated_fill_prices.get(r.market_id, 0.0),
                "liquidity": float(r.liquidity),
            }
            for r in rows
        ]

        out: list[Signal] = []
        total_legs = len(rows)
        for idx, row in enumerate(rows):
            target = target_prices.get(row.market_id, 0.0)
            if target <= 0:
                continue

            payload = {
                "signal_source": "negrisk_conversion_arb",
                "event_id": event_id,
                "basket_id": basket_id,
                "basket_atomic": True,
                "basket_batch_id": basket_id,
                "basket_expected_legs": total_legs,
                "direction": candidate.direction,
                "selection_mode": candidate.selection_mode,
                "sum_yes": post_quote_sum_yes,
                "deviation": candidate.deviation,
                "gross_edge": post_quote_gross_edge,
                "gross_edge_bps": post_quote_gross_edge_bps,
                "effective_edge_bps": effective_edge_bps,
                "pre_quote_sum_yes": pre_quote_sum_yes,
                "post_quote_sum_yes": post_quote_sum_yes,
                "post_slippage_sum_yes": post_slippage_sum_yes,
                "pre_quote_gross_edge_bps": pre_quote_gross_edge_bps,
                "post_quote_gross_edge_bps": post_quote_gross_edge_bps,
                "post_slippage_gross_edge_bps": post_slippage_gross_edge_bps,
                "post_quote_effective_edge_bps": post_quote_effective_edge_bps,
                "post_slippage_effective_edge_bps": post_slippage_effective_edge_bps,
                "price_floor": executable_price_floor,
                "floor_adjusted_legs": floor_adjusted_legs,
                "floor_adjusted_leg_share": floor_adjusted_leg_share,
                "max_floor_adjusted_leg_share": max_floor_adjusted_leg_share,
                "tiny_price_legs": tiny_price_legs,
                "tiny_price_leg_share": tiny_price_leg_share,
                "max_tiny_price_leg_share": max_tiny_price_leg_share,
                "estimated_fill_price": estimated_fill_prices.get(row.market_id, target),
                "basket_qty": qty,
                "basket_notional": basket_notional,
                "convert_value": 1.0,
                "raw_leg_count": candidate.raw_leg_count,
                "unique_canonical_count": total_legs,
                "max_leg_weight": candidate.max_leg_weight,
                "legs_count": total_legs,
                "leg_index": idx,
                "basket_markets": basket_markets,
                "cost_model": {
                    "fee_bps_per_leg": fee_bps,
                    "slippage_bps_per_leg": slippage_bps,
                    "depth_reference_liquidity": depth_reference_liquidity,
                    "depth_penalty_max_bps_per_leg": depth_penalty_max_bps,
                    "weighted_total_cost_bps": weighted_total_cost_bps,
                    "weighted_non_slippage_cost_bps": weighted_non_slippage_cost_bps,
                    "max_weighted_total_cost_bps": max_weighted_total_cost_bps,
                    "leg_total_cost_bps": leg_cost_bps.get(row.market_id, 0.0),
                    "max_leg_total_cost_bps": max_leg_total_cost_bps,
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

        if not out:
            return [], "non_positive_leg_price", pricing_diag

        return out, None, pricing_diag
