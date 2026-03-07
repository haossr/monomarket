from __future__ import annotations

import hashlib
import math
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


@dataclass(slots=True)
class _PairPricingDiagnostics:
    pre_floor_parity_sum: float
    post_floor_parity_sum: float
    post_slippage_parity_sum: float
    pre_floor_gross_edge_bps: float
    post_floor_gross_edge_bps: float
    post_slippage_gross_edge_bps: float
    post_floor_effective_edge_bps: float
    post_slippage_effective_edge_bps: float
    floor_adjusted_legs: int
    tiny_price_legs: int


@dataclass(slots=True)
class _PairSearchDiagnostics:
    candidate_pairs_scanned: int = 0
    market_guard_pass: int = 0
    event_guard_pass: int = 0
    condition_guard_pass: int = 0
    rejected_by_market_guard: int = 0
    rejected_by_event_guard: int = 0
    rejected_by_condition_guard: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "candidate_pairs_scanned": int(self.candidate_pairs_scanned),
            "market_guard_pass": int(self.market_guard_pass),
            "event_guard_pass": int(self.event_guard_pass),
            "condition_guard_pass": int(self.condition_guard_pass),
            "rejected_by_market_guard": int(self.rejected_by_market_guard),
            "rejected_by_event_guard": int(self.rejected_by_event_guard),
            "rejected_by_condition_guard": int(self.rejected_by_condition_guard),
        }


def _merge_pair_search_diag(target: _PairSearchDiagnostics, source: _PairSearchDiagnostics) -> None:
    target.candidate_pairs_scanned += int(source.candidate_pairs_scanned)
    target.market_guard_pass += int(source.market_guard_pass)
    target.event_guard_pass += int(source.event_guard_pass)
    target.condition_guard_pass += int(source.condition_guard_pass)
    target.rejected_by_market_guard += int(source.rejected_by_market_guard)
    target.rejected_by_event_guard += int(source.rejected_by_event_guard)
    target.rejected_by_condition_guard += int(source.rejected_by_condition_guard)


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


def _normalize_text(raw: str) -> str:
    return " ".join(str(raw).strip().lower().split())


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


class S9YesNoParityArb(Strategy):
    """
    S9: same-condition YES/NO parity arbitrage.

    Signal construction:
      - buy carry: buy YES + NO parity leg when yes+no < 1
      - sell overround: sell YES + NO parity leg when yes+no > 1

    Default leg matching is same-market (`require_same_market=true`),
    with optional cross-market pairing for experiments.
    Effective edge is local and depth/fee/slippage aware.
    """

    name = "s9"

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

    @staticmethod
    def _edge_from_parity(*, side: str, parity_sum: float) -> float:
        if side == "buy":
            return 1.0 - parity_sum
        return parity_sum - 1.0

    @classmethod
    def _edge_bps_from_parity(cls, *, side: str, parity_sum: float) -> float:
        gross_edge = cls._edge_from_parity(side=side, parity_sum=parity_sum)
        return (gross_edge / max(1e-9, parity_sum)) * 10000.0

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
        max_total_cost_bps = max(0.0, float(cfg.get("max_total_cost_bps", 140.0)))
        liquidity_fraction = max(0.0, float(cfg.get("liquidity_fraction", 0.01)))
        min_qty = max(0.1, float(cfg.get("min_qty", 1.0)))
        max_signals = max(0, int(float(cfg.get("max_signals", 120))))
        quote_improve = max(0.0, float(cfg.get("quote_improve", 0.0)))
        executable_price_floor = max(0.01, float(cfg.get("executable_price_floor", 0.01)))
        allow_sell_parity = _as_bool(cfg.get("allow_sell_parity"), False)
        require_same_event = _as_bool(cfg.get("require_same_event"), True)
        require_same_condition = _as_bool(cfg.get("require_same_condition"), True)
        require_same_market = _as_bool(cfg.get("require_same_market"), True)
        max_pairs_per_event = max(0, int(float(cfg.get("max_pairs_per_event", 2))))
        max_event_pair_notional = float(cfg.get("max_event_pair_notional", 20.0))
        diagnostics_event_top_k = max(0, int(float(cfg.get("diagnostics_event_top_k", 20))))

        if max_order_notional <= 0 or max_signals <= 0:
            self.last_diagnostics = {
                "canonical_groups_total": 0,
                "events_total": 0,
                "pair_attempts": 0,
                "pairs_emitted": 0,
                "signals_emitted": 0,
                "direction_attempts": {"buy": 0, "sell": 0},
                "direction_pass": {"buy": 0, "sell": 0},
                "candidate_reject_reasons": {"strategy_disabled": 1},
                "candidate_reject_reasons_by_event": {},
                "candidate_reject_reasons_by_event_top_k": diagnostics_event_top_k,
                "candidate_reject_reasons_by_event_top": {},
                "pair_search": {
                    "candidate_pairs_scanned": 0,
                    "market_guard_pass": 0,
                    "event_guard_pass": 0,
                    "condition_guard_pass": 0,
                    "rejected_by_market_guard": 0,
                    "rejected_by_event_guard": 0,
                    "rejected_by_condition_guard": 0,
                    "by_side": {
                        "buy": {
                            "candidate_pairs_scanned": 0,
                            "market_guard_pass": 0,
                            "event_guard_pass": 0,
                            "condition_guard_pass": 0,
                            "rejected_by_market_guard": 0,
                            "rejected_by_event_guard": 0,
                            "rejected_by_condition_guard": 0,
                        },
                        "sell": {
                            "candidate_pairs_scanned": 0,
                            "market_guard_pass": 0,
                            "event_guard_pass": 0,
                            "condition_guard_pass": 0,
                            "rejected_by_market_guard": 0,
                            "rejected_by_event_guard": 0,
                            "rejected_by_condition_guard": 0,
                        },
                    },
                },
                "pricing_consistency": {
                    "price_floor": executable_price_floor,
                    "pair_candidates_priced": 0,
                    "pairs_with_floor_adjustment": 0,
                    "tiny_price_legs": 0,
                    "tiny_price_pairs": 0,
                    "avg_pre_floor_parity_sum": 0.0,
                    "avg_post_floor_parity_sum": 0.0,
                    "avg_post_slippage_parity_sum": 0.0,
                    "avg_pre_floor_gross_edge_bps": 0.0,
                    "avg_post_floor_gross_edge_bps": 0.0,
                    "avg_post_slippage_gross_edge_bps": 0.0,
                    "avg_post_floor_effective_edge_bps": 0.0,
                    "avg_post_slippage_effective_edge_bps": 0.0,
                    "filtered_post_floor_non_positive": 0,
                    "filtered_post_slippage_non_positive": 0,
                    "filtered_post_slippage_effective_edge_below_min": 0,
                },
            }
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

        event_pair_counts: dict[str, int] = defaultdict(int)
        event_pair_notional: dict[str, float] = defaultdict(float)
        signals: list[Signal] = []
        reject_reasons: dict[str, int] = {}
        reject_reasons_by_event: dict[str, dict[str, int]] = {}
        direction_attempts: dict[str, int] = {"buy": 0, "sell": 0}
        direction_pass: dict[str, int] = {"buy": 0, "sell": 0}
        pairs_emitted = 0

        pricing_diag_count = 0
        pricing_diag_sum_pre_floor_parity = 0.0
        pricing_diag_sum_post_floor_parity = 0.0
        pricing_diag_sum_post_slippage_parity = 0.0
        pricing_diag_sum_pre_floor_gross_bps = 0.0
        pricing_diag_sum_post_floor_gross_bps = 0.0
        pricing_diag_sum_post_slippage_gross_bps = 0.0
        pricing_diag_sum_post_floor_effective_bps = 0.0
        pricing_diag_sum_post_slippage_effective_bps = 0.0
        pricing_diag_floor_adjusted_pairs = 0
        pricing_diag_tiny_price_legs = 0
        pricing_diag_tiny_price_pairs = 0
        pricing_diag_filtered_post_floor_non_positive = 0
        pricing_diag_filtered_post_slippage_non_positive = 0
        pricing_diag_filtered_effective_below_min = 0

        pair_search_by_side: dict[str, _PairSearchDiagnostics] = {
            "buy": _PairSearchDiagnostics(),
            "sell": _PairSearchDiagnostics(),
        }

        for canonical_id in sorted(by_canonical):
            rows = by_canonical[canonical_id]
            event_hint = self._event_hint_for_rows(rows)
            if len(rows) < 2:
                _record_reject_reason(
                    reject_reasons=reject_reasons,
                    reject_reasons_by_event=reject_reasons_by_event,
                    event_id=event_hint,
                    reason="canonical_under_two_legs",
                )
                continue

            candidate_sides: list[tuple[str, bool]] = [("buy", True)]
            if allow_sell_parity:
                candidate_sides.append(("sell", False))

            for side, buy_mode in candidate_sides:
                direction_attempts[side] = int(direction_attempts.get(side, 0)) + 1
                pair, pair_search_diag = self._best_pair(
                    rows,
                    buy_mode=buy_mode,
                    require_same_event=require_same_event,
                    require_same_condition=require_same_condition,
                    require_same_market=require_same_market,
                )
                side_pair_search_diag = pair_search_by_side.setdefault(
                    side, _PairSearchDiagnostics()
                )
                _merge_pair_search_diag(side_pair_search_diag, pair_search_diag)

                if pair is None:
                    pair_not_found_reason = self._pair_not_found_reason(
                        search_diag=pair_search_diag,
                        require_same_event=require_same_event,
                        require_same_condition=require_same_condition,
                    )
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_hint,
                        reason=f"{side}:{pair_not_found_reason}",
                    )
                    continue

                yes_leg, no_leg = pair
                pair_event_hint = self._event_hint_for_pair(yes_leg=yes_leg, no_leg=no_leg)
                event_id = self._resolve_pair_event_id(
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    require_same_event=require_same_event,
                )
                if event_id is None:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=pair_event_hint,
                        reason=f"{side}:pair_event_mismatch",
                    )
                    continue

                condition_key = self._resolve_pair_condition_key(
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    require_same_condition=require_same_condition,
                )
                if condition_key is None:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_id,
                        reason=f"{side}:pair_condition_mismatch",
                    )
                    continue

                if max_pairs_per_event > 0 and event_pair_counts[event_id] >= max_pairs_per_event:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_id,
                        reason=f"{side}:event_pair_limit_reached",
                    )
                    continue

                used_before = float(event_pair_notional[event_id])
                event_notional_budget: float | None = None
                if max_event_pair_notional > 0:
                    event_notional_budget = max_event_pair_notional - used_before
                    if event_notional_budget <= 0:
                        _record_reject_reason(
                            reject_reasons=reject_reasons,
                            reject_reasons_by_event=reject_reasons_by_event,
                            event_id=event_id,
                            reason=f"{side}:event_notional_budget_exhausted",
                        )
                        continue

                pair_signals, pair_reject_reason, pair_pricing_diag = self._emit_pair_signals(
                    canonical_id=canonical_id,
                    event_id=event_id,
                    condition_key=condition_key,
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    side=side,
                    quote_improve=quote_improve,
                    min_effective_edge_bps=min_effective_edge_bps,
                    max_order_notional=max_order_notional,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    depth_reference_liquidity=depth_reference_liquidity,
                    depth_penalty_max_bps=depth_penalty_max_bps,
                    max_total_cost_bps=max_total_cost_bps,
                    liquidity_fraction=liquidity_fraction,
                    min_qty=min_qty,
                    event_notional_budget=event_notional_budget,
                    event_notional_used_before=used_before,
                    event_pair_index=event_pair_counts[event_id] + 1,
                    max_pairs_per_event=max_pairs_per_event,
                    executable_price_floor=executable_price_floor,
                )

                if pair_pricing_diag is not None:
                    pricing_diag_count += 1
                    pricing_diag_sum_pre_floor_parity += pair_pricing_diag.pre_floor_parity_sum
                    pricing_diag_sum_post_floor_parity += pair_pricing_diag.post_floor_parity_sum
                    pricing_diag_sum_post_slippage_parity += (
                        pair_pricing_diag.post_slippage_parity_sum
                    )
                    pricing_diag_sum_pre_floor_gross_bps += (
                        pair_pricing_diag.pre_floor_gross_edge_bps
                    )
                    pricing_diag_sum_post_floor_gross_bps += (
                        pair_pricing_diag.post_floor_gross_edge_bps
                    )
                    pricing_diag_sum_post_slippage_gross_bps += (
                        pair_pricing_diag.post_slippage_gross_edge_bps
                    )
                    pricing_diag_sum_post_floor_effective_bps += (
                        pair_pricing_diag.post_floor_effective_edge_bps
                    )
                    pricing_diag_sum_post_slippage_effective_bps += (
                        pair_pricing_diag.post_slippage_effective_edge_bps
                    )
                    pricing_diag_tiny_price_legs += int(pair_pricing_diag.tiny_price_legs)
                    if pair_pricing_diag.floor_adjusted_legs > 0:
                        pricing_diag_floor_adjusted_pairs += 1
                    if pair_pricing_diag.tiny_price_legs > 0:
                        pricing_diag_tiny_price_pairs += 1

                if pair_reject_reason == "post_floor_non_positive_gross_edge":
                    pricing_diag_filtered_post_floor_non_positive += 1
                elif pair_reject_reason == "post_slippage_non_positive_gross_edge":
                    pricing_diag_filtered_post_slippage_non_positive += 1
                elif pair_reject_reason == "effective_edge_below_min":
                    pricing_diag_filtered_effective_below_min += 1

                if not pair_signals:
                    _record_reject_reason(
                        reject_reasons=reject_reasons,
                        reject_reasons_by_event=reject_reasons_by_event,
                        event_id=event_id,
                        reason=(
                            f"{side}:{pair_reject_reason}"
                            if pair_reject_reason
                            else f"{side}:emit_rejected_unknown"
                        ),
                    )
                    continue

                pair_notional = float(pair_signals[0].payload.get("pair_notional", 0.0))
                event_pair_notional[event_id] = used_before + max(0.0, pair_notional)
                event_pair_counts[event_id] += 1
                direction_pass[side] = int(direction_pass.get(side, 0)) + 1
                pairs_emitted += 1
                signals.extend(pair_signals)

                if len(signals) >= max_signals:
                    break

            if len(signals) >= max_signals:
                break

        signals.sort(key=lambda s: s.score, reverse=True)
        selected_signals = signals[:max_signals]
        reject_by_event = _normalize_reject_reasons_by_event(reject_reasons_by_event)
        events_total = len(
            {
                str(row.event_id).strip()
                for grouped_rows in by_canonical.values()
                for row in grouped_rows
                if str(row.event_id).strip()
            }
        )
        pair_search_total = _PairSearchDiagnostics()
        for side_diag in pair_search_by_side.values():
            _merge_pair_search_diag(pair_search_total, side_diag)

        self.last_diagnostics = {
            "canonical_groups_total": len(by_canonical),
            "events_total": events_total,
            "pair_attempts": sum(int(v) for v in direction_attempts.values()),
            "pairs_emitted": pairs_emitted,
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
            "pair_search": {
                **pair_search_total.as_dict(),
                "by_side": {
                    side: diag.as_dict() for side, diag in sorted(pair_search_by_side.items())
                },
            },
            "pricing_consistency": {
                "price_floor": executable_price_floor,
                "pair_candidates_priced": pricing_diag_count,
                "pairs_with_floor_adjustment": pricing_diag_floor_adjusted_pairs,
                "tiny_price_legs": pricing_diag_tiny_price_legs,
                "tiny_price_pairs": pricing_diag_tiny_price_pairs,
                "avg_pre_floor_parity_sum": self._safe_avg(
                    pricing_diag_sum_pre_floor_parity,
                    pricing_diag_count,
                ),
                "avg_post_floor_parity_sum": self._safe_avg(
                    pricing_diag_sum_post_floor_parity,
                    pricing_diag_count,
                ),
                "avg_post_slippage_parity_sum": self._safe_avg(
                    pricing_diag_sum_post_slippage_parity,
                    pricing_diag_count,
                ),
                "avg_pre_floor_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_pre_floor_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_floor_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_floor_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_slippage_gross_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_slippage_gross_bps,
                    pricing_diag_count,
                ),
                "avg_post_floor_effective_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_floor_effective_bps,
                    pricing_diag_count,
                ),
                "avg_post_slippage_effective_edge_bps": self._safe_avg(
                    pricing_diag_sum_post_slippage_effective_bps,
                    pricing_diag_count,
                ),
                "filtered_post_floor_non_positive": pricing_diag_filtered_post_floor_non_positive,
                "filtered_post_slippage_non_positive": pricing_diag_filtered_post_slippage_non_positive,
                "filtered_post_slippage_effective_edge_below_min": (
                    pricing_diag_filtered_effective_below_min
                ),
            },
        }
        return selected_signals

    @staticmethod
    def _event_hint_for_rows(rows: list[MarketView]) -> str:
        event_ids = {
            str(row.event_id or "").strip() for row in rows if str(row.event_id or "").strip()
        }
        if not event_ids:
            return "unknown"
        if len(event_ids) == 1:
            return next(iter(event_ids))
        return "mixed"

    @staticmethod
    def _event_hint_for_pair(*, yes_leg: MarketView, no_leg: MarketView) -> str:
        yes_event = str(yes_leg.event_id or "").strip()
        no_event = str(no_leg.event_id or "").strip()
        if yes_event and no_event and yes_event == no_event:
            return yes_event
        if yes_event or no_event:
            return "mixed"
        return "unknown"

    @staticmethod
    def _resolve_pair_event_id(
        *,
        yes_leg: MarketView,
        no_leg: MarketView,
        require_same_event: bool,
    ) -> str | None:
        yes_event = str(yes_leg.event_id or "").strip()
        no_event = str(no_leg.event_id or "").strip()
        if yes_event and no_event and yes_event == no_event:
            return yes_event
        if require_same_event:
            return None
        return yes_event or no_event or None

    @classmethod
    def _condition_key_for_market(cls, market: MarketView) -> str:
        question = _normalize_text(str(market.question or ""))
        if question:
            return question
        return _normalize_text(str(market.canonical_id or ""))

    @classmethod
    def _resolve_pair_condition_key(
        cls,
        *,
        yes_leg: MarketView,
        no_leg: MarketView,
        require_same_condition: bool,
    ) -> str | None:
        yes_key = cls._condition_key_for_market(yes_leg)
        no_key = cls._condition_key_for_market(no_leg)
        if yes_key and no_key and yes_key == no_key:
            return yes_key
        if require_same_condition:
            return None
        return yes_key or no_key or None

    @staticmethod
    def _pair_not_found_reason(
        *,
        search_diag: _PairSearchDiagnostics,
        require_same_event: bool,
        require_same_condition: bool,
    ) -> str:
        if search_diag.candidate_pairs_scanned <= 0:
            return "pair_not_found_no_candidates"
        if search_diag.market_guard_pass <= 0 and search_diag.rejected_by_market_guard > 0:
            return "pair_not_found_market_guard"
        if (
            require_same_event
            and search_diag.event_guard_pass <= 0
            and search_diag.rejected_by_event_guard > 0
        ):
            return "pair_not_found_event_guard"
        if (
            require_same_condition
            and search_diag.condition_guard_pass <= 0
            and search_diag.rejected_by_condition_guard > 0
        ):
            return "pair_not_found_condition_guard"
        return "pair_not_found"

    @classmethod
    def _best_pair(
        cls,
        rows: list[MarketView],
        *,
        buy_mode: bool,
        require_same_event: bool,
        require_same_condition: bool,
        require_same_market: bool,
    ) -> tuple[tuple[MarketView, MarketView] | None, _PairSearchDiagnostics]:
        stats = _PairSearchDiagnostics()

        if buy_mode:
            yes_sorted = sorted(rows, key=lambda m: float(m.yes_price or 1.0))
            no_sorted = sorted(rows, key=lambda m: float(m.no_price or 1.0))
        else:
            yes_sorted = sorted(rows, key=lambda m: float(m.yes_price or 0.0), reverse=True)
            no_sorted = sorted(rows, key=lambda m: float(m.no_price or 0.0), reverse=True)

        if not yes_sorted or not no_sorted:
            return None, stats

        for yes_leg in yes_sorted:
            for no_leg in no_sorted:
                stats.candidate_pairs_scanned += 1

                if require_same_market:
                    if yes_leg.market_id != no_leg.market_id:
                        stats.rejected_by_market_guard += 1
                        continue
                elif yes_leg.market_id == no_leg.market_id:
                    stats.rejected_by_market_guard += 1
                    continue
                stats.market_guard_pass += 1

                if require_same_event and str(yes_leg.event_id) != str(no_leg.event_id):
                    stats.rejected_by_event_guard += 1
                    continue
                stats.event_guard_pass += 1

                if require_same_condition and (
                    cls._condition_key_for_market(yes_leg) != cls._condition_key_for_market(no_leg)
                ):
                    stats.rejected_by_condition_guard += 1
                    continue
                stats.condition_guard_pass += 1

                return (yes_leg, no_leg), stats

        return None, stats

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
        event_id: str,
        condition_key: str,
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
        max_total_cost_bps: float,
        liquidity_fraction: float,
        min_qty: float,
        event_notional_budget: float | None,
        event_notional_used_before: float,
        event_pair_index: int,
        max_pairs_per_event: int,
        executable_price_floor: float,
    ) -> tuple[list[Signal], str | None, _PairPricingDiagnostics | None]:
        yes_px = float(yes_leg.yes_price or 0.0)
        no_px = float(no_leg.no_price or 0.0)
        if yes_px <= 0 or no_px <= 0:
            return [], "non_positive_leg_price", None

        if side == "buy":
            direction = "buy_carry"
            yes_target_raw = yes_px + quote_improve
            no_target_raw = no_px + quote_improve
        else:
            direction = "sell_overround"
            yes_target_raw = yes_px - quote_improve
            no_target_raw = no_px - quote_improve

        pre_floor_parity_sum = yes_target_raw + no_target_raw
        pre_floor_gross_edge = self._edge_from_parity(side=side, parity_sum=pre_floor_parity_sum)

        yes_target = self._clamp_target_price(yes_target_raw, price_floor=executable_price_floor)
        no_target = self._clamp_target_price(no_target_raw, price_floor=executable_price_floor)

        tiny_price_legs = int(yes_target_raw < executable_price_floor) + int(
            no_target_raw < executable_price_floor
        )
        floor_adjusted_legs = int(abs(yes_target - yes_target_raw) > 1e-12) + int(
            abs(no_target - no_target_raw) > 1e-12
        )

        post_floor_parity_sum = yes_target + no_target
        post_floor_gross_edge = self._edge_from_parity(side=side, parity_sum=post_floor_parity_sum)

        yes_fill_est = self._expected_fill_price(
            side=side,
            target_price=yes_target,
            slippage_bps=slippage_bps,
            price_floor=executable_price_floor,
        )
        no_fill_est = self._expected_fill_price(
            side=side,
            target_price=no_target,
            slippage_bps=slippage_bps,
            price_floor=executable_price_floor,
        )
        post_slippage_parity_sum = yes_fill_est + no_fill_est
        post_slippage_gross_edge = self._edge_from_parity(
            side=side,
            parity_sum=post_slippage_parity_sum,
        )

        pre_floor_gross_edge_bps = self._edge_bps_from_parity(
            side=side, parity_sum=pre_floor_parity_sum
        )
        post_floor_gross_edge_bps = self._edge_bps_from_parity(
            side=side,
            parity_sum=post_floor_parity_sum,
        )
        post_slippage_gross_edge_bps = self._edge_bps_from_parity(
            side=side,
            parity_sum=post_slippage_parity_sum,
        )

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
        non_slippage_cost_bps = (
            max(0.0, fee_bps)
            + yes_cost.depth_penalty_bps
            + max(0.0, fee_bps)
            + no_cost.depth_penalty_bps
        )
        post_floor_effective_edge_bps = post_floor_gross_edge_bps - total_cost_bps
        effective_edge_bps = post_slippage_gross_edge_bps - non_slippage_cost_bps

        pricing_diag = _PairPricingDiagnostics(
            pre_floor_parity_sum=pre_floor_parity_sum,
            post_floor_parity_sum=post_floor_parity_sum,
            post_slippage_parity_sum=post_slippage_parity_sum,
            pre_floor_gross_edge_bps=pre_floor_gross_edge_bps,
            post_floor_gross_edge_bps=post_floor_gross_edge_bps,
            post_slippage_gross_edge_bps=post_slippage_gross_edge_bps,
            post_floor_effective_edge_bps=post_floor_effective_edge_bps,
            post_slippage_effective_edge_bps=effective_edge_bps,
            floor_adjusted_legs=floor_adjusted_legs,
            tiny_price_legs=tiny_price_legs,
        )

        if pre_floor_gross_edge <= 0:
            return [], "non_positive_gross_edge", pricing_diag

        if post_floor_gross_edge <= 0:
            return [], "post_floor_non_positive_gross_edge", pricing_diag

        if post_slippage_gross_edge <= 0:
            return [], "post_slippage_non_positive_gross_edge", pricing_diag

        if max_total_cost_bps > 0 and total_cost_bps > max_total_cost_bps:
            return [], "total_cost_cap_exceeded", pricing_diag

        if effective_edge_bps < min_effective_edge_bps:
            return [], "effective_edge_below_min", pricing_diag

        pricing_parity_ref = max(1e-9, post_floor_parity_sum)
        min_leg_liq = min(float(yes_leg.liquidity), float(no_leg.liquidity))
        pair_qty = max(min_qty, min_leg_liq * liquidity_fraction)
        if max_order_notional > 0:
            pair_qty = min(pair_qty, max_order_notional / pricing_parity_ref)

        if event_notional_budget is not None:
            budget = max(0.0, float(event_notional_budget))
            if budget <= 1e-12:
                return [], "event_notional_budget_non_positive", pricing_diag
            pair_qty = min(pair_qty, budget / pricing_parity_ref)

        if pair_qty <= 1e-12:
            return [], "qty_non_positive", pricing_diag

        pair_notional = pair_qty * post_floor_parity_sum
        if pair_notional <= 1e-12:
            return [], "pair_notional_non_positive", pricing_diag

        confidence = min(0.98, 0.50 + max(0.0, effective_edge_bps) / 600.0)
        score = (effective_edge_bps / 10000.0) * (1.0 + min_leg_liq / depth_reference_liquidity)

        condition_fingerprint = hashlib.sha1(condition_key.encode("utf-8")).hexdigest()[:12]
        pair_id = (
            f"{canonical_id}:{event_id}:{condition_fingerprint}:{direction}:"
            f"{yes_leg.market_id}:{no_leg.market_id}:{yes_leg.source}:{no_leg.source}"
        )
        pair_batch_id = f"{pair_id}:{event_pair_index}"

        if side == "buy":
            rationale = (
                f"同条件平价套利({direction}): post_floor_yes+no={post_floor_parity_sum:.4f}<1, "
                f"gross_exec={post_slippage_gross_edge_bps:.1f}bps, "
                f"eff_exec={effective_edge_bps:.1f}bps"
            )
        else:
            rationale = (
                f"同条件平价套利({direction}): post_floor_yes+no={post_floor_parity_sum:.4f}>1, "
                f"gross_exec={post_slippage_gross_edge_bps:.1f}bps, "
                f"eff_exec={effective_edge_bps:.1f}bps"
            )

        event_budget_payload: float | None = None
        if event_notional_budget is not None and math.isfinite(event_notional_budget):
            event_budget_payload = max(0.0, float(event_notional_budget))

        common_payload = {
            "signal_source": "yes_no_parity_arb",
            "event_id": event_id,
            "canonical_id": canonical_id,
            "condition_key": condition_key,
            "pair_id": pair_id,
            "pair_batch_id": pair_batch_id,
            "pair_expected_legs": 2,
            "pair_atomic": True,
            "pair_mode": (
                "same_market" if str(yes_leg.market_id) == str(no_leg.market_id) else "cross_market"
            ),
            "pair_same_market": str(yes_leg.market_id) == str(no_leg.market_id),
            "pair_leg_tokens": [OUTCOME_TOKEN_YES, OUTCOME_TOKEN_NO],
            "direction": direction,
            "parity_sum": post_floor_parity_sum,
            "gross_edge": post_floor_gross_edge,
            "gross_edge_bps": post_floor_gross_edge_bps,
            "effective_edge_bps": effective_edge_bps,
            "pre_floor_parity_sum": pre_floor_parity_sum,
            "post_floor_parity_sum": post_floor_parity_sum,
            "post_slippage_parity_sum": post_slippage_parity_sum,
            "pre_floor_gross_edge_bps": pre_floor_gross_edge_bps,
            "post_floor_gross_edge_bps": post_floor_gross_edge_bps,
            "post_slippage_gross_edge_bps": post_slippage_gross_edge_bps,
            "post_floor_effective_edge_bps": post_floor_effective_edge_bps,
            "post_slippage_effective_edge_bps": effective_edge_bps,
            "price_floor": executable_price_floor,
            "floor_adjusted_legs": floor_adjusted_legs,
            "tiny_price_legs": tiny_price_legs,
            "estimated_fill_prices": {
                "yes": yes_fill_est,
                "no": no_fill_est,
            },
            "pair_qty": pair_qty,
            "pair_notional": pair_notional,
            "event_pair_index": event_pair_index,
            "event_notional_budget": event_budget_payload,
            "event_notional_used_before": event_notional_used_before,
            "event_notional_used_after": event_notional_used_before + pair_notional,
            "max_pairs_per_event": max_pairs_per_event,
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
                "non_slippage_cost_bps": non_slippage_cost_bps,
                "max_total_cost_bps": max_total_cost_bps,
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

        return (
            [
                Signal(
                    strategy=self.name,
                    market_id=yes_leg.market_id,
                    event_id=event_id,
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
                    event_id=event_id,
                    side=side,
                    score=score,
                    confidence=confidence,
                    target_price=no_target,
                    size_hint=pair_qty,
                    rationale=rationale,
                    payload=no_payload,
                ),
            ],
            None,
            pricing_diag,
        )
