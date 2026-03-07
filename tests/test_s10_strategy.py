from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s10_negrisk_conversion import S10NegRiskConversionArb


def _market(
    i: int,
    *,
    event_id: str,
    canonical_id: str,
    yes: float,
    liq: float,
    neg_risk: bool = True,
) -> MarketView:
    return MarketView(
        source="gamma" if i % 2 else "clob",
        market_id=f"m{i}",
        canonical_id=canonical_id,
        event_id=event_id,
        question=f"Q{i}",
        status="open",
        neg_risk=neg_risk,
        liquidity=liq,
        volume=100,
        yes_price=yes,
        no_price=1 - yes,
        best_bid=None,
        best_ask=None,
        mid_price=yes,
    )


def test_s10_generates_buy_conversion_basket() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(1, event_id="e1", canonical_id="c1", yes=0.28, liq=900),
        _market(2, event_id="e1", canonical_id="c2", yes=0.31, liq=850),
        _market(3, event_id="e1", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 8.0,
        },
    )

    assert len(signals) == 3
    assert all(s.strategy == "s10" for s in signals)
    assert all(s.side == "buy" for s in signals)
    assert all(str(s.payload.get("direction")) == "buy_conversion" for s in signals)
    assert all(float(s.payload.get("effective_edge_bps", 0.0)) > 0 for s in signals)


def test_s10_emits_basket_atomic_metadata() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(1, event_id="e1", canonical_id="c1", yes=0.28, liq=900),
        _market(2, event_id="e1", canonical_id="c2", yes=0.31, liq=850),
        _market(3, event_id="e1", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 8.0,
        },
    )

    assert len(signals) == 3
    batch_ids = {str(s.payload.get("basket_batch_id", "")) for s in signals}
    assert len(batch_ids) == 1
    assert all(bool(s.payload.get("basket_atomic", False)) for s in signals)
    assert all(int(s.payload.get("basket_expected_legs", 0)) == 3 for s in signals)
    assert {int(s.payload.get("leg_index", -1)) for s in signals} == {0, 1, 2}


def test_s10_cost_model_blocks_weak_conversion() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(1, event_id="e2", canonical_id="c1", yes=0.33, liq=160),
        _market(2, event_id="e2", canonical_id="c2", yes=0.33, liq=140),
        _market(3, event_id="e2", canonical_id="c3", yes=0.33, liq=120),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.001,
            "min_effective_edge_bps": 80.0,
            "fee_bps": 14.0,
            "slippage_bps": 12.0,
            "depth_reference_liquidity": 1000.0,
            "depth_penalty_max_bps": 40.0,
        },
    )

    assert signals == []


def test_s10_allow_sell_conversion_emits_sell_legs() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(1, event_id="e3", canonical_id="c1", yes=0.41, liq=900),
        _market(2, event_id="e3", canonical_id="c2", yes=0.37, liq=860),
        _market(3, event_id="e3", canonical_id="c3", yes=0.30, liq=840),
    ]

    signals = strategy.generate(
        markets,
        {
            "allow_sell_conversion": True,
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert len(signals) == 3
    assert all(s.side == "sell" for s in signals)
    assert all(str(s.payload.get("direction")) == "sell_conversion" for s in signals)


def test_s10_dedupes_per_canonical_using_min_yes_for_buy() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(1, event_id="e4", canonical_id="c1", yes=0.25, liq=500),
        _market(2, event_id="e4", canonical_id="c1", yes=0.30, liq=900),
        _market(3, event_id="e4", canonical_id="c2", yes=0.32, liq=860),
        _market(4, event_id="e4", canonical_id="c3", yes=0.34, liq=840),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 8.0,
            "min_unique_canonicals": 3,
            "max_leg_weight": 0.7,
        },
    )

    assert len(signals) == 3
    payload = signals[0].payload
    assert str(payload.get("selection_mode")) == "min_yes_per_canonical"
    assert int(payload.get("raw_leg_count", 0)) == 4
    assert int(payload.get("unique_canonical_count", 0)) == 3

    selected_ids = {
        str(row.get("market_id", ""))
        for row in payload.get("basket_markets", [])
        if isinstance(row, dict)
    }
    assert selected_ids == {"m1", "m3", "m4"}


def test_s10_filters_by_unique_canonicals_and_leg_weight() -> None:
    strategy = S10NegRiskConversionArb()

    too_few_unique = [
        _market(1, event_id="e5", canonical_id="c1", yes=0.25, liq=500),
        _market(2, event_id="e5", canonical_id="c1", yes=0.28, liq=550),
        _market(3, event_id="e5", canonical_id="c2", yes=0.36, liq=600),
    ]
    signals_unique = strategy.generate(
        too_few_unique,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "min_unique_canonicals": 3,
        },
    )
    assert signals_unique == []

    concentrated = [
        _market(11, event_id="e6", canonical_id="c1", yes=0.79, liq=800),
        _market(12, event_id="e6", canonical_id="c2", yes=0.10, liq=760),
        _market(13, event_id="e6", canonical_id="c3", yes=0.08, liq=740),
    ]
    signals_concentrated = strategy.generate(
        concentrated,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_leg_weight": 0.7,
            "min_unique_canonicals": 3,
        },
    )
    assert signals_concentrated == []


def test_s10_weighted_cost_cap_blocks_candidate() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(21, event_id="e7", canonical_id="c1", yes=0.29, liq=150),
        _market(22, event_id="e7", canonical_id="c2", yes=0.32, liq=140),
        _market(23, event_id="e7", canonical_id="c3", yes=0.34, liq=130),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 20.0,
            "slippage_bps": 12.0,
            "depth_reference_liquidity": 1000.0,
            "depth_penalty_max_bps": 40.0,
            "max_weighted_total_cost_bps": 60.0,
            "max_leg_total_cost_bps": 200.0,
        },
    )

    assert signals == []
    reject_reasons = strategy.last_diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy_conversion:weighted_cost_cap_exceeded") == 1
    assert reject_reasons.get("event_no_actionable_candidate") == 1


def test_s10_leg_cost_cap_blocks_single_expensive_leg() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(31, event_id="e8", canonical_id="c1", yes=0.29, liq=900),
        _market(32, event_id="e8", canonical_id="c2", yes=0.32, liq=110),
        _market(33, event_id="e8", canonical_id="c3", yes=0.34, liq=880),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 8.0,
            "slippage_bps": 6.0,
            "depth_reference_liquidity": 1000.0,
            "depth_penalty_max_bps": 70.0,
            "max_weighted_total_cost_bps": 200.0,
            "max_leg_total_cost_bps": 70.0,
        },
    )

    assert signals == []
    reject_reasons = strategy.last_diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy_conversion:leg_cost_cap_exceeded") == 1
    assert reject_reasons.get("event_no_actionable_candidate") == 1


def test_s10_diagnostics_counts_excluded_events() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(41, event_id="blocked-e", canonical_id="c1", yes=0.28, liq=900),
        _market(42, event_id="blocked-e", canonical_id="c2", yes=0.31, liq=850),
        _market(43, event_id="blocked-e", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "exclude_event_ids": ["blocked-e"],
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    assert diagnostics.get("events_total") == 1
    assert diagnostics.get("signals_emitted") == 0
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("event_excluded") == 1

    reject_by_event = diagnostics.get("candidate_reject_reasons_by_event", {})
    blocked_event_reasons = reject_by_event.get("blocked-e", {})
    assert blocked_event_reasons.get("event_excluded") == 1


def test_s10_diagnostics_tracks_reject_reasons_by_event() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(51, event_id="blocked-e", canonical_id="c1", yes=0.28, liq=900),
        _market(52, event_id="blocked-e", canonical_id="c2", yes=0.31, liq=850),
        _market(53, event_id="blocked-e", canonical_id="c3", yes=0.34, liq=870),
        _market(61, event_id="thin-e", canonical_id="c1", yes=0.25, liq=900),
        _market(62, event_id="thin-e", canonical_id="c1", yes=0.27, liq=860),
    ]

    signals = strategy.generate(
        markets,
        {
            "exclude_event_ids": ["blocked-e"],
            "min_unique_canonicals": 3,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("event_excluded") == 1
    assert reject_reasons.get("event_under_min_unique") == 1

    reject_by_event = diagnostics.get("candidate_reject_reasons_by_event", {})
    assert reject_by_event.get("blocked-e", {}).get("event_excluded") == 1
    assert reject_by_event.get("thin-e", {}).get("event_under_min_unique") == 1


def test_s10_diagnostics_event_top_k_summary() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(71, event_id="expensive-e", canonical_id="c1", yes=0.29, liq=150),
        _market(72, event_id="expensive-e", canonical_id="c2", yes=0.32, liq=140),
        _market(73, event_id="expensive-e", canonical_id="c3", yes=0.34, liq=130),
        _market(81, event_id="blocked-e", canonical_id="c1", yes=0.28, liq=900),
        _market(82, event_id="blocked-e", canonical_id="c2", yes=0.31, liq=850),
        _market(83, event_id="blocked-e", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "exclude_event_ids": ["blocked-e"],
            "diagnostics_event_top_k": 1,
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 20.0,
            "slippage_bps": 12.0,
            "depth_reference_liquidity": 1000.0,
            "depth_penalty_max_bps": 40.0,
            "max_weighted_total_cost_bps": 60.0,
            "max_leg_total_cost_bps": 200.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics

    reject_by_event = diagnostics.get("candidate_reject_reasons_by_event", {})
    assert set(reject_by_event.keys()) == {"blocked-e", "expensive-e"}

    assert diagnostics.get("candidate_reject_reasons_by_event_top_k") == 1
    reject_top = diagnostics.get("candidate_reject_reasons_by_event_top", {})
    assert list(reject_top.keys()) == ["expensive-e"]
    assert reject_top["expensive-e"].get("buy_conversion:weighted_cost_cap_exceeded") == 1


def test_s10_rejects_when_post_quote_turns_negative() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(91, event_id="e-quote", canonical_id="c1", yes=0.28, liq=900),
        _market(92, event_id="e-quote", canonical_id="c2", yes=0.31, liq=850),
        _market(93, event_id="e-quote", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "quote_improve": 0.03,
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy_conversion:post_quote_non_positive_gross_edge") == 1

    pricing_diag = diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("filtered_post_floor_non_positive", 0.0))) == 1


def test_s10_rejects_when_post_slippage_turns_negative() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(101, event_id="e-slip", canonical_id="c1", yes=0.28, liq=900),
        _market(102, event_id="e-slip", canonical_id="c2", yes=0.31, liq=850),
        _market(103, event_id="e-slip", canonical_id="c3", yes=0.34, liq=870),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 1000.0,
            "depth_penalty_max_bps": 0.0,
            "max_weighted_total_cost_bps": 2000.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy_conversion:post_slippage_non_positive_gross_edge") == 1

    pricing_diag = diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("filtered_post_slippage_non_positive", 0.0))) == 1


def test_s10_floor_applied_to_tiny_leg_targets_and_payload() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(111, event_id="e-floor", canonical_id="c1", yes=0.005, liq=1000),
        _market(112, event_id="e-floor", canonical_id="c2", yes=0.30, liq=1000),
        _market(113, event_id="e-floor", canonical_id="c3", yes=0.33, liq=1000),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "max_abs_deviation": 0.50,
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 8.0,
        },
    )

    assert len(signals) == 3
    tiny_leg_signal = next(s for s in signals if s.market_id == "m111")
    assert abs(float(tiny_leg_signal.target_price) - 0.01) < 1e-12
    assert int(float(tiny_leg_signal.payload.get("tiny_price_legs", 0.0))) == 1
    assert int(float(tiny_leg_signal.payload.get("floor_adjusted_legs", 0.0))) == 1

    pricing_diag = strategy.last_diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("pair_candidates_priced", 0.0))) == 1
    assert int(float(pricing_diag.get("tiny_price_pairs", 0.0))) == 1
    assert int(float(pricing_diag.get("pairs_with_floor_adjustment", 0.0))) == 1


def test_s10_rejects_when_tiny_price_leg_share_exceeds_cap() -> None:
    strategy = S10NegRiskConversionArb()
    markets = [
        _market(121, event_id="e-tiny-cap", canonical_id="c1", yes=0.005, liq=1000),
        _market(122, event_id="e-tiny-cap", canonical_id="c2", yes=0.30, liq=1000),
        _market(123, event_id="e-tiny-cap", canonical_id="c3", yes=0.33, liq=1000),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.01,
            "max_abs_deviation": 0.50,
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 8.0,
            "max_tiny_price_leg_share": 0.20,
        },
    )

    assert signals == []
    reject_reasons = strategy.last_diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy_conversion:tiny_price_leg_share_exceeded") == 1

    pricing_diag = strategy.last_diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("filtered_tiny_price_leg_share_exceeded", 0.0))) == 1
