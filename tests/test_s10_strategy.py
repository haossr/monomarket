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
