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
