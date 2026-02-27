from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s2_negrisk_rebalance import S2NegRiskRebalance


def _market(i: int, *, event_id: str, yes: float) -> MarketView:
    return MarketView(
        source="gamma",
        market_id=f"m{i}",
        canonical_id=f"c{i}",
        event_id=event_id,
        question=f"Q{i}",
        status="open",
        neg_risk=True,
        liquidity=500 + i,
        volume=100,
        yes_price=yes,
        no_price=1 - yes,
        best_bid=None,
        best_ask=None,
        mid_price=yes,
    )


def test_s2_generates_for_deviated_event() -> None:
    strategy = S2NegRiskRebalance()
    markets = [
        _market(1, event_id="e1", yes=0.80),
        _market(2, event_id="e1", yes=0.30),
    ]

    signals = strategy.generate(markets, {"prob_sum_tolerance": 0.04})

    assert len(signals) == 2
    assert {s.event_id for s in signals} == {"e1"}


def test_s2_exclude_event_ids_filters_event() -> None:
    strategy = S2NegRiskRebalance()
    markets = [
        _market(1, event_id="drop-me", yes=0.80),
        _market(2, event_id="drop-me", yes=0.30),
        _market(3, event_id="keep-me", yes=0.80),
        _market(4, event_id="keep-me", yes=0.30),
    ]

    signals = strategy.generate(
        markets,
        {
            "prob_sum_tolerance": 0.04,
            "exclude_event_ids": ["drop-me"],
        },
    )

    assert len(signals) == 2
    assert {s.event_id for s in signals} == {"keep-me"}
