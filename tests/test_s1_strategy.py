from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s1_cross_venue import S1CrossVenueScanner


def _market(
    i: int,
    *,
    canonical_id: str,
    event_id: str,
    mid: float,
    liquidity: float = 500.0,
) -> MarketView:
    return MarketView(
        source="gamma",
        market_id=f"m{i}",
        canonical_id=canonical_id,
        event_id=event_id,
        question=f"Q{i}",
        status="open",
        neg_risk=False,
        liquidity=liquidity,
        volume=100,
        yes_price=mid,
        no_price=1 - mid,
        best_bid=None,
        best_ask=None,
        mid_price=mid,
    )


def test_s1_generates_for_spread() -> None:
    strategy = S1CrossVenueScanner()
    markets = [
        _market(1, canonical_id="c1", event_id="e1", mid=0.20),
        _market(2, canonical_id="c1", event_id="e1", mid=0.35),
    ]

    signals = strategy.generate(markets, {"min_spread": 0.03, "max_order_notional": 1.0})

    assert len(signals) >= 1
    assert all(s.strategy == "s1" for s in signals)


def test_s1_non_positive_notional_disables_strategy() -> None:
    strategy = S1CrossVenueScanner()
    markets = [
        _market(1, canonical_id="c1", event_id="e1", mid=0.20),
        _market(2, canonical_id="c1", event_id="e1", mid=0.35),
    ]

    signals = strategy.generate(markets, {"min_spread": 0.03, "max_order_notional": 0.0})

    assert signals == []
