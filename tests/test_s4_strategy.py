from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s4_low_prob_yes import S4LowProbYesBasket


def _market(i: int, yes: float = 0.08) -> MarketView:
    return MarketView(
        source="gamma",
        market_id=f"m{i}",
        canonical_id=f"c{i}",
        event_id=f"e{i}",
        question=f"Q{i}",
        status="open",
        neg_risk=False,
        liquidity=500 + i,
        volume=100,
        yes_price=yes,
        no_price=1 - yes,
        best_bid=None,
        best_ask=None,
        mid_price=yes,
    )


def test_s4_default_max_candidates_is_40() -> None:
    strategy = S4LowProbYesBasket()
    markets = [_market(i) for i in range(45)]

    signals = strategy.generate(markets, {"yes_price_min": 0.01, "yes_price_max": 0.15})

    assert len(signals) == 40


def test_s4_respects_max_candidates_override() -> None:
    strategy = S4LowProbYesBasket()
    markets = [_market(i) for i in range(45)]

    signals = strategy.generate(
        markets,
        {"yes_price_min": 0.01, "yes_price_max": 0.15, "max_candidates": 24},
    )

    assert len(signals) == 24
