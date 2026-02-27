from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s8_no_carry_tailhedge import S8NoCarryTailHedge


def _market(i: int, *, event_id: str, yes: float, liq: float) -> MarketView:
    return MarketView(
        source="gamma",
        market_id=f"m{i}",
        canonical_id=f"c{i}",
        event_id=event_id,
        question=f"Q{i}",
        status="open",
        neg_risk=False,
        liquidity=liq,
        volume=100,
        yes_price=yes,
        no_price=1 - yes,
        best_bid=None,
        best_ask=None,
        mid_price=yes,
    )


def test_s8_generates_no_carry_signal() -> None:
    strategy = S8NoCarryTailHedge()
    markets = [
        _market(1, event_id="main-1", yes=0.20, liq=500),
        _market(2, event_id="tail-1", yes=0.05, liq=120),
    ]

    signals = strategy.generate(markets, {"yes_price_max_for_no": 0.25})

    assert len(signals) >= 1
    assert all(s.strategy == "s8" for s in signals)
    assert any(s.event_id == "main-1" for s in signals)


def test_s8_exclude_event_ids_filters_main_event() -> None:
    strategy = S8NoCarryTailHedge()
    markets = [
        _market(1, event_id="drop-me", yes=0.20, liq=500),
        _market(2, event_id="keep-me", yes=0.18, liq=520),
        _market(3, event_id="tail-1", yes=0.05, liq=90),
    ]

    signals = strategy.generate(
        markets,
        {
            "yes_price_max_for_no": 0.25,
            "exclude_event_ids": ["drop-me"],
        },
    )

    assert len(signals) >= 1
    assert {s.event_id for s in signals} == {"keep-me"}
