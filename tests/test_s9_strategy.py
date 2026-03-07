from __future__ import annotations

from monomarket.models import MarketView
from monomarket.signals.strategies.s9_yes_no_parity import S9YesNoParityArb


def _market(
    i: int,
    *,
    canonical_id: str,
    event_id: str,
    yes: float,
    no: float,
    liq: float,
) -> MarketView:
    return MarketView(
        source="gamma" if i % 2 else "clob",
        market_id=f"m{i}",
        canonical_id=canonical_id,
        event_id=event_id,
        question=f"Q{i}",
        status="open",
        neg_risk=False,
        liquidity=liq,
        volume=100,
        yes_price=yes,
        no_price=no,
        best_bid=None,
        best_ask=None,
        mid_price=yes,
    )


def test_s9_generates_buy_carry_pair() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(1, canonical_id="c1", event_id="e1", yes=0.44, no=0.58, liq=900),
        _market(2, canonical_id="c1", event_id="e1", yes=0.49, no=0.53, liq=850),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 10.0,
        },
    )

    assert len(signals) == 2
    assert all(s.strategy == "s9" for s in signals)
    tokens = {str(s.payload.get("primary_leg", {}).get("token", "")) for s in signals}
    assert tokens == {"YES", "NO"}
    assert all(float(s.payload.get("effective_edge_bps", 0.0)) > 0 for s in signals)


def test_s9_cost_model_blocks_weak_edge() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(1, canonical_id="c1", event_id="e1", yes=0.49, no=0.52, liq=150),
        _market(2, canonical_id="c1", event_id="e1", yes=0.50, no=0.51, liq=120),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 10.0,
            "fee_bps": 12.0,
            "slippage_bps": 8.0,
            "depth_reference_liquidity": 1000.0,
            "depth_penalty_max_bps": 40.0,
        },
    )

    assert signals == []


def test_s9_allow_sell_parity_emits_sell_legs() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(1, canonical_id="c2", event_id="e2", yes=0.57, no=0.47, liq=900),
        _market(2, canonical_id="c2", event_id="e2", yes=0.52, no=0.46, liq=850),
    ]

    signals = strategy.generate(
        markets,
        {
            "allow_sell_parity": True,
            "min_effective_edge_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    sell_signals = [s for s in signals if s.side == "sell"]
    assert len(sell_signals) == 2
    assert all(str(s.payload.get("direction")) == "sell_overround" for s in sell_signals)
