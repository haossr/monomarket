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
    question: str | None = None,
) -> MarketView:
    return MarketView(
        source="gamma" if i % 2 else "clob",
        market_id=f"m{i}",
        canonical_id=canonical_id,
        event_id=event_id,
        question=question or f"Q{i}",
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
        _market(
            1,
            canonical_id="c1",
            event_id="e1",
            yes=0.44,
            no=0.53,
            liq=900,
            question="Will Team A win?",
        ),
        _market(
            2,
            canonical_id="c1",
            event_id="e1",
            yes=0.49,
            no=0.58,
            liq=850,
            question="Will Team A win?",
        ),
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
    assert all(bool(s.payload.get("pair_atomic", False)) for s in signals)
    pair_batch_ids = {str(s.payload.get("pair_batch_id", "")) for s in signals}
    assert len(pair_batch_ids) == 1
    condition_keys = {str(s.payload.get("condition_key", "")) for s in signals}
    assert len(condition_keys) == 1
    market_ids = {str(s.market_id) for s in signals}
    assert len(market_ids) == 1
    assert all(bool(s.payload.get("pair_same_market", False)) for s in signals)
    assert all(str(s.payload.get("pair_mode", "")) == "same_market" for s in signals)


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


def test_s9_event_pair_guard_limits_pairs_per_event() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c1",
            event_id="e1",
            yes=0.44,
            no=0.53,
            liq=900,
            question="Will Team A win?",
        ),
        _market(
            2,
            canonical_id="c1",
            event_id="e1",
            yes=0.47,
            no=0.58,
            liq=880,
            question="Will Team A win?",
        ),
        _market(
            3,
            canonical_id="c2",
            event_id="e1",
            yes=0.43,
            no=0.55,
            liq=910,
            question="Will Team B win?",
        ),
        _market(
            4,
            canonical_id="c2",
            event_id="e1",
            yes=0.49,
            no=0.52,
            liq=860,
            question="Will Team B win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_pairs_per_event": 1,
            "max_event_pair_notional": 20.0,
            "max_order_notional": 10.0,
        },
    )

    assert len(signals) == 2
    assert {s.event_id for s in signals} == {"e1"}
    assert all(int(s.payload.get("event_pair_index", 0)) == 1 for s in signals)


def test_s9_event_notional_budget_caps_pair_size() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c1",
            event_id="e1",
            yes=0.44,
            no=0.53,
            liq=1200,
            question="Will Team A win?",
        ),
        _market(
            2,
            canonical_id="c1",
            event_id="e1",
            yes=0.49,
            no=0.58,
            liq=1200,
            question="Will Team A win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "liquidity_fraction": 0.02,
            "max_order_notional": 20.0,
            "max_event_pair_notional": 2.0,
        },
    )

    assert len(signals) == 2
    pair_notional = float(signals[0].payload.get("pair_notional", 0.0))
    assert pair_notional <= 2.0 + 1e-9
    assert all(abs(float(s.size_hint) - float(signals[0].size_hint)) < 1e-12 for s in signals)


def test_s9_default_blocks_cross_event_pair() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(1, canonical_id="c1", event_id="e1", yes=0.44, no=0.58, liq=900),
        _market(2, canonical_id="c1", event_id="e2", yes=0.49, no=0.53, liq=900),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []


def test_s9_default_same_market_guard_blocks_cross_market_only_edge() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c-cross-only",
            event_id="e-cross",
            yes=0.44,
            no=0.58,
            liq=900,
            question="Will Team X win?",
        ),
        _market(
            2,
            canonical_id="c-cross-only",
            event_id="e-cross",
            yes=0.49,
            no=0.53,
            liq=900,
            question="Will Team X win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []

    cross_market_signals = strategy.generate(
        markets,
        {
            "require_same_market": False,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert len(cross_market_signals) == 2
    assert len({str(s.market_id) for s in cross_market_signals}) == 2
    assert all(not bool(s.payload.get("pair_same_market", True)) for s in cross_market_signals)


def test_s9_default_blocks_cross_condition_pair_within_event() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="democratic party",
            event_id="e-same",
            yes=0.44,
            no=0.58,
            liq=900,
            question="Will Democrats win the Senate?",
        ),
        _market(
            2,
            canonical_id="democratic party",
            event_id="e-same",
            yes=0.49,
            no=0.53,
            liq=900,
            question="Will Democrats win the House?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []


def test_s9_can_opt_out_same_condition_guard() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="democratic party",
            event_id="e-same",
            yes=0.44,
            no=0.58,
            liq=900,
            question="Will Democrats win the Senate?",
        ),
        _market(
            2,
            canonical_id="democratic party",
            event_id="e-same",
            yes=0.49,
            no=0.53,
            liq=900,
            question="Will Democrats win the House?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "require_same_condition": False,
            "require_same_event": True,
            "require_same_market": False,
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert len(signals) == 2


def test_s9_diagnostics_tracks_reject_reason_by_event() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c-mixed",
            event_id="e1",
            yes=0.44,
            no=0.58,
            liq=900,
            question="Will Team A win?",
        ),
        _market(
            2,
            canonical_id="c-mixed",
            event_id="e2",
            yes=0.49,
            no=0.53,
            liq=900,
            question="Will Team A win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 5.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "require_same_event": True,
            "require_same_condition": True,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy:non_positive_gross_edge") == 1

    reject_by_event = diagnostics.get("candidate_reject_reasons_by_event", {})
    assert reject_by_event.get("e1", {}).get("buy:non_positive_gross_edge") == 1


def test_s9_diagnostics_event_top_k_summary() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c-main",
            event_id="e1",
            yes=0.44,
            no=0.58,
            liq=900,
            question="Will Team A win?",
        ),
        _market(
            2,
            canonical_id="c-main",
            event_id="e1",
            yes=0.49,
            no=0.53,
            liq=900,
            question="Will Team A win?",
        ),
        _market(
            3,
            canonical_id="c-thin",
            event_id="e2",
            yes=0.40,
            no=0.61,
            liq=900,
            question="Will Team B win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "allow_sell_parity": True,
            "min_effective_edge_bps": 900.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "diagnostics_event_top_k": 1,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics

    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("canonical_under_two_legs") == 1
    assert reject_reasons.get("buy:non_positive_gross_edge") == 1
    assert reject_reasons.get("sell:effective_edge_below_min") == 1

    reject_by_event = diagnostics.get("candidate_reject_reasons_by_event", {})
    assert reject_by_event.get("e1", {}).get("buy:non_positive_gross_edge") == 1
    assert reject_by_event.get("e1", {}).get("sell:effective_edge_below_min") == 1
    assert reject_by_event.get("e2", {}).get("canonical_under_two_legs") == 1

    assert diagnostics.get("candidate_reject_reasons_by_event_top_k") == 1
    reject_top = diagnostics.get("candidate_reject_reasons_by_event_top", {})
    assert list(reject_top.keys()) == ["e1"]


def test_s9_floor_applied_to_tiny_leg_targets_and_payload() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c-floor",
            event_id="e-floor",
            yes=0.0005,
            no=0.82,
            liq=1000,
            question="Will Team C win?",
        ),
        _market(
            2,
            canonical_id="c-floor",
            event_id="e-floor",
            yes=0.30,
            no=0.15,
            liq=1000,
            question="Will Team C win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
            "max_order_notional": 10.0,
        },
    )

    assert len(signals) == 2
    yes_signal = next(
        s for s in signals if str(s.payload.get("primary_leg", {}).get("token")) == "YES"
    )
    assert abs(float(yes_signal.target_price) - 0.01) < 1e-12
    assert float(yes_signal.payload.get("pre_floor_parity_sum", 0.0)) < float(
        yes_signal.payload.get("post_floor_parity_sum", 0.0)
    )
    assert int(float(yes_signal.payload.get("tiny_price_legs", 0.0))) == 1
    assert int(float(yes_signal.payload.get("floor_adjusted_legs", 0.0))) == 1

    pricing_diag = strategy.last_diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("tiny_price_pairs", 0.0))) == 1
    assert int(float(pricing_diag.get("pairs_with_floor_adjustment", 0.0))) == 1


def test_s9_rejects_when_post_floor_parity_turns_negative() -> None:
    strategy = S9YesNoParityArb()
    markets = [
        _market(
            1,
            canonical_id="c-floor-reject",
            event_id="e-floor-reject",
            yes=0.0005,
            no=0.995,
            liq=1000,
            question="Will Team D win?",
        ),
        _market(
            2,
            canonical_id="c-floor-reject",
            event_id="e-floor-reject",
            yes=0.85,
            no=0.60,
            liq=1000,
            question="Will Team D win?",
        ),
    ]

    signals = strategy.generate(
        markets,
        {
            "min_effective_edge_bps": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "depth_penalty_max_bps": 0.0,
        },
    )

    assert signals == []
    diagnostics = strategy.last_diagnostics
    reject_reasons = diagnostics.get("candidate_reject_reasons", {})
    assert reject_reasons.get("buy:post_floor_non_positive_gross_edge") == 1

    pricing_diag = diagnostics.get("pricing_consistency", {})
    assert int(float(pricing_diag.get("filtered_post_floor_non_positive", 0.0))) == 1
    assert int(float(pricing_diag.get("tiny_price_pairs", 0.0))) == 1
