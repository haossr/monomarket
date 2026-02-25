from __future__ import annotations

from monomarket.backtest.reject_reason import (
    aggregate_reject_reasons,
    format_reject_top,
    normalize_reject_reason,
)


def test_normalize_reject_reason_prefix_rules() -> None:
    assert (
        normalize_reject_reason("strategy notional limit exceeded: 1019.19 > 1000.00")
        == "strategy notional limit exceeded"
    )
    assert (
        normalize_reject_reason("circuit breaker open: rejected=10, threshold=5")
        == "circuit breaker open"
    )
    assert normalize_reject_reason("riskA") == "riskA"


def test_aggregate_reject_reasons_normalized_merges_and_sorts() -> None:
    aggregated = aggregate_reject_reasons(
        {
            "strategy notional limit exceeded: 1019.19 > 1000.00": 3,
            "strategy notional limit exceeded: 1019.21 > 1000.00": 2,
            "circuit breaker open: rejected=10, threshold=5": 1,
        },
        normalize=True,
    )
    assert aggregated == [
        ("strategy notional limit exceeded", 5),
        ("circuit breaker open", 1),
    ]


def test_format_reject_top_disabled_none_and_tie_order() -> None:
    assert format_reject_top({}, top_k=0, normalize=False) == ("disabled", [])
    assert format_reject_top({}, top_k=2, normalize=False) == ("none", [])

    text, pairs = format_reject_top(
        {
            "z-reason": 5,
            "a-reason": 5,
            "m-reason": 4,
        },
        top_k=2,
        normalize=False,
    )
    assert text == "a-reason:5;z-reason:5"
    assert pairs == [("a-reason", 5), ("z-reason", 5)]
