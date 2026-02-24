from __future__ import annotations

from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_nightly_report.sh"


def test_nightly_summary_contains_canonical_alias_fields() -> None:
    content = SCRIPT_PATH.read_text()

    required_tokens = [
        "positive_window_rate=",
        "empty_window_count=",
        "range_hours=",
        "coverage_ratio=",
        "overlap_ratio=",
        "rolling_reject_top_k=",
    ]
    for token in required_tokens:
        assert token in content


def test_nightly_reject_topk_zero_semantics_documented_in_script() -> None:
    content = SCRIPT_PATH.read_text()

    assert "0=disabled" in content
    assert 'rolling_reject_top = "disabled" if rolling_reject_top_k <= 0 else "none"' in content
