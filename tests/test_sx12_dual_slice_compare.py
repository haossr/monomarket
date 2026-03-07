from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "sx12_dual_slice_compare.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("sx12_dual_slice_compare", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load sx12_dual_slice_compare module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_slice_specs() -> None:
    module = _load_module()

    specs = module.parse_slice_specs("recent24h:24,recent7d:168")
    assert [s.label for s in specs] == ["recent24h", "recent7d"]
    assert [s.hours for s in specs] == [24.0, 168.0]


def test_slice_default_includes_recent14d() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(["--baseline-config", "base.yaml", "--candidate-config", "cand.yaml"])

    specs = module.parse_slice_specs(str(args.slices))
    assert [s.label for s in specs] == ["recent24h", "recent7d", "recent14d"]
    assert [s.hours for s in specs] == [24.0, 168.0, 336.0]


def test_summarize_strategy_normalizes_reject_reason_prefix() -> None:
    module = _load_module()

    report = {
        "results": [
            {
                "strategy": "s9",
                "pnl": -0.25,
                "max_drawdown": 0.8,
                "trade_count": 2,
                "closed_winrate": 0.5,
                "mtm_winrate": 0.25,
                "closed_sample_count": 2,
                "mtm_sample_count": 4,
            }
        ],
        "replay": [
            {
                "strategy": "s9",
                "risk_allowed": False,
                "risk_reason": "strategy notional limit exceeded: 1200.00 > 1000.00",
            },
            {
                "strategy": "s9",
                "risk_allowed": False,
                "risk_reason": "strategy notional limit exceeded: 1400.00 > 1000.00",
            },
            {
                "strategy": "s9",
                "risk_allowed": True,
                "risk_reason": "none",
            },
            {
                "strategy": "s10",
                "risk_allowed": False,
                "risk_reason": "circuit breaker open: rejected=10, threshold=5",
            },
        ],
    }

    summary = module.summarize_strategy(report, strategy="s9")
    assert summary["strategy"] == "s9"
    assert abs(float(summary["max_drawdown"]) - 0.8) < 1e-12
    assert summary["replay_rows"] == 3
    assert summary["rejected_rows"] == 2
    assert summary["executed_rows"] == 1
    assert abs(float(summary["reject_share"]) - (2.0 / 3.0)) < 1e-12
    assert summary["top_reject_reason"] == "strategy notional limit exceeded:2"


def test_summary_delta_and_markdown_render() -> None:
    module = _load_module()

    baseline = {
        "s9": {
            "pnl": -0.5,
            "max_drawdown": 1.2,
            "trade_count": 1,
            "executed_rows": 3,
            "rejected_rows": 2,
            "mtm_winrate": 0.2,
            "closed_winrate": 0.0,
            "top_reject_reason": "none",
        }
    }
    candidate = {
        "s9": {
            "pnl": 0.1,
            "max_drawdown": 0.9,
            "trade_count": 2,
            "executed_rows": 5,
            "rejected_rows": 1,
            "mtm_winrate": 0.4,
            "closed_winrate": 0.5,
            "top_reject_reason": "none",
        }
    }
    delta = module._summary_delta(baseline, candidate, strategies=["s9"])
    assert abs(float(delta["s9"]["pnl"]) - 0.6) < 1e-12
    assert abs(float(delta["s9"]["max_drawdown"]) + 0.3) < 1e-12
    assert int(delta["s9"]["executed_rows"]) == 2
    assert int(delta["s9"]["rejected_rows"]) == -1

    rendered = module.render_markdown(
        {
            "generated_at": "2026-03-07T17:40:00Z",
            "anchor_ts": "2026-03-07T17:39:00Z",
            "baseline_config": "/tmp/base.yaml",
            "candidate_config": "/tmp/candidate.yaml",
            "strategies": ["s9"],
            "slices": [
                {
                    "label": "recent24h",
                    "hours": 24,
                    "baseline": {"by_strategy": baseline},
                    "candidate": {"by_strategy": candidate},
                    "delta": delta,
                }
            ],
        }
    )
    assert "# Sx12 Dual-Slice Compare" in rendered
    assert "## recent24h (24h)" in rendered
    assert "| s9 | -0.5000 | 0.1000 | +0.6000 |" in rendered
