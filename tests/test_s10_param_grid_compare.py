from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "s10_param_grid_compare.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("s10_param_grid_compare", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load s10_param_grid_compare module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_float_values_dedup_and_range() -> None:
    module = _load_module()

    values = module.parse_float_values(
        "0.02,0.03,0.02",
        field_name="prob_sum_tolerance_grid",
        min_value=0.0,
        max_value=1.0,
    )
    assert values == [0.02, 0.03]


def test_build_s10_param_grid_cartesian_size() -> None:
    module = _load_module()

    grid = module.build_s10_param_grid(
        prob_sum_tolerance=[0.01, 0.02],
        max_abs_deviation=[0.1],
        max_tiny_price_leg_share=[0.2, 0.3],
        max_floor_adjusted_leg_share=[0.25],
    )

    assert len(grid) == 4
    assert {tuple(sorted(item.items())) for item in grid} == {
        tuple(
            sorted(
                {
                    "prob_sum_tolerance": prob_tol,
                    "max_abs_deviation": 0.1,
                    "max_tiny_price_leg_share": tiny,
                    "max_floor_adjusted_leg_share": 0.25,
                }.items()
            )
        )
        for prob_tol in [0.01, 0.02]
        for tiny in [0.2, 0.3]
    }


def test_s10_grid_default_slices_include_recent14d() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(["--baseline-config", "base.yaml"])

    assert str(args.slices) == "recent24h:24,recent7d:168,recent14d:336"


def test_apply_s10_overrides_keeps_base_immutable() -> None:
    module = _load_module()

    base = {
        "strategies": {
            "s9": {"max_order_notional": 12.0},
            "s10": {"prob_sum_tolerance": 0.02, "max_abs_deviation": 0.2},
        }
    }

    updated = module.apply_s10_overrides(
        base,
        {
            "prob_sum_tolerance": 0.015,
            "max_abs_deviation": 0.15,
            "max_tiny_price_leg_share": 0.2,
            "max_floor_adjusted_leg_share": 0.25,
        },
    )

    assert base["strategies"]["s10"]["prob_sum_tolerance"] == 0.02
    assert updated["strategies"]["s9"]["max_order_notional"] == 12.0
    assert updated["strategies"]["s10"]["prob_sum_tolerance"] == 0.015
    assert updated["strategies"]["s10"]["max_abs_deviation"] == 0.15
    assert updated["strategies"]["s10"]["max_tiny_price_leg_share"] == 0.2


def test_candidate_sort_key_prefers_constraint_pass_then_slice_stability() -> None:
    module = _load_module()

    rows = [
        {
            "candidate_id": "cand-b",
            "passes_constraints": True,
            "min_slice_delta_pnl": 0.0,
            "max_slice_delta_max_drawdown": 0.03,
            "total_delta_pnl": 0.2,
            "total_delta_exec": 1,
            "total_delta_rej": 0,
            "total_delta_max_drawdown": 0.03,
        },
        {
            "candidate_id": "cand-a",
            "passes_constraints": True,
            "min_slice_delta_pnl": 0.0,
            "max_slice_delta_max_drawdown": 0.01,
            "total_delta_pnl": 0.2,
            "total_delta_exec": 1,
            "total_delta_rej": 0,
            "total_delta_max_drawdown": 0.01,
        },
        {
            "candidate_id": "cand-c",
            "passes_constraints": False,
            "min_slice_delta_pnl": 0.5,
            "max_slice_delta_max_drawdown": -0.2,
            "total_delta_pnl": 1.0,
            "total_delta_exec": 3,
            "total_delta_rej": -1,
            "total_delta_max_drawdown": -0.3,
        },
    ]

    ranked = sorted(rows, key=module.candidate_sort_key)
    assert [row["candidate_id"] for row in ranked] == ["cand-a", "cand-b", "cand-c"]


def test_summarize_compare_payload_and_markdown() -> None:
    module = _load_module()

    payload = {
        "slices": [
            {
                "label": "recent24h",
                "hours": 24,
                "delta": {
                    "s10": {
                        "pnl": 0.6,
                        "max_drawdown": -0.2,
                        "executed_rows": 3,
                        "rejected_rows": -1,
                        "mtm_winrate": 0.1,
                    }
                },
            },
            {
                "label": "recent7d",
                "hours": 168,
                "delta": {
                    "s10": {
                        "pnl": -0.1,
                        "max_drawdown": -0.1,
                        "executed_rows": 1,
                        "rejected_rows": -2,
                        "mtm_winrate": -0.02,
                    }
                },
            },
        ]
    }

    summary = module.summarize_compare_payload(payload, objective_strategy="s10")
    assert abs(float(summary["total_delta_pnl"]) - 0.5) < 1e-12
    assert abs(float(summary["min_slice_delta_pnl"]) - (-0.1)) < 1e-12
    assert int(summary["non_negative_slice_count"]) == 1
    assert int(summary["total_delta_exec"]) == 4
    assert int(summary["total_delta_rej"]) == -3
    assert abs(float(summary["total_delta_max_drawdown"]) - (-0.3)) < 1e-12
    assert abs(float(summary["max_slice_delta_max_drawdown"]) - (-0.1)) < 1e-12
    assert abs(float(summary["total_delta_mtm_winrate"]) - 0.08) < 1e-12

    markdown = module.render_markdown(
        {
            "generated_at": "2026-03-07T18:30:00Z",
            "anchor_ts": "2026-03-07T18:30:00Z",
            "baseline_config": "/tmp/base.yaml",
            "candidate_base_config": "/tmp/candidate.yaml",
            "objective_strategy": "s10",
            "min_slice_delta_pnl_threshold": 0.0,
            "max_slice_delta_max_drawdown_threshold": 0.0,
            "total_candidates": 1,
            "candidates": [
                {
                    "rank": 1,
                    "candidate_id": "cand-001",
                    "overrides": {
                        "prob_sum_tolerance": 0.015,
                        "max_abs_deviation": 0.15,
                        "max_tiny_price_leg_share": 0.2,
                        "max_floor_adjusted_leg_share": 0.25,
                    },
                    "min_slice_delta_pnl": 0.1,
                    "max_slice_delta_max_drawdown": -0.1,
                    "passes_constraints": True,
                    "total_delta_pnl": 0.5,
                    "total_delta_exec": 4,
                    "total_delta_rej": -3,
                    "total_delta_max_drawdown": -0.2,
                    "total_delta_mtm_winrate": 0.08,
                }
            ],
        }
    )

    assert "# S10 Parameter Grid Compare" in markdown
    assert "| rank | candidate | prob_tol |" in markdown
    assert (
        "| rank | candidate | prob_tol | max_abs | tiny_share | floor_share | min(Δpnl) | max(ΔmaxDD) |"
        in markdown
    )
    assert (
        "| 1 | cand-001 | 0.0150 | 0.1500 | 0.2000 | 0.2500 | +0.1000 | -0.1000 | +0.5000 | +4 | -3 | -0.2000 | +0.0800 | yes |"
        in markdown
    )
