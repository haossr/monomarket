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


def test_parse_bool_values_dedup_and_aliases() -> None:
    module = _load_module()

    values = module.parse_bool_values(
        "false,TRUE,0,yes,false",
        field_name="require_same_source_grid",
    )
    assert values == [False, True]


def test_build_s10_param_grid_cartesian_size() -> None:
    module = _load_module()

    grid = module.build_s10_param_grid(
        prob_sum_tolerance=[0.01, 0.02],
        max_abs_deviation=[0.1],
        max_tiny_price_leg_share=[0.2, 0.3],
        max_floor_adjusted_leg_share=[0.25],
        require_same_source=[False, True],
    )

    assert len(grid) == 8
    assert {tuple(sorted(item.items())) for item in grid} == {
        tuple(
            sorted(
                {
                    "prob_sum_tolerance": prob_tol,
                    "max_abs_deviation": 0.1,
                    "max_tiny_price_leg_share": tiny,
                    "max_floor_adjusted_leg_share": 0.25,
                    "require_same_source": same_source,
                }.items()
            )
        )
        for prob_tol in [0.01, 0.02]
        for tiny in [0.2, 0.3]
        for same_source in [False, True]
    }


def test_s10_grid_default_slices_include_recent14d() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(["--baseline-config", "base.yaml"])

    assert str(args.slices) == "recent24h:24,recent7d:168,recent14d:336"
    assert str(args.require_same_source_grid) == "false,true"
    assert bool(args.rebuild_signals_window) is False
    assert float(args.rebuild_step_hours) == 12.0
    assert int(args.rebuild_market_limit) == 2000
    assert int(args.rebuild_ingest_limit) == 300
    assert bool(args.skip_ingest_rebuild) is False
    assert bool(args.enforce_settle_profile_match) is True
    assert bool(args.inject_candidate_settle_mismatch) is False


def test_s10_grid_parser_allows_settle_profile_mismatch_override() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(
        [
            "--baseline-config",
            "base.yaml",
            "--allow-settle-profile-mismatch",
        ]
    )

    assert bool(args.enforce_settle_profile_match) is False
    assert bool(args.inject_candidate_settle_mismatch) is False


def test_s10_grid_parser_allows_candidate_settle_mismatch_injection() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(
        [
            "--baseline-config",
            "base.yaml",
            "--inject-candidate-settle-mismatch",
        ]
    )

    assert bool(args.inject_candidate_settle_mismatch) is True
    assert bool(args.enforce_settle_profile_match) is True


def test_build_dual_slice_compare_cmd_includes_rebuild_flags() -> None:
    module = _load_module()

    cmd = module._build_dual_slice_compare_cmd(
        compare_script=Path("/tmp/sx12.py"),
        baseline_config=Path("/tmp/base.yaml"),
        candidate_config=Path("/tmp/cand.yaml"),
        strategies="s9,s10",
        slices="recent24h:24,recent7d:168",
        out_dir=Path("/tmp/out"),
        anchor_ts="2026-03-07T23:00:00Z",
        rebuild_signals_window=True,
        rebuild_step_hours=6.0,
        rebuild_market_limit=123,
        rebuild_ingest_limit=45,
        skip_ingest_rebuild=True,
        baseline_settle_window_end=True,
        candidate_settle_window_end=False,
    )

    assert cmd[:2] == [sys.executable, "/tmp/sx12.py"]
    assert "--rebuild-signals-window" in cmd
    assert "--rebuild-step-hours" in cmd
    assert "6.0" in cmd
    assert "--rebuild-market-limit" in cmd
    assert "123" in cmd
    assert "--rebuild-ingest-limit" in cmd
    assert "45" in cmd
    assert "--skip-ingest-rebuild" in cmd
    assert "--candidate-no-settle-window-end" in cmd
    assert "--baseline-no-settle-window-end" not in cmd


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
            "require_same_source": True,
        },
    )

    assert base["strategies"]["s10"]["prob_sum_tolerance"] == 0.02
    assert updated["strategies"]["s9"]["max_order_notional"] == 12.0
    assert updated["strategies"]["s10"]["prob_sum_tolerance"] == 0.015
    assert updated["strategies"]["s10"]["max_abs_deviation"] == 0.15
    assert updated["strategies"]["s10"]["max_tiny_price_leg_share"] == 0.2
    assert bool(updated["strategies"]["s10"]["require_same_source"]) is True


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


def test_summarize_same_source_rollup_groups_and_ranks_candidates() -> None:
    module = _load_module()

    entries = [
        {
            "candidate_id": "cand-a",
            "overrides": {"require_same_source": False},
            "passes_constraints": True,
            "min_slice_delta_pnl": 0.10,
            "max_slice_delta_max_drawdown": 0.00,
            "total_delta_pnl": 1.20,
            "total_delta_exec": 4,
            "total_delta_rej": -1,
        },
        {
            "candidate_id": "cand-b",
            "overrides": {"require_same_source": False},
            "passes_constraints": False,
            "min_slice_delta_pnl": -0.20,
            "max_slice_delta_max_drawdown": 0.30,
            "total_delta_pnl": -0.40,
            "total_delta_exec": -1,
            "total_delta_rej": 2,
        },
        {
            "candidate_id": "cand-c",
            "overrides": {"require_same_source": True},
            "passes_constraints": True,
            "min_slice_delta_pnl": 0.05,
            "max_slice_delta_max_drawdown": 0.01,
            "total_delta_pnl": 0.80,
            "total_delta_exec": 2,
            "total_delta_rej": 0,
        },
    ]

    rollup = module.summarize_same_source_rollup(entries)
    assert len(rollup) == 2

    same_source_off = next(item for item in rollup if item["require_same_source"] is False)
    assert int(same_source_off["candidate_count"]) == 2
    assert int(same_source_off["pass_count"]) == 1
    assert abs(float(same_source_off["pass_rate"]) - 0.5) < 1e-12
    assert str(same_source_off["best_candidate_id"]) == "cand-a"
    assert str(same_source_off["best_pass_candidate_id"]) == "cand-a"
    assert abs(float(same_source_off["avg_total_delta_pnl"]) - 0.4) < 1e-12

    same_source_on = next(item for item in rollup if item["require_same_source"] is True)
    assert int(same_source_on["candidate_count"]) == 1
    assert int(same_source_on["pass_count"]) == 1
    assert abs(float(same_source_on["pass_rate"]) - 1.0) < 1e-12
    assert str(same_source_on["best_candidate_id"]) == "cand-c"
    assert str(same_source_on["best_pass_candidate_id"]) == "cand-c"


def test_apply_constraint_flags_respects_settle_profile_guard() -> None:
    module = _load_module()

    strict_entry = {
        "min_slice_delta_pnl": 0.1,
        "max_slice_delta_max_drawdown": -0.05,
        "settle_mismatch_slice_count": 1,
    }
    module.apply_constraint_flags(
        strict_entry,
        min_slice_delta_pnl_threshold=0.0,
        max_slice_delta_max_drawdown_threshold=0.0,
        enforce_settle_profile_match=True,
    )
    assert bool(strict_entry["passes_min_slice_delta_pnl"]) is True
    assert bool(strict_entry["passes_max_slice_delta_max_drawdown"]) is True
    assert bool(strict_entry["passes_settle_profile_match"]) is False
    assert bool(strict_entry["passes_constraints"]) is False

    relaxed_entry = {
        "min_slice_delta_pnl": 0.1,
        "max_slice_delta_max_drawdown": -0.05,
        "settle_mismatch_slice_count": 1,
    }
    module.apply_constraint_flags(
        relaxed_entry,
        min_slice_delta_pnl_threshold=0.0,
        max_slice_delta_max_drawdown_threshold=0.0,
        enforce_settle_profile_match=False,
    )
    assert bool(relaxed_entry["passes_settle_profile_match"]) is True
    assert bool(relaxed_entry["passes_constraints"]) is True


def test_candidate_sort_key_downgrades_settle_mismatch_candidate() -> None:
    module = _load_module()

    matched = {
        "candidate_id": "cand-match",
        "min_slice_delta_pnl": 0.0,
        "max_slice_delta_max_drawdown": 0.0,
        "total_delta_pnl": 0.0,
        "total_delta_exec": 0,
        "total_delta_rej": 0,
        "total_delta_max_drawdown": 0.0,
        "settle_mismatch_slice_count": 0,
    }
    mismatched = {
        "candidate_id": "cand-mismatch",
        "min_slice_delta_pnl": 0.4,
        "max_slice_delta_max_drawdown": -0.1,
        "total_delta_pnl": 1.2,
        "total_delta_exec": 3,
        "total_delta_rej": -2,
        "total_delta_max_drawdown": -0.1,
        "settle_mismatch_slice_count": 1,
    }

    module.apply_constraint_flags(
        matched,
        min_slice_delta_pnl_threshold=0.0,
        max_slice_delta_max_drawdown_threshold=0.0,
        enforce_settle_profile_match=True,
    )
    module.apply_constraint_flags(
        mismatched,
        min_slice_delta_pnl_threshold=0.0,
        max_slice_delta_max_drawdown_threshold=0.0,
        enforce_settle_profile_match=True,
    )

    assert bool(matched["passes_settle_profile_match"]) is True
    assert bool(matched["passes_constraints"]) is True
    assert bool(mismatched["passes_settle_profile_match"]) is False
    assert bool(mismatched["passes_constraints"]) is False

    ranked = sorted([mismatched, matched], key=module.candidate_sort_key)
    assert [row["candidate_id"] for row in ranked] == ["cand-match", "cand-mismatch"]


def test_summarize_compare_payload_tracks_settle_mismatch_rate_and_labels() -> None:
    module = _load_module()

    payload = {
        "slices": [
            {
                "label": "recent24h",
                "baseline": {"settle_window_end": True, "settle_window_end_source": "cycle_meta"},
                "candidate": {
                    "settle_window_end": True,
                    "settle_window_end_source": "cycle_meta",
                },
                "delta": {"s10": {"pnl": 0.1}},
            },
            {
                "label": "recent7d",
                "baseline": {"settle_window_end": True, "settle_window_end_source": "cycle_meta"},
                "candidate": {
                    "settle_window_end": False,
                    "settle_window_end_source": "execution_config",
                },
                "delta": {"s10": {"pnl": 0.0}},
            },
            {
                "label": "recent14d",
                "baseline": {"settle_window_end": True, "settle_window_end_source": "cycle_meta"},
                "candidate": {
                    "settle_window_end": False,
                    "settle_window_end_source": "execution_config",
                },
                "delta": {"s10": {"pnl": -0.1}},
            },
        ]
    }

    summary = module.summarize_compare_payload(payload, objective_strategy="s10")
    assert int(summary["slice_count"]) == 3
    assert int(summary["settle_mismatch_slice_count"]) == 2
    assert abs(float(summary["settle_mismatch_slice_rate"]) - (2.0 / 3.0)) < 1e-12
    assert summary["settle_mismatch_slice_labels"] == ["recent7d", "recent14d"]


def test_summarize_compare_payload_and_markdown() -> None:
    module = _load_module()

    payload = {
        "slices": [
            {
                "label": "recent24h",
                "hours": 24,
                "baseline": {
                    "by_strategy": {"s10": {"generation_top_reject_event": "evt-old-a:4"}}
                },
                "candidate": {
                    "by_strategy": {"s10": {"generation_top_reject_event": "evt-new-a:2"}}
                },
                "delta": {
                    "s10": {
                        "pnl": 0.6,
                        "max_drawdown": -0.2,
                        "executed_rows": 3,
                        "rejected_rows": -1,
                        "mtm_winrate": 0.1,
                        "generation_pass": 5,
                        "generation_rejected_candidates": -4,
                        "generation_top_reject_event_count": -2,
                        "generation_top_reject_event_shift": 1,
                    }
                },
            },
            {
                "label": "recent7d",
                "hours": 168,
                "baseline": {
                    "by_strategy": {"s10": {"generation_top_reject_event": "evt-stable:3"}}
                },
                "candidate": {
                    "by_strategy": {"s10": {"generation_top_reject_event": "evt-stable:2"}}
                },
                "delta": {
                    "s10": {
                        "pnl": -0.1,
                        "max_drawdown": -0.1,
                        "executed_rows": 1,
                        "rejected_rows": -2,
                        "mtm_winrate": -0.02,
                        "generation_pass": 2,
                        "generation_rejected_candidates": -3,
                        "generation_top_reject_event_count": -1,
                        "generation_top_reject_event_shift": 0,
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
    assert int(summary["total_delta_generation_pass"]) == 7
    assert int(summary["total_delta_generation_rejected_candidates"]) == -7
    assert int(summary["total_delta_generation_top_reject_event_count"]) == -3
    assert int(summary["generation_top_reject_event_shift_slices"]) == 1
    assert str(summary["baseline_settle_window_end_profile"]) == "all_off"
    assert str(summary["candidate_settle_window_end_profile"]) == "all_off"
    assert str(summary["baseline_settle_window_end_source_profile"]) == "unknown"
    assert str(summary["candidate_settle_window_end_source_profile"]) == "unknown"
    assert int(summary["settle_mismatch_slice_count"]) == 0
    assert abs(float(summary["settle_mismatch_slice_rate"]) - 0.0) < 1e-12
    assert summary["settle_mismatch_slice_labels"] == []

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
            "same_source_rollup": [
                {
                    "require_same_source": True,
                    "candidate_count": 1,
                    "pass_count": 1,
                    "pass_rate": 1.0,
                    "avg_total_delta_pnl": 0.5,
                    "avg_min_slice_delta_pnl": 0.1,
                    "avg_total_delta_exec": 4.0,
                    "avg_total_delta_rej": -3.0,
                    "best_candidate_id": "cand-001",
                    "best_total_delta_pnl": 0.5,
                    "best_min_slice_delta_pnl": 0.1,
                    "best_pass_candidate_id": "cand-001",
                    "best_pass_total_delta_pnl": 0.5,
                }
            ],
            "candidates": [
                {
                    "rank": 1,
                    "candidate_id": "cand-001",
                    "overrides": {
                        "prob_sum_tolerance": 0.015,
                        "max_abs_deviation": 0.15,
                        "max_tiny_price_leg_share": 0.2,
                        "max_floor_adjusted_leg_share": 0.25,
                        "require_same_source": True,
                    },
                    "min_slice_delta_pnl": 0.1,
                    "max_slice_delta_max_drawdown": -0.1,
                    "passes_constraints": True,
                    "total_delta_pnl": 0.5,
                    "total_delta_exec": 4,
                    "total_delta_rej": -3,
                    "total_delta_generation_pass": 7,
                    "total_delta_generation_rejected_candidates": -7,
                    "total_delta_generation_top_reject_event_count": -3,
                    "generation_top_reject_event_shift_slices": 1,
                    "total_delta_max_drawdown": -0.2,
                    "total_delta_mtm_winrate": 0.08,
                }
            ],
        }
    )

    assert "# S10 Parameter Grid Compare" in markdown
    assert "| rank | candidate | prob_tol |" in markdown
    assert (
        "| rank | candidate | prob_tol | max_abs | tiny_share | floor_share | same_source | base_settle | cand_settle | settle_mismatch | mismatch_rate | mismatch_slices | pass_settle? | min(Δpnl) | max(ΔmaxDD) |"
        in markdown
    )
    assert (
        "| 1 | cand-001 | 0.0150 | 0.1500 | 0.2000 | 0.2500 | true | unknown | unknown | +0 | 0.00% | none | yes | +0.1000 | -0.1000 | +0.5000 | +4 | -3 | +7 | -7 | -3 | +1 | -0.2000 | +0.0800 | yes |"
        in markdown
    )
    assert "## Same-source guard rollup" in markdown
    assert (
        "| true | 1 | 1 | 100.00% | +0.5000 | +0.1000 | +4.00 | -3.00 | cand-001 | +0.5000 | +0.1000 | cand-001 | +0.5000 |"
        in markdown
    )
