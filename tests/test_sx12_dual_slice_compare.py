from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

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
    assert bool(args.skip_ingest_rebuild) is False
    assert bool(args.baseline_no_settle_window_end) is False
    assert bool(args.candidate_no_settle_window_end) is False
    assert bool(args.strategy_config_diff_only) is False


def test_slice_parser_accepts_settle_override_flags() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(
        [
            "--baseline-config",
            "base.yaml",
            "--candidate-config",
            "cand.yaml",
            "--baseline-no-settle-window-end",
            "--candidate-no-settle-window-end",
        ]
    )

    assert bool(args.baseline_no_settle_window_end) is True
    assert bool(args.candidate_no_settle_window_end) is True


def test_slice_parser_accepts_strategy_config_diff_only_flag() -> None:
    module = _load_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(
        [
            "--baseline-config",
            "base.yaml",
            "--candidate-config",
            "cand.yaml",
            "--strategy-config-diff-only",
        ]
    )

    assert bool(args.strategy_config_diff_only) is True


def test_prepare_isolated_config_copies_db_and_rewrites_config(tmp_path: Path) -> None:
    module = _load_module()

    source_db = tmp_path / "data" / "mono.db"
    source_db.parent.mkdir(parents=True, exist_ok=True)
    source_db.write_text("db-seed")

    source_cfg = tmp_path / "config.yaml"
    source_cfg.write_text("app:\n  db_path: data/mono.db\n")

    run_dir = tmp_path / "run"
    isolated_cfg = module.prepare_isolated_config(
        source_config_path=source_cfg,
        run_dir=run_dir,
        config_tag="baseline",
    )

    assert isolated_cfg.exists()
    payload = yaml.safe_load(isolated_cfg.read_text())
    assert isinstance(payload, dict)
    app = payload.get("app", {})
    assert isinstance(app, dict)
    isolated_db_path = Path(str(app.get("db_path", "")))
    assert isolated_db_path.exists()
    assert isolated_db_path.read_text() == "db-seed"
    assert isolated_db_path.parent == run_dir / "db"


def test_build_backtest_cycle_cmd_includes_rebuild_flags(tmp_path: Path) -> None:
    module = _load_module()

    cmd = module._build_backtest_cycle_cmd(
        config_path=tmp_path / "cfg.yaml",
        from_ts="2026-03-07T00:00:00Z",
        to_ts="2026-03-07T01:00:00Z",
        output_dir=tmp_path / "out",
        rebuild_step_hours=6.0,
        rebuild_market_limit=120,
        rebuild_ingest_limit=40,
        skip_ingest=True,
        settle_window_end=False,
    )

    cmd_str = " ".join(str(x) for x in cmd)
    assert "scripts/backtest_cycle.sh" in cmd_str
    assert "--clear-signals-window" in cmd
    assert "--rebuild-signals-window" in cmd
    assert "--rebuild-step-hours" in cmd
    assert "--ingest-limit" in cmd
    assert "--market-limit" in cmd
    assert "--skip-ingest" in cmd
    assert "--no-settle-window-end" in cmd


def test_extract_settle_window_end_prefers_cycle_meta_then_execution_config() -> None:
    module = _load_module()

    settle_from_cycle_meta = module._extract_settle_window_end(
        {"execution_config": {"settle_window_end_positions": False}},
        cycle_meta={"signal_generation": {"settle_window_end": True}},
    )
    assert settle_from_cycle_meta == (
        True,
        "cycle_meta.signal_generation.settle_window_end",
    )

    settle_from_execution_config = module._extract_settle_window_end(
        {"execution_config": {"settle_window_end_positions": True}},
        cycle_meta=None,
    )
    assert settle_from_execution_config == (
        True,
        "report.execution_config.settle_window_end_positions",
    )

    settle_default = module._extract_settle_window_end({}, cycle_meta=None)
    assert settle_default == (False, "default(false)")


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
    assert int(summary["generation_raw"]) == 0
    assert int(summary["generation_pass"]) == 0
    assert int(summary["generation_fail"]) == 0
    assert int(summary["generation_rejected_candidates"]) == 0
    assert str(summary["generation_top_reject_reason"]) == "none"
    assert str(summary["generation_top_reject_event"]) == "none"
    assert int(summary["generation_top_reject_event_count"]) == 0
    assert str(summary["generation_top_reject_event_reason"]) == "none"


def test_summarize_strategy_extracts_generation_from_cycle_meta() -> None:
    module = _load_module()

    summary = module.summarize_strategy(
        {"results": [], "replay": []},
        strategy="s10",
        cycle_meta={
            "signal_generation": {
                "edge_gate": {
                    "by_strategy": {
                        "s10": {
                            "raw": 7,
                            "pass": 3,
                            "fail": 4,
                            "pass_rate": 3.0 / 7.0,
                            "strategy_diagnostics": {
                                "candidate_reject_reasons": {
                                    "buy_conversion:effective_edge_below_min": 5,
                                    "event_no_actionable_candidate": 2,
                                },
                                "candidate_reject_reasons_by_event_top": {
                                    "evt-alpha": {
                                        "buy_conversion:effective_edge_below_min": 4,
                                        "event_no_actionable_candidate": 1,
                                    },
                                    "evt-beta": {
                                        "event_no_actionable_candidate": 2,
                                    },
                                },
                            },
                        }
                    }
                }
            }
        },
    )

    assert int(summary["generation_raw"]) == 7
    assert int(summary["generation_pass"]) == 3
    assert int(summary["generation_fail"]) == 4
    assert abs(float(summary["generation_pass_rate"]) - (3.0 / 7.0)) < 1e-12
    assert int(summary["generation_rejected_candidates"]) == 7
    assert (
        str(summary["generation_top_reject_reason"]) == "buy_conversion:effective_edge_below_min:5"
    )
    assert str(summary["generation_top_reject_event"]) == "evt-alpha:5"
    assert int(summary["generation_top_reject_event_count"]) == 5
    assert (
        str(summary["generation_top_reject_event_reason"])
        == "evt-alpha|buy_conversion:effective_edge_below_min:4"
    )


def test_extract_strategy_config_context_only_keeps_focus_keys() -> None:
    module = _load_module()

    context = module._extract_strategy_config_context(
        config_payload={
            "strategies": {
                "s9": {
                    "min_effective_edge_bps": 20.0,
                    "require_same_market": True,
                    "unknown_key": "ignored",
                },
                "s10": {
                    "convert_value": 1.02,
                    "conversion_fee_bps": 12.0,
                    "max_weighted_total_cost_bps": 120.0,
                    "list_value": [1, 2, 3],
                },
                "s1": {
                    "max_signals": 10,
                },
            }
        },
        strategies=["s9", "s10", "s1"],
    )

    assert context == {
        "s9": {
            "min_effective_edge_bps": 20.0,
            "require_same_market": True,
        },
        "s10": {
            "convert_value": 1.02,
            "conversion_fee_bps": 12.0,
            "max_weighted_total_cost_bps": 120.0,
        },
    }


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
            "generation_pass": 1,
            "generation_rejected_candidates": 5,
            "generation_top_reject_reason": "effective edge below min:4",
            "generation_top_reject_event": "evt-old:4",
            "generation_top_reject_event_count": 4,
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
            "generation_pass": 4,
            "generation_rejected_candidates": 2,
            "generation_top_reject_reason": "none",
            "generation_top_reject_event": "evt-new:2",
            "generation_top_reject_event_count": 2,
        }
    }
    delta = module._summary_delta(baseline, candidate, strategies=["s9"])
    assert abs(float(delta["s9"]["pnl"]) - 0.6) < 1e-12
    assert abs(float(delta["s9"]["max_drawdown"]) + 0.3) < 1e-12
    assert int(delta["s9"]["executed_rows"]) == 2
    assert int(delta["s9"]["rejected_rows"]) == -1
    assert int(delta["s9"]["generation_pass"]) == 3
    assert int(delta["s9"]["generation_rejected_candidates"]) == -3
    assert int(delta["s9"]["generation_top_reject_event_count"]) == -2
    assert int(delta["s9"]["generation_top_reject_event_shift"]) == 1

    rendered = module.render_markdown(
        {
            "generated_at": "2026-03-07T17:40:00Z",
            "anchor_ts": "2026-03-07T17:39:00Z",
            "baseline_config": "/tmp/base.yaml",
            "candidate_config": "/tmp/candidate.yaml",
            "baseline_strategy_config": {
                "s9": {
                    "min_effective_edge_bps": 20.0,
                    "require_same_market": True,
                }
            },
            "candidate_strategy_config": {
                "s9": {
                    "min_effective_edge_bps": 35.0,
                    "require_same_market": False,
                }
            },
            "strategies": ["s9"],
            "slices": [
                {
                    "label": "recent24h",
                    "hours": 24,
                    "baseline": {
                        "settle_window_end": True,
                        "settle_window_end_source": "cycle_meta.signal_generation.settle_window_end",
                        "by_strategy": baseline,
                    },
                    "candidate": {
                        "settle_window_end": False,
                        "settle_window_end_source": "report.execution_config.settle_window_end_positions",
                        "by_strategy": candidate,
                    },
                    "delta": delta,
                }
            ],
        }
    )
    assert "# Sx12 Dual-Slice Compare" in rendered
    assert "## recent24h (24h)" in rendered
    assert (
        "- settle_window_end: base=true (cycle_meta.signal_generation.settle_window_end), "
        "cand=false (report.execution_config.settle_window_end_positions)"
    ) in rendered
    assert "## Strategy config context" in rendered
    assert "- s9 baseline: min_effective_edge_bps=20, require_same_market=true" in rendered
    assert "- s9 candidate: min_effective_edge_bps=35, require_same_market=false" in rendered
    assert "- s9 diff: min_effective_edge_bps:20->35, require_same_market:true->false" in rendered
    assert "| s9 | -0.5000 | 0.1000 | +0.6000 |" in rendered
    assert "| 1 | 4 | +3 | 5 | 2 | -3 |" in rendered
    assert (
        "| none | none | effective edge below min:4 | none | evt-old:4 | evt-new:2 | -2 | +1 |"
        in rendered
    )


def test_render_markdown_strategy_config_diff_only_filters_unchanged_keys() -> None:
    module = _load_module()

    rendered = module.render_markdown(
        {
            "generated_at": "2026-03-08T07:00:00Z",
            "anchor_ts": "2026-03-08T06:59:59Z",
            "baseline_config": "/tmp/base.yaml",
            "candidate_config": "/tmp/candidate.yaml",
            "strategies": ["s9", "s10"],
            "strategy_config_diff_only": True,
            "baseline_strategy_config": {
                "s9": {
                    "min_effective_edge_bps": 20.0,
                    "require_same_market": True,
                },
                "s10": {
                    "convert_value": 1.0,
                    "conversion_fee_bps": 0.0,
                },
            },
            "candidate_strategy_config": {
                "s9": {
                    "min_effective_edge_bps": 35.0,
                    "require_same_market": True,
                },
                "s10": {
                    "convert_value": 1.0,
                    "conversion_fee_bps": 0.0,
                },
            },
            "slices": [],
        }
    )

    assert "- strategy_config_diff_only: True" in rendered
    assert "## Strategy config context" in rendered
    assert "- s9 baseline: min_effective_edge_bps=20" in rendered
    assert "- s9 candidate: min_effective_edge_bps=35" in rendered
    assert "- s9 diff: min_effective_edge_bps:20->35" in rendered
    assert "require_same_market" not in rendered
    assert "- s10 baseline:" not in rendered
    assert "- no strategy config diffs" not in rendered
