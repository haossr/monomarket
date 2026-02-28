from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from monomarket.backtest import (
    NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO,
    compute_nightly_summary_sidecar_checksum,
    validate_nightly_summary_sidecar,
    verify_nightly_summary_sidecar_checksum,
)
from monomarket.cli import app

ROOT = Path(__file__).resolve().parents[1]
BASH_SCRIPT_PATH = ROOT / "scripts" / "backtest_nightly_report.sh"
SUMMARY_SCRIPT_PATH = ROOT / "scripts" / "nightly_summary_line.py"
PDF_SCRIPT_PATH = ROOT / "scripts" / "backtest_pdf_report.py"


def _load_pdf_report_module() -> Any:
    spec = importlib.util.spec_from_file_location("backtest_pdf_report", PDF_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load backtest_pdf_report module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_summary_contains_canonical_alias_fields() -> None:
    content = SUMMARY_SCRIPT_PATH.read_text()

    required_tokens = [
        "closed_winrate=",
        "closed_samples=",
        "mtm_winrate=",
        "mtm_samples=",
        "main_coverage=",
        "history_limited=",
        "window_note=",
        "fixed_window=",
        "generated_signals=",
        "generated_in_window=",
        "clear_signals_window=",
        "cleared_signals_in_window=",
        "rebuild_signals_window=",
        "rebuild_step_h=",
        "rebuild_sampled_steps=",
        "generated_share=",
        "generated_span_h=",
        "generated_window_coverage=",
        "generated_low_influence=",
        "generated_low_sample_count=",
        "generated_low_temporal_coverage=",
        "historical_replay_only=",
        "experiment_interpretable=",
        "experiment_reason=",
        "negative_strategies=",
        "worst_negative_strategy=",
        "worst_negative_pnl=",
        "negative_events=",
        "negative_event_unique_count=",
        "negative_event_source=",
        "worst_negative_event=",
        "worst_negative_event_strategy=",
        "worst_negative_event_pnl=",
        "rolling_negative_strategies=",
        "rolling_worst_negative_strategy=",
        "rolling_worst_avg_pnl=",
        "rolling_active_negative_strategies=",
        "rolling_active_worst_negative_strategy=",
        "rolling_active_worst_avg_pnl=",
        "rolling_total_signals=",
        "rolling_executed_signals=",
        "rolling_rejected_signals=",
        "positive_window_rate=",
        "empty_window_count=",
        "range_hours=",
        "coverage_ratio=",
        "overlap_ratio=",
        "rolling_reject_top_k=",
        "rolling_reject_top_normalized=",
        "rolling_reject_top_effective=",
        "rolling_reject_top_effective_primary_reason=",
        "rolling_reject_top_effective_primary_count=",
        "reject_strategy_top=",
        "reject_strategy_top_reason=",
        "reject_strategy_top_rejected=",
        "reject_strategy_top_share=",
        "best_strategy_basis=",
    ]
    for token in required_tokens:
        assert token in content


def test_pdf_report_includes_main_window_coverage_section_tokens() -> None:
    content = PDF_SCRIPT_PATH.read_text()
    required_tokens = [
        "Closed winrate summary",
        "MTM winrate summary",
        "Main window coverage",
        "Main history limited",
        "Main window note",
        "Rolling execution rate",
        "Rolling empty windows",
        "Rolling coverage label",
        "Rolling reject top",
        "Interpretation note: main-window metrics are history-limited",
        "Interpretation note: no replay rows in main window",
        "PDF_ROLLING_REJECT_TOP_K = 2",
        "def _load_payload_results_rows",
        "winrate_source_rows = _load_payload_results_rows(payload) or strategy_rows",
        "def _extract_rolling_summary",
        "from monomarket.backtest.reject_reason import format_reject_top",
        "def _format_rolling_reject_top",
    ]
    for token in required_tokens:
        assert token in content


def test_pdf_format_rolling_reject_top_none_and_top2() -> None:
    module = _load_pdf_report_module()
    fmt = module._format_rolling_reject_top

    assert fmt({}) == "none"
    assert fmt({"risk_rejection_reasons": {}}) == "none"

    actual = fmt(
        {
            "risk_rejection_reasons": {
                "reason-a": 3,
                "reason-b": 2,
                "reason-c": 1,
            }
        }
    )
    assert actual == "reason-a:3;reason-b:2"

    normalized = fmt(
        {
            "risk_rejection_reasons": {
                "strategy notional limit exceeded: 1019.19 > 1000.00": 3,
                "strategy notional limit exceeded: 1019.21 > 1000.00": 2,
                "circuit breaker open: rejected=10, threshold=5": 1,
            }
        }
    )
    assert normalized == "strategy notional limit exceeded:5;circuit breaker open:1"


def test_pdf_format_rolling_reject_top_tie_is_stable() -> None:
    module = _load_pdf_report_module()
    fmt = module._format_rolling_reject_top

    actual = fmt(
        {
            "risk_rejection_reasons": {
                "z-reason": 5,
                "a-reason": 5,
                "m-reason": 4,
            }
        }
    )
    assert actual == "a-reason:5;z-reason:5"


def test_nightly_best_strategy_na_when_no_executed_signals(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 10,
        "executed_signals": 0,
        "rejected_signals": 10,
        # keep non-empty results to ensure gating is based on executed_signals
        "results": [{"strategy": "s1", "pnl": 1.2}],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 3,
            "execution_rate": 0.0,
            "positive_window_rate": 0.0,
            "empty_window_count": 3,
            "range_hours": 24,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.2,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "best_strategy=n/a" in line
    assert "negative_strategies=0" in line
    assert "worst_negative_strategy=n/a" in line
    assert "negative_event_source=missing" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    best_obj = sidecar["best"]
    assert isinstance(best_obj, dict)
    assert bool(best_obj["available"]) is False
    assert str(best_obj["strategy"]) == ""
    assert float(best_obj["pnl"]) == 0.0
    assert str(best_obj["selection_basis"]) == "none"
    assert int(best_obj["candidate_count"]) == 0
    assert int(best_obj["active_candidate_count"]) == 0
    assert str(best_obj["text"]) == "best_strategy=n/a"
    assert str(sidecar["best_text"]) == "best_strategy=n/a"

    negative_obj = sidecar["negative_strategies"]
    assert isinstance(negative_obj, dict)
    assert int(negative_obj["count"]) == 0
    assert str(negative_obj["worst_strategy"]) == ""
    assert float(negative_obj["worst_pnl"]) == 0.0

    negative_event_obj = sidecar["negative_events"]
    assert isinstance(negative_event_obj, dict)
    assert int(negative_event_obj["count"]) == 0
    assert int(negative_event_obj["unique_count"]) == 0
    assert bool(negative_event_obj["source_present"]) is False
    assert str(negative_event_obj["worst_event_id"]) == ""
    assert str(negative_event_obj["worst_strategy"]) == ""
    assert float(negative_event_obj["worst_pnl"]) == 0.0

    rolling_obj = sidecar["rolling"]
    assert str(rolling_obj["reject_top_effective"]) == "none"
    assert str(rolling_obj["reject_top_effective_primary_reason"]) == "none"
    assert float(rolling_obj["reject_top_effective_primary_count"]) == 0.0

    rolling_negative_obj = sidecar["rolling"]["negative_strategies"]
    assert isinstance(rolling_negative_obj, dict)
    assert int(rolling_negative_obj["count"]) == 0
    assert str(rolling_negative_obj["worst_strategy"]) == ""
    assert float(rolling_negative_obj["worst_avg_pnl"]) == 0.0

    rolling_negative_active_obj = sidecar["rolling"]["negative_strategies_active"]
    assert isinstance(rolling_negative_active_obj, dict)
    assert int(rolling_negative_active_obj["count"]) == 0
    assert str(rolling_negative_active_obj["worst_strategy"]) == ""
    assert float(rolling_negative_active_obj["worst_avg_pnl"]) == 0.0


def test_nightly_best_strategy_prefers_active_strategies(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 4,
        "executed_signals": 4,
        "rejected_signals": 0,
        "results": [
            {"strategy": "s1", "pnl": 0.0, "trades": 0},
            {"strategy": "s2", "pnl": 0.0, "trades": 0},
            {"strategy": "s8", "pnl": -0.25, "trades": 4, "mtm_wins": 0, "mtm_losses": 4},
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 2,
            "execution_rate": 1.0,
            "positive_window_rate": 0.0,
            "empty_window_count": 0,
            "range_hours": 24,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.5,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "best_strategy=s8 pnl=-0.2500" in line
    assert "best_strategy_basis=active_first" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    best_obj = sidecar["best"]
    assert isinstance(best_obj, dict)
    assert str(best_obj["strategy"]) == "s8"
    assert abs(float(best_obj["pnl"]) - (-0.25)) < 1e-9
    assert str(best_obj["selection_basis"]) == "active_first"
    assert int(best_obj["candidate_count"]) == 3
    assert int(best_obj["active_candidate_count"]) == 1


def test_nightly_summary_reports_negative_strategy_metadata(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 10,
        "executed_signals": 10,
        "rejected_signals": 0,
        "results": [
            {"strategy": "s1", "pnl": -1.25, "mtm_wins": 0, "mtm_losses": 1},
            {"strategy": "s8", "pnl": 0.75, "mtm_wins": 1, "mtm_losses": 0},
        ],
        "event_results": [
            {
                "strategy": "s1",
                "event_id": 111,
                "pnl": -1.25,
                "trade_count": 1,
                "mtm_wins": 0,
                "mtm_losses": 1,
            },
            {
                "strategy": "s8",
                "event_id": 222,
                "pnl": 0.75,
                "trade_count": 1,
                "mtm_wins": 1,
                "mtm_losses": 0,
            },
        ],
        "replay": [{"ts": "2026-02-24T00:00:00Z", "realized_change": 0.0}],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 1.0,
            "positive_window_rate": 1.0,
            "empty_window_count": 0,
            "range_hours": 2,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "negative_strategies=1" in line
    assert "worst_negative_strategy=s1" in line
    assert "worst_negative_pnl=-1.2500" in line
    assert "negative_events=1" in line
    assert "negative_event_unique_count=1" in line
    assert "negative_event_source=present" in line
    assert "worst_negative_event=111" in line
    assert "worst_negative_event_strategy=s1" in line
    assert "worst_negative_event_pnl=-1.2500" in line
    assert "rolling_negative_strategies=0" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    negative_obj = sidecar["negative_strategies"]
    assert int(negative_obj["count"]) == 1
    assert str(negative_obj["worst_strategy"]) == "s1"
    assert float(negative_obj["worst_pnl"]) == -1.25

    negative_event_obj = sidecar["negative_events"]
    assert isinstance(negative_event_obj, dict)
    assert int(negative_event_obj["count"]) == 1
    assert int(negative_event_obj["unique_count"]) == 1
    assert bool(negative_event_obj["source_present"]) is True
    assert str(negative_event_obj["worst_event_id"]) == "111"
    assert str(negative_event_obj["worst_strategy"]) == "s1"
    assert float(negative_event_obj["worst_pnl"]) == -1.25


def test_nightly_summary_reports_rolling_negative_strategy_metadata(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 10,
        "executed_signals": 10,
        "rejected_signals": 0,
        "results": [{"strategy": "s1", "pnl": 0.5, "mtm_wins": 1, "mtm_losses": 0}],
        "replay": [{"ts": "2026-02-24T00:00:00Z", "realized_change": 0.0}],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 2,
            "execution_rate": 1.0,
            "positive_window_rate": 0.5,
            "empty_window_count": 0,
            "range_hours": 24,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.5,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        },
        "strategy_aggregate": [
            {
                "strategy": "s1",
                "windows": 2,
                "active_windows": 2,
                "active_window_rate": 1.0,
                "avg_pnl": 0.2,
            },
            {
                "strategy": "s2",
                "windows": 2,
                "active_windows": 2,
                "active_window_rate": 1.0,
                "avg_pnl": -0.1,
            },
        ],
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "rolling_negative_strategies=1" in line
    assert "rolling_worst_negative_strategy=s2" in line
    assert "rolling_worst_avg_pnl=-0.1000" in line
    assert "rolling_active_negative_strategies=1" in line
    assert "rolling_active_worst_negative_strategy=s2" in line
    assert "rolling_active_worst_avg_pnl=-0.1000" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    rolling_negative_obj = sidecar["rolling"]["negative_strategies"]
    assert int(rolling_negative_obj["count"]) == 1
    assert str(rolling_negative_obj["worst_strategy"]) == "s2"
    assert float(rolling_negative_obj["worst_avg_pnl"]) == -0.1

    rolling_negative_active_obj = sidecar["rolling"]["negative_strategies_active"]
    assert int(rolling_negative_active_obj["count"]) == 1
    assert str(rolling_negative_active_obj["worst_strategy"]) == "s2"
    assert float(rolling_negative_active_obj["worst_avg_pnl"]) == -0.1


def test_nightly_summary_reports_closed_and_mtm_winrate(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 10,
        "executed_signals": 8,
        "rejected_signals": 2,
        "results": [
            {
                "strategy": "s1",
                "pnl": 1.2,
                "wins": 2,
                "losses": 1,
                "mtm_wins": 3,
                "mtm_losses": 1,
            },
            {
                "strategy": "s2",
                "pnl": 0.8,
                "wins": 1,
                "losses": 2,
                "mtm_wins": 1,
                "mtm_losses": 3,
            },
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 3,
            "execution_rate": 0.8,
            "positive_window_rate": 0.66,
            "empty_window_count": 1,
            "range_hours": 24,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.2,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "closed_winrate=50.00%" in line
    assert "closed_samples=6" in line
    assert "mtm_winrate=50.00%" in line
    assert "mtm_samples=8" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    winrate = sidecar["winrate"]
    assert isinstance(winrate, dict)
    assert abs(float(winrate["closed_winrate"]) - 0.5) < 1e-9
    assert int(winrate["closed_sample_count"]) == 6
    assert int(winrate["closed_wins"]) == 3
    assert int(winrate["closed_losses"]) == 3
    assert abs(float(winrate["mtm_winrate"]) - 0.5) < 1e-9
    assert int(winrate["mtm_sample_count"]) == 8
    assert int(winrate["mtm_wins"]) == 4
    assert int(winrate["mtm_losses"]) == 4


def test_nightly_reject_by_strategy_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 6,
        "executed_signals": 2,
        "rejected_signals": 4,
        "results": [{"strategy": "s4", "pnl": -1.0}],
        "replay": [
            {
                "strategy": "s4",
                "risk_allowed": False,
                "risk_reason": "strategy notional limit exceeded: 1010 > 1000",
            },
            {
                "strategy": "s4",
                "risk_allowed": False,
                "risk_reason": "strategy notional limit exceeded: 1020 > 1000",
            },
            {"strategy": "s4", "risk_allowed": True, "risk_reason": "ok"},
            {
                "strategy": "s8",
                "risk_allowed": False,
                "risk_reason": "circuit breaker open: 3 >= 3",
            },
            {"strategy": "s8", "risk_allowed": True, "risk_reason": "ok"},
            {"strategy": "s1", "risk_allowed": False, "risk_reason": "custom reason"},
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 0.1,
            "positive_window_rate": 0.0,
            "empty_window_count": 0,
            "range_hours": 2,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "reject_strategy_top=s4:2;s1:1;s8:1" in line
    assert "reject_strategy_top_reason=strategy notional limit exceeded:2" in line
    assert "reject_strategy_top_rejected=2" in line
    assert "reject_strategy_top_share=50.00%" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    reject_by_strategy = sidecar["reject_by_strategy"]
    assert isinstance(reject_by_strategy, dict)
    assert reject_by_strategy["top"] == "s4:2;s1:1;s8:1"
    assert reject_by_strategy["top_reason"] == "strategy notional limit exceeded:2"
    assert int(reject_by_strategy["top_rejected"]) == 2
    assert abs(float(reject_by_strategy["top_share"]) - 0.5) < 1e-12
    rows = reject_by_strategy["rows"]
    assert isinstance(rows, list)
    assert rows[0]["strategy"] == "s4"
    assert rows[0]["top_reason"] == "strategy notional limit exceeded:2"
    assert rows[1]["strategy"] == "s1"
    assert rows[1]["top_reason"] == "custom reason:1"
    assert rows[2]["strategy"] == "s8"
    assert rows[2]["top_reason"] == "circuit breaker open:1"


def test_nightly_cycle_meta_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    cycle_meta_json = tmp_path / "cycle-meta.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 2,
        "executed_signals": 1,
        "rejected_signals": 1,
        "results": [{"strategy": "s1", "pnl": 1.0}],
        "replay": [{"strategy": "s1", "risk_allowed": False, "risk_reason": "x"}],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 0.5,
            "positive_window_rate": 0.0,
            "empty_window_count": 0,
            "range_hours": 2,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    cycle_meta_payload = {
        "fixed_window_mode": True,
        "signal_generation": {
            "new_signals_total": 82,
            "new_signals_in_window": 0,
            "new_signals_first_ts": "",
            "new_signals_last_ts": "",
            "clear_signals_window": False,
            "cleared_signals_in_window": 0,
            "rebuild_signals_window": False,
            "rebuild_step_hours": 12.0,
            "rebuild_sampled_steps": 0,
            "generated_share_of_total": 0.0,
            "generated_span_hours": 0.0,
            "generated_window_coverage_ratio": 0.0,
            "generated_low_influence": True,
            "generated_low_temporal_coverage": False,
            "historical_replay_only": True,
            "experiment_interpretable": False,
            "experiment_reason": "historical_replay_only",
        },
    }
    cycle_meta_json.write_text(json.dumps(cycle_meta_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--cycle-meta-json",
            str(cycle_meta_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "fixed_window=true" in line
    assert "generated_signals=82" in line
    assert "generated_in_window=0" in line
    assert "clear_signals_window=false" in line
    assert "cleared_signals_in_window=0" in line
    assert "rebuild_signals_window=false" in line
    assert "rebuild_step_h=12.00" in line
    assert "rebuild_sampled_steps=0" in line
    assert "generated_share=0.00%" in line
    assert "generated_span_h=0.00" in line
    assert "generated_window_coverage=0.00%" in line
    assert "generated_low_influence=true" in line
    assert "generated_low_sample_count=true" in line
    assert "generated_low_temporal_coverage=false" in line
    assert "historical_replay_only=true" in line
    assert "experiment_interpretable=false" in line
    assert "experiment_reason=historical_replay_only" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    cycle_meta = sidecar["cycle_meta"]
    assert isinstance(cycle_meta, dict)
    assert cycle_meta["fixed_window_mode"] is True
    signal_generation = cycle_meta["signal_generation"]
    assert isinstance(signal_generation, dict)
    assert int(signal_generation["new_signals_total"]) == 82
    assert int(signal_generation["new_signals_in_window"]) == 0
    assert signal_generation["new_signals_first_ts"] == ""
    assert signal_generation["new_signals_last_ts"] == ""
    assert signal_generation["clear_signals_window"] is False
    assert int(signal_generation["cleared_signals_in_window"]) == 0
    assert signal_generation["rebuild_signals_window"] is False
    assert abs(float(signal_generation["rebuild_step_hours"]) - 12.0) < 1e-9
    assert int(signal_generation["rebuild_sampled_steps"]) == 0
    assert abs(float(signal_generation["generated_share_of_total"])) < 1e-9
    assert abs(float(signal_generation["generated_span_hours"])) < 1e-9
    assert abs(float(signal_generation["generated_window_coverage_ratio"])) < 1e-9
    assert signal_generation["generated_low_influence"] is True
    assert signal_generation["generated_low_sample_count"] is True
    assert signal_generation["generated_low_temporal_coverage"] is False
    assert signal_generation["historical_replay_only"] is True
    assert signal_generation["experiment_interpretable"] is False
    assert signal_generation["experiment_reason"] == "historical_replay_only"


def test_nightly_cycle_meta_history_limited_does_not_trigger_low_temporal_coverage(
    tmp_path: Path,
) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    cycle_meta_json = tmp_path / "cycle-meta.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T10:00:00Z",
        "total_signals": 20,
        "executed_signals": 20,
        "rejected_signals": 0,
        "results": [
            {
                "strategy": "s8",
                "pnl": 1.0,
                "mtm_wins": 10,
                "mtm_losses": 10,
            }
        ],
        "replay": [
            {"ts": "2026-02-24T08:00:00Z", "realized_change": 0.0},
            {"ts": "2026-02-24T09:00:00Z", "realized_change": 0.0},
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 1.0,
            "positive_window_rate": 1.0,
            "empty_window_count": 0,
            "range_hours": 10,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    cycle_meta_payload = {
        "fixed_window_mode": True,
        "signal_generation": {
            "new_signals_total": 20,
            "new_signals_in_window": 20,
            "new_signals_first_ts": "2026-02-24T08:00:00Z",
            "new_signals_last_ts": "2026-02-24T09:00:00Z",
            "clear_signals_window": True,
            "cleared_signals_in_window": 20,
            "rebuild_signals_window": True,
            "rebuild_step_hours": 6.0,
            "rebuild_sampled_steps": 2,
            "historical_replay_only": False,
            "edge_gate": {"total_raw": 0, "total_pass": 0, "total_fail": 0, "pass_rate": 0.0},
        },
    }
    cycle_meta_json.write_text(json.dumps(cycle_meta_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--cycle-meta-json",
            str(cycle_meta_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "history_limited=true" in line
    assert "generated_low_temporal_coverage=false" in line
    assert "generated_low_sample_count=false" in line
    assert "experiment_interpretable=true" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    signal_generation = sidecar["cycle_meta"]["signal_generation"]
    assert signal_generation["generated_low_temporal_coverage"] is False
    assert signal_generation["generated_low_sample_count"] is False
    assert signal_generation["experiment_interpretable"] is True


def test_nightly_window_coverage_history_limited_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T10:00:00Z",
        "total_signals": 2,
        "executed_signals": 0,
        "rejected_signals": 2,
        "results": [],
        "replay": [
            {"ts": "2026-02-24T08:00:00Z", "realized_change": 0.0},
            {"ts": "2026-02-24T09:00:00Z", "realized_change": 0.0},
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 0.0,
            "positive_window_rate": 0.0,
            "empty_window_count": 1,
            "range_hours": 10,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "history_limited=true" in line
    assert "window_note=history_limited" in line
    assert "main_coverage=20.00%" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    window_coverage = sidecar["window_coverage"]
    assert isinstance(window_coverage, dict)
    assert bool(window_coverage["history_limited"]) is True
    assert str(window_coverage["note"]) == "history_limited"
    assert abs(float(window_coverage["window_hours"]) - 10.0) < 1e-9
    assert abs(float(window_coverage["covered_hours"]) - 2.0) < 1e-9
    assert abs(float(window_coverage["coverage_ratio"]) - 0.2) < 1e-9


def test_nightly_window_coverage_full_history_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T10:00:00Z",
        "total_signals": 2,
        "executed_signals": 0,
        "rejected_signals": 2,
        "results": [],
        "replay": [
            {"ts": "2026-02-24T00:00:00Z", "realized_change": 0.0},
            {"ts": "2026-02-24T09:00:00Z", "realized_change": 0.0},
        ],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 0.0,
            "positive_window_rate": 0.0,
            "empty_window_count": 1,
            "range_hours": 10,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "history_limited=false" in line
    assert "window_note=full_history" in line
    assert "main_coverage=100.00%" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    window_coverage = sidecar["window_coverage"]
    assert isinstance(window_coverage, dict)
    assert bool(window_coverage["history_limited"]) is False
    assert str(window_coverage["note"]) == "full_history"
    assert abs(float(window_coverage["coverage_ratio"]) - 1.0) < 1e-9


def test_nightly_window_coverage_no_replay_rows_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T10:00:00Z",
        "total_signals": 0,
        "executed_signals": 0,
        "rejected_signals": 0,
        "results": [],
        "replay": [],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    rolling_payload = {
        "summary": {
            "run_count": 1,
            "execution_rate": 0.0,
            "positive_window_rate": 0.0,
            "empty_window_count": 1,
            "range_hours": 10,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.0,
            "coverage_label": "full",
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload))

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    line = summary_txt.read_text().strip()
    assert "history_limited=false" in line
    assert "window_note=no_replay_rows" in line
    assert "main_coverage=0.00%" in line

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    window_coverage = sidecar["window_coverage"]
    assert isinstance(window_coverage, dict)
    assert bool(window_coverage["history_limited"]) is False
    assert str(window_coverage["note"]) == "no_replay_rows"
    assert abs(float(window_coverage["covered_hours"]) - 0.0) < 1e-9
    assert abs(float(window_coverage["coverage_ratio"]) - 0.0) < 1e-9
    assert str(window_coverage["effective_from_ts"]) == ""


def test_nightly_reject_topk_zero_disabled_and_none_runtime(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_payload = {
        "from_ts": "2026-02-24T00:00:00Z",
        "to_ts": "2026-02-24T02:00:00Z",
        "total_signals": 10,
        "executed_signals": 8,
        "rejected_signals": 2,
        "results": [{"strategy": "s1", "pnl": 1.2}],
    }
    backtest_json.write_text(json.dumps(backtest_payload))

    base_summary = {
        "run_count": 3,
        "execution_rate": 0.8,
        "positive_window_rate": 0.66,
        "empty_window_count": 1,
        "range_hours": 24,
        "coverage_ratio": 1.0,
        "overlap_ratio": 0.2,
        "coverage_label": "full",
    }

    # k=0 => disabled (even if reasons exist)
    rolling_payload_disabled = {
        "summary": {
            **base_summary,
            "risk_rejection_reasons": {"riskA": 3, "riskB": 1},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload_disabled))
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "0",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line_disabled = summary_txt.read_text().strip()
    assert "rolling_reject_top_k=0" in line_disabled
    assert "rolling_reject_top_delim=;" in line_disabled
    assert "rolling_reject_top=disabled" in line_disabled
    assert "rolling_reject_top_normalized=disabled" in line_disabled
    assert "rolling_reject_top_effective=disabled" in line_disabled
    assert "rolling_reject_top_effective_primary_reason=disabled" in line_disabled
    assert "rolling_reject_top_effective_primary_count=0" in line_disabled
    assert "positive_window_rate=" in line_disabled
    assert "empty_window_count=" in line_disabled
    assert "range_hours=" in line_disabled
    assert "coverage_ratio=" in line_disabled
    assert "overlap_ratio=" in line_disabled

    disabled_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(disabled_sidecar)
    assert str(disabled_sidecar["schema_version"]) == "nightly-summary-sidecar-1.0"
    assert str(disabled_sidecar["schema_note_version"]) == "1.0"
    assert str(disabled_sidecar["schema_note"]).startswith("best is structured object")
    assert str(disabled_sidecar["best_version"]) == "1.0"
    assert str(disabled_sidecar["checksum_algo"]) == NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
    assert verify_nightly_summary_sidecar_checksum(disabled_sidecar)
    assert int(disabled_sidecar["rolling"]["reject_top_k"]) == 0
    assert str(disabled_sidecar["rolling"]["reject_top_delimiter"]) == ";"
    assert str(disabled_sidecar["rolling"]["reject_top"]) == "disabled"
    assert str(disabled_sidecar["rolling"]["reject_top_normalized"]) == "disabled"
    assert str(disabled_sidecar["rolling"]["reject_top_effective"]) == "disabled"
    assert str(disabled_sidecar["rolling"]["reject_top_effective_primary_reason"]) == "disabled"
    assert float(disabled_sidecar["rolling"]["reject_top_effective_primary_count"]) == 0.0
    assert disabled_sidecar["rolling"]["reject_top_pairs_normalized"] == []
    best_obj = disabled_sidecar["best"]
    assert isinstance(best_obj, dict)
    assert str(best_obj["strategy"]) == "s1"
    assert abs(float(best_obj["pnl"]) - 1.2) < 1e-9
    assert str(best_obj["text"]).startswith("best_strategy=")
    assert "coverage_ratio" in disabled_sidecar["rolling"]
    assert "overlap_ratio" in disabled_sidecar["rolling"]

    # k>0 with no reasons => none
    rolling_payload_none = {
        "summary": {
            **base_summary,
            "risk_rejection_reasons": {},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload_none))
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line_none = summary_txt.read_text().strip()
    assert "rolling_reject_top_k=2" in line_none
    assert "rolling_reject_top_delim=;" in line_none
    assert "rolling_reject_top=none" in line_none
    assert "rolling_reject_top_normalized=none" in line_none
    assert "rolling_reject_top_effective=none" in line_none
    assert "rolling_reject_top_effective_primary_reason=none" in line_none
    assert "rolling_reject_top_effective_primary_count=0" in line_none

    none_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(none_sidecar)
    assert str(none_sidecar["checksum_algo"]) == NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
    assert verify_nightly_summary_sidecar_checksum(none_sidecar)
    assert int(none_sidecar["rolling"]["reject_top_k"]) == 2
    assert str(none_sidecar["rolling"]["reject_top_delimiter"]) == ";"
    assert str(none_sidecar["rolling"]["reject_top"]) == "none"
    assert str(none_sidecar["rolling"]["reject_top_normalized"]) == "none"
    assert str(none_sidecar["rolling"]["reject_top_effective"]) == "none"
    assert str(none_sidecar["rolling"]["reject_top_effective_primary_reason"]) == "none"
    assert float(none_sidecar["rolling"]["reject_top_effective_primary_count"]) == 0.0
    assert none_sidecar["rolling"]["reject_top_pairs_normalized"] == []

    # k>0 with reasons containing comma: delimiter must remain ';'
    rolling_payload_reasons = {
        "summary": {
            **base_summary,
            "risk_rejection_reasons": {"risk,A": 3, "riskB": 1},
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload_reasons))
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line_reasons = summary_txt.read_text().strip()
    assert "rolling_reject_top_delim=;" in line_reasons
    assert "rolling_reject_top=risk,A:3;riskB:1" in line_reasons
    assert "rolling_reject_top_normalized=risk,A:3;riskB:1" in line_reasons
    assert "rolling_reject_top_effective=risk,A:3;riskB:1" in line_reasons
    assert "rolling_reject_top_effective_primary_reason=risk,A" in line_reasons
    assert "rolling_reject_top_effective_primary_count=3" in line_reasons

    reasons_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(reasons_sidecar)
    assert verify_nightly_summary_sidecar_checksum(reasons_sidecar)

    rolling_norm = reasons_sidecar["rolling"]
    assert str(rolling_norm["reject_top_normalized"]) == "risk,A:3;riskB:1"
    assert str(rolling_norm["reject_top_effective_primary_reason"]) == "risk,A"
    assert float(rolling_norm["reject_top_effective_primary_count"]) == 3.0

    # normalization should collapse noisy numeric suffixes into family-level reasons
    rolling_payload_normalized = {
        "summary": {
            **base_summary,
            "risk_rejection_reasons": {
                "strategy notional limit exceeded: 1019.19 > 1000.00": 3,
                "strategy notional limit exceeded: 1019.21 > 1000.00": 2,
                "circuit breaker open: rejected=10, threshold=5": 1,
            },
        }
    }
    rolling_json.write_text(json.dumps(rolling_payload_normalized))
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
            "--with-checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line_normalized = summary_txt.read_text().strip()
    assert (
        "rolling_reject_top_normalized=" "strategy notional limit exceeded:5;circuit breaker open:1"
    ) in line_normalized
    assert (
        "rolling_reject_top_effective=" "strategy notional limit exceeded:5;circuit breaker open:1"
    ) in line_normalized
    assert (
        "rolling_reject_top_effective_primary_reason=strategy notional limit exceeded"
    ) in line_normalized
    assert "rolling_reject_top_effective_primary_count=5" in line_normalized

    normalized_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(normalized_sidecar)
    assert verify_nightly_summary_sidecar_checksum(normalized_sidecar)
    assert (
        str(normalized_sidecar["rolling"]["reject_top_effective"])
        == "strategy notional limit exceeded:5;circuit breaker open:1"
    )
    assert (
        str(normalized_sidecar["rolling"]["reject_top_effective_primary_reason"])
        == "strategy notional limit exceeded"
    )
    assert float(normalized_sidecar["rolling"]["reject_top_effective_primary_count"]) == 5.0
    rolling_norm_pairs = normalized_sidecar["rolling"]["reject_top_pairs_normalized"]
    assert isinstance(rolling_norm_pairs, list)
    assert rolling_norm_pairs == [
        {"reason": "strategy notional limit exceeded", "count": 5},
        {"reason": "circuit breaker open", "count": 1},
    ]


def test_nightly_summary_line_without_checksum_omits_checksum_fields(tmp_path: Path) -> None:
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_json.write_text(
        json.dumps(
            {
                "from_ts": "2026-02-24T00:00:00Z",
                "to_ts": "2026-02-24T02:00:00Z",
                "total_signals": 1,
                "executed_signals": 1,
                "rejected_signals": 0,
                "results": [{"strategy": "s1", "pnl": 1.0}],
            }
        )
    )
    rolling_json.write_text(
        json.dumps(
            {
                "summary": {
                    "run_count": 1,
                    "execution_rate": 1.0,
                    "positive_window_rate": 1.0,
                    "empty_window_count": 0,
                    "range_hours": 2.0,
                    "coverage_ratio": 1.0,
                    "overlap_ratio": 0.0,
                    "coverage_label": "full",
                    "risk_rejection_reasons": {},
                }
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT_PATH),
            "--backtest-json",
            str(backtest_json),
            "--pdf-path",
            str(pdf_path),
            "--rolling-json",
            str(rolling_json),
            "--summary-path",
            str(summary_txt),
            "--summary-json-path",
            str(summary_json),
            "--nightly-date",
            "2026-02-24",
            "--rolling-reject-top-k",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(payload)
    assert "checksum_algo" not in payload
    assert "checksum_sha256" not in payload


def _write_nightly_summary_sidecar(tmp_path: Path, *, with_checksum: bool) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    backtest_json = tmp_path / "latest.json"
    rolling_json = tmp_path / "rolling.json"
    summary_txt = tmp_path / "summary.txt"
    summary_json = tmp_path / "summary.json"
    pdf_path = tmp_path / "report.pdf"

    backtest_json.write_text(
        json.dumps(
            {
                "from_ts": "2026-02-24T00:00:00Z",
                "to_ts": "2026-02-24T02:00:00Z",
                "total_signals": 1,
                "executed_signals": 1,
                "rejected_signals": 0,
                "results": [{"strategy": "s1", "pnl": 1.0}],
            }
        )
    )
    rolling_json.write_text(
        json.dumps(
            {
                "summary": {
                    "run_count": 1,
                    "execution_rate": 1.0,
                    "positive_window_rate": 1.0,
                    "empty_window_count": 0,
                    "range_hours": 2.0,
                    "coverage_ratio": 1.0,
                    "overlap_ratio": 0.0,
                    "coverage_label": "full",
                    "risk_rejection_reasons": {},
                }
            }
        )
    )

    cmd = [
        sys.executable,
        str(SUMMARY_SCRIPT_PATH),
        "--backtest-json",
        str(backtest_json),
        "--pdf-path",
        str(pdf_path),
        "--rolling-json",
        str(rolling_json),
        "--summary-path",
        str(summary_txt),
        "--summary-json-path",
        str(summary_json),
        "--nightly-date",
        "2026-02-24",
        "--rolling-reject-top-k",
        "2",
    ]
    if with_checksum:
        cmd.append("--with-checksum")

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return summary_json


def test_cli_nightly_summary_verify_modes(tmp_path: Path) -> None:
    runner = CliRunner()

    summary_json_with = _write_nightly_summary_sidecar(tmp_path / "with", with_checksum=True)
    res_with = runner.invoke(
        app,
        ["nightly-summary-verify", "--summary-json", str(summary_json_with)],
    )
    assert res_with.exit_code == 0, res_with.output
    assert "nightly sidecar ok" in res_with.output
    assert "with-checksum" in res_with.output

    res_with_strict_ok = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_with),
            "--strict-schema-note-version",
            "1.0",
        ],
    )
    assert res_with_strict_ok.exit_code == 0, res_with_strict_ok.output

    res_with_strict_bad = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_with),
            "--strict-schema-note-version",
            "2.0",
        ],
    )
    assert res_with_strict_bad.exit_code == 1, res_with_strict_bad.output
    assert "schema_note_version mismatch" in res_with_strict_bad.output

    res_with_best_strict_ok = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_with),
            "--strict-best-version",
            "1.0",
        ],
    )
    assert res_with_best_strict_ok.exit_code == 0, res_with_best_strict_ok.output

    res_with_best_strict_bad = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_with),
            "--strict-best-version",
            "2.0",
        ],
    )
    assert res_with_best_strict_bad.exit_code == 1, res_with_best_strict_bad.output
    assert "best_version mismatch" in res_with_best_strict_bad.output

    summary_json_without = _write_nightly_summary_sidecar(tmp_path / "without", with_checksum=False)
    res_without = runner.invoke(
        app,
        ["nightly-summary-verify", "--summary-json", str(summary_json_without)],
    )
    assert res_without.exit_code == 1, res_without.output
    assert "missing checksum fields" in res_without.output

    res_without_allow = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_without),
            "--allow-missing-checksum",
        ],
    )
    assert res_without_allow.exit_code == 0, res_without_allow.output
    assert "nightly sidecar ok" in res_without_allow.output
    assert "no-checksum" in res_without_allow.output

    res_without_allow_note_strict = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_without),
            "--allow-missing-checksum",
            "--strict-schema-note-version",
            "1.0",
        ],
    )
    assert res_without_allow_note_strict.exit_code == 0, res_without_allow_note_strict.output

    res_without_allow_best_strict = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json_without),
            "--allow-missing-checksum",
            "--strict-best-version",
            "1.0",
        ],
    )
    assert res_without_allow_best_strict.exit_code == 0, res_without_allow_best_strict.output


def test_cli_nightly_summary_verify_reject_tampered_checksum(tmp_path: Path) -> None:
    runner = CliRunner()
    summary_json = _write_nightly_summary_sidecar(tmp_path / "tampered", with_checksum=True)

    payload = json.loads(summary_json.read_text())
    assert isinstance(payload["signals"], dict)
    payload["signals"]["executed"] = 0
    summary_json.write_text(json.dumps(payload))

    res = runner.invoke(
        app,
        ["nightly-summary-verify", "--summary-json", str(summary_json)],
    )
    assert res.exit_code == 1, res.output
    assert "nightly sidecar invalid" in res.output


def test_cli_nightly_summary_verify_reject_missing_schema_note_version_when_strict(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    summary_json = _write_nightly_summary_sidecar(
        tmp_path / "missing-note-version", with_checksum=True
    )

    payload = json.loads(summary_json.read_text())
    payload.pop("schema_note_version", None)
    old_checksum = str(payload["checksum_sha256"])
    # recompute checksum after removing optional field so strict gate, not checksum, triggers the failure
    payload["checksum_sha256"] = compute_nightly_summary_sidecar_checksum(payload)
    assert str(payload["checksum_sha256"]) != old_checksum
    summary_json.write_text(json.dumps(payload))

    res = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json),
            "--strict-schema-note-version",
            "1.0",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "schema_note_version mismatch" in res.output


def test_cli_nightly_summary_verify_reject_missing_best_version_when_strict(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    summary_json = _write_nightly_summary_sidecar(
        tmp_path / "missing-best-version", with_checksum=True
    )

    payload = json.loads(summary_json.read_text())
    payload.pop("best_version", None)
    old_checksum = str(payload["checksum_sha256"])
    payload["checksum_sha256"] = compute_nightly_summary_sidecar_checksum(payload)
    assert str(payload["checksum_sha256"]) != old_checksum
    summary_json.write_text(json.dumps(payload))

    res = runner.invoke(
        app,
        [
            "nightly-summary-verify",
            "--summary-json",
            str(summary_json),
            "--strict-best-version",
            "1.0",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "best_version mismatch" in res.output


def test_nightly_script_help_mentions_disabled_semantics() -> None:
    content = BASH_SCRIPT_PATH.read_text()
    assert "0=disabled" in content
    assert "--no-checksum" in content
    assert "--from-ts" in content
    assert "--to-ts" in content
    assert "--clear-signals-window" in content
    assert "--rebuild-signals-window" in content
    assert "--rebuild-step-hours" in content
    assert "--require-interpretable" in content
    assert "${CYCLE_WINDOW_ARGS[@]-}" in content
    assert "${CYCLE_CLEAR_ARGS[@]-}" in content
    assert "${CYCLE_REBUILD_ARGS[@]-}" in content
    assert '--rolling-json "$ROLLING_JSON"' in content
