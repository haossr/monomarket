from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
        "positive_window_rate=",
        "empty_window_count=",
        "range_hours=",
        "coverage_ratio=",
        "overlap_ratio=",
        "rolling_reject_top_k=",
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
    ]
    for token in required_tokens:
        assert token in content


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

    sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(sidecar)
    assert verify_nightly_summary_sidecar_checksum(sidecar)

    best_obj = sidecar["best"]
    assert isinstance(best_obj, dict)
    assert bool(best_obj["available"]) is False
    assert str(best_obj["strategy"]) == ""
    assert float(best_obj["pnl"]) == 0.0
    assert str(best_obj["text"]) == "best_strategy=n/a"
    assert str(sidecar["best_text"]) == "best_strategy=n/a"


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

    none_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(none_sidecar)
    assert str(none_sidecar["checksum_algo"]) == NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
    assert verify_nightly_summary_sidecar_checksum(none_sidecar)
    assert int(none_sidecar["rolling"]["reject_top_k"]) == 2
    assert str(none_sidecar["rolling"]["reject_top_delimiter"]) == ";"
    assert str(none_sidecar["rolling"]["reject_top"]) == "none"

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

    reasons_sidecar = json.loads(summary_json.read_text())
    validate_nightly_summary_sidecar(reasons_sidecar)
    assert verify_nightly_summary_sidecar_checksum(reasons_sidecar)


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
