from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from monomarket.backtest import (
    NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO,
    validate_nightly_summary_sidecar,
    verify_nightly_summary_sidecar_checksum,
)

ROOT = Path(__file__).resolve().parents[1]
BASH_SCRIPT_PATH = ROOT / "scripts" / "backtest_nightly_report.sh"
SUMMARY_SCRIPT_PATH = ROOT / "scripts" / "nightly_summary_line.py"


def test_nightly_summary_contains_canonical_alias_fields() -> None:
    content = SUMMARY_SCRIPT_PATH.read_text()

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


def test_nightly_script_help_mentions_disabled_semantics() -> None:
    content = BASH_SCRIPT_PATH.read_text()
    assert "0=disabled" in content
    assert "--no-checksum" in content
