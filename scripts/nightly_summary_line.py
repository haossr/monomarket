#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from monomarket.backtest import (
    NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO,
    compute_nightly_summary_sidecar_checksum,
)

ROLLING_REJECT_TOP_DELIMITER = ";"


def _f(raw: object) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _best_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results") or []
    best: dict[str, Any] | None = None
    if isinstance(rows, list) and rows:
        candidates = [r for r in rows if isinstance(r, dict)]
        if candidates:
            best = max(candidates, key=lambda r: _f(r.get("pnl")))

    if not isinstance(best, dict):
        return {
            "available": False,
            "strategy": "",
            "pnl": 0.0,
            "text": "best_strategy=n/a",
        }

    strategy = str(best.get("strategy", ""))
    pnl = _f(best.get("pnl"))
    return {
        "available": True,
        "strategy": strategy,
        "pnl": pnl,
        "text": f"best_strategy={strategy} pnl={pnl:.4f}",
    }


def build_summary_bundle(
    *,
    payload: dict[str, Any],
    rolling_payload: dict[str, Any] | None,
    pdf_path: Path,
    rolling_path: Path,
    nightly_date: str,
    rolling_reject_top_k: int,
) -> tuple[str, dict[str, Any]]:
    best_info = _best_strategy(payload)
    best_text = str(best_info.get("text") or "best_strategy=n/a")

    rolling_runs = 0
    rolling_exec_rate = 0.0
    rolling_range_hours = 0.0
    rolling_coverage_ratio = 0.0
    rolling_overlap_ratio = 0.0
    rolling_coverage_label = "unknown"
    rolling_positive_window_rate = 0.0
    rolling_empty_window_count = 0

    k_norm = max(0, int(rolling_reject_top_k))
    rolling_reject_top = "disabled" if k_norm <= 0 else "none"
    rolling_reject_top_pairs: list[tuple[str, int]] = []

    rolling_summary = None
    if isinstance(rolling_payload, dict):
        rolling_summary = rolling_payload.get("summary")

    if isinstance(rolling_summary, dict):
        rolling_runs = int(_f(rolling_summary.get("run_count")))
        rolling_exec_rate = _f(rolling_summary.get("execution_rate"))
        rolling_range_hours = _f(rolling_summary.get("range_hours"))
        rolling_coverage_ratio = _f(rolling_summary.get("coverage_ratio"))
        rolling_overlap_ratio = _f(rolling_summary.get("overlap_ratio"))
        rolling_positive_window_rate = _f(rolling_summary.get("positive_window_rate"))
        rolling_empty_window_count = int(_f(rolling_summary.get("empty_window_count")))
        coverage_label_raw = rolling_summary.get("coverage_label")
        if isinstance(coverage_label_raw, str) and coverage_label_raw.strip():
            rolling_coverage_label = coverage_label_raw.strip()

        raw_reasons = rolling_summary.get("risk_rejection_reasons")
        if isinstance(raw_reasons, dict):
            reason_items: list[tuple[str, int]] = []
            for key, value in raw_reasons.items():
                reason = str(key).strip() or "unknown"
                count = int(_f(value))
                if count <= 0:
                    continue
                reason_items.append((reason, count))
            if reason_items and k_norm > 0:
                reason_items.sort(key=lambda x: (-x[1], x[0]))
                rolling_reject_top_pairs = reason_items[:k_norm]
                rolling_reject_top = ROLLING_REJECT_TOP_DELIMITER.join(
                    f"{reason}:{count}" for reason, count in rolling_reject_top_pairs
                )

    line = (
        f"Nightly {nightly_date} | window={payload.get('from_ts', '')} -> {payload.get('to_ts', '')} "
        f"| signals total={payload.get('total_signals', 0)} executed={payload.get('executed_signals', 0)} "
        f"rejected={payload.get('rejected_signals', 0)} | {best_text} "
        f"| rolling runs={rolling_runs} exec_rate={rolling_exec_rate:.2%} "
        f"pos_win_rate={rolling_positive_window_rate:.2%} empty_windows={rolling_empty_window_count} "
        f"positive_window_rate={rolling_positive_window_rate:.2%} "
        f"empty_window_count={rolling_empty_window_count} "
        f"range_h={rolling_range_hours:.2f} coverage={rolling_coverage_ratio:.2%} "
        f"overlap={rolling_overlap_ratio:.2%} "
        f"range_hours={rolling_range_hours:.2f} coverage_ratio={rolling_coverage_ratio:.2%} "
        f"overlap_ratio={rolling_overlap_ratio:.2%} coverage_label={rolling_coverage_label} "
        f"rolling_reject_top_k={k_norm} "
        f"rolling_reject_top_delim={ROLLING_REJECT_TOP_DELIMITER} "
        f"rolling_reject_top={rolling_reject_top} "
        f"| pdf={pdf_path.resolve()} | rolling_json={rolling_path.resolve()}"
    )

    sidecar = {
        "schema_version": "nightly-summary-sidecar-1.0",
        "schema_note_version": "1.0",
        "schema_note": "best is structured object; prefer rolling.reject_top_pairs for machine parsing",
        "best_version": "1.0",
        "nightly_date": nightly_date,
        "window": {
            "from_ts": str(payload.get("from_ts", "")),
            "to_ts": str(payload.get("to_ts", "")),
        },
        "signals": {
            "total": int(_f(payload.get("total_signals"))),
            "executed": int(_f(payload.get("executed_signals"))),
            "rejected": int(_f(payload.get("rejected_signals"))),
        },
        "best": {
            "available": bool(best_info.get("available", False)),
            "strategy": str(best_info.get("strategy", "")),
            "pnl": float(best_info.get("pnl", 0.0)),
            "text": best_text,
        },
        "best_text": best_text,
        "rolling": {
            "runs": rolling_runs,
            "execution_rate": rolling_exec_rate,
            "pos_win_rate": rolling_positive_window_rate,
            "empty_windows": rolling_empty_window_count,
            "positive_window_rate": rolling_positive_window_rate,
            "empty_window_count": rolling_empty_window_count,
            "range_h": rolling_range_hours,
            "coverage": rolling_coverage_ratio,
            "overlap": rolling_overlap_ratio,
            "range_hours": rolling_range_hours,
            "coverage_ratio": rolling_coverage_ratio,
            "overlap_ratio": rolling_overlap_ratio,
            "coverage_label": rolling_coverage_label,
            "reject_top_k": k_norm,
            "reject_top_delimiter": ROLLING_REJECT_TOP_DELIMITER,
            "reject_top": rolling_reject_top,
            "reject_top_pairs": [
                {"reason": reason, "count": count} for reason, count in rolling_reject_top_pairs
            ],
        },
        "paths": {
            "pdf": str(pdf_path.resolve()),
            "rolling_json": str(rolling_path.resolve()),
        },
    }

    return line, sidecar


def build_summary_line(
    *,
    payload: dict[str, Any],
    rolling_payload: dict[str, Any] | None,
    pdf_path: Path,
    rolling_path: Path,
    nightly_date: str,
    rolling_reject_top_k: int,
) -> str:
    line, _ = build_summary_bundle(
        payload=payload,
        rolling_payload=rolling_payload,
        pdf_path=pdf_path,
        rolling_path=rolling_path,
        nightly_date=nightly_date,
        rolling_reject_top_k=rolling_reject_top_k,
    )
    return line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-json", required=True)
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--rolling-json", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--summary-json-path", default=None)
    parser.add_argument("--nightly-date", required=True)
    parser.add_argument("--rolling-reject-top-k", type=int, default=2)
    parser.add_argument("--with-checksum", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.backtest_json).read_text())
    rolling_path = Path(args.rolling_json)
    rolling_payload: dict[str, Any] | None = None
    if rolling_path.exists():
        rolling_payload = json.loads(rolling_path.read_text())

    line, sidecar = build_summary_bundle(
        payload=payload,
        rolling_payload=rolling_payload,
        pdf_path=Path(args.pdf_path),
        rolling_path=rolling_path,
        nightly_date=args.nightly_date,
        rolling_reject_top_k=args.rolling_reject_top_k,
    )
    Path(args.summary_path).write_text(line + "\n")
    if args.summary_json_path:
        if args.with_checksum:
            sidecar["checksum_algo"] = NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
            sidecar["checksum_sha256"] = compute_nightly_summary_sidecar_checksum(sidecar)
        Path(args.summary_json_path).write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n"
        )
    print(line)


if __name__ == "__main__":
    main()
