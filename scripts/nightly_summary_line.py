#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monomarket.backtest import (
    NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO,
    compute_nightly_summary_sidecar_checksum,
)
from monomarket.backtest.reject_reason import format_reject_top, normalize_reject_reason

ROLLING_REJECT_TOP_DELIMITER = ";"


def _f(raw: object) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _best_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    executed_signals = int(_f(payload.get("executed_signals")))
    if executed_signals <= 0:
        return {
            "available": False,
            "strategy": "",
            "pnl": 0.0,
            "text": "best_strategy=n/a",
        }

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


def _aggregate_winrate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results")
    closed_wins = 0
    closed_losses = 0
    mtm_wins = 0
    mtm_losses = 0

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue

            row_closed_wins = int(_f(row.get("wins")))
            row_closed_losses = int(_f(row.get("losses")))
            if row_closed_wins <= 0 and row_closed_losses <= 0:
                closed_sample = int(_f(row.get("closed_sample_count")))
                if closed_sample > 0:
                    closed_rate = _f(row.get("closed_winrate", row.get("winrate")))
                    row_closed_wins = max(
                        0, min(closed_sample, int(round(closed_sample * closed_rate)))
                    )
                    row_closed_losses = max(0, closed_sample - row_closed_wins)

            row_mtm_wins = int(_f(row.get("mtm_wins")))
            row_mtm_losses = int(_f(row.get("mtm_losses")))
            if row_mtm_wins <= 0 and row_mtm_losses <= 0:
                mtm_sample = int(_f(row.get("mtm_sample_count")))
                if mtm_sample > 0:
                    mtm_rate = _f(row.get("mtm_winrate"))
                    row_mtm_wins = max(0, min(mtm_sample, int(round(mtm_sample * mtm_rate))))
                    row_mtm_losses = max(0, mtm_sample - row_mtm_wins)

            closed_wins += max(0, row_closed_wins)
            closed_losses += max(0, row_closed_losses)
            mtm_wins += max(0, row_mtm_wins)
            mtm_losses += max(0, row_mtm_losses)

    closed_sample_count = closed_wins + closed_losses
    mtm_sample_count = mtm_wins + mtm_losses

    closed_winrate = (closed_wins / closed_sample_count) if closed_sample_count > 0 else 0.0
    mtm_winrate = (mtm_wins / mtm_sample_count) if mtm_sample_count > 0 else 0.0

    return {
        "closed_winrate": closed_winrate,
        "closed_sample_count": closed_sample_count,
        "closed_wins": closed_wins,
        "closed_losses": closed_losses,
        "mtm_winrate": mtm_winrate,
        "mtm_sample_count": mtm_sample_count,
        "mtm_wins": mtm_wins,
        "mtm_losses": mtm_losses,
    }


def _format_winrate(rate: float, sample_count: int) -> str:
    if sample_count <= 0:
        return "n/a"
    return f"{rate:.2%}"


def _parse_iso_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _compute_window_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    from_ts_raw = payload.get("from_ts")
    to_ts_raw = payload.get("to_ts")
    from_ts = _parse_iso_ts(from_ts_raw)
    to_ts = _parse_iso_ts(to_ts_raw)

    replay = payload.get("replay")
    replay_rows = replay if isinstance(replay, list) else []

    first_replay_dt: datetime | None = None
    for row in replay_rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_iso_ts(row.get("ts"))
        if ts is None:
            continue
        if first_replay_dt is None or ts < first_replay_dt:
            first_replay_dt = ts

    if from_ts is None or to_ts is None:
        return {
            "window_hours": 0.0,
            "covered_hours": 0.0,
            "coverage_ratio": 0.0,
            "history_limited": False,
            "note": "window_parse_error",
            "first_replay_ts": (
                first_replay_dt.isoformat().replace("+00:00", "Z")
                if first_replay_dt is not None
                else ""
            ),
            "effective_from_ts": "",
        }

    window_hours = max(0.0, (to_ts - from_ts).total_seconds() / 3600.0)
    effective_from_dt: datetime | None = from_ts
    history_limited = False
    note = "full_history"

    if first_replay_dt is not None and first_replay_dt > from_ts:
        effective_from_dt = first_replay_dt
        history_limited = True
        note = "history_limited"
    elif first_replay_dt is None:
        effective_from_dt = None
        note = "no_replay_rows"

    if effective_from_dt is None:
        covered_hours = 0.0
    else:
        covered_hours = max(0.0, (to_ts - effective_from_dt).total_seconds() / 3600.0)
    coverage_ratio = (covered_hours / window_hours) if window_hours > 0 else 0.0

    return {
        "window_hours": window_hours,
        "covered_hours": covered_hours,
        "coverage_ratio": max(0.0, min(1.0, coverage_ratio)),
        "history_limited": history_limited,
        "note": note,
        "first_replay_ts": (
            first_replay_dt.isoformat().replace("+00:00", "Z")
            if first_replay_dt is not None
            else ""
        ),
        "effective_from_ts": (
            effective_from_dt.isoformat().replace("+00:00", "Z")
            if effective_from_dt is not None
            else ""
        ),
    }


def _reject_by_strategy(payload: dict[str, Any], *, top_k: int = 3) -> dict[str, Any]:
    replay = payload.get("replay")
    rows = replay if isinstance(replay, list) else []

    strategy_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue

        strategy = str(row.get("strategy") or "").strip() or "unknown"
        item = strategy_stats.setdefault(
            strategy,
            {
                "strategy": strategy,
                "total": 0,
                "rejected": 0,
                "reason_counts": {},
            },
        )
        item["total"] = int(item["total"]) + 1

        risk_allowed_raw = row.get("risk_allowed")
        risk_allowed = True if risk_allowed_raw is None else bool(risk_allowed_raw)
        risk_reason_raw = str(row.get("risk_reason") or "").strip()
        rejected = (not risk_allowed) or (
            risk_reason_raw != "" and risk_reason_raw.lower() != "ok"
        )
        if not rejected:
            continue

        item["rejected"] = int(item["rejected"]) + 1
        reason = normalize_reject_reason(risk_reason_raw) if risk_reason_raw else "unknown"
        reason_counts = item.get("reason_counts")
        if isinstance(reason_counts, dict):
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    summary_rows: list[dict[str, Any]] = []
    for strategy, item in strategy_stats.items():
        total = int(item.get("total", 0))
        rejected = int(item.get("rejected", 0))
        if rejected <= 0:
            continue

        reason_counts = item.get("reason_counts")
        top_reason_text = "none"
        if isinstance(reason_counts, dict) and reason_counts:
            top_reason_text, _ = format_reject_top(
                reason_counts,
                top_k=1,
                delimiter=ROLLING_REJECT_TOP_DELIMITER,
                normalize=True,
            )

        summary_rows.append(
            {
                "strategy": strategy,
                "total": total,
                "rejected": rejected,
                "reject_rate": (rejected / total) if total > 0 else 0.0,
                "top_reason": top_reason_text,
            }
        )

    summary_rows.sort(key=lambda row: (-int(row["rejected"]), str(row["strategy"])))
    top_rows = summary_rows[: max(0, int(top_k))]
    top_text = (
        ROLLING_REJECT_TOP_DELIMITER.join(
            f"{row['strategy']}:{int(row['rejected'])}" for row in top_rows
        )
        if top_rows
        else "none"
    )

    return {
        "top_k": int(top_k),
        "delimiter": ROLLING_REJECT_TOP_DELIMITER,
        "top": top_text,
        "rows": top_rows,
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

    winrate_info = _aggregate_winrate(payload)
    closed_winrate = float(winrate_info.get("closed_winrate", 0.0))
    closed_sample_count = int(winrate_info.get("closed_sample_count", 0))
    mtm_winrate = float(winrate_info.get("mtm_winrate", 0.0))
    mtm_sample_count = int(winrate_info.get("mtm_sample_count", 0))

    window_coverage = _compute_window_coverage(payload)
    window_coverage_ratio = float(window_coverage.get("coverage_ratio", 0.0))
    window_note = str(window_coverage.get("note", ""))
    window_history_limited = bool(window_coverage.get("history_limited", False))

    reject_strategy_info = _reject_by_strategy(payload, top_k=3)
    reject_strategy_top = str(reject_strategy_info.get("top", "none"))

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
    rolling_reject_top_normalized = "disabled" if k_norm <= 0 else "none"
    rolling_reject_top_pairs_normalized: list[tuple[str, int]] = []

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
            (
                rolling_reject_top,
                rolling_reject_top_pairs,
            ) = format_reject_top(
                raw_reasons, top_k=k_norm, delimiter=ROLLING_REJECT_TOP_DELIMITER, normalize=False
            )
            (
                rolling_reject_top_normalized,
                rolling_reject_top_pairs_normalized,
            ) = format_reject_top(
                raw_reasons, top_k=k_norm, delimiter=ROLLING_REJECT_TOP_DELIMITER, normalize=True
            )

    line = (
        f"Nightly {nightly_date} | window={payload.get('from_ts', '')} -> {payload.get('to_ts', '')} "
        f"| signals total={payload.get('total_signals', 0)} executed={payload.get('executed_signals', 0)} "
        f"rejected={payload.get('rejected_signals', 0)} "
        f"| closed_winrate={_format_winrate(closed_winrate, closed_sample_count)} "
        f"closed_samples={closed_sample_count} "
        f"mtm_winrate={_format_winrate(mtm_winrate, mtm_sample_count)} "
        f"mtm_samples={mtm_sample_count} "
        f"main_coverage={window_coverage_ratio:.2%} "
        f"history_limited={str(window_history_limited).lower()} "
        f"window_note={window_note} "
        f"| {best_text} "
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
        f"rolling_reject_top_normalized={rolling_reject_top_normalized} "
        f"reject_strategy_top={reject_strategy_top} "
        f"| pdf={pdf_path.resolve()} | rolling_json={rolling_path.resolve()}"
    )

    sidecar = {
        "schema_version": "nightly-summary-sidecar-1.0",
        "schema_note_version": "1.0",
        "schema_note": "best is structured object; prefer rolling.reject_top_pairs(_normalized) and reject_by_strategy.rows for machine parsing",
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
        "winrate": {
            "closed_winrate": closed_winrate,
            "closed_sample_count": closed_sample_count,
            "closed_wins": int(winrate_info.get("closed_wins", 0)),
            "closed_losses": int(winrate_info.get("closed_losses", 0)),
            "mtm_winrate": mtm_winrate,
            "mtm_sample_count": mtm_sample_count,
            "mtm_wins": int(winrate_info.get("mtm_wins", 0)),
            "mtm_losses": int(winrate_info.get("mtm_losses", 0)),
        },
        "window_coverage": {
            "window_hours": float(window_coverage.get("window_hours", 0.0)),
            "covered_hours": float(window_coverage.get("covered_hours", 0.0)),
            "coverage_ratio": window_coverage_ratio,
            "history_limited": window_history_limited,
            "note": window_note,
            "first_replay_ts": str(window_coverage.get("first_replay_ts", "")),
            "effective_from_ts": str(window_coverage.get("effective_from_ts", "")),
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
            "reject_top_normalized": rolling_reject_top_normalized,
            "reject_top_pairs_normalized": [
                {"reason": reason, "count": count}
                for reason, count in rolling_reject_top_pairs_normalized
            ],
        },
        "reject_by_strategy": {
            "top_k": int(reject_strategy_info.get("top_k", 0)),
            "delimiter": str(reject_strategy_info.get("delimiter", ROLLING_REJECT_TOP_DELIMITER)),
            "top": reject_strategy_top,
            "rows": [
                {
                    "strategy": str(row.get("strategy", "")),
                    "total": int(_f(row.get("total"))),
                    "rejected": int(_f(row.get("rejected"))),
                    "reject_rate": float(_f(row.get("reject_rate"))),
                    "top_reason": str(row.get("top_reason", "none")),
                }
                for row in reject_strategy_info.get("rows", [])
                if isinstance(row, dict)
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
