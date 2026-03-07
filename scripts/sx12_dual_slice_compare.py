#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from monomarket.backtest.reject_reason import aggregate_reject_reasons


@dataclass(slots=True, frozen=True)
class SliceSpec:
    label: str
    hours: float


def _safe_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(raw: object, default: int = 0) -> int:
    try:
        return int(float(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_iso_utc(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_slice_specs(raw: str) -> list[SliceSpec]:
    specs: list[SliceSpec] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"invalid slice token (expected label:hours): {token}")
        label_raw, hours_raw = token.split(":", 1)
        label = label_raw.strip()
        if not label:
            raise ValueError(f"empty slice label in token: {token}")
        try:
            hours = float(hours_raw.strip())
        except ValueError as exc:
            raise ValueError(f"invalid slice hours in token: {token}") from exc
        if hours <= 0:
            raise ValueError(f"slice hours must be > 0 in token: {token}")
        specs.append(SliceSpec(label=label, hours=hours))

    if not specs:
        raise ValueError("no valid slices provided")
    return specs


def summarize_strategy(
    report: dict[str, Any],
    *,
    strategy: str,
    normalize_reject_reasons: bool = True,
) -> dict[str, Any]:
    target = strategy.strip().lower()
    results = report.get("results", []) if isinstance(report.get("results"), list) else []
    result_row = next(
        (
            row
            for row in results
            if isinstance(row, dict) and str(row.get("strategy", "")).strip().lower() == target
        ),
        {},
    )

    replay_rows = []
    for row in report.get("replay", []) if isinstance(report.get("replay"), list) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get("strategy", "")).strip().lower() != target:
            continue
        replay_rows.append(row)

    rejected_rows = [row for row in replay_rows if not bool(row.get("risk_allowed", True))]
    reason_counts: Counter[str] = Counter()
    for row in rejected_rows:
        reason = str(row.get("risk_reason") or "unknown").strip() or "unknown"
        reason_counts[reason] += 1

    top_pairs = aggregate_reject_reasons(reason_counts, normalize=normalize_reject_reasons)
    top_reason = "none" if not top_pairs else f"{top_pairs[0][0]}:{top_pairs[0][1]}"

    return {
        "strategy": target,
        "pnl": _safe_float(result_row.get("pnl")),
        "max_drawdown": _safe_float(result_row.get("max_drawdown")),
        "trade_count": _safe_int(result_row.get("trade_count")),
        "closed_winrate": _safe_float(result_row.get("closed_winrate")),
        "mtm_winrate": _safe_float(result_row.get("mtm_winrate")),
        "closed_sample_count": _safe_int(result_row.get("closed_sample_count")),
        "mtm_sample_count": _safe_int(result_row.get("mtm_sample_count")),
        "replay_rows": len(replay_rows),
        "executed_rows": len(replay_rows) - len(rejected_rows),
        "rejected_rows": len(rejected_rows),
        "reject_share": (
            float(len(rejected_rows)) / float(len(replay_rows)) if replay_rows else 0.0
        ),
        "top_reject_reason": top_reason,
        "reject_reasons": [[reason, count] for reason, count in top_pairs],
    }


def _summary_delta(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    strategies: list[str],
) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        b = baseline.get(strategy, {})
        c = candidate.get(strategy, {})
        delta[strategy] = {
            "pnl": _safe_float(c.get("pnl")) - _safe_float(b.get("pnl")),
            "max_drawdown": _safe_float(c.get("max_drawdown")) - _safe_float(b.get("max_drawdown")),
            "trade_count": float(_safe_int(c.get("trade_count")) - _safe_int(b.get("trade_count"))),
            "executed_rows": float(
                _safe_int(c.get("executed_rows")) - _safe_int(b.get("executed_rows"))
            ),
            "rejected_rows": float(
                _safe_int(c.get("rejected_rows")) - _safe_int(b.get("rejected_rows"))
            ),
            "mtm_winrate": _safe_float(c.get("mtm_winrate")) - _safe_float(b.get("mtm_winrate")),
            "closed_winrate": _safe_float(c.get("closed_winrate"))
            - _safe_float(b.get("closed_winrate")),
        }
    return delta


def _run_backtest(
    *,
    config_path: Path,
    strategies_csv: str,
    from_ts: str,
    to_ts: str,
    out_json: Path,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "monomarket",
        "backtest",
        "--strategies",
        strategies_csv,
        "--from",
        from_ts,
        "--to",
        to_ts,
        "--replay-limit",
        "0",
        "--config",
        str(config_path),
        "--out-json",
        str(out_json),
    ]
    env = os.environ.copy()
    env["ENABLE_LIVE_TRADING"] = "false"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "backtest failed"
            f"\ncmd: {' '.join(cmd)}"
            f"\nstdout:\n{proc.stdout.strip()}"
            f"\nstderr:\n{proc.stderr.strip()}"
        )
    if not out_json.exists():
        raise RuntimeError(f"backtest produced no json artifact: {out_json}")

    payload = json.loads(out_json.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid backtest json payload: {out_json}")
    return payload


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Sx12 Dual-Slice Compare",
        "",
        f"- generated_at: {result.get('generated_at')}",
        f"- anchor_ts: {result.get('anchor_ts')}",
        f"- baseline_config: {result.get('baseline_config')}",
        f"- candidate_config: {result.get('candidate_config')}",
        f"- strategies: {','.join(result.get('strategies', []))}",
        "",
    ]

    for item in result.get("slices", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "slice"))
        hours = _safe_float(item.get("hours"))
        lines.append(f"## {label} ({hours:g}h)")
        lines.append("")
        lines.append(
            "| strategy | base_pnl | cand_pnl | Δpnl | base_exec | cand_exec | Δexec | "
            "base_rej | cand_rej | Δrej | base_maxdd | cand_maxdd | Δmaxdd | base_mtm_wr | cand_mtm_wr | Δmtm_wr | "
            "base_top_reject | cand_top_reject |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
        )

        baseline = item.get("baseline", {}) if isinstance(item.get("baseline"), dict) else {}
        candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
        delta = item.get("delta", {}) if isinstance(item.get("delta"), dict) else {}
        baseline_by_strategy = (
            baseline.get("by_strategy", {}) if isinstance(baseline.get("by_strategy"), dict) else {}
        )
        candidate_by_strategy = (
            candidate.get("by_strategy", {})
            if isinstance(candidate.get("by_strategy"), dict)
            else {}
        )

        for strategy in result.get("strategies", []):
            strategy_key = str(strategy)
            b = baseline_by_strategy.get(strategy_key, {})
            c = candidate_by_strategy.get(strategy_key, {})
            d = delta.get(strategy_key, {}) if isinstance(delta.get(strategy_key), dict) else {}
            lines.append(
                "| "
                f"{strategy_key} | "
                f"{_safe_float(b.get('pnl')):.4f} | {_safe_float(c.get('pnl')):.4f} | {(_safe_float(d.get('pnl'))):+.4f} | "
                f"{_safe_int(b.get('executed_rows'))} | {_safe_int(c.get('executed_rows'))} | {(_safe_int(c.get('executed_rows')) - _safe_int(b.get('executed_rows'))):+d} | "
                f"{_safe_int(b.get('rejected_rows'))} | {_safe_int(c.get('rejected_rows'))} | {(_safe_int(c.get('rejected_rows')) - _safe_int(b.get('rejected_rows'))):+d} | "
                f"{_safe_float(b.get('max_drawdown')):.4f} | {_safe_float(c.get('max_drawdown')):.4f} | {(_safe_float(d.get('max_drawdown'))):+.4f} | "
                f"{_safe_float(b.get('mtm_winrate')):.4f} | {_safe_float(c.get('mtm_winrate')):.4f} | {(_safe_float(d.get('mtm_winrate'))):+.4f} | "
                f"{str(b.get('top_reject_reason', 'none'))} | {str(c.get('top_reject_reason', 'none'))} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Sx12 (S9/S10 by default) dual-slice backtests for baseline vs candidate "
            "configs and emit JSON/Markdown compare artifacts."
        )
    )
    parser.add_argument("--baseline-config", required=True, help="Baseline config path")
    parser.add_argument("--candidate-config", required=True, help="Candidate config path")
    parser.add_argument(
        "--strategies",
        default="s9,s10",
        help="Comma-separated strategy ids (default: s9,s10)",
    )
    parser.add_argument(
        "--slices",
        default="recent24h:24,recent7d:168",
        help="Slice specs label:hours,... (default: recent24h:24,recent7d:168)",
    )
    parser.add_argument(
        "--anchor-ts",
        default=None,
        help="Anchor ISO timestamp (UTC). Default: now.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: artifacts/backtest/sx12-dual-slice-<anchor>",
    )
    parser.add_argument(
        "--no-normalize-reject-reasons",
        action="store_true",
        help="Keep raw reject reasons (disable prefix normalization)",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    baseline_config = Path(args.baseline_config).resolve()
    candidate_config = Path(args.candidate_config).resolve()
    if not baseline_config.exists():
        raise FileNotFoundError(f"baseline config not found: {baseline_config}")
    if not candidate_config.exists():
        raise FileNotFoundError(f"candidate config not found: {candidate_config}")

    strategies = [s.strip().lower() for s in str(args.strategies).split(",") if s.strip()]
    if not strategies:
        raise ValueError("no strategies selected")

    specs = parse_slice_specs(str(args.slices))
    anchor = _parse_iso_utc(args.anchor_ts) if args.anchor_ts else datetime.now(UTC)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (
            Path("artifacts/backtest") / f"sx12-dual-slice-{anchor.strftime('%Y%m%dT%H%M%SZ')}"
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize_reject_reasons = not bool(args.no_normalize_reject_reasons)
    strategies_csv = ",".join(strategies)

    result: dict[str, Any] = {
        "generated_at": _iso_z(datetime.now(UTC)),
        "anchor_ts": _iso_z(anchor),
        "baseline_config": str(baseline_config),
        "candidate_config": str(candidate_config),
        "strategies": strategies,
        "normalize_reject_reasons": normalize_reject_reasons,
        "slices": [],
    }

    for spec in specs:
        to_dt = anchor
        from_dt = anchor - timedelta(hours=spec.hours)
        from_ts = _iso_z(from_dt)
        to_ts = _iso_z(to_dt)

        baseline_out = out_dir / "baseline" / f"{spec.label}.json"
        candidate_out = out_dir / "candidate" / f"{spec.label}.json"
        baseline_out.parent.mkdir(parents=True, exist_ok=True)
        candidate_out.parent.mkdir(parents=True, exist_ok=True)

        baseline_report = _run_backtest(
            config_path=baseline_config,
            strategies_csv=strategies_csv,
            from_ts=from_ts,
            to_ts=to_ts,
            out_json=baseline_out,
        )
        candidate_report = _run_backtest(
            config_path=candidate_config,
            strategies_csv=strategies_csv,
            from_ts=from_ts,
            to_ts=to_ts,
            out_json=candidate_out,
        )

        baseline_by_strategy = {
            strategy: summarize_strategy(
                baseline_report,
                strategy=strategy,
                normalize_reject_reasons=normalize_reject_reasons,
            )
            for strategy in strategies
        }
        candidate_by_strategy = {
            strategy: summarize_strategy(
                candidate_report,
                strategy=strategy,
                normalize_reject_reasons=normalize_reject_reasons,
            )
            for strategy in strategies
        }

        result["slices"].append(
            {
                "label": spec.label,
                "hours": spec.hours,
                "from_ts": from_ts,
                "to_ts": to_ts,
                "baseline": {
                    "report_path": str(baseline_out),
                    "total_signals": _safe_int(baseline_report.get("total_signals")),
                    "executed_signals": _safe_int(baseline_report.get("executed_signals")),
                    "rejected_signals": _safe_int(baseline_report.get("rejected_signals")),
                    "by_strategy": baseline_by_strategy,
                },
                "candidate": {
                    "report_path": str(candidate_out),
                    "total_signals": _safe_int(candidate_report.get("total_signals")),
                    "executed_signals": _safe_int(candidate_report.get("executed_signals")),
                    "rejected_signals": _safe_int(candidate_report.get("rejected_signals")),
                    "by_strategy": candidate_by_strategy,
                },
                "delta": _summary_delta(
                    baseline_by_strategy,
                    candidate_by_strategy,
                    strategies=strategies,
                ),
            }
        )

    compare_json = out_dir / "compare.json"
    compare_md = out_dir / "compare.md"
    compare_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    compare_md.write_text(render_markdown(result))

    print(f"sx12 dual-slice compare completed: {out_dir}")
    print(f"  json: {compare_json}")
    print(f"  markdown: {compare_md}")
    for item in result.get("slices", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "slice"))
        delta = item.get("delta", {}) if isinstance(item.get("delta"), dict) else {}
        for strategy in strategies:
            sd = delta.get(strategy, {}) if isinstance(delta.get(strategy), dict) else {}
            print(
                f"  [{label}] {strategy} Δpnl={_safe_float(sd.get('pnl')):+.4f} "
                f"Δexec={_safe_int(sd.get('executed_rows')):+d} "
                f"Δrej={_safe_int(sd.get('rejected_rows')):+d}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
