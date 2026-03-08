#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from monomarket.backtest.reject_reason import aggregate_reject_reasons

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _resolve_db_path(config_payload: dict[str, Any], *, config_path: Path) -> Path:
    app = config_payload.get("app", {})
    if not isinstance(app, dict):
        raise ValueError(f"config app section must be a mapping: {config_path}")
    db_raw = str(app.get("db_path") or "").strip()
    if not db_raw:
        raise ValueError(f"config missing app.db_path: {config_path}")

    db_path = Path(db_raw)
    if db_path.is_absolute():
        return db_path
    return (config_path.parent / db_path).resolve()


def prepare_isolated_config(
    *,
    source_config_path: Path,
    run_dir: Path,
    config_tag: str,
) -> Path:
    cfg = _load_yaml_mapping(source_config_path)
    source_db_path = _resolve_db_path(cfg, config_path=source_config_path)

    isolated_db_dir = run_dir / "db"
    isolated_db_dir.mkdir(parents=True, exist_ok=True)
    isolated_db_path = isolated_db_dir / f"{config_tag}-{source_db_path.name}"
    if source_db_path.exists():
        shutil.copy2(source_db_path, isolated_db_path)

    app = cfg.setdefault("app", {})
    if not isinstance(app, dict):
        raise ValueError("config app section must be a mapping")
    app["db_path"] = str(isolated_db_path)

    isolated_config_path = run_dir / f"{config_tag}.config.yaml"
    isolated_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return isolated_config_path


def _build_backtest_cycle_cmd(
    *,
    config_path: Path,
    from_ts: str,
    to_ts: str,
    output_dir: Path,
    rebuild_step_hours: float,
    rebuild_market_limit: int,
    rebuild_ingest_limit: int,
    skip_ingest: bool,
    settle_window_end: bool,
) -> list[str]:
    cmd = [
        "bash",
        "scripts/backtest_cycle.sh",
        "--from-ts",
        from_ts,
        "--to-ts",
        to_ts,
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--market-limit",
        str(rebuild_market_limit),
        "--ingest-limit",
        str(rebuild_ingest_limit),
        "--clear-signals-window",
        "--rebuild-signals-window",
        "--rebuild-step-hours",
        str(rebuild_step_hours),
    ]
    if skip_ingest:
        cmd.append("--skip-ingest")
    if not settle_window_end:
        cmd.append("--no-settle-window-end")
    return cmd


def _extract_settle_window_end(
    report: dict[str, Any],
    *,
    cycle_meta: dict[str, Any] | None,
) -> tuple[bool, str]:
    if isinstance(cycle_meta, dict):
        signal_generation = cycle_meta.get("signal_generation")
        if isinstance(signal_generation, dict) and "settle_window_end" in signal_generation:
            return (
                bool(signal_generation.get("settle_window_end")),
                "cycle_meta.signal_generation.settle_window_end",
            )

    execution_config = report.get("execution_config")
    if isinstance(execution_config, dict) and "settle_window_end_positions" in execution_config:
        return (
            bool(execution_config.get("settle_window_end_positions")),
            "report.execution_config.settle_window_end_positions",
        )

    return (False, "default(false)")


def summarize_strategy(
    report: dict[str, Any],
    *,
    strategy: str,
    normalize_reject_reasons: bool = True,
    cycle_meta: dict[str, Any] | None = None,
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

    generation_by_strategy: dict[str, Any] = {}
    if isinstance(cycle_meta, dict):
        signal_generation = cycle_meta.get("signal_generation")
        if isinstance(signal_generation, dict):
            edge_gate = signal_generation.get("edge_gate")
            if isinstance(edge_gate, dict):
                by_strategy = edge_gate.get("by_strategy")
                if isinstance(by_strategy, dict):
                    generation_by_strategy = {
                        str(k).strip().lower(): v
                        for k, v in by_strategy.items()
                        if isinstance(v, dict)
                    }

    generation_row = generation_by_strategy.get(target, {})
    generation_raw = _safe_int(generation_row.get("raw"))
    generation_pass = _safe_int(generation_row.get("pass"))
    generation_fail = _safe_int(generation_row.get("fail"))
    generation_pass_rate = _safe_float(generation_row.get("pass_rate"))

    generation_reason_counts: Counter[str] = Counter()
    raw_strategy_diag = generation_row.get("strategy_diagnostics")
    strategy_diag = raw_strategy_diag if isinstance(raw_strategy_diag, dict) else {}
    candidate_reject_reasons = strategy_diag.get("candidate_reject_reasons")
    if isinstance(candidate_reject_reasons, dict):
        for reason, raw_count in candidate_reject_reasons.items():
            count = _safe_int(raw_count)
            if count <= 0:
                continue
            generation_reason_counts[str(reason)] += count

    generation_reject_top_pairs = aggregate_reject_reasons(
        generation_reason_counts,
        normalize=normalize_reject_reasons,
    )
    generation_top_reject_reason = (
        "none"
        if not generation_reject_top_pairs
        else f"{generation_reject_top_pairs[0][0]}:{generation_reject_top_pairs[0][1]}"
    )
    generation_rejected_candidates = sum(generation_reason_counts.values())

    generation_reject_by_event_raw = strategy_diag.get("candidate_reject_reasons_by_event_top")
    if not isinstance(generation_reject_by_event_raw, dict):
        generation_reject_by_event_raw = strategy_diag.get("candidate_reject_reasons_by_event")

    generation_reject_by_event: dict[str, dict[str, int]] = {}
    if isinstance(generation_reject_by_event_raw, dict):
        for raw_event_id, raw_reasons in generation_reject_by_event_raw.items():
            event_id = str(raw_event_id or "").strip()
            if not event_id or not isinstance(raw_reasons, dict):
                continue
            normalized_reasons: dict[str, int] = {}
            for raw_reason, raw_count in raw_reasons.items():
                reason = str(raw_reason or "unknown").strip() or "unknown"
                count = _safe_int(raw_count)
                if count <= 0:
                    continue
                normalized_reasons[reason] = count
            if normalized_reasons:
                generation_reject_by_event[event_id] = normalized_reasons

    generation_reject_event_totals = sorted(
        (
            (event_id, sum(int(v) for v in event_reason_counts.values()))
            for event_id, event_reason_counts in generation_reject_by_event.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    generation_top_reject_event = (
        "none"
        if not generation_reject_event_totals
        else f"{generation_reject_event_totals[0][0]}:{generation_reject_event_totals[0][1]}"
    )
    generation_top_reject_event_count = (
        0 if not generation_reject_event_totals else int(generation_reject_event_totals[0][1])
    )
    generation_top_reject_event_reason = "none"
    if generation_reject_event_totals:
        top_event_id = generation_reject_event_totals[0][0]
        top_event_reason_pairs = aggregate_reject_reasons(
            Counter(generation_reject_by_event.get(top_event_id, {})),
            normalize=normalize_reject_reasons,
        )
        if top_event_reason_pairs:
            generation_top_reject_event_reason = (
                f"{top_event_id}|{top_event_reason_pairs[0][0]}:{top_event_reason_pairs[0][1]}"
            )

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
        "generation_raw": generation_raw,
        "generation_pass": generation_pass,
        "generation_fail": generation_fail,
        "generation_pass_rate": generation_pass_rate,
        "generation_rejected_candidates": generation_rejected_candidates,
        "generation_top_reject_reason": generation_top_reject_reason,
        "generation_reject_reasons": [
            [reason, count] for reason, count in generation_reject_top_pairs
        ],
        "generation_top_reject_event": generation_top_reject_event,
        "generation_top_reject_event_count": generation_top_reject_event_count,
        "generation_top_reject_event_reason": generation_top_reject_event_reason,
        "generation_reject_events": [
            [event_id, total] for event_id, total in generation_reject_event_totals
        ],
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
        base_top_event = str(b.get("generation_top_reject_event") or "none")
        cand_top_event = str(c.get("generation_top_reject_event") or "none")
        top_event_shift = int(
            (base_top_event != cand_top_event)
            and not (base_top_event == "none" and cand_top_event == "none")
        )
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
            "generation_raw": float(
                _safe_int(c.get("generation_raw")) - _safe_int(b.get("generation_raw"))
            ),
            "generation_pass": float(
                _safe_int(c.get("generation_pass")) - _safe_int(b.get("generation_pass"))
            ),
            "generation_fail": float(
                _safe_int(c.get("generation_fail")) - _safe_int(b.get("generation_fail"))
            ),
            "generation_rejected_candidates": float(
                _safe_int(c.get("generation_rejected_candidates"))
                - _safe_int(b.get("generation_rejected_candidates"))
            ),
            "generation_top_reject_event_count": float(
                _safe_int(c.get("generation_top_reject_event_count"))
                - _safe_int(b.get("generation_top_reject_event_count"))
            ),
            "generation_top_reject_event_shift": float(top_event_shift),
        }
    return delta


def _run_backtest(
    *,
    config_path: Path,
    strategies_csv: str,
    from_ts: str,
    to_ts: str,
    out_json: Path,
    rebuild_signals_window: bool,
    rebuild_step_hours: float,
    rebuild_market_limit: int,
    rebuild_ingest_limit: int,
    skip_ingest_rebuild: bool,
    isolated_run_dir: Path | None,
    settle_window_end: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["ENABLE_LIVE_TRADING"] = "false"
    cycle_meta_payload: dict[str, Any] | None = None

    if rebuild_signals_window:
        if isolated_run_dir is None:
            raise ValueError("isolated_run_dir is required when rebuild_signals_window=true")

        isolated_run_dir.mkdir(parents=True, exist_ok=True)
        isolated_config = prepare_isolated_config(
            source_config_path=config_path,
            run_dir=isolated_run_dir,
            config_tag="isolated",
        )
        cycle_output_dir = isolated_run_dir / "cycle-run"
        cmd = _build_backtest_cycle_cmd(
            config_path=isolated_config,
            from_ts=from_ts,
            to_ts=to_ts,
            output_dir=cycle_output_dir,
            rebuild_step_hours=rebuild_step_hours,
            rebuild_market_limit=rebuild_market_limit,
            rebuild_ingest_limit=rebuild_ingest_limit,
            skip_ingest=skip_ingest_rebuild,
            settle_window_end=settle_window_end,
        )
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "backtest cycle failed"
                f"\ncmd: {' '.join(cmd)}"
                f"\nstdout:\n{proc.stdout.strip()}"
                f"\nstderr:\n{proc.stderr.strip()}"
            )

        latest_json = cycle_output_dir / "latest.json"
        if not latest_json.exists():
            raise RuntimeError(f"backtest cycle produced no latest.json: {cycle_output_dir}")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_json, out_json)

        cycle_meta_json = cycle_output_dir / "cycle-meta.json"
        if cycle_meta_json.exists():
            raw_cycle_meta = json.loads(cycle_meta_json.read_text())
            if isinstance(raw_cycle_meta, dict):
                cycle_meta_payload = raw_cycle_meta
    else:
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
            "--settle-window-end" if settle_window_end else "--no-settle-window-end",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            cwd=PROJECT_ROOT,
        )
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
    if isinstance(cycle_meta_payload, dict):
        payload["_cycle_meta"] = cycle_meta_payload
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
        f"- rebuild_signals_window: {bool(result.get('rebuild_signals_window', False))}",
        f"- rebuild_step_hours: {result.get('rebuild_step_hours')}",
        f"- rebuild_market_limit: {result.get('rebuild_market_limit')}",
        f"- rebuild_ingest_limit: {result.get('rebuild_ingest_limit')}",
        f"- skip_ingest_rebuild: {bool(result.get('skip_ingest_rebuild', False))}",
        f"- baseline_settle_window_end: {bool(result.get('baseline_settle_window_end', True))}",
        f"- candidate_settle_window_end: {bool(result.get('candidate_settle_window_end', True))}",
        "",
    ]

    for item in result.get("slices", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "slice"))
        hours = _safe_float(item.get("hours"))
        lines.append(f"## {label} ({hours:g}h)")

        baseline = item.get("baseline", {}) if isinstance(item.get("baseline"), dict) else {}
        candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
        base_settle = bool(baseline.get("settle_window_end", False))
        cand_settle = bool(candidate.get("settle_window_end", False))
        base_settle_source = str(baseline.get("settle_window_end_source") or "unknown")
        cand_settle_source = str(candidate.get("settle_window_end_source") or "unknown")
        lines.append(
            "- settle_window_end: "
            f"base={str(base_settle).lower()} ({base_settle_source}), "
            f"cand={str(cand_settle).lower()} ({cand_settle_source})"
        )
        lines.append("")
        lines.append(
            "| strategy | base_pnl | cand_pnl | Δpnl | base_exec | cand_exec | Δexec | "
            "base_rej | cand_rej | Δrej | base_gen_pass | cand_gen_pass | Δgen_pass | "
            "base_gen_reject | cand_gen_reject | Δgen_reject | base_maxdd | cand_maxdd | Δmaxdd | "
            "base_mtm_wr | cand_mtm_wr | Δmtm_wr | base_top_reject | cand_top_reject | "
            "base_gen_top_reject | cand_gen_top_reject | base_gen_top_event | cand_gen_top_event | "
            "Δgen_top_event_count | event_shift |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---:|"
        )

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
                f"{_safe_int(b.get('generation_pass'))} | {_safe_int(c.get('generation_pass'))} | {(_safe_int(c.get('generation_pass')) - _safe_int(b.get('generation_pass'))):+d} | "
                f"{_safe_int(b.get('generation_rejected_candidates'))} | {_safe_int(c.get('generation_rejected_candidates'))} | {(_safe_int(c.get('generation_rejected_candidates')) - _safe_int(b.get('generation_rejected_candidates'))):+d} | "
                f"{_safe_float(b.get('max_drawdown')):.4f} | {_safe_float(c.get('max_drawdown')):.4f} | {(_safe_float(d.get('max_drawdown'))):+.4f} | "
                f"{_safe_float(b.get('mtm_winrate')):.4f} | {_safe_float(c.get('mtm_winrate')):.4f} | {(_safe_float(d.get('mtm_winrate'))):+.4f} | "
                f"{str(b.get('top_reject_reason', 'none'))} | {str(c.get('top_reject_reason', 'none'))} | "
                f"{str(b.get('generation_top_reject_reason', 'none'))} | {str(c.get('generation_top_reject_reason', 'none'))} | "
                f"{str(b.get('generation_top_reject_event', 'none'))} | {str(c.get('generation_top_reject_event', 'none'))} | "
                f"{(_safe_int(c.get('generation_top_reject_event_count')) - _safe_int(b.get('generation_top_reject_event_count'))):+d} | "
                f"{_safe_int(d.get('generation_top_reject_event_shift')):+d} |"
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
        default="recent24h:24,recent7d:168,recent14d:336",
        help=("Slice specs label:hours,... " "(default: recent24h:24,recent7d:168,recent14d:336)"),
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
    parser.add_argument(
        "--rebuild-signals-window",
        action="store_true",
        help=(
            "Use backtest_cycle window rebuild path (clear + rebuild signals in slice) "
            "for each baseline/candidate run. Runs on isolated DB copies."
        ),
    )
    parser.add_argument(
        "--rebuild-step-hours",
        type=float,
        default=12.0,
        help="Step hours for rebuild-signals-window mode (default: 12)",
    )
    parser.add_argument(
        "--rebuild-market-limit",
        type=int,
        default=2000,
        help="Market limit passed to backtest_cycle in rebuild mode (default: 2000)",
    )
    parser.add_argument(
        "--rebuild-ingest-limit",
        type=int,
        default=300,
        help="Ingest limit passed to backtest_cycle in rebuild mode (default: 300)",
    )
    parser.add_argument(
        "--skip-ingest-rebuild",
        action="store_true",
        help=(
            "When rebuild-signals-window is enabled, pass --skip-ingest to backtest_cycle "
            "for deterministic baseline/candidate comparisons on the same DB snapshot."
        ),
    )
    parser.add_argument(
        "--baseline-no-settle-window-end",
        action="store_true",
        help=(
            "Run baseline slices with --no-settle-window-end (default baseline settle is on)."
        ),
    )
    parser.add_argument(
        "--candidate-no-settle-window-end",
        action="store_true",
        help=(
            "Run candidate slices with --no-settle-window-end (default candidate settle is on)."
        ),
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
        "rebuild_signals_window": bool(args.rebuild_signals_window),
        "rebuild_step_hours": float(args.rebuild_step_hours),
        "rebuild_market_limit": int(args.rebuild_market_limit),
        "rebuild_ingest_limit": int(args.rebuild_ingest_limit),
        "skip_ingest_rebuild": bool(args.skip_ingest_rebuild),
        "baseline_settle_window_end": not bool(args.baseline_no_settle_window_end),
        "candidate_settle_window_end": not bool(args.candidate_no_settle_window_end),
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

        baseline_isolated_dir = (
            out_dir / "isolated" / spec.label / "baseline" if args.rebuild_signals_window else None
        )
        candidate_isolated_dir = (
            out_dir / "isolated" / spec.label / "candidate" if args.rebuild_signals_window else None
        )

        baseline_report = _run_backtest(
            config_path=baseline_config,
            strategies_csv=strategies_csv,
            from_ts=from_ts,
            to_ts=to_ts,
            out_json=baseline_out,
            rebuild_signals_window=bool(args.rebuild_signals_window),
            rebuild_step_hours=float(args.rebuild_step_hours),
            rebuild_market_limit=int(args.rebuild_market_limit),
            rebuild_ingest_limit=int(args.rebuild_ingest_limit),
            skip_ingest_rebuild=bool(args.skip_ingest_rebuild),
            isolated_run_dir=baseline_isolated_dir,
            settle_window_end=not bool(args.baseline_no_settle_window_end),
        )
        candidate_report = _run_backtest(
            config_path=candidate_config,
            strategies_csv=strategies_csv,
            from_ts=from_ts,
            to_ts=to_ts,
            out_json=candidate_out,
            rebuild_signals_window=bool(args.rebuild_signals_window),
            rebuild_step_hours=float(args.rebuild_step_hours),
            rebuild_market_limit=int(args.rebuild_market_limit),
            rebuild_ingest_limit=int(args.rebuild_ingest_limit),
            skip_ingest_rebuild=bool(args.skip_ingest_rebuild),
            isolated_run_dir=candidate_isolated_dir,
            settle_window_end=not bool(args.candidate_no_settle_window_end),
        )

        baseline_cycle_meta = baseline_report.get("_cycle_meta")
        baseline_cycle_meta_payload = (
            baseline_cycle_meta if isinstance(baseline_cycle_meta, dict) else None
        )
        candidate_cycle_meta = candidate_report.get("_cycle_meta")
        candidate_cycle_meta_payload = (
            candidate_cycle_meta if isinstance(candidate_cycle_meta, dict) else None
        )

        baseline_settle_window_end, baseline_settle_window_end_source = _extract_settle_window_end(
            baseline_report,
            cycle_meta=baseline_cycle_meta_payload,
        )
        candidate_settle_window_end, candidate_settle_window_end_source = _extract_settle_window_end(
            candidate_report,
            cycle_meta=candidate_cycle_meta_payload,
        )

        baseline_by_strategy = {
            strategy: summarize_strategy(
                baseline_report,
                strategy=strategy,
                normalize_reject_reasons=normalize_reject_reasons,
                cycle_meta=baseline_cycle_meta_payload,
            )
            for strategy in strategies
        }
        candidate_by_strategy = {
            strategy: summarize_strategy(
                candidate_report,
                strategy=strategy,
                normalize_reject_reasons=normalize_reject_reasons,
                cycle_meta=candidate_cycle_meta_payload,
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
                    "settle_window_end": baseline_settle_window_end,
                    "settle_window_end_source": baseline_settle_window_end_source,
                    "by_strategy": baseline_by_strategy,
                },
                "candidate": {
                    "report_path": str(candidate_out),
                    "total_signals": _safe_int(candidate_report.get("total_signals")),
                    "executed_signals": _safe_int(candidate_report.get("executed_signals")),
                    "rejected_signals": _safe_int(candidate_report.get("rejected_signals")),
                    "settle_window_end": candidate_settle_window_end,
                    "settle_window_end_source": candidate_settle_window_end_source,
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
        baseline = item.get("baseline", {}) if isinstance(item.get("baseline"), dict) else {}
        candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
        print(
            f"  [{label}] settle_window_end "
            f"base={str(bool(baseline.get('settle_window_end', False))).lower()} "
            f"cand={str(bool(candidate.get('settle_window_end', False))).lower()}"
        )
        delta = item.get("delta", {}) if isinstance(item.get("delta"), dict) else {}
        for strategy in strategies:
            sd = delta.get(strategy, {}) if isinstance(delta.get(strategy), dict) else {}
            print(
                f"  [{label}] {strategy} Δpnl={_safe_float(sd.get('pnl')):+.4f} "
                f"Δexec={_safe_int(sd.get('executed_rows')):+d} "
                f"Δrej={_safe_int(sd.get('rejected_rows')):+d} "
                f"Δgen_pass={_safe_int(sd.get('generation_pass')):+d} "
                f"Δgen_reject={_safe_int(sd.get('generation_rejected_candidates')):+d} "
                f"Δgen_top_event_count={_safe_int(sd.get('generation_top_reject_event_count')):+d} "
                f"event_shift={_safe_int(sd.get('generation_top_reject_event_shift')):+d}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
