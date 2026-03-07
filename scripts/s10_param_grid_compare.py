#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


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


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_float_values(
    raw: str,
    *,
    field_name: str,
    min_value: float,
    max_value: float,
) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"invalid float for {field_name}: {text}") from exc
        if value < min_value or value > max_value:
            raise ValueError(f"{field_name} out of range [{min_value}, {max_value}]: {value}")
        if value not in values:
            values.append(value)

    if not values:
        raise ValueError(f"no values provided for {field_name}")
    return values


def build_s10_param_grid(
    *,
    prob_sum_tolerance: list[float],
    max_abs_deviation: list[float],
    max_tiny_price_leg_share: list[float],
    max_floor_adjusted_leg_share: list[float],
) -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for prob_tol, max_abs, tiny_share, floor_share in itertools.product(
        prob_sum_tolerance,
        max_abs_deviation,
        max_tiny_price_leg_share,
        max_floor_adjusted_leg_share,
    ):
        grid.append(
            {
                "prob_sum_tolerance": float(prob_tol),
                "max_abs_deviation": float(max_abs),
                "max_tiny_price_leg_share": float(tiny_share),
                "max_floor_adjusted_leg_share": float(floor_share),
            }
        )
    return grid


def apply_s10_overrides(
    base_config: dict[str, Any],
    overrides: dict[str, float],
) -> dict[str, Any]:
    out = copy.deepcopy(base_config)

    strategies_raw = out.setdefault("strategies", {})
    if not isinstance(strategies_raw, dict):
        raise ValueError("config field 'strategies' must be a mapping")

    s10_raw = strategies_raw.setdefault("s10", {})
    if not isinstance(s10_raw, dict):
        raise ValueError("config field 'strategies.s10' must be a mapping")

    for key, value in overrides.items():
        s10_raw[key] = float(value)

    return out


def summarize_compare_payload(
    payload: dict[str, Any],
    *,
    objective_strategy: str,
) -> dict[str, Any]:
    objective = objective_strategy.strip().lower()

    total_delta_pnl = 0.0
    total_delta_exec = 0
    total_delta_rej = 0
    total_delta_mtm_wr = 0.0
    total_delta_max_drawdown = 0.0

    by_slice: list[dict[str, Any]] = []
    slices_raw = payload.get("slices", [])
    slices = slices_raw if isinstance(slices_raw, list) else []

    for item in slices:
        if not isinstance(item, dict):
            continue
        delta_raw = item.get("delta", {})
        delta = delta_raw if isinstance(delta_raw, dict) else {}
        strategy_delta_raw = delta.get(objective, {})
        strategy_delta = strategy_delta_raw if isinstance(strategy_delta_raw, dict) else {}

        pnl = _safe_float(strategy_delta.get("pnl"))
        exec_rows = _safe_int(strategy_delta.get("executed_rows"))
        rej_rows = _safe_int(strategy_delta.get("rejected_rows"))
        mtm_wr = _safe_float(strategy_delta.get("mtm_winrate"))
        max_drawdown = _safe_float(strategy_delta.get("max_drawdown"))

        total_delta_pnl += pnl
        total_delta_exec += exec_rows
        total_delta_rej += rej_rows
        total_delta_mtm_wr += mtm_wr
        total_delta_max_drawdown += max_drawdown

        by_slice.append(
            {
                "label": str(item.get("label", "slice")),
                "hours": _safe_float(item.get("hours")),
                "delta_pnl": pnl,
                "delta_exec": exec_rows,
                "delta_rej": rej_rows,
                "delta_mtm_winrate": mtm_wr,
                "delta_max_drawdown": max_drawdown,
            }
        )

    min_slice_delta_pnl = min((item["delta_pnl"] for item in by_slice), default=0.0)
    non_negative_slice_count = sum(1 for item in by_slice if float(item["delta_pnl"]) >= 0.0)

    return {
        "objective_strategy": objective,
        "slice_count": len(by_slice),
        "total_delta_pnl": total_delta_pnl,
        "total_delta_exec": total_delta_exec,
        "total_delta_rej": total_delta_rej,
        "total_delta_mtm_winrate": total_delta_mtm_wr,
        "total_delta_max_drawdown": total_delta_max_drawdown,
        "min_slice_delta_pnl": min_slice_delta_pnl,
        "non_negative_slice_count": non_negative_slice_count,
        "by_slice": by_slice,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# S10 Parameter Grid Compare",
        "",
        f"- generated_at: {result.get('generated_at')}",
        f"- anchor_ts: {result.get('anchor_ts')}",
        f"- baseline_config: {result.get('baseline_config')}",
        f"- candidate_base_config: {result.get('candidate_base_config')}",
        f"- objective_strategy: {result.get('objective_strategy')}",
        f"- min_slice_delta_pnl_threshold: {result.get('min_slice_delta_pnl_threshold')}",
        f"- total_candidates: {result.get('total_candidates')}",
        "",
        "| rank | candidate | prob_tol | max_abs | tiny_share | floor_share | "
        "min(Δpnl) | ΣΔpnl | ΣΔexec | ΣΔrej | ΣΔmaxDD | ΣΔmtm_wr | pass? |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for item in result.get("candidates", []):
        if not isinstance(item, dict):
            continue
        ov_raw = item.get("overrides", {})
        ov = ov_raw if isinstance(ov_raw, dict) else {}
        lines.append(
            "| "
            f"{_safe_int(item.get('rank'))} | "
            f"{str(item.get('candidate_id', 'cand'))} | "
            f"{_safe_float(ov.get('prob_sum_tolerance')):.4f} | "
            f"{_safe_float(ov.get('max_abs_deviation')):.4f} | "
            f"{_safe_float(ov.get('max_tiny_price_leg_share')):.4f} | "
            f"{_safe_float(ov.get('max_floor_adjusted_leg_share')):.4f} | "
            f"{_safe_float(item.get('min_slice_delta_pnl')):+.4f} | "
            f"{_safe_float(item.get('total_delta_pnl')):+.4f} | "
            f"{_safe_int(item.get('total_delta_exec')):+d} | "
            f"{_safe_int(item.get('total_delta_rej')):+d} | "
            f"{_safe_float(item.get('total_delta_max_drawdown')):+.4f} | "
            f"{_safe_float(item.get('total_delta_mtm_winrate')):+.4f} | "
            f"{'yes' if bool(item.get('passes_min_slice_delta_pnl')) else 'no'} |"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def _run_dual_slice_compare(
    *,
    compare_script: Path,
    baseline_config: Path,
    candidate_config: Path,
    strategies: str,
    slices: str,
    out_dir: Path,
    anchor_ts: str | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(compare_script),
        "--baseline-config",
        str(baseline_config),
        "--candidate-config",
        str(candidate_config),
        "--strategies",
        strategies,
        "--slices",
        slices,
        "--out-dir",
        str(out_dir),
    ]
    if anchor_ts:
        cmd.extend(["--anchor-ts", anchor_ts])

    env = os.environ.copy()
    env["ENABLE_LIVE_TRADING"] = "false"

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "sx12_dual_slice_compare failed"
            f"\ncmd: {' '.join(cmd)}"
            f"\nstdout:\n{proc.stdout.strip()}"
            f"\nstderr:\n{proc.stderr.strip()}"
        )

    compare_json = out_dir / "compare.json"
    if not compare_json.exists():
        raise RuntimeError(f"missing compare artifact: {compare_json}")
    payload = json.loads(compare_json.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid compare artifact payload: {compare_json}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run S10 parameter grid search by generating candidate config variants and "
            "calling scripts/sx12_dual_slice_compare.py for each candidate."
        )
    )
    parser.add_argument("--baseline-config", required=True, help="Baseline config path")
    parser.add_argument(
        "--candidate-base-config",
        default=None,
        help="Base candidate config path (default: baseline config)",
    )
    parser.add_argument(
        "--strategies",
        default="s9,s10",
        help="Comma-separated strategies passed to sx12 compare (default: s9,s10)",
    )
    parser.add_argument(
        "--slices",
        default="recent24h:24,recent7d:168",
        help="Slice specs passed to sx12 compare",
    )
    parser.add_argument("--anchor-ts", default=None, help="Anchor ISO timestamp (UTC)")
    parser.add_argument(
        "--objective-strategy",
        default="s10",
        help="Strategy used for aggregate ranking objective",
    )
    parser.add_argument(
        "--min-slice-delta-pnl",
        type=float,
        default=-1_000_000_000.0,
        help=(
            "Minimum per-slice Δpnl required for pass (default keeps all candidates). "
            "Use 0 to require non-negative Δpnl on every slice."
        ),
    )

    parser.add_argument(
        "--prob-sum-tolerance-grid",
        default="0.015,0.02,0.03",
        help="Comma-separated grid for strategies.s10.prob_sum_tolerance",
    )
    parser.add_argument(
        "--max-abs-deviation-grid",
        default="0.15,0.20,0.25",
        help="Comma-separated grid for strategies.s10.max_abs_deviation",
    )
    parser.add_argument(
        "--max-tiny-price-leg-share-grid",
        default="0.20,0.25,0.35",
        help="Comma-separated grid for strategies.s10.max_tiny_price_leg_share",
    )
    parser.add_argument(
        "--max-floor-adjusted-leg-share-grid",
        default="0.25,0.35,0.50",
        help="Comma-separated grid for strategies.s10.max_floor_adjusted_leg_share",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Cap candidates evaluated (0 means all combinations)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: artifacts/backtest/s10-grid-<anchor>)",
    )
    parser.add_argument(
        "--compare-script",
        default="scripts/sx12_dual_slice_compare.py",
        help="Path to sx12 dual-slice compare script",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    baseline_config = Path(args.baseline_config).resolve()
    candidate_base_config = (
        Path(args.candidate_base_config).resolve()
        if args.candidate_base_config
        else baseline_config
    )
    compare_script = Path(args.compare_script).resolve()

    if not baseline_config.exists():
        raise FileNotFoundError(f"baseline config not found: {baseline_config}")
    if not candidate_base_config.exists():
        raise FileNotFoundError(f"candidate base config not found: {candidate_base_config}")
    if not compare_script.exists():
        raise FileNotFoundError(f"compare script not found: {compare_script}")

    base_raw = yaml.safe_load(candidate_base_config.read_text())
    base_config = base_raw if isinstance(base_raw, dict) else {}

    grid = build_s10_param_grid(
        prob_sum_tolerance=parse_float_values(
            str(args.prob_sum_tolerance_grid),
            field_name="prob_sum_tolerance_grid",
            min_value=0.0,
            max_value=1.0,
        ),
        max_abs_deviation=parse_float_values(
            str(args.max_abs_deviation_grid),
            field_name="max_abs_deviation_grid",
            min_value=0.0,
            max_value=1.0,
        ),
        max_tiny_price_leg_share=parse_float_values(
            str(args.max_tiny_price_leg_share_grid),
            field_name="max_tiny_price_leg_share_grid",
            min_value=0.0,
            max_value=1.0,
        ),
        max_floor_adjusted_leg_share=parse_float_values(
            str(args.max_floor_adjusted_leg_share_grid),
            field_name="max_floor_adjusted_leg_share_grid",
            min_value=0.0,
            max_value=1.0,
        ),
    )

    if args.max_candidates > 0:
        grid = grid[: int(args.max_candidates)]

    anchor = datetime.now(UTC)
    anchor_z = _iso_z(anchor)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (
            Path("artifacts/backtest") / f"s10-grid-{anchor.strftime('%Y%m%dT%H%M%SZ')}"
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for idx, overrides in enumerate(grid, start=1):
        candidate_id = f"cand-{idx:03d}"
        candidate_cfg = apply_s10_overrides(base_config, overrides)

        candidate_cfg_path = out_dir / "candidates" / f"{candidate_id}.yaml"
        candidate_cfg_path.write_text(yaml.safe_dump(candidate_cfg, sort_keys=False))

        compare_out_dir = out_dir / "runs" / candidate_id
        compare_out_dir.mkdir(parents=True, exist_ok=True)

        compare_payload = _run_dual_slice_compare(
            compare_script=compare_script,
            baseline_config=baseline_config,
            candidate_config=candidate_cfg_path,
            strategies=str(args.strategies),
            slices=str(args.slices),
            out_dir=compare_out_dir,
            anchor_ts=str(args.anchor_ts) if args.anchor_ts else None,
        )
        summary = summarize_compare_payload(
            compare_payload,
            objective_strategy=str(args.objective_strategy),
        )

        entry = {
            "candidate_id": candidate_id,
            "overrides": overrides,
            "candidate_config": str(candidate_cfg_path),
            "compare_json": str(compare_out_dir / "compare.json"),
            **summary,
        }
        entry["passes_min_slice_delta_pnl"] = _safe_float(
            entry.get("min_slice_delta_pnl")
        ) >= float(args.min_slice_delta_pnl)
        entries.append(entry)

        print(
            f"[{candidate_id}]"
            f" min(Δpnl)={_safe_float(entry.get('min_slice_delta_pnl')):+.4f}"
            f" ΣΔpnl={_safe_float(entry.get('total_delta_pnl')):+.4f}"
            f" ΣΔexec={_safe_int(entry.get('total_delta_exec')):+d}"
            f" ΣΔrej={_safe_int(entry.get('total_delta_rej')):+d}"
            f" ΣΔmaxDD={_safe_float(entry.get('total_delta_max_drawdown')):+.4f}"
            f" pass={bool(entry.get('passes_min_slice_delta_pnl'))}"
            f" overrides={overrides}"
        )

    entries.sort(
        key=lambda item: (
            -int(bool(item.get("passes_min_slice_delta_pnl"))),
            -_safe_float(item.get("min_slice_delta_pnl")),
            -_safe_float(item.get("total_delta_pnl")),
            -_safe_int(item.get("total_delta_exec")),
            _safe_int(item.get("total_delta_rej")),
            _safe_float(item.get("total_delta_max_drawdown")),
            str(item.get("candidate_id", "")),
        )
    )
    for rank, item in enumerate(entries, start=1):
        item["rank"] = rank

    result: dict[str, Any] = {
        "generated_at": _iso_z(datetime.now(UTC)),
        "anchor_ts": anchor_z,
        "baseline_config": str(baseline_config),
        "candidate_base_config": str(candidate_base_config),
        "objective_strategy": str(args.objective_strategy).strip().lower(),
        "strategies": [s.strip().lower() for s in str(args.strategies).split(",") if s.strip()],
        "slices": str(args.slices),
        "min_slice_delta_pnl_threshold": float(args.min_slice_delta_pnl),
        "total_candidates": len(entries),
        "candidates": entries,
    }

    summary_json = out_dir / "grid-results.json"
    summary_md = out_dir / "grid-results.md"
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary_md.write_text(render_markdown(result))

    print(f"s10 grid compare completed: {out_dir}")
    print(f"  summary json: {summary_json}")
    print(f"  summary md: {summary_md}")
    if entries:
        top = entries[0]
        print(
            "  best: "
            f"{top.get('candidate_id')} "
            f"min(Δpnl)={_safe_float(top.get('min_slice_delta_pnl')):+.4f} "
            f"ΣΔpnl={_safe_float(top.get('total_delta_pnl')):+.4f} "
            f"ΣΔexec={_safe_int(top.get('total_delta_exec')):+d} "
            f"ΣΔrej={_safe_int(top.get('total_delta_rej')):+d} "
            f"ΣΔmaxDD={_safe_float(top.get('total_delta_max_drawdown')):+.4f} "
            f"pass={bool(top.get('passes_min_slice_delta_pnl'))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
