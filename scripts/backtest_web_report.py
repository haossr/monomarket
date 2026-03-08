#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from typing import Any

RAW_REQUIRED_FILENAMES = ("latest.json", "strategy.csv", "event.csv", "replay.csv")
RAW_OPTIONAL_FILENAMES = ("cycle-meta.json",)
MAX_REPLAY_CHART_POINTS = 120


@dataclass
class StrategyRow:
    strategy: str
    pnl: float
    trade_count: int
    winrate: float
    winrate_label: str
    max_drawdown: float


@dataclass
class AssumptionRow:
    item: str
    value: str
    source: str


def _iso_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static HTML report for one backtest run")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing latest.json/strategy.csv/event.csv/replay.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: artifacts/backtest/web/<timestamp>/",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="How many recent run points to include in history chart.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_raw_artifacts(run_dir: Path, output_dir: Path) -> dict[str, str]:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for name in (*RAW_REQUIRED_FILENAMES, *RAW_OPTIONAL_FILENAMES):
        src = run_dir / name
        if src.exists():
            dst = raw_dir / name
            copy2(src, dst)
            links[name] = f"raw/{name}"
    return links


def _parse_iso_maybe(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _fmt_num(v: float, digits: int = 4) -> str:
    return f"{v:,.{digits}f}"


def _fmt_pct(v: float) -> str:
    return f"{(v * 100):.2f}%"


def _fmt_opt(value: Any) -> str:
    if value is None:
        return "n/a"
    text = str(value).strip()
    return text if text else "n/a"


def _as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _get_nested(data: dict[str, Any] | None, path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_non_none(candidates: list[tuple[str, Any]]) -> tuple[str, Any | None]:
    for source, value in candidates:
        if value is not None:
            return source, value
    return "n/a", None


def _collect_strategies(payload: dict[str, Any]) -> list[StrategyRow]:
    rows: list[StrategyRow] = []
    for item in payload.get("results", []):
        closed_samples = int(item.get("closed_sample_count", 0) or 0)
        mtm_samples = int(item.get("mtm_sample_count", 0) or 0)
        if closed_samples > 0:
            winrate = float(item.get("closed_winrate", 0.0) or 0.0)
            label = "closed_winrate"
        elif mtm_samples > 0:
            winrate = float(item.get("mtm_winrate", 0.0) or 0.0)
            label = "mtm_winrate"
        else:
            winrate = float(item.get("winrate", 0.0) or 0.0)
            label = "winrate"
        rows.append(
            StrategyRow(
                strategy=str(item.get("strategy", "")),
                pnl=float(item.get("pnl", 0.0) or 0.0),
                trade_count=int(item.get("trade_count", 0) or 0),
                winrate=winrate,
                winrate_label=label,
                max_drawdown=float(item.get("max_drawdown", 0.0) or 0.0),
            )
        )
    rows.sort(key=lambda x: x.strategy)
    return rows


def _collect_replay_timeline(payload: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    replay = payload.get("replay", [])

    records: list[tuple[datetime | None, str, str, float, float | None]] = []

    def _as_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    if isinstance(replay, list) and replay:
        for row in replay:
            if not isinstance(row, dict):
                continue
            ts = str(row.get("ts", ""))
            strategy = str(row.get("strategy", "")).strip().lower() or "unknown"
            change = _as_float_or_none(row.get("realized_change")) or 0.0
            strategy_equity = _as_float_or_none(row.get("strategy_equity"))
            records.append((_parse_iso_maybe(ts), ts, strategy, change, strategy_equity))
    else:
        replay_csv = run_dir / "replay.csv"
        if replay_csv.exists():
            with replay_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    ts = str(row.get("ts", ""))
                    strategy = str(row.get("strategy", "")).strip().lower() or "unknown"
                    change = _as_float_or_none(row.get("realized_change")) or 0.0
                    strategy_equity = _as_float_or_none(row.get("strategy_equity"))
                    records.append((_parse_iso_maybe(ts), ts, strategy, change, strategy_equity))

    records.sort(
        key=lambda x: (
            x[0] or datetime.min.replace(tzinfo=UTC),
            x[1],
            x[2],
        )
    )

    # Build both realized-only and total-equity(MTM) timeline.
    cumulative_realized = 0.0
    last_equity_by_strategy: dict[str, float] = {}
    points: list[tuple[datetime | None, str, float, float]] = []

    for dt, ts, strategy, change, strategy_equity in records:
        cumulative_realized += change
        if strategy_equity is not None:
            last_equity_by_strategy[strategy] = strategy_equity
        total_equity = (
            sum(last_equity_by_strategy.values())
            if last_equity_by_strategy
            else cumulative_realized
        )
        points.append((dt, ts, cumulative_realized, total_equity))

    # Collapse identical timestamps (keep latest values at that ts) to reduce clutter.
    collapsed: list[tuple[datetime | None, str, float, float]] = []
    for dt, ts, cumulative_realized, total_equity in points:
        key = ts.strip()
        if collapsed and collapsed[-1][1] == key:
            collapsed[-1] = (dt, key, cumulative_realized, total_equity)
        else:
            collapsed.append((dt, key, cumulative_realized, total_equity))

    # Downsample very dense timelines for mobile readability.
    if len(collapsed) > MAX_REPLAY_CHART_POINTS:
        target = MAX_REPLAY_CHART_POINTS
        span = len(collapsed) - 1
        idxs = sorted({round(i * span / (target - 1)) for i in range(target)})
        collapsed = [collapsed[i] for i in idxs]

    out: list[dict[str, Any]] = []
    for idx, (_, ts, cumulative_realized, total_equity) in enumerate(collapsed, start=1):
        short_label = ts
        dt = _parse_iso_maybe(ts)
        if dt is not None:
            short_label = dt.strftime("%m-%d %H:%M")
        elif not short_label:
            short_label = f"#{idx}"
        out.append(
            {
                "label": short_label,
                "full_label": ts or short_label,
                "index": idx,
                "cumulative_realized_pnl": cumulative_realized,
                "cumulative_total_equity": total_equity,
            }
        )
    return out


def _collect_run_history(run_dir: Path, history_limit: int) -> list[dict[str, Any]]:
    runs_root = run_dir.parent
    entries: list[dict[str, Any]] = []
    if not runs_root.exists():
        return entries

    for child in runs_root.iterdir():
        latest_path = child / "latest.json"
        if not latest_path.exists():
            continue
        try:
            payload = _load_json(latest_path)
        except Exception:
            continue

        total_pnl = sum(float(r.get("pnl", 0.0) or 0.0) for r in payload.get("results", []))
        total_signals = int(payload.get("total_signals", 0) or 0)
        executed_signals = int(payload.get("executed_signals", 0) or 0)
        execution_rate = (executed_signals / total_signals) if total_signals > 0 else 0.0

        generated_at_raw = payload.get("generated_at")
        generated_at_dt = _parse_iso_maybe(generated_at_raw)
        if generated_at_dt is None:
            generated_at_dt = datetime.fromtimestamp(latest_path.stat().st_mtime, tz=UTC)
        generated_at = generated_at_dt.isoformat()

        entries.append(
            {
                "run": child.name,
                "generated_at": generated_at,
                "generated_at_dt": generated_at_dt,
                "from_ts": _fmt_opt(payload.get("from_ts")),
                "to_ts": _fmt_opt(payload.get("to_ts")),
                "total_pnl": total_pnl,
                "total_signals": total_signals,
                "executed_signals": executed_signals,
                "execution_rate": execution_rate,
            }
        )

    entries.sort(key=lambda item: item["generated_at_dt"])
    if history_limit > 0:
        entries = entries[-history_limit:]

    for row in entries:
        row.pop("generated_at_dt", None)
    return entries


def _collect_assumptions(
    payload: dict[str, Any], cycle_meta: dict[str, Any], strategy_rows: list[StrategyRow]
) -> list[AssumptionRow]:
    strategy_set: set[str] = {r.strategy for r in strategy_rows if r.strategy}
    strategy_set.update(
        str(item.get("strategy", ""))
        for item in payload.get("replay", [])
        if str(item.get("strategy", "")).strip()
    )

    by_strategy = _get_nested(cycle_meta, "signal_generation.edge_gate.by_strategy")
    if isinstance(by_strategy, dict):
        strategy_set.update(str(k) for k in by_strategy.keys())

    live_source, live_value = _first_non_none(
        [
            (
                "latest.json.trading.enable_live_trading",
                _get_nested(payload, "trading.enable_live_trading"),
            ),
            ("latest.json.enable_live_trading", payload.get("enable_live_trading")),
            (
                "latest.json.execution_config.enable_live_trading",
                _get_nested(payload, "execution_config.enable_live_trading"),
            ),
            (
                "cycle-meta.json.trading.enable_live_trading",
                _get_nested(cycle_meta, "trading.enable_live_trading"),
            ),
            (
                "cycle-meta.json.enable_live_trading",
                _get_nested(cycle_meta, "enable_live_trading"),
            ),
        ]
    )
    live_status_bool = _as_bool_or_none(live_value)
    if live_status_bool is None:
        live_status = "n/a"
    else:
        live_status = "disabled" if not live_status_bool else "enabled"

    execution_cfg = payload.get("execution_config", {})
    risk_cfg = payload.get("risk_config", {})

    fill_mode = (
        "partial_fill="
        f"{_fmt_opt(execution_cfg.get('enable_partial_fill'))}, "
        "fill_probability="
        f"{_fmt_opt(execution_cfg.get('enable_fill_probability'))}, "
        "dynamic_slippage="
        f"{_fmt_opt(execution_cfg.get('enable_dynamic_slippage'))}"
    )
    slippage_fee = (
        f"slippage_bps={_fmt_opt(execution_cfg.get('slippage_bps'))}, "
        f"fee_bps={_fmt_opt(execution_cfg.get('fee_bps'))}"
    )
    risk_limits = (
        f"max_daily_loss={_fmt_opt(risk_cfg.get('max_daily_loss'))}, "
        f"max_strategy_notional={_fmt_opt(risk_cfg.get('max_strategy_notional'))}, "
        f"max_event_notional={_fmt_opt(risk_cfg.get('max_event_notional'))}, "
        "circuit_breaker_rejections="
        f"{_fmt_opt(risk_cfg.get('circuit_breaker_rejections'))}"
    )

    signal_mode = (
        f"fixed_window_mode={_fmt_opt(cycle_meta.get('fixed_window_mode'))}, "
        "historical_replay_only="
        f"{_fmt_opt(_get_nested(cycle_meta, 'signal_generation.historical_replay_only'))}"
    )
    signal_window_controls = (
        f"clear_signals_window={_fmt_opt(_get_nested(cycle_meta, 'signal_generation.clear_signals_window'))}, "
        "rebuild_signals_window="
        f"{_fmt_opt(_get_nested(cycle_meta, 'signal_generation.rebuild_signals_window'))}, "
        "rebuild_step_hours="
        f"{_fmt_opt(_get_nested(cycle_meta, 'signal_generation.rebuild_step_hours'))}"
    )

    assumptions = [
        AssumptionRow(
            item="Window (from -> to)",
            value=f"{_fmt_opt(payload.get('from_ts'))} -> {_fmt_opt(payload.get('to_ts'))}",
            source="latest.json.from_ts/to_ts",
        ),
        AssumptionRow(
            item="Strategies",
            value=", ".join(sorted(strategy_set)) if strategy_set else "n/a",
            source="latest.json.results/replay + cycle-meta.json.signal_generation.edge_gate.by_strategy",
        ),
        AssumptionRow(
            item="Live trading",
            value=live_status,
            source=live_source,
        ),
        AssumptionRow(
            item="Fill model",
            value=fill_mode,
            source="latest.json.execution_config.*",
        ),
        AssumptionRow(
            item="Slippage / Fee",
            value=slippage_fee,
            source="latest.json.execution_config.slippage_bps/fee_bps",
        ),
        AssumptionRow(
            item="Risk limits",
            value=risk_limits,
            source="latest.json.risk_config.*",
        ),
        AssumptionRow(
            item="Signal generation mode",
            value=signal_mode,
            source="cycle-meta.json.fixed_window_mode + signal_generation.historical_replay_only",
        ),
        AssumptionRow(
            item="Signal window controls",
            value=signal_window_controls,
            source="cycle-meta.json.signal_generation.*",
        ),
    ]
    return assumptions


def _cycle_reject_reasons(cycle_meta: dict[str, Any], strategy: str, top_k: int = 3) -> list[str]:
    reasons = _get_nested(
        cycle_meta,
        f"signal_generation.edge_gate.by_strategy.{strategy}.strategy_diagnostics.candidate_reject_reasons",
    )
    if not isinstance(reasons, dict):
        return []

    ranked = sorted(
        ((str(reason), int(count)) for reason, count in reasons.items()),
        key=lambda item: (-item[1], item[0]),
    )
    out = [f"{reason}: {count}" for reason, count in ranked[:top_k] if count > 0]
    return out


def _strategy_catalog(cycle_meta: dict[str, Any]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = [
        {
            "strategy": "s1",
            "name": "Cross Venue Scanner",
            "core_logic": "同 canonical 或同 event 下，寻找 YES 价格低腿与高腿的价差并打分。",
            "signal_trigger": "当 spread >= min_spread（或 min_event_spread）且满足流动性门槛时，生成 buy 信号。",
            "constraints": [
                "strategies.s1.min_spread",
                "strategies.s1.min_event_spread",
                "strategies.s1.min_liquidity",
                "strategies.s1.max_order_notional",
                "strategies.s1.max_signals_per_event",
                "strategies.s1.event_notional_cap",
                "strategies.s1.inventory_decay",
                "strategies.s1.enable_cross_market_arb",
            ],
            "known_risks_or_rejects": [
                "max_order_notional <= 0 会直接禁用策略（返回空信号）。",
                "价差不足/流动性不足时不会发信号，事件预算与每事件信号上限会进一步裁剪。",
            ],
            "source_refs": [
                "docs/strategies.md",
                "src/monomarket/signals/strategies/s1_cross_venue.py",
            ],
        },
        {
            "strategy": "s2",
            "name": "NegRisk Rebalance",
            "core_logic": "对 neg_risk=true 的同事件市场，计算 sum(yes) 偏离 1 的程度，做篮子再平衡。",
            "signal_trigger": "|sum_yes - 1| >= prob_sum_tolerance 时触发，偏离<0 买入，偏离>0 卖出。",
            "constraints": [
                "strategies.s2.prob_sum_tolerance",
                "strategies.s2.max_order_notional",
                "strategies.s2.exclude_event_ids",
                "strategies.s2.edge_gate_min_bps",
                "strategies.s2.edge_gate_budget_penalty_bps",
                "strategies.s2.edge_gate_budget_cap_notional",
            ],
            "known_risks_or_rejects": [
                "偏离不足容差或事件被 exclude_event_ids 命中时不触发。",
                "本地 edge gate 的 effective_edge_bps 低于门槛时拒绝输出。",
            ],
            "source_refs": [
                "docs/strategies.md",
                "src/monomarket/signals/strategies/s2_negrisk_rebalance.py",
            ],
        },
        {
            "strategy": "s4",
            "name": "Low Prob YES Basket",
            "core_logic": "在低 yes 价格区间内构建篮子，按 A/B/C 分层与阶梯报价下单。",
            "signal_trigger": "yes 价格、no 价格、edge buffer、流动性等条件同时满足时触发 buy。",
            "constraints": [
                "strategies.s4.yes_price_min",
                "strategies.s4.yes_price_max",
                "strategies.s4.min_edge_buffer",
                "strategies.s4.min_no_price",
                "strategies.s4.min_liquidity",
                "strategies.s4.max_candidates",
                "strategies.s4.max_order_notional",
                "strategies.s4.max_signals_per_event",
                "strategies.s4.event_notional_cap",
                "strategies.s4.inventory_decay",
                "strategies.s4.exclude_event_ids",
            ],
            "known_risks_or_rejects": [
                "价格/流动性条件不达标不会入篮；exclude_event_ids 会直接过滤事件。",
                "事件预算与 max_signals_per_event 会导致候选被裁剪。",
            ],
            "source_refs": [
                "docs/strategies.md",
                "src/monomarket/signals/strategies/s4_low_prob_yes.py",
            ],
        },
        {
            "strategy": "s8",
            "name": "NO Carry + Tail Hedge",
            "core_logic": "主仓做高胜率 NO carry，尾部用超低 yes 市场做 hedge。",
            "signal_trigger": "yes<=yes_price_max_for_no 且流动性达标时触发，按 hedge_budget_ratio 生成对冲建议。",
            "constraints": [
                "strategies.s8.yes_price_max_for_no",
                "strategies.s8.hedge_budget_ratio",
                "strategies.s8.max_order_notional",
                "strategies.s8.max_candidates",
                "strategies.s8.max_signals_per_event",
                "strategies.s8.exclude_event_ids",
                "strategies.s8.edge_gate_min_bps",
                "strategies.s8.edge_gate_budget_penalty_bps",
                "strategies.s8.edge_gate_budget_cap_notional",
            ],
            "known_risks_or_rejects": [
                "主仓/对冲池流动性不足时不会入选。",
                "local edge gate（effective_edge_bps）低于阈值会拒绝发信号。",
            ],
            "source_refs": [
                "docs/strategies.md",
                "src/monomarket/signals/strategies/s8_no_carry_tailhedge.py",
            ],
        },
        {
            "strategy": "s9",
            "name": "YES/NO Parity Arb",
            "core_logic": "同条件 YES+NO 平价套利：sum<1 做 buy carry，sum>1（可选）做 sell overround。",
            "signal_trigger": "满足同市场/同事件/同条件配对门槛后，effective edge 超过阈值才发双腿信号。",
            "constraints": [
                "strategies.s9.min_leg_liquidity",
                "strategies.s9.min_effective_edge_bps",
                "strategies.s9.max_total_cost_bps",
                "strategies.s9.max_order_notional",
                "strategies.s9.max_pairs_per_event",
                "strategies.s9.max_event_pair_notional",
                "strategies.s9.require_same_market",
                "strategies.s9.require_same_event",
                "strategies.s9.require_same_condition",
                "strategies.s9.cross_market_require_same_source",
                "strategies.s9.cross_market_extra_edge_bps",
                "strategies.s9.executable_price_floor",
            ],
            "known_risks_or_rejects": [
                "常见拒因包含 pair_not_found_*、non_positive_gross_edge、effective_edge_below_min。",
                "事件配额与成本上限（total_cost_cap / event_notional_budget）会拒绝候选。",
            ],
            "source_refs": [
                "src/monomarket/signals/strategies/s9_yes_no_parity.py",
            ],
        },
        {
            "strategy": "s10",
            "name": "NegRisk Conversion Arb",
            "core_logic": "在 negRisk 事件上做转换套利：sum(yes)<1-tol 买入篮子，sum(yes)>1+tol（可选）卖出。",
            "signal_trigger": "事件满足 unique canonical、deviation 与成本/edge 门槛后，发 basket 原子多腿信号。",
            "constraints": [
                "strategies.s10.prob_sum_tolerance",
                "strategies.s10.max_abs_deviation",
                "strategies.s10.min_unique_canonicals",
                "strategies.s10.max_leg_weight",
                "strategies.s10.max_legs_per_event",
                "strategies.s10.min_effective_edge_bps",
                "strategies.s10.max_weighted_total_cost_bps",
                "strategies.s10.max_leg_total_cost_bps",
                "strategies.s10.max_tiny_price_leg_share",
                "strategies.s10.max_floor_adjusted_leg_share",
                "strategies.s10.exclude_event_ids",
            ],
            "known_risks_or_rejects": [
                "常见拒因包含 deviation_not_below_tolerance、event_under_min_unique、tiny_price_leg_share_exceeded。",
                "若 weighted/leg cost 超限、effective_edge_below_min，则整篮子拒绝。",
            ],
            "source_refs": [
                "src/monomarket/signals/strategies/s10_negrisk_conversion.py",
            ],
        },
    ]

    for row in catalog:
        cycle_rejects = _cycle_reject_reasons(cycle_meta, row["strategy"])
        row["cycle_rejects"] = cycle_rejects

    return catalog


def _build_chart_data(
    *,
    strategy_rows: list[StrategyRow],
    replay_timeline: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cumulative_pnl": {
            "labels": [row["label"] for row in replay_timeline],
            "full_labels": [row.get("full_label", row["label"]) for row in replay_timeline],
            "realized_values": [row["cumulative_realized_pnl"] for row in replay_timeline],
            "total_values": [
                row.get("cumulative_total_equity", row["cumulative_realized_pnl"])
                for row in replay_timeline
            ],
        },
        "strategy_pnl": {
            "labels": [row.strategy for row in strategy_rows],
            "values": [row.pnl for row in strategy_rows],
        },
        "strategy_trade_count": {
            "labels": [row.strategy for row in strategy_rows],
            "values": [row.trade_count for row in strategy_rows],
        },
        "strategy_winrate": {
            "labels": [row.strategy for row in strategy_rows],
            "values": [row.winrate * 100.0 for row in strategy_rows],
            "sources": [row.winrate_label for row in strategy_rows],
        },
        "history_total_pnl": {
            "labels": [row["run"] for row in run_history],
            "values": [row["total_pnl"] for row in run_history],
        },
    }


def _render_html(
    *,
    payload: dict[str, Any],
    cycle_meta: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    links: dict[str, str],
    history_limit: int,
) -> str:
    strategy_rows = _collect_strategies(payload)
    assumptions = _collect_assumptions(payload, cycle_meta, strategy_rows)
    strategy_catalog = _strategy_catalog(cycle_meta)
    replay_timeline = _collect_replay_timeline(payload, run_dir)
    run_history = _collect_run_history(run_dir, history_limit=history_limit)
    chart_data = _build_chart_data(
        strategy_rows=strategy_rows,
        replay_timeline=replay_timeline,
        run_history=run_history,
    )

    total_pnl = sum(r.pnl for r in strategy_rows)
    max_dd = max((r.max_drawdown for r in strategy_rows), default=0.0)

    table_rows = "\n".join(
        "".join(
            [
                "<tr>",
                f"<td>{html.escape(r.strategy)}</td>",
                f"<td class='num'>{_fmt_num(r.pnl)}</td>",
                f"<td class='num'>{r.trade_count}</td>",
                f"<td class='num'>{_fmt_pct(r.winrate)}</td>",
                f"<td>{html.escape(r.winrate_label)}</td>",
                "</tr>",
            ]
        )
        for r in strategy_rows
    )

    realized_pnl = (
        float(replay_timeline[-1].get("cumulative_realized_pnl", 0.0)) if replay_timeline else 0.0
    )

    summary_items = [
        ("Total PnL (MTM)", _fmt_num(total_pnl)),
        ("Realized PnL", _fmt_num(realized_pnl)),
        ("Max Drawdown", _fmt_num(max_dd)),
        ("Executed Signals", str(int(payload.get("executed_signals", 0) or 0))),
        ("Rejected Signals", str(int(payload.get("rejected_signals", 0) or 0))),
        ("History points", str(len(run_history))),
    ]
    cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(k)}</div>"
        f"<div class='value'>{html.escape(v)}</div></div>"
        for k, v in summary_items
    )

    assumption_rows = "\n".join(
        "".join(
            [
                "<tr>",
                f"<td>{html.escape(row.item)}</td>",
                f"<td>{html.escape(row.value)}</td>",
                f"<td><code>{html.escape(row.source)}</code></td>",
                "</tr>",
            ]
        )
        for row in assumptions
    )

    catalog_rows = []
    for row in strategy_catalog:
        constraints = "".join(
            f"<li><code>{html.escape(str(item))}</code></li>" for item in row["constraints"]
        )
        known_rejects = "".join(
            f"<li>{html.escape(str(item))}</li>" for item in row["known_risks_or_rejects"]
        )
        cycle_rejects = row.get("cycle_rejects", [])
        cycle_rejects_html = (
            "".join(f"<li>{html.escape(str(item))}</li>" for item in cycle_rejects)
            if cycle_rejects
            else "<li>n/a</li>"
        )
        refs = "".join(
            f"<li><code>{html.escape(str(ref))}</code></li>" for ref in row["source_refs"]
        )
        catalog_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td><strong>{html.escape(row['strategy'])}</strong><br/>"
                    f"<span class='muted'>{html.escape(row['name'])}</span></td>",
                    f"<td>{html.escape(row['core_logic'])}</td>",
                    f"<td>{html.escape(row['signal_trigger'])}</td>",
                    f"<td><ul>{constraints}</ul></td>",
                    f"<td><ul>{known_rejects}</ul><div class='muted'>"
                    "本次 cycle 常见拒因</div><ul>"
                    f"{cycle_rejects_html}</ul></td>",
                    f"<td><ul>{refs}</ul></td>",
                    "</tr>",
                ]
            )
        )

    link_items = "\n".join(
        f"<li><a href='{html.escape(rel)}'>{html.escape(name)}</a></li>"
        for name, rel in links.items()
    )

    chart_data_json = json.dumps(chart_data, ensure_ascii=False)

    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Monomarket Backtest Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 16px; color: #111; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #555; margin-bottom: 18px; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fafafa; }}
    .label {{ font-size: 12px; color: #666; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; min-width: 640px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; font-size: 14px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f3f4f6; border-radius: 4px; padding: 2px 4px; word-break: break-all; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin: 16px 0 24px;
    }}
    .chart-grid.large {{
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
    }}
    .chart-card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff; }}
    .chart-grid.large .chart-card {{ padding: 14px; }}
    .chart-title {{ margin: 0 0 8px; font-size: 14px; font-weight: 600; }}
    .chart-grid.large .chart-title {{ font-size: 15px; }}
    canvas {{ width: 100% !important; height: auto !important; display: block; }}
    ul {{ margin: 6px 0; padding-left: 18px; }}

    @media (max-width: 640px) {{
      body {{ margin: 10px; }}
      .cards {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }}
      .value {{ font-size: 18px; }}
      .chart-grid {{ grid-template-columns: 1fr; gap: 12px; }}
      .chart-grid.large {{ grid-template-columns: 1fr; gap: 14px; }}
      th, td {{ font-size: 12px; padding: 6px 8px; }}
    }}
  </style>
</head>
<body>
  <h1>Monomarket Backtest Report</h1>
  <div class='meta'>
    <div><strong>Generated:</strong> {html.escape(datetime.now(UTC).isoformat())}</div>
    <div><strong>Window:</strong> {html.escape(str(payload.get('from_ts', 'n/a')))} → {html.escape(str(payload.get('to_ts', 'n/a')))}</div>
    <div><strong>Run dir:</strong> <code>{html.escape(str(run_dir))}</code></div>
    <div><strong>Report dir:</strong> <code>{html.escape(str(output_dir))}</code></div>
    <div><strong>Schema:</strong> {html.escape(str(payload.get('schema_version', 'n/a')))}</div>
  </div>

  <h2>Summary</h2>
  <div class='cards'>
    {cards}
  </div>

  <h2>Backtest Assumptions</h2>
  <div class='table-wrap'>
    <table>
      <thead>
        <tr>
          <th>Item</th>
          <th>Value</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {assumption_rows}
      </tbody>
    </table>
  </div>

  <h2>待续稿 · Charts</h2>
  <div class='muted'>关键图表已放大（mobile/desktop 都更易读）。</div>
  <div class='chart-grid large'>
    <div class='chart-card'>
      <div class='chart-title'>Cumulative PnL (Total MTM vs Realized)</div>
      <canvas id='chart-cumulative-pnl'></canvas>
    </div>
    <div class='chart-card'>
      <div class='chart-title'>Per-strategy PnL</div>
      <canvas id='chart-strategy-pnl'></canvas>
    </div>
    <div class='chart-card'>
      <div class='chart-title'>Per-strategy Trade Count</div>
      <canvas id='chart-strategy-trades'></canvas>
    </div>
    <div class='chart-card'>
      <div class='chart-title'>Per-strategy Winrate (%)</div>
      <canvas id='chart-strategy-winrate'></canvas>
    </div>
    <div class='chart-card'>
      <div class='chart-title'>History: Recent Runs Total PnL Trend</div>
      <canvas id='chart-history-total-pnl'></canvas>
    </div>
  </div>

  <h2>Per-strategy</h2>
  <div class='table-wrap'>
    <table>
      <thead>
        <tr>
          <th>strategy</th>
          <th>pnl</th>
          <th>trade_count</th>
          <th>winrate</th>
          <th>winrate_source</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

  <h2>Strategy Catalog</h2>
  <div class='table-wrap'>
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>策略核心逻辑</th>
          <th>信号方向/触发条件</th>
          <th>主要约束（关键阈值配置项名）</th>
          <th>已知风险或常见拒因</th>
          <th>来源</th>
        </tr>
      </thead>
      <tbody>
        {''.join(catalog_rows)}
      </tbody>
    </table>
  </div>

  <h2>Raw Artifacts</h2>
  <ul>
    {link_items}
  </ul>

  <script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>
  <script>
    const REPORT_DATA = {chart_data_json};
    const CHART_ASPECT = window.innerWidth <= 640 ? 1.45 : 1.75;

    function _mkChart(id, cfg) {{
      const el = document.getElementById(id);
      if (!el) return;
      const hasData = cfg && cfg.data && cfg.data.datasets && cfg.data.datasets.some(
        (d) => Array.isArray(d.data) && d.data.length > 0
      );
      if (!hasData) {{
        const parent = el.parentElement;
        if (parent) {{
          const p = document.createElement('div');
          p.className = 'muted';
          p.textContent = 'n/a';
          parent.appendChild(p);
        }}
        el.style.display = 'none';
        return;
      }}
      new Chart(el.getContext('2d'), cfg);
    }}

    _mkChart('chart-cumulative-pnl', {{
      type: 'line',
      data: {{
        labels: REPORT_DATA.cumulative_pnl.labels,
        datasets: [
          {{
            label: 'Total pnl (mtm)',
            data: REPORT_DATA.cumulative_pnl.total_values,
            borderColor: '#16a34a',
            backgroundColor: 'rgba(22,163,74,0.10)',
            tension: 0.25,
            fill: true,
            pointRadius: 0,
          }},
          {{
            label: 'Realized pnl',
            data: REPORT_DATA.cumulative_pnl.realized_values,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37,99,235,0.08)',
            tension: 0.2,
            fill: false,
            pointRadius: 0,
            borderDash: [6, 4],
          }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: CHART_ASPECT,
        interaction: {{ mode: 'index', intersect: false }},
        scales: {{
          x: {{
            ticks: {{ autoSkip: true, maxTicksLimit: 8, maxRotation: 0, minRotation: 0 }},
            grid: {{ display: false }},
          }},
        }},
        plugins: {{
          legend: {{ display: true }},
          tooltip: {{
            callbacks: {{
              title: (items) => {{
                if (!items || items.length === 0) return '';
                const idx = items[0].dataIndex;
                return REPORT_DATA.cumulative_pnl.full_labels[idx] || REPORT_DATA.cumulative_pnl.labels[idx] || '';
              }},
            }},
          }},
        }},
      }},
    }});

    _mkChart('chart-strategy-pnl', {{
      type: 'bar',
      data: {{
        labels: REPORT_DATA.strategy_pnl.labels,
        datasets: [{{
          label: 'PnL',
          data: REPORT_DATA.strategy_pnl.values,
          backgroundColor: REPORT_DATA.strategy_pnl.values.map((v) => v >= 0 ? '#16a34a' : '#dc2626'),
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: CHART_ASPECT,
        scales: {{ x: {{ ticks: {{ maxRotation: 0, minRotation: 0 }} }} }},
      }},
    }});

    _mkChart('chart-strategy-trades', {{
      type: 'bar',
      data: {{
        labels: REPORT_DATA.strategy_trade_count.labels,
        datasets: [{{
          label: 'Trade count',
          data: REPORT_DATA.strategy_trade_count.values,
          backgroundColor: '#7c3aed',
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: CHART_ASPECT,
        scales: {{ x: {{ ticks: {{ maxRotation: 0, minRotation: 0 }} }} }},
      }},
    }});

    _mkChart('chart-strategy-winrate', {{
      type: 'bar',
      data: {{
        labels: REPORT_DATA.strategy_winrate.labels,
        datasets: [{{
          label: 'Winrate %',
          data: REPORT_DATA.strategy_winrate.values,
          backgroundColor: '#0ea5e9',
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: CHART_ASPECT,
        scales: {{
          x: {{ ticks: {{ maxRotation: 0, minRotation: 0 }} }},
          y: {{ min: 0, max: 100 }},
        }},
        plugins: {{
          tooltip: {{
            callbacks: {{
              afterLabel: (ctx) => 'source=' + (REPORT_DATA.strategy_winrate.sources[ctx.dataIndex] || 'n/a'),
            }},
          }},
        }},
      }},
    }});

    _mkChart('chart-history-total-pnl', {{
      type: 'line',
      data: {{
        labels: REPORT_DATA.history_total_pnl.labels,
        datasets: [{{
          label: 'Recent runs total pnl',
          data: REPORT_DATA.history_total_pnl.values,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.12)',
          tension: 0.2,
          fill: true,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: CHART_ASPECT,
        scales: {{ x: {{ ticks: {{ autoSkip: true, maxTicksLimit: 8, maxRotation: 0, minRotation: 0 }} }} }},
      }},
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    for required in RAW_REQUIRED_FILENAMES:
        if not (run_dir / required).exists():
            raise SystemExit(f"missing required artifact: {run_dir / required}")

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (Path("artifacts/backtest/web") / _iso_utc_now()).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(run_dir / "latest.json")
    cycle_meta_path = run_dir / "cycle-meta.json"
    cycle_meta = _load_json(cycle_meta_path) if cycle_meta_path.exists() else {}

    links = _copy_raw_artifacts(run_dir, out_dir)
    html_text = _render_html(
        payload=payload,
        cycle_meta=cycle_meta,
        run_dir=run_dir,
        output_dir=out_dir,
        links=links,
        history_limit=max(0, int(args.history_limit)),
    )

    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

    with (out_dir / "report-meta.json").open("w", encoding="utf-8", newline="") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "run_dir": str(run_dir),
                "report_dir": str(out_dir),
                "raw_links": links,
                "has_cycle_meta": bool(cycle_meta),
                "total_pnl": sum(
                    float(r.get("pnl", 0.0) or 0.0) for r in payload.get("results", [])
                ),
                "max_drawdown": max(
                    (float(r.get("max_drawdown", 0.0) or 0.0) for r in payload.get("results", [])),
                    default=0.0,
                ),
                "executed_signals": int(payload.get("executed_signals", 0) or 0),
                "rejected_signals": int(payload.get("rejected_signals", 0) or 0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (run_dir / "strategy.csv").open("r", encoding="utf-8", newline="") as f:
        _ = list(csv.DictReader(f))

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
