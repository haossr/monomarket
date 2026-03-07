#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monomarket.config import load_settings


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _topic_category(question: str) -> str:
    q = (question or "").lower()
    sports_kw = (
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "mvp",
        "rookie",
        "playoff",
        "conference",
        "fifa",
        "world cup",
        "champions league",
        "super bowl",
        "team",
        "win the",
    )
    politics_kw = (
        "election",
        "president",
        "presidential",
        "senate",
        "house",
        "governor",
        "mayor",
        "democratic",
        "republican",
        "primary",
        "nomination",
        "white house",
        "congress",
    )
    crypto_kw = (
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "crypto",
        "token",
        "fdv",
        "market cap",
        "memecoin",
        "airdrop",
        "defi",
        "altcoin",
    )

    if any(k in q for k in sports_kw):
        return "sports"
    if any(k in q for k in politics_kw):
        return "politics"
    if any(k in q for k in crypto_kw):
        return "crypto"
    return "other"


def _load_question_map(db_path: Path, event_ids: list[str]) -> dict[str, str]:
    if not event_ids:
        return {}

    conn = sqlite3.connect(str(db_path))
    try:
        qmarks = ",".join(["?"] * len(event_ids))
        query = (
            f"SELECT event_id, question, liquidity FROM markets WHERE event_id IN ({qmarks})"
        )
        rows = conn.execute(query, event_ids).fetchall()
    finally:
        conn.close()

    best: dict[str, tuple[str, float]] = {}
    for raw_event_id, question, liquidity in rows:
        event_id = str(raw_event_id)
        liq = _safe_float(liquidity)
        prev = best.get(event_id)
        if prev is None or liq > prev[1]:
            best[event_id] = (str(question or ""), liq)

    return {k: v[0] for k, v in best.items()}


def _compute_equity_series(replay: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    strategies = sorted(replay["strategy"].dropna().astype(str).unique().tolist())
    state = {s: 0.0 for s in strategies}

    portfolio_values: list[float] = []
    benchmark_values: list[float] = []

    def benchmark_from_state() -> float:
        if "s8" in state:
            return float(state["s8"])
        if not state:
            return 0.0
        return float(np.mean(list(state.values())))

    for row in replay.itertuples(index=False):
        strategy = str(getattr(row, "strategy"))
        strategy_equity = _safe_float(getattr(row, "strategy_equity"))
        state[strategy] = strategy_equity
        portfolio_values.append(float(sum(state.values())))
        benchmark_values.append(benchmark_from_state())

    portfolio_eq = pd.Series(portfolio_values, index=replay["ts"])
    benchmark_eq = pd.Series(benchmark_values, index=replay["ts"])
    benchmark_name = "s8 strategy equity curve" if "s8" in strategies else "equal-weight strategy basket"
    return portfolio_eq, benchmark_eq, benchmark_name


def _sharpe(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    vals = series.astype(float).to_numpy()
    std = float(np.std(vals, ddof=0))
    if std == 0:
        return float("nan")
    return float(np.mean(vals) / std)


def _beta_alpha(y_vals: np.ndarray, x_vals: np.ndarray) -> tuple[float, float]:
    n = min(len(y_vals), len(x_vals))
    if n < 2:
        return float("nan"), float("nan")

    y = y_vals[:n]
    x = x_vals[:n]
    var_x = float(np.var(x, ddof=0))
    if var_x == 0:
        return float("nan"), float(np.mean(y))

    cov = float(np.cov(y, x, ddof=0)[0, 1])
    beta = cov / var_x
    alpha = float(np.mean(y) - beta * np.mean(x))
    return float(beta), float(alpha)


def _build_topic_frames(
    *,
    replay: pd.DataFrame,
    event_csv: pd.DataFrame,
    db_path: Path,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_agg = (
        event_csv.groupby("event_id", as_index=False)
        .agg(pnl=("pnl", "sum"), trade_count=("trade_count", "sum"))
        .copy()
    )
    event_agg["event_id"] = event_agg["event_id"].astype(str)

    exec_counts = (
        replay[replay["risk_allowed_bool"]]
        .groupby("event_id", as_index=False)
        .size()
        .rename(columns={"size": "executed_signals"})
    )
    exec_counts["event_id"] = exec_counts["event_id"].astype(str)

    event_ids = event_agg["event_id"].tolist()
    qmap = _load_question_map(db_path, event_ids)

    topics = event_agg.merge(exec_counts, on="event_id", how="left")
    topics["executed_signals"] = topics["executed_signals"].fillna(0).astype(int)
    topics["topic"] = topics["event_id"].map(qmap).fillna("N/A")
    topics["category"] = topics["topic"].map(_topic_category)
    topics = topics.sort_values(["executed_signals", "pnl"], ascending=[False, False]).reset_index(
        drop=True
    )

    top_topics = topics[["event_id", "topic", "category", "executed_signals", "trade_count", "pnl"]].head(
        top_k
    )

    distribution = (
        topics.groupby("category", as_index=False)
        .agg(
            executed_signals=("executed_signals", "sum"),
            topic_count=("event_id", "nunique"),
            pnl=("pnl", "sum"),
        )
        .sort_values("executed_signals", ascending=False)
        .reset_index(drop=True)
    )

    return top_topics, distribution


def _plot_performance_dashboard(
    *,
    out_path: Path,
    replay: pd.DataFrame,
    portfolio_eq: pd.Series,
    benchmark_eq: pd.Series,
    benchmark_name: str,
    rolling_sharpe: pd.Series,
    rolling_beta: pd.Series,
    rolling_alpha: pd.Series,
    sharpe_step: float,
    beta_step: float,
    alpha_step: float,
    max_drawdown_abs: float,
    run_label: str,
) -> None:
    drawdown = portfolio_eq - portfolio_eq.cummax()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].plot(replay["ts"], portfolio_eq.values, label="Portfolio MTM equity", lw=2)
    axes[0, 0].plot(replay["ts"], benchmark_eq.values, label=f"Benchmark ({benchmark_name})", lw=1.5)
    axes[0, 0].axhline(0, color="gray", lw=1)
    axes[0, 0].set_title("PnL / Equity Curve")
    axes[0, 0].set_ylabel("PnL")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].fill_between(replay["ts"], drawdown.values, 0, alpha=0.35, color="tab:red")
    axes[0, 1].plot(replay["ts"], drawdown.values, color="tab:red", lw=1.2)
    axes[0, 1].axhline(0, color="gray", lw=1)
    axes[0, 1].set_title(f"Drawdown (max={max_drawdown_abs:.3f})")
    axes[0, 1].set_ylabel("PnL drawdown")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(replay["ts"], rolling_sharpe.values, label="Rolling Sharpe (30 steps)")
    if not math.isnan(sharpe_step):
        axes[1, 0].axhline(sharpe_step, ls="--", lw=1, label=f"Full Sharpe={sharpe_step:.3f}")
    axes[1, 0].axhline(0, color="gray", lw=1)
    axes[1, 0].set_title("Sharpe ratio (step-PnL)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(replay["ts"], rolling_beta.values, color="tab:blue", label="Rolling beta")
    if not math.isnan(beta_step):
        ax.axhline(beta_step, ls="--", color="tab:blue", alpha=0.7, label=f"Beta={beta_step:.3f}")
    ax2 = ax.twinx()
    ax2.plot(replay["ts"], rolling_alpha.values, color="tab:orange", label="Rolling alpha")
    if not math.isnan(alpha_step):
        ax2.axhline(alpha_step, ls="--", color="tab:orange", alpha=0.7, label=f"Alpha={alpha_step:.4f}")
    ax.set_title(f"Beta / Alpha vs {benchmark_name}")
    ax.set_ylabel("Beta")
    ax2.set_ylabel("Alpha")
    ax.grid(alpha=0.25)
    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="best")

    fig.suptitle(f"Monomarket backtest diagnostics ({run_label})", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_topic_distribution(*, out_path: Path, distribution: pd.DataFrame) -> None:
    plot_df = distribution.copy()
    if plot_df.empty:
        plot_df = pd.DataFrame(
            {
                "category": ["none"],
                "executed_signals": [0],
                "topic_count": [0],
                "pnl": [0.0],
            }
        )

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bars1 = ax1.bar(
        x - width / 2,
        plot_df["executed_signals"].astype(float).to_numpy(),
        width,
        label="Executed signals",
        color="#1f77b4",
    )
    bars2 = ax1.bar(
        x + width / 2,
        plot_df["topic_count"].astype(float).to_numpy(),
        width,
        label="Unique topics",
        color="#17becf",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["category"].astype(str).tolist())
    ax1.set_ylabel("Count")
    ax1.set_title("Topic distribution (executed signals + unique topics)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["pnl"].astype(float).to_numpy(), color="#d62728", marker="o", label="PnL")
    ax2.set_ylabel("PnL")

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=8)

    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_top_topics(*, out_path: Path, top_topics: pd.DataFrame) -> None:
    plot_df = top_topics.head(10).copy()
    if plot_df.empty:
        plot_df = pd.DataFrame(
            {
                "event_id": ["n/a"],
                "topic": ["N/A"],
                "executed_signals": [0],
                "pnl": [0.0],
            }
        )

    plot_df["label"] = (
        plot_df["event_id"].astype(str)
        + " | "
        + plot_df["topic"].astype(str).str.slice(0, 52)
    )
    plot_df = plot_df.iloc[::-1]

    colors = np.where(plot_df["pnl"].astype(float).to_numpy() >= 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    bars = ax.barh(plot_df["label"], plot_df["executed_signals"].astype(float).to_numpy(), color=colors, alpha=0.78)

    ax.set_xlabel("Executed signals")
    ax.set_title("Top bet topics by executed signals (color = pnl sign)")
    ax.grid(axis="x", alpha=0.25)

    for bar, pnl in zip(bars, plot_df["pnl"].astype(float).to_numpy(), strict=False):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + 0.4, y, f"pnl={pnl:.2f}", va="center", fontsize=9)

    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monomarket topic/perf analysis charts")
    parser.add_argument("--backtest-json", required=True)
    parser.add_argument("--replay-csv", required=True)
    parser.add_argument("--event-csv", required=True)
    parser.add_argument("--config", required=True, help="Monomarket config path (for db_path)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backtest_json = Path(args.backtest_json)
    replay_csv = Path(args.replay_csv)
    event_csv = Path(args.event_csv)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(str(config_path))
    db_path = Path(settings.app.db_path)
    if not db_path.is_absolute():
        db_path = (Path.cwd() / db_path).resolve()

    replay = pd.read_csv(replay_csv)
    replay["ts"] = pd.to_datetime(replay["ts"], utc=True, format="mixed")
    replay = replay.sort_values("ts").reset_index(drop=True)
    replay["risk_allowed_bool"] = replay["risk_allowed"].astype(str).str.lower().isin(["true", "1", "yes"])

    events = pd.read_csv(event_csv)

    portfolio_eq, benchmark_eq, benchmark_name = _compute_equity_series(replay)
    port_step = portfolio_eq.diff().fillna(0.0)
    bench_step = benchmark_eq.diff().fillna(0.0)

    drawdown = portfolio_eq - portfolio_eq.cummax()
    max_drawdown_abs = float(drawdown.min()) if len(drawdown) else float("nan")
    max_drawdown_pct = float((drawdown / portfolio_eq.cummax().replace(0, np.nan)).min()) if len(drawdown) else float("nan")

    rolling_sharpe = port_step.rolling(30).apply(
        lambda x: np.nan if np.std(x, ddof=0) == 0 else np.mean(x) / np.std(x, ddof=0),
        raw=True,
    )

    roll_beta: list[float] = []
    roll_alpha: list[float] = []
    for idx in range(len(port_step)):
        if idx + 1 < 40:
            roll_beta.append(float("nan"))
            roll_alpha.append(float("nan"))
            continue
        y = port_step.to_numpy()[idx + 1 - 40 : idx + 1]
        x = bench_step.to_numpy()[idx + 1 - 40 : idx + 1]
        b, a = _beta_alpha(y, x)
        roll_beta.append(b)
        roll_alpha.append(a)

    rolling_beta = pd.Series(roll_beta, index=port_step.index)
    rolling_alpha = pd.Series(roll_alpha, index=port_step.index)

    daily_df = pd.DataFrame(
        {
            "p": portfolio_eq.groupby(portfolio_eq.index.floor("D")).last().diff(),
            "b": benchmark_eq.groupby(benchmark_eq.index.floor("D")).last().diff(),
        }
    ).dropna()

    beta_step, alpha_step = _beta_alpha(port_step.to_numpy(), bench_step.to_numpy())
    beta_daily, alpha_daily = _beta_alpha(daily_df["p"].to_numpy(), daily_df["b"].to_numpy())
    sharpe_step = _sharpe(port_step)
    sharpe_daily_ann = float(_sharpe(daily_df["p"]) * np.sqrt(365)) if len(daily_df) >= 2 else float("nan")

    top_topics, distribution = _build_topic_frames(
        replay=replay,
        event_csv=events,
        db_path=db_path,
        top_k=max(1, args.top_k),
    )

    perf_png = output_dir / "perf_dashboard.png"
    dist_png = output_dir / "topic_distribution.png"
    topics_png = output_dir / "bet_topics_top10.png"
    top_csv = output_dir / "bet_topics_top12.csv"
    summary_json = output_dir / "topic-analysis.json"

    run_label = backtest_json.parent.name
    _plot_performance_dashboard(
        out_path=perf_png,
        replay=replay,
        portfolio_eq=portfolio_eq,
        benchmark_eq=benchmark_eq,
        benchmark_name=benchmark_name,
        rolling_sharpe=rolling_sharpe,
        rolling_beta=rolling_beta,
        rolling_alpha=rolling_alpha,
        sharpe_step=sharpe_step,
        beta_step=beta_step,
        alpha_step=alpha_step,
        max_drawdown_abs=max_drawdown_abs,
        run_label=run_label,
    )
    _plot_topic_distribution(out_path=dist_png, distribution=distribution)
    _plot_top_topics(out_path=topics_png, top_topics=top_topics)
    top_topics.to_csv(top_csv, index=False)

    latest_payload = json.loads(backtest_json.read_text())
    official_total_pnl = float(
        sum(_safe_float(row.get("pnl")) for row in (latest_payload.get("results") or []))
    )

    summary: dict[str, Any] = {
        "schema_version": "topic-analysis-1.0",
        "run_dir": str(backtest_json.parent.resolve()),
        "benchmark_definition": benchmark_name,
        "official_backtest_total_pnl": official_total_pnl,
        "mtm_curve_final_pnl": float(portfolio_eq.iloc[-1]) if len(portfolio_eq) else 0.0,
        "max_drawdown_abs": max_drawdown_abs,
        "max_drawdown_pct": max_drawdown_pct,
        "beta_step": beta_step,
        "alpha_step": alpha_step,
        "beta_daily": beta_daily,
        "alpha_daily": alpha_daily,
        "sharpe_step": sharpe_step,
        "sharpe_daily_annualized": sharpe_daily_ann,
        "topic_distribution": distribution.to_dict(orient="records"),
        "hot_topics": top_topics.to_dict(orient="records"),
        "files": {
            "perf_dashboard_png": str(perf_png.resolve()),
            "topic_distribution_png": str(dist_png.resolve()),
            "top_topics_png": str(topics_png.resolve()),
            "top_topics_csv": str(top_csv.resolve()),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"[topic-analysis] wrote: {summary_json}")
    print(f"[topic-analysis] wrote: {perf_png}")
    print(f"[topic-analysis] wrote: {dist_png}")
    print(f"[topic-analysis] wrote: {topics_png}")
    print(f"[topic-analysis] wrote: {top_csv}")


if __name__ == "__main__":
    main()
