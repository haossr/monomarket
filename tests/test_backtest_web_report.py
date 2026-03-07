from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "backtest_web_report.py"


def _write_run(run_dir: Path, *, pnl: float, generated_at: str, include_cycle_meta: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "from_ts": "2026-03-06T00:00:00Z",
        "to_ts": "2026-03-07T00:00:00Z",
        "total_signals": 3,
        "executed_signals": 2,
        "rejected_signals": 1,
        "execution_config": {
            "enable_live_trading": False,
            "enable_partial_fill": True,
            "enable_fill_probability": True,
            "enable_dynamic_slippage": True,
            "slippage_bps": 5.0,
            "fee_bps": 0.0,
            "liquidity_full_fill": 1000.0,
            "min_fill_ratio": 0.2,
            "min_fill_probability": 0.1,
            "spread_slippage_weight_bps": 20.0,
            "liquidity_slippage_weight_bps": 40.0,
            "liquidity_reference": 1000.0,
        },
        "risk_config": {
            "max_daily_loss": 250.0,
            "max_strategy_notional": 1000.0,
            "max_event_notional": 1500.0,
            "circuit_breaker_rejections": 5,
        },
        "results": [
            {
                "strategy": "s1",
                "pnl": pnl,
                "trade_count": 2,
                "closed_winrate": 0.5,
                "closed_sample_count": 2,
                "mtm_winrate": 0.5,
                "mtm_sample_count": 2,
                "max_drawdown": 0.3,
            },
            {
                "strategy": "s10",
                "pnl": 1.2,
                "trade_count": 1,
                "closed_winrate": 1.0,
                "closed_sample_count": 1,
                "mtm_winrate": 1.0,
                "mtm_sample_count": 1,
                "max_drawdown": 0.1,
            },
        ],
        "replay": [
            {
                "ts": "2026-03-06T00:00:00Z",
                "strategy": "s1",
                "realized_change": 0.1,
            },
            {
                "ts": "2026-03-06T01:00:00Z",
                "strategy": "s10",
                "realized_change": -0.05,
            },
        ],
    }
    (run_dir / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    (run_dir / "strategy.csv").write_text(
        "strategy,pnl,trade_count\n" f"s1,{pnl},2\n" "s10,1.2,1\n",
        encoding="utf-8",
    )
    (run_dir / "event.csv").write_text(
        "strategy,event_id,pnl\n" f"s1,e1,{pnl}\n",
        encoding="utf-8",
    )
    (run_dir / "replay.csv").write_text(
        "schema_version,ts,strategy,realized_change\n"
        "1.0,2026-03-06T00:00:00Z,s1,0.1\n"
        "1.0,2026-03-06T01:00:00Z,s10,-0.05\n",
        encoding="utf-8",
    )

    if include_cycle_meta:
        cycle_meta = {
            "fixed_window_mode": False,
            "signal_generation": {
                "historical_replay_only": False,
                "clear_signals_window": True,
                "rebuild_signals_window": True,
                "rebuild_step_hours": 12.0,
                "edge_gate": {
                    "by_strategy": {
                        "s9": {
                            "strategy_diagnostics": {
                                "candidate_reject_reasons": {
                                    "canonical_under_two_legs": 3,
                                    "buy:non_positive_gross_edge": 1,
                                }
                            }
                        }
                    }
                },
            },
        }
        (run_dir / "cycle-meta.json").write_text(json.dumps(cycle_meta), encoding="utf-8")


def test_backtest_web_report_contains_assumptions_charts_and_strategy_catalog(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    old_run = runs_root / "20260306T000000Z"
    cur_run = runs_root / "20260307T215857Z"

    _write_run(
        old_run, pnl=0.25, generated_at="2026-03-06T00:10:00+00:00", include_cycle_meta=False
    )
    _write_run(
        cur_run, pnl=-0.42, generated_at="2026-03-07T22:00:00+00:00", include_cycle_meta=True
    )

    out_dir = tmp_path / "web"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--run-dir",
            str(cur_run),
            "--output-dir",
            str(out_dir),
            "--history-limit",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    index_html = (out_dir / "index.html").read_text(encoding="utf-8")

    # Assumptions section and key fields
    assert "Backtest Assumptions" in index_html
    assert "Window (from -&gt; to)" in index_html
    assert "Live trading" in index_html
    assert "disabled" in index_html
    assert "Risk limits" in index_html

    # Chart markers
    assert "chart-cumulative-pnl" in index_html
    assert "chart-strategy-pnl" in index_html
    assert "chart-strategy-trades" in index_html
    assert "chart-strategy-winrate" in index_html
    assert "chart-history-total-pnl" in index_html

    # Strategy catalog markers and required strategy coverage
    assert "Strategy Catalog" in index_html
    for sid in ("s1", "s2", "s4", "s8", "s9", "s10"):
        assert f"<strong>{sid}</strong>" in index_html
    assert "strategies.s1.min_spread" in index_html
    assert "strategies.s10.prob_sum_tolerance" in index_html
    assert "canonical_under_two_legs: 3" in index_html

    # History chart should include previous run label
    assert "20260306T000000Z" in index_html

    assert (out_dir / "report-meta.json").exists()
    assert (out_dir / "raw" / "cycle-meta.json").exists()
