from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from monomarket.cli import app


class _FakeSignalEngine:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.last_generation_stats: dict[str, Any] = {}

    def generate(self, _selected: list[str], market_limit: int = 2000) -> list[Any]:
        assert market_limit == 2000
        self.last_generation_stats = {
            "edge_gate": {
                "total_raw": 2,
                "total_pass": 1,
                "total_fail": 1,
                "pass_rate": 0.5,
                "by_strategy": {
                    "s9": {
                        "raw": 2,
                        "pass": 1,
                        "fail": 1,
                        "pass_rate": 0.5,
                        "avg_estimated_edge_bps_pass": 42.0,
                        "strategy_diagnostics": {
                            "candidate_reject_reasons": {
                                "buy:effective_edge_below_min": 1,
                            },
                            "candidate_reject_reasons_by_event_top_k": 20,
                            "candidate_reject_reasons_by_event_top": {
                                "e1": {"buy:effective_edge_below_min": 1},
                            },
                            "pricing_consistency": {
                                "price_floor": 0.01,
                                "pair_candidates_priced": 2,
                                "pairs_with_floor_adjustment": 1,
                                "tiny_price_pairs": 1,
                                "avg_pre_floor_gross_edge_bps": 35.0,
                                "avg_post_floor_gross_edge_bps": 20.0,
                                "avg_post_slippage_gross_edge_bps": 10.0,
                                "avg_post_slippage_effective_edge_bps": 5.0,
                                "filtered_post_floor_non_positive": 1,
                                "filtered_post_slippage_non_positive": 0,
                                "filtered_post_slippage_effective_edge_below_min": 1,
                            },
                        },
                    }
                },
            }
        }
        return []


def test_generate_signals_prints_pricing_consistency_diagnostics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    db = tmp_path / "mono.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                f"  db_path: {db}",
                "trading:",
                "  mode: paper",
            ]
        )
    )

    monkeypatch.setattr("monomarket.cli.SignalEngine", _FakeSignalEngine)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "generate-signals",
            "--strategies",
            "s9",
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "generated 0 signals" in res.output
    assert "Strategy pricing consistency diagnostics" in res.output
    assert "buy:effective_edge_below_min" in res.output
