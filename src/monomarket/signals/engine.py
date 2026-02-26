from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from monomarket.config import Settings
from monomarket.db.storage import Storage
from monomarket.models import Signal
from monomarket.signals.edge_gate import EdgeGate
from monomarket.signals.strategies import (
    S1CrossVenueScanner,
    S2NegRiskRebalance,
    S4LowProbYesBasket,
    S8NoCarryTailHedge,
    Strategy,
)


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


class SignalEngine:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.registry: dict[str, Strategy] = {
            "s1": S1CrossVenueScanner(),
            "s2": S2NegRiskRebalance(),
            "s4": S4LowProbYesBasket(),
            "s8": S8NoCarryTailHedge(),
        }
        self.last_generation_stats: dict[str, object] = {}

    def generate(
        self, strategies: Iterable[str] | None = None, market_limit: int = 2000
    ) -> list[Signal]:
        started_at = datetime.now(UTC).isoformat()
        selected = [s.lower() for s in strategies] if strategies else list(self.registry.keys())
        markets = self.storage.fetch_markets(limit=market_limit, status="open")
        liquidity_by_market = {m.market_id: float(m.liquidity or 0.0) for m in markets}

        edge_gate = EdgeGate(self.settings.edge_gate)
        out: list[Signal] = []
        per_strategy: dict[str, dict[str, object]] = {}

        for name in selected:
            impl = self.registry.get(name)
            if impl is None:
                continue
            cfg = self.settings.strategies.get(name, {})
            raw_signals = impl.generate(markets, cfg)

            stats = per_strategy.setdefault(
                name,
                {
                    "raw": 0,
                    "pass": 0,
                    "fail": 0,
                    "pass_rate": 0.0,
                    "avg_estimated_edge_bps_pass": 0.0,
                    "avg_estimated_edge_bps_fail": 0.0,
                    "fail_reasons": {},
                },
            )
            pass_edges: list[float] = []
            fail_edges: list[float] = []

            for signal in raw_signals:
                stats["raw"] = _to_int(stats.get("raw")) + 1
                decision = edge_gate.evaluate(
                    signal,
                    market_liquidity=liquidity_by_market.get(signal.market_id),
                )
                if decision.passed:
                    stats["pass"] = _to_int(stats.get("pass")) + 1
                    pass_edges.append(float(decision.estimated_edge_bps))
                    out.append(edge_gate.attach(signal, decision))
                else:
                    stats["fail"] = _to_int(stats.get("fail")) + 1
                    fail_edges.append(float(decision.estimated_edge_bps))
                    fail_reasons = stats.get("fail_reasons")
                    if isinstance(fail_reasons, dict):
                        key = str(decision.reason)
                        fail_reasons[key] = _to_int(fail_reasons.get(key, 0)) + 1

            raw_count = _to_int(stats.get("raw"))
            pass_count = _to_int(stats.get("pass"))
            stats["pass_rate"] = (pass_count / raw_count) if raw_count > 0 else 0.0
            stats["avg_estimated_edge_bps_pass"] = (
                (sum(pass_edges) / len(pass_edges)) if pass_edges else 0.0
            )
            stats["avg_estimated_edge_bps_fail"] = (
                (sum(fail_edges) / len(fail_edges)) if fail_edges else 0.0
            )

        out.sort(key=lambda x: x.score, reverse=True)
        self.storage.insert_signals(out)

        total_raw = sum(_to_int(v.get("raw", 0)) for v in per_strategy.values())
        total_pass = sum(_to_int(v.get("pass", 0)) for v in per_strategy.values())
        total_fail = sum(_to_int(v.get("fail", 0)) for v in per_strategy.values())
        pass_rate = (total_pass / total_raw) if total_raw > 0 else 0.0

        stats_payload: dict[str, object] = {
            "edge_gate": {
                "total_raw": total_raw,
                "total_pass": total_pass,
                "total_fail": total_fail,
                "pass_rate": pass_rate,
                "by_strategy": per_strategy,
            }
        }
        self.last_generation_stats = stats_payload

        self.storage.insert_signal_generation_run(
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            strategies=selected,
            market_limit=market_limit,
            total_raw=total_raw,
            total_pass=total_pass,
            total_fail=total_fail,
            diagnostics=stats_payload,
        )

        return out
