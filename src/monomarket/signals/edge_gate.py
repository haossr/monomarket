from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from monomarket.config import EdgeGateSettings
from monomarket.models import Signal


@dataclass(slots=True)
class EdgeGateDecision:
    passed: bool
    reason: str
    gross_edge_bps: float
    estimated_edge_bps: float
    min_edge_bps: float
    transaction_cost_bps: float
    liquidity_penalty_bps: float
    latency_penalty_bps: float


class EdgeGate:
    def __init__(self, settings: EdgeGateSettings):
        self.settings = settings

    @staticmethod
    def _f(raw: object, default: float = 0.0) -> float:
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(raw: object, default: bool = False) -> bool:
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return default
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _strategy_override(self, strategy: str) -> Mapping[str, Any]:
        raw = self.settings.per_strategy.get(strategy, {})
        return raw if isinstance(raw, Mapping) else {}

    def _gross_edge_bps(self, signal: Signal) -> float:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        if "edge_hint_bps" in payload:
            return max(0.0, self._f(payload.get("edge_hint_bps"), 0.0))
        target = max(1e-6, self._f(signal.target_price, 0.0))
        return max(
            0.0,
            self._f(signal.score, 0.0) * 100.0 + self._f(signal.confidence, 0.0) * 10.0 / target,
        )

    def evaluate(self, signal: Signal, *, market_liquidity: float | None) -> EdgeGateDecision:
        strategy = str(signal.strategy or "").strip().lower()
        strategy_cfg = self._strategy_override(strategy)

        enabled = self._as_bool(strategy_cfg.get("enabled"), self.settings.enabled)
        min_edge_bps = self._f(strategy_cfg.get("min_edge_bps"), self.settings.min_edge_bps)
        fee_bps = self._f(strategy_cfg.get("fee_bps"), self.settings.fee_bps)
        slippage_bps = self._f(strategy_cfg.get("slippage_bps"), self.settings.slippage_bps)
        latency_penalty_bps = self._f(
            strategy_cfg.get("latency_penalty_bps"), self.settings.latency_penalty_bps
        )
        liq_ref = max(
            1e-9,
            self._f(strategy_cfg.get("liquidity_reference"), self.settings.liquidity_reference),
        )
        liq_penalty_max = max(
            0.0,
            self._f(
                strategy_cfg.get("liquidity_penalty_max_bps"),
                self.settings.liquidity_penalty_max_bps,
            ),
        )

        gross = self._gross_edge_bps(signal)
        transaction_cost = max(0.0, fee_bps) + max(0.0, slippage_bps)
        liq = max(0.0, self._f(market_liquidity, 0.0))
        illiquid_ratio = max(0.0, min(1.0, 1.0 - (liq / liq_ref)))
        liq_penalty = illiquid_ratio * liq_penalty_max

        estimated = gross - transaction_cost - liq_penalty - max(0.0, latency_penalty_bps)
        if not enabled:
            estimated = gross
        passed = (estimated >= min_edge_bps) if enabled else True

        return EdgeGateDecision(
            passed=passed,
            reason="ok" if passed else "edge_below_threshold",
            gross_edge_bps=gross,
            estimated_edge_bps=estimated,
            min_edge_bps=min_edge_bps,
            transaction_cost_bps=transaction_cost,
            liquidity_penalty_bps=liq_penalty,
            latency_penalty_bps=max(0.0, latency_penalty_bps),
        )

    @staticmethod
    def attach(signal: Signal, decision: EdgeGateDecision) -> Signal:
        payload = dict(signal.payload or {})
        payload["edge_gate"] = {
            "passed": decision.passed,
            "reason": decision.reason,
            "gross_edge_bps": decision.gross_edge_bps,
            "estimated_edge_bps": decision.estimated_edge_bps,
            "min_edge_bps": decision.min_edge_bps,
            "transaction_cost_bps": decision.transaction_cost_bps,
            "liquidity_penalty_bps": decision.liquidity_penalty_bps,
            "latency_penalty_bps": decision.latency_penalty_bps,
        }
        return Signal(
            strategy=signal.strategy,
            market_id=signal.market_id,
            event_id=signal.event_id,
            side=signal.side,
            score=signal.score,
            confidence=signal.confidence,
            target_price=signal.target_price,
            size_hint=signal.size_hint,
            rationale=signal.rationale,
            payload=payload,
        )
