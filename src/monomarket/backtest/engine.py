from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from monomarket.db.storage import Storage

OUTCOME_TOKEN_YES = "YES"  # nosec B105
BACKTEST_ARTIFACT_SCHEMA_VERSION = "1.0"
_METRIC_EPS = 1e-12


@dataclass(slots=True)
class BacktestExecutionConfig:
    slippage_bps: float = 5.0
    fee_bps: float = 0.0
    enable_partial_fill: bool = False
    liquidity_full_fill: float = 1000.0
    min_fill_ratio: float = 0.1
    enable_fill_probability: bool = False
    min_fill_probability: float = 0.05
    enable_dynamic_slippage: bool = False
    spread_slippage_weight_bps: float = 50.0
    liquidity_slippage_weight_bps: float = 25.0
    liquidity_reference: float = 1000.0


@dataclass(slots=True)
class BacktestRiskConfig:
    max_daily_loss: float = 1e18
    max_strategy_notional: float = 1e18
    max_event_notional: float = 1e18
    circuit_breaker_rejections: int = 10**9


@dataclass(slots=True)
class BacktestStrategyResult:
    strategy: str
    pnl: float
    # legacy alias kept for downstream compatibility; equals closed_winrate.
    winrate: float
    closed_winrate: float
    mtm_winrate: float
    max_drawdown: float
    trade_count: int
    wins: int
    losses: int
    closed_sample_count: int
    mtm_wins: int
    mtm_losses: int
    mtm_sample_count: int
    # Risk-adjusted metrics (computed from executed replay fills that close inventory,
    # i.e. fills where realized_change != 0 under the current toy execution model).
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_trade_return: float
    return_volatility: float
    expectancy: float
    best_trade_return: float
    worst_trade_return: float


@dataclass(slots=True)
class BacktestEventResult:
    strategy: str
    event_id: str
    pnl: float
    # legacy alias kept for downstream compatibility; equals closed_winrate.
    winrate: float
    closed_winrate: float
    mtm_winrate: float
    max_drawdown: float
    trade_count: int
    wins: int
    losses: int
    closed_sample_count: int
    mtm_wins: int
    mtm_losses: int
    mtm_sample_count: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_trade_return: float
    return_volatility: float
    expectancy: float
    best_trade_return: float
    worst_trade_return: float


@dataclass(slots=True)
class BacktestReplayRow:
    ts: str
    strategy: str
    event_id: str
    market_id: str
    token_id: str
    side: str
    qty: float
    executed_qty: float
    fill_ratio: float
    fill_probability: float
    slippage_bps_applied: float
    target_price: float
    fill_price: float
    realized_change: float
    strategy_equity: float
    event_equity: float
    risk_allowed: bool
    risk_reason: str
    risk_notional: float
    risk_realized_pnl_before: float
    risk_strategy_notional_before: float
    risk_event_notional_before: float
    risk_rejections_before: int
    risk_max_daily_loss: float
    risk_max_strategy_notional: float
    risk_max_event_notional: float
    risk_circuit_breaker_rejections: int


@dataclass(slots=True)
class BacktestReport:
    schema_version: str
    generated_at: datetime
    from_ts: str
    to_ts: str
    total_signals: int
    executed_signals: int
    rejected_signals: int
    execution_config: dict[str, Any]
    risk_config: dict[str, Any]
    results: list[BacktestStrategyResult]
    event_results: list[BacktestEventResult]
    replay: list[BacktestReplayRow]


@dataclass(slots=True)
class _Position:
    net_qty: float = 0.0
    avg_price: float = 0.0


@dataclass(slots=True)
class _FillOutcome:
    realized_delta: float
    closed_trade: bool


@dataclass(slots=True)
class _RiskDecision:
    ok: bool
    reason: str
    notional: float
    realized_pnl_before: float
    strategy_notional_before: float
    event_notional_before: float
    rejections_before: int


@dataclass(slots=True)
class _RiskAdjustedMetrics:
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    return_volatility: float = 0.0
    expectancy: float = 0.0
    best_trade_return: float = 0.0
    worst_trade_return: float = 0.0


def _parse_ts(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class BacktestEngine:
    def __init__(
        self,
        storage: Storage,
        execution: BacktestExecutionConfig | None = None,
        risk: BacktestRiskConfig | None = None,
    ):
        self.storage = storage
        self.execution = execution or BacktestExecutionConfig()
        self.risk = risk or BacktestRiskConfig()

    def run(self, strategies: list[str], from_ts: str, to_ts: str) -> BacktestReport:
        from_iso = _parse_ts(from_ts).isoformat()
        to_iso = _parse_ts(to_ts).isoformat()
        selected = [s.strip().lower() for s in strategies if s.strip()]

        rows = self.storage.list_signals_in_window(from_iso, to_iso, strategies=selected or None)

        positions: dict[tuple[str, str, str], _Position] = {}
        strategy_market_event: dict[tuple[str, str], str] = {}

        realized_by_strategy: dict[str, float] = defaultdict(float)
        total_realized = 0.0

        trade_count: dict[str, int] = defaultdict(int)
        wins: dict[str, int] = defaultdict(int)
        losses: dict[str, int] = defaultdict(int)
        mtm_wins: dict[str, int] = defaultdict(int)
        mtm_losses: dict[str, int] = defaultdict(int)
        equity_curve: dict[str, list[float]] = defaultdict(lambda: [0.0])
        # Per-closed-fill return legs for risk-adjusted metrics.
        # Return leg: realized_change / abs(fill_price * executed_qty).
        closed_trade_returns: dict[str, list[float]] = defaultdict(list)
        closed_trade_realized: dict[str, list[float]] = defaultdict(list)

        realized_by_event: dict[tuple[str, str], float] = defaultdict(float)
        event_trade_count: dict[tuple[str, str], int] = defaultdict(int)
        event_wins: dict[tuple[str, str], int] = defaultdict(int)
        event_losses: dict[tuple[str, str], int] = defaultdict(int)
        event_mtm_wins: dict[tuple[str, str], int] = defaultdict(int)
        event_mtm_losses: dict[tuple[str, str], int] = defaultdict(int)
        event_equity_curve: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0])
        event_closed_trade_returns: dict[tuple[str, str], list[float]] = defaultdict(list)
        event_closed_trade_realized: dict[tuple[str, str], list[float]] = defaultdict(list)

        replay: list[BacktestReplayRow] = []
        rejection_count = 0
        rejection_streak_by_strategy: dict[str, int] = defaultdict(int)
        executed_signals = 0

        for row in rows:
            strategy = str(row["strategy"]).lower()
            market_id = str(row["market_id"])
            event_id = str(row["event_id"])
            side = str(row["side"]).lower()
            created_at = str(row["created_at"])

            strategy_market_event[(strategy, market_id)] = event_id
            event_key = (strategy, event_id)

            token_id, target_price, requested_qty = self._signal_execution_fields(row)
            fill_price, applied_slippage_bps = self._fill_price(
                market_id=market_id,
                ts=created_at,
                target_price=target_price,
                side=side,
            )
            executed_qty, fill_ratio, fill_probability = self._effective_fill_qty(
                market_id=market_id,
                ts=created_at,
                side=side,
                requested_qty=requested_qty,
            )

            if executed_qty > 1e-12:
                capped_qty = self._cap_qty_to_strategy_headroom(
                    qty=executed_qty,
                    price=fill_price,
                    strategy=strategy,
                    positions=positions,
                    ts=created_at,
                )
                if capped_qty < executed_qty:
                    executed_qty = capped_qty
                    fill_ratio = (
                        max(0.0, min(1.0, executed_qty / requested_qty))
                        if requested_qty > 1e-12
                        else 0.0
                    )

            # Risk checks should use executable size when available;
            # otherwise fallback to requested size so no-liquidity paths keep clear reasons.
            risk_qty = executed_qty if executed_qty > 1e-12 else requested_qty
            risk_decision = self._risk_check(
                qty=risk_qty,
                price=fill_price,
                strategy=strategy,
                event_id=event_id,
                positions=positions,
                strategy_market_event=strategy_market_event,
                realized_pnl=total_realized,
                rejection_streak=rejection_streak_by_strategy[strategy],
                ts=created_at,
            )

            if not risk_decision.ok:
                rejection_count += 1
                if self._breaker_counts_rejection_reason(risk_decision.reason):
                    rejection_streak_by_strategy[strategy] += 1
                else:
                    rejection_streak_by_strategy[strategy] = 0
                strategy_equity = realized_by_strategy[strategy] + self._strategy_unrealized(
                    positions,
                    strategy,
                    created_at,
                )
                event_equity = realized_by_event[event_key] + self._event_unrealized(
                    positions,
                    strategy,
                    event_id,
                    created_at,
                    strategy_market_event,
                )

                equity_curve[strategy].append(strategy_equity)
                event_equity_curve[event_key].append(event_equity)

                replay.append(
                    BacktestReplayRow(
                        ts=created_at,
                        strategy=strategy,
                        event_id=event_id,
                        market_id=market_id,
                        token_id=token_id,
                        side=side,
                        qty=requested_qty,
                        executed_qty=0.0,
                        fill_ratio=0.0,
                        fill_probability=0.0,
                        slippage_bps_applied=applied_slippage_bps,
                        target_price=target_price,
                        fill_price=fill_price,
                        realized_change=0.0,
                        strategy_equity=strategy_equity,
                        event_equity=event_equity,
                        risk_allowed=False,
                        risk_reason=risk_decision.reason,
                        risk_notional=risk_decision.notional,
                        risk_realized_pnl_before=risk_decision.realized_pnl_before,
                        risk_strategy_notional_before=risk_decision.strategy_notional_before,
                        risk_event_notional_before=risk_decision.event_notional_before,
                        risk_rejections_before=risk_decision.rejections_before,
                        risk_max_daily_loss=self.risk.max_daily_loss,
                        risk_max_strategy_notional=self.risk.max_strategy_notional,
                        risk_max_event_notional=self.risk.max_event_notional,
                        risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                    )
                )
                continue

            if executed_qty <= 1e-12:
                rejection_count += 1
                # Liquidity misses are execution outcomes; keep breaker focused on
                # consecutive risk-check failures.
                rejection_streak_by_strategy[strategy] = 0
                strategy_equity = realized_by_strategy[strategy] + self._strategy_unrealized(
                    positions,
                    strategy,
                    created_at,
                )
                event_equity = realized_by_event[event_key] + self._event_unrealized(
                    positions,
                    strategy,
                    event_id,
                    created_at,
                    strategy_market_event,
                )

                equity_curve[strategy].append(strategy_equity)
                event_equity_curve[event_key].append(event_equity)

                replay.append(
                    BacktestReplayRow(
                        ts=created_at,
                        strategy=strategy,
                        event_id=event_id,
                        market_id=market_id,
                        token_id=token_id,
                        side=side,
                        qty=requested_qty,
                        executed_qty=0.0,
                        fill_ratio=0.0,
                        fill_probability=0.0,
                        slippage_bps_applied=applied_slippage_bps,
                        target_price=target_price,
                        fill_price=fill_price,
                        realized_change=0.0,
                        strategy_equity=strategy_equity,
                        event_equity=event_equity,
                        risk_allowed=False,
                        risk_reason="no executable liquidity",
                        risk_notional=risk_decision.notional,
                        risk_realized_pnl_before=risk_decision.realized_pnl_before,
                        risk_strategy_notional_before=risk_decision.strategy_notional_before,
                        risk_event_notional_before=risk_decision.event_notional_before,
                        risk_rejections_before=risk_decision.rejections_before,
                        risk_max_daily_loss=self.risk.max_daily_loss,
                        risk_max_strategy_notional=self.risk.max_strategy_notional,
                        risk_max_event_notional=self.risk.max_event_notional,
                        risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                    )
                )
                continue

            fee = fill_price * executed_qty * max(0.0, self.execution.fee_bps) / 10000.0

            strategy_equity_before = realized_by_strategy[strategy] + self._strategy_unrealized(
                positions,
                strategy,
                created_at,
            )
            event_equity_before = realized_by_event[event_key] + self._event_unrealized(
                positions,
                strategy,
                event_id,
                created_at,
                strategy_market_event,
            )

            key = (strategy, market_id, token_id)
            pos = positions.get(key)
            if pos is None:
                pos = _Position()
                positions[key] = pos

            outcome = self._apply_fill(pos, side=side, price=fill_price, qty=executed_qty)
            realized_change = outcome.realized_delta - fee

            realized_by_strategy[strategy] += realized_change
            total_realized += realized_change
            trade_count[strategy] += 1
            executed_signals += 1
            rejection_streak_by_strategy[strategy] = 0

            realized_by_event[event_key] += realized_change
            event_trade_count[event_key] += 1

            if outcome.closed_trade:
                trade_notional = abs(fill_price * executed_qty)
                trade_return = (
                    (realized_change / trade_notional) if trade_notional > _METRIC_EPS else 0.0
                )
                closed_trade_returns[strategy].append(trade_return)
                closed_trade_realized[strategy].append(realized_change)
                event_closed_trade_returns[event_key].append(trade_return)
                event_closed_trade_realized[event_key].append(realized_change)

                if realized_change > 0:
                    wins[strategy] += 1
                    event_wins[event_key] += 1
                elif realized_change < 0:
                    losses[strategy] += 1
                    event_losses[event_key] += 1

            strategy_equity = realized_by_strategy[strategy] + self._strategy_unrealized(
                positions,
                strategy,
                created_at,
            )
            event_equity = realized_by_event[event_key] + self._event_unrealized(
                positions,
                strategy,
                event_id,
                created_at,
                strategy_market_event,
            )

            mtm_delta = strategy_equity - strategy_equity_before
            if mtm_delta > 1e-12:
                mtm_wins[strategy] += 1
            elif mtm_delta < -1e-12:
                mtm_losses[strategy] += 1

            event_mtm_delta = event_equity - event_equity_before
            if event_mtm_delta > 1e-12:
                event_mtm_wins[event_key] += 1
            elif event_mtm_delta < -1e-12:
                event_mtm_losses[event_key] += 1
            equity_curve[strategy].append(strategy_equity)
            event_equity_curve[event_key].append(event_equity)

            replay.append(
                BacktestReplayRow(
                    ts=created_at,
                    strategy=strategy,
                    event_id=event_id,
                    market_id=market_id,
                    token_id=token_id,
                    side=side,
                    qty=requested_qty,
                    executed_qty=executed_qty,
                    fill_ratio=fill_ratio,
                    fill_probability=fill_probability,
                    slippage_bps_applied=applied_slippage_bps,
                    target_price=target_price,
                    fill_price=fill_price,
                    realized_change=realized_change,
                    strategy_equity=strategy_equity,
                    event_equity=event_equity,
                    risk_allowed=True,
                    risk_reason="ok",
                    risk_notional=risk_decision.notional,
                    risk_realized_pnl_before=risk_decision.realized_pnl_before,
                    risk_strategy_notional_before=risk_decision.strategy_notional_before,
                    risk_event_notional_before=risk_decision.event_notional_before,
                    risk_rejections_before=risk_decision.rejections_before,
                    risk_max_daily_loss=self.risk.max_daily_loss,
                    risk_max_strategy_notional=self.risk.max_strategy_notional,
                    risk_max_event_notional=self.risk.max_event_notional,
                    risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                )
            )

        # Guarantee selected strategies appear in report, even if no signal in the window.
        strategies_for_report = sorted(
            set(selected or []) | {str(r["strategy"]).lower() for r in rows}
        )

        results: list[BacktestStrategyResult] = []
        for strategy in strategies_for_report:
            final_equity = realized_by_strategy[strategy] + self._strategy_unrealized(
                positions,
                strategy,
                to_iso,
            )
            equity_curve[strategy].append(final_equity)

            closed_total = wins[strategy] + losses[strategy]
            closed_winrate = (wins[strategy] / closed_total) if closed_total else 0.0
            mtm_total = mtm_wins[strategy] + mtm_losses[strategy]
            mtm_winrate = (mtm_wins[strategy] / mtm_total) if mtm_total else 0.0
            strategy_max_drawdown = self._max_drawdown(equity_curve[strategy])
            strategy_metrics = self._risk_adjusted_metrics(
                trade_returns=closed_trade_returns[strategy],
                realized_changes=closed_trade_realized[strategy],
                pnl=final_equity,
                max_drawdown=strategy_max_drawdown,
            )

            results.append(
                BacktestStrategyResult(
                    strategy=strategy,
                    pnl=final_equity,
                    winrate=closed_winrate,
                    closed_winrate=closed_winrate,
                    mtm_winrate=mtm_winrate,
                    max_drawdown=strategy_max_drawdown,
                    trade_count=trade_count[strategy],
                    wins=wins[strategy],
                    losses=losses[strategy],
                    closed_sample_count=closed_total,
                    mtm_wins=mtm_wins[strategy],
                    mtm_losses=mtm_losses[strategy],
                    mtm_sample_count=mtm_total,
                    sharpe_ratio=strategy_metrics.sharpe_ratio,
                    sortino_ratio=strategy_metrics.sortino_ratio,
                    calmar_ratio=strategy_metrics.calmar_ratio,
                    profit_factor=strategy_metrics.profit_factor,
                    avg_trade_return=strategy_metrics.avg_trade_return,
                    return_volatility=strategy_metrics.return_volatility,
                    expectancy=strategy_metrics.expectancy,
                    best_trade_return=strategy_metrics.best_trade_return,
                    worst_trade_return=strategy_metrics.worst_trade_return,
                )
            )

        event_keys_for_report = sorted(
            set(event_trade_count.keys())
            | {(str(r["strategy"]).lower(), str(r["event_id"])) for r in rows}
        )

        event_results: list[BacktestEventResult] = []
        for strategy, event_id in event_keys_for_report:
            event_key = (strategy, event_id)
            final_event_equity = realized_by_event[event_key] + self._event_unrealized(
                positions,
                strategy,
                event_id,
                to_iso,
                strategy_market_event,
            )
            event_equity_curve[event_key].append(final_event_equity)

            closed_total = event_wins[event_key] + event_losses[event_key]
            closed_winrate = (event_wins[event_key] / closed_total) if closed_total else 0.0
            mtm_total = event_mtm_wins[event_key] + event_mtm_losses[event_key]
            mtm_winrate = (event_mtm_wins[event_key] / mtm_total) if mtm_total else 0.0
            event_max_drawdown = self._max_drawdown(event_equity_curve[event_key])
            event_metrics = self._risk_adjusted_metrics(
                trade_returns=event_closed_trade_returns[event_key],
                realized_changes=event_closed_trade_realized[event_key],
                pnl=final_event_equity,
                max_drawdown=event_max_drawdown,
            )
            event_results.append(
                BacktestEventResult(
                    strategy=strategy,
                    event_id=event_id,
                    pnl=final_event_equity,
                    winrate=closed_winrate,
                    closed_winrate=closed_winrate,
                    mtm_winrate=mtm_winrate,
                    max_drawdown=event_max_drawdown,
                    trade_count=event_trade_count[event_key],
                    wins=event_wins[event_key],
                    losses=event_losses[event_key],
                    closed_sample_count=closed_total,
                    mtm_wins=event_mtm_wins[event_key],
                    mtm_losses=event_mtm_losses[event_key],
                    mtm_sample_count=mtm_total,
                    sharpe_ratio=event_metrics.sharpe_ratio,
                    sortino_ratio=event_metrics.sortino_ratio,
                    calmar_ratio=event_metrics.calmar_ratio,
                    profit_factor=event_metrics.profit_factor,
                    avg_trade_return=event_metrics.avg_trade_return,
                    return_volatility=event_metrics.return_volatility,
                    expectancy=event_metrics.expectancy,
                    best_trade_return=event_metrics.best_trade_return,
                    worst_trade_return=event_metrics.worst_trade_return,
                )
            )

        results.sort(key=lambda x: x.strategy)
        replay.sort(key=lambda x: (x.ts, x.strategy, x.market_id, x.token_id))

        return BacktestReport(
            schema_version=BACKTEST_ARTIFACT_SCHEMA_VERSION,
            generated_at=datetime.now(UTC),
            from_ts=from_iso,
            to_ts=to_iso,
            total_signals=len(rows),
            executed_signals=executed_signals,
            rejected_signals=len(rows) - executed_signals,
            execution_config=asdict(self.execution),
            risk_config=asdict(self.risk),
            results=results,
            event_results=event_results,
            replay=replay,
        )

    @staticmethod
    def _signal_execution_fields(row: dict[str, Any]) -> tuple[str, float, float]:
        token = OUTCOME_TOKEN_YES
        target_price = float(row["target_price"])
        qty = float(row["size_hint"])

        payload = row.get("payload")
        if isinstance(payload, dict):
            primary_leg = payload.get("primary_leg")
            if isinstance(primary_leg, dict):
                token = str(primary_leg.get("token", token)).upper()
                target_price = float(primary_leg.get("price", target_price))
                qty = float(primary_leg.get("qty", qty))

        return token, target_price, qty

    def _fill_price(
        self,
        *,
        market_id: str,
        ts: str,
        target_price: float,
        side: str,
    ) -> tuple[float, float]:
        px = max(0.01, min(0.99, target_price))
        total_bps = max(0.0, self.execution.slippage_bps)

        if self.execution.enable_dynamic_slippage:
            total_bps += self._dynamic_slippage_bps(market_id=market_id, ts=ts)

        slippage = px * total_bps / 10000.0
        if side == "buy":
            return min(0.99, px + slippage), total_bps
        return max(0.01, px - slippage), total_bps

    def _dynamic_slippage_bps(self, *, market_id: str, ts: str) -> float:
        yes_px, no_px = self.storage.get_snapshot_yes_no_at(market_id, ts)
        liquidity = self.storage.get_snapshot_liquidity_at(market_id, ts)

        spread_proxy = 0.0
        if yes_px is not None and no_px is not None:
            spread_proxy = abs(1.0 - (yes_px + no_px))

        spread_component = spread_proxy * max(0.0, self.execution.spread_slippage_weight_bps)

        if liquidity is None:
            liquidity_component = 0.0
        else:
            liq_ref = max(1e-9, self.execution.liquidity_reference)
            illiquid_ratio = max(0.0, min(1.0, 1.0 - (float(liquidity) / liq_ref)))
            liquidity_component = illiquid_ratio * max(
                0.0, self.execution.liquidity_slippage_weight_bps
            )

        return spread_component + liquidity_component

    def _effective_fill_qty(
        self,
        *,
        market_id: str,
        ts: str,
        side: str,
        requested_qty: float,
    ) -> tuple[float, float, float]:
        if requested_qty <= 0:
            return 0.0, 0.0, 0.0

        liquidity = self.storage.get_snapshot_liquidity_at(market_id, ts)

        ratio = 1.0
        if self.execution.enable_partial_fill:
            full_fill_liq = max(1e-9, self.execution.liquidity_full_fill)
            if liquidity is not None:
                raw_ratio = max(0.0, min(1.0, float(liquidity) / full_fill_liq))
                min_ratio = max(0.0, min(1.0, self.execution.min_fill_ratio))
                ratio = max(min_ratio, raw_ratio)

        fill_probability = 1.0
        if self.execution.enable_fill_probability:
            fill_probability = self._fill_probability(liquidity, side)

        final_ratio = max(0.0, min(1.0, ratio * fill_probability))
        executed_qty = requested_qty * final_ratio
        return executed_qty, final_ratio, fill_probability

    def _fill_probability(self, liquidity: float | None, side: str) -> float:
        if liquidity is None:
            return 1.0

        liq = float(liquidity)
        if liq >= 1000.0:
            base = 1.0
        elif liq >= 800.0:
            base = 0.9
        elif liq >= 500.0:
            base = 0.75
        elif liq >= 200.0:
            base = 0.55
        else:
            base = 0.35

        side_adj = 0.95 if side.lower() == "buy" else 1.0
        min_prob = max(0.0, min(1.0, self.execution.min_fill_probability))
        return max(min_prob, min(1.0, base * side_adj))

    @staticmethod
    def _apply_fill(pos: _Position, side: str, price: float, qty: float) -> _FillOutcome:
        signed = qty if side == "buy" else -qty
        if pos.net_qty == 0.0 or pos.net_qty * signed > 0:
            new_qty = pos.net_qty + signed
            if abs(new_qty) < 1e-9:
                pos.avg_price = 0.0
            else:
                pos.avg_price = ((pos.avg_price * abs(pos.net_qty)) + (price * abs(signed))) / abs(
                    new_qty
                )
            pos.net_qty = new_qty
            return _FillOutcome(realized_delta=0.0, closed_trade=False)

        close_qty = min(abs(pos.net_qty), abs(signed))
        direction = 1.0 if pos.net_qty > 0 else -1.0
        pnl_delta = close_qty * (price - pos.avg_price) * direction

        remainder = pos.net_qty + signed
        if abs(remainder) < 1e-9:
            pos.net_qty = 0.0
            pos.avg_price = 0.0
        elif pos.net_qty * remainder > 0:
            pos.net_qty = remainder
        else:
            pos.net_qty = remainder
            pos.avg_price = price

        return _FillOutcome(realized_delta=pnl_delta, closed_trade=close_qty > 0)

    def _strategy_unrealized(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
        ts: str,
    ) -> float:
        total = 0.0
        for (s, market_id, token_id), pos in positions.items():
            if s != strategy or abs(pos.net_qty) < 1e-9:
                continue
            mark = self.storage.get_snapshot_price_at(market_id, token_id, ts)
            if mark is None:
                mark = pos.avg_price
            total += pos.net_qty * (mark - pos.avg_price)
        return total

    def _event_unrealized(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
        event_id: str,
        ts: str,
        strategy_market_event: dict[tuple[str, str], str],
    ) -> float:
        total = 0.0
        for (s, market_id, token_id), pos in positions.items():
            if s != strategy or abs(pos.net_qty) < 1e-9:
                continue
            market_event_id = strategy_market_event.get((strategy, market_id))
            if market_event_id != event_id:
                continue
            mark = self.storage.get_snapshot_price_at(market_id, token_id, ts)
            if mark is None:
                mark = pos.avg_price
            total += pos.net_qty * (mark - pos.avg_price)
        return total

    @staticmethod
    def _max_drawdown(points: list[float]) -> float:
        if not points:
            return 0.0

        peak = points[0]
        max_dd = 0.0
        for x in points:
            if x > peak:
                peak = x
            dd = peak - x
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _sample_std(values: list[float]) -> float:
        n = len(values)
        if n <= 1:
            return 0.0

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        if variance <= _METRIC_EPS:
            return 0.0
        return math.sqrt(variance)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if abs(denominator) <= _METRIC_EPS:
            return 0.0
        return numerator / denominator

    @classmethod
    def _risk_adjusted_metrics(
        cls,
        *,
        trade_returns: list[float],
        realized_changes: list[float],
        pnl: float,
        max_drawdown: float,
    ) -> _RiskAdjustedMetrics:
        # Formulas mirror common tear-sheet conventions, adapted to replay trades:
        #   trade_return_i = realized_change_i / abs(fill_price_i * executed_qty_i)
        #   avg_trade_return = mean(trade_return_i)
        #   return_volatility = sample_std(trade_return_i)
        #   sharpe_ratio = sqrt(N) * avg_trade_return / return_volatility (rf=0)
        #   sortino_ratio = sqrt(N) * avg_trade_return / downside_deviation
        #       downside_deviation = sqrt(mean(min(trade_return_i, 0)^2))
        #   calmar_ratio = pnl / max_drawdown
        #   profit_factor = gross_profit / abs(gross_loss)
        #
        # Edge handling (deterministic/safe): return 0 for undefined denominators
        # (e.g., no trades, zero variance, no losses, zero drawdown).
        #
        # TODO(hsheng): annual return/CAGR, annualized volatility, drawdown duration,
        # Omega/skew/kurtosis/tail-ratio/VaR, exposure/turnover require a canonical
        # capital base and regularized time-series returns in replay artifacts.
        if not trade_returns or not realized_changes:
            return _RiskAdjustedMetrics()

        n = len(trade_returns)
        avg_trade_return = sum(trade_returns) / n
        return_volatility = cls._sample_std(trade_returns)
        sharpe_ratio = cls._safe_ratio(avg_trade_return, return_volatility) * math.sqrt(float(n))

        downside_dev = math.sqrt(sum(min(x, 0.0) ** 2 for x in trade_returns) / n)
        sortino_ratio = cls._safe_ratio(avg_trade_return, downside_dev) * math.sqrt(float(n))

        gross_profit = sum(x for x in realized_changes if x > 0.0)
        gross_loss = -sum(x for x in realized_changes if x < 0.0)

        expectancy = sum(realized_changes) / len(realized_changes)

        return _RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=cls._safe_ratio(pnl, max_drawdown),
            profit_factor=cls._safe_ratio(gross_profit, gross_loss),
            avg_trade_return=avg_trade_return,
            return_volatility=return_volatility,
            expectancy=expectancy,
            best_trade_return=max(trade_returns),
            worst_trade_return=min(trade_returns),
        )

    @staticmethod
    def _breaker_counts_rejection_reason(reason: str) -> bool:
        text = reason.strip().lower()
        if text.startswith("strategy notional limit exceeded"):
            return False
        if text.startswith("event notional limit exceeded"):
            return False
        return True

    def _strategy_notional(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
        ts: str | None = None,
    ) -> float:
        total = 0.0
        for (s, market_id, token_id), pos in positions.items():
            if s != strategy:
                continue
            ref_price = pos.avg_price
            if ts:
                mark = self.storage.get_snapshot_price_at(market_id, token_id, ts)
                if mark is not None and mark > 0:
                    ref_price = float(mark)
            total += abs(pos.net_qty * ref_price)
        return total

    def _event_notional(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
        event_id: str,
        strategy_market_event: dict[tuple[str, str], str],
        ts: str | None = None,
    ) -> float:
        total = 0.0
        for (s, market_id, token_id), pos in positions.items():
            if s != strategy:
                continue
            if strategy_market_event.get((strategy, market_id)) != event_id:
                continue
            ref_price = pos.avg_price
            if ts:
                mark = self.storage.get_snapshot_price_at(market_id, token_id, ts)
                if mark is not None and mark > 0:
                    ref_price = float(mark)
            total += abs(pos.net_qty * ref_price)
        return total

    def _cap_qty_to_strategy_headroom(
        self,
        *,
        qty: float,
        price: float,
        strategy: str,
        positions: dict[tuple[str, str, str], _Position],
        ts: str,
    ) -> float:
        if qty <= 1e-12 or price <= 1e-12 or self.risk.max_strategy_notional <= 0:
            return qty

        strategy_notional = self._strategy_notional(positions, strategy, ts)
        headroom = self.risk.max_strategy_notional - strategy_notional
        if headroom <= 1e-12:
            return qty

        max_qty = headroom / price
        if max_qty <= 1e-12 or max_qty >= qty:
            return qty

        return max_qty

    def _risk_check(
        self,
        *,
        qty: float,
        price: float,
        strategy: str,
        event_id: str,
        positions: dict[tuple[str, str, str], _Position],
        strategy_market_event: dict[tuple[str, str], str],
        realized_pnl: float,
        rejection_streak: int,
        ts: str,
    ) -> _RiskDecision:
        notional = abs(qty * price)
        strategy_notional = self._strategy_notional(positions, strategy, ts)
        event_notional = self._event_notional(
            positions, strategy, event_id, strategy_market_event, ts
        )

        if qty <= 0 or price <= 0:
            return _RiskDecision(
                ok=False,
                reason="invalid order qty/price",
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_streak,
            )

        if realized_pnl <= -abs(self.risk.max_daily_loss):
            return _RiskDecision(
                ok=False,
                reason=f"global stop-loss triggered: realized={realized_pnl:.2f}",
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_streak,
            )

        if strategy_notional + notional > self.risk.max_strategy_notional:
            return _RiskDecision(
                ok=False,
                reason=(
                    f"strategy notional limit exceeded: {strategy_notional + notional:.2f} "
                    f"> {self.risk.max_strategy_notional:.2f}"
                ),
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_streak,
            )

        if event_notional + notional > self.risk.max_event_notional:
            return _RiskDecision(
                ok=False,
                reason=(
                    f"event notional limit exceeded: {event_notional + notional:.2f} "
                    f"> {self.risk.max_event_notional:.2f}"
                ),
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_streak,
            )

        if rejection_streak >= self.risk.circuit_breaker_rejections:
            return _RiskDecision(
                ok=False,
                reason=(
                    "circuit breaker open: "
                    f"consecutive_rejected={rejection_streak}, "
                    f"threshold={self.risk.circuit_breaker_rejections}"
                ),
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_streak,
            )

        return _RiskDecision(
            ok=True,
            reason="ok",
            notional=notional,
            realized_pnl_before=realized_pnl,
            strategy_notional_before=strategy_notional,
            event_notional_before=event_notional,
            rejections_before=rejection_streak,
        )
