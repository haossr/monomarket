from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from monomarket.db.storage import Storage

OUTCOME_TOKEN_YES = "YES"  # nosec B105
OUTCOME_TOKEN_NO = "NO"  # nosec B105
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

        execution_batches: list[list[dict[str, Any]]] = []
        pending_s9_batches: dict[str, dict[str, Any]] = {}
        pending_s10_batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_batch_key = self._s9_pair_batch_key(row)
            if pair_batch_key is not None:
                pending = pending_s9_batches.pop(pair_batch_key, None)
                if pending is None:
                    pending_s9_batches[pair_batch_key] = row
                else:
                    execution_batches.append([pending, row])
                continue

            basket_batch_key = self._s10_basket_batch_key(row)
            if basket_batch_key is not None:
                pending_s10_batches[basket_batch_key].append(row)
                continue

            execution_batches.append([row])

        execution_batches.extend([[row] for row in pending_s9_batches.values()])
        execution_batches.extend(pending_s10_batches.values())
        execution_batches.sort(key=lambda batch: (str(batch[0]["created_at"]), int(batch[0]["id"])))

        def _prepare_row(row: dict[str, Any]) -> dict[str, Any]:
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

            return {
                "row": row,
                "strategy": strategy,
                "market_id": market_id,
                "event_id": event_id,
                "event_key": event_key,
                "side": side,
                "created_at": created_at,
                "token_id": token_id,
                "target_price": target_price,
                "requested_qty": requested_qty,
                "fill_price": fill_price,
                "applied_slippage_bps": applied_slippage_bps,
                "executed_qty": executed_qty,
                "fill_ratio": fill_ratio,
                "fill_probability": fill_probability,
                "risk_decision": risk_decision,
            }

        def _append_rejection(
            prep: dict[str, Any],
            reason: str,
            *,
            count_for_breaker: bool,
        ) -> None:
            nonlocal rejection_count
            strategy = str(prep["strategy"])
            event_key = tuple(prep["event_key"])
            created_at = str(prep["created_at"])
            event_id = str(prep["event_id"])
            risk_decision = prep["risk_decision"]

            rejection_count += 1
            if count_for_breaker:
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
                    market_id=str(prep["market_id"]),
                    token_id=str(prep["token_id"]),
                    side=str(prep["side"]),
                    qty=float(prep["requested_qty"]),
                    executed_qty=0.0,
                    fill_ratio=0.0,
                    fill_probability=0.0,
                    slippage_bps_applied=float(prep["applied_slippage_bps"]),
                    target_price=float(prep["target_price"]),
                    fill_price=float(prep["fill_price"]),
                    realized_change=0.0,
                    strategy_equity=strategy_equity,
                    event_equity=event_equity,
                    risk_allowed=False,
                    risk_reason=reason,
                    risk_notional=float(risk_decision.notional),
                    risk_realized_pnl_before=float(risk_decision.realized_pnl_before),
                    risk_strategy_notional_before=float(risk_decision.strategy_notional_before),
                    risk_event_notional_before=float(risk_decision.event_notional_before),
                    risk_rejections_before=int(risk_decision.rejections_before),
                    risk_max_daily_loss=self.risk.max_daily_loss,
                    risk_max_strategy_notional=self.risk.max_strategy_notional,
                    risk_max_event_notional=self.risk.max_event_notional,
                    risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                )
            )

        def _execute_prepared(prep: dict[str, Any]) -> None:
            nonlocal total_realized, executed_signals

            strategy = str(prep["strategy"])
            market_id = str(prep["market_id"])
            event_id = str(prep["event_id"])
            side = str(prep["side"])
            created_at = str(prep["created_at"])
            event_key = tuple(prep["event_key"])
            risk_decision = prep["risk_decision"]

            fill_price = float(prep["fill_price"])
            executed_qty = float(prep["executed_qty"])
            requested_qty = float(prep["requested_qty"])
            target_price = float(prep["target_price"])
            applied_slippage_bps = float(prep["applied_slippage_bps"])
            fill_ratio = float(prep["fill_ratio"])
            fill_probability = float(prep["fill_probability"])

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

            key = (strategy, market_id, str(prep["token_id"]))
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
                    token_id=str(prep["token_id"]),
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
                    risk_notional=float(risk_decision.notional),
                    risk_realized_pnl_before=float(risk_decision.realized_pnl_before),
                    risk_strategy_notional_before=float(risk_decision.strategy_notional_before),
                    risk_event_notional_before=float(risk_decision.event_notional_before),
                    risk_rejections_before=int(risk_decision.rejections_before),
                    risk_max_daily_loss=self.risk.max_daily_loss,
                    risk_max_strategy_notional=self.risk.max_strategy_notional,
                    risk_max_event_notional=self.risk.max_event_notional,
                    risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                )
            )

        def _execute_s10_conversion_basket(batch_preps: list[dict[str, Any]]) -> tuple[bool, str | None]:
            nonlocal total_realized, executed_signals

            if not batch_preps:
                return False, "s10 basket atomic guard: empty basket"

            strategy = str(batch_preps[0]["strategy"])
            if strategy != "s10":
                return False, "s10 basket atomic guard: inconsistent strategy legs"

            side = str(batch_preps[0]["side"]).lower()
            if side not in {"buy", "sell"}:
                return False, "s10 basket atomic guard: invalid basket side"

            executed_qtys = [float(prep["executed_qty"]) for prep in batch_preps]
            min_executed_qty = min(executed_qtys)
            max_executed_qty = max(executed_qtys)
            if min_executed_qty <= 1e-12:
                return False, "s10 basket atomic guard: basket leg has no executable liquidity"

            qty_tolerance = max(1e-9, min_executed_qty * 1e-6)
            if max_executed_qty - min_executed_qty > qty_tolerance:
                return False, "s10 basket atomic guard: basket legs must share executable qty"

            fill_qty = min_executed_qty

            first_payload = batch_preps[0]["row"].get("payload")
            convert_value = 1.0
            if isinstance(first_payload, dict):
                raw_convert_value = first_payload.get("convert_value")
                if raw_convert_value is not None:
                    try:
                        convert_value = float(raw_convert_value)
                    except (TypeError, ValueError):
                        return False, "s10 basket atomic guard: invalid convert_value"
            if convert_value <= 0:
                return False, "s10 basket atomic guard: invalid convert_value"

            leg_notionals = [float(prep["fill_price"]) * fill_qty for prep in batch_preps]
            total_leg_notional = sum(leg_notionals)
            total_fee = (
                total_leg_notional * max(0.0, self.execution.fee_bps) / 10000.0
            )
            payout_notional = fill_qty * convert_value

            if side == "buy":
                basket_realized = payout_notional - total_leg_notional - total_fee
            else:
                basket_realized = total_leg_notional - total_fee - payout_notional

            if total_leg_notional > _METRIC_EPS:
                allocated_realized = [
                    basket_realized * (leg_notional / total_leg_notional)
                    for leg_notional in leg_notionals
                ]
                allocated_realized[-1] = basket_realized - sum(allocated_realized[:-1])
            else:
                allocated_realized = [basket_realized / float(len(batch_preps))] * len(batch_preps)

            for idx, prep in enumerate(batch_preps):
                created_at = str(prep["created_at"])
                event_id = str(prep["event_id"])
                market_id = str(prep["market_id"])
                event_key = tuple(prep["event_key"])
                risk_decision = prep["risk_decision"]

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

                realized_change = float(allocated_realized[idx])

                realized_by_strategy[strategy] += realized_change
                total_realized += realized_change
                trade_count[strategy] += 1
                executed_signals += 1
                rejection_streak_by_strategy[strategy] = 0

                realized_by_event[event_key] += realized_change
                event_trade_count[event_key] += 1

                trade_notional = abs(float(prep["fill_price"]) * fill_qty)
                trade_return = (
                    (realized_change / trade_notional)
                    if trade_notional > _METRIC_EPS
                    else 0.0
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

                requested_qty = float(prep["requested_qty"])
                fill_ratio = (
                    max(0.0, min(1.0, fill_qty / requested_qty))
                    if requested_qty > 1e-12
                    else 0.0
                )

                replay.append(
                    BacktestReplayRow(
                        ts=created_at,
                        strategy=strategy,
                        event_id=event_id,
                        market_id=market_id,
                        token_id=str(prep["token_id"]),
                        side=side,
                        qty=requested_qty,
                        executed_qty=fill_qty,
                        fill_ratio=fill_ratio,
                        fill_probability=float(prep["fill_probability"]),
                        slippage_bps_applied=float(prep["applied_slippage_bps"]),
                        target_price=float(prep["target_price"]),
                        fill_price=float(prep["fill_price"]),
                        realized_change=realized_change,
                        strategy_equity=strategy_equity,
                        event_equity=event_equity,
                        risk_allowed=True,
                        risk_reason="ok",
                        risk_notional=float(risk_decision.notional),
                        risk_realized_pnl_before=float(risk_decision.realized_pnl_before),
                        risk_strategy_notional_before=float(
                            risk_decision.strategy_notional_before
                        ),
                        risk_event_notional_before=float(risk_decision.event_notional_before),
                        risk_rejections_before=int(risk_decision.rejections_before),
                        risk_max_daily_loss=self.risk.max_daily_loss,
                        risk_max_strategy_notional=self.risk.max_strategy_notional,
                        risk_max_event_notional=self.risk.max_event_notional,
                        risk_circuit_breaker_rejections=self.risk.circuit_breaker_rejections,
                    )
                )

            return True, None

        for batch in execution_batches:
            if len(batch) == 1:
                row = batch[0]
                prep = _prepare_row(row)
                pair_batch_key = self._s9_pair_batch_key(row)
                if pair_batch_key is not None:
                    _append_rejection(
                        prep,
                        "s9 pair atomic guard: missing paired leg",
                        count_for_breaker=False,
                    )
                    continue

                basket_batch_key = self._s10_basket_batch_key(row)
                if basket_batch_key is not None:
                    _append_rejection(
                        prep,
                        "s10 basket atomic guard: missing basket legs",
                        count_for_breaker=False,
                    )
                    continue

                risk_decision = prep["risk_decision"]
                if not risk_decision.ok:
                    _append_rejection(
                        prep,
                        str(risk_decision.reason),
                        count_for_breaker=self._breaker_counts_rejection_reason(
                            str(risk_decision.reason)
                        ),
                    )
                    continue

                if float(prep["executed_qty"]) <= 1e-12:
                    _append_rejection(prep, "no executable liquidity", count_for_breaker=False)
                    continue

                _execute_prepared(prep)
                continue

            batch_key = self._s9_pair_batch_key(batch[0])
            if (
                len(batch) == 2
                and batch_key is not None
                and batch_key == self._s9_pair_batch_key(batch[1])
            ):
                ordered_rows = sorted(batch, key=lambda r: int(r["id"]))
                consistent, reject_reason = self._s9_pair_rows_consistent(ordered_rows)
                pair_preps = [_prepare_row(row) for row in ordered_rows]

                if not consistent:
                    for prep in pair_preps:
                        _append_rejection(prep, reject_reason, count_for_breaker=False)
                    continue

                first_risk_failure = next(
                    (prep for prep in pair_preps if not prep["risk_decision"].ok),
                    None,
                )
                if first_risk_failure is not None:
                    reason = str(first_risk_failure["risk_decision"].reason)
                    count_for_breaker = self._breaker_counts_rejection_reason(reason)
                    for prep in pair_preps:
                        _append_rejection(prep, reason, count_for_breaker=count_for_breaker)
                    continue

                if any(float(prep["executed_qty"]) <= 1e-12 for prep in pair_preps):
                    for prep in pair_preps:
                        _append_rejection(
                            prep,
                            "s9 pair atomic guard: paired leg has no executable liquidity",
                            count_for_breaker=False,
                        )
                    continue

                total_pair_notional = sum(
                    float(prep["risk_decision"].notional) for prep in pair_preps
                )
                strategy_notional_before = float(
                    pair_preps[0]["risk_decision"].strategy_notional_before
                )
                if strategy_notional_before + total_pair_notional > self.risk.max_strategy_notional:
                    reason = (
                        "strategy notional limit exceeded (pair-atomic): "
                        f"{strategy_notional_before + total_pair_notional:.2f} "
                        f"> {self.risk.max_strategy_notional:.2f}"
                    )
                    for prep in pair_preps:
                        _append_rejection(prep, reason, count_for_breaker=False)
                    continue

                if pair_preps[0]["event_id"] == pair_preps[1]["event_id"]:
                    event_notional_before = float(
                        pair_preps[0]["risk_decision"].event_notional_before
                    )
                    if event_notional_before + total_pair_notional > self.risk.max_event_notional:
                        reason = (
                            "event notional limit exceeded (pair-atomic): "
                            f"{event_notional_before + total_pair_notional:.2f} "
                            f"> {self.risk.max_event_notional:.2f}"
                        )
                        for prep in pair_preps:
                            _append_rejection(prep, reason, count_for_breaker=False)
                        continue

                for prep in pair_preps:
                    _execute_prepared(prep)
                continue

            s10_batch_key = self._s10_basket_batch_key(batch[0])
            if s10_batch_key is not None and all(
                self._s10_basket_batch_key(row) == s10_batch_key for row in batch
            ):
                ordered_rows = sorted(batch, key=lambda r: int(r["id"]))
                consistent, reject_reason = self._s10_basket_rows_consistent(ordered_rows)
                basket_preps = [_prepare_row(row) for row in ordered_rows]

                if not consistent:
                    for prep in basket_preps:
                        _append_rejection(prep, reject_reason, count_for_breaker=False)
                    continue

                first_risk_failure = next(
                    (prep for prep in basket_preps if not prep["risk_decision"].ok),
                    None,
                )
                if first_risk_failure is not None:
                    reason = str(first_risk_failure["risk_decision"].reason)
                    count_for_breaker = self._breaker_counts_rejection_reason(reason)
                    for prep in basket_preps:
                        _append_rejection(prep, reason, count_for_breaker=count_for_breaker)
                    continue

                if any(float(prep["executed_qty"]) <= 1e-12 for prep in basket_preps):
                    for prep in basket_preps:
                        _append_rejection(
                            prep,
                            "s10 basket atomic guard: basket leg has no executable liquidity",
                            count_for_breaker=False,
                        )
                    continue

                total_basket_notional = sum(
                    float(prep["risk_decision"].notional) for prep in basket_preps
                )
                strategy_notional_before = float(
                    basket_preps[0]["risk_decision"].strategy_notional_before
                )
                if (
                    strategy_notional_before + total_basket_notional
                    > self.risk.max_strategy_notional
                ):
                    reason = (
                        "strategy notional limit exceeded (basket-atomic): "
                        f"{strategy_notional_before + total_basket_notional:.2f} "
                        f"> {self.risk.max_strategy_notional:.2f}"
                    )
                    for prep in basket_preps:
                        _append_rejection(prep, reason, count_for_breaker=False)
                    continue

                event_ids = {str(prep["event_id"]) for prep in basket_preps}
                if len(event_ids) == 1:
                    event_notional_before = float(
                        basket_preps[0]["risk_decision"].event_notional_before
                    )
                    if event_notional_before + total_basket_notional > self.risk.max_event_notional:
                        reason = (
                            "event notional limit exceeded (basket-atomic): "
                            f"{event_notional_before + total_basket_notional:.2f} "
                            f"> {self.risk.max_event_notional:.2f}"
                        )
                        for prep in basket_preps:
                            _append_rejection(prep, reason, count_for_breaker=False)
                        continue

                executed, conversion_reject_reason = _execute_s10_conversion_basket(
                    basket_preps
                )
                if not executed:
                    reason = conversion_reject_reason or "s10 basket atomic guard: conversion failed"
                    for prep in basket_preps:
                        _append_rejection(prep, reason, count_for_breaker=False)
                    continue

                continue

            for row in batch:
                prep = _prepare_row(row)
                risk_decision = prep["risk_decision"]
                if not risk_decision.ok:
                    _append_rejection(
                        prep,
                        str(risk_decision.reason),
                        count_for_breaker=self._breaker_counts_rejection_reason(
                            str(risk_decision.reason)
                        ),
                    )
                elif float(prep["executed_qty"]) <= 1e-12:
                    _append_rejection(prep, "no executable liquidity", count_for_breaker=False)
                else:
                    _execute_prepared(prep)

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

    @staticmethod
    def _s9_pair_batch_key(row: dict[str, Any]) -> str | None:
        if str(row.get("strategy", "")).lower() != "s9":
            return None

        payload = row.get("payload")
        if not isinstance(payload, dict):
            return None

        pair_id = str(payload.get("pair_batch_id") or payload.get("pair_id") or "").strip()
        if not pair_id:
            return None

        is_atomic = bool(payload.get("pair_atomic", False))
        if not is_atomic:
            # Legacy S9 payloads (before pair_atomic fields) still carry pair_id + partner info.
            partner_market_id = str(payload.get("partner_market_id") or "").strip()
            partner_token = str(payload.get("partner_token") or "").strip().upper()
            if not partner_market_id or partner_token not in {OUTCOME_TOKEN_YES, OUTCOME_TOKEN_NO}:
                return None

        created_at = str(row.get("created_at", "")).strip()
        side = str(row.get("side", "")).strip().lower()
        if not created_at or not side:
            return None

        return f"{created_at}|{side}|{pair_id}"

    @classmethod
    def _s9_pair_rows_consistent(cls, pair_rows: list[dict[str, Any]]) -> tuple[bool, str]:
        if len(pair_rows) != 2:
            return False, "s9 pair atomic guard: expected exactly 2 legs"

        first, second = pair_rows
        if (
            str(first.get("strategy", "")).lower() != "s9"
            or str(second.get("strategy", "")).lower() != "s9"
        ):
            return False, "s9 pair atomic guard: inconsistent strategy legs"

        side_a = str(first.get("side", "")).lower()
        side_b = str(second.get("side", "")).lower()
        if not side_a or side_a != side_b:
            return False, "s9 pair atomic guard: inconsistent pair side"

        payload_a = first.get("payload") if isinstance(first.get("payload"), dict) else {}
        payload_b = second.get("payload") if isinstance(second.get("payload"), dict) else {}
        if not isinstance(payload_a, dict) or not isinstance(payload_b, dict):
            return False, "s9 pair atomic guard: invalid pair payload"

        pair_id_a = str(payload_a.get("pair_id") or "").strip()
        pair_id_b = str(payload_b.get("pair_id") or "").strip()
        if pair_id_a and pair_id_b and pair_id_a != pair_id_b:
            return False, "s9 pair atomic guard: mismatched pair_id"

        event_a = str(first.get("event_id", "")).strip()
        event_b = str(second.get("event_id", "")).strip()
        if event_a and event_b and event_a != event_b:
            return False, "s9 pair atomic guard: mismatched event_id"

        cond_a = str(payload_a.get("condition_key") or "").strip().lower()
        cond_b = str(payload_b.get("condition_key") or "").strip().lower()
        if cond_a and cond_b and cond_a != cond_b:
            return False, "s9 pair atomic guard: mismatched condition_key"

        expected_legs = int(float(payload_a.get("pair_expected_legs", 2) or 2))
        if expected_legs != 2:
            return False, "s9 pair atomic guard: unexpected pair_expected_legs"

        token_a, _, _ = cls._signal_execution_fields(first)
        token_b, _, _ = cls._signal_execution_fields(second)
        if {token_a, token_b} != {OUTCOME_TOKEN_YES, OUTCOME_TOKEN_NO}:
            return False, "s9 pair atomic guard: legs must include YES and NO"

        market_a = str(first.get("market_id", "")).strip()
        market_b = str(second.get("market_id", "")).strip()
        if market_a and market_b and market_a != market_b:
            return False, "s9 pair atomic guard: pair legs must share market_id"

        return True, "ok"

    @staticmethod
    def _s10_basket_batch_key(row: dict[str, Any]) -> str | None:
        if str(row.get("strategy", "")).lower() != "s10":
            return None

        payload = row.get("payload")
        if not isinstance(payload, dict):
            return None
        if not bool(payload.get("basket_atomic", False)):
            return None

        basket_id = str(payload.get("basket_batch_id") or payload.get("basket_id") or "").strip()
        if not basket_id:
            return None

        created_at = str(row.get("created_at", "")).strip()
        side = str(row.get("side", "")).strip().lower()
        if not created_at or not side:
            return None

        return f"{created_at}|{side}|{basket_id}"

    @staticmethod
    def _s10_basket_rows_consistent(batch_rows: list[dict[str, Any]]) -> tuple[bool, str]:
        if len(batch_rows) < 2:
            return False, "s10 basket atomic guard: expected at least 2 basket legs"

        first = batch_rows[0]
        first_side = str(first.get("side", "")).lower()
        first_event = str(first.get("event_id", "")).strip()
        first_payload = first.get("payload") if isinstance(first.get("payload"), dict) else {}
        if not isinstance(first_payload, dict):
            return False, "s10 basket atomic guard: invalid basket payload"

        first_batch_id = str(
            first_payload.get("basket_batch_id") or first_payload.get("basket_id") or ""
        ).strip()
        if not first_batch_id:
            return False, "s10 basket atomic guard: missing basket identifier"

        expected_legs = int(float(first_payload.get("basket_expected_legs", len(batch_rows)) or 0))
        if expected_legs <= 1:
            return False, "s10 basket atomic guard: invalid basket_expected_legs"
        if len(batch_rows) != expected_legs:
            return (
                False,
                f"s10 basket atomic guard: incomplete basket legs ({len(batch_rows)}/{expected_legs})",
            )

        seen_leg_indexes: set[int] = set()
        for row in batch_rows:
            if str(row.get("strategy", "")).lower() != "s10":
                return False, "s10 basket atomic guard: inconsistent strategy legs"

            side = str(row.get("side", "")).lower()
            if side != first_side:
                return False, "s10 basket atomic guard: inconsistent basket side"

            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if not isinstance(payload, dict):
                return False, "s10 basket atomic guard: invalid basket payload"

            batch_id = str(payload.get("basket_batch_id") or payload.get("basket_id") or "").strip()
            if batch_id != first_batch_id:
                return False, "s10 basket atomic guard: mismatched basket_id"

            event_id = str(row.get("event_id", "")).strip()
            if event_id != first_event:
                return False, "s10 basket atomic guard: mismatched event_id"

            try:
                leg_index = int(payload.get("leg_index", -1))
            except (TypeError, ValueError):
                return False, "s10 basket atomic guard: invalid leg_index"
            if leg_index < 0 or leg_index >= expected_legs:
                return False, "s10 basket atomic guard: leg_index out of expected range"
            if leg_index in seen_leg_indexes:
                return False, "s10 basket atomic guard: duplicated leg_index"
            seen_leg_indexes.add(leg_index)

            token, _, _ = BacktestEngine._signal_execution_fields(row)
            if token != OUTCOME_TOKEN_YES:
                return False, "s10 basket atomic guard: legs must target YES token"

        return True, "ok"

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
