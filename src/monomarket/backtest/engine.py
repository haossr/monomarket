from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from monomarket.db.storage import Storage

OUTCOME_TOKEN_YES = "YES"  # nosec B105


@dataclass(slots=True)
class BacktestExecutionConfig:
    slippage_bps: float = 5.0
    fee_bps: float = 0.0


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
    winrate: float
    max_drawdown: float
    trade_count: int
    wins: int
    losses: int


@dataclass(slots=True)
class BacktestEventResult:
    strategy: str
    event_id: str
    pnl: float
    winrate: float
    max_drawdown: float
    trade_count: int
    wins: int
    losses: int


@dataclass(slots=True)
class BacktestReplayRow:
    ts: str
    strategy: str
    event_id: str
    market_id: str
    token_id: str
    side: str
    qty: float
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
    generated_at: datetime
    from_ts: str
    to_ts: str
    total_signals: int
    executed_signals: int
    rejected_signals: int
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
        equity_curve: dict[str, list[float]] = defaultdict(lambda: [0.0])

        realized_by_event: dict[tuple[str, str], float] = defaultdict(float)
        event_trade_count: dict[tuple[str, str], int] = defaultdict(int)
        event_wins: dict[tuple[str, str], int] = defaultdict(int)
        event_losses: dict[tuple[str, str], int] = defaultdict(int)
        event_equity_curve: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0])

        replay: list[BacktestReplayRow] = []
        rejection_count = 0
        executed_signals = 0

        for row in rows:
            strategy = str(row["strategy"]).lower()
            market_id = str(row["market_id"])
            event_id = str(row["event_id"])
            side = str(row["side"]).lower()
            created_at = str(row["created_at"])

            strategy_market_event[(strategy, market_id)] = event_id
            event_key = (strategy, event_id)

            token_id, target_price, qty = self._signal_execution_fields(row)
            fill_price = self._fill_price(target_price, side)

            risk_decision = self._risk_check(
                qty=qty,
                price=fill_price,
                strategy=strategy,
                event_id=event_id,
                positions=positions,
                strategy_market_event=strategy_market_event,
                realized_pnl=total_realized,
                rejection_count=rejection_count,
            )

            if not risk_decision.ok:
                rejection_count += 1
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
                        qty=qty,
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

            fee = fill_price * qty * max(0.0, self.execution.fee_bps) / 10000.0

            key = (strategy, market_id, token_id)
            pos = positions.get(key)
            if pos is None:
                pos = _Position()
                positions[key] = pos

            outcome = self._apply_fill(pos, side=side, price=fill_price, qty=qty)
            realized_change = outcome.realized_delta - fee

            realized_by_strategy[strategy] += realized_change
            total_realized += realized_change
            trade_count[strategy] += 1
            executed_signals += 1

            realized_by_event[event_key] += realized_change
            event_trade_count[event_key] += 1

            if outcome.closed_trade:
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
                    qty=qty,
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
            winrate = (wins[strategy] / closed_total) if closed_total else 0.0

            results.append(
                BacktestStrategyResult(
                    strategy=strategy,
                    pnl=final_equity,
                    winrate=winrate,
                    max_drawdown=self._max_drawdown(equity_curve[strategy]),
                    trade_count=trade_count[strategy],
                    wins=wins[strategy],
                    losses=losses[strategy],
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
            winrate = (event_wins[event_key] / closed_total) if closed_total else 0.0
            event_results.append(
                BacktestEventResult(
                    strategy=strategy,
                    event_id=event_id,
                    pnl=final_event_equity,
                    winrate=winrate,
                    max_drawdown=self._max_drawdown(event_equity_curve[event_key]),
                    trade_count=event_trade_count[event_key],
                    wins=event_wins[event_key],
                    losses=event_losses[event_key],
                )
            )

        results.sort(key=lambda x: x.strategy)
        replay.sort(key=lambda x: (x.ts, x.strategy, x.market_id, x.token_id))

        return BacktestReport(
            generated_at=datetime.now(UTC),
            from_ts=from_iso,
            to_ts=to_iso,
            total_signals=len(rows),
            executed_signals=executed_signals,
            rejected_signals=len(rows) - executed_signals,
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

    def _fill_price(self, target_price: float, side: str) -> float:
        px = max(0.01, min(0.99, target_price))
        slippage = px * max(0.0, self.execution.slippage_bps) / 10000.0
        if side == "buy":
            return min(0.99, px + slippage)
        return max(0.01, px - slippage)

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

    def _strategy_notional(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
    ) -> float:
        total = 0.0
        for (s, _, _), pos in positions.items():
            if s != strategy:
                continue
            total += abs(pos.net_qty * pos.avg_price)
        return total

    def _event_notional(
        self,
        positions: dict[tuple[str, str, str], _Position],
        strategy: str,
        event_id: str,
        strategy_market_event: dict[tuple[str, str], str],
    ) -> float:
        total = 0.0
        for (s, market_id, _), pos in positions.items():
            if s != strategy:
                continue
            if strategy_market_event.get((strategy, market_id)) != event_id:
                continue
            total += abs(pos.net_qty * pos.avg_price)
        return total

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
        rejection_count: int,
    ) -> _RiskDecision:
        notional = abs(qty * price)
        strategy_notional = self._strategy_notional(positions, strategy)
        event_notional = self._event_notional(positions, strategy, event_id, strategy_market_event)

        if qty <= 0 or price <= 0:
            return _RiskDecision(
                ok=False,
                reason="invalid order qty/price",
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_count,
            )

        if realized_pnl <= -abs(self.risk.max_daily_loss):
            return _RiskDecision(
                ok=False,
                reason=f"global stop-loss triggered: realized={realized_pnl:.2f}",
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_count,
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
                rejections_before=rejection_count,
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
                rejections_before=rejection_count,
            )

        if rejection_count >= self.risk.circuit_breaker_rejections:
            return _RiskDecision(
                ok=False,
                reason=(
                    "circuit breaker open: "
                    f"rejected={rejection_count}, "
                    f"threshold={self.risk.circuit_breaker_rejections}"
                ),
                notional=notional,
                realized_pnl_before=realized_pnl,
                strategy_notional_before=strategy_notional,
                event_notional_before=event_notional,
                rejections_before=rejection_count,
            )

        return _RiskDecision(
            ok=True,
            reason="ok",
            notional=notional,
            realized_pnl_before=realized_pnl,
            strategy_notional_before=strategy_notional,
            event_notional_before=event_notional,
            rejections_before=rejection_count,
        )
