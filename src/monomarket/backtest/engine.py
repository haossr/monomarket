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
class BacktestStrategyResult:
    strategy: str
    pnl: float
    winrate: float
    max_drawdown: float
    trade_count: int
    wins: int
    losses: int


@dataclass(slots=True)
class BacktestReport:
    generated_at: datetime
    from_ts: str
    to_ts: str
    total_signals: int
    results: list[BacktestStrategyResult]


@dataclass(slots=True)
class _Position:
    net_qty: float = 0.0
    avg_price: float = 0.0


@dataclass(slots=True)
class _FillOutcome:
    realized_delta: float
    closed_trade: bool


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
    ):
        self.storage = storage
        self.execution = execution or BacktestExecutionConfig()

    def run(self, strategies: list[str], from_ts: str, to_ts: str) -> BacktestReport:
        from_iso = _parse_ts(from_ts).isoformat()
        to_iso = _parse_ts(to_ts).isoformat()
        selected = [s.strip().lower() for s in strategies if s.strip()]

        rows = self.storage.list_signals_in_window(from_iso, to_iso, strategies=selected or None)

        positions: dict[tuple[str, str, str], _Position] = {}
        realized_by_strategy: dict[str, float] = defaultdict(float)
        trade_count: dict[str, int] = defaultdict(int)
        wins: dict[str, int] = defaultdict(int)
        losses: dict[str, int] = defaultdict(int)
        equity_curve: dict[str, list[float]] = defaultdict(lambda: [0.0])

        for row in rows:
            strategy = str(row["strategy"]).lower()
            market_id = str(row["market_id"])
            side = str(row["side"]).lower()
            created_at = str(row["created_at"])

            token_id, target_price, qty = self._signal_execution_fields(row)
            fill_price = self._fill_price(target_price, side)
            fee = fill_price * qty * max(0.0, self.execution.fee_bps) / 10000.0

            key = (strategy, market_id, token_id)
            pos = positions.get(key)
            if pos is None:
                pos = _Position()
                positions[key] = pos

            outcome = self._apply_fill(pos, side=side, price=fill_price, qty=qty)
            realized_change = outcome.realized_delta - fee
            realized_by_strategy[strategy] += realized_change
            trade_count[strategy] += 1

            if outcome.closed_trade:
                if realized_change > 0:
                    wins[strategy] += 1
                elif realized_change < 0:
                    losses[strategy] += 1

            equity_curve[strategy].append(
                realized_by_strategy[strategy]
                + self._strategy_unrealized(positions, strategy, created_at)
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

        results.sort(key=lambda x: x.strategy)
        return BacktestReport(
            generated_at=datetime.now(UTC),
            from_ts=from_iso,
            to_ts=to_iso,
            total_signals=len(rows),
            results=results,
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
