from __future__ import annotations

from datetime import UTC, datetime

from monomarket.db.storage import Storage
from monomarket.models import MetricsReport
from monomarket.pnl.tracker import PnlTracker


class MetricsReporter:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.tracker = PnlTracker(storage)

    def report(self) -> MetricsReport:
        order_stats = self.storage.order_stats()
        pnl = self.tracker.report()
        equity_curve = self._equity_curve_points()

        return MetricsReport(
            generated_at=datetime.now(UTC),
            total_orders=int(order_stats["total"]),
            filled_orders=int(order_stats["filled"]),
            rejected_orders=int(order_stats["rejected"]),
            fill_rate=float(order_stats["fill_rate"]),
            rejection_rate=float(order_stats["rejection_rate"]),
            realized_pnl=pnl.realized_total,
            unrealized_pnl=pnl.unrealized_total,
            max_drawdown=self._max_drawdown(equity_curve),
        )

    def _equity_curve_points(self) -> list[float]:
        positions = self.storage.get_positions()
        # MVP approximation: per-position realized snapshot as pseudo-curve.
        values = [float(p["realized_pnl"]) for p in positions]
        if not values:
            return [0.0]
        out: list[float] = []
        running = 0.0
        for v in values:
            running += v
            out.append(running)
        return out

    @staticmethod
    def _max_drawdown(points: list[float]) -> float:
        peak = points[0] if points else 0.0
        max_dd = 0.0
        for x in points:
            if x > peak:
                peak = x
            dd = peak - x
            if dd > max_dd:
                max_dd = dd
        return max_dd
