from __future__ import annotations

from datetime import UTC, datetime

from monomarket.db.storage import Storage
from monomarket.models import PnlLine, PnlReport


class PnlTracker:
    def __init__(self, storage: Storage):
        self.storage = storage

    def report(self) -> PnlReport:
        prices = self.storage.get_market_prices()
        positions = self.storage.get_positions()

        lines: list[PnlLine] = []
        realized_total = 0.0
        unrealized_total = 0.0
        by_strategy: dict[str, float] = {}

        for p in positions:
            strategy = str(p["strategy"])
            market_id = str(p["market_id"])
            token = str(p["token_id"]).upper()
            net_qty = float(p["net_qty"])
            avg_price = float(p["avg_price"])
            realized = float(p["realized_pnl"])
            mark = float(prices.get((market_id, token), avg_price))
            unrealized = net_qty * (mark - avg_price)

            lines.append(
                PnlLine(
                    strategy=strategy,
                    market_id=market_id,
                    token_id=token,
                    net_qty=net_qty,
                    avg_price=avg_price,
                    mark_price=mark,
                    unrealized_pnl=unrealized,
                    realized_pnl=realized,
                )
            )

            realized_total += realized
            unrealized_total += unrealized
            by_strategy[strategy] = by_strategy.get(strategy, 0.0) + realized + unrealized

        return PnlReport(
            generated_at=datetime.now(UTC),
            realized_total=realized_total,
            unrealized_total=unrealized_total,
            by_strategy=by_strategy,
            lines=lines,
        )
