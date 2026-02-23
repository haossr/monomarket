from __future__ import annotations

from monomarket.db.storage import Storage
from monomarket.models import OrderRequest, OrderResult


class PaperExecutor:
    def __init__(self, storage: Storage):
        self.storage = storage

    def execute(self, req: OrderRequest) -> OrderResult:
        order_id = self.storage.insert_order(req, status="filled", message="paper fill")
        self.storage.record_fill(
            order_id=order_id,
            strategy=req.strategy,
            market_id=req.market_id,
            event_id=req.event_id,
            token_id=req.token_id,
            side=req.side,
            price=req.price,
            qty=req.qty,
            fee=0.0,
        )
        return OrderResult(
            accepted=True,
            status="filled",
            order_id=order_id,
            message="paper order filled",
            external_id=None,
        )
