from __future__ import annotations

import os

from monomarket.db.storage import Storage
from monomarket.models import OrderRequest, OrderResult


class LiveExecutor:
    """Stubbed live executor.

    This MVP intentionally avoids fake external fills. It records an accepted
    intent only when credentials exist; otherwise it rejects with a clear message.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    def execute(self, req: OrderRequest) -> OrderResult:
        api_key = os.getenv("POLYMARKET_API_KEY")
        if not api_key:
            order_id = self.storage.insert_order(
                req,
                status="rejected",
                message="live credentials missing: POLYMARKET_API_KEY",
            )
            return OrderResult(
                accepted=False,
                status="rejected",
                order_id=order_id,
                message="live credentials missing: POLYMARKET_API_KEY",
            )

        # Placeholder for real API integration; keep behavior explicit and auditable.
        order_id = self.storage.insert_order(
            req,
            status="accepted",
            message="live order accepted (API integration pending)",
        )
        return OrderResult(
            accepted=True,
            status="accepted",
            order_id=order_id,
            message="live order accepted (API integration pending)",
            external_id=f"live-{order_id}",
        )
