from __future__ import annotations

from pathlib import Path

from monomarket.db.storage import Storage
from monomarket.execution.live import LiveExecutor, LiveOrderReport
from monomarket.models import OrderRequest


class _FakeLiveClient:
    def __init__(self) -> None:
        self.place_report = LiveOrderReport(
            external_id="ext-1",
            status="accepted",
            filled_qty=0.0,
            avg_fill_price=0.0,
            message="accepted",
        )
        self.cancel_report = LiveOrderReport(
            external_id="ext-1",
            status="canceled",
            filled_qty=0.0,
            avg_fill_price=0.0,
            message="canceled",
        )
        self.order_report = LiveOrderReport(
            external_id="ext-1",
            status="accepted",
            filled_qty=0.0,
            avg_fill_price=0.0,
            message="accepted",
        )

    def place_order(self, req: OrderRequest) -> LiveOrderReport:
        del req
        return self.place_report

    def cancel_order(self, external_id: str) -> LiveOrderReport:
        del external_id
        return self.cancel_report

    def get_order(self, external_id: str) -> LiveOrderReport:
        del external_id
        return self.order_report


def _req() -> OrderRequest:
    return OrderRequest(
        strategy="manual",
        market_id="m1",
        event_id="e1",
        token_id="YES",
        side="buy",
        action="open",
        price=0.42,
        qty=3.0,
        mode="live",
    )


def test_live_execute_rejects_without_credentials(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    monkeypatch.delenv("POLYMARKET_CLOB_HEADERS_JSON", raising=False)
    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_API_SECRET", raising=False)
    monkeypatch.delenv("POLYMARKET_API_PASSPHRASE", raising=False)

    executor = LiveExecutor(storage, client=_FakeLiveClient())
    result = executor.execute(_req())

    assert result.accepted is False
    assert result.status == "rejected"
    assert result.order_id is not None
    row = storage.get_order(result.order_id)
    assert row is not None
    assert row["status"] == "rejected"


def test_live_execute_cancel_and_sync_closed_loop(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    monkeypatch.setenv("POLYMARKET_CLOB_HEADERS_JSON", "{}")

    fake = _FakeLiveClient()
    executor = LiveExecutor(storage, client=fake)

    place_result = executor.execute(_req())
    assert place_result.accepted is True
    assert place_result.status == "accepted"
    assert place_result.order_id is not None

    order_id = int(place_result.order_id)
    row = storage.get_order(order_id)
    assert row is not None
    assert row["external_id"] == "ext-1"

    fake.order_report = LiveOrderReport(
        external_id="ext-1",
        status="filled",
        filled_qty=2.0,
        avg_fill_price=0.43,
        message="filled",
    )
    sync_summary = executor.sync_open_orders(limit=10)

    assert sync_summary["orders_scanned"] == 1
    assert sync_summary["orders_updated"] == 1
    assert sync_summary["errors"] == 0
    assert float(sync_summary["filled_delta_qty"]) == 2.0
    assert abs(storage.order_filled_qty(order_id) - 2.0) < 1e-9

    cancel_result = executor.cancel(order_id)
    assert cancel_result.accepted is True
    assert cancel_result.status == "canceled"

    row_after_cancel = storage.get_order(order_id)
    assert row_after_cancel is not None
    assert row_after_cancel["status"] == "canceled"
