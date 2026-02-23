from __future__ import annotations

from pathlib import Path

import requests

from monomarket.data.clients import FetchBatch, RequestStats, SourceClient
from monomarket.data.ingestion import IngestionService
from monomarket.db.storage import Storage
from monomarket.models import MarketView


def _market(market_id: str) -> MarketView:
    return MarketView(
        market_id=market_id,
        canonical_id=f"c-{market_id}",
        source="gamma",
        event_id=f"e-{market_id}",
        question=f"Q-{market_id}",
        status="open",
        neg_risk=False,
        liquidity=100.0,
        volume=10.0,
        yes_price=0.4,
        no_price=0.6,
        mid_price=0.4,
    )


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_source_client_retry_and_backoff(monkeypatch) -> None:
    client = SourceClient(
        base_url="https://example.com",
        timeout_sec=2,
        max_retries=2,
        backoff_base_sec=0.1,
        rate_limit_per_sec=1000,
    )

    calls = {"n": 0}

    def _fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.RequestException("boom")
        return _FakeResponse({"ok": True})

    sleeps: list[float] = []

    monkeypatch.setattr(requests, "get", _fake_get)
    monkeypatch.setattr("monomarket.data.clients.time.sleep", lambda sec: sleeps.append(sec))

    payload, stats, err = client.get_json("/markets", {"limit": 1})

    assert payload == {"ok": True}
    assert err == ""
    assert stats.requests == 2
    assert stats.failures == 1
    assert stats.retries == 1
    assert sleeps
    assert abs(sleeps[0] - 0.1) < 1e-6


def test_source_client_rate_limit_throttle(monkeypatch) -> None:
    client = SourceClient(
        base_url="https://example.com",
        timeout_sec=2,
        max_retries=0,
        backoff_base_sec=0.0,
        rate_limit_per_sec=2.0,
    )
    client._last_request_monotonic = 10.0
    stats = RequestStats()

    sleeps: list[float] = []
    monkeypatch.setattr("monomarket.data.clients.time.monotonic", lambda: 10.2)
    monkeypatch.setattr("monomarket.data.clients.time.sleep", lambda sec: sleeps.append(sec))

    client._throttle(stats)

    assert len(sleeps) == 1
    assert abs(sleeps[0] - 0.3) < 1e-6
    assert abs(stats.throttled_sec - 0.3) < 1e-6


class _FakeClients:
    def __init__(self):
        self.calls: list[str | None] = []

    def fetch_gamma(self, limit: int, since=None, incremental: bool = False) -> FetchBatch:
        del limit, incremental
        self.calls.append(None if since is None else since.isoformat())
        if since is None:
            return FetchBatch(
                rows=[_market("m1"), _market("m2")],
                stats=RequestStats(requests=1, failures=0, retries=0),
                max_timestamp="2026-02-20T00:00:00+00:00",
            )
        return FetchBatch(
            rows=[_market("m3")],
            stats=RequestStats(requests=1, failures=0, retries=0),
            max_timestamp="2026-02-21T00:00:00+00:00",
        )


def test_ingestion_incremental_checkpoint(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    fake_clients = _FakeClients()
    svc = IngestionService(fake_clients, storage)

    r1 = svc.ingest("gamma", limit=2, incremental=True)
    r2 = svc.ingest("gamma", limit=2, incremental=True)

    assert r1.status == "ok"
    assert r1.rows == 2
    assert r2.status == "ok"
    assert r2.rows == 1

    assert fake_clients.calls[0] is None
    assert fake_clients.calls[1] == "2026-02-20T00:00:00+00:00"
    assert storage.get_ingestion_checkpoint("gamma") == "2026-02-21T00:00:00+00:00"

    with storage.conn() as conn:
        row = conn.execute(
            "SELECT total_requests, total_failures FROM ingestion_state WHERE source='gamma'"
        ).fetchone()
    assert row is not None
    assert int(row["total_requests"]) == 2
    assert int(row["total_failures"]) == 0
