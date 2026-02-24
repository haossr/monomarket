from __future__ import annotations

from datetime import datetime
from pathlib import Path

import requests
from typer.testing import CliRunner

from monomarket.cli import app
from monomarket.config import DataSettings
from monomarket.data.clients import (
    FetchBatch,
    MarketDataClients,
    RequestStats,
    SourceClient,
    normalize_market,
)
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


class _FakeErrorResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def raise_for_status(self) -> None:
        response = requests.Response()
        response.status_code = self.status_code
        raise requests.HTTPError(f"status={self.status_code}", response=response)

    def json(self) -> dict[str, object]:
        return {}


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


def test_source_client_http_4xx_no_retry(monkeypatch) -> None:
    client = SourceClient(
        base_url="https://example.com",
        timeout_sec=2,
        max_retries=3,
        backoff_base_sec=0.1,
        rate_limit_per_sec=1000,
    )

    calls = {"n": 0}

    def _fake_get(*_args, **_kwargs):
        calls["n"] += 1
        return _FakeErrorResponse(400)

    monkeypatch.setattr(requests, "get", _fake_get)
    payload, stats, err = client.get_json("/markets", {"limit": 1})

    assert payload == []
    assert "HTTP 400" in err
    assert stats.requests == 1
    assert stats.retries == 0
    assert stats.failures == 1
    assert stats.error_buckets.get("http_4xx") == 1


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


def test_normalize_market_parses_string_outcome_prices_and_events_event_id() -> None:
    row = normalize_market(
        {
            "id": "m-json",
            "question": "Will X happen?",
            "active": True,
            "closed": False,
            "liquidity": 123,
            "outcomePrices": '["0.029", "0.971"]',
            "outcomes": '["Yes", "No"]',
            "events": [{"id": "evt-123", "slug": "evt-slug"}],
        },
        source="gamma",
    )

    assert row is not None
    assert row.event_id == "evt-123"
    assert row.yes_price == 0.029
    assert row.no_price == 0.971


def test_fetch_gamma_includes_closed_and_archived_filters(monkeypatch) -> None:
    clients = MarketDataClients(DataSettings())
    captured: dict[str, object] = {}

    def _fake_get_json(
        _self: SourceClient,
        path: str,
        params: dict[str, object] | None = None,
    ) -> tuple[object, RequestStats, str]:
        captured["path"] = path
        captured["params"] = dict(params or {})
        return [], RequestStats(requests=1), ""

    monkeypatch.setattr(SourceClient, "get_json", _fake_get_json)

    batch = clients.fetch_gamma(limit=321, incremental=False)

    assert batch.error == ""
    assert batch.rows == []
    assert captured["path"] == "markets"
    assert captured["params"] == {
        "limit": 321,
        "active": "true",
        "closed": "false",
        "archived": "false",
    }


def test_fetch_data_skips_404_market_endpoints(monkeypatch) -> None:
    settings = DataSettings(data_base_url="https://data.example")
    clients = MarketDataClients(settings)
    called_paths: list[str] = []

    def _fake_get_json(
        self: SourceClient,
        path: str,
        params: dict[str, object] | None = None,
    ) -> tuple[object, RequestStats, str]:
        del params
        if self.base_url == "https://data.example":
            called_paths.append(path)
            stats = RequestStats(requests=1, failures=1, retries=0)
            stats.error_buckets["http_4xx"] = 1
            stats.last_error_bucket = "http_4xx"
            return [], stats, "HTTP 404: status=404"

        return [], RequestStats(requests=1, failures=0, retries=0), ""

    monkeypatch.setattr(SourceClient, "get_json", _fake_get_json)

    batch = clients.fetch_data(limit=10, incremental=False)

    assert batch.error == ""
    assert batch.rows == []
    assert called_paths == ["markets", "v1/markets", "v2/markets"]
    assert batch.stats.requests == 3
    assert batch.stats.failures == 0
    assert batch.stats.error_buckets == {"source_skipped_404": 1}


class _FakeClients:
    def __init__(self):
        self.calls: list[str | None] = []

    def fetch_gamma(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
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

    def fetch_data(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        return self.fetch_gamma(limit=limit, since=since, incremental=incremental)

    def fetch_clob(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        return self.fetch_gamma(limit=limit, since=since, incremental=incremental)


class _BreakerFakeClients:
    def __init__(self):
        self.calls = 0

    def fetch_gamma(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        del limit, since, incremental
        self.calls += 1
        stats = RequestStats(requests=1, failures=1, retries=0)
        stats.error_buckets["http_5xx"] = 1
        stats.last_error_bucket = "http_5xx"
        return FetchBatch(rows=[], stats=stats, error="HTTP 503")

    def fetch_data(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        return self.fetch_gamma(limit=limit, since=since, incremental=incremental)

    def fetch_clob(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        return self.fetch_gamma(limit=limit, since=since, incremental=incremental)


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


def test_ingestion_circuit_breaker_and_error_buckets(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    fake_clients = _BreakerFakeClients()
    svc = IngestionService(
        fake_clients,
        storage,
        breaker_failure_threshold=1,
        breaker_cooldown_sec=600,
    )

    r1 = svc.ingest("gamma", limit=1, incremental=True)
    r2 = svc.ingest("gamma", limit=1, incremental=True)

    assert r1.status == "partial"
    assert r1.error_buckets.get("http_5xx") == 1

    assert r2.status == "partial"
    assert r2.error_buckets.get("circuit_open") == 1
    assert fake_clients.calls == 1

    with storage.conn() as conn:
        b1 = conn.execute(
            """
            SELECT total_count FROM ingestion_error_buckets
            WHERE source='gamma' AND error_bucket='http_5xx'
            """
        ).fetchone()
        b2 = conn.execute(
            """
            SELECT total_count FROM ingestion_error_buckets
            WHERE source='gamma' AND error_bucket='circuit_open'
            """
        ).fetchone()
    assert b1 is not None and int(b1["total_count"]) == 1
    assert b2 is not None and int(b2["total_count"]) == 1

    transitions = storage.list_ingestion_breaker_transitions(source="gamma")
    by_state = {str(x["state"]): int(x["transition_count"]) for x in transitions}
    assert by_state.get("open") == 1


def test_ingestion_half_open_single_probe(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    fake_clients = _BreakerFakeClients()
    storage.set_ingestion_breaker(
        source="gamma",
        consecutive_failures=1,
        open_until_ts="2000-01-01T00:00:00+00:00",
        last_error_bucket="http_5xx",
    )

    svc = IngestionService(
        fake_clients,
        storage,
        breaker_failure_threshold=1,
        breaker_cooldown_sec=600,
    )

    r1 = svc.ingest("gamma", limit=1, incremental=True)
    r2 = svc.ingest("gamma", limit=1, incremental=True)

    # cooldown passed: allow exactly one probe request
    assert r1.status == "partial"
    assert r1.error_buckets.get("http_5xx") == 1
    assert fake_clients.calls == 1

    # probe failed -> breaker re-opened, no second request
    assert r2.status == "partial"
    assert r2.error_buckets.get("circuit_open") == 1
    assert fake_clients.calls == 1

    transitions = storage.list_ingestion_breaker_transitions(source="gamma")
    by_state = {str(x["state"]): int(x["transition_count"]) for x in transitions}
    assert by_state.get("half_open") == 1
    assert by_state.get("open") == 1


def test_cli_ingest_health(tmp_path: Path) -> None:
    db = tmp_path / "mono.db"
    storage = Storage(str(db))
    storage.init_db()

    storage.update_ingestion_error_bucket(
        source="gamma",
        error_bucket="http_5xx",
        count=3,
        last_error="HTTP 503",
    )
    storage.update_ingestion_error_bucket(
        source="gamma",
        error_bucket="http_5xx",
        count=1,
        last_error="HTTP 502",
    )
    storage.update_ingestion_error_bucket(
        source="gamma",
        error_bucket="timeout",
        count=1,
        last_error="timeout-1",
    )
    storage.update_ingestion_error_bucket(
        source="gamma",
        error_bucket="timeout",
        count=5,
        last_error="timeout-2",
    )
    storage.set_ingestion_breaker(
        source="gamma",
        consecutive_failures=2,
        open_until_ts="2026-02-24T00:00:00+00:00",
        last_error_bucket="http_5xx",
    )
    storage.record_ingestion(
        source="gamma",
        status="ok",
        started_at="2026-02-22T00:00:00+00:00",
        finished_at="2026-02-22T00:01:00+00:00",
        rows_ingested=10,
        error="",
        request_count=10,
        failure_count=0,
        retry_count=1,
    )
    storage.record_ingestion(
        source="gamma",
        status="partial",
        started_at="2026-02-22T01:00:00+00:00",
        finished_at="2026-02-22T01:01:00+00:00",
        rows_ingested=5,
        error="HTTP 503",
        request_count=10,
        failure_count=3,
        retry_count=2,
        error_buckets={"http_5xx": 3, "timeout": 1},
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                f"  db_path: {db}",
                "trading:",
                "  mode: paper",
            ]
        )
    )

    summary = storage.list_ingestion_run_summary_by_source(source="gamma", run_window=5)
    assert len(summary) == 1
    row = summary[0]
    assert int(row["total_runs"]) == 2
    assert int(row["non_ok_runs"]) == 1
    assert float(row["avg_failures"]) == 1.5
    assert float(row["avg_retries"]) == 1.5
    assert float(row["total_failures"]) == 3.0
    assert float(row["total_requests"]) == 20.0

    trends = storage.list_ingestion_error_bucket_trends(
        source="gamma",
        window=1,
        sort_by_abs_delta=True,
    )
    assert len(trends) == 2

    # timeout: recent=5, prev=1 => |delta|=4 (top mover)
    trend_top = trends[0]
    assert str(trend_top["error_bucket"]) == "timeout"
    assert int(trend_top["recent_count"] or 0) == 5
    assert int(trend_top["prev_count"] or 0) == 1

    # http_5xx: recent=1, prev=3 => |delta|=2
    trend_second = trends[1]
    assert str(trend_second["error_bucket"]) == "http_5xx"
    assert int(trend_second["recent_count"] or 0) == 1
    assert int(trend_second["prev_count"] or 0) == 3

    shares = storage.list_ingestion_error_bucket_share_by_source(source="gamma", run_window=5)
    assert len(shares) == 2
    assert str(shares[0]["error_bucket"]) == "http_5xx"
    assert int(shares[0]["bucket_count"] or 0) == 3
    assert abs(float(shares[0]["bucket_share"] or 0.0) - 0.75) < 1e-9
    assert int(shares[0]["runs_with_error"] or 0) == 1
    assert int(shares[0]["total_runs"] or 0) == 2

    assert str(shares[1]["error_bucket"]) == "timeout"
    assert int(shares[1]["bucket_count"] or 0) == 1
    assert abs(float(shares[1]["bucket_share"] or 0.0) - 0.25) < 1e-9

    shares_top1 = storage.list_ingestion_error_bucket_share_by_source(
        source="gamma",
        run_window=5,
        top_k_per_source=1,
    )
    assert len(shares_top1) == 1
    assert str(shares_top1[0]["error_bucket"]) == "http_5xx"

    shares_min50 = storage.list_ingestion_error_bucket_share_by_source(
        source="gamma",
        run_window=5,
        min_share=0.5,
    )
    assert len(shares_min50) == 1
    assert str(shares_min50[0]["error_bucket"]) == "http_5xx"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ingest-health",
            "--source",
            "gamma",
            "--run-window",
            "5",
            "--error-trend-window",
            "1",
            "--error-share-top-k",
            "1",
            "--error-share-min-share",
            "0.5",
            "--error-sample-limit",
            "3",
            "--config",
            str(config_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Ingestion error buckets" in res.output
    assert "Ingestion error bucket trends" in res.output
    assert "top movers by |delta|" in res.output
    assert "Ingestion breakers" in res.output
    assert "Breaker transitions" in res.output
    assert "Ingestion run summary by source" in res.output
    assert "Ingestion error bucket share by source" in res.output
    assert "top_k_per_source=1" in res.output
    assert "min_share=50.00%" in res.output
    assert "Recent ingestion errors" in res.output
    assert "gamma" in res.output
    assert "http_5xx" in res.output
    assert "timeout" in res.output
    assert "HTTP 503" in res.output
