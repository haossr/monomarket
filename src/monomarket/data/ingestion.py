from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from monomarket.data.clients import FetchBatch, MarketDataClients
from monomarket.db.storage import Storage


@dataclass(slots=True)
class IngestionResult:
    source: str
    rows: int
    started_at: str
    finished_at: str
    status: str
    error: str = ""
    request_count: int = 0
    failure_count: int = 0
    retry_count: int = 0


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class IngestionService:
    def __init__(self, clients: MarketDataClients, storage: Storage):
        self.clients = clients
        self.storage = storage

    def _fetch_one(self, source: str, limit: int, incremental: bool) -> FetchBatch:
        since = (
            _parse_iso_ts(self.storage.get_ingestion_checkpoint(source)) if incremental else None
        )

        if source == "gamma":
            return self.clients.fetch_gamma(limit, since=since, incremental=incremental)
        if source == "data":
            return self.clients.fetch_data(limit, since=since, incremental=incremental)
        if source == "clob":
            return self.clients.fetch_clob(limit, since=since, incremental=incremental)
        raise ValueError(f"Unsupported source: {source}")

    def ingest(self, source: str, limit: int = 200, incremental: bool = True) -> IngestionResult:
        started = datetime.now(UTC).isoformat()
        source_l = source.lower().strip()

        try:
            if source_l == "all":
                targets = ["gamma", "data", "clob"]
            elif source_l in {"gamma", "data", "clob"}:
                targets = [source_l]
            else:
                raise ValueError(f"Unsupported source: {source}")

            inserted = 0
            req_count = 0
            fail_count = 0
            retry_count = 0
            errors: list[str] = []

            for target in targets:
                batch = self._fetch_one(target, limit=limit, incremental=incremental)
                inserted += self.storage.upsert_markets(batch.rows)
                req_count += batch.stats.requests
                fail_count += batch.stats.failures
                retry_count += batch.stats.retries

                if batch.max_timestamp:
                    self.storage.update_ingestion_checkpoint(target, batch.max_timestamp)
                self.storage.update_ingestion_counters(
                    source=target,
                    request_count=batch.stats.requests,
                    failure_count=batch.stats.failures,
                )

                if batch.error:
                    errors.append(f"{target}: {batch.error}")

            finished = datetime.now(UTC).isoformat()
            status = "partial" if errors else "ok"
            self.storage.record_ingestion(
                source_l,
                status,
                started,
                finished,
                inserted,
                "; ".join(errors),
            )
            return IngestionResult(
                source=source_l,
                rows=inserted,
                started_at=started,
                finished_at=finished,
                status=status,
                error="; ".join(errors),
                request_count=req_count,
                failure_count=fail_count,
                retry_count=retry_count,
            )
        except Exception as exc:  # noqa: BLE001
            finished = datetime.now(UTC).isoformat()
            self.storage.record_ingestion(source_l, "error", started, finished, 0, str(exc))
            return IngestionResult(
                source=source_l,
                rows=0,
                started_at=started,
                finished_at=finished,
                status="error",
                error=str(exc),
            )
