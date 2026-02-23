from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from monomarket.data.clients import MarketDataClients
from monomarket.db.storage import Storage


@dataclass(slots=True)
class IngestionResult:
    source: str
    rows: int
    started_at: str
    finished_at: str
    status: str
    error: str = ""


class IngestionService:
    def __init__(self, clients: MarketDataClients, storage: Storage):
        self.clients = clients
        self.storage = storage

    def ingest(self, source: str, limit: int = 200) -> IngestionResult:
        started = datetime.now(UTC).isoformat()
        try:
            source_l = source.lower()
            if source_l == "gamma":
                rows = self.clients.fetch_gamma(limit)
            elif source_l == "data":
                rows = self.clients.fetch_data(limit)
            elif source_l == "clob":
                rows = self.clients.fetch_clob(limit)
            elif source_l == "all":
                rows = [
                    *self.clients.fetch_gamma(limit),
                    *self.clients.fetch_data(limit),
                    *self.clients.fetch_clob(limit),
                ]
            else:
                raise ValueError(f"Unsupported source: {source}")

            inserted = self.storage.upsert_markets(rows)
            finished = datetime.now(UTC).isoformat()
            self.storage.record_ingestion(source_l, "ok", started, finished, inserted, "")
            return IngestionResult(
                source=source_l,
                rows=inserted,
                started_at=started,
                finished_at=finished,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            finished = datetime.now(UTC).isoformat()
            self.storage.record_ingestion(source.lower(), "error", started, finished, 0, str(exc))
            return IngestionResult(
                source=source.lower(),
                rows=0,
                started_at=started,
                finished_at=finished,
                status="error",
                error=str(exc),
            )
