from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from monomarket.data.clients import FetchBatch
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
    error_buckets: dict[str, int] = field(default_factory=dict)


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


class IngestionClients(Protocol):
    def fetch_gamma(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch: ...

    def fetch_data(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch: ...

    def fetch_clob(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch: ...


class IngestionService:
    def __init__(
        self,
        clients: IngestionClients,
        storage: Storage,
        breaker_failure_threshold: int = 3,
        breaker_cooldown_sec: int = 90,
    ):
        self.clients = clients
        self.storage = storage
        self.breaker_failure_threshold = max(1, breaker_failure_threshold)
        self.breaker_cooldown_sec = max(1, breaker_cooldown_sec)

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

    def _breaker_gate(self, source: str) -> tuple[bool, str, bool]:
        row = self.storage.get_ingestion_breaker(source)
        if row is None:
            return False, "", False

        open_until = _parse_iso_ts(str(row.get("open_until_ts") or ""))
        if open_until is None:
            return False, "", False

        now = datetime.now(UTC)
        if now < open_until:
            return True, f"circuit open until {open_until.isoformat()}", False

        # cooldown passed: allow one half-open probe request
        self.storage.record_ingestion_breaker_transition(source, "half_open")
        return False, "half-open probe", True

    def _note_failure(self, source: str, error_bucket: str) -> None:
        row = self.storage.get_ingestion_breaker(source)
        consecutive = int(row["consecutive_failures"]) if row is not None else 0
        consecutive += 1

        open_until: str | None = None
        if consecutive >= self.breaker_failure_threshold:
            open_until = (
                datetime.now(UTC) + timedelta(seconds=self.breaker_cooldown_sec)
            ).isoformat()

        self.storage.set_ingestion_breaker(
            source=source,
            consecutive_failures=consecutive,
            open_until_ts=open_until,
            last_error_bucket=error_bucket,
        )
        if open_until is not None:
            self.storage.record_ingestion_breaker_transition(source, "open")

    def _note_probe_failure(self, source: str, error_bucket: str) -> None:
        # half-open single probe failed: immediately re-open breaker
        self.storage.set_ingestion_breaker(
            source=source,
            consecutive_failures=self.breaker_failure_threshold,
            open_until_ts=(
                datetime.now(UTC) + timedelta(seconds=self.breaker_cooldown_sec)
            ).isoformat(),
            last_error_bucket=error_bucket,
        )
        self.storage.record_ingestion_breaker_transition(source, "open")

    def _note_success(self, source: str) -> None:
        prev = self.storage.get_ingestion_breaker(source)
        self.storage.set_ingestion_breaker(
            source=source,
            consecutive_failures=0,
            open_until_ts=None,
            last_error_bucket="",
        )
        if prev is not None and (
            int(prev.get("consecutive_failures") or 0) > 0 or prev.get("open_until_ts")
        ):
            self.storage.record_ingestion_breaker_transition(source, "closed")

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
            error_buckets: dict[str, int] = {}

            for target in targets:
                is_open, breaker_msg, probe_mode = self._breaker_gate(target)
                if is_open:
                    errors.append(f"{target}: {breaker_msg}")
                    error_buckets["circuit_open"] = error_buckets.get("circuit_open", 0) + 1
                    self.storage.update_ingestion_error_bucket(
                        source=target,
                        error_bucket="circuit_open",
                        count=1,
                        last_error=breaker_msg,
                    )
                    continue

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

                for bucket, count in batch.stats.error_buckets.items():
                    error_buckets[bucket] = error_buckets.get(bucket, 0) + count
                    self.storage.update_ingestion_error_bucket(
                        source=target,
                        error_bucket=bucket,
                        count=count,
                        last_error=batch.error,
                    )

                if batch.error:
                    errors.append(f"{target}: {batch.error}")
                    failure_bucket = batch.stats.last_error_bucket or "unknown"
                    if probe_mode:
                        self._note_probe_failure(target, failure_bucket)
                    else:
                        self._note_failure(target, failure_bucket)
                else:
                    self._note_success(target)

            finished = datetime.now(UTC).isoformat()
            status = "partial" if errors else "ok"
            self.storage.record_ingestion(
                source_l,
                status,
                started,
                finished,
                inserted,
                "; ".join(errors),
                request_count=req_count,
                failure_count=fail_count,
                retry_count=retry_count,
                error_buckets=error_buckets,
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
                error_buckets=error_buckets,
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
