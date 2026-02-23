from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

from monomarket.config import DataSettings
from monomarket.models import MarketView


@dataclass(slots=True)
class RequestStats:
    requests: int = 0
    failures: int = 0
    retries: int = 0
    throttled_sec: float = 0.0


@dataclass(slots=True)
class FetchBatch:
    rows: list[MarketView]
    stats: RequestStats = field(default_factory=RequestStats)
    max_timestamp: str | None = None
    error: str = ""


@dataclass(slots=True)
class SourceClient:
    base_url: str
    timeout_sec: int
    max_retries: int
    backoff_base_sec: float
    rate_limit_per_sec: float
    _last_request_monotonic: float | None = field(default=None, init=False, repr=False)
    _min_interval_sec: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._min_interval_sec = (
            1.0 / self.rate_limit_per_sec if self.rate_limit_per_sec > 0 else 0.0
        )

    def _throttle(self, stats: RequestStats) -> None:
        if self._min_interval_sec <= 0.0 or self._last_request_monotonic is None:
            return

        elapsed = time.monotonic() - self._last_request_monotonic
        sleep_for = self._min_interval_sec - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
            stats.throttled_sec += sleep_for

    def get_json(
        self, path: str, params: dict[str, Any] | None = None
    ) -> tuple[Any, RequestStats, str]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        stats = RequestStats()

        for attempt in range(self.max_retries + 1):
            self._throttle(stats)
            stats.requests += 1
            try:
                resp = requests.get(url, params=params, timeout=self.timeout_sec)
                self._last_request_monotonic = time.monotonic()
                resp.raise_for_status()
                return resp.json(), stats, ""
            except (requests.RequestException, ValueError) as exc:
                self._last_request_monotonic = time.monotonic()
                stats.failures += 1
                if attempt >= self.max_retries:
                    return [], stats, str(exc)

                backoff = self.backoff_base_sec * (2**attempt)
                if backoff > 0:
                    time.sleep(backoff)
                stats.retries += 1

        return [], stats, "unreachable"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extract_prices(item: dict[str, Any]) -> tuple[float | None, float | None]:
    yes = item.get("yes_price")
    no = item.get("no_price")

    if yes is not None or no is not None:
        return (
            _as_float(yes, 0.0) if yes is not None else None,
            _as_float(no, 0.0) if no is not None else None,
        )

    outcomes = item.get("outcomePrices") or item.get("outcomes")
    if isinstance(outcomes, list) and len(outcomes) >= 2:
        try:
            return float(outcomes[0]), float(outcomes[1])
        except (TypeError, ValueError):
            return None, None

    if isinstance(outcomes, dict):
        y = outcomes.get("YES") or outcomes.get("Yes") or outcomes.get("yes")
        n = outcomes.get("NO") or outcomes.get("No") or outcomes.get("no")
        return (
            _as_float(y, 0.0) if y is not None else None,
            _as_float(n, 0.0) if n is not None else None,
        )

    return None, None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (TypeError, ValueError, OSError):
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (TypeError, ValueError, OSError):
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def normalize_market(item: dict[str, Any], source: str) -> MarketView | None:
    market_id = str(
        item.get("market_id")
        or item.get("id")
        or item.get("conditionId")
        or item.get("condition_id")
        or ""
    ).strip()
    if not market_id:
        return None

    question = str(item.get("question") or item.get("title") or item.get("name") or market_id)
    event_id = str(item.get("event_id") or item.get("eventId") or item.get("event") or "unknown")
    canonical_id = str(
        item.get("canonical_id")
        or item.get("groupItemTitle")
        or item.get("slug")
        or f"{event_id}:{question[:60]}"
    )

    yes, no = _extract_prices(item)
    best_bid = item.get("bestBid") or item.get("best_bid")
    best_ask = item.get("bestAsk") or item.get("best_ask")
    best_bid_f = _as_float(best_bid, 0.0) if best_bid is not None else None
    best_ask_f = _as_float(best_ask, 0.0) if best_ask is not None else None

    mid: float | None = None
    if best_bid_f is not None and best_ask_f is not None and best_bid_f > 0 and best_ask_f > 0:
        mid = (best_bid_f + best_ask_f) / 2
    elif yes is not None:
        mid = yes

    closed = _as_bool(item.get("closed"), False)
    active = _as_bool(item.get("active"), not closed)
    status = "closed" if closed or not active else "open"

    return MarketView(
        market_id=market_id,
        canonical_id=canonical_id,
        source=source,
        event_id=event_id,
        question=question,
        status=status,
        neg_risk=_as_bool(item.get("negRisk") or item.get("neg_risk"), False),
        liquidity=_as_float(item.get("liquidity") or item.get("liquidityNum") or 0.0),
        volume=_as_float(item.get("volume") or item.get("volumeNum") or 0.0),
        yes_price=yes,
        no_price=no,
        best_bid=best_bid_f,
        best_ask=best_ask_f,
        mid_price=mid,
    )


class MarketDataClients:
    def __init__(self, settings: DataSettings):
        self.gamma = SourceClient(
            settings.gamma_base_url,
            settings.timeout_sec,
            settings.max_retries,
            settings.backoff_base_sec,
            settings.rate_limit_per_sec,
        )
        self.data = SourceClient(
            settings.data_base_url,
            settings.timeout_sec,
            settings.max_retries,
            settings.backoff_base_sec,
            settings.rate_limit_per_sec,
        )
        self.clob = SourceClient(
            settings.clob_base_url,
            settings.timeout_sec,
            settings.max_retries,
            settings.backoff_base_sec,
            settings.rate_limit_per_sec,
        )

    @staticmethod
    def _extract_item_timestamp(item: dict[str, Any]) -> datetime | None:
        for key in (
            "updatedAt",
            "updated_at",
            "lastUpdated",
            "last_updated",
            "endDate",
            "end_date",
            "createdAt",
            "created_at",
        ):
            ts = _parse_timestamp(item.get(key))
            if ts is not None:
                return ts
        return None

    @classmethod
    def _normalize_rows(
        cls,
        payload: Any,
        source: str,
        since: datetime | None = None,
    ) -> tuple[list[MarketView], str | None]:
        rows: list[dict[str, Any]]
        if isinstance(payload, dict):
            for key in ("markets", "data", "items", "results"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            else:
                rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        normalized: list[MarketView] = []
        max_seen_ts: datetime | None = None

        for item in rows:
            if not isinstance(item, dict):
                continue

            ts = cls._extract_item_timestamp(item)
            if ts is not None and (max_seen_ts is None or ts > max_seen_ts):
                max_seen_ts = ts
            if since is not None and ts is not None and ts <= since:
                continue

            n = normalize_market(item, source)
            if n is not None:
                normalized.append(n)

        max_seen_iso = max_seen_ts.isoformat() if max_seen_ts is not None else None
        return normalized, max_seen_iso

    @staticmethod
    def _with_incremental_params(
        base_params: dict[str, Any],
        since: datetime | None,
        incremental: bool,
    ) -> dict[str, Any]:
        params = dict(base_params)
        if incremental and since is not None:
            since_iso = since.astimezone(UTC).isoformat()
            params["updated_after"] = since_iso
            params["since"] = since_iso
        return params

    def fetch_gamma(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        params = self._with_incremental_params(
            {"limit": limit, "active": "true"},
            since=since,
            incremental=incremental,
        )
        payload, stats, error = self.gamma.get_json("markets", params=params)
        rows, max_timestamp = self._normalize_rows(
            payload,
            "gamma",
            since=since if incremental else None,
        )
        return FetchBatch(rows=rows, stats=stats, max_timestamp=max_timestamp, error=error)

    def fetch_data(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        params = self._with_incremental_params(
            {"limit": limit},
            since=since,
            incremental=incremental,
        )
        payload, stats, error = self.data.get_json("markets", params=params)
        rows, max_timestamp = self._normalize_rows(
            payload, "data", since=since if incremental else None
        )
        return FetchBatch(rows=rows, stats=stats, max_timestamp=max_timestamp, error=error)

    def fetch_clob(
        self,
        limit: int = 200,
        since: datetime | None = None,
        incremental: bool = False,
    ) -> FetchBatch:
        params = self._with_incremental_params(
            {"limit": limit},
            since=since,
            incremental=incremental,
        )
        payload, stats, error = self.clob.get_json("markets", params=params)
        rows, max_timestamp = self._normalize_rows(
            payload, "clob", since=since if incremental else None
        )
        return FetchBatch(rows=rows, stats=stats, max_timestamp=max_timestamp, error=error)
