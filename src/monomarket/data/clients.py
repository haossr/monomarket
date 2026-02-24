from __future__ import annotations

import json
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
    error_buckets: dict[str, int] = field(default_factory=dict)
    last_error_bucket: str = ""


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

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, requests.Timeout):
            return "timeout"

        if isinstance(exc, requests.HTTPError):
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                return "http_429"
            if status_code is not None and 500 <= status_code <= 599:
                return "http_5xx"
            if status_code is not None and 400 <= status_code <= 499:
                return "http_4xx"
            return "http_other"

        if isinstance(exc, requests.ConnectionError):
            return "network"

        if isinstance(exc, requests.RequestException):
            return "request"

        if isinstance(exc, ValueError):
            return "decode"

        return "unknown"

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status_code = exc.response.status_code
            text = str(exc).strip()
            if text:
                return f"HTTP {status_code}: {text}"
            return f"HTTP {status_code}"
        return str(exc)

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

                bucket = self._classify_error(exc)
                stats.last_error_bucket = bucket
                stats.error_buckets[bucket] = stats.error_buckets.get(bucket, 0) + 1
                error_message = self._error_message(exc)

                retryable = bucket not in {"http_4xx"}
                if attempt >= self.max_retries or not retryable:
                    return [], stats, error_message

                backoff_multiplier = 1.0
                if bucket == "http_429":
                    backoff_multiplier = 4.0
                elif bucket == "http_5xx":
                    backoff_multiplier = 2.0

                backoff = self.backoff_base_sec * (2**attempt) * backoff_multiplier
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


def _parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value
    if not (text.startswith("[") or text.startswith("{")):
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _to_outcome_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_json_like(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _to_list(value: Any) -> list[Any]:
    parsed = _parse_json_like(value)
    if isinstance(parsed, list):
        return parsed
    return []


def _extract_prices(item: dict[str, Any]) -> tuple[float | None, float | None]:
    yes = item.get("yes_price")
    no = item.get("no_price")

    if yes is not None or no is not None:
        return (
            _as_float(yes, 0.0) if yes is not None else None,
            _as_float(no, 0.0) if no is not None else None,
        )

    outcome_prices = item.get("outcomePrices")
    outcome_names = item.get("outcomes")

    prices_list = _to_list(outcome_prices)
    names_list = _to_list(outcome_names)

    if prices_list and len(prices_list) >= 2:
        if names_list and len(names_list) >= 2:
            by_name: dict[str, Any] = {}
            for idx, raw_name in enumerate(names_list):
                if idx >= len(prices_list):
                    break
                key = str(raw_name).strip().lower()
                if key:
                    by_name[key] = prices_list[idx]

            y = by_name.get("yes")
            n = by_name.get("no")
            if y is not None or n is not None:
                return (
                    _as_float(y, 0.0) if y is not None else None,
                    _as_float(n, 0.0) if n is not None else None,
                )

        try:
            return float(prices_list[0]), float(prices_list[1])
        except (TypeError, ValueError):
            return None, None

    outcome_prices_dict = _to_outcome_dict(outcome_prices)
    if outcome_prices_dict:
        y = (
            outcome_prices_dict.get("YES")
            or outcome_prices_dict.get("Yes")
            or outcome_prices_dict.get("yes")
        )
        n = (
            outcome_prices_dict.get("NO")
            or outcome_prices_dict.get("No")
            or outcome_prices_dict.get("no")
        )
        if y is not None or n is not None:
            return (
                _as_float(y, 0.0) if y is not None else None,
                _as_float(n, 0.0) if n is not None else None,
            )

    outcomes = _parse_json_like(item.get("outcomes"))
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

    if isinstance(value, int | float):
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


def _extract_event_id(item: dict[str, Any]) -> str:
    for key in ("event_id", "eventId"):
        value = item.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text

    event = item.get("event")
    if isinstance(event, dict):
        for key in ("id", "event_id", "eventId", "slug", "ticker"):
            value = event.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    elif event is not None:
        text = str(event).strip()
        if text:
            return text

    events = _to_list(item.get("events"))
    if events:
        first = events[0]
        if isinstance(first, dict):
            for key in ("id", "event_id", "eventId", "slug", "ticker"):
                value = first.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text

    return "unknown"


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
    event_id = _extract_event_id(item)
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
            {
                "limit": limit,
                "active": "true",
                "closed": "false",
                "archived": "false",
            },
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

        merged_stats = RequestStats()

        for path in ("markets", "v1/markets", "v2/markets"):
            payload, stats, error = self.data.get_json(path, params=params)
            merged_stats.requests += stats.requests
            merged_stats.failures += stats.failures
            merged_stats.retries += stats.retries
            merged_stats.throttled_sec += stats.throttled_sec
            merged_stats.last_error_bucket = stats.last_error_bucket
            for bucket, count in stats.error_buckets.items():
                merged_stats.error_buckets[bucket] = (
                    merged_stats.error_buckets.get(bucket, 0) + count
                )

            if not error:
                rows, max_timestamp = self._normalize_rows(
                    payload,
                    "data",
                    since=since if incremental else None,
                )
                return FetchBatch(
                    rows=rows,
                    stats=merged_stats,
                    max_timestamp=max_timestamp,
                    error="",
                )

            not_found = stats.last_error_bucket == "http_4xx" and "404" in error
            if not not_found:
                return FetchBatch(rows=[], stats=merged_stats, max_timestamp=None, error=error)

        # data-api 在部分部署中不提供 markets 路径：降级为跳过，不触发 breaker。
        merged_stats.failures = 0
        merged_stats.last_error_bucket = ""
        merged_stats.error_buckets = {"source_skipped_404": 1}
        return FetchBatch(rows=[], stats=merged_stats, max_timestamp=None, error="")

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
