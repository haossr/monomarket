from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from monomarket.config import DataSettings
from monomarket.models import MarketView


@dataclass(slots=True)
class SourceClient:
    base_url: str
    timeout_sec: int

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout_sec)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return []


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
        self.gamma = SourceClient(settings.gamma_base_url, settings.timeout_sec)
        self.data = SourceClient(settings.data_base_url, settings.timeout_sec)
        self.clob = SourceClient(settings.clob_base_url, settings.timeout_sec)

    @staticmethod
    def _normalize_rows(payload: Any, source: str) -> list[MarketView]:
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
        for item in rows:
            if not isinstance(item, dict):
                continue
            n = normalize_market(item, source)
            if n is not None:
                normalized.append(n)
        return normalized

    def fetch_gamma(self, limit: int = 200) -> list[MarketView]:
        payload = self.gamma.get_json("markets", params={"limit": limit, "active": "true"})
        return self._normalize_rows(payload, "gamma")

    def fetch_data(self, limit: int = 200) -> list[MarketView]:
        payload = self.data.get_json("markets", params={"limit": limit})
        return self._normalize_rows(payload, "data")

    def fetch_clob(self, limit: int = 200) -> list[MarketView]:
        payload = self.clob.get_json("markets", params={"limit": limit})
        return self._normalize_rows(payload, "clob")
