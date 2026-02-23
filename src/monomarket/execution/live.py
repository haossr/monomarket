from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from monomarket.db.storage import Storage
from monomarket.models import OrderRequest, OrderResult

DEFAULT_CLOB_BASE_URL = "https://clob.polymarket.com"
ORDER_STATUS_SYNCABLE = {"submitted", "accepted", "open", "partially_filled", "cancel_pending"}


class LiveApiError(RuntimeError):
    pass


@dataclass(slots=True)
class LiveOrderReport:
    external_id: str | None
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    message: str = ""
    external_fill_id: str | None = None
    raw: dict[str, Any] | None = None


class ClobApiClient(Protocol):
    def place_order(self, req: OrderRequest) -> LiveOrderReport: ...

    def cancel_order(self, external_id: str) -> LiveOrderReport: ...

    def get_order(self, external_id: str) -> LiveOrderReport: ...


@dataclass(slots=True)
class RequestsClobClient:
    base_url: str
    timeout_sec: int = 15

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        raw_json = os.getenv("POLYMARKET_CLOB_HEADERS_JSON", "").strip()
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise LiveApiError("POLYMARKET_CLOB_HEADERS_JSON is not valid json") from exc
            if not isinstance(parsed, dict):
                raise LiveApiError("POLYMARKET_CLOB_HEADERS_JSON must be a json object")
            for k, v in parsed.items():
                headers[str(k)] = str(v)

        api_key = os.getenv("POLYMARKET_API_KEY", "").strip()
        api_secret = os.getenv("POLYMARKET_API_SECRET", "").strip()
        passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()
        if api_key and api_secret and passphrase:
            headers["X-API-KEY"] = api_key
            headers["X-API-SECRET"] = api_secret
            headers["X-API-PASSPHRASE"] = passphrase

        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else {"data": body}
        except (requests.RequestException, ValueError) as exc:
            raise LiveApiError(f"{method.upper()} {path}: {exc}") from exc

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _normalize_report(
        cls,
        payload: dict[str, Any],
        fallback_external_id: str | None = None,
    ) -> LiveOrderReport:
        root = payload.get("order")
        if isinstance(root, dict):
            body = root
        else:
            body = payload

        external_id = body.get("id") or body.get("order_id") or body.get("orderId")
        if external_id is None:
            external_id = fallback_external_id

        status_raw = body.get("status") or payload.get("status") or "accepted"
        status = str(status_raw).strip().lower()

        filled_qty = cls._as_float(
            body.get("filled_size")
            or body.get("filledSize")
            or body.get("filled_qty")
            or body.get("filledQty")
            or body.get("filled")
            or payload.get("filled")
        )
        avg_fill_price = cls._as_float(
            body.get("avg_price")
            or body.get("avgPrice")
            or body.get("average_fill_price")
            or body.get("averageFillPrice")
            or body.get("price")
            or payload.get("price")
        )

        fill_id = body.get("fill_id") or body.get("trade_id") or payload.get("fill_id")
        message = str(body.get("message") or payload.get("message") or status)

        return LiveOrderReport(
            external_id=str(external_id) if external_id is not None else None,
            status=status,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            message=message,
            external_fill_id=str(fill_id) if fill_id is not None else None,
            raw=payload,
        )

    def place_order(self, req: OrderRequest) -> LiveOrderReport:
        payload = {
            "market_id": req.market_id,
            "event_id": req.event_id,
            "token_id": req.token_id,
            "side": req.side.lower(),
            "action": req.action.lower(),
            "price": req.price,
            "size": req.qty,
            "order_type": "GTC",
            "client_tag": "monomarket",
        }
        raw = self._request("post", "/order", payload=payload)
        return self._normalize_report(raw)

    def cancel_order(self, external_id: str) -> LiveOrderReport:
        try:
            raw = self._request("delete", f"/order/{external_id}")
        except LiveApiError:
            # fallback endpoint some gateways expose
            raw = self._request("post", "/order/cancel", payload={"order_id": external_id})
        report = self._normalize_report(raw, fallback_external_id=external_id)
        if report.status in {"ok", "success", "accepted"}:
            report.status = "canceled"
        return report

    def get_order(self, external_id: str) -> LiveOrderReport:
        raw = self._request("get", f"/order/{external_id}")
        return self._normalize_report(raw, fallback_external_id=external_id)


class LiveExecutor:
    """Live CLOB executor with place/cancel/report closed loop.

    Live path is still guarded by router switches. This executor only runs after
    ENABLE_LIVE_TRADING checks passed.
    """

    def __init__(
        self,
        storage: Storage,
        base_url: str = DEFAULT_CLOB_BASE_URL,
        timeout_sec: int = 15,
        client: ClobApiClient | None = None,
    ):
        self.storage = storage
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self._client = client

    @staticmethod
    def _has_live_credentials() -> bool:
        if os.getenv("POLYMARKET_CLOB_HEADERS_JSON", "").strip():
            return True

        key = os.getenv("POLYMARKET_API_KEY", "").strip()
        secret = os.getenv("POLYMARKET_API_SECRET", "").strip()
        passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()
        return bool(key and secret and passphrase)

    def _client_or_default(self) -> ClobApiClient:
        if self._client is not None:
            return self._client
        self._client = RequestsClobClient(base_url=self.base_url, timeout_sec=self.timeout_sec)
        return self._client

    @staticmethod
    def _to_local_status(status: str) -> str:
        s = status.strip().lower()
        if s in {"filled", "closed", "done"}:
            return "filled"
        if s in {"partial", "partially_filled", "partially-filled"}:
            return "partially_filled"
        if s in {"canceled", "cancelled", "cancel"}:
            return "canceled"
        if s in {"rejected", "error", "failed"}:
            return "rejected"
        if s in {"open", "accepted", "new", "live", "pending"}:
            return "accepted"
        if s == "submitted":
            return "submitted"
        return "accepted"

    @staticmethod
    def _is_accepted_status(status: str) -> bool:
        return status in {"submitted", "accepted", "partially_filled", "filled", "canceled"}

    def _apply_fill_from_report(
        self,
        order_id: int,
        req: OrderRequest,
        report: LiveOrderReport,
    ) -> float:
        total_filled = max(0.0, report.filled_qty)
        if total_filled <= 0.0:
            return 0.0

        already_filled = self.storage.order_filled_qty(order_id)
        delta = total_filled - already_filled
        if delta <= 1e-9:
            return 0.0

        fill_price = report.avg_fill_price if report.avg_fill_price > 0 else req.price
        self.storage.record_fill(
            order_id=order_id,
            strategy=req.strategy,
            market_id=req.market_id,
            event_id=req.event_id,
            token_id=req.token_id,
            side=req.side,
            price=fill_price,
            qty=delta,
            fee=0.0,
            external_fill_id=report.external_fill_id,
            raw_report_json=(json.dumps(report.raw, ensure_ascii=False) if report.raw else ""),
        )
        return delta

    def execute(self, req: OrderRequest) -> OrderResult:
        if not self._has_live_credentials():
            order_id = self.storage.insert_order(
                req,
                status="rejected",
                message=(
                    "live credentials missing: set POLYMARKET_CLOB_HEADERS_JSON "
                    "or POLYMARKET_API_KEY+POLYMARKET_API_SECRET+POLYMARKET_API_PASSPHRASE"
                ),
            )
            return OrderResult(
                accepted=False,
                status="rejected",
                order_id=order_id,
                message=(
                    "live credentials missing: set POLYMARKET_CLOB_HEADERS_JSON "
                    "or POLYMARKET_API_KEY+POLYMARKET_API_SECRET+POLYMARKET_API_PASSPHRASE"
                ),
            )

        order_id = self.storage.insert_order(req, status="submitted", message="live submit pending")

        try:
            report = self._client_or_default().place_order(req)
        except LiveApiError as exc:
            message = f"live submit failed: {exc}"
            self.storage.update_order_status(order_id, status="rejected", message=message)
            return OrderResult(False, "rejected", order_id, message)

        status = self._to_local_status(report.status)
        message = report.message or f"live {status}"

        self.storage.update_order_status(
            order_id,
            status=status,
            message=message,
            external_id=report.external_id,
        )
        self._apply_fill_from_report(order_id, req, report)

        return OrderResult(
            accepted=self._is_accepted_status(status),
            status=status,
            order_id=order_id,
            message=message,
            external_id=report.external_id,
        )

    def cancel(self, order_id: int) -> OrderResult:
        row = self.storage.get_order(order_id)
        if row is None:
            return OrderResult(False, "rejected", None, f"order {order_id} not found")

        if str(row.get("mode", "")).lower() != "live":
            return OrderResult(False, "rejected", order_id, "only live orders can be canceled")

        external_id = str(row.get("external_id") or "").strip()
        if not external_id:
            return OrderResult(False, "rejected", order_id, "live order missing external_id")

        try:
            report = self._client_or_default().cancel_order(external_id)
        except LiveApiError as exc:
            message = f"live cancel failed: {exc}"
            self.storage.update_order_status(order_id, status=str(row["status"]), message=message)
            return OrderResult(
                False, str(row["status"]), order_id, message, external_id=external_id
            )

        status = self._to_local_status(report.status)
        if status not in {"canceled", "rejected"}:
            status = "canceled"
        message = report.message or "live canceled"
        self.storage.update_order_status(
            order_id, status=status, message=message, external_id=external_id
        )

        return OrderResult(
            accepted=True,
            status=status,
            order_id=order_id,
            message=message,
            external_id=external_id,
        )

    def sync_open_orders(self, limit: int = 100) -> dict[str, int | float]:
        rows = self.storage.list_orders(
            mode="live", statuses=sorted(ORDER_STATUS_SYNCABLE), limit=limit
        )

        orders_updated = 0
        errors = 0
        filled_delta_qty = 0.0

        for row in rows:
            local_order_id = int(row["id"])
            external_id = str(row.get("external_id") or "").strip()
            if not external_id:
                continue

            req = OrderRequest(
                strategy=str(row["strategy"]),
                market_id=str(row["market_id"]),
                event_id=str(row["event_id"]),
                token_id=str(row["token_id"]),
                side=str(row["side"]),
                action=str(row["action"]),
                price=float(row["price"]),
                qty=float(row["qty"]),
                mode="live",
                reason="live-sync",
            )

            try:
                report = self._client_or_default().get_order(external_id)
            except LiveApiError as exc:
                errors += 1
                self.storage.update_order_status(
                    local_order_id,
                    status=str(row["status"]),
                    message=f"live sync failed: {exc}",
                    external_id=external_id,
                )
                continue

            status = self._to_local_status(report.status)
            message = report.message or f"sync {status}"
            self.storage.update_order_status(
                local_order_id,
                status=status,
                message=message,
                external_id=external_id,
            )
            filled_delta_qty += self._apply_fill_from_report(local_order_id, req, report)
            orders_updated += 1

        return {
            "orders_scanned": len(rows),
            "orders_updated": orders_updated,
            "errors": errors,
            "filled_delta_qty": round(filled_delta_qty, 8),
        }
