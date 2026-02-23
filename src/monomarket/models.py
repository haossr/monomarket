from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MarketView:
    market_id: str
    canonical_id: str
    source: str
    event_id: str
    question: str
    status: str
    neg_risk: bool
    liquidity: float
    volume: float
    yes_price: float | None = None
    no_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None


@dataclass(slots=True)
class Signal:
    strategy: str
    market_id: str
    event_id: str
    side: str
    score: float
    confidence: float
    target_price: float
    size_hint: float
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    strategy: str
    market_id: str
    event_id: str
    token_id: str
    side: str  # buy / sell
    action: str  # open / close
    price: float
    qty: float
    mode: str = "paper"
    reason: str = ""


@dataclass(slots=True)
class OrderResult:
    accepted: bool
    status: str
    order_id: int | None
    message: str
    external_id: str | None = None


@dataclass(slots=True)
class PnlLine:
    strategy: str
    market_id: str
    token_id: str
    net_qty: float
    avg_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass(slots=True)
class PnlReport:
    generated_at: datetime
    realized_total: float
    unrealized_total: float
    by_strategy: dict[str, float]
    lines: list[PnlLine]


@dataclass(slots=True)
class MetricsReport:
    generated_at: datetime
    total_orders: int
    filled_orders: int
    rejected_orders: int
    fill_rate: float
    rejection_rate: float
    realized_pnl: float
    unrealized_pnl: float
    max_drawdown: float
