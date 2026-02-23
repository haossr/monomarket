from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from monomarket.models import MarketView, Signal


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        markets: list[MarketView],
        strategy_config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        raise NotImplementedError
