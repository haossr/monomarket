from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}


@dataclass(slots=True)
class AppSettings:
    db_path: str = "data/monomarket.db"
    log_level: str = "INFO"


@dataclass(slots=True)
class TradingSettings:
    mode: str = "paper"
    enable_live_trading: bool = False
    require_manual_confirm: bool = True
    kill_switch: bool = False


@dataclass(slots=True)
class RiskSettings:
    max_daily_loss: float = 250.0
    max_strategy_notional: float = 1000.0
    max_event_notional: float = 1500.0
    circuit_breaker_rejections: int = 5


@dataclass(slots=True)
class DataSettings:
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    timeout_sec: int = 15


@dataclass(slots=True)
class Settings:
    app: AppSettings
    trading: TradingSettings
    risk: RiskSettings
    data: DataSettings
    strategies: dict[str, dict[str, Any]]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return _as_bool(raw, False)


def _candidate_paths(config_path: str | None) -> list[Path]:
    out: list[Path] = []
    if config_path:
        out.append(Path(config_path))
    out.append(Path("configs/config.yaml"))
    out.append(Path("configs/config.example.yaml"))
    return out


def load_settings(config_path: str | None = None) -> Settings:
    chosen: Path | None = None
    for candidate in _candidate_paths(config_path):
        if candidate.exists():
            chosen = candidate
            break

    raw: dict[str, Any] = {}
    if chosen is not None:
        raw = yaml.safe_load(chosen.read_text()) or {}

    app_raw = raw.get("app", {})
    trading_raw = raw.get("trading", {})
    risk_raw = raw.get("risk", {})
    data_raw = raw.get("data", {})

    trading = TradingSettings(
        mode=str(trading_raw.get("mode", "paper")).lower(),
        enable_live_trading=_as_bool(trading_raw.get("enable_live_trading"), False),
        require_manual_confirm=_as_bool(trading_raw.get("require_manual_confirm"), True),
        kill_switch=_as_bool(trading_raw.get("kill_switch"), False),
    )

    env_live = _pick_env_bool("ENABLE_LIVE_TRADING")
    env_manual = _pick_env_bool("REQUIRE_MANUAL_CONFIRM")
    env_kill = _pick_env_bool("KILL_SWITCH")
    env_mode = os.getenv("MONOMARKET_MODE")

    if env_live is not None:
        trading.enable_live_trading = env_live
    if env_manual is not None:
        trading.require_manual_confirm = env_manual
    if env_kill is not None:
        trading.kill_switch = env_kill
    if env_mode:
        trading.mode = env_mode.strip().lower()

    return Settings(
        app=AppSettings(
            db_path=str(app_raw.get("db_path", "data/monomarket.db")),
            log_level=str(app_raw.get("log_level", "INFO")),
        ),
        trading=trading,
        risk=RiskSettings(
            max_daily_loss=_as_float(risk_raw.get("max_daily_loss"), 250.0),
            max_strategy_notional=_as_float(risk_raw.get("max_strategy_notional"), 1000.0),
            max_event_notional=_as_float(risk_raw.get("max_event_notional"), 1500.0),
            circuit_breaker_rejections=_as_int(risk_raw.get("circuit_breaker_rejections"), 5),
        ),
        data=DataSettings(
            gamma_base_url=str(data_raw.get("gamma_base_url", "https://gamma-api.polymarket.com")),
            data_base_url=str(data_raw.get("data_base_url", "https://data-api.polymarket.com")),
            clob_base_url=str(data_raw.get("clob_base_url", "https://clob.polymarket.com")),
            timeout_sec=_as_int(data_raw.get("timeout_sec"), 15),
        ),
        strategies=raw.get("strategies", {}),
    )
