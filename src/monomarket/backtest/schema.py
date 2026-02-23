from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_BACKTEST_SCHEMA_MAJOR = 1

REQUIRED_BACKTEST_JSON_FIELDS_V1 = {
    "schema_version",
    "generated_at",
    "from_ts",
    "to_ts",
    "total_signals",
    "executed_signals",
    "rejected_signals",
    "execution_config",
    "risk_config",
    "results",
    "event_results",
    "replay",
}


def parse_schema_version(raw: str) -> tuple[int, int]:
    text = raw.strip()
    parts = text.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"invalid schema_version format: {raw!r}, expected '<major>.<minor>'")
    return int(parts[0]), int(parts[1])


def is_schema_compatible(
    raw: str, *, supported_major: int = SUPPORTED_BACKTEST_SCHEMA_MAJOR
) -> bool:
    major, _ = parse_schema_version(raw)
    return major == supported_major


def assert_schema_compatible(
    raw: str,
    *,
    supported_major: int = SUPPORTED_BACKTEST_SCHEMA_MAJOR,
) -> tuple[int, int]:
    major, minor = parse_schema_version(raw)
    if major != supported_major:
        raise ValueError(f"unsupported schema_version={raw!r}; supported major={supported_major}.x")
    return major, minor


def validate_backtest_json_artifact(
    payload: Mapping[str, Any],
    *,
    supported_major: int = SUPPORTED_BACKTEST_SCHEMA_MAJOR,
) -> tuple[int, int]:
    if "schema_version" not in payload:
        raise ValueError("missing required field: schema_version")

    major, minor = assert_schema_compatible(
        str(payload["schema_version"]),
        supported_major=supported_major,
    )

    if major == 1:
        missing = sorted(REQUIRED_BACKTEST_JSON_FIELDS_V1 - set(payload.keys()))
        if missing:
            raise ValueError(f"missing required v1 fields: {', '.join(missing)}")

        if not isinstance(payload.get("execution_config"), Mapping):
            raise ValueError("v1 execution_config must be an object")
        if not isinstance(payload.get("risk_config"), Mapping):
            raise ValueError("v1 risk_config must be an object")
        if not isinstance(payload.get("results"), list):
            raise ValueError("v1 results must be an array")
        if not isinstance(payload.get("event_results"), list):
            raise ValueError("v1 event_results must be an array")
        if not isinstance(payload.get("replay"), list):
            raise ValueError("v1 replay must be an array")

    return major, minor
