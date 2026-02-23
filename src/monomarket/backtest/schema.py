from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
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

# Placeholder v2 contract for migration rehearsal.
REQUIRED_BACKTEST_JSON_FIELDS_V2 = {
    "schema_version",
    "meta",
    "summary",
    "configs",
    "attribution",
    "replay",
}

BacktestJsonArtifactValidator = Callable[[Mapping[str, Any]], None]


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


def validate_backtest_json_artifact_v1(payload: Mapping[str, Any]) -> None:
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


def validate_backtest_json_artifact_v2(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_BACKTEST_JSON_FIELDS_V2 - set(payload.keys()))
    if missing:
        raise ValueError(f"missing required v2 fields: {', '.join(missing)}")

    if not isinstance(payload.get("meta"), Mapping):
        raise ValueError("v2 meta must be an object")
    if not isinstance(payload.get("summary"), Mapping):
        raise ValueError("v2 summary must be an object")
    if not isinstance(payload.get("configs"), Mapping):
        raise ValueError("v2 configs must be an object")
    if not isinstance(payload.get("attribution"), Mapping):
        raise ValueError("v2 attribution must be an object")
    if not isinstance(payload.get("replay"), list):
        raise ValueError("v2 replay must be an array")


def migrate_backtest_artifact_v1_to_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_backtest_json_artifact_v1(payload)

    return {
        "schema_version": "2.0",
        "meta": {
            "migration": "v1_to_v2",
            "source_schema_version": str(payload["schema_version"]),
        },
        "summary": {
            "generated_at": payload["generated_at"],
            "from_ts": payload["from_ts"],
            "to_ts": payload["to_ts"],
            "total_signals": payload["total_signals"],
            "executed_signals": payload["executed_signals"],
            "rejected_signals": payload["rejected_signals"],
        },
        "configs": {
            "execution": deepcopy(payload["execution_config"]),
            "risk": deepcopy(payload["risk_config"]),
        },
        "attribution": {
            "strategy": deepcopy(payload["results"]),
            "event": deepcopy(payload["event_results"]),
        },
        "replay": deepcopy(payload["replay"]),
    }


DEFAULT_BACKTEST_JSON_VALIDATORS: dict[int, BacktestJsonArtifactValidator] = {
    1: validate_backtest_json_artifact_v1,
    2: validate_backtest_json_artifact_v2,
}


def validate_backtest_json_artifact(
    payload: Mapping[str, Any],
    *,
    supported_major: int | None = SUPPORTED_BACKTEST_SCHEMA_MAJOR,
    validators: Mapping[int, BacktestJsonArtifactValidator] | None = None,
) -> tuple[int, int]:
    if "schema_version" not in payload:
        raise ValueError("missing required field: schema_version")

    major, minor = parse_schema_version(str(payload["schema_version"]))

    if supported_major is not None and major != supported_major:
        raise ValueError(
            f"unsupported schema_version={payload['schema_version']!r}; "
            f"supported major={supported_major}.x"
        )

    validator_map = dict(DEFAULT_BACKTEST_JSON_VALIDATORS)
    if validators:
        validator_map.update(validators)

    validator = validator_map.get(major)
    if validator is None:
        supported = ", ".join(str(x) for x in sorted(validator_map.keys()))
        raise ValueError(
            f"no validator registered for schema major={major}; available majors: {supported}"
        )

    validator(payload)
    return major, minor
