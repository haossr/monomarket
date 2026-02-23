from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypedDict

SUPPORTED_BACKTEST_SCHEMA_MAJOR = 1
BACKTEST_MIGRATION_MAP_SCHEMA_VERSION = "1.0"
BACKTEST_MIGRATION_MAP_CHECKSUM_ALGO = "sha256"

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


class BacktestMigrationFieldMapping(TypedDict):
    v1_path: str
    v2_path: str
    transform: str
    reversible: bool
    note: str


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


def backtest_migration_v1_to_v2_field_map() -> list[BacktestMigrationFieldMapping]:
    return [
        {
            "v1_path": "schema_version",
            "v2_path": "schema_version",
            "transform": "constant('2.0')",
            "reversible": False,
            "note": "升级版本号；原始值保存在 meta.source_schema_version",
        },
        {
            "v1_path": "schema_version",
            "v2_path": "meta.source_schema_version",
            "transform": "copy",
            "reversible": True,
            "note": "记录迁移前 schema",
        },
        {
            "v1_path": "generated_at",
            "v2_path": "summary.generated_at",
            "transform": "copy",
            "reversible": True,
            "note": "时间字段平移到 summary",
        },
        {
            "v1_path": "from_ts",
            "v2_path": "summary.from_ts",
            "transform": "copy",
            "reversible": True,
            "note": "时间窗口起点",
        },
        {
            "v1_path": "to_ts",
            "v2_path": "summary.to_ts",
            "transform": "copy",
            "reversible": True,
            "note": "时间窗口终点",
        },
        {
            "v1_path": "total_signals",
            "v2_path": "summary.total_signals",
            "transform": "copy",
            "reversible": True,
            "note": "汇总统计",
        },
        {
            "v1_path": "executed_signals",
            "v2_path": "summary.executed_signals",
            "transform": "copy",
            "reversible": True,
            "note": "汇总统计",
        },
        {
            "v1_path": "rejected_signals",
            "v2_path": "summary.rejected_signals",
            "transform": "copy",
            "reversible": True,
            "note": "汇总统计",
        },
        {
            "v1_path": "execution_config",
            "v2_path": "configs.execution",
            "transform": "deepcopy",
            "reversible": True,
            "note": "执行参数快照",
        },
        {
            "v1_path": "risk_config",
            "v2_path": "configs.risk",
            "transform": "deepcopy",
            "reversible": True,
            "note": "风控参数快照",
        },
        {
            "v1_path": "results",
            "v2_path": "attribution.strategy",
            "transform": "deepcopy",
            "reversible": True,
            "note": "策略归因",
        },
        {
            "v1_path": "event_results",
            "v2_path": "attribution.event",
            "transform": "deepcopy",
            "reversible": True,
            "note": "事件归因",
        },
        {
            "v1_path": "replay",
            "v2_path": "replay",
            "transform": "deepcopy",
            "reversible": True,
            "note": "回放明细",
        },
    ]


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def compute_backtest_migration_map_checksum(payload: Mapping[str, Any]) -> str:
    # checksum field itself is excluded from digest material
    normalized = dict(payload)
    normalized.pop("checksum_sha256", None)
    digest = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    return digest


def build_backtest_migration_map_artifact(*, with_checksum: bool = False) -> dict[str, Any]:
    mappings = backtest_migration_v1_to_v2_field_map()
    reversible_count = sum(1 for x in mappings if x["reversible"])

    artifact = {
        "schema_version": BACKTEST_MIGRATION_MAP_SCHEMA_VERSION,
        "kind": "backtest_migration_map",
        "checksum_algo": BACKTEST_MIGRATION_MAP_CHECKSUM_ALGO,
        "from_schema_major": 1,
        "to_schema_major": 2,
        "mappings": mappings,
        "summary": {
            "total_fields": len(mappings),
            "reversible_fields": reversible_count,
            "non_reversible_fields": len(mappings) - reversible_count,
        },
    }

    if with_checksum:
        artifact["checksum_sha256"] = compute_backtest_migration_map_checksum(artifact)

    return artifact


def verify_backtest_migration_map_checksum(payload: Mapping[str, Any]) -> bool:
    checksum = payload.get("checksum_sha256")
    if not isinstance(checksum, str) or not checksum:
        return False
    return checksum == compute_backtest_migration_map_checksum(payload)


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
