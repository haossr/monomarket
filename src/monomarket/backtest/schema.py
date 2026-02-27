from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypedDict

SUPPORTED_BACKTEST_SCHEMA_MAJOR = 1
BACKTEST_ARTIFACT_CHECKSUM_ALGO = "sha256"
BACKTEST_MIGRATION_MAP_SCHEMA_VERSION = "1.0"
BACKTEST_MIGRATION_MAP_CHECKSUM_ALGO = "sha256"
NIGHTLY_SUMMARY_SIDECAR_SCHEMA_VERSION = "nightly-summary-sidecar-1.0"
NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO = "sha256"

REQUIRED_NIGHTLY_SUMMARY_SIDECAR_FIELDS = {
    "schema_version",
    "nightly_date",
    "window",
    "signals",
    "best",
    "rolling",
    "paths",
}

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


def compute_backtest_json_artifact_checksum(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("checksum_sha256", None)
    digest = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    return digest


def verify_backtest_json_artifact_checksum(payload: Mapping[str, Any]) -> bool:
    checksum = payload.get("checksum_sha256")
    if not isinstance(checksum, str) or not checksum:
        return False

    algo = payload.get("checksum_algo")
    if algo is not None and algo != BACKTEST_ARTIFACT_CHECKSUM_ALGO:
        return False

    return checksum == compute_backtest_json_artifact_checksum(payload)


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


def compute_nightly_summary_sidecar_checksum(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("checksum_sha256", None)
    digest = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    return digest


def verify_nightly_summary_sidecar_checksum(payload: Mapping[str, Any]) -> bool:
    checksum = payload.get("checksum_sha256")
    if not isinstance(checksum, str) or not checksum:
        return False

    algo = payload.get("checksum_algo")
    if algo is not None and algo != NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO:
        return False

    return checksum == compute_nightly_summary_sidecar_checksum(payload)


def validate_nightly_summary_sidecar(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_NIGHTLY_SUMMARY_SIDECAR_FIELDS - set(payload.keys()))
    if missing:
        raise ValueError(f"missing required nightly sidecar fields: {', '.join(missing)}")

    schema_version = payload.get("schema_version")
    if schema_version != NIGHTLY_SUMMARY_SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            "unsupported nightly sidecar schema_version="
            f"{schema_version!r}; expected {NIGHTLY_SUMMARY_SIDECAR_SCHEMA_VERSION!r}"
        )

    has_checksum = "checksum_sha256" in payload or "checksum_algo" in payload
    if has_checksum:
        if "checksum_sha256" not in payload or "checksum_algo" not in payload:
            raise ValueError(
                "nightly sidecar checksum requires both checksum_algo and checksum_sha256"
            )
        algo = payload.get("checksum_algo")
        if algo != NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO:
            raise ValueError(
                "unsupported nightly sidecar checksum_algo="
                f"{algo!r}; expected {NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO!r}"
            )
        if not verify_nightly_summary_sidecar_checksum(payload):
            raise ValueError("nightly sidecar checksum verification failed")

    schema_note_version = payload.get("schema_note_version")
    if schema_note_version is not None and not isinstance(schema_note_version, str):
        raise ValueError("nightly sidecar schema_note_version must be a string")

    schema_note = payload.get("schema_note")
    if schema_note is not None and not isinstance(schema_note, str):
        raise ValueError("nightly sidecar schema_note must be a string")

    best_version = payload.get("best_version")
    if best_version is not None and not isinstance(best_version, str):
        raise ValueError("nightly sidecar best_version must be a string")

    nightly_date = payload.get("nightly_date")
    if not isinstance(nightly_date, str) or not nightly_date.strip():
        raise ValueError("nightly sidecar nightly_date must be a non-empty string")

    window = payload.get("window")
    if not isinstance(window, Mapping):
        raise ValueError("nightly sidecar window must be an object")
    for key in ("from_ts", "to_ts"):
        raw = window.get(key)
        if not isinstance(raw, str):
            raise ValueError(f"nightly sidecar window.{key} must be a string")

    signals = payload.get("signals")
    if not isinstance(signals, Mapping):
        raise ValueError("nightly sidecar signals must be an object")
    for key in ("total", "executed", "rejected"):
        raw = signals.get(key)
        if not isinstance(raw, int | float):
            raise ValueError(f"nightly sidecar signals.{key} must be numeric")

    winrate = payload.get("winrate")
    if winrate is not None:
        if not isinstance(winrate, Mapping):
            raise ValueError("nightly sidecar winrate must be an object")
        for key in (
            "closed_winrate",
            "closed_sample_count",
            "closed_wins",
            "closed_losses",
            "mtm_winrate",
            "mtm_sample_count",
            "mtm_wins",
            "mtm_losses",
        ):
            raw = winrate.get(key)
            if not isinstance(raw, int | float):
                raise ValueError(f"nightly sidecar winrate.{key} must be numeric")

    window_coverage = payload.get("window_coverage")
    if window_coverage is not None:
        if not isinstance(window_coverage, Mapping):
            raise ValueError("nightly sidecar window_coverage must be an object")
        for key in ("window_hours", "covered_hours", "coverage_ratio"):
            raw = window_coverage.get(key)
            if not isinstance(raw, int | float):
                raise ValueError(f"nightly sidecar window_coverage.{key} must be numeric")
        history_limited = window_coverage.get("history_limited")
        if not isinstance(history_limited, bool):
            raise ValueError("nightly sidecar window_coverage.history_limited must be a bool")
        for key in ("note", "first_replay_ts", "effective_from_ts"):
            raw = window_coverage.get(key)
            if not isinstance(raw, str):
                raise ValueError(f"nightly sidecar window_coverage.{key} must be a string")

    best = payload.get("best")
    if isinstance(best, Mapping):
        best_available = best.get("available")
        if not isinstance(best_available, bool):
            raise ValueError("nightly sidecar best.available must be a bool")
        best_strategy = best.get("strategy")
        if not isinstance(best_strategy, str):
            raise ValueError("nightly sidecar best.strategy must be a string")
        best_pnl = best.get("pnl")
        if not isinstance(best_pnl, int | float):
            raise ValueError("nightly sidecar best.pnl must be numeric")
        best_text = best.get("text")
        if not isinstance(best_text, str):
            raise ValueError("nightly sidecar best.text must be a string")
    elif isinstance(best, str):
        # Backward compatibility for legacy sidecars where best was plain text.
        pass
    else:
        raise ValueError("nightly sidecar best must be an object or string")

    best_text_top = payload.get("best_text")
    if best_text_top is not None and not isinstance(best_text_top, str):
        raise ValueError("nightly sidecar best_text must be a string")

    negative_strategies = payload.get("negative_strategies")
    if not isinstance(negative_strategies, Mapping):
        raise ValueError("nightly sidecar negative_strategies must be an object")

    negative_count = negative_strategies.get("count")
    if not isinstance(negative_count, int):
        raise ValueError("nightly sidecar negative_strategies.count must be an integer")
    if negative_count < 0:
        raise ValueError("nightly sidecar negative_strategies.count must be >= 0")

    worst_strategy = negative_strategies.get("worst_strategy")
    if not isinstance(worst_strategy, str):
        raise ValueError("nightly sidecar negative_strategies.worst_strategy must be a string")

    worst_pnl = negative_strategies.get("worst_pnl")
    if not isinstance(worst_pnl, int | float):
        raise ValueError("nightly sidecar negative_strategies.worst_pnl must be numeric")

    negative_text = negative_strategies.get("text")
    if not isinstance(negative_text, str):
        raise ValueError("nightly sidecar negative_strategies.text must be a string")

    if negative_count == 0:
        if worst_strategy not in {"", "n/a"}:
            raise ValueError(
                "nightly sidecar negative_strategies.worst_strategy must be empty/n-a when count=0"
            )
        if float(worst_pnl) != 0.0:
            raise ValueError("nightly sidecar negative_strategies.worst_pnl must be 0 when count=0")
    else:
        if not worst_strategy:
            raise ValueError(
                "nightly sidecar negative_strategies.worst_strategy must be non-empty when count>0"
            )
        if float(worst_pnl) >= 0.0:
            raise ValueError(
                "nightly sidecar negative_strategies.worst_pnl must be negative when count>0"
            )

    rolling = payload.get("rolling")
    if not isinstance(rolling, Mapping):
        raise ValueError("nightly sidecar rolling must be an object")

    rolling_required_numeric = (
        "runs",
        "execution_rate",
        "positive_window_rate",
        "empty_window_count",
        "range_hours",
        "coverage_ratio",
        "overlap_ratio",
        "reject_top_k",
    )
    for key in rolling_required_numeric:
        raw = rolling.get(key)
        if not isinstance(raw, int | float):
            raise ValueError(f"nightly sidecar rolling.{key} must be numeric")

    for key in ("coverage_label", "reject_top", "reject_top_effective"):
        raw = rolling.get(key)
        if not isinstance(raw, str):
            raise ValueError(f"nightly sidecar rolling.{key} must be a string")

    reject_top_delimiter = rolling.get("reject_top_delimiter")
    if reject_top_delimiter is not None and not isinstance(reject_top_delimiter, str):
        raise ValueError("nightly sidecar rolling.reject_top_delimiter must be a string")

    reject_pairs = rolling.get("reject_top_pairs")
    if not isinstance(reject_pairs, list):
        raise ValueError("nightly sidecar rolling.reject_top_pairs must be an array")
    for idx, row in enumerate(reject_pairs):
        if not isinstance(row, Mapping):
            raise ValueError(f"nightly sidecar rolling.reject_top_pairs[{idx}] must be an object")
        reason = row.get("reason")
        count = row.get("count")
        if not isinstance(reason, str):
            raise ValueError(
                f"nightly sidecar rolling.reject_top_pairs[{idx}].reason must be a string"
            )
        if not isinstance(count, int | float):
            raise ValueError(
                f"nightly sidecar rolling.reject_top_pairs[{idx}].count must be numeric"
            )

    reject_top_normalized = rolling.get("reject_top_normalized")
    if reject_top_normalized is not None and not isinstance(reject_top_normalized, str):
        raise ValueError("nightly sidecar rolling.reject_top_normalized must be a string")

    reject_pairs_normalized = rolling.get("reject_top_pairs_normalized")
    if reject_pairs_normalized is not None:
        if not isinstance(reject_pairs_normalized, list):
            raise ValueError("nightly sidecar rolling.reject_top_pairs_normalized must be an array")
        for idx, row in enumerate(reject_pairs_normalized):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"nightly sidecar rolling.reject_top_pairs_normalized[{idx}] must be an object"
                )
            reason = row.get("reason")
            count = row.get("count")
            if not isinstance(reason, str):
                raise ValueError(
                    "nightly sidecar rolling.reject_top_pairs_normalized"
                    f"[{idx}].reason must be a string"
                )
            if not isinstance(count, int | float):
                raise ValueError(
                    "nightly sidecar rolling.reject_top_pairs_normalized"
                    f"[{idx}].count must be numeric"
                )

    reject_top_raw = str(rolling.get("reject_top", ""))
    reject_top_norm = (
        str(reject_top_normalized) if isinstance(reject_top_normalized, str) else reject_top_raw
    )
    reject_top_effective = str(rolling.get("reject_top_effective", ""))
    expected_reject_top_effective = (
        reject_top_norm if reject_top_norm not in {"none", "disabled"} else reject_top_raw
    )
    if reject_top_effective != expected_reject_top_effective:
        raise ValueError(
            "nightly sidecar rolling.reject_top_effective must match normalized fallback logic"
        )

    rolling_negative = rolling.get("negative_strategies")
    if not isinstance(rolling_negative, Mapping):
        raise ValueError("nightly sidecar rolling.negative_strategies must be an object")

    rolling_negative_count = rolling_negative.get("count")
    if not isinstance(rolling_negative_count, int):
        raise ValueError("nightly sidecar rolling.negative_strategies.count must be an integer")
    if rolling_negative_count < 0:
        raise ValueError("nightly sidecar rolling.negative_strategies.count must be >= 0")

    rolling_worst_strategy = rolling_negative.get("worst_strategy")
    if not isinstance(rolling_worst_strategy, str):
        raise ValueError(
            "nightly sidecar rolling.negative_strategies.worst_strategy must be a string"
        )

    rolling_worst_avg_pnl = rolling_negative.get("worst_avg_pnl")
    if not isinstance(rolling_worst_avg_pnl, int | float):
        raise ValueError(
            "nightly sidecar rolling.negative_strategies.worst_avg_pnl must be numeric"
        )

    rolling_negative_text = rolling_negative.get("text")
    if not isinstance(rolling_negative_text, str):
        raise ValueError("nightly sidecar rolling.negative_strategies.text must be a string")

    if rolling_negative_count == 0:
        if rolling_worst_strategy not in {"", "n/a"}:
            raise ValueError(
                "nightly sidecar rolling.negative_strategies.worst_strategy"
                " must be empty/n-a when count=0"
            )
        if float(rolling_worst_avg_pnl) != 0.0:
            raise ValueError(
                "nightly sidecar rolling.negative_strategies.worst_avg_pnl"
                " must be 0 when count=0"
            )
    else:
        if not rolling_worst_strategy:
            raise ValueError(
                "nightly sidecar rolling.negative_strategies.worst_strategy"
                " must be non-empty when count>0"
            )
        if float(rolling_worst_avg_pnl) >= 0.0:
            raise ValueError(
                "nightly sidecar rolling.negative_strategies.worst_avg_pnl"
                " must be negative when count>0"
            )

    reject_by_strategy = payload.get("reject_by_strategy")
    if reject_by_strategy is not None:
        if not isinstance(reject_by_strategy, Mapping):
            raise ValueError("nightly sidecar reject_by_strategy must be an object")

        top_k = reject_by_strategy.get("top_k")
        if not isinstance(top_k, int | float):
            raise ValueError("nightly sidecar reject_by_strategy.top_k must be numeric")

        delimiter = reject_by_strategy.get("delimiter")
        if not isinstance(delimiter, str):
            raise ValueError("nightly sidecar reject_by_strategy.delimiter must be a string")

        top = reject_by_strategy.get("top")
        if not isinstance(top, str):
            raise ValueError("nightly sidecar reject_by_strategy.top must be a string")

        rows = reject_by_strategy.get("rows")
        if not isinstance(rows, list):
            raise ValueError("nightly sidecar reject_by_strategy.rows must be an array")

        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}] must be an object"
                )
            strategy = row.get("strategy")
            total = row.get("total")
            rejected = row.get("rejected")
            reject_rate = row.get("reject_rate")
            top_reason = row.get("top_reason")
            if not isinstance(strategy, str):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}].strategy must be a string"
                )
            if not isinstance(total, int | float):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}].total must be numeric"
                )
            if not isinstance(rejected, int | float):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}].rejected must be numeric"
                )
            if not isinstance(reject_rate, int | float):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}].reject_rate must be numeric"
                )
            if not isinstance(top_reason, str):
                raise ValueError(
                    f"nightly sidecar reject_by_strategy.rows[{idx}].top_reason must be a string"
                )

    cycle_meta = payload.get("cycle_meta")
    if cycle_meta is not None:
        if not isinstance(cycle_meta, Mapping):
            raise ValueError("nightly sidecar cycle_meta must be an object")

        fixed_window_mode = cycle_meta.get("fixed_window_mode")
        if not isinstance(fixed_window_mode, bool):
            raise ValueError("nightly sidecar cycle_meta.fixed_window_mode must be a boolean")

        signal_generation = cycle_meta.get("signal_generation")
        if not isinstance(signal_generation, Mapping):
            raise ValueError("nightly sidecar cycle_meta.signal_generation must be an object")

        new_total = signal_generation.get("new_signals_total")
        if not isinstance(new_total, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.new_signals_total must be numeric"
            )
        new_in_window = signal_generation.get("new_signals_in_window")
        if not isinstance(new_in_window, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.new_signals_in_window must be numeric"
            )
        new_first_ts = signal_generation.get("new_signals_first_ts")
        if not isinstance(new_first_ts, str):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.new_signals_first_ts must be a string"
            )
        new_last_ts = signal_generation.get("new_signals_last_ts")
        if not isinstance(new_last_ts, str):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.new_signals_last_ts must be a string"
            )

        clear_signals_window = signal_generation.get("clear_signals_window")
        if not isinstance(clear_signals_window, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.clear_signals_window must be a boolean"
            )

        cleared_signals_in_window = signal_generation.get("cleared_signals_in_window")
        if not isinstance(cleared_signals_in_window, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.cleared_signals_in_window must be numeric"
            )

        rebuild_signals_window = signal_generation.get("rebuild_signals_window")
        if not isinstance(rebuild_signals_window, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.rebuild_signals_window must be a boolean"
            )

        rebuild_step_hours = signal_generation.get("rebuild_step_hours")
        if not isinstance(rebuild_step_hours, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.rebuild_step_hours must be numeric"
            )

        rebuild_sampled_steps = signal_generation.get("rebuild_sampled_steps")
        if not isinstance(rebuild_sampled_steps, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.rebuild_sampled_steps must be numeric"
            )

        generated_share_of_total = signal_generation.get("generated_share_of_total")
        if not isinstance(generated_share_of_total, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_share_of_total must be numeric"
            )

        generated_span_hours = signal_generation.get("generated_span_hours")
        if not isinstance(generated_span_hours, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_span_hours must be numeric"
            )

        generated_window_coverage_ratio = signal_generation.get("generated_window_coverage_ratio")
        if not isinstance(generated_window_coverage_ratio, int | float):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_window_coverage_ratio must be numeric"
            )

        generated_low_influence = signal_generation.get("generated_low_influence")
        if not isinstance(generated_low_influence, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_low_influence must be a boolean"
            )

        generated_low_sample_count = signal_generation.get("generated_low_sample_count")
        if not isinstance(generated_low_sample_count, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_low_sample_count must be a boolean"
            )

        generated_low_temporal_coverage = signal_generation.get("generated_low_temporal_coverage")
        if not isinstance(generated_low_temporal_coverage, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.generated_low_temporal_coverage must be a boolean"
            )

        historical_replay_only = signal_generation.get("historical_replay_only")
        if not isinstance(historical_replay_only, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.historical_replay_only must be a boolean"
            )

        experiment_interpretable = signal_generation.get("experiment_interpretable")
        if not isinstance(experiment_interpretable, bool):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.experiment_interpretable must be a boolean"
            )

        experiment_reason = signal_generation.get("experiment_reason")
        if not isinstance(experiment_reason, str):
            raise ValueError(
                "nightly sidecar cycle_meta.signal_generation.experiment_reason must be a string"
            )

    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("nightly sidecar paths must be an object")
    for key in ("pdf", "rolling_json"):
        raw = paths.get(key)
        if not isinstance(raw, str):
            raise ValueError(f"nightly sidecar paths.{key} must be a string")


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
