from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from monomarket.backtest import (
    BACKTEST_ARTIFACT_CHECKSUM_ALGO,
    BACKTEST_MIGRATION_MAP_CHECKSUM_ALGO,
    BACKTEST_MIGRATION_MAP_SCHEMA_VERSION,
    NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO,
    NIGHTLY_SUMMARY_SIDECAR_SCHEMA_VERSION,
    REQUIRED_BACKTEST_JSON_FIELDS_V1,
    REQUIRED_BACKTEST_JSON_FIELDS_V2,
    REQUIRED_NIGHTLY_SUMMARY_SIDECAR_FIELDS,
    SUPPORTED_BACKTEST_SCHEMA_MAJOR,
    assert_schema_compatible,
    backtest_migration_v1_to_v2_field_map,
    build_backtest_migration_map_artifact,
    compute_backtest_json_artifact_checksum,
    compute_backtest_migration_map_checksum,
    compute_nightly_summary_sidecar_checksum,
    is_schema_compatible,
    migrate_backtest_artifact_v1_to_v2,
    parse_schema_version,
    validate_backtest_json_artifact,
    validate_backtest_json_artifact_v1,
    validate_backtest_json_artifact_v2,
    validate_nightly_summary_sidecar,
    verify_backtest_json_artifact_checksum,
    verify_backtest_migration_map_checksum,
    verify_nightly_summary_sidecar_checksum,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "backtest"


def test_parse_schema_version() -> None:
    assert parse_schema_version("1.0") == (1, 0)
    assert parse_schema_version("12.34") == (12, 34)


def test_is_schema_compatible() -> None:
    assert is_schema_compatible("1.0")
    assert is_schema_compatible("1.9")
    assert not is_schema_compatible("2.0")


def test_assert_schema_compatible() -> None:
    major, minor = assert_schema_compatible("1.2")
    assert major == SUPPORTED_BACKTEST_SCHEMA_MAJOR
    assert minor == 2


@pytest.mark.parametrize("raw", ["1", "v1.0", "1.a", "", "1.0.0"])
def test_parse_schema_version_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_schema_version(raw)


def test_assert_schema_compatible_reject_new_major() -> None:
    with pytest.raises(ValueError):
        assert_schema_compatible("2.0")


def _v1_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-02-20T00:00:00+00:00",
        "from_ts": "2026-02-20T00:00:00+00:00",
        "to_ts": "2026-02-20T01:00:00+00:00",
        "total_signals": 1,
        "executed_signals": 1,
        "rejected_signals": 0,
        "execution_config": {},
        "risk_config": {},
        "results": [],
        "event_results": [],
        "replay": [],
    }


def test_validate_backtest_json_artifact_v1_ok() -> None:
    major, minor = validate_backtest_json_artifact(_v1_payload())
    assert (major, minor) == (1, 0)


def test_validate_backtest_json_artifact_missing_required_field() -> None:
    payload: dict[str, object] = {
        k: [] if k in {"results", "event_results", "replay"} else {}
        for k in REQUIRED_BACKTEST_JSON_FIELDS_V1
    }
    payload["schema_version"] = "1.0"
    payload["generated_at"] = "2026-02-20T00:00:00+00:00"
    payload["from_ts"] = "2026-02-20T00:00:00+00:00"
    payload["to_ts"] = "2026-02-20T01:00:00+00:00"
    payload["total_signals"] = 1
    payload["executed_signals"] = 1
    payload["rejected_signals"] = 0
    payload.pop("risk_config")

    with pytest.raises(ValueError):
        validate_backtest_json_artifact(payload)


def test_validate_backtest_json_artifact_reject_incompatible_major() -> None:
    with pytest.raises(ValueError):
        validate_backtest_json_artifact({"schema_version": "2.0"})


def test_validate_backtest_json_artifact_dual_stack_dispatch() -> None:
    called = {"v2": False}

    def _validate_v2(payload: Mapping[str, object]) -> None:
        called["v2"] = True
        assert payload["schema_version"] == "2.1"
        assert payload["meta"] == "ok"

    major, minor = validate_backtest_json_artifact(
        {"schema_version": "2.1", "meta": "ok"},
        supported_major=None,
        validators={2: _validate_v2},
    )
    assert (major, minor) == (2, 1)
    assert called["v2"] is True


def test_validate_backtest_json_artifact_unknown_major_without_validator() -> None:
    with pytest.raises(ValueError):
        validate_backtest_json_artifact(
            {"schema_version": "3.0"},
            supported_major=None,
            validators={1: validate_backtest_json_artifact_v1},
        )


def test_validate_backtest_json_artifact_v1_helper_reuse() -> None:
    validate_backtest_json_artifact_v1(_v1_payload())


def test_validate_backtest_json_artifact_v2_helper_reuse() -> None:
    payload = {
        "schema_version": "2.0",
        "meta": {},
        "summary": {},
        "configs": {},
        "attribution": {},
        "replay": [],
    }
    validate_backtest_json_artifact_v2(payload)


def test_validate_backtest_json_artifact_v2_missing_required() -> None:
    payload: dict[str, object] = {k: {} for k in REQUIRED_BACKTEST_JSON_FIELDS_V2}
    payload["schema_version"] = "2.0"
    payload["replay"] = []
    payload.pop("meta")

    with pytest.raises(ValueError):
        validate_backtest_json_artifact_v2(payload)


def test_migrate_backtest_artifact_v1_to_v2() -> None:
    migrated = migrate_backtest_artifact_v1_to_v2(_v1_payload())

    assert migrated["schema_version"] == "2.0"
    assert migrated["meta"]["migration"] == "v1_to_v2"
    assert migrated["summary"]["total_signals"] == 1
    assert migrated["configs"]["execution"] == {}
    assert migrated["attribution"]["strategy"] == []

    assert validate_backtest_json_artifact(migrated, supported_major=None) == (2, 0)


def test_backtest_migration_v1_to_v2_field_map_contract() -> None:
    rows = backtest_migration_v1_to_v2_field_map()
    assert rows

    keyset = {"v1_path", "v2_path", "transform", "reversible", "note"}
    for row in rows:
        assert set(row.keys()) == keyset

    assert any(row["v1_path"] == "schema_version" for row in rows)
    assert any(row["reversible"] is False for row in rows)


def test_build_backtest_migration_map_artifact() -> None:
    artifact = build_backtest_migration_map_artifact(with_checksum=True)

    assert artifact["schema_version"] == BACKTEST_MIGRATION_MAP_SCHEMA_VERSION
    assert artifact["kind"] == "backtest_migration_map"
    assert artifact["checksum_algo"] == BACKTEST_MIGRATION_MAP_CHECKSUM_ALGO
    assert artifact["from_schema_major"] == 1
    assert artifact["to_schema_major"] == 2
    assert isinstance(artifact["mappings"], list)
    assert artifact["summary"]["total_fields"] == len(artifact["mappings"])
    assert isinstance(artifact["checksum_sha256"], str)
    assert len(artifact["checksum_sha256"]) == 64

    assert verify_backtest_migration_map_checksum(artifact)
    assert artifact["checksum_sha256"] == compute_backtest_migration_map_checksum(artifact)


def test_verify_backtest_migration_map_checksum_tampered() -> None:
    artifact = build_backtest_migration_map_artifact(with_checksum=True)
    artifact["summary"]["total_fields"] = int(artifact["summary"]["total_fields"]) + 1
    assert not verify_backtest_migration_map_checksum(artifact)


def test_verify_backtest_json_artifact_checksum() -> None:
    payload = _v1_payload()
    payload["checksum_algo"] = BACKTEST_ARTIFACT_CHECKSUM_ALGO
    payload["checksum_sha256"] = compute_backtest_json_artifact_checksum(payload)

    assert verify_backtest_json_artifact_checksum(payload)
    assert payload["checksum_sha256"] == compute_backtest_json_artifact_checksum(payload)


def test_verify_backtest_json_artifact_checksum_tampered() -> None:
    payload = _v1_payload()
    payload["checksum_algo"] = BACKTEST_ARTIFACT_CHECKSUM_ALGO
    payload["checksum_sha256"] = compute_backtest_json_artifact_checksum(payload)
    payload["total_signals"] = int(payload["total_signals"]) + 1

    assert not verify_backtest_json_artifact_checksum(payload)


def _nightly_sidecar_payload() -> dict[str, object]:
    return {
        "schema_version": NIGHTLY_SUMMARY_SIDECAR_SCHEMA_VERSION,
        "nightly_date": "2026-02-24",
        "window": {
            "from_ts": "2026-02-24T00:00:00Z",
            "to_ts": "2026-02-24T02:00:00Z",
        },
        "signals": {
            "total": 10,
            "executed": 8,
            "rejected": 2,
        },
        "best": {
            "available": True,
            "strategy": "s1",
            "pnl": 1.2,
            "text": "best_strategy=s1 pnl=1.2000",
        },
        "best_text": "best_strategy=s1 pnl=1.2000",
        "rolling": {
            "runs": 3,
            "execution_rate": 0.8,
            "positive_window_rate": 0.66,
            "empty_window_count": 1,
            "range_hours": 24.0,
            "coverage_ratio": 1.0,
            "overlap_ratio": 0.2,
            "coverage_label": "full",
            "reject_top_k": 2,
            "reject_top_delimiter": ";",
            "reject_top": "riskA:3;riskB:1",
            "reject_top_pairs": [
                {"reason": "riskA", "count": 3},
                {"reason": "riskB", "count": 1},
            ],
        },
        "paths": {
            "pdf": "/tmp/report.pdf",
            "rolling_json": "/tmp/rolling-summary.json",
        },
    }


def test_validate_nightly_summary_sidecar_ok() -> None:
    validate_nightly_summary_sidecar(_nightly_sidecar_payload())


def test_validate_nightly_summary_sidecar_legacy_best_string_ok() -> None:
    payload = _nightly_sidecar_payload()
    payload["best"] = "best_strategy=s1 pnl=1.2000"
    payload.pop("best_text", None)
    validate_nightly_summary_sidecar(payload)


def test_verify_nightly_summary_sidecar_checksum() -> None:
    payload = _nightly_sidecar_payload()
    payload["checksum_algo"] = NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
    payload["checksum_sha256"] = compute_nightly_summary_sidecar_checksum(payload)

    assert verify_nightly_summary_sidecar_checksum(payload)
    validate_nightly_summary_sidecar(payload)


def test_verify_nightly_summary_sidecar_checksum_tampered() -> None:
    payload = _nightly_sidecar_payload()
    payload["checksum_algo"] = NIGHTLY_SUMMARY_SIDECAR_CHECKSUM_ALGO
    payload["checksum_sha256"] = compute_nightly_summary_sidecar_checksum(payload)
    rolling = payload["rolling"]
    assert isinstance(rolling, dict)
    rolling["runs"] = int(rolling["runs"]) + 1

    assert not verify_nightly_summary_sidecar_checksum(payload)
    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_reject_partial_checksum_fields() -> None:
    payload = _nightly_sidecar_payload()
    payload["checksum_sha256"] = "abc"

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_reject_wrong_checksum_algo() -> None:
    payload = _nightly_sidecar_payload()
    payload["checksum_algo"] = "sha1"
    payload["checksum_sha256"] = "abc"

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_missing_required() -> None:
    payload = _nightly_sidecar_payload()
    payload.pop("rolling")

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_reject_wrong_schema() -> None:
    payload = _nightly_sidecar_payload()
    payload["schema_version"] = "nightly-summary-sidecar-2.0"

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_reject_bad_pairs() -> None:
    payload = _nightly_sidecar_payload()
    rolling = payload["rolling"]
    assert isinstance(rolling, dict)
    rolling["reject_top_pairs"] = [{"reason": "riskA", "count": "x"}]

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_validate_nightly_summary_sidecar_reject_bad_delimiter_type() -> None:
    payload = _nightly_sidecar_payload()
    rolling = payload["rolling"]
    assert isinstance(rolling, dict)
    rolling["reject_top_delimiter"] = 1

    with pytest.raises(ValueError):
        validate_nightly_summary_sidecar(payload)


def test_nightly_sidecar_required_fields_contract() -> None:
    assert REQUIRED_NIGHTLY_SUMMARY_SIDECAR_FIELDS == {
        "schema_version",
        "nightly_date",
        "window",
        "signals",
        "best",
        "rolling",
        "paths",
    }


def test_validate_backtest_json_artifact_fixture_samples() -> None:
    payload_v1 = json.loads((FIXTURE_DIR / "artifact_v1.json").read_text())
    payload_v2 = json.loads((FIXTURE_DIR / "artifact_v2.json").read_text())

    assert validate_backtest_json_artifact(payload_v1) == (1, 0)
    assert validate_backtest_json_artifact(payload_v2, supported_major=None) == (2, 0)
