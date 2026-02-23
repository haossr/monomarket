from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from monomarket.backtest import (
    REQUIRED_BACKTEST_JSON_FIELDS_V1,
    REQUIRED_BACKTEST_JSON_FIELDS_V2,
    SUPPORTED_BACKTEST_SCHEMA_MAJOR,
    assert_schema_compatible,
    is_schema_compatible,
    parse_schema_version,
    validate_backtest_json_artifact,
    validate_backtest_json_artifact_v1,
    validate_backtest_json_artifact_v2,
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


def test_validate_backtest_json_artifact_v1_ok() -> None:
    payload = {
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
    major, minor = validate_backtest_json_artifact(payload)
    assert (major, minor) == (1, 0)


def test_validate_backtest_json_artifact_missing_required_field() -> None:
    payload = {
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
        )


def test_validate_backtest_json_artifact_v1_helper_reuse() -> None:
    payload = {
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
    validate_backtest_json_artifact_v1(payload)


def test_validate_backtest_json_artifact_v2_helper_reuse() -> None:
    payload = {
        "schema_version": "2.0",
        "meta": {},
        "results": [],
    }
    validate_backtest_json_artifact_v2(payload)


def test_validate_backtest_json_artifact_v2_missing_required() -> None:
    payload = {k: {} for k in REQUIRED_BACKTEST_JSON_FIELDS_V2}
    payload["schema_version"] = "2.0"
    payload["results"] = []
    payload.pop("meta")

    with pytest.raises(ValueError):
        validate_backtest_json_artifact_v2(payload)


def test_validate_backtest_json_artifact_fixture_samples() -> None:
    payload_v1 = json.loads((FIXTURE_DIR / "artifact_v1.json").read_text())
    payload_v2 = json.loads((FIXTURE_DIR / "artifact_v2.json").read_text())

    assert validate_backtest_json_artifact(payload_v1) == (1, 0)
    assert validate_backtest_json_artifact(payload_v2, supported_major=None) == (2, 0)
