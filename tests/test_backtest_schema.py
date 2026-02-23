from __future__ import annotations

import pytest

from monomarket.backtest import (
    REQUIRED_BACKTEST_JSON_FIELDS_V1,
    SUPPORTED_BACKTEST_SCHEMA_MAJOR,
    assert_schema_compatible,
    is_schema_compatible,
    parse_schema_version,
    validate_backtest_json_artifact,
)


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
