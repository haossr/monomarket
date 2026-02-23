from __future__ import annotations

import pytest

from monomarket.backtest import (
    SUPPORTED_BACKTEST_SCHEMA_MAJOR,
    assert_schema_compatible,
    is_schema_compatible,
    parse_schema_version,
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
