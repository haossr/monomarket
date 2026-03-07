from __future__ import annotations

import inspect

from monomarket import cli

_EXPECTED_DEFAULT = "s1,s2,s4,s8,s9,s10"


def _strategies_default(fn: object) -> str:
    sig = inspect.signature(fn)
    default = sig.parameters["strategies"].default
    value = getattr(default, "default", None)
    assert isinstance(value, str)
    return value


def test_generate_signals_default_includes_s9_s10() -> None:
    assert _strategies_default(cli.generate_signals) == _EXPECTED_DEFAULT


def test_backtest_default_includes_s9_s10() -> None:
    assert _strategies_default(cli.backtest) == _EXPECTED_DEFAULT


def test_backtest_rolling_default_includes_s9_s10() -> None:
    assert _strategies_default(cli.backtest_rolling) == _EXPECTED_DEFAULT
