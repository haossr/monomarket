from __future__ import annotations

SUPPORTED_BACKTEST_SCHEMA_MAJOR = 1


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
