from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_REJECT_REASON_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("strategy notional limit exceeded:", "strategy notional limit exceeded"),
    ("circuit breaker open:", "circuit breaker open"),
)


def _safe_int(raw: object) -> int:
    try:
        return int(float(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def normalize_reject_reason(
    reason: str,
    *,
    rules: tuple[tuple[str, str], ...] = DEFAULT_REJECT_REASON_PREFIX_RULES,
) -> str:
    text = reason.strip()
    for prefix, normalized in rules:
        if text.startswith(prefix):
            return normalized
    return text


def aggregate_reject_reasons(
    raw_reasons: Mapping[str, Any],
    *,
    normalize: bool,
    rules: tuple[tuple[str, str], ...] = DEFAULT_REJECT_REASON_PREFIX_RULES,
) -> list[tuple[str, int]]:
    reason_items: list[tuple[str, int]] = []
    for key, value in raw_reasons.items():
        reason_raw = str(key).strip() or "unknown"
        reason = normalize_reject_reason(reason_raw, rules=rules) if normalize else reason_raw
        count = _safe_int(value)
        if count <= 0:
            continue
        reason_items.append((reason, count))

    if not reason_items:
        return []

    if normalize:
        merged: dict[str, int] = {}
        for reason, count in reason_items:
            merged[reason] = merged.get(reason, 0) + count
        reason_items = list(merged.items())

    reason_items.sort(key=lambda item: (-item[1], item[0]))
    return reason_items


def format_reject_top(
    raw_reasons: Mapping[str, Any],
    *,
    top_k: int,
    delimiter: str = ";",
    normalize: bool,
    rules: tuple[tuple[str, str], ...] = DEFAULT_REJECT_REASON_PREFIX_RULES,
) -> tuple[str, list[tuple[str, int]]]:
    if top_k <= 0:
        return "disabled", []

    reason_items = aggregate_reject_reasons(
        raw_reasons,
        normalize=normalize,
        rules=rules,
    )
    if not reason_items:
        return "none", []

    top_pairs = reason_items[:top_k]
    top_text = delimiter.join(f"{reason}:{count}" for reason, count in top_pairs)
    return top_text, top_pairs
