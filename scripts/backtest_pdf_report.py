#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_strategy_rows(payload: dict[str, Any], strategy_csv: Path | None) -> list[dict[str, Any]]:
    if strategy_csv and strategy_csv.exists():
        rows: list[dict[str, Any]] = []
        with strategy_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
        return rows

    raw_rows = payload.get("results") or []
    if not isinstance(raw_rows, list):
        return []
    return [dict(row) for row in raw_rows if isinstance(row, dict)]


def _load_event_rows(payload: dict[str, Any], event_csv: Path | None) -> list[dict[str, Any]]:
    if event_csv and event_csv.exists():
        rows: list[dict[str, Any]] = []
        with event_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
        return rows

    raw_rows = payload.get("event_results") or []
    if not isinstance(raw_rows, list):
        return []
    return [dict(row) for row in raw_rows if isinstance(row, dict)]


def _format_pct(raw: object) -> str:
    return f"{_safe_float(raw) * 100.0:.2f}%"


def render_pdf(
    *,
    payload: dict[str, Any],
    strategy_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - runtime dependency guidance
        raise SystemExit(
            "reportlab is required to render PDF. "
            "Try: uv run --with reportlab python scripts/backtest_pdf_report.py ..."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin_x = 40
    margin_y = 40
    y = height - margin_y

    def write_line(text: str, *, bold: bool = False, size: int = 10, leading: int = 14) -> None:
        nonlocal y
        if y < margin_y + leading:
            pdf.showPage()
            y = height - margin_y
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        pdf.drawString(margin_x, y, text)
        y -= leading

    def write_wrapped(
        text: str, *, bold: bool = False, size: int = 10, width_chars: int = 95
    ) -> None:
        chunks = textwrap.wrap(text, width=width_chars) or [""]
        for chunk in chunks:
            write_line(chunk, bold=bold, size=size)

    write_line(title, bold=True, size=16, leading=22)
    write_line(f"Generated at: {payload.get('generated_at', '')}")
    write_line(f"Schema version: {payload.get('schema_version', '')}")
    write_line("")

    write_line("Time Window", bold=True, size=12, leading=18)
    write_line(f"From: {payload.get('from_ts', '')}")
    write_line(f"To:   {payload.get('to_ts', '')}")
    write_line("")

    total_signals = _safe_int(payload.get("total_signals"))
    executed_signals = _safe_int(payload.get("executed_signals"))
    rejected_signals = _safe_int(payload.get("rejected_signals"))

    write_line("Signals Summary", bold=True, size=12, leading=18)
    write_line(f"Total signals:    {total_signals}")
    write_line(f"Executed signals: {executed_signals}")
    write_line(f"Rejected signals: {rejected_signals}")
    write_line("")

    write_line("Strategy Metrics", bold=True, size=12, leading=18)
    if not strategy_rows:
        write_line("(No strategy rows)")
    else:
        for row in strategy_rows:
            strategy = str(row.get("strategy", ""))
            pnl = _safe_float(row.get("pnl"))
            winrate = _format_pct(row.get("winrate"))
            max_dd = _safe_float(row.get("max_drawdown"))
            trades = _safe_int(row.get("trade_count"))
            wins = _safe_int(row.get("wins"))
            losses = _safe_int(row.get("losses"))
            write_wrapped(
                " - "
                + f"{strategy}: pnl={pnl:.4f}, winrate={winrate}, "
                + f"max_drawdown={max_dd:.4f}, trades={trades}, wins={wins}, losses={losses}"
            )

    write_line("")
    write_line("Top Events by |PnL|", bold=True, size=12, leading=18)
    if not event_rows:
        write_line("(No event rows)")
    else:
        sorted_events = sorted(
            event_rows,
            key=lambda row: abs(_safe_float(row.get("pnl"))),
            reverse=True,
        )[:10]
        for row in sorted_events:
            strategy = str(row.get("strategy", ""))
            event_id = str(row.get("event_id", ""))
            pnl = _safe_float(row.get("pnl"))
            trades = _safe_int(row.get("trade_count"))
            write_wrapped(
                " - "
                + f"{strategy}/{event_id}: pnl={pnl:.4f}, "
                + f"winrate={_format_pct(row.get('winrate'))}, trades={trades}"
            )

    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render monomarket backtest PDF report")
    parser.add_argument("--backtest-json", required=True, help="Path to backtest JSON artifact")
    parser.add_argument("--strategy-csv", help="Optional strategy attribution CSV")
    parser.add_argument("--event-csv", help="Optional event attribution CSV")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--title", default="Monomarket Backtest Report", help="PDF title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtest_json = Path(args.backtest_json)
    strategy_csv = Path(args.strategy_csv) if args.strategy_csv else None
    event_csv = Path(args.event_csv) if args.event_csv else None
    output = Path(args.output)

    payload = _load_payload(backtest_json)
    strategy_rows = _load_strategy_rows(payload, strategy_csv)
    event_rows = _load_event_rows(payload, event_csv)

    render_pdf(
        payload=payload,
        strategy_rows=strategy_rows,
        event_rows=event_rows,
        output_path=output,
        title=args.title,
    )
    print(f"[backtest-pdf] report generated: {output}")


if __name__ == "__main__":
    main()
