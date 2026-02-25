#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monomarket.backtest.reject_reason import format_reject_top

PDF_ROLLING_REJECT_TOP_K = 2


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


def _nice_step(span: float, target_ticks: int = 5) -> float:
    if span <= 0:
        return 1.0
    rough = span / max(1, target_ticks)
    magnitude = 10 ** math.floor(math.log10(abs(rough)))
    normalized = rough / magnitude
    if normalized <= 1:
        nice = 1.0
    elif normalized <= 2:
        nice = 2.0
    elif normalized <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def _truncate_label(text: str, max_len: int = 16) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


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


def _load_payload_results_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("results") or []
    if not isinstance(raw_rows, list):
        return []
    return [dict(row) for row in raw_rows if isinstance(row, dict)]


def _build_cumulative_realized_points(payload: dict[str, Any]) -> list[tuple[float, float]]:
    replay = payload.get("replay")
    if not isinstance(replay, list):
        return []

    points: list[tuple[float, float]] = []
    cumulative = 0.0
    idx = 1
    for row in replay:
        if not isinstance(row, dict):
            continue
        cumulative += _safe_float(row.get("realized_change"))
        points.append((float(idx), cumulative))
        idx += 1
    return points


def _build_strategy_pnl_bars(
    strategy_rows: list[dict[str, Any]], max_items: int = 12
) -> list[tuple[str, float]]:
    bars: list[tuple[str, float]] = []
    for row in strategy_rows:
        strategy = str(row.get("strategy", "")).strip() or "(unknown)"
        bars.append((strategy, _safe_float(row.get("pnl"))))
    bars.sort(key=lambda item: item[1], reverse=True)
    return bars[:max_items]


def _parse_iso_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _compute_main_window_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    from_ts = _parse_iso_ts(payload.get("from_ts"))
    to_ts = _parse_iso_ts(payload.get("to_ts"))

    replay = payload.get("replay")
    replay_rows = replay if isinstance(replay, list) else []

    first_replay_dt: datetime | None = None
    for row in replay_rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_iso_ts(row.get("ts"))
        if ts is None:
            continue
        if first_replay_dt is None or ts < first_replay_dt:
            first_replay_dt = ts

    if from_ts is None or to_ts is None:
        return {
            "coverage_ratio": 0.0,
            "history_limited": False,
            "note": "window_parse_error",
            "first_replay_ts": "",
            "effective_from_ts": "",
        }

    effective_from_dt: datetime | None = from_ts
    history_limited = False
    note = "full_history"

    if first_replay_dt is not None and first_replay_dt > from_ts:
        effective_from_dt = first_replay_dt
        history_limited = True
        note = "history_limited"
    elif first_replay_dt is None:
        effective_from_dt = None
        note = "no_replay_rows"

    window_hours = max(0.0, (to_ts - from_ts).total_seconds() / 3600.0)
    covered_hours = (
        max(0.0, (to_ts - effective_from_dt).total_seconds() / 3600.0)
        if effective_from_dt is not None
        else 0.0
    )
    coverage_ratio = (covered_hours / window_hours) if window_hours > 0 else 0.0

    return {
        "coverage_ratio": max(0.0, min(1.0, coverage_ratio)),
        "history_limited": history_limited,
        "note": note,
        "first_replay_ts": (
            first_replay_dt.isoformat().replace("+00:00", "Z")
            if first_replay_dt is not None
            else ""
        ),
        "effective_from_ts": (
            effective_from_dt.isoformat().replace("+00:00", "Z")
            if effective_from_dt is not None
            else ""
        ),
    }


def _format_pct(raw: object) -> str:
    return f"{_safe_float(raw) * 100.0:.2f}%"


def _format_rate_with_samples(rate_raw: object, sample_count_raw: object) -> str:
    sample_count = _safe_int(sample_count_raw)
    if sample_count <= 0:
        return "n/a"
    return _format_pct(rate_raw)


def _extract_rolling_summary(rolling_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(rolling_payload, dict):
        return None
    summary = rolling_payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def _format_rolling_reject_top(
    rolling_summary: dict[str, Any], *, top_k: int = PDF_ROLLING_REJECT_TOP_K
) -> str:
    raw = rolling_summary.get("risk_rejection_reasons")
    if not isinstance(raw, dict):
        return "none"

    top_text, _ = format_reject_top(raw, top_k=top_k, delimiter=";", normalize=True)
    if top_text == "disabled":
        return "none"
    return top_text


def _aggregate_winrate_from_rows(strategy_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    closed_wins = 0
    closed_losses = 0
    mtm_wins = 0
    mtm_losses = 0

    for row in strategy_rows:
        row_closed_wins = _safe_int(row.get("wins"))
        row_closed_losses = _safe_int(row.get("losses"))
        if row_closed_wins <= 0 and row_closed_losses <= 0:
            row_closed_samples = _safe_int(row.get("closed_sample_count"))
            if row_closed_samples > 0:
                row_closed_rate = _safe_float(row.get("closed_winrate", row.get("winrate")))
                row_closed_wins = max(
                    0, min(row_closed_samples, int(round(row_closed_samples * row_closed_rate)))
                )
                row_closed_losses = max(0, row_closed_samples - row_closed_wins)

        row_mtm_wins = _safe_int(row.get("mtm_wins"))
        row_mtm_losses = _safe_int(row.get("mtm_losses"))
        if row_mtm_wins <= 0 and row_mtm_losses <= 0:
            row_mtm_samples = _safe_int(row.get("mtm_sample_count"))
            if row_mtm_samples > 0:
                row_mtm_rate = _safe_float(row.get("mtm_winrate"))
                row_mtm_wins = max(
                    0, min(row_mtm_samples, int(round(row_mtm_samples * row_mtm_rate)))
                )
                row_mtm_losses = max(0, row_mtm_samples - row_mtm_wins)

        closed_wins += max(0, row_closed_wins)
        closed_losses += max(0, row_closed_losses)
        mtm_wins += max(0, row_mtm_wins)
        mtm_losses += max(0, row_mtm_losses)

    closed_samples = closed_wins + closed_losses
    mtm_samples = mtm_wins + mtm_losses
    closed_rate = (closed_wins / closed_samples) if closed_samples > 0 else 0.0
    mtm_rate = (mtm_wins / mtm_samples) if mtm_samples > 0 else 0.0

    return {
        "closed_wins": closed_wins,
        "closed_losses": closed_losses,
        "closed_samples": closed_samples,
        "closed_rate": closed_rate,
        "mtm_wins": mtm_wins,
        "mtm_losses": mtm_losses,
        "mtm_samples": mtm_samples,
        "mtm_rate": mtm_rate,
    }


def render_pdf(
    *,
    payload: dict[str, Any],
    strategy_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    rolling_payload: dict[str, Any] | None,
    output_path: Path,
    title: str,
) -> None:
    try:
        from reportlab.graphics import renderPDF
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.lineplots import LinePlot
        from reportlab.graphics.shapes import Drawing, Line, String
        from reportlab.lib import colors
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
    content_width = width - margin_x * 2
    y = height - margin_y

    def ensure_space(required_height: float) -> None:
        nonlocal y
        if y - required_height < margin_y:
            pdf.showPage()
            y = height - margin_y

    def write_line(text: str, *, bold: bool = False, size: int = 10, leading: int = 14) -> None:
        nonlocal y
        ensure_space(leading)
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        pdf.drawString(margin_x, y, text)
        y -= leading

    def write_wrapped(
        text: str, *, bold: bool = False, size: int = 10, width_chars: int = 95
    ) -> None:
        chunks = textwrap.wrap(text, width=width_chars) or [""]
        for chunk in chunks:
            write_line(chunk, bold=bold, size=size)

    def draw_chart(drawing: Any) -> None:
        nonlocal y
        ensure_space(float(drawing.height) + 8)
        draw_y = y - float(drawing.height)
        renderPDF.draw(drawing, pdf, margin_x, draw_y)
        y = draw_y - 8

    def make_cumulative_chart() -> tuple[Any | None, str | None]:
        points = _build_cumulative_realized_points(payload)
        if len(points) < 2:
            return None, "Not enough replay points for cumulative realized PnL chart"

        drawing = Drawing(content_width, 220)
        chart = LinePlot()
        chart.x = 48
        chart.y = 40
        chart.width = content_width - 70
        chart.height = 135
        chart.data = [points]
        chart.lines[0].strokeColor = colors.HexColor("#1f77b4")
        chart.lines[0].strokeWidth = 1.6

        x_max = max(2, len(points))
        chart.xValueAxis.valueMin = 1
        chart.xValueAxis.valueMax = x_max
        chart.xValueAxis.valueStep = max(1, x_max // 5)
        chart.xValueAxis.labelTextFormat = "%d"

        y_values = [p[1] for p in points]
        y_min = min(y_values)
        y_max = max(y_values)
        if abs(y_max - y_min) < 1e-9:
            pad = max(0.5, abs(y_max) * 0.1 + 0.1)
        else:
            pad = max(0.2, (y_max - y_min) * 0.08)
        y_min -= pad
        y_max += pad
        chart.yValueAxis.valueMin = y_min
        chart.yValueAxis.valueMax = y_max
        chart.yValueAxis.valueStep = _nice_step(y_max - y_min, target_ticks=5)
        chart.yValueAxis.labelTextFormat = "%.2f"

        if y_min < 0 < y_max:
            zero_ratio = (0.0 - y_min) / (y_max - y_min)
            zero_y = chart.y + chart.height * zero_ratio
            drawing.add(
                Line(
                    chart.x,
                    zero_y,
                    chart.x + chart.width,
                    zero_y,
                    strokeColor=colors.HexColor("#bbbbbb"),
                    strokeWidth=0.8,
                )
            )

        drawing.add(chart)
        drawing.add(
            String(
                content_width / 2,
                204,
                "Cumulative Realized PnL (from replay.realized_change)",
                textAnchor="middle",
                fontName="Helvetica-Bold",
                fontSize=10,
            )
        )
        drawing.add(
            String(
                content_width / 2,
                190,
                f"points={len(points)} final={points[-1][1]:.4f}",
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=9,
                fillColor=colors.HexColor("#444444"),
            )
        )
        return drawing, None

    def make_strategy_bar_chart() -> tuple[Any | None, str | None]:
        bars = _build_strategy_pnl_bars(strategy_rows)
        if not bars:
            return None, "No strategy attribution rows for strategy PnL bar chart"

        labels = [_truncate_label(name) for name, _ in bars]
        values = [pnl for _, pnl in bars]

        drawing = Drawing(content_width, 240)
        chart = VerticalBarChart()
        chart.x = 48
        chart.y = 58
        chart.width = content_width - 70
        chart.height = 140
        chart.data = [values]
        chart.groupSpacing = 8

        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.labels.angle = 28
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 8

        y_min = min(values + [0.0])
        y_max = max(values + [0.0])
        if abs(y_max - y_min) < 1e-9:
            pad = max(0.5, abs(y_max) * 0.1 + 0.1)
        else:
            pad = max(0.2, (y_max - y_min) * 0.1)
        y_min -= pad
        y_max += pad
        chart.valueAxis.valueMin = y_min
        chart.valueAxis.valueMax = y_max
        chart.valueAxis.valueStep = _nice_step(y_max - y_min, target_ticks=5)
        chart.valueAxis.labelTextFormat = "%.2f"

        for idx, value in enumerate(values):
            chart.bars[(0, idx)].fillColor = (
                colors.HexColor("#2ca02c") if value >= 0 else colors.HexColor("#d62728")
            )
            chart.bars[(0, idx)].strokeColor = colors.HexColor("#333333")

        drawing.add(chart)
        drawing.add(
            String(
                content_width / 2,
                220,
                "Strategy PnL Attribution",
                textAnchor="middle",
                fontName="Helvetica-Bold",
                fontSize=10,
            )
        )
        drawing.add(
            String(
                content_width / 2,
                206,
                "green>=0, red<0",
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=9,
                fillColor=colors.HexColor("#444444"),
            )
        )
        return drawing, None

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

    # Keep PDF summary winrate source aligned with nightly summary line (payload.results).
    winrate_source_rows = _load_payload_results_rows(payload) or strategy_rows
    winrate_summary = _aggregate_winrate_from_rows(winrate_source_rows)
    closed_summary_samples = _safe_int(winrate_summary.get("closed_samples"))
    mtm_summary_samples = _safe_int(winrate_summary.get("mtm_samples"))
    closed_summary_wins = _safe_int(winrate_summary.get("closed_wins"))
    mtm_summary_wins = _safe_int(winrate_summary.get("mtm_wins"))
    closed_summary_rate = _safe_float(winrate_summary.get("closed_rate"))
    mtm_summary_rate = _safe_float(winrate_summary.get("mtm_rate"))

    main_cov = _compute_main_window_coverage(payload)
    main_cov_ratio = _safe_float(main_cov.get("coverage_ratio"))
    main_cov_history_limited = bool(main_cov.get("history_limited", False))
    main_cov_note = str(main_cov.get("note", ""))
    first_replay_ts = str(main_cov.get("first_replay_ts", ""))
    effective_from_ts = str(main_cov.get("effective_from_ts", ""))

    write_line("Sample Coverage", bold=True, size=12, leading=18)
    write_line(
        "Closed winrate summary: "
        f"{_format_rate_with_samples(closed_summary_rate, closed_summary_samples)} "
        f"({closed_summary_wins}/{closed_summary_samples})"
    )
    write_line(
        "MTM winrate summary:    "
        f"{_format_rate_with_samples(mtm_summary_rate, mtm_summary_samples)} "
        f"({mtm_summary_wins}/{mtm_summary_samples})"
    )
    write_line(f"Main window coverage:   {main_cov_ratio * 100.0:.2f}%")
    write_line(f"Main history limited:   {'true' if main_cov_history_limited else 'false'}")
    write_line(f"Main window note:       {main_cov_note}")
    if main_cov_history_limited:
        write_wrapped(
            "Interpretation note: main-window metrics are history-limited; "
            "prioritize Rolling Summary for trend-level decisions."
        )
    elif main_cov_note == "no_replay_rows":
        write_wrapped(
            "Interpretation note: no replay rows in main window; "
            "treat main-window metrics as unavailable."
        )
    if first_replay_ts:
        write_line(f"First replay ts:        {first_replay_ts}")
    if effective_from_ts:
        write_line(f"Effective from ts:      {effective_from_ts}")
    write_line("(winrate shown as n/a when sample count is zero)")
    write_line("")

    rolling_summary = _extract_rolling_summary(rolling_payload)
    write_line("Rolling Summary", bold=True, size=12, leading=18)
    if rolling_summary is None:
        write_line("(Rolling summary unavailable)")
    else:
        rolling_runs = _safe_int(rolling_summary.get("run_count"))
        rolling_exec_rate = _safe_float(rolling_summary.get("execution_rate"))
        rolling_positive_window_rate = _safe_float(rolling_summary.get("positive_window_rate"))
        rolling_empty_windows = _safe_int(rolling_summary.get("empty_window_count"))
        rolling_coverage_label = str(rolling_summary.get("coverage_label", "unknown"))
        rolling_reject_top = _format_rolling_reject_top(
            rolling_summary, top_k=PDF_ROLLING_REJECT_TOP_K
        )
        write_line(f"Rolling runs:            {rolling_runs}")
        write_line(f"Rolling execution rate:  {rolling_exec_rate * 100.0:.2f}%")
        write_line("Rolling positive windows: " f"{rolling_positive_window_rate * 100.0:.2f}%")
        write_line(f"Rolling empty windows:   {rolling_empty_windows}")
        write_line(f"Rolling coverage label:  {rolling_coverage_label}")
        write_wrapped(f"Rolling reject top:      {rolling_reject_top}")
    write_line("")

    write_line("Strategy Metrics", bold=True, size=12, leading=18)
    if not strategy_rows:
        write_line("(No strategy rows)")
    else:
        for row in strategy_rows:
            strategy = str(row.get("strategy", ""))
            pnl = _safe_float(row.get("pnl"))
            closed_winrate = _format_rate_with_samples(
                row.get("closed_winrate", row.get("winrate")), row.get("closed_sample_count")
            )
            mtm_winrate = _format_rate_with_samples(
                row.get("mtm_winrate"), row.get("mtm_sample_count")
            )
            max_dd = _safe_float(row.get("max_drawdown"))
            trades = _safe_int(row.get("trade_count"))
            wins = _safe_int(row.get("wins"))
            losses = _safe_int(row.get("losses"))
            closed_samples = _safe_int(row.get("closed_sample_count"))
            mtm_samples = _safe_int(row.get("mtm_sample_count"))
            write_wrapped(
                " - "
                + f"{strategy}: pnl={pnl:.4f}, closed_winrate={closed_winrate}"
                + f" ({wins}/{closed_samples}), mtm_winrate={mtm_winrate}, "
                + f"max_drawdown={max_dd:.4f}, trades={trades}, losses={losses}, mtm_samples={mtm_samples}"
            )

    write_line("")
    write_line("PnL Charts", bold=True, size=12, leading=18)

    cumulative_chart, cumulative_msg = make_cumulative_chart()
    if cumulative_chart is not None:
        draw_chart(cumulative_chart)
    else:
        write_wrapped(f" - Cumulative realized PnL chart unavailable: {cumulative_msg}")

    strategy_chart, strategy_chart_msg = make_strategy_bar_chart()
    if strategy_chart is not None:
        draw_chart(strategy_chart)
    else:
        write_wrapped(f" - Strategy PnL chart unavailable: {strategy_chart_msg}")

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
            closed_wr = _format_rate_with_samples(
                row.get("closed_winrate", row.get("winrate")), row.get("closed_sample_count")
            )
            mtm_wr = _format_rate_with_samples(row.get("mtm_winrate"), row.get("mtm_sample_count"))
            write_wrapped(
                " - "
                + f"{strategy}/{event_id}: pnl={pnl:.4f}, "
                + f"closed_winrate={closed_wr}, mtm_winrate={mtm_wr}, trades={trades}"
            )

    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render monomarket backtest PDF report")
    parser.add_argument("--backtest-json", required=True, help="Path to backtest JSON artifact")
    parser.add_argument("--strategy-csv", help="Optional strategy attribution CSV")
    parser.add_argument("--event-csv", help="Optional event attribution CSV")
    parser.add_argument("--rolling-json", help="Optional rolling summary JSON")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--title", default="Monomarket Backtest Report", help="PDF title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtest_json = Path(args.backtest_json)
    strategy_csv = Path(args.strategy_csv) if args.strategy_csv else None
    event_csv = Path(args.event_csv) if args.event_csv else None
    rolling_json = Path(args.rolling_json) if args.rolling_json else None
    output = Path(args.output)

    payload = _load_payload(backtest_json)
    strategy_rows = _load_strategy_rows(payload, strategy_csv)
    event_rows = _load_event_rows(payload, event_csv)
    rolling_payload: dict[str, Any] | None = None
    if rolling_json and rolling_json.exists():
        rolling_payload = _load_payload(rolling_json)

    render_pdf(
        payload=payload,
        strategy_rows=strategy_rows,
        event_rows=event_rows,
        rolling_payload=rolling_payload,
        output_path=output,
        title=args.title,
    )
    print(f"[backtest-pdf] report generated: {output}")


if __name__ == "__main__":
    main()
