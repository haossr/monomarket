#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2


RAW_FILENAMES = ("latest.json", "strategy.csv", "event.csv", "replay.csv")


@dataclass
class StrategyRow:
    strategy: str
    pnl: float
    trade_count: int
    winrate: float
    winrate_label: str
    max_drawdown: float


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static HTML report for one backtest run")
    parser.add_argument("--run-dir", required=True, help="Run directory containing latest.json/strategy.csv/event.csv/replay.csv")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: artifacts/backtest/web/<timestamp>/",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_raw_artifacts(run_dir: Path, output_dir: Path) -> dict[str, str]:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for name in RAW_FILENAMES:
        src = run_dir / name
        if src.exists():
            dst = raw_dir / name
            copy2(src, dst)
            links[name] = f"raw/{name}"
    return links


def _collect_strategies(payload: dict) -> list[StrategyRow]:
    rows: list[StrategyRow] = []
    for item in payload.get("results", []):
        closed_samples = int(item.get("closed_sample_count", 0) or 0)
        mtm_samples = int(item.get("mtm_sample_count", 0) or 0)
        if closed_samples > 0:
            winrate = float(item.get("closed_winrate", 0.0) or 0.0)
            label = "closed_winrate"
        elif mtm_samples > 0:
            winrate = float(item.get("mtm_winrate", 0.0) or 0.0)
            label = "mtm_winrate"
        else:
            winrate = float(item.get("winrate", 0.0) or 0.0)
            label = "winrate"
        rows.append(
            StrategyRow(
                strategy=str(item.get("strategy", "")),
                pnl=float(item.get("pnl", 0.0) or 0.0),
                trade_count=int(item.get("trade_count", 0) or 0),
                winrate=winrate,
                winrate_label=label,
                max_drawdown=float(item.get("max_drawdown", 0.0) or 0.0),
            )
        )
    rows.sort(key=lambda x: x.strategy)
    return rows


def _fmt_num(v: float, digits: int = 4) -> str:
    return f"{v:,.{digits}f}"


def _fmt_pct(v: float) -> str:
    return f"{(v * 100):.2f}%"


def _render_html(*, payload: dict, run_dir: Path, output_dir: Path, links: dict[str, str]) -> str:
    rows = _collect_strategies(payload)
    total_pnl = sum(r.pnl for r in rows)
    max_dd = max((r.max_drawdown for r in rows), default=0.0)

    table_rows = "\n".join(
        "".join(
            [
                "<tr>",
                f"<td>{html.escape(r.strategy)}</td>",
                f"<td class='num'>{_fmt_num(r.pnl)}</td>",
                f"<td class='num'>{r.trade_count}</td>",
                f"<td class='num'>{_fmt_pct(r.winrate)}</td>",
                f"<td>{html.escape(r.winrate_label)}</td>",
                "</tr>",
            ]
        )
        for r in rows
    )

    summary_items = [
        ("Total PnL", _fmt_num(total_pnl)),
        ("Max Drawdown", _fmt_num(max_dd)),
        ("Executed Signals", str(int(payload.get("executed_signals", 0) or 0))),
        ("Rejected Signals", str(int(payload.get("rejected_signals", 0) or 0))),
    ]
    cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(k)}</div><div class='value'>{html.escape(v)}</div></div>"
        for k, v in summary_items
    )

    link_items = "\n".join(
        f"<li><a href='{html.escape(rel)}'>{html.escape(name)}</a></li>" for name, rel in links.items()
    )

    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Monomarket Backtest Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #555; margin-bottom: 18px; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fafafa; }}
    .label {{ font-size: 12px; color: #666; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; font-size: 14px; }}
    th {{ background: #f5f5f5; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f3f4f6; border-radius: 4px; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Monomarket Backtest Report</h1>
  <div class='meta'>
    <div><strong>Generated:</strong> {html.escape(datetime.now(timezone.utc).isoformat())}</div>
    <div><strong>Window:</strong> {html.escape(str(payload.get('from_ts', 'n/a')))} → {html.escape(str(payload.get('to_ts', 'n/a')))}</div>
    <div><strong>Run dir:</strong> <code>{html.escape(str(run_dir))}</code></div>
    <div><strong>Report dir:</strong> <code>{html.escape(str(output_dir))}</code></div>
    <div><strong>Schema:</strong> {html.escape(str(payload.get('schema_version', 'n/a')))}</div>
  </div>

  <h2>Summary</h2>
  <div class='cards'>
    {cards}
  </div>

  <h2>Per-strategy</h2>
  <table>
    <thead>
      <tr>
        <th>strategy</th>
        <th>pnl</th>
        <th>trade_count</th>
        <th>winrate</th>
        <th>winrate_source</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <h2>Raw Artifacts</h2>
  <ul>
    {link_items}
  </ul>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    for required in RAW_FILENAMES:
        if not (run_dir / required).exists():
            raise SystemExit(f"missing required artifact: {run_dir / required}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (Path("artifacts/backtest/web") / _iso_utc_now()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(run_dir / "latest.json")
    links = _copy_raw_artifacts(run_dir, out_dir)
    html_text = _render_html(payload=payload, run_dir=run_dir, output_dir=out_dir, links=links)

    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

    # Small machine-readable sidecar for quick integrations.
    with (out_dir / "report-meta.json").open("w", encoding="utf-8", newline="") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "report_dir": str(out_dir),
                "raw_links": links,
                "total_pnl": sum(float(r.get("pnl", 0.0) or 0.0) for r in payload.get("results", [])),
                "max_drawdown": max((float(r.get("max_drawdown", 0.0) or 0.0) for r in payload.get("results", [])), default=0.0),
                "executed_signals": int(payload.get("executed_signals", 0) or 0),
                "rejected_signals": int(payload.get("rejected_signals", 0) or 0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Optional validation: strategy CSV exists and can be parsed (sanity only).
    with (run_dir / "strategy.csv").open("r", encoding="utf-8", newline="") as f:
        _ = list(csv.DictReader(f))

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
