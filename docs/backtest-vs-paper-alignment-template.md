# Backtest vs Paper 对齐报告模板

> 用途：对同一时间窗、同一策略集合的 backtest 与 paper soak 结果做一致性审计。

## 1) 报告元信息

- 报告时间：`<YYYY-MM-DD HH:MM TZ>`
- 对齐窗口：`<from_ts>` ~ `<to_ts>`
- 策略集合：`<s1,s2,s4,s8>`
- 版本：
  - git commit: `<sha>`
  - backtest schema_version: `<x.y>`
  - soak run id: `<artifacts/soak/paper-...>`
- 配置快照：
  - execution_config: `<json path>`
  - risk_config: `<json path>`

## 2) 数据来源

- Backtest 工件：
  - JSON: `<artifacts/backtest/latest.json>`
  - replay CSV: `<artifacts/backtest/replay.csv>`
  - strategy CSV: `<artifacts/backtest/strategy.csv>`
  - event CSV: `<artifacts/backtest/event.csv>`
- Paper 工件：
  - soak status history: `<artifacts/soak/.../status/history.jsonl>`
  - DB: `<data/monomarket-paper-soak.db>`
  - 导出快照（若有）：`<path>`

## 3) 核心对齐指标（必填）

| 指标 | Backtest | Paper | Diff | Diff% | 备注 |
|---|---:|---:|---:|---:|---|
| total_signals | `<n>` | `<n>` | `<n>` | `<%>` | |
| executed_signals | `<n>` | `<n>` | `<n>` | `<%>` | |
| rejected_signals | `<n>` | `<n>` | `<n>` | `<%>` | |
| fill_rate | `<v>` | `<v>` | `<v>` | `<%>` | |
| realized_pnl | `<v>` | `<v>` | `<v>` | `<%>` | |
| unrealized_pnl | `<v>` | `<v>` | `<v>` | `<%>` | |
| max_drawdown | `<v>` | `<v>` | `<v>` | `<%>` | |
| avg_fill_ratio | `<v>` | `<v>` | `<v>` | `<%>` | |
| avg_fill_probability | `<v>` | `<v>` | `<v>` | `<%>` | |
| avg_slippage_bps | `<v>` | `<v>` | `<v>` | `<%>` | |

## 4) 策略维度对齐

| strategy | backtest_pnl | paper_pnl | diff | diff% | backtest_trades | paper_trades | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| s1 | | | | | | | |
| s2 | | | | | | | |
| s4 | | | | | | | |
| s8 | | | | | | | |

## 5) 事件维度对齐（Top N 偏差）

| rank | event_id | strategy | pnl_bt | pnl_paper | pnl_diff | trade_bt | trade_paper | reject_bt | reject_paper | 风险原因 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | | | | | | | | | | |
| 2 | | | | | | | | | | |
| 3 | | | | | | | | | | |

## 6) 风控与执行轨迹核对

- 风控拒单原因分布（backtest vs paper）：
  - `event notional limit exceeded`: `<bt/paper>`
  - `strategy notional limit exceeded`: `<bt/paper>`
  - `global stop-loss triggered`: `<bt/paper>`
  - `circuit breaker open`: `<bt/paper>`
- 执行偏差检查：
  - partial fill 假设 vs 实际：`<summary>`
  - fill probability 假设 vs 实际：`<summary>`
  - dynamic slippage 假设 vs 实际：`<summary>`

## 7) 结论与行动项

- 对齐结论：`<aligned / partially_aligned / not_aligned>`
- 主要偏差来源（最多 3 条）：
  1. `<...>`
  2. `<...>`
  3. `<...>`
- 后续行动：
  - [ ] `<参数校准项>`
  - [ ] `<执行模型修正项>`
  - [ ] `<数据质量修复项>`

## 8) 附录（建议）

- 关键命令与参数（可复现）：
  - backtest 命令：`<cmd>`
  - soak 命令：`<cmd>`
- 工件校验：
  - backtest JSON schema 校验：`<pass/fail>`
  - migration map checksum（如使用）：`<sha256>`
