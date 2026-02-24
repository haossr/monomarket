# Runbook

## 1) 初始化

```bash
cp configs/config.example.yaml configs/config.yaml
monomarket init-db
```

## 2) 数据抓取

```bash
# 默认增量抓取（带重试退避、基础限流、错误分类、source 级熔断）
monomarket ingest --source gamma --limit 300 --incremental
monomarket ingest --source data --limit 300 --incremental
monomarket ingest --source clob --limit 300 --incremental
# 或
monomarket ingest --source all --limit 300 --incremental

# 强制全量（忽略 checkpoint）
monomarket ingest --source all --limit 300 --full
```

`ingest` 输出会额外包含 `error_buckets`（如 `http_429/http_5xx/circuit_open`），用于观测分级重试与熔断恢复状态。
当 breaker 冷却到期后，系统会执行单次 `half-open` 探测请求：成功则关闭 breaker，失败则立即重新打开。

查看聚合健康状态（错误分桶 + bucket 趋势 + breaker 状态）：

```bash
monomarket ingest-health --source gamma --run-window 20 --error-trend-window 20 --error-trend-top-movers --error-share-top-k 3 --error-share-min-share 0.05 --error-share-min-count 2 --error-share-min-runs-with-error 1 --error-share-min-total-runs 5 --error-share-min-source-bucket-total 10 --error-sample-limit 5
# 或查看全部 source
monomarket ingest-health --run-window 20 --error-trend-window 20 --error-trend-top-movers --error-share-top-k 3 --error-share-min-share 0.05 --error-share-min-count 2 --error-share-min-runs-with-error 1 --error-share-min-total-runs 5 --error-share-min-source-bucket-total 10 --error-sample-limit 5
```

输出包含：
- error buckets 聚合
- error bucket 趋势（最近窗口 vs 前一窗口，默认按 |delta| top movers 排序）
- breaker 状态
- breaker 状态过渡计数（open/half_open/closed）与最近转移时间
- 近 N 次 run 的 source 级摘要：`non_ok_rate / avg_failures / avg_retries / failure_per_req`
- 近 N 次 run 的 source 级错误类别占比（http_429/http_5xx/timeout 等，支持 top-k + min-share + min-count + min-runs + min-total-runs + min-source-total 过滤）
- 当过滤条件过严导致 share 子表为空时，CLI 会提示“error share empty after filters”
- 按 source 的最近错误样本（top-N last_error）

## 3) 生成与查看信号

```bash
monomarket generate-signals --strategies s1,s2,s4,s8
monomarket list-signals --status new --limit 30
```

## 4) 执行

### 4.1 纸上交易（默认）

```bash
monomarket execute-signal 1 --mode paper
```

### 4.2 实盘开关流程

```bash
monomarket set-switch ENABLE_LIVE_TRADING true
monomarket set-switch REQUIRE_MANUAL_CONFIRM true
monomarket set-switch KILL_SWITCH false
monomarket switches
```

执行 live 订单时需显式确认：

```bash
monomarket execute-signal 1 --mode live --confirm-live
```

live 凭据（二选一）：
- `POLYMARKET_CLOB_HEADERS_JSON`（完整请求头 JSON）
- `POLYMARKET_API_KEY` + `POLYMARKET_API_SECRET` + `POLYMARKET_API_PASSPHRASE`

若缺少凭据，系统会拒单并记录原因。

live 闭环操作：

```bash
# 同步 live 订单回报（状态/成交）
monomarket live-sync --limit 100

# 撤销指定本地订单 ID
monomarket live-cancel <local_order_id>
```

## 5) 手工开平仓（Rocky 独立操作）

```bash
# 开仓
monomarket place-order \
  --strategy manual --market-id <market> --event-id <event> \
  --token-id YES --side buy --action open --price 0.13 --qty 30 --mode paper

# 平仓
monomarket place-order \
  --strategy manual --market-id <market> --event-id <event> \
  --token-id YES --side sell --action close --price 0.22 --qty 30 --mode paper
```

## 6) 风险与收益检查

```bash
monomarket pnl-report
monomarket metrics-report
```

## 7) 24h paper soak test（可复现 + 状态输出 + 失败重试）

```bash
bash scripts/paper_soak_24h.sh \
  --hours 24 \
  --interval-sec 300 \
  --max-signals-per-cycle 10 \
  --retry-max 3 \
  --config configs/soak.paper.yaml

# 查看状态
bash scripts/paper_soak_status.sh
```

产物：
- `artifacts/soak/paper-<ts>/soak.log`
- `artifacts/soak/paper-<ts>/status/latest.json`
- `artifacts/soak/paper-<ts>/status/history.jsonl`

## 8) 回测（signals replay + 策略/事件归因）

```bash
monomarket backtest --strategies s1,s2,s4,s8 \
  --from 2026-02-20T00:00:00Z --to 2026-02-22T23:59:59Z \
  --partial-fill --liquidity-full-fill 1000 --min-fill-ratio 0.10 \
  --fill-probability --min-fill-probability 0.05 \
  --dynamic-slippage --spread-slippage-weight-bps 50 --liquidity-slippage-weight-bps 25 \
  --replay-limit 30 \
  --out-json artifacts/backtest/latest.json --with-checksum \
  --out-replay-csv artifacts/backtest/replay.csv \
  --out-strategy-csv artifacts/backtest/strategy.csv \
  --out-event-csv artifacts/backtest/event.csv \
  --with-csv-digest-sidecar
```

输出包含：
- 按策略归因：`pnl / winrate / max drawdown / 交易次数`
- 按事件归因：`strategy + event + pnl / winrate / max drawdown / 交易次数`
- 回放账本：逐条 signal replay（时间、market、token、requested/filled qty、fill ratio、fill probability、slippage bps、realized、累计权益、risk allow/reject + reason）
- 机器可读导出：
  - `--out-json`：完整报告（summary + execution/risk config snapshot + strategy/event attribution + replay rows）
  - 可配 `--with-checksum`：在 JSON 中附带 `checksum_algo/checksum_sha256`
  - `--out-replay-csv`：replay ledger CSV（便于审计/外部分析）
  - `--out-strategy-csv`：策略维度归因 CSV
  - `--out-event-csv`：事件维度归因 CSV
  - 可配 `--with-csv-digest-sidecar`：为每个导出 CSV 生成 `.sha256` sidecar
  - 导出工件统一带 `schema_version` 字段（兼容解析）
  - 解析端可使用 `monomarket.backtest.parse_schema_version` / `assert_schema_compatible`
  - JSON 解析可进一步使用 `monomarket.backtest.validate_backtest_json_artifact`
  - 双栈校验可传 `supported_major=None` 并注入 `validators={2: ...}`
  - 内置占位校验器：`monomarket.backtest.validate_backtest_json_artifact_v2`

滚动窗口多样本回测（稳定性观察）：

```bash
monomarket backtest-rolling --strategies s1,s2,s4,s8 \
  --from 2026-02-20T00:00:00Z --to 2026-02-23T00:00:00Z \
  --window-hours 24 --step-hours 12 \
  --out-json artifacts/backtest/rolling-summary.json
```

输出包含每个窗口的 `signals/executed/rejected/execution_rate/pnl_total`，以及策略级 `avg_pnl/avg_winrate/total_trades` 聚合。
rolling JSON 工件还包含：
- `schema_version`（当前 `rolling-1.0`）
- `overlap_mode`（`overlap`/`tiled`/`gapped`）标记窗口重叠语义
- `execution_config` / `risk_config` 参数快照
- `summary` 扩展指标：`empty_window_count` / `positive_window_rate` / `pnl_sum` / `pnl_avg`
- 覆盖与复用指标：`coverage_label`（full/partial/sparse）+ `range_hours` / `coverage_ratio` / `overlap_ratio`（含 `sampled_hours` / `covered_hours` / `overlap_hours`）
- 风控拒单原因分布（`risk_rejection_reasons`，窗口级 + 汇总级）

v1 -> v2 迁移命令：

```bash
monomarket backtest-migrate-v1-to-v2 \
  --in artifacts/backtest/latest.json \
  --out artifacts/backtest/latest.v2.json

# 查看字段级映射与可逆性
monomarket backtest-migration-map
monomarket backtest-migration-map --format json
# 导出 machine-readable mapping artifact
monomarket backtest-migration-map --format json --with-checksum \
  --out-json artifacts/backtest/migration-map.json
```

回测 vs paper 对齐报告建议使用：
- `docs/backtest-vs-paper-alignment-template.md`

## 9) 停机保护

```bash
monomarket set-switch KILL_SWITCH true
```

KILL_SWITCH 生效后所有新单拒绝。

## 10) 单轮回测流水线（可复用脚本）

```bash
bash scripts/backtest_cycle.sh \
  --lookback-hours 24 \
  --market-limit 2000 \
  --ingest-limit 300 \
  --config configs/soak.paper.yaml
```

产物目录：`artifacts/backtest/runs/<timestamp>/`
- `latest.json`
- `replay.csv` + `replay.csv.sha256`
- `strategy.csv` + `strategy.csv.sha256`
- `event.csv` + `event.csv.sha256`
- `summary.md`

并会更新 latest 指针：
- `artifacts/backtest/latest-run.json`
- `artifacts/backtest/latest`（symlink）

## 11) 生成 PDF 报告

```bash
uv run --with reportlab python scripts/backtest_pdf_report.py \
  --backtest-json artifacts/backtest/runs/<timestamp>/latest.json \
  --strategy-csv artifacts/backtest/runs/<timestamp>/strategy.csv \
  --event-csv artifacts/backtest/runs/<timestamp>/event.csv \
  --output artifacts/backtest/runs/<timestamp>/report.pdf
```

> 若环境已安装 reportlab，也可直接 `python scripts/backtest_pdf_report.py ...`。

## 12) Nightly 一键产出（回测 + PDF）

```bash
bash scripts/backtest_nightly_report.sh \
  --lookback-hours 24 \
  --market-limit 2000 \
  --ingest-limit 300 \
  --rolling-window-hours 24 \
  --rolling-step-hours 12 \
  --rolling-reject-top-k 2 \
  --config configs/soak.paper.yaml
```

夜间目录：`artifacts/backtest/nightly/<YYYY-MM-DD>/`
- `report.pdf`
- `summary.txt`（含 rolling `pos_win_rate/empty_windows` 与 canonical 别名 `positive_window_rate/empty_window_count`，以及 `range_h/coverage/overlap` 与 canonical 别名 `range_hours/coverage_ratio/overlap_ratio`、`coverage_label`、`rolling_reject_top_k`、主要拒单原因摘要）
- `summary.json`（结构化 sidecar，便于机器解析）
- `rolling-summary.json`（滚动窗口多样本回测汇总）
- `run-<timestamp>/`（本轮 JSON/CSV/summary.md 工件）

`--rolling-reject-top-k` 语义：`0=disabled`（关闭拒单原因摘要输出），`N>0` 输出前 N 个原因（无数据时为 `none`）。
`rolling_reject_top` 使用 `;` 作为原因分隔符（如 `reasonA:3;reasonB:1`）；消费端优先读取 `summary.json` 的 `reject_top_pairs`。

## 13) 指标解释（回测与报告通用）

- `executed_signals / rejected_signals`：信号执行/拒绝数量（风控与流动性影响的核心观测项）
- `winrate`：已闭合交易中的胜率（`wins / (wins + losses)`）
- `max_drawdown`：权益曲线历史峰值到后续低点的最大回撤
- `pnl`：策略（或事件）维度最终盈亏
- `trade_count`：成交交易次数
- `replay.csv`：逐笔 replay 账本（含 fill ratio、slippage、risk reason）
