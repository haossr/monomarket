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

查看聚合健康状态（错误分桶 + breaker 状态）：

```bash
monomarket ingest-health --source gamma --run-window 20
# 或查看全部 source
monomarket ingest-health --run-window 20
```

输出包含：
- error buckets 聚合
- breaker 状态
- breaker 状态过渡计数（open/half_open/closed）与最近转移时间
- 近 N 次 run 的 source 级失败率摘要（non_ok_rate）

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

> 若缺少 `POLYMARKET_API_KEY`，系统会拒单并记录原因。

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

## 7) 回测（signals replay + 策略/事件归因）

```bash
monomarket backtest --strategies s1,s2,s4,s8 \
  --from 2026-02-20T00:00:00Z --to 2026-02-22T23:59:59Z \
  --partial-fill --liquidity-full-fill 1000 --min-fill-ratio 0.10 \
  --fill-probability --min-fill-probability 0.05 \
  --dynamic-slippage --spread-slippage-weight-bps 50 --liquidity-slippage-weight-bps 25 \
  --replay-limit 30 \
  --out-json artifacts/backtest/latest.json \
  --out-replay-csv artifacts/backtest/replay.csv \
  --out-strategy-csv artifacts/backtest/strategy.csv \
  --out-event-csv artifacts/backtest/event.csv
```

输出包含：
- 按策略归因：`pnl / winrate / max drawdown / 交易次数`
- 按事件归因：`strategy + event + pnl / winrate / max drawdown / 交易次数`
- 回放账本：逐条 signal replay（时间、market、token、requested/filled qty、fill ratio、fill probability、slippage bps、realized、累计权益、risk allow/reject + reason）
- 机器可读导出：
  - `--out-json`：完整报告（summary + execution/risk config snapshot + strategy/event attribution + replay rows）
  - `--out-replay-csv`：replay ledger CSV（便于审计/外部分析）
  - `--out-strategy-csv`：策略维度归因 CSV
  - `--out-event-csv`：事件维度归因 CSV
  - 导出工件统一带 `schema_version` 字段（兼容解析）
  - 解析端可使用 `monomarket.backtest.parse_schema_version` / `assert_schema_compatible`

## 8) 停机保护

```bash
monomarket set-switch KILL_SWITCH true
```

KILL_SWITCH 生效后所有新单拒绝。
