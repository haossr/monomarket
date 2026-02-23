# Runbook

## 1) 初始化

```bash
cp configs/config.example.yaml configs/config.yaml
monomarket init-db
```

## 2) 数据抓取

```bash
# 默认增量抓取（带重试退避、基础限流）
monomarket ingest --source gamma --limit 300 --incremental
monomarket ingest --source data --limit 300 --incremental
monomarket ingest --source clob --limit 300 --incremental
# 或
monomarket ingest --source all --limit 300 --incremental

# 强制全量（忽略 checkpoint）
monomarket ingest --source all --limit 300 --full
```

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

## 7) 回测（signals replay + 归因）

```bash
monomarket backtest --strategies s1,s2,s4,s8 \
  --from 2026-02-20T00:00:00Z --to 2026-02-22T23:59:59Z
```

输出包含：按策略的 `pnl / winrate / max drawdown / 交易次数`。

## 8) 停机保护

```bash
monomarket set-switch KILL_SWITCH true
```

KILL_SWITCH 生效后所有新单拒绝。
