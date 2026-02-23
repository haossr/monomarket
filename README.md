# monomarket

Monomarket 是一个面向二元市场（Polymarket 风格）的可运行交易 MVP，目标是先达到：

- 可稳定抓取数据（gamma/data/clob，含分级重试、source 级熔断与 half-open 探测）
- 统一归一化存储（SQLite）
- 信号引擎（可插拔，优先 S1/S2/S4/S8）
- 执行路由（paper/live 双模式，默认 paper）
- 统一风控（全局止损、单策略上限、单事件上限、熔断）
- PnL + 指标报表
- 时间窗口回测（signals replay + paper fills + 可选 liquidity partial fill + 策略归因 + 事件级归因 + 风控决策轨迹 + JSON/CSV 导出）

> 安全默认值：`paper`，且 `ENABLE_LIVE_TRADING=false`。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp configs/config.example.yaml configs/config.yaml
```

## 启动流程（MVP）

```bash
# 1) 初始化数据库
monomarket init-db

# 2) 抓取市场数据（默认增量；可选 source=gamma/data/clob/all）
monomarket ingest --source all --limit 300 --incremental

# 2.1) 查看抓取健康（错误分桶 + breaker + 状态过渡 + 近 N 次失败率 + 最近错误样本）
monomarket ingest-health --run-window 20 --error-sample-limit 5

# 3) 生成策略信号（S1/S2/S4/S8）
monomarket generate-signals --strategies s1,s2,s4,s8

# 4) 查看候选信号
monomarket list-signals --status new --limit 20

# 5) 执行信号（默认 paper）
monomarket execute-signal 1

# 6) 查看收益与指标
monomarket pnl-report
monomarket metrics-report

# 7) 时间窗口回测（按策略/事件归因 + 回放账本）
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

回放账本（终端 + CSV）包含 `requested/filled qty`、`fill_ratio`、`fill_probability`、`slippage_bps_applied` 以及风控决策字段：`risk_allowed` / `risk_reason` 与阈值快照，便于离线审计。
所有导出工件（JSON + 各 CSV）均包含 `schema_version` 字段，用于向后兼容与审计解析。
JSON 额外包含 `execution_config` / `risk_config` 快照，用于可重复回放与参数审计。
归因结果可分别导出 strategy/event CSV，便于接审计流水线与 BI。

v1->v2 迁移（校验 + 转换）：

```bash
monomarket backtest-migrate-v1-to-v2 \
  --in artifacts/backtest/latest.json \
  --out artifacts/backtest/latest.v2.json

# 查看字段映射与可逆性
monomarket backtest-migration-map
# 导出 machine-readable mapping artifact
monomarket backtest-migration-map --format json --out-json artifacts/backtest/migration-map.json
```

## 交易控制开关（Rocky 可独立控制）

支持 3 个统一开关：

- `ENABLE_LIVE_TRADING`
- `REQUIRE_MANUAL_CONFIRM`
- `KILL_SWITCH`

查看当前开关：

```bash
monomarket switches
```

设置开关（写入 SQLite runtime switches）：

```bash
monomarket set-switch ENABLE_LIVE_TRADING true
monomarket set-switch REQUIRE_MANUAL_CONFIRM true
monomarket set-switch KILL_SWITCH false
```

手工下单（不依赖 Hao 介入）：

```bash
# 手工开仓（paper）
monomarket place-order \
  --strategy manual --market-id <market_id> --event-id <event_id> \
  --token-id YES --side buy --action open --price 0.12 --qty 20 --mode paper

# 手工实盘尝试（需要开关 + --confirm-live + 凭据）
monomarket place-order \
  --strategy manual --market-id <market_id> --event-id <event_id> \
  --token-id NO --side buy --action open --price 0.78 --qty 10 \
  --mode live --confirm-live
```

## Live 说明

- 默认关闭 live（`ENABLE_LIVE_TRADING=false`）
- 即便切换 live，若缺少 `POLYMARKET_API_KEY` 会被明确拒单，不会伪造成交
- `REQUIRE_MANUAL_CONFIRM=true` 时，live 指令必须带 `--confirm-live`
- `KILL_SWITCH=true` 时所有新单拒绝

## 本地复现 CI

```bash
make ci
# 或
bash scripts/ci_local.sh
```

## 文档

- `docs/architecture.md`：架构图与模块说明
- `docs/runbook.md`：运行手册
- `docs/backtest-schema.md`：回测导出 schema 与迁移约定
- `docs/strategies.md`：S1/S2/S4/S8 策略细节
