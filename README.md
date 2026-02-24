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

# 2.1) 查看抓取健康（错误分桶 + bucket 趋势 top movers + 错误类别占比 + breaker + 状态过渡 + 近 N 次失败率/重试 + 最近错误样本）
monomarket ingest-health --run-window 20 --error-trend-window 20 --error-trend-top-movers --error-share-top-k 3 --error-share-min-share 0.05 --error-share-min-count 2 --error-share-min-runs-with-error 1 --error-share-min-total-runs 5 --error-sample-limit 5

# 3) 生成策略信号（S1/S2/S4/S8）
monomarket generate-signals --strategies s1,s2,s4,s8

# 4) 查看候选信号
monomarket list-signals --status new --limit 20

# 5) 执行信号（默认 paper）
monomarket execute-signal 1

# 6) 查看收益与指标
monomarket pnl-report
monomarket metrics-report

# 6.1) live 报单回报同步 / 撤单（仅在你显式开启 live 后使用）
monomarket live-sync --limit 100
monomarket live-cancel <local_order_id>

# 7) 时间窗口回测（按策略/事件归因 + 回放账本）
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

回放账本（终端 + CSV）包含 `requested/filled qty`、`fill_ratio`、`fill_probability`、`slippage_bps_applied` 以及风控决策字段：`risk_allowed` / `risk_reason` 与阈值快照，便于离线审计。
所有导出工件（JSON + 各 CSV）均包含 `schema_version` 字段，用于向后兼容与审计解析。
JSON 额外包含 `execution_config` / `risk_config` 快照，用于可重复回放与参数审计。
使用 `backtest --out-json ... --with-checksum` 时，还会附带 `checksum_algo/checksum_sha256` 便于跨系统完整性校验。
使用 `--with-csv-digest-sidecar` 时，会为每个导出 CSV 写入同名 `.sha256` sidecar。
归因结果可分别导出 strategy/event CSV，便于接审计流水线与 BI。

多样本滚动回测（用于策略稳定性观察）：

```bash
monomarket backtest-rolling --strategies s1,s2,s4,s8 \
  --from 2026-02-20T00:00:00Z --to 2026-02-23T00:00:00Z \
  --window-hours 24 --step-hours 12 \
  --out-json artifacts/backtest/rolling-summary.json
```

rolling summary 会额外输出：
- `schema_version`（当前 `rolling-1.0`）
- `overlap_mode`（`overlap`/`tiled`/`gapped`）用于标记窗口重叠语义
- `execution_config` / `risk_config` 快照
- `summary` 扩展指标：`empty_window_count` / `positive_window_rate` / `pnl_sum` / `pnl_avg`
- 覆盖与复用指标：`coverage_label`（full/partial/sparse）+ `range_hours` / `coverage_ratio` / `overlap_ratio`（含 `sampled_hours` / `covered_hours` / `overlap_hours`）
- 风控拒单原因分布（`risk_rejection_reasons`，窗口级 + 汇总级）

v1->v2 迁移（校验 + 转换）：

```bash
monomarket backtest-migrate-v1-to-v2 \
  --in artifacts/backtest/latest.json \
  --out artifacts/backtest/latest.v2.json

# 查看字段映射与可逆性
monomarket backtest-migration-map
# 导出 machine-readable mapping artifact（含校验和）
monomarket backtest-migration-map --format json --with-checksum \
  --out-json artifacts/backtest/migration-map.json
```

## 本周持续回测 + 每晚 PDF 报告

单轮流水线（init-db -> ingest gamma incremental -> generate-signals -> backtest）：

```bash
bash scripts/backtest_cycle.sh \
  --lookback-hours 24 \
  --market-limit 2000 \
  --ingest-limit 300 \
  --config configs/soak.paper.yaml
```

输出：`artifacts/backtest/runs/<timestamp>/latest.json|replay.csv|strategy.csv|event.csv|summary.md`
（并为各 CSV 生成 `.sha256` sidecar）。
并自动更新 latest 指针：`artifacts/backtest/latest-run.json`（及 `artifacts/backtest/latest` symlink）。

生成 PDF：

```bash
uv run --with reportlab python scripts/backtest_pdf_report.py \
  --backtest-json artifacts/backtest/runs/<timestamp>/latest.json \
  --strategy-csv artifacts/backtest/runs/<timestamp>/strategy.csv \
  --event-csv artifacts/backtest/runs/<timestamp>/event.csv \
  --output artifacts/backtest/runs/<timestamp>/report.pdf
```

报告会自动包含收益图表（累计 realized PnL 曲线 + 策略 PnL 柱状图；数据不足时显示降级提示）。

Nightly 一键：

```bash
bash scripts/backtest_nightly_report.sh \
  --lookback-hours 24 \
  --market-limit 2000 \
  --ingest-limit 300 \
  --rolling-reject-top-k 2 \
  --config configs/soak.paper.yaml
```

Nightly 输出目录：`artifacts/backtest/nightly/<YYYY-MM-DD>/`
- `report.pdf`
- `summary.txt`（含 rolling `pos_win_rate/empty_windows` 与 canonical 别名 `positive_window_rate/empty_window_count`，以及 `range_h/coverage/overlap` 与 canonical 别名 `range_hours/coverage_ratio/overlap_ratio`、`coverage_label`、`rolling_reject_top_k`、主要拒单原因摘要）
- `rolling-summary.json`
- `run-<timestamp>/`（本轮 JSON/CSV/summary.md 工件）

`--rolling-reject-top-k` 语义：`0=disabled`（关闭拒单原因摘要输出），`N>0` 输出前 N 个原因（无数据时为 `none`）。

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
- live executor 已接入真实 CLOB HTTP 闭环：下单（`/order`）→ 回报同步（`live-sync`）→ 撤单（`live-cancel`）
- 凭据要求（二选一）：
  - `POLYMARKET_CLOB_HEADERS_JSON`（完整请求头 JSON）
  - `POLYMARKET_API_KEY` + `POLYMARKET_API_SECRET` + `POLYMARKET_API_PASSPHRASE`
- `REQUIRE_MANUAL_CONFIRM=true` 时，live 指令必须带 `--confirm-live`
- `KILL_SWITCH=true` 时所有新单拒绝（不影响已有单的状态同步/撤单）

## 24h Paper Soak Test

```bash
# 推荐使用专用配置（安全默认：paper + live=false）
bash scripts/paper_soak_24h.sh \
  --hours 24 \
  --interval-sec 300 \
  --max-signals-per-cycle 10 \
  --retry-max 3 \
  --config configs/soak.paper.yaml

# 查看实时状态
bash scripts/paper_soak_status.sh
```

产物目录：`artifacts/soak/paper-<timestamp>/`
- `soak.log`：完整运行日志
- `status/latest.json`：最新状态
- `status/history.jsonl`：每轮状态历史

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
- `docs/backtest-vs-paper-alignment-template.md`：回测 vs paper 对齐报告模板
- `docs/strategies.md`：S1/S2/S4/S8 策略细节
