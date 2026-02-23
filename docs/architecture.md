# Architecture (MVP)

```text
[gamma/data/clob APIs]
         |
         v
   data ingestion
         |
         v
 normalized sqlite storage
         |
         +--> signal engine (S1/S2/S4/S8)
         |          |
         |          v
         |      opportunities/signals
         |          |
         +------> execution router (paper/live)
                    |
                    +--> unified risk guard
                    |
                    v
                 orders/fills
                    |
                    v
              pnl + metrics report
```

## 模块

- `monomarket.data`
  - `clients.py`: gamma/data/clob HTTP 读取 + 归一化
  - `ingestion.py`: source ingest 入口（支持 gamma/data/clob/all）

- `monomarket.db`
  - `storage.py`: SQLite schema + CRUD
  - 表：`markets/signals/orders/fills/positions/switches/ingestion_runs`

- `monomarket.signals`
  - 可插拔策略引擎
  - 已实现：
    - `s1`: 跨平台价差扫描与排名（semi-auto 执行 payload）
    - `s2`: NegRisk 组合重平衡机会检测
    - `s4`: 低概率 Yes 篮子与分层挂单参数
    - `s8`: 高胜率 No Carry + 尾部对冲配比

- `monomarket.execution`
  - `router.py`: paper/live 路由、开关治理、统一风控
  - `risk.py`: 全局止损 + 策略上限 + 事件上限 + 熔断
  - `paper.py`: 本地撮合（立即成交）
  - `live.py`: 预留 live 执行通道（无凭据则拒单）

- `monomarket.pnl`
  - `tracker.py`: 持仓与已实现/未实现 PnL
  - `metrics.py`: fill rate/rejection rate/max drawdown 等指标
