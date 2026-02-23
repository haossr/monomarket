# Strategy Notes (S1/S2/S4/S8)

## S1 跨平台价差扫描

- 输入：同 canonical/event 的多来源市场（gamma/data/clob）
- 输出：机会排名（spread、流动性、置信度）
- 当前执行：先输出 scanner + ranking；payload 提示半自动双腿执行

## S2 NegRisk 重平衡

- 识别 `neg_risk=true` 且同 event 的市场集合
- 计算 `sum(yes_price)` 与 1 的偏离
- 若偏离超过容差，生成多腿 rebalance 信号

## S4 低概率 Yes 篮子

- 选择 yes 价格在 `[yes_price_min, yes_price_max]`
- 根据价格与流动性分层（A/B/C）
- 每个标的给出分层挂单（ladder_prices）与仓位参数

## S8 高胜率 No Carry + 尾部对冲

- 主仓：筛选低 yes（高 no 胜率）市场，建立 NO carry
- 对冲：从超低 yes 池里选 tail hedge，按 `hedge_budget_ratio` 配比
- 输出 payload 包含主仓 + 对冲建议

## 参数位置

策略参数在 `configs/config.yaml` / `configs/config.example.yaml` 的 `strategies.*` 下。
