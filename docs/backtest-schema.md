# Backtest Artifact Schema

当前回测导出工件（JSON + CSV）使用统一 `schema_version`，用于兼容解析。

- 当前版本：`1.0`
- 兼容策略：`major` 相同即兼容（`1.x` 互相兼容）
- 代码中已预留 v2 校验占位：`validate_backtest_json_artifact_v2`（用于双栈迁移演练）

## v1.x 兼容约定

1. 不删除已有字段
2. 新增字段仅追加（消费者应忽略未知字段）
3. 小版本（`1.0` -> `1.1`）允许新增可选字段
4. 大版本（`1.x` -> `2.0`）表示不兼容变更

## 消费端建议

- 先读取 `schema_version`
- 用 `parse_schema_version` / `assert_schema_compatible` 做版本检查
- 读取 JSON 可调用 `validate_backtest_json_artifact(payload)` 做 v1 结构校验
- 如需双栈校验，可用 `validate_backtest_json_artifact(payload, supported_major=None, validators={2: ...})` 按 major 分派
- 对 CSV 采用“已知字段优先 + 忽略未知列”
- 参考测试样本：`tests/fixtures/backtest/artifact_v1.json`、`artifact_v2.json`
- 迁移助手：`migrate_backtest_artifact_v1_to_v2(payload)`（CLI: `backtest-migrate-v1-to-v2`）
- 字段映射清单：`backtest_migration_v1_to_v2_field_map()`（CLI: `backtest-migration-map`）

## 字段映射与可逆性说明

- `schema_version` 在迁移中会强制写为 `2.0`（不可逆），原值保存在 `meta.source_schema_version`
- 其余核心字段采用 copy/deepcopy 映射，可用于审计对照

查看映射：

```bash
monomarket backtest-migration-map
monomarket backtest-migration-map --format json
monomarket backtest-migration-map --format json --out-json artifacts/backtest/migration-map.json
```

mapping artifact 含：`schema_version/kind/from_schema_major/to_schema_major/mappings/summary`。

## v2 Breaking Changes Checklist（草案）

在引入 `2.0` 之前，逐项确认：

1. **变更分类明确**：字段删除/重命名/语义变化已列出（不是仅新增字段）
2. **迁移脚本可用**：提供 `v1_to_v2`，并覆盖 JSON + CSV 样本
3. **双栈读取窗口**：读取端在一个发布周期同时支持 `1.x` 与 `2.x`
4. **回滚策略**：可回退到 `1.x` 输出（feature flag 或兼容开关）
5. **测试覆盖**：
   - `1.x` 兼容不回退
   - `2.x` 结构校验
   - `1.x -> 2.x` 迁移正确性
6. **文档同步**：runbook、字段字典、下游消费者告警窗口

## 迁移指南（v1.x -> v2 预案）

当需要进行不兼容调整（例如字段重命名、语义变化）时：

1. 回测导出改为 `2.0`
2. 同期提供迁移脚本（`v1_to_v2`）
3. 在一个发布周期内并行支持：
   - 读取端支持 `1.x` 与 `2.x`
   - 写出端默认 `2.x`（可选开关回写 `1.x`）
4. 迁移稳定后移除 `1.x` 读取支持
