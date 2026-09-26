# deprecated/ — 废弃产物归档

本目录收录已被取代、不再被任何脚本或报告引用的历史产物，仅保留作审计追溯。
**请勿在复现管线或交付文档中引用本目录内容。**

## 归档清单

| 文件 | 迁移前路径 | 迁移日期 | 废弃原因 |
|---|---|---|---|
| `opt_search_results.csv`、`opt_sample_weights.csv`、`opt_stacking_compare.csv`、`opt_nested_cv.csv`、`opt_loeo_results.csv`、`opt_final_metrics.csv`、`opt_extval_results.csv`、`opt_extval_metrics.csv`、`opt_best_params.json` | `data_processed/opt_*` | 2025-09-09 | 第一轮 H\* 优化（scripts/12）草稿产物，搜索/评估采用**随机 KFold 口径**，与项目统一的 GroupKFold（规范化组成）零泄漏协议不一致；已由 GroupKFold 口径重做版取代，现行产物在 `outputs/opt_*`（同名但数值不同，勿混淆） |
| `08_optimization.md` | `notes/08_optimization.md` | 2025-09-09 | 上述随机 KFold 草稿对应的旧版优化报告；重做版为 `notes/12_optimization.md` |

## 对应关系（旧 → 现行）

- 旧报告：`deprecated/08_optimization.md` → 现行：`notes/12_optimization.md`
- 旧数据产物：`deprecated/opt_*.csv/json` → 现行：`outputs/opt_*.csv/json` + `outputs/opt_comparison.csv`（新增全线对比表）

> 注意：`deprecated/opt_best_params.json` 等与 `outputs/` 下现行文件**同名不同值**
> （旧版如 `max_depth=11`，现行重做版为 `max_depth=9`），引用时务必确认路径。
