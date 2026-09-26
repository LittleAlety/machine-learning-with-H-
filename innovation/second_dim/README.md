# Phase 6 stage_S3_second_dim — 追"第二维度"（v3 缺口的 ~0.03 eV）

预注册：`config/preregistered_phase6.yaml` → `stage_S3_second_dim`
（method：v3 SHAP 交互值排名 + 残差蒸馏；rule：机制命名须交互+蒸馏两臂互证）。

前情：Phase 5 路线 1（`innovation/bottleneck/`）判定 gap——k=2 非线性瓶颈比 v3
OOF 0.1134 eV 掉 ~0.031 eV，且潜维度退化为单一 group_wmean 方向，提示存在
SHAP 单特征排名未暴露的**非线性第二维度**。本阶段用两臂追它的性质。

## 协议（锁死，与 v3 完全一致）
- 数据 `data_processed/hstar_features_v3_84feat.csv`（1836 样本，84 特征），
  GroupKFold(5, groups=comp)，XGBoost（lr 0.03, depth 7, n 400, subsample 0.8）。
- 臂 (a) SHAP 交互：v3 全量拟合（seed 42），`booster.predict(pred_interactions=True)`，
  样本均值 |交互值| 去对角排序 → `interactions_top20.csv`。
- 臂 (b) 残差蒸馏：10 种子 OOF（MAE 0.1137±0.0004 eV，复现 v3 基线），
  r = y − OOF_mean（|r| 均值 0.098 eV，std 0.233 eV）；锁死小模型
  （depth≤3 树 / 标准化 Ridge/LogReg），同折 OOF 预测 |r|（幅度）与 sign(r)（符号）。

## 结果

### 臂 (a) Top5 交互对（mean |interaction|, eV）
| # | 特征对 | 强度 |
|---|---|---|
| 1 | **group_wmean × melting_point_max** | **0.0438** |
| 2 | group_wmean × group_mode | 0.0232 |
| 3 | melting_point_wmean × melting_point_max | 0.0097 |
| 4 | group_wmean × nd_unfilled_wmean | 0.0095 |
| 5 | melting_point_max × nd_unfilled_wmean | 0.0084 |

最强对（group_wmean × melting_point_max，0.0438 eV）是第 3 名起的 ~4.5 倍，
但它不是"group_wmean 之外"的新方向——它是 group_wmean 主方向的**条件化纹理**
（熔点族极端值调制 d 带代理族效应）。交互/主效应均强比见 verdict.json。

### 臂 (b) 残差可预测性
- 幅度 |r|：Ridge OOF R² = **0.097**（MAE 0.1015 vs 常数基线 0.1069，仅 ~5% 改善）；
  depth-3 树 OOF R² = **−0.023**（不可预测）。
- 符号 sign(r)：树 0.521 / LogReg 0.526，均**低于**多数类基线 0.558 → 符号完全不可预测。
- 蒸馏 Top 特征（树，全量归因）：melting_point_max (0.30)、mendeleev_number_mode (0.27)、
  mendeleev_number_wmean (0.16)、melting_point_range (0.15)、n_valence_max (0.07)。

### 命名判定（两臂互证规则）
- 臂 (a) Top20 族对最集中为 group × melting_point，但只占 4/20（< 锁死阈值 5/20）。
- 臂 (b) 残差仅幅度被 Ridge 弱线性预测（R²≈0.10），符号不可预测、树不可预测——
  不存在一个"缺失特征"能被小模型从残差里挖出来。
- **判定：`second_dim_name = "unidentified (interaction texture)"（未识别——交互纹理）**。
  机制性读法：第二维度不是某个缺失的物理特征，而是 v3 已在用、但单特征 SHAP
  排名看不见的**特征对交互纹理**（以 group_wmean × 熔点族极端值为主导项）；
  瓶颈模型压缩到 2 维时丢失的正是这部分纹理，与 Phase 5 gap 判定自洽。

## 产物与复现
```bash
python innovation/second_dim/run_second_dim.py   # 幂等，覆盖写全部产物
```
`run_second_dim.py`、`interactions_top20.csv`、`residual_distill.json`、
`verdict.json`、`second_dim.png`（300 dpi）、`run.log`、`README.md`。
