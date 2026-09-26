# Phase 3 Workstream I — 跨泛函不变性分析（Functional Invariance）

预注册依据：`config/preregistered_phase3.yaml` → `workstream_I_invariance`
（红线 15：先写判读规则后跑实验；公平契约 min_gain 0.005 eV / 10-10 seeds；
seeds = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]）。

复现：`python -u innovation/functional_invariance/run_invariance.py`（幂等，覆盖同名产物；
依赖 mendeleev/xgboost/torch；内置口径自检：重算主域 1836 行 84 特征与
`data_processed/hstar_features_v3_84feat.csv` 快照逐值比对，0 失配方继续）。

## 三域（特征 = v3 84 维描述符，同一 mendeleev 口径）

| 域 | 来源 | n | 唯一组成 |
|---|---|---|---|
| Mamun / BEEF-vdW（主域） | `data_processed/hstar_features_v3_84feat.csv` | 1836 | 1836 |
| EqV2 / RPBE | `data_raw/eqv2_361.csv` + `eqv2_500.csv`（复用 scripts/11 口径：规范化 comp、同表面取最低 E_ads、structure one-hot=0、facet_111） | 491 | 115 |
| CatHub / PBE 系 | `data_processed/extval2_homogeneous.csv` 中 `dftFunctional==PBE`（scripts/14 聚合口径） | 203 | 46 |

兜底说明：mendeleev 1.3.0 中 Sn 熔点因同素异形体返回 None，按主域快照实测
505.08 K 手工兜底（与 06/23 脚本 HAND_LOOKUP 传统一致）；Nd/Sm 无 group 号 →
对应统计置 NaN（XGBoost 原生处理，同 23 号脚本口径）。

## i1 重叠组成盘点（"罗塞塔"锚点对）

- 三域共同组成 **3 个**：FeNi3、Ir3Ti、MnNi3（→ `i1_overlap.csv`，含各域 E_ads 均值/极值）。
- 两两重叠：Mamun∩EqV2 = 68；Mamun∩CatHub = 34；EqV2∩CatHub = 3（`i1_pair_counts.csv`）。
- 锚点对上清晰可见泛函系统性偏移，如 FeNi3：BEEF-vdW −0.300 / RPBE −0.389 / PBE −0.568 eV
  （PBE 过束缚、RPBE 欠束缚，符合已知泛函趋势）。

## i2 单描述符跨域稳定性（`i2_stability.csv` + `i2_stability_heatmap.png`）

逐域拟合每个描述符与 E_ads 的一元斜率/Pearson r，按跨域斜率 std 排序
（structure/facet one-hot 在外部域为常量，不参与排序）。

- **最稳定 Top5**（全部为 melting_point 族）：melting_point_range / _mode / _min / _wstd / _max。
- **最不稳 Bottom5**（全部为电负性族）：en_range、en_mode、en_min、en_wstd、en_wmean
  —— en_wmean 斜率 Mamun +0.80 / EqV2 −0.22 / CatHub +0.42 eV/单位，发生**符号翻转**。
- 84 特征中 78 个三域可比，其中 **74 个存在跨域符号翻转** —— 单描述符层面几乎不存在
  跨泛函不变的单调关系，这是"描述符–泛函耦合"的直接证据。

## i3 三方案同台（同折同种子 KFold5；XGBoost 锁死出厂 tuned 参数，无内层调参）

评估口径：三域 pooled 5-fold OOF（10 种子 × 同折），报告 EqV2 行的 OOF MAE 与
Spearman ρ；改善相对 v3 出厂模型零重训 EqV2 基线 **0.347 eV**
（`data_processed/extval_metrics.csv`, ALL=0.3469996）。

| arm | EqV2 MAE (mean±sd) | gain vs 0.347 | 10/10 gain≥0.005 | ρ mean |
|---|---|---|---|---|
| naive_pooled | 0.1309 ± 0.0037 | +0.216 | ✅ | 0.456 |
| **domain_indicator** | **0.1226 ± 0.0050** | **+0.224** | ✅ | **0.485** |
| IRM_lite（线性 IRMv1, λ=100, 2000 epochs, 锁死） | 0.2299 ± 0.0045 | +0.117 | ✅ | 0.097 |

## 结局判定：**A**（`i3_verdict.json`）

最优 arm = domain_indicator：MAE 0.123 ≤ 0.25 ✅，改善 0.224 > 0.05 ✅，ρ 0.485 ≥ 0.35 ✅，
且公平契约 10/10 种子 gain ≥ 0.005 eV（最小 gain 0.217）✅ → 存在可利用的"不变核 +
域偏移"结构，支持协议/域感知建模路线。

机制解读（讨论用，非判读依据）：
- i2 显示单描述符层面**无**跨域不变量（74/78 符号翻转），但 i3 显示树模型可从
  少量域内样本学到域条件化修正 —— 二者一致：不变性存在于**高维联合映射**而非任何
  单轴描述符；domain_indicator 以极小代价（2 个哑变量）拿下最大改善，说明跨库误差的
  主要成分是**加性域偏移 + 交互修正**，而非特征空间失效。
- IRM_lite 幅度项达标（MAE 0.230、gain 0.117）但排序崩坏（ρ=0.097）：线性函数类上
  强行惩罚环境间梯度一致性，把 i2 中符号翻转的电负性族压成零斜率，牺牲了域内排序 —
  与"线性不变核不存在"的 i2 证据互洽。
- 注意口径：本评估为 pooled CV（EqV2 行出现在训练折中），结论"存在可学习的不变核"
  以**小比例域内锚点可用**为前提；零锚点纯零样本跨库仍受描述符–泛函耦合伪相关限制
  （出厂模型 0.347 eV 即此情形）。

## 产物清单

- `run_invariance.py`（幂等主脚本，含口径自检）、`run.log`
- `i1_overlap.csv`、`i1_pair_counts.csv`
- `i2_stability.csv`（84 特征 × 逐域斜率/r + 跨域 std + 排序）、
  `i2_stability_heatmap.png`（300 dpi，YlOrBr 暖色系，英文标注）
- `i3_fold_metrics.csv`（3 arms × 10 seeds 逐种子）、`i3_summary.csv`、
  `i3_verdict.json`（判读规则逐条核对）、`invariance_summary.json`
