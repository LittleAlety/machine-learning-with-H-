# Phase 4 Workstream L — L0 论证卫生包（结果片段）

预注册: `config/preregistered_phase4.yaml` workstream_L；红线 16/17/20。
测试面：EqV2 491 行 OOF（与 0.347 eV 零锚点基线同一键集合，已逐键核查）；
锚点只进训练折（红线 16）；lockbox `splits/lockbox_l0_anchors.json` 先于任何分析冻结（SHA256 留证）。

| 方案 | EqV2 MAE (eV) | 95% CI (bootstrap 10,000) | Spearman ρ |
|---|---|---|---|
| B1_mean_shift | 0.3492 | [0.3089, 0.3996] | 0.217 |
| B2_linear_shift | 0.1360 | [0.1056, 0.1777] | -0.242 |
| B3_domain_linear | 0.2293 | [0.1948, 0.2753] | 0.236 |
| B4_domain_gbdt | 0.1195 | [0.0878, 0.1615] | 0.512 |

**B4−B2 改善 = 0.0165 eV，95% CI [0.0031, 0.0267]** → **结局 B**：锚点本体论：跨库上限由锚点数量决定，GBDT 相对线性偏移无显著红利

参照：v3 零锚点跨库基线 0.347 eV；B4 对 Phase 3 I 既有 0.1226 eV 口径复现值 0.1195 eV。
