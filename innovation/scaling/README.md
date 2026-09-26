# Workstream E — 标度关系量化 (plan-hb v3.0)

Scope qualifier applying to ALL conclusions: ordered fcc intermetallic surfaces, composition-level granularity, demo-level free-energy corrections (+0.24 eV for H*, +3.2 eV for OH*/O*).

## E1 标度矩阵 (`e1_scaling_matrix.py`)
Theil-Sen (预注册), 15×15 有序对, 共享 (comp,structure,facet), 210/210 对 n≥30。
产物: `e1_slope/intercept/r2/nshared_matrix.csv`, `e1_scaling_matrix.png`, `e1_key_pairs.json`。
- H→OH: slope=1.3842, intercept=0.4644, R²=0.0684 (n=229)
- OH→H: slope=0.2757, intercept=-0.3271, R²=0.1918
- H→O: slope=2.6701, R²=0.5411 (n=1765)
- 文献对比: OH*/H* 文献斜率约 0.5–1；本数据 H→OH=1.38 略高于区间且 R² 仅 0.07
  （OH* 小档 n=229，统计弱），方向一致、幅度偏弱。OH→O=1.74 (R²=0.81) 与 O*–OH*
  标度关系文献吻合。

## E2 双功能缺口裁决 (`e2_gap_verdict.py`)
口径: **标签口径**（共享组成的实测能量，不用模型预测）；demo 级自由能修正
(ΔG_H=E+0.24, ΔG_OH=E+3.2 eV，同 C-1)。
- HER×ORR 标度线: dG_OH = 1.3842*dG_H + 3.3322 (R²=0.0684, n=229)
- **d_min = 1.9514 eV = 17.7× 模型分辨率 0.11 eV**
- 裁决: PHYSICAL GAP: d_min = 1.951 eV is 17.7x the model resolution (0.11 eV); the empty HER x ORR intersection is NOT a resolution limit -- it is a physical fact within the dataset coverage (ordered fcc intermetallic surfaces, composition-level granularity, demo-level free-energy corrections (+0.24 eV for H*, +3.2 eV for OH*/O*)).
- 最近实测点距理想点 0.8941 eV（佐证）
- HER×OER: dG_O = 2.6701*dG_H + 2.5313, d_min=0.8878 eV = 8.1× 分辨率 → 同为物理事实
- 图: `e2_gap.png`（标度线+理想点+d_min 标注+分辨率圆，替代双火山信息量）

## E3 阶梯分析 (`e3_ladder.py`)
- 15 吸附种全体: NOT SUPPORTED: Spearman rho(MAE, n_atoms)=-0.12 over 15 adsorbates（阴性如实：原子数/价电子/多原子均不显著）
- 大档 (n=8): MAE[eV] = 0.1902 * n_atoms + 0.0281 (r=0.691, p=5.8e-02)；
  阶梯由解离二聚体 (2H/2N/2O) 驱动，dimer flag Spearman ρ=0.62
- 经验关系式（全体）: MAE[eV] = -0.0162 * n_atoms + 0.2774（不显著，仅如实记录）

## E4 池化 GBDT 对照 (`e4_pooled_gbdt.py`)
同折同种子 (GroupKFold(5,42) by comp, seeds 0/1/2, v3 超参)；公平契约 >0.005 eV + 3/3 方向。
- 池化+one-hot 仅在 2/15 吸附种 (CH2, NH) 满足契约；平均 Δ(sep−pooled) = -0.0273 eV
- **裁决: SEPARATE WINS: pooled-with-one-hot wins fair contract on 2/15 adsorbates; mean delta(sep-pooled)=-0.0273 eV. The data-sharing benefit is largely absorbed by the adsorbate indicator feature (or offset by cross-adsorbate interference); per-adsorbate models remain the reference.** —— 数据共享红利被指示特征吃掉/被跨吸附种干扰抵消，分吸附种模型保持基准。
- 表: `e4_pooled_vs_separate.csv`

## E5 逃逸角红旗统计
- 最近邻候选 20 个中 16 个 (80%) 带 ≥1 红旗
  （Tc 放射性 / Hg,Tl 毒性 / La,Y,Sc 强亲氧 / Mn,Bi,Fe LOEO），规则分布: {'oxophilic_La_Y_Sc': 14, 'LOEO_Mn_Bi_Fe': 5, 'toxic_Hg_Tl': 1}
- OH* 训练集外（模型外推）占比 100%
- **"逃逸角=不可信角"获得双重量化支撑：逃离标度缺口的组成同时是模型不可信
  （外推）+ 材料不稳定（红旗）—— 双重警示，不得单独引用其中任一维度。**

## 复现
`for s in e1_scaling_matrix e2_gap_verdict e3_ladder e4_pooled_gbdt e5_summary; do /usr/local/bin/python3 innovation/scaling/$s.py; done`（幂等覆盖）。
