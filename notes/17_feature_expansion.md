# 17 — Round 4 特征扩展消融实验（Agent G）

日期：2026-09-09。脚本：`scripts/17_feature_expand.py`。种子 42；`site_spread` 禁入。
公平性契约（plan_round4.md）：基线 XGBoost_tuned + v2 原 36 特征 + GroupKFold(5, 规范化 comp)；
改善 >0.005 eV 才判"有效"；有效者补嵌套 CV + SHAP 防假阳性；阴性结果照常报告。

> 基线参数说明：契约的 0.1198 来自 **XGBoost_tuned**（lr=0.03, max_depth=7,
> n_estimators=400, subsample=0.8，见 `hstar_best_params.json`）。任务书所谓"出厂参数"
> 与 0.1198 不能同时成立——XGBoost_default 的 GroupKFold MAE=0.1296
> （`hstar_group_cv_results.csv`）。本实验以 tuned 参数为基线，实测精确复现
> **0.11978±0.00872** ✓。

## 1. 特征组构造

### (a) 位点特征组（site_H_official → 6 列多热）
- 来源：`hstar_facet_check.csv`（CatHub 官方校验位点元数据），与 v2 按 comp **1:1 完全对齐**。
- 缺失率 **15/1836 = 0.82%**（<20% 阈值，整组保留；缺失行 5 个类型列置 NaN 交 XGBoost，
  `site_missing=1`）。
- 解析出 7 种官方位点注解：top、top-tilt、bridge、bridge-tilt、hollow、hollow-tilt、4fold；
  按契约归桶为 `site_top / site_bridge / site_hollow_fcc / site_hollow_hcp / site_other /
  site_missing`（*-tilt 归入基础类型，4fold 归入 other；hollow 按 FCC/HCP 尾标区分）。
- 结构内方差充足（如 A1 中 site_top=1 的比例仅 0.839），**并非 structure one-hot 的重编码**。

### (b) mendeleev 扩展属性组（18 属性 × 5 统计 = 90 特征）
- 数据集实际含 **37** 个相关元素（任务书说 39，以实际为准）。
- 枚举 mendeleev 1.2.0 中除现有 6 属性外的 23 个数值型候选；
  `density / fusion_heat / evaporation_heat` 在 mendeleev 1.2.0 中**不存在**（已逐一确认）。
- 按"元素缺失率 >10% 弃用"规则弃用 5 个（明细见 `outputs/featexp_dropped_properties.csv`）：
  electron_affinity（10.8%，Cd/Hg/Mn/Zn）、thermal_conductivity（10.8%，Mn/Mo/Os/Y）、
  covalent_radius_bragg（62.2%）、gas_basicity（59.5%）、proton_affinity（59.5%）。
- 保留 18 个：boiling_point, melting_point, specific_heat_capacity, covalent_radius,
  covalent_radius_cordero, covalent_radius_pyykko, atomic_volume, dipole_polarizability,
  lattice_constant, en_ghosh, en_allen, atomic_weight, vdw_radius, vdw_radius_alvarez,
  metallic_radius, metallic_radius_c12, abundance_crust, atomic_radius_rahm。
- 按现有 wmean/max/min/range/wstd 五统计展开；组元缺失 → 该组成该属性五统计全置 NaN
  （共 2,500/165,240 = 1.5% 单元格，集中于含 Sn/Tc/La/Tl 的组成），XGBoost 原生处理。

## 2. 消融结果（同一批 GroupKFold(5, comp) fold）

| 变体 | 特征数 | MAE (eV) | Δ vs 基线 | 逐折配对差 min | 判定 |
|---|---|---|---|---|---|
| baseline_v2_36f | 36 | 0.11978±0.00872 | — | — | — |
| plus_site | 42 | **0.10229±0.00820** | **+0.01749** | +0.0121 | 数值"有效" |
| plus_mendeleev | 126 | 0.12096±0.00798 | −0.00118 | −0.0082 | **无效** |
| plus_site_mendeleev | 132 | 0.10373±0.00720 | +0.01604 | +0.0068 | 数值"有效"但逊于单独 site |

`outputs/featexp_ablation.csv`。mendeleev 组单独加入**无改善**（还略变差 0.0012 eV）；
组合组不如单独位点组。

## 3. 防假阳性验证（对两个"有效"变体）

**嵌套 CV**（外 GroupKFold5 / 内 GroupKFold4，内层随机搜索 30 组，基线同样重调参作对照，
`outputs/featexp_nested_cv.csv`）：

| 变体 | 嵌套 CV MAE (eV) |
|---|---|
| baseline（重调参） | 0.1218±0.0123 |
| plus_site（重调参） | **0.1007±0.0077**（Δ=+0.0211） |
| plus_site_mendeleev（重调参） | 0.1051±0.0076（Δ=+0.0167） |

位点组的改善在重调参对照下依然成立且更大 → **不是固定参数的调参假阳性**。

**SHAP**（全量训练，TreeExplainer；`outputs/featexp_shap_importance_*.csv`）：
plus_site 中 site_hollow_hcp 排第 4、site_top 排第 8、site_bridge 第 19、site_hollow_fcc 第 21
——新特征**确有真实贡献**，不是噪声占位。

## 4. 关键反证：位点增益不可移植（决定采纳建议的核心证据）

数值"有效"之后补做了两个可移植性/机制检验，结果推翻了采纳价值：

1. **结构众数规则降级测试**：把逐组成的位点特征替换为"structure→众数位点模式"
   （页面唯一能实现的规则推断），GroupKFold MAE 回到 **0.1198，与基线完全一致**。
   → 0.0175 eV 的增益 **100% 来自逐组成特异模式**，几何可枚举的部分零信号。
2. **n_configs 中介检验**：`n_configs_official`（该组成官方构型数）单独作为特征即可把
   MAE 从 0.1198 降到 0.1121（+0.0077）；corr(n_configs, 位点类型数)=0.78；
   且 site+n_configs（0.1022）相比 site（0.1023）无额外增益。
   → 位点特征携带的信息包含并强于"数据可用性指纹"。目标值是逐组成取最低吸附能，
   构型数本身通过极值统计向下偏置 E_min——这是一个**聚合层面的数据集伪迹通道**。

机制解释：site_H_official 记录的是"CatHub 历史上为该组成计算/标注过哪些位点构型"，
其中哪些构型存在、哪些缺失、哪些弛豫成 tilt，依赖该组成的计算历史与收敛命运，
与吸附能存在间接相关；但**对任意新组成，这一切都无法预先知道**。

## 5. 诚实结论与采纳建议

- **mendeleev 扩展属性组：无效（Δ=−0.0012 eV），不采纳。** 现有 6 属性已覆盖其可解释方差。
- **位点特征组：数值上"有效"（固定参数 +0.0175，嵌套 CV +0.0211，SHAP 排位前列），
  但可移植性检验证明增益 100% 来自不可泛化的逐组成元数据（部分还是构型数伪迹），
  按任务规则标注「仅训练集可用，页面不适用」。建议：不采纳进 H\* 主线模型。**
  若主代理仍想利用，只能作为训练集内部分析特征，且必须披露 n_configs 伪迹通道；
  部署形态与训练特征分布系统性不一致（实测降级后与基线无差）。
- **最终建议：H\* 主线保留现有 36 特征 XGBoost_tuned 模型。** 本轮特征扩展实验总体阴性。

## 6. JS 可移植性声明

| 特征组 | 可移植性 | 结论 |
|---|---|---|
| 位点特征（6 列） | **不可泛化**。任意新组成的官方位点集合无法推断——几何可枚举部分等价于 structure/facet（已有 one-hot），实测逐组成特异部分才是信号来源；只能用结构众数规则近似，而该近似实测零增益 | **仅训练集可用，页面不适用**（未被采纳，web 前端无需改动） |
| mendeleev 扩展属性（18×5） | 技术上完全可移植：37 元素 × 18 属性的静态表可导出 elements.json，
JS 端按化学计量做 wmean/max/min/range/wstd 即可 | 未被采纳（无效），无需导出 |

## 7. 产物清单

- `scripts/17_feature_expand.py`（新增，本轮唯一新增脚本）
- `outputs/featexp_ablation.csv`（消融主表）
- `outputs/featexp_dropped_properties.csv`（mendeleev 弃用属性明细）
- `outputs/featexp_nested_cv.csv`（嵌套 CV 逐折结果，含每折最优参数）
- `outputs/featexp_shap_importance_plus_site.csv`、`featexp_shap_importance_plus_site_mendeleev.csv`
- 未改动任何共享产物（v2 数据集、候选排序、web/assets、scripts 01–15 输出均未触碰）。
