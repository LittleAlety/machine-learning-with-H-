# 多吸附种统一模型报告（scripts/09 + 10，SPEC 第 7 节）

## 1. 数据构建（09_multiads_data.py）

- 数据源：`data_raw/MamunHighT2019_adsorption.json.gz`（catbench.org 镜像，与 05 同源同口径），
  全量 **45,130** 条吸附反应，覆盖 **15 种吸附种标签**：H, C, N, O, S, 2H, 2N, 2O,
  CH, CH₂, CH₃, NH, OH, SH, H₂O。
- 解析与清洗完全复用 H* 线口径：前缀 → 元素计量字典 → comp 规范化（元素字母序+gcd 约简）；
  structure/facet 由 12 原子超胞计量模式推断（M12→A1(111)，A9B3→L12(111)，A6B6→L10(101)）；
  同一 (comp, structure, adsorbate) 的 `_N` 后缀条目为不同吸附构型，**取最低 E_ads**
  （最稳定位点），n_raw 记录构型数。
- 去重后 **16,499** 行，1,916 个唯一规范化组成（多吸附种共表面）；能量范围
  [−12.90, +11.20] eV（远宽于 H* 子线的 [−2.31, +3.75]，因 C/N/O/S 为强键合多价吸附种）。

### 各吸附种条数（去重前 → 去重后）

| adsorbate | 吸附原子 | 原始记录 | 去重后 |
|---|---|---|---|
| H | H | 7,048 | 1,836 |
| S | S | 5,812 | 1,806 |
| 2H | H | 1,806 | 1,802 |
| N | N | 5,696 | 1,796 |
| O | O | 5,689 | 1,795 |
| C | C | 5,291 | 1,776 |
| 2N | N | 1,761 | 1,761 |
| 2O | O | 1,680 | 1,680 |
| NH | N | 1,558 | 352 |
| CH | C | 1,508 | 347 |
| SH | S | 1,770 | 342 |
| H₂O | O | 1,534 | 326 |
| CH₂ | C | 1,458 | 319 |
| CH₃ | C | 1,284 | 318 |
| OH | O | 1,235 | 243 |

（明细：`data_processed/multiads_adsorbate_counts.csv`；图：`figures/multiads_adsorbate_counts.png`）
原子吸附种（H/C/N/O/S、2H/2N/2O）几乎每表面一条（去重后 ~1,680–1,836）；复合吸附种
（CHx/NH/OH/SH/H₂O）构型离散大，取最低值后每表面一条（243–352 条，覆盖表面子集较小）。

H* 子集去重后 1,836 行，与 05 的 hstar_dataset_v1.csv（1,836 个唯一表面）一致，口径自洽。

## 2. 特征化（复用 06 + 吸附种特征）

- 组成/结构描述符：importlib 按路径直接加载 `06_hstar_features.py` 的
  `parse_comp / build_element_table / wmean / wstd / PROPS / STATS`，保证与 H* 线**逐函数一致**：
  n_elements + 6 元素性质（en/radius/group/period/ie1/d_el）× 5 统计
  （wmean/max/min/range/wstd）+ structure/facet one-hot。mendeleev 全齐，手工兜底未启用。
- 吸附种特征（新增 9 个）：
  - `ads_atom`：与表面成键的原子（CHx→C，NH→N，OH/H₂O→O，SH→S，2X→X），one-hot 5 列；
  - `ads_en / ads_radius / ads_group`：吸附原子的 Pauling 电负性 / 原子半径 / 族号（mendeleev）；
  - `n_ads_units`：吸附单元数（"2X*" 解离吸附=2，其余=1），纯结构信息，区分 H* 与 2H*。
- **泄漏防控**：未引入任何由标签能量派生的特征（site_spread 教训，见 notes/05）；
  multiads_dataset.csv 中连 site_spread 分析列都不再生成。
- 输出 `multiads_features.csv`：16,499 行 × 52 列（45 个建模特征 + 7 个标识列）。

## 3. 统一模型性能（10_multiads_model.py）

XGBoost 小网格与 07 完全相同（n_estimators×max_depth×learning_rate×subsample，24 组）；
最优参数 {lr=0.1, max_depth=7, n_estimators=400, subsample=1.0}。
随机 5-fold CV（种子 42）+ GroupKFold（groups=规范化 comp，整组跨吸附种留出）：

| 模型 | 随机 CV MAE (eV) | 分组 CV MAE (eV) | 随机 R² | 分组 R² |
|---|---|---|---|---|
| Ridge（线性基线） | 1.224 ± 0.026 | 1.224 ± 0.014 | 0.698 | 0.698 |
| XGBoost_default | 0.344 ± 0.011 | 0.394 ± 0.008 | 0.962 | 0.950 |
| **XGBoost_tuned** | **0.316 ± 0.010** | **0.363 ± 0.013** | **0.964** | **0.953** |

注意能量跨度 24.1 eV（含强键合吸附种），MAE 绝对值不可直接与 H* 线（跨度 6.1 eV）比；
R²=0.96 且显著优于线性基线，满足验收口径。

### 与 H* 单吸附种模型对比（逐吸附种随机折 OOF MAE，eV）

| adsorbate | n | MAE | | adsorbate | n | MAE |
|---|---|---|---|---|---|---|
| **H** | 1,836 | **0.104** | | 2O | 1,680 | 0.382 |
| S | 1,806 | 0.191 | | SH | 342 | 0.409 |
| 2H | 1,802 | 0.216 | | 2N | 1,761 | 0.438 |
| N | 1,796 | 0.226 | | C | 1,776 | 0.472 |
| O | 1,795 | 0.255 | | NH | 352 | 0.549 |
| OH | 243 | 0.280 | | H₂O | 326 | 0.585 |
| CH₂ | 319 | 0.314 | | CH₃ | 318 | 1.080 |
| CH | 347 | 0.377 | | | | |

- **统一模型在 H* 子集上的随机折 OOF MAE = 0.104 eV，优于 H* 专线模型的 0.120 eV**
  （07 去泄漏后口径）——多任务效应：其余吸附种提供的组成-能量结构信息对 H* 预测有
  正迁移。**但需注意信息对等口径**：随机折中同一表面的其他吸附种能量大概率出现在
  训练折内，H* 子集 OOF 等价于"已知同表面其他吸附种能量"的**插值场景**；而在
  GroupKFold(comp)（整组留出、对**全新表面**的外推场景）下独立复测，统一模型 H* 子集
  OOF MAE = **0.152 eV**，反而差于 H* 专线模型的 0.120 eV（终审独立实测：GroupKFold
  OOF 预测按 adsorbate=H 子集取 MAE）。**正确结论：统一模型的优势仅存在于"已知同表面
  其他吸附种能量"的插值场景（随机折 0.104）；对全新表面的外推场景，H* 专线模型仍是
  最优（0.120 vs 0.152），H* 主线地位不变。**
- 难度排序符合化学直觉：单原子吸附（H/S/N/O）最易；分子型/多构型吸附种（CH₃、H₂O、NH）
  最难（吸附几何自由度大，纯组成+结构 one-hot 无法分辨位点几何）；2N/2O 能量跨度大
  （17–18 eV），绝对误差随之放大。
- 分组 CV 相对随机 CV 的退化（0.316→0.363，+15%）明显小于单吸附种线的同族退化，
  因同一组成的其余吸附种能量在训练折中可见，等价于"已测该表面其他吸附能"的信息条件。

## 4. SHAP 分析（figures/multiads_shap_summary.png，种子 42 抽样 5,000 行）

Top-5（mean |SHAP|, eV）：

| 特征 | mean |SHAP| | 化学解读 |
|---|---|---|---|
| ads_group | 1.367 | 吸附原子族号（H=1 / C=14 / N=15 / O,S=16），本质是吸附种身份的最强编码，决定能量量程 |
| n_ads_units | 0.945 | 解离吸附（2X*）使总能量大幅负移（两根 M–X 键），区分 H* 与 2H* 等 |
| group_wmean | 0.672 | 表面组成 d 带填充度代理（同 H* 线 top-1） |
| ads_radius | 0.618 | 吸附原子尺寸 → 吸附位几何偏好（on-top vs 空位）与键强 |
| d_el_wmean | 0.378 | 表面 d 电子数（d 带理论核心描述符） |

解读：统一模型的首要任务被 SHAP 如实反映——先识别"吸附的是什么"（ads_group、n_ads_units、
ads_radius），再用表面描述符（group_wmean、d_el_wmean）调制强弱，与 H* 单吸附种线的
主导特征（group_wmean/d_el_wmean 居首）衔接一致，吸附种特征与组成特征各司其职。

## 5. 标度关系归纳（核心结论）

五种原子吸附种（H/C/N/O/S）在 **1,657** 个表面上能量齐备，两两相关矩阵
（`multiads_scaling_corr.csv`，图 `figures/multiads_scaling_matrix.png`）：

Spearman（下三角）/ Pearson（括号内）：

| | H | C | N | O | S |
|---|---|---|---|---|---|
| H | 1 | | | | |
| C | 0.80 (0.81) | 1 | | | |
| N | 0.88 (0.83) | 0.81 (0.83) | 1 | | |
| O | 0.82 (0.75) | 0.63 (0.64) | 0.92 (0.90) | 1 | |
| S | 0.90 (0.87) | 0.85 (0.85) | 0.94 (0.93) | 0.87 (0.86) | 1 |

以 H* 为锚的标度线（`figures/multiads_scaling_examples.png`，
`multiads_scaling_vs_hstar.csv`）：

| 配对 | n | Spearman | Pearson | 斜率 | 截距 (eV) |
|---|---|---|---|---|---|
| O* vs H* | 1,765 | 0.812 | 0.750 | 2.25 | −0.25 |
| C* vs H* | 1,743 | 0.801 | 0.794 | 2.00 | +3.03 |
| N* vs H* | 1,761 | 0.880 | 0.822 | 2.51 | +0.31 |
| S* vs H* | 1,772 | 0.895 | 0.862 | 1.62 | −1.48 |

核心发现：

1. **经典标度关系命题在双金属高通量数据上定量成立**：原子吸附种两两 Spearman
   均 ≥ 0.80，唯二例外为 C*–O*（0.63）与 H*–C*（0.799，恰低于 0.80 临界）。
   "测一种吸附能即可近似预测另一种"——对同一表面，
   E_ads(X*) 随 E_ads(H*) 单调共变，与 Abild-Pedersen 等提出的吸附能标度关系
   （以 H*/N* 为描述符）一致，且本数据集把该结论从纯金属/氮化物扩展到 1,916 种有序双金属。
2. **最强相关：N*–S*（ρ=0.94）、N*–O*（0.92）、S*–H*（0.90）、N*–H*（0.88）**；
   即亲硫/亲氮/亲氧性高度同源——都由表面 d 态填充与给电子能力驱动。
3. **最弱相关：C*–O*（Spearman 0.63）**。C* 偏好多空位强共价键合（渗碳倾向），
   O* 偏好电荷转移型键合，位点偏好差异最大（散点图中 C* vs H* 的离群点即来自此）。
   这意味着"用 O* 能量预测 C*"误差最大，反之亦然——对 Fischer-Tropsch 等 C/O 共存
   体系的催化剂筛选，两描述符近似独立，反而提供了一定的解耦设计空间。
4. **斜率均 >1（1.6–2.5）**：X* 能量对表面化学的敏感性是 H* 的 1.6–2.5 倍
   （多价吸附种成键数多，d 带移动的响应被放大）。H* 是"低增益探针"，N* 是"高增益探针"——
   用 H* 能量做初筛会低估表面间真实差异，这是单一 HER 描述符外推的系统性风险。
5. 工程含义：在这些有序双金属表面上，H* 与 O*/N*/S* 吸附**难以独立调变**
   （ρ≈0.8–0.9）——强化氢结合的表面几乎必然强化氧/硫结合，解释了 HER 催化剂
   抗硫中毒/抗氧化设计的内在困难；想要打破标度关系需引入本数据集未覆盖的自由度
   （配体效应之外的应变、缺陷、覆盖度效应等）。

## 6. 局限

- **能量参考态跨吸附种不可直接比**：各反应的 ref_ads_eng 以各自气相分子为零点
  （H₂/CH₄/NH₃/H₂S 等），E_ads 绝对值仅在同一吸附种内有意义；统一模型正是靠
  ads_group/ads_en 等特征吸收这一系统性偏移，SHAP 中 ads_group 居首部分源于此。
- **structure/facet 由计量模式推断**（镜像版无 atoms_json），复合吸附种的最稳定构型
  位点信息丢失，CH₃/H₂O 等 MAE 偏高与此直接相关。
- GroupKFold 按 comp 分组，同一 comp 的不同吸附种被同时留出；实际部署中"已测同表面
  其他吸附种"的场景更接近随机 CV，0.316 eV 与 0.363 eV 可视作该场景区间的两端。
  对 H* 子集同理：随机折 OOF 0.104 eV 属"已知同表面其他吸附种能量"的插值口径，
  GroupKFold(comp) 下统一模型 H* 子集 OOF MAE = 0.152 eV（终审独立实测），
  高于 H* 专线模型的 0.120 eV——全新表面外推场景仍应使用 H* 专线模型。
- 标度关系基于"最低 E_ads"口径，位点多样性（site_spread）信息被压缩；散点中残余
  离散部分来自位点偏好差异，非纯噪声。
- 2H/2N/2O 的 E_ads 为两个吸附单元的总能量，与单原子吸附不在同一标度上，
  标度矩阵仅含单原子吸附种（H/C/N/O/S）。

## 7. 产物清单

- 脚本：`scripts/09_multiads_data.py`、`scripts/10_multiads_model.py`
- 数据：`multiads_dataset.csv`（16,499 行）、`multiads_features.csv`（52 列）、
  `multiads_adsorbate_counts.csv`、`multiads_model_compare.csv`、
  `multiads_per_adsorbate_mae.csv`、`multiads_xgb_grid_results.csv`、
  `multiads_best_params.json`、`multiads_shap_importance.csv`、
  `multiads_scaling_corr.csv`、`multiads_scaling_vs_hstar.csv`
- 图：`figures/multiads_adsorbate_counts.png`、`multiads_shap_summary.png`、
  `multiads_scaling_matrix.png`、`multiads_scaling_examples.png`（均 300 dpi、暖色系）
