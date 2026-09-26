# ai_surface_cat：可解释机器学习表面吸附能预测与催化位点筛选

基于公开 DFT 吸附能数据库，构建"组成/结构描述符 + 树模型 + SHAP"的
可解释机器学习管线，预测表面吸附能并按 Sabatier 原则筛选候选催化表面。

- **主线（H\*，HER 方向）**：Mamun 2019 高通量双金属数据集 → HER 催化位点筛选
- **方法验证线（CO\*/O\*）**：CMR CatApp → 管线方法学验证
- **扩展线**：多吸附种统一模型（15 种吸附种）+ 两轮外部验证 + Catalysis-Hub 官方库交叉核对

## 最终指标（终审修复后）

| 管线 | 最优模型 | 随机 5-fold MAE | GroupKFold MAE | LOEO MAE | R²（随机） |
|---|---|---|---|---|---|
| H* 主线（1836 表面） | XGBoost_tuned | **0.120 ± 0.011 eV** | 0.120 ± 0.009 | 0.155（汇集） | 0.817 |
| CO*/O* 验证线（937 行） | XGBoost_tuned | **0.227 ± 0.017 eV** | 0.219 ± 0.017（规范化组成） | — | 0.890 |

- H* 线 LOEO 最差折：Mn 0.482 / Fe 0.283 / Bi 0.279 eV（明细 `hstar_loeo_results.csv`）
- 筛选产物：`candidate_rankings_hstar.csv`（HER，ΔG_H* = E_ads + 0.24 eV）、`candidate_rankings.csv`（CO）

## 数据来源

### 主线：H*（Mamun 2019，经 CatBench 整理）
- 文件：`data_raw/MamunHighT2019_adsorption.json.gz`（gzip 压缩，只读）
- 实际下载来源：`https://catbench.org/benchmark/MamunHighT2019.json.gz`（CatBench 官方镜像；
  原 Zenodo 记录 `https://zenodo.org/records/17157086/files/MamunHighT2019_adsorption.json?download=1`
  在实施期间持续 504，05 脚本会先试 Zenodo 再自动回退镜像）
- 文献：Mamun et al., *Sci. Data* 6:76, 2019（原数据 DOI: 10.1038/s41597-019-0082-4，CC-BY-4.0）；
  CatBench 整理版见 Moon et al., *Cell Rep. Phys. Sci.* 2025（Zenodo concept DOI: 10.5281/zenodo.17157085）
- 规模：45,131 条吸附反应；本项目取稀释 H\*（0.5H₂(g)+\*→H\*）7,048 条 → 1,836 个唯一表面（37 种金属）
- 能量语义：`ref_ads_eng` = E(H\*slab) − E(slab) − 0.5·E(H₂)，**负值 = 放热吸附**（已抽样验证）
- 注意：镜像版不含 atoms_json，structure/facet 由 12 原子超胞计量模式推断
  （M12→A1(111)、A9B3→L1₂(111)、A6B6→L1₀(101)），详见 `notes/05_hstar_report.md`

### 验证线：CO*/O*（CMR CatApp）
- 文件：`data_raw/catappdata.csv`（只读，不修改）
- 文献：Hummelshøj et al., *Angew. Chem. Int. Ed.* 2012, DOI: [10.1002/anie.201107947](https://doi.org/10.1002/anie.201107947)；RPBE 泛函
- 能量语义：经检验为"数值越大=结合越强"（≈ −E_ads+常数，CatApp 参考态校正所致），见 `notes/04_screening.md`；
  BEP 行（经验关系估算）已在清洗阶段剔除

### 权威交叉验证：Catalysis-Hub GraphQL API
- `scripts/13_cathub_fetch.py` 直连官方库核对 H* 能量并补齐 facet/sites 元数据，详见 `notes/09_cathub_integration.md`
- `scripts/14_extval2_homogeneous.py` 用官方库同质金属 HER 子集做第二外部验证（零重训 MAE 0.177 eV，详见 `notes/14_extval2.md`）
- **数据权威性结论**：主线镜像数据经 Catalysis-Hub 官方库逐条交叉核对——组成匹配率 **99.2%**（1,821/1,836）、
  能量逐条匹配率 **98.4%**（6,893/7,005）、匹配对最大偏差 5.2e-4 eV，主线数据可信度获官方权威确认
- **私密配置：项目根 `.env` 含 Catalysis-Hub API Key（私密），勿提交入库，分发前务必删除**

## 目录结构

```
ai_surface_cat/
├── SPEC.md            # 模块契约（唯一事实来源）
├── README.md
├── requirements.txt   # 依赖清单（宽松下界约束）
├── data_raw/          # catappdata.csv + MamunHighT2019_adsorption.json.gz（只读）
├── data_processed/    # 数据集与指标表（dataset_v1/v2、hstar_*、multiads_*、extval_* 等）
├── outputs/           # 优化线产物（opt_*，GroupKFold 口径）与 extval2_metrics.csv
├── figures/           # *.png（300 dpi，低饱和暖色）
├── scripts/           # 01-04 CO*/O* 验证线；05-08 H* 主线；09-10 多吸附种；
│                      # 11/14 外部验证；12 H* 深度优化；13 官方库交叉核对
├── notes/             # 方法学笔记：各阶段报告（01-14）+ 描述符表 + 99 终审核验
├── thesis/            # 成果汇总报告
├── deprecated/        # 废弃草稿产物归档（迁移说明见 deprecated/README.md）
└── web/               # 终端展示页面（即将交付，见 web/ 目录）
```

## 管线阶段

| 阶段 | 脚本 | 输出 | 状态 |
|---|---|---|---|
| 1 数据清洗 | `scripts/01_clean_data.py` | `dataset_v1.csv`、`notes/01_cleaning_report.md`、3 张 EDA 图 | ✅ |
| 2 特征工程 | `scripts/02_features.py` | `dataset_v2.csv`、`notes/descriptor_table.md` | ✅ |
| 3 建模 | `scripts/03_models.py` | `model_compare.csv`、`group_cv_results.csv`、`xgb_grid_results.csv`、`best_params.json`、2 张图 | ✅ |
| 4 解释与筛选 | `scripts/04_interpret_screen.py` | `shap_*.png`、`candidate_rankings.csv`、`notes/04_screening.md` | ✅ |
| 5 H* 数据获取 | `scripts/05_fetch_hstar.py` | `hstar_dataset_v1.csv`、`hstar_group_spread.csv`、`notes/05_hstar_report.md` | ✅ |
| 6 H* 特征化 | `scripts/06_hstar_features.py` | `hstar_dataset_v2.csv`、`notes/hstar_descriptor_table.md` | ✅ |
| 7 H* 建模 | `scripts/07_hstar_models.py` | `hstar_model_compare.csv`、`hstar_group_cv_results.csv`、`hstar_loeo_results.csv`、2 张图 | ✅ |
| 8 H* 解释与筛选 | `scripts/08_hstar_interpret.py` | `hstar_shap_*.png`、`candidate_rankings_hstar.csv`、`hstar_candidates.png` | ✅ |
| 9 多吸附种数据 | `scripts/09_multiads_data.py` | `multiads_dataset.csv`、`multiads_features.csv` | ✅ |
| 10 多吸附种建模 | `scripts/10_multiads_model.py` | `multiads_model_compare.csv`、`multiads_scaling_corr.csv`、`notes/06_multiads_report.md` | ✅ |
| 11 跨库外部验证 | `scripts/11_external_validation.py` | `extval_results.csv`、`extval_metrics.csv`、`notes/07_external_validation.md` | ✅ |
| 12 H* 深度优化 | `scripts/12_optimize_hstar.py` | `outputs/opt_*`、`figures/opt_search_trajectory.png`、`notes/12_optimization.md` | ✅ |
| 13 官方库交叉核对 | `scripts/13_cathub_fetch.py` | `cathub_mamun.csv`、`cathub_hstar_extra*.csv`、`hstar_facet_check.csv`、`notes/09_cathub_integration.md` | ✅ |
| 14 第二外部验证 | `scripts/14_extval2_homogeneous.py` | `extval2_homogeneous.csv`、`outputs/extval2_metrics.csv`、`notes/14_extval2.md` | ✅ |

另有文献锚点人工核对笔记 `notes/10_literature_benchmark.md`（25 个权威文献锚点，
配套数据 `data_processed/literature_anchors.csv`）与终审独立核验记录 `notes/99_final_audit.md`。

## 复现步骤

环境：Python 3（依赖见 requirements.txt：pandas / numpy / scipy / matplotlib / mendeleev /
scikit-learn / xgboost / shap），随机种子统一 42。

```bash
cd ai_surface_cat
# 验证线 CO*/O*
python scripts/01_clean_data.py        # 清洗 + EDA
python scripts/02_features.py          # 特征化
python scripts/03_models.py            # 5 模型对比 + XGBoost 小网格调参 + 随机/分组 CV
python scripts/04_interpret_screen.py  # SHAP 解释 + Sabatier 候选筛选
# 主线 H*（HER）
python scripts/05_fetch_hstar.py       # 下载（Zenodo→catbench 镜像回退）+ 解析 + 去重
python scripts/06_hstar_features.py    # 特征化
python scripts/07_hstar_models.py      # 建模 + CV + LOEO（分组键均为规范化组成）
python scripts/08_hstar_interpret.py   # SHAP + HER 筛选（ΔG_H* = E_ads + 0.24 eV）
# 扩展与核验（可选）
python scripts/09_multiads_data.py       # 多吸附种数据集（15 种吸附种）
python scripts/10_multiads_model.py      # 多吸附种统一模型 + 标度关系分析
python scripts/11_external_validation.py # 跨库外部验证（EqV2-HER）
python scripts/12_optimize_hstar.py      # H* 深度优化（GroupKFold 口径，产物在 outputs/）
python scripts/13_cathub_fetch.py        # Catalysis-Hub 官方库交叉核对（需 CATHUB_API_KEY）
python scripts/14_extval2_homogeneous.py # 第二外部验证（依赖 13 的产物）
```

每个脚本可独立复跑（01/05 只依赖原始数据或网络；02 依赖 01；06 依赖 05；
03/04 依赖 02，07/08 依赖 06；04 另依赖 03、08 另依赖 07 的指标/参数文件）。

## 关键数据处理决策与修复历史

### CO*/O* 验证线（详见 notes/01_cleaning_report.md、notes/04_screening.md）
- 只保留 CO\* / O\*；剔除 dataset 含 "BEP" 的经验估算行
- surface 列尾部 `AA/AB/BB` 为 L1₂ 合金表面终止层标签，解析为独立列 `term`（621 行，合金数据主体）
- 无法解析的非标准条目（MoS₂/MoSe₂ 类 15 行、ZnO(0001) 四位晶面 1 行）剔除并逐条记录
- 去重（comp, facet, term, adsorbate）：组内能量极差 ≤ 0.5 eV 取中位数；
  极差 > 0.5 eV 判为冲突记录整组剔除（11 组纯金属 (211) CO* 双文献矛盾记录，逐组列于报告）
- 1.5×IQR 法则（分吸附种）剔除能量异常值 → 937 行
- **修复**：GroupKFold 分组键由原始 comp 字符串改为规范化组成（元素排序+计量约简），
  消除 AgAu3 vs Au3Ag 类化学等价组成的隐性泄漏
- 注意：数据能量标度经检验为"数值越大=结合越强"（与最初预期相反），详见 `notes/04_screening.md`

### H* 主线（详见 notes/05_hstar_report.md）
- 组成规范化：元素排序 + gcd 计量约简（Pt9Ti3→Pt3Ti）；实测无同素异写冲突
- `_N` 后缀 = 同表面不同吸附构型（非重复测量），按 HER 惯例取**最低** E_ads（最稳定位点）；
  min-vs-mean 口径敏感性已在 notes/05 第 4 节披露（top-20 交集 0/20，结论解读为候选簇而非确定排名）
- **修复（终审）**：`site_spread`（组内能量极差）为目标泄漏特征（含目标值本身，新表面不可得），
  已从建模特征移除（保留为分析列）；MAE 由 0.104 → 0.120 eV（去泄漏后）
- **修复（终审）**：补充 LOEO（留一元素外测），与验证线口径对齐；候选表新增实用性红旗列
  （放射性 Tc、强亲氧 La/Y/Sc、LOEO 高误差 Mn/Bi/Fe）
- 已知局限：`structure_L10 ≡ facet_101` 恒等特征对（SHAP 解释时重要性不可分割）；
  镜像数据无 atoms_json，结构信息为计量模式推断
