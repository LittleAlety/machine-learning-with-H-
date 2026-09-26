# SPEC — ai_surface_cat：可解释机器学习表面吸附能预测与催化位点筛选

## 0. 项目定位
基于公开 DFT 吸附能数据库，构建"描述符 + 树模型 + SHAP"的可解释 ML 管线，
预测表面吸附能并按 Sabatier 原则筛选候选催化表面。
- **主线（H*）**：Mamun 2019 高通量双金属数据集 → HER 催化位点筛选
- **方法验证线（CO*/O*）**：CMR CatApp → 管线方法学验证（已完成）
管线设计为**吸附种无关**。

## 1. 目录契约
```
ai_surface_cat/
├── SPEC.md            # 本文件
├── README.md          # 项目说明（来源、复现步骤）
├── data_raw/          # 原始数据（只读不改）
├── data_processed/    # 数据集与结果表格
├── outputs/           # 优化线（scripts/12）与第二外部验证（scripts/14）产物
├── figures/           # *.png（300 dpi，低饱和配色，无蓝紫渐变）
├── scripts/           # 01-04: CO*/O* 线；05-08: H* 主线；09-10: 多吸附种；
│                      # 11/14: 外部验证；12: H* 深度优化；13: 官方库交叉核对
├── notes/             # 方法学笔记、局限分析
├── thesis/            # 成果汇总报告
└── deprecated/        # 废弃草稿产物归档（见 deprecated/README.md）
```

## 2. CO*/O* 线数据源（已完成）
- 文件：data_raw/catappdata.csv（3269 行）
- 出处：CMR CatApp, Hummelshøj et al., Angew. Chem. Int. Ed. 2012, DOI 10.1002/anie.201107947；RPBE 泛函
- 列：reaction_energy, activation_energy, surface, ab, a, b, reference, url, dataset
- 能量语义修正（第 5 节验收时确认）：该数据**数值越大=结合越强**（≈ −E_ads + 常数），见 notes/04_screening.md

## 3. CO*/O* 线模块契约（01-04，已完成并验收）
### scripts/01_clean_data.py → data_processed/dataset_v1.csv
- 只保留 ab ∈ {CO*, O*}；剔除 dataset 含 "BEP" 的行
- surface 解析：`^(?P<comp>.+?)\((?P<facet>-?\d{3})\)(?: (?P<term>AA|AB|BB))?$`
  - 尾部 AA/AB/BB 是表面终止层标签，解析成独立列 `term` 作特征（621 行，合金数据主体）
  - 无法解析约 16 行（MoS2 类），剔除并记录
- 输出列：comp, facet, term, adsorbate ∈ {CO, O}, energy_eV, dataset_src, n_raw
- 去重规则：按 (comp, facet, term, adsorbate) 分组；组内极差 ≤0.5 eV 取中位数合并；
  极差 >0.5 eV 判为冲突记录整组剔除（11 组纯金属 (211) CO* 两文献矛盾记录已剔除，逐组列于报告）
- IQR 异常值剔除（分吸附种，1.5×IQR），各步删行数写入 notes/01_cleaning_report.md

### scripts/02_features.py → data_processed/dataset_v2.csv
- 元素描述符（mendeleev）：Pauling 电负性、原子半径、族号、周期、第一电离能、d 电子数
  （注意 mendeleev 组态无角标问题，Sc/Y/La 的 nd¹ 需特判）
- 组合统计（化学计量分数加权）：wmean/max/min/range/wstd + n_elements
- 结构描述符：facet/term/adsorbate one-hot
- 产物 notes/descriptor_table.md：每特征一句化学含义

### scripts/03_models.py
- Linear/Ridge/Lasso（StandardScaler+Pipeline）、RandomForest、XGBoost（小网格调参）
- 随机 5-fold CV + GroupKFold → model_compare.csv / group_cv_results.csv
- 图：model_cv_compare.png、pred_vs_true.png

### scripts/04_interpret_screen.py
- SHAP TreeExplainer（最优树模型）：shap_summary.png、shap_dependence_top.png
- CO* 筛选窗口 [0.7, 1.3] eV（依据见 notes/04_screening.md），候选标注"统计不可区分"
- 已知局限：按 comp 字符串分组存在隐性泄漏（AgAu3 vs Au3Ag 同素异写），H* 线已修正

## 4. 运行环境
- 用 shell 的 python（/usr/local/bin/python，已装 pandas/sklearn/xgboost/shap/mendeleev）
- ipython 内核缺 xgboost/shap，不要在 ipython 里跑建模脚本
- 随机种子统一 42

## 5. 验收标准
- 每个脚本可独立 `python scripts/0X_*.py` 无报错复跑
- 数据集行数、清洗记录、指标表、图、候选清单一一对应
- 最优模型随机 CV 的 MAE 显著优于线性基线

---

## 6. H* 主线（05-08，优先级最高）

### 数据源
- CatBench Zenodo 镜像（Mamun et al., Sci. Data 6:76, 2019，CC-BY-4.0）：
  `https://zenodo.org/records/17157086/files/MamunHighT2019_adsorption.json?download=1`（195.4 MB，Zenodo 偶发 504，需重试）
- 已验证：45,130 条吸附反应，其中**稀释 H\*（0.5H₂(g)+*→H\*）7,048 条**，1,836 个唯一表面、37 种金属；结构类型 A1（纯金属，(111)）、L1₂（(111)）、L1₀（(101)）；能量 −2.31~+3.75 eV
- 关键字段：ref_ads_eng（目标）、adsorbate_indices、atoms_json（ASE 结构）、键名含组成（如 Pt9Ti3）
- README 提示存在重复几何（弛豫后重构），必须去重

### 脚本与产物
- `scripts/05_fetch_hstar.py` → 下载+解析+去重 → `data_processed/hstar_dataset_v1.csv`
  （列：comp 规范化组成（元素排序！）、structure ∈ {A1,L10,L12}、facet、energy_eV、n_raw；
  去重同 CO 线规则：极差>0.5 eV 冲突组剔除并逐组列明）
- `scripts/06_hstar_features.py` → `data_processed/hstar_dataset_v2.csv`（复用 02 的元素描述符体系 + structure/facet one-hot）
- `scripts/07_hstar_models.py` → 同 03，**GroupKFold 的 groups 必须用规范化组成（元素排序后），避免 CO 线的隐性泄漏**
- `scripts/08_hstar_interpret.py` → SHAP + HER 候选筛选
- 产物命名加 `hstar_` 前缀；方法学文档 `notes/05_hstar_report.md`

### HER 筛选规则（核心化学逻辑）
- ΔG_H* ≈ E_ads(H*) + 0.24 eV（零点能+熵修正，Nørskov 标准近似）
- 按 |ΔG_H*| 升序排名，最优 ≈ 0；与第 1 名差距 < 模型 MAE 的候选标注"统计不可区分"
- 文献锚点交叉核对：Pt 系 ΔG_H* ≈ −0.1 eV 应落在前列，若严重偏离需解释
- 输出 `candidate_rankings_hstar.csv` + `figures/hstar_candidates.png`（|ΔG| 分布与候选高亮）

### H* 线验收标准
- 随机 CV MAE ≤ 0.25 eV 量级，且显著优于线性基线
- 候选 top-10 有文献合理性讨论

---

## 7. 数据扩展与权威性强化（第二轮，09-11）

### scripts/09_multiads_data.py + 10_multiads_model.py（多吸附种统一模型）
- 数据：data_raw/MamunHighT2019_adsorption.json.gz 全量 45,130 条（H/C/N/O/S 及 CHx/NH/OH/SH）
- 清洗：按吸附种分组，同 (comp, structure, adsorbate) 多构型取最低 E_ads（与 H* 线口径一致）；记录各吸附种条数
- 特征：复用 02/06 的元素+结构描述符体系，新增吸附种描述符（吸附原子身份 one-hot + 吸附原子的元素性质：电负性、半径、族号）
- 建模：XGBoost 为主（随机 CV + GroupKFold 规范化组成），与单吸附种模型对比
- 归纳分析（权威性核心）：标度关系——以 H* 为锚，输出 E_ads(X*) vs E_ads(H*) 的相关矩阵与散点图（X ∈ C,N,O,S），讨论"一种吸附能不能预测另一种"的经典命题
- 产物：multiads_dataset.csv、multiads_model_compare.csv、figures/multiads_scaling_*.png、multiads_shap_summary.png、notes/06_multiads_report.md

### scripts/11_external_validation.py（跨库外部验证）
- 外部源 1：EqV2-HER-Discovery（GitHub ergroup，ACS Catal. 2025 配套，~861 条 DFT H 吸附能，含组成/晶面/表面能）——下载到 data_raw/eqv2_*.csv
- 外部源 2（可选，设可行性闸门）：GASdb docs.pkl（1.25GB，H 吸附能 ~21k 条）；2 次解析失败即放弃并在 notes 记录原因
- 方法：用 H* 线最优模型（07 的 XGBoost_tuned）对外部数据做零重训预测；报跨库 MAE/RMSE/R²、系统性偏差（均值偏移，反映泛函/参考态差异：Mamun=BEEF-vdW vs EqV2=RPBE?/其他）、按组成分层误差
- 产物：extval_results.csv、figures/extval_parity.png、notes/07_external_validation.md

---

## 8. 模型优化（第三轮，scripts/12）

### scripts/12_optimize_hstar.py（H* 主线深度优化）
- 目标：在零泄漏前提下优化 H* 主线模型（当前基线：XGB MAE 随机CV 0.120 / LOEO 0.155）
- 超参搜索：RandomizedSearchCV ≥150 组，空间含 max_depth/min_child_weight/gamma/subsample/colsample_bytree/colsample_bylevel/reg_alpha/reg_lambda/learning_rate（配 n_estimators 上限 2000 + early stopping）
- 样本权重实验：uniform vs n_raw 加权 vs 组成频次逆加权，三种对比
- 模型融合：Stacking（XGB + RF + LinearRegression 元学习器）
- 诚实评估铁律：
  a) 搜索阶段用内层 CV；最终报告用**嵌套 CV（外层 5-fold）无偏估计**
  b) 同步报告 GroupKFold（规范化comp）与 LOEO
  c) 最优模型零重训复测 EqV2 外部集（复用 11 的流程）
- 若最终模型优于现基线：用该模型重跑候选筛选（复用 08 逻辑），刷新 candidate_rankings_hstar.csv 与 hstar_candidates.png
- 产物：opt_search_results.csv（全部搜索记录）、opt_final_metrics.csv（嵌套CV/分组/LOEO/外部验证四口径）、notes/12_optimization.md（搜索过程、权重实验结论、最优配置、与基线的诚实对比——若提升 <0.005 eV 需明说"收益边际"）。
  （实施注记：全部产物实际落盘于 outputs/opt_*；旧随机 KFold 草稿 notes/08_optimization.md 与 data_processed/opt_* 已归档 deprecated/）
- 红线：禁止任何标签派生特征；种子 42；不得改动 01-11 的已有产物口径（优化产物用 opt_ 前缀，候选刷新除外）
