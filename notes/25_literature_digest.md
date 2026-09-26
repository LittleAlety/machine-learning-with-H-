# 文献消化报告：4 篇催化表面吸附能 ML 文献

> 项目背景：1,836 个过渡金属表面 H* 吸附能（BEEF-vdW），36 个组成级特征（mendeleev 6 种元素性质 × 5 种加权统计 + 结构/晶面 one-hot），XGBoost，OOF MAE 0.1198 eV。已建成 SHAP、候选排名、分位数 XGBoost 不确定度、主动学习采集清单。
> 本报告所有数字/定义均摘自 PDF 原文并标注页码（以 PDF 印刷页码为准）；原文未给出者标"未提及"。

---

## 文献 1：Usuga, Praveen & Comas-Vives（J. Mater. Chem. A, 2024）—— 局部环境描述符 + 聚类异常检测

### 1.1 题录
- **标题**：Local descriptors-based machine learning model refined by cluster analysis for accurately predicting adsorption energies on bimetallic alloys
- **作者**：A. F. Usuga, C. S. Praveen, A. Comas-Vives（TU Wien / UAB / CUSAT）
- **期刊/年份**：J. Mater. Chem. A, 2024, 12, 2708–2721（2023-10-17 收稿，2023-12-04 在线）
- **DOI**：10.1039/d3ta06316j
- **代码**：https://github.com/Anfeus02/Localized-chemical-E_ads-ML（p.2718）

### 1.2 数据集与计算设置
- 17,343 个数据点，取自 CatalysisHub 的 Mamun et al.（Sci. Data 2019, 6, 76）数据集（p.2709–2710）：FCC 双金属 (111)/(101) 晶面，A:B 比 0/25/50/75/100%；A ∈ {Ag, Au, Cu, Fe, Pt, Zn}，B 覆盖 3–15 族、4–6 周期金属及 Al；吸附质 C, CH, CH₂, CH₃, H, N, NH, O, OH, H₂O, S, SH；位点类型 top/bridge/hollow。
- 目标：E_ads = E_ads/surf − (E_surf + E_ads)（p.2709）。
- 描述符用 DFT 单点：Quantum ESPRESSO，BEEF-vdW，GBRV 超软赝势，截断 35/350 Ry，Monkhorst–Pack 4×4×1 k 点（p.2710）——**与我们的 BEEF-vdW 泛函一致**。

### 1.3 描述符清单（34 个，p.2711–2712；完整表在 ESI Table S2）
**几何构造法（可移植的核心思想）**：在吸附位点正上方、距顶层 1.5 Å 处放置一个截断球，球心 XY 坐标取弛豫后吸附原子位置；球内表面原子视为负责成键的局部环境；每个特征取**球内所有原子的平均**（"the average property across all atoms contained within the cutoff sphere"，p.2711）。球半径/高度启发式调节至配位数合理。

| 类别 | 特征（原文记号） | 定义与来源 | 组成级可移植？ |
|---|---|---|---|
| 几何 | atoms_surf | 与吸附质直接成键的表面原子数（位点编码相关） | 半可移植（需位点类型，不需坐标） |
| 几何 | stm_surf | 吸附位点处电子密度（STM-like，DFT） | 否 |
| 电子 | Fermi energy、d-band center、work function | 清洁表面单点 DFT（p.2711，"do not inherently include geometrical effects"） | 否（需 DFT） |
| 电子 | d_charge_surf / s-partial orbital charge | 表面原子 d/s 分波电荷（球内平均），**SHAP 最重要特征**（p.2714） | 否 |
| 元素 | 原子半径、电负性等 tabulated 性质 | 球内原子平均（p.2711 "atomic-tabulated properties"） | **是**（思想可移植：对位点邻域做加权平均而非全组成平均） |
| 位点 | 吸附位点类型 | 数值编码 1/2/3/4 = top/bridge/fcc-hollow/hcp-hollow（p.2711） | 是 |
| 吸附质 | HOMO_ads（吸附质 HOMO 能量，DFT 单点）、H_ads（吸附质中 H 原子数）、p_charge_ads、电负性 | 仅考虑与表面成键的主元素（C/H/N/O/S），数值分层（stratified）表示（p.2712） | 部分（H_ads、电负性可；HOMO 需 DFT） |

**关键发现**：Pearson 线性相关普遍接近 0；只有 atoms_surf 和 stm_surf 有中等单特征线性相关（p.2712）——印证组成/单描述符必须靠非线性模型。电子类 > 几何类 > 元素类（SHAP 影响力排序，p.2718 结论）。

### 1.4 模型与验证设计
- 预处理：RobustScaler；随机 70/30 划分；**10 折 K-fold CV** 调参（p.2710）。
- 模型：MLR、KRR（Laplacian 核）、RFR + 梯度提升（XGBoost / CatBoost / LightGBM）。
- 超参搜索范围（p.2710）：n_estimators 300–1500（步长 200）；learning_rate 0.06–0.20（步长 0.02）；max_depth 3–10；min child samples 0–10；L2 0–2.0（步长 0.02）。
- 精度（Table 1/2, p.2713）：CatBoost 最优，K-fold R²=0.908；MAE train 0.166 eV / test 0.280 eV（XGBoost 0.175/0.289；LightGBM 0.170/0.299；KRR 0.151/0.308；MLR 0.671/0.660）。
- **外部验证/留一外推：未提及**（仅随机划分 + K-fold，无按成分/元素留出）。

### 1.5 聚类异常检测工作流（本文最大特色，无主动学习）
- UMAP 降维 + DBSCAN 聚类；**关键技巧：对 SHAP 值（而非原始特征）做 UMAP/DBSCAN**（"supervised clustering"，借鉴 Cooper et al. 2021，p.2714–2715）——SHAP 空间聚类显著优于原始特征空间。
- 异常定位：MAE > 1 eV 的点仅占训练集 4.3%，集中于 **top 位点**（atoms_surf=1）且处于弱吸附区（E_ads 在 −4~0 eV 段模型高估）（p.2713, 2715）。
- 剔除 top 位点（17,343 → 13,894，−19.9%）后重训：CatBoost MAE 0.019 eV (train) / **0.174 eV (test)**，test MAE 降 37.9%、MSE 降 58.1%（Table 4, p.2716–2717）。
- 注意：摘要中著名的 "0.019/0.174 eV" 是**剔除 top 位点后**的数字，全数据是 0.166/0.280 eV——引用时勿混淆。
- 无主动学习/迭代采集；无 ASE/FireWorks（QE 单点，结构来自既有数据库）。

### 1.6 对本项目的可移植点
1. **SHAP 值空间聚类做异常检测**（UMAP+DBSCAN on SHAP）——直接可嫁接到我们已建成的 SHAP 管线，定位高偏差子群（如某晶面/某位点），指导主动学习采集。
2. **"局部邻域加权平均"思想**：我们的 wmean 是全组成平均；若能按"位点近邻原子"做加权（仅需位点近邻组成，不需坐标），即其 atoms_surf/平均半径思想的组成级版本。
3. **位点类型数值编码**（top/bridge/hollow 分级）与我们的晶面 one-hot 互补。
4. 误差分区诊断：弱吸附区（−4~0 eV）偏差大 → 可对照我们 1,836 点中 H* 弱吸附端的表现。
5. 超参搜索网格范围可直接复用。

---

## 文献 2：Chen, Huang, Hua, He & Schwaller（Nat. Commun. 2025）—— AdsMT 多模态 Transformer 预测 GMAE

### 2.1 题录
- **标题**：A multi-modal transformer for predicting global minimum adsorption energy
- **作者**：Junwu Chen, Xu Huang（共同一作）, Cheng Hua, Yulian He, Philippe Schwaller（EPFL LIAC / NCCR Catalysis / SJTU）
- **期刊/年份**：Nature Communications (2025) 16:3232（2024-08-16 收稿，2025-03-17 接收）
- **DOI**：10.1038/s41467-025-58499-7
- **代码/数据**：github.com/schwallergroup/AdsMT；Zenodo 10.5281/zenodo.12104162（p.10–11）

### 2.2 描述符清单
**表面（图模态）**——全部需要原子结构，组成级不可直接移植：
- 周期表面图 G=(V,E)：节点特征 = 原子序数 z_i 的 embedding；边 = 半径截断 r_c 内近邻（含自连接边实现面内周期性，p.8 eq.1）；边特征 = 距离 d_ij 的 RBF 展开。
- **位置编码 δ_i = (h_i − h_min)/(h_max − h_min)**（p.3 Fig.2a、p.8 eq.2）：原子沿 c 轴相对高度，δ=1 顶层 / δ=0 底层，经 RBF 展开为位置 embedding——解决 GNN 无法区分顶层/底层原子的问题。
- 深度 embedding s_i（cross-attention 的 K/V 中拼接，p.9 eq.11）。
- 图编码器可替换：AdsGT（本文）、CGCNN、SchNet、DimeNet++、GemNet-OC、ET、eSCN。

**吸附质（向量模态）**——**组成级、无需结构**：
- **RDKit 208 维分子描述符向量**（p.8 Methods "Adsorbate feature"，k″=208）。"molecular descriptors based on expert knowledge … especially for small adsorbates"（p.8）——对小吸附质（*H、*O 单原子无法用图消息传递）用专家描述符而非图。

**化学空间可视化**：表面用 SOAP 描述符、吸附质用 RDKit 描述符做 UMAP（p.4 Fig.3e）。

### 2.3 三个基准数据集（p.3–4, Fig.3）
| 数据集 | 来源 | 组合数 | 表面 | 吸附质 | GMAE 范围 |
|---|---|---|---|---|---|
| Alloy-GMAE | Catalysis Hub（Mamun et al.，**与我们同源的合金数据**） | 11,260 | 1,916 双金属合金（37 种金属） | 12 种 <5 原子小分子 | −4.3 ~ 9.1 eV |
| FG-GMAE | GAME-Net 'functional groups' 数据集 | 3,308 | 14 纯金属 | 202 种官能团分子 | −4.0 ~ 0.8 eV |
| OCD-GMAE | OC20-Dense | 973 | 967 无机表面（54 元素） | 74 种 O/H、C₁/₂、N 基 | −8.0 ~ 6.4 eV |
- GMAE 定义：对每个表面/吸附质组合枚举所有吸附构型，取最低吸附能（p.8 data cleaning）。
- 另建 **OC20-LMAE**（363,937 组合，训练 345,254 / 验证 18,683）用于预训练（p.5, 8）。

### 2.4 模型与验证设计
- 架构：图编码器 + MLP 向量编码器 + 跨模态编码器（cross-attention Q=吸附质+图 embedding，K/V=原子+深度 embedding；后接 self-attention 学表面内部位移效应；MLP 能量头）（p.9 eq.10–20）。
- 训练：MAE 损失、AdamW、reduce-on-plateau；**随机 8:1:1 划分**；10 个随机种子重复；单卡 A100 fp32；目标标准化（p.9）。
- **外部验证（最相关）**：按**表面类型**或**吸附质类型**做分组划分（80% 类型训练/10% 验证/10% 测试），测试集表面/吸附质类型完全不出现在训练中（p.4）：表面分组 MAE ↑ ~0.02 eV、成功率 ↓ ~6%（Alloy-GMAE 最佳 MAE 0.158 eV / SR 60.1%）；吸附质分组 MAE ↑ ~0.04 eV、SR ↓ ~8%（FG-GMAE 最佳 0.123 eV / 65.3%）。
- 精度（随机划分，p.4–5）：Alloy-GMAE MAE 0.143 eV / SR（|误差|<0.1 eV 比例）66.3%；FG-GMAE 0.095 eV / 71.9%；OCD-GMAE 0.571 eV / 13.5%（数据 <1000 且 54 元素，差）。迁移学习（OC20-LMAE 预训练 + 冻结图编码器微调）后 OCD-GMAE 达 **0.389 eV / 22.0%**（摘要的 0.09/0.14/0.39 即三个数据集的最佳 MAE）。
- **UQ（p.7, 9–10）**：**深度集成**——10 个同架构不同种子副本，预测取均值、不确定度取标准差；校准曲线接近对角线、miscalibration area < 0.1；不确定度与 MAE 的 Spearman 相关 > 0.98。
- 解释性：cross-attention 分数识别最优吸附位点，Alloy-GMAE 准确率 0.48（ET 编码器）、OCD-GMAE 0.56（AdsGT），远超随机基线（p.6–7）。
- 速度：比 DFT 快约 8 个数量级、比 MLIP+启发式搜索快 4 个数量级（p.7）。

### 2.5 主动学习/高通量工作流
- **无主动学习实施**；仅在讨论中提出未来方向："integrate AdsMT with active learning … iterative expansion of the training datasets towards underexplored regions"，以及"AdsMT 初筛（目标 GMAE + 低不确定度）→ DFT 复核 top 候选"的两级筛选策略（p.7）。
- 工具链：PyTorch Geometric 2.2.0 / PyTorch 1.13.1；ASE 3.22.1 处理表面结构；RDKit 2022.9.5（p.9）。
- **"390,625 位点"在本文未出现**（该数字出自 Tran & Ulissi, Nat. Catal. 2018，本文 ref 20 仅引用，未展开）。

### 2.6 对本项目的可移植点
1. **分组外部验证设计**（按表面/成分类型 80/10/10 留出，报告性能衰减幅度 ~0.02 eV / ~6% SR）——为我们的"留一元素/留一合金"外推评估提供标准做法与预期衰减基准。
2. **深度集成 UQ + 校准曲线 + miscalibration area < 0.1 + Spearman(σ, MAE)**——可与我们分位数 XGBoost 互为对照的第二条 UQ 路线与校准评估指标。
3. **成功率 SR（|误差|<0.1 eV 比例）**作为除 MAE 外的筛选导向指标。
4. RDKit/专家描述符处理单原子吸附质的思路（对我们 H* 体系，说明 H 端特征可极简）。
5. 两级筛选叙事（ML 初筛 + DFT 复核 top 候选）可直接用于论文工作流图。

---

## 文献 3：Montoya & Persson（npj Comput. Mater. 2017）—— 吸附能高通量自动化框架（⚠️ 非 Tran & Ulissi）

### 3.1 题录
- **标题**：A high-throughput framework for determining adsorption energies on solid surfaces
- **作者**：Joseph H. Montoya, Kristin A. Persson（LBNL / UC Berkeley）
- **期刊/年份**：npj Computational Materials (2017) 3:14（2016-12-14 收稿，2017-02-28 接收）
- **DOI**：10.1038/s41524-017-0017-z
- **注意**：此 PDF 是 **Montoya & Persson 的高通量工作流论文，不是 Tran & Ulissi 的主动学习经典论文**（后者为 Nat. Catal. 2018, 1, 696–703，本文仅在 ref 24 被引用 Ulissi 2016 JPC Lett.）。任务背景中的预判有误。

### 3.2 描述符清单
- **无 ML 描述符**（非 ML 论文）。仅在引言末尾引用"recent machine-learning approaches and more advanced structural descriptors（指 Calle-Vallejo 配位数结构敏感性描述符, Nat. Chem. 2015）may help"（p.2–3）。
- 唯一可视为"结构描述"的内容：**吸附位点自动枚举算法**（p.3 Methods）：选表面位点（手动/距最大 z 的窗口/pymatgen Voronoi 欠配位判定）→ 2D Voronoi 镶嵌（投影到垂直 Miller 指数的平面）→ **on-top=顶点、bridge=边中点、hollow=Voronoi 面心** → 双重过滤：距离 <0.1 Å（默认）的近重复位点剔除 + 按 slab 对称操作剔除对称等价位点 → 得"对称且几何互异"的最小位点集。Ni(111) 示例正确生成 on-top/bridge/fcc/hcp（p.3 Fig.3）。

### 3.3 模型与验证设计（=DFT 基准验证，非 ML）
- 基准集：**CE27 实验化学吸附数据库**（H₂, N₂, CO, O, NO 在单质低指数晶面）（p.1–2）。
- 表面能 vs Tran et al. (Sci. Data 2016)：MAE **0.02 eV**（p.2）。
- 吸附能 vs Wellendorf et al. (PRB 2012) PBE 计算基准：MAE **0.2 eV**（差异归因于优化例程、DFT 形式、GPAW vs VASP 赝势、未校正电子能参考态）（p.2）。
- 实验基准：**RPBE 与实验定量吻合明显优于 PBE；PBE 系统性低估分子化学吸附能（过强吸附）**（p.2 Fig.2）——对我们 BEEF-vdW 数据的实验交叉验证有参照意义。
- 局限自述：复杂吸附质（H₂O, HOO*, C₆H₆）转动自由度需人工干预；台阶/扭折位点更多；不含动力学势垒与覆盖度效应（p.2–3）。

### 3.4 主动学习/高通量工作流细节（本文核心贡献）
- 无主动学习。高通量自动化栈（p.3–4）：
  - **atomate** 生成 workflow（结合 **FireWorks** + **pymatgen**）；
  - **VASP** 计算，**custodian** 做 on-the-fly 错误纠正的作业管理；
  - 流程：bulk 结构优化（晶格常数收敛）→ 分支为 slab/吸附构型的离子弛豫；每个 FireWork = 预处理（系统特定参数注入）→ VASP+custodian → 后处理（结果存 JSON / 上传数据库）；
  - **刻意不用 FireWorks 动态工作流**：开跑前一次性固定全部任务数（"dynamic workflows are often difficult to debug"，p.3–4）；
  - 输入极简：bulk 结构 + VASP 参数 + "adsorbate configuration"（化学身份、几何、Miller 指数）；
  - 规模：200+ DFT 计算压缩为单次提交，"potentially one to two orders of magnitude more"（p.1）。
- SI 附 IPython notebooks（CE27 基准 + OER 中间体在所有低指数晶面/不同终端的扩展示例）。

### 3.5 对本项目的可移植点
1. **位点枚举的"对称去重 + 距离去重"双过滤准则**（0.1 Å 默认阈值、Voronoi 位点分类）——若未来从成分预测走向位点级 DFT 补算，这是标准做法。
2. **atomate/FireWorks/custodian 自动化栈选型与"固定任务数"工程经验**——主动学习补点后的 DFT 批量执行可直接采用。
3. **基准态度**：吸附能需同时对照计算基准（MAE 0.2 eV 量级视为可复现）与实验基准（泛函选择导致系统性偏移），为我们 BEEF-vdW 模型的外部验证误差容忍度提供标尺。
4. CE27 作为公开实验基准集，可用于我们模型的实验交叉验证（已有 24_experimental_crosscheck.md）。

---

## 文献 4：Vipin K E & Padhan（手稿，无期刊信息）—— HER/OER 吸附自由能堆叠集成

### 4.1 题录
- **标题**：Data-Driven Catalyst Design: A Machine Learning Approach to Predicting Electrocatalytic Performance in Hydrogen Evolution and Oxygen Evolution Reactions
- **作者**：Vipin K E, Prahallad Padhan（IIT Madras, Department of Physics）
- **期刊/年份**：**PDF 中未标注期刊、年份、DOI**（版式为手稿/preprint；通讯邮箱 ph18d200@smail.iitm.ac.in）；收稿/DOI 未提及。
- 页码：PDF 无印刷页码，以下按节标注。

### 4.2 数据集
- Catalysis-hub（Mamun et al. Sci. Data 2019，同文献 1、2 的合金数据源）：**16,226 点 = HER 8,856 + OER 7,370**；9 种吸附质 C/O/N/H/S/CHx/OH/NH/SH；**2,035 个双金属合金表面**；5 种化学计量比 0/25/50/75/100%；L1₀ (AB) 与 L1₂ (A₃B) FCC 结构；37 种元素（"Dataset Composition" 节）。目标：吸附 Gibbs 自由能 ΔG。

### 4.3 描述符清单（最终 80 维 HER / 76 维 OER）
1. **组成特征：Matminer 包**生成（"Composition and site based features" 节），SHAP 揭示实际命中的全部是 **MagpieData 系列**：
   - `MagpieData mean MeltingT`（HER 最重要）、max/min MeltingT
   - `MagpieData mean MendeleevNumber`、min MendeleevNumber
   - `MagpieData mode NdUnfilled`、mode NUnfilled（未填满电子数）
   - `MagpieData mode Electronegativity`、range Electronegativity
   - `MagpieData maximum CovalentRadius`（OER 关键）
   - `MagpieData mean NdValence`、min NdValence、mode Column（OER）
   - 加权方案：**mean/mode/min/max/range 等统计量 × Magpie 元素属性**，未说明是否分数加权（原文仅称 "compositional analysis"，加权细节未提及）——**与我们 mendeleev 6 性质 × 5 统计的方案同构**，可直接对照扩充（MeltingT、MendeleevNumber、NUnfilled、NdValence、CovalentRadius、Column 均为 mendeleev/matminer 可得的纯组成属性）。
2. **位点特征**：20 个 one-hot 位点描述符 → **PCA 降到 5 个主成分**（PCA_Site_1..5；HER 的 PCA_Site_3、OER 的 PCA_Site_3/2 均为头部特征）——降维去多重共线。
3. **特征筛选**：Pearson 相关分析剔除高相关特征（阈值未提及）。
- 全部特征均为组成级/类别级，**无需原子坐标**——与我们管线完全同层。

### 4.4 模型与验证设计
- **Stacking 集成**：基模型 Random Forest + XGBoost，**元模型 SVR**（"ML model" 节）。
- 划分：**90% 训练 / 10% 独立测试**；未提及 K-fold CV、未提及按成分/元素的外部留出验证、未提及超参搜索细节。
- 精度：**HER R²=0.98, MAE=0.251 eV；OER R²=0.94, MAE=0.121 eV**（摘要与 "ML model" 节）。注意 HER 的 MAE 反而大于 OER，且 R²=0.98 对应 MAE 0.251 暗示目标方差大——随机划分 + 同源数据（与训练集同合金家族）下数字偏乐观，无 OOD 评估。
- 解释：SHAP summary（HER 图 3、OER 图 4）。

### 4.5 主动学习/高通量工作流
- **无主动学习、无高通量 DFT、无不确定度量化、无自动化工具**（全部未提及）。数据直接取自 Catalysis-hub。

### 4.6 对本项目的可移植点
1. **MagpieData 属性扩充清单**（组成级、零结构成本）：MeltingT、MendeleevNumber、NdUnfilled/NUnfilled、NdValence、CovalentRadius、Column——我们目前只用 6 种 mendeleev 性质，这些是文献验证过对 HER/OER ΔG 高 SHAP 的候选扩充。
2. **mode/range 统计量**：我们已有 wmean/wstd/min/max/range，可补 mode（众数，对 L1₀/L1₂ 化学计量有区分力）。
3. **位点 one-hot → PCA（20→5）**：与我们晶面/结构 one-hot 的降维替代方案，供特征膨胀时参考。
4. Stacking（RF+XGBoost→SVR 元模型）作为单 XGBoost 之外的集成对照；但注意其验证设计弱于我们（无 OOF/OOD），数字不宜直接对标我们的 OOF MAE 0.1198 eV。

---

## 跨文献汇总表

### A. 可移植描述符候选清单（组成级优先）

| 描述符 | 定义/来源 | 出处（页码） | 类型 | 移植成本 | 优先级 |
|---|---|---|---|---|---|
| MagpieData mean/max/min MeltingT | 组成元素熔点统计（matminer） | 文献4 SHAP-HER | 组成级 | 极低 | ★★★ |
| MagpieData mean/min MendeleevNumber | 门捷列夫数统计 | 文献4 SHAP-HER/OER | 组成级 | 极低 | ★★★ |
| MagpieData mode NdUnfilled / NUnfilled | 未填满 d/总电子数众数 | 文献4 SHAP-HER | 组成级（d 电子相关，我们已有 d 电子数可对照） | 极低 | ★★★ |
| MagpieData mean/min NdValence | d 价电子数统计 | 文献4 SHAP-OER | 组成级 | 极低 | ★★★ |
| MagpieData max CovalentRadius | 共价半径最大值 | 文献4 SHAP-OER | 组成级 | 极低 | ★★★ |
| mode Column / mode & range Electronegativity | 族号众数、电负性众数/极差 | 文献4 | 组成级（我们已有族号/电负性 wmean 等，补 mode 统计） | 极低 | ★★★ |
| 吸附位点类型分级编码 | top/bridge/fcc-hollow/hcp-hollow = 1/2/3/4 | 文献1 p.2711 | 类别级 | 低 | ★★☆ |
| 位点近邻组成加权平均（局部化 wmean） | 截断球内原子的元素性质平均（文献1 核心思想，p.2711）；我们可用"位点近邻元素"做组成级近似 | 文献1 | 半组成级 | 中 | ★★☆ |
| atoms_surf（与吸附质成键的表面原子数） | 位点连通性 | 文献1 p.2712 | 半组成级 | 中 | ★★☆ |
| 吸附质 H 原子数 H_ads / 主元素电负性 | 吸附质侧极简特征 | 文献1 p.2712 | 组成级（多吸附质扩展时用） | 低 | ★☆☆ |
| d-band center / Fermi 能 / 功函数 / d-partial 电荷 | 清洁表面 DFT 单点 | 文献1 p.2711 | **需 DFT，不可直接移植**；文献1 SHAP 显示 d 电荷最重要，可作为未来单点 DFT 增强方向 | 高 | ★☆☆ |
| 表面图 + 高度位置编码 δ_i；RDKit 208 维吸附质向量 | AdsMT 输入 | 文献2 p.8 | 需结构 / 吸附质专用 | 高 | ☆（仅参考） |

### B. 可移植验证 / 工作流设计清单

| 设计 | 内容 | 出处 | 对本项目的用法 |
|---|---|---|---|
| 分组外部验证 | 按表面/吸附质**类型** 80/10/10 留出；报告 MAE 增幅（+0.02~0.04 eV）与 SR 降幅（6~8%） | 文献2 p.4 | 标准化我们的留一元素/留一合金 OOD 评估与报告格式 |
| 成功率 SR 指标 | 预测误差 <0.1 eV 的样本比例 | 文献2 p.4 | 与 OOF MAE 并列报告，面向筛选 |
| 深度集成 UQ + 校准评估 | 10 个种子集成 σ；校准曲线、miscalibration area <0.1、Spearman(σ, MAE)>0.98 | 文献2 p.7, 9–10 | 与分位数 XGBoost UQ 互验；引入 miscalibration area 指标 |
| SHAP 空间聚类异常检测 | 对 SHAP 值（非原始特征）做 UMAP+DBSCAN，定位高偏差子群（文献1 定位到 top 位点） | 文献1 p.2714–2715 | 接到现有 SHAP 管线做误差子群诊断，指导主动学习采集清单 |
| 误差分区诊断 | 弱吸附区（−4~0 eV）偏差集中、MAE>1 eV 仅 4.3% | 文献1 p.2713, 2715 | 对 H* 弱吸附端做分段误差审计 |
| 超参网格 | n_est 300–1500、lr 0.06–0.20、depth 3–10、min_child 0–10、L2 0–2.0 | 文献1 p.2710 | 复用到 XGBoost 调参 |
| 位点枚举双过滤 | Voronoi 位点分类 + 0.1 Å 距离去重 + 对称等价去重 | 文献3 p.3 | 未来位点级 DFT 补算的位点生成标准 |
| HT 自动化栈 | atomate + FireWorks + pymatgen + custodian + VASP；固定任务数（避免动态 workflow 调试难）；JSON 入库 | 文献3 p.3–4 | 主动学习补点的 DFT 执行层选型 |
| 泛函基准态度 | 计算间复现 MAE~0.2 eV；PBE 系统性低估实验化学吸附、RPBE 更准；CE27 实验基准 | 文献3 p.2 | 外部验证误差容忍度标尺 + CE27 实验交叉验证 |
| Stacking 集成 | RF+XGBoost 基模型、SVR 元模型；90/10 划分（验证设计偏弱，慎用其精度数字） | 文献4 | 集成对照实验 |
| 位点 one-hot PCA 降维 | 20→5 主成分 | 文献4 | 特征膨胀时的降维备选 |
| 两级筛选叙事 | ML 初筛（目标值+低不确定度）→ DFT 复核 top 候选 | 文献2 p.7 | 项目论文工作流图与讨论 |

### C. 关键提醒
- 文献1 摘要数字 **0.019/0.174 eV 是剔除 top 位点（−19.9% 数据）后的结果**；全数据为 0.166/0.280 eV。
- 文献3 **不是** Tran & Ulissi 主动学习论文；"390,625 位点"在 4 篇 PDF 中均未出现（出自 Tran & Ulissi, Nat. Catal. 2018, 1, 696–703，仅被文献1/2 引用）。若需该论文细节需另行获取。
- 文献4 无期刊/DOI 信息（手稿形态），精度数字基于随机 90/10 划分、无 OOD 验证，与我们的 OOF MAE 0.1198 eV 不可直接比较。
- 4 篇均**未实施主动学习**（文献2 仅在展望中提及）；我们的主动学习采集清单在该维度上超出了这 4 篇的覆盖范围。
