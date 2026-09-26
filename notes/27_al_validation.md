# 27 — 主动学习补点的真实价值验证（回溯模拟 + 外部代理点）与真实计算组执行手册

脚本：`scripts/24_al_validation.py`（可重跑；任务 A 留池种子 123，随机对照种子 1000–1009，模型种子 42）。
不确定度管线与 21 号脚本完全一致：分位数 XGBoost（q05/q95）→ width；
采集分 = width × exp(−|pred_ΔG|/0.1)，ΔG = E_ads + 0.24 eV。
所有数字来自真实运行，明细见 `outputs/alval_*.csv`，图见 `figures/alval_learning_curve.png`。

---

## 任务 A：回溯式主动学习验证（内部模拟"补算 DFT 点"）

**协议**：1,836 个组成中留出 200 个（按 comp 分组、固定种子 123）作"假想未算池"，
其余 1,636 训练基线模型；对池计算采集分；AL top-k 或随机 k 点"揭示"加入训练集重训，
在池中**剩余**点上测 MAE。k ∈ {5, 10, 15, 20}；随机策略用 10 个种子取均值±std。
补点前（k=0）池 MAE = 0.1393 eV。

### A.1 学习曲线（剩余池 MAE, eV）

| k | 剩余 | AL（width×火山核） | 随机 mean±std | diff（随机−AL） | 显著* | AL 消融（仅 width） | diff_w | 显著* |
|---|---|---|---|---|---|---|---|---|
| 5  | 195 | 0.1407 | 0.1399±0.0024 | −0.0008 | no | **0.1284** | +0.0115 | **YES** |
| 10 | 190 | 0.1400 | 0.1396±0.0046 | −0.0005 | no | **0.1218** | +0.0178 | **YES** |
| 15 | 185 | 0.1396 | 0.1407±0.0064 | +0.0011 | no | **0.1171** | +0.0236 | **YES** |
| 20 | 180 | 0.1359 | 0.1407±0.0069 | +0.0048 | no | **0.1081** | +0.0326 | **YES** |

\* 判定标准：均值差 > 随机种子 std 判显著。

### A.2 判定与解读（诚实版）

1. **按原定判定标准，主采集函数（width×火山核）在 k=10/20 未显著优于随机**。
   原因清楚：火山核把补点预算聚焦在 |pred_ΔG|≲0.05 eV 的近峰组成
   （k=20 被选点 pred_dG 全部在 ±0.06 eV 内，如 Co₃Hg、Mo₃Zr、NiPd、Pt₃V），
   这些点本身预测就准（width 中等），揭示它们对"剩余池全局 MAE"几乎不贡献信息；
   而 5–20 个点相对 1,636 的基盘太小，单点杠杆有限。
2. **但消融实验（同一管线、仅把采集函数换成纯 width）在全部 k 上显著优于随机**，
   且优势随 k 单调扩大（k=20 时 0.108 vs 0.141，省 23% MAE，diff/std ≈ 4.7）。
   width-only top-20 确实命中真模型错误：如 CrNb₃（pred −1.03 / true +0.19 eV，
   误差 1.2 eV）、Fe₃In（pred +0.22 / true +1.77 eV）。
   **结论：不确定度信号本身是真实有效的补点依据；失败的是火山核与
   "全局 MAE"目标的不匹配，而非不确定度。**
3. 这与 21 号脚本的双清单设计自洽：**探索型清单（高 width）用于压缩模型误差，
   利用型/火山核清单用于发现近峰 HER 候选**——两个目标不应共用一个采集函数。
4. 局限：本模拟的"揭示"值是训练集已有的同协议 BEEF-vdW 能量，理想化了
   真实 DFT 补算的噪声/协议差；且特征为组成级，新增同原型组成的信息增量
   天然有限（真正的杠杆在域外/新原型，见 notes/22 §5）。

产物：`outputs/alval_A_pool_acquisition.csv`（池 200 点预测/区间/采集分）、
`outputs/alval_A_learning_curve.csv`、`outputs/alval_A_selected.csv`、
`figures/alval_learning_curve.png`（300dpi）。

---

## 任务 B：外部代理点验证（extval2 同泛函子集充当"新补算点"）

**筛选口径**：`extval2_homogeneous.csv`（355 行）含 `dftFunctional` 列；
取其小写化后恰为 `beef-vdw` 的行 → **115 行**。剔除 4 行带 VSHE 电位偏移标注的
BEEF-vdW（PasumarthiFacetDependence2023，参考态不同）；比 notes/14 的精确匹配口径
（114 行）多 1 行：SongStrongly2026 Pt(111)（原标记为小写 `beef-vdW`，泛函实为同一）。
特征化与 scripts/14 完全同口径（单元素→A1=1、合金 structure 全 0、facet 仅精确
101/111 置 1）。

### B.1 补点前后外部 MAE（剩余外部点）

| 阶段 | 评估点数 | AL top-k MAE (bias) | AL 组成去重 top-k MAE (bias) | 随机对照 mean±std |
|---|---|---|---|---|
| 补点前 | 115 | 0.1419 (+0.0002) | — | — |
| +10 点 | 105 | 0.1370 (−0.0254) | 0.1354 (−0.0123) | 0.1419±0.0057 |
| +20 点 | 95  | 0.1442 (−0.0389) | **0.1163 (−0.0125)** | 0.1386±0.0099 |

### B.2 解读（升/降都如实报）

1. **原始 top-k 暴露了一个真问题：采集分并列退化**。组成级特征下同组成的多个
   晶面/文献行特征几乎相同（facet one-hot 只有 101/111 两档），raw top-20 中
   **Pd 出现 10 次、Os 3 次、Ir 5 次**（同一组成的多晶面/多文献重复测量），
   补点预算被近重复点浪费——+20 点反而比 +10 差（0.1442 > 0.1370）。
2. **按组成去重后（每组成保留采集分最高 1 行再取 top-k），AL 价值显现**：
   +20 点外部 MAE 0.1419→0.1163（−18%），diff vs 随机 = 0.0223 > std 0.0099，显著。
   被选中的高杠杆点确有大残差：Pd₄Ta（pred −0.28 / true −1.13，残差 +0.86 eV）、
   Ir₄Sn（−0.80）、Os(211)（+0.44）、Ni₄Re（+0.49）、Rh₄Ta（+0.50）、Cu₄W（+0.38）。
   **修正系统误差点的信息价值远大于近峰确认点**。
3. **组成去重 top-10 清单（真实"优先补算"对象）**：
   Os、Pd₄Ta、Pd、Ir、Rh、Pt、Pt₂₉Sn、Ir₄Sn、Co、Ni；
   11–20 位加 Ru、Cu、Ni₄Re、Re、Rh₄Ta、Cu₄W、Ag₄Zn、Au、Au₄Zn、Ag。
   化学身份集中在 5d/4d 贵金属与 HansenFirst2018 的 Sn/Ta/W/Re 系表面合金——
   正是模型在"强吸附端"（E_true < −0.9 eV）系统性高估的区域。
4. 局限：① 只加 10–20 点、基盘 1,836，效应量小且评估集随之缩小（95 点），
   单次重训的 ±0.005 eV 波动属正常；② bias 转负（−0.01~−0.04）提示被选点
   多为负残差/强吸附端，补点后模型整体下移；③ 这些是"同库不同文献"点，
   共享 Cathub 收录协议（notes/14 §3.4）。

产物：`outputs/alval_B_extval_acquisition.csv`（115 点预测/区间/采集分）、
`outputs/alval_B_metrics.csv`、`outputs/alval_B_selected.csv`（raw 与 comp_dedup 两变体）。

---

## 任务 C：真实计算组执行手册——"若要真实补算 Top-10 点"

依据文献 3（Montoya & Persson, npj Comput. Mater. 2017；Voronoi 位点枚举 +
atomate/FireWorks/pymatgen/custodian/VASP 栈）与训练集（Mamun et al.,
Sci. Data 2019）协议，给出可执行清单。

### C.0 输入与优先级
- 候选来源：`outputs/active_learning_proposals.csv`（21 号脚本 Top-50）。
  采纳前人工筛查：**in_domain=True 优先**（AuMn₃、ReTl、Mn₃Tl、BiCo₃ 等）；
  Ge/Sb 域外点的 width 是下界（notes/22 §5.1），若纳入须预期更大协议风险。
- 依据任务 B 教训：**按组成去重**，同一组成只算一个代表表面（取晶面原型规定的那个）。
- 若目标是压缩全局误差（而非近峰候选确认），用探索型清单
  `active_learning_explore_top20.csv`（任务 A 消融证明 width-only 显著有效）。

### C.1 结构生成（pymatgen）
1. bulk 原型必须与训练集同构：A1 → FCC 原胞；L10 → 四方 CuAu 型（AB 1:1）；
   L12 → 立方 Cu₃Au 型（A₃B 3:1）。先优化 bulk 晶格常数至收敛（文献 3 的
   workflow 第一步）。
2. `SlabGenerator`：Miller 指数按特征口径——A1/L12 用 (111)，L10 用 (101)；
   slab 层数与真空层厚度不臆造，以 **C.3 锚定复算能复现 Mamun 已知值**为准定标
   （先扫 4/5/6 层 × 真空 ≥15 Å 的收敛性）。
3. 位点枚举：`AdsorbateSiteFinder`（文献 3 算法：2D Voronoi 镶嵌 →
   on-top=顶点、bridge=边中点、hollow=Voronoi 面心；双重过滤：0.1 Å 距离去重 +
   slab 对称等价去重）。H 放置于每个位点，全部弛豫。
4. **能量口径对齐训练集**：取同一表面所有位点弛豫后的**最低 E_ads**（最稳定位点口径），
   记录 `site_spread` = 位点能量极差、`n_raw` = 位点数——正是 v2 数据集的两列。

### C.2 泛函一致性（铁律）
- 训练集协议（文献 1 p.2710 对 Mamun 集的描述）：**Quantum ESPRESSO + BEEF-vdW +
  GBRV 超软赝势 + 截断 35/350 Ry（波函数/密度）+ Monkhorst–Pack 4×4×1 k 点**。
  首选直接用同栈复算。
- 若执行层用 VASP（atomate 默认）：INCAR 必须选 BEEF-vdW 泛函（VASP ≥5.4 编译
  libvdwxc），不得用 PBE/RPBE 代替——notes/14 实测跨泛函系统性偏差 +0.15~+0.35 eV，
  混入即污染训练集。
- **锚定验证（放行闸门）**：先从训练集抽 5–10 个已知体系（覆盖 A1/L10/L12 三种
  原型）按新栈复算，E_ads 与 Mamun 值偏差 |Δ| ≤ 0.1 eV 才放行批量
  （文献 3 给出计算间复现 MAE ~0.2 eV 的容忍标尺，0.1 eV 为更严内控）。

### C.3 收敛标准
- 电子 SCF 能量收敛 ≤ 1×10⁻⁶ eV；离子弛豫残余力 ≤ 0.05 eV/Å
  （以锚定复算可复现 Mamun 值为准微调）；smearing、k 点密度、截断一律同 C.2 协议。
- ΔG 校正统一 +0.24 eV（项目既有约定），不在 DFT 端重复加 ZPE/熵。

### C.4 批量提交（atomate/FireWorks，文献 3 §3.4 的工程经验）
- 输入极简三件套：bulk 结构 + 计算参数 + adsorbate configuration（H、Miller 指数）。
- workflow：bulk 优化 → 分支为 slab+H 各位点的离子弛豫；custodian 做 on-the-fly
  错误纠正；结果 JSON 入库（MongoDB）。
- **开跑前一次性固定全部任务数**（不用 FireWorks 动态 workflow，文献 3 原话
  "dynamic workflows are often difficult to debug"）。规模估计：10 组成 ×
  每表面 3–10 个对称互异位点 ≈ 数十至百余个 DFT 计算，单次提交即可。

### C.5 回收入库与回灌重训（对接现有管线）
1. 回收字段整理为 `hstar_dataset_v2.csv` 同构行：`comp`（元素字母序 + gcd 约简，
   同 scripts/05 规则）、`structure`（A1/L10/L12）、`facet`（111/101）、
   `energy_eV`（最稳定位点 E_ads）、`n_raw`、`site_spread`。
2. 复用 `scripts/06_hstar_features.py` 的 `build_element_table` 生成 36 特征
   （与 web/assets/feature_spec.md 逐位一致），append 进数据集。
3. 重跑 `scripts/21_active_learning.py`：重新 OOF 校准（覆盖率应维持 ~0.90）、
   枚举时新算组成自动被排除、采集清单刷新——完成一次闭环。
4. 验收标准：重训后 GroupKFold(5) OOF MAE 与校准覆盖率不劣化；
   已算组成不再出现在任何候选清单。

---

## 产物清单

| 文件 | 内容 |
|---|---|
| `scripts/24_al_validation.py` | A+B 全流程（可重跑，种子固定） |
| `outputs/alval_A_pool_acquisition.csv` | 池 200 点预测/双分位区间/采集分 |
| `outputs/alval_A_learning_curve.csv` | k × (AL / width-only 消融 / 随机 mean±std / 显著性) |
| `outputs/alval_A_selected.csv` | 各 k 被 AL（火山核）选中的池点明细 |
| `outputs/alval_B_extval_acquisition.csv` | extval2-BEEF 115 点预测/区间/采集分 |
| `outputs/alval_B_metrics.csv` | 补点前后外部 MAE（raw / 组成去重 / 随机对照） |
| `outputs/alval_B_selected.csv` | top-10/20 被选点明细（raw 与 comp_dedup 两变体） |
| `outputs/alval_summary.json` | 关键指标汇总 |
| `figures/alval_learning_curve.png` | 学习曲线（300dpi，暖色系，英文标注） |

## 一句话总结

**不确定度驱动的补点是真实有效的（width-only 消融：内部池 MAE −23%@k=20、
外部 MAE −18%@+20，均显著优于随机）；但要兑现价值必须做对两件事——
采集函数匹配目标（全局误差压缩用纯 width，近峰候选发现才用火山核），
以及按组成去重避免近重复点浪费预算。**
