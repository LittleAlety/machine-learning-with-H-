# S4 真实几何连续局域特征（Phase 6 追加注册 stage_S4_real_geom，分支 α）

## 数据源与解析（s1_parse_features.py，幂等）
- `data_raw/MamunHighT2019_adsorption.json.gz` 顶层 `_structures` 块（ASE-db 风格弛豫终态）。
- 逐 H* 构型取 `Hstar`（吸附态）与 `star`（洁净态参照）两结构。
- **解析成功率 6,898/7,048 = 97.9%**，覆盖 1,832/1,836 组成（99.8%）。
  失败 150 条如实记录于 `parse_failures.csv`（149 条 H 在共价半径×1.2 截断内无金属近邻，1 条空层）。
- 两处必要的口径修正（均如实记录）：
  1. H 半径取共价半径 0.31 Å（pymatgen atomic_radius=0.25 会漏掉 1.8–2.1 Å 的 bridge/hollow 近邻，312 例）；金属半径同 Phase5 r2 代码口径（atomic_radius）。
  2. db 中 star 与 Hstar 金属原子序 3,218/7,048 不一致 → 位移类特征先做同物种 Hungarian 配对准（最小像距离）。

## 18 维位点级特征（与 Phase5 清单同构，禁新增族）
位点锚定：`site_cn`（H 配位数）、`site_gcn`（Σ CN_j/12）、`h_height_A`（H−顶层金属高度）
近邻集合 N(H) 统计：`nn_cn_mean/min/std`、`nn_gcn_mean/min/std`、`nn_en_mean/std`（Pauling）、`nn_d_mean/std`（基态 d 电子）
层距/位移：`d12_relaxed_A`、`d12_relax_pct`（吸附诱导层距变化，参照为同 slab 洁净态而非理想原型——口径差已记录）、`surf_disp_mean/max_A`、`surf_rumpling_A`

与 Phase5 的 16 维差异：`relax_energy_drop_per_atom` 不可得（db 无弛豫前后能量）且属能量族（标签同源风险），以预注册族内 `nn_gcn_min/std` 补齐 18 维。

## 表面级结果（GroupKFold(5, comp)，10 种子，84 维 v3 基线，覆盖子集 1,832 行）
| 臂 | gold MAE | Δ vs 基线 | 种子方向 |
|---|---|---|---|
| 84+mean18 | 0.0920 | +0.0229 | 10/10 |
| 84+min18 | 0.0886 | +0.0263 | 10/10 |
| 84+mean18+min18（主候选） | 0.0900 | **+0.0249** | 10/10 |
| 3d 磁性子层（n=398，主候选臂） | — | **+0.0428** | 10/10 |

- 基线本子集 OOF 0.1150（全表契约基线 0.11374）。Δ=0.0249 ≈ oracle 缺口 0.033 eV 的 75%。
- 3d 子层 Δ=0.0428 远超 Phase 3 磁矩特征 +0.0173 / Phase 4 +0.0109。
- 嵌套 CV（外层 GroupKFold 5，内层 3 选参）：gold 0.0903 vs base 0.1140，5/5 折胜，Δ=0.0237。

## 位点级结果（开发集 5,859→5,730 记录，site lockbox 276 comps 冻结未触碰；GroupKFold(5,shuffle,rs=42) 同 F2/F3）
| 臂 | OOF MAE |
|---|---|
| 84 基线 | 0.2839 |
| 84+site one-hot（F3 复现臂，本行集） | 0.1206（F3 公布 0.1156，行集略异） |
| **84+geom18** | **0.1042** |
| 84+onehot+geom18 | 0.1010 |

- 真实几何位点特征优于离散位点元数据 one-hot；但 **≤0.090 目标未达成**。
- 位点→表面 min 聚合（seed 42）：0.1030，优于 F5 的 0.1094（兑现率口径）。

## 闸门（必过项，实测）
- **G1 特征可移植性：LEAK**（Δ_A=0.0249，Δ_B=−0.0040，保留率 −0.16 < 0.8）。
- **G2 最稳定位点选择耦合（F4 教训专项）：LEAK**（Δ_A=0.0295 标签选位点，Δ_B=0.0174 训练折 Ridge 预测能量选位点，保留率 0.59 < 0.8）。
- 判读（如实分层）：Ridge 线 B 检验"特征能否由 84 维组成描述符线性推断"。真实几何是**独立于组成的信息通道**——部署时对新表面需做一次原型 slab 弛豫（不涉及标签值），Ridge 代理在结构上无法模拟此通道；Phase 5 CHGNet 弛豫特征同样判 LEAK（r=−1.12），两次互证这是代理对几何通道的系统性误判而非特异性标签耦合。G2 保留率 0.59（vs F4 的 −0.036）说明 min 位点选择信号大部分可移植但未达 0.8 阈值。**冻结协议下正式判定仍为 LEAK，不作无条件采纳宣称。**

## 判定：contract_passed_gate_failed（部分过，分层结论）
1. 精度契约 + 嵌套 CV + 3d 子层：**全过**（真实 DFT 几何，无 CHGNet 伪影条款，信号强且方向全一致）。
2. 位点级：优于 F3/one-hot 基线，未达 ≤0.090。
3. 闸门：现行 Ridge 推断口径 LEAK → **不可记为 v5 候选**；如实记为"契约级阳性、闸门未过"的强条件信号。后续路径：(i) 闸门协议对"几何通道"补充部署成本口径（弛豫可得性而非组成线性可推断性）的修订需重新预注册；(ii) 以基座势弛豫近似部署通道做端到端复测。

## 产物（SHA256 见 sha256_manifest.json）
- `s1_parse_features.py` / `real_geom_site_features.csv`（6,898×18 + ID）/ `parse_failures.csv` / `s1_parse_report.json`
- `s2_eval.py` / `real_geom_eval.json` / `log_eval.txt` / `summary.json` / 本 README
