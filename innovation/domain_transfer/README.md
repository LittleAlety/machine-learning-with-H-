# Phase 4 Workstream L — 跨库域迁移（锚点协议）

预注册：`config/preregistered_phase4.yaml`（workstream_L；红线 16：锚点只进训练折，
禁止进入任何测试评估面）。数据口径复用
`innovation/functional_invariance/run_invariance.py` 的三域加载与 84 维特征对齐
（Mamun/BEEF-vdW n=1836、EqV2/RPBE n=491、CatHub/PBE n=203，口径自检 0 失配）。
主方案 B4 = 域指示 GBDT（84 描述符 + 2 域哑变量，XGBoost 锁死
`data_processed/hstar_best_params.json` 出厂 tuned 参数），10 种子
`[0,1,2,7,13,42,99,123,2024,31337]`。

## L0 四方案基线（B1/B2/B3/B4 同台）

见 `README_L0.md`（L0 代理主写）。

## L1 锚点效率曲线（`l1_anchor_curve.py` → `l1_curve.csv` / `l1_curve.png`）

协议：测试域 = EqV2（跨库主目标）；训练集 = Mamun + CatHub 全域 + 从测试域随机抽
n ∈ {0(参考), 5, 10, 20, 50} 个锚点（仅进训练折）；评估面 = 测试域其余行。

| n_anchor | MAE (eV, mean±sd) | Spearman ρ (mean±sd) |
|---|---|---|
| 0（参考） | 0.369 ± 0.006 | 0.235 ± 0.008 |
| 5  | 0.329 ± 0.024 | 0.238 ± 0.020 |
| 10 | 0.288 ± 0.030 | 0.228 ± 0.025 |
| 20 | 0.255 ± 0.020 | 0.221 ± 0.029 |
| 50 | 0.206 ± 0.026 | 0.259 ± 0.056 |

读法：MAE 近似线性随锚点数下降，50 锚点（测试域 ~10%）仍未到 pooled-CV 口径的
0.123 eV；排序 ρ 在 ≤20 锚点基本平坦（~0.22–0.24），仅 50 锚点略升——**锚点主要
修复幅度（域偏移），对排序的边际收益很小**，与"跨库误差主成分为域条件化偏移"
的解读一致。

## L2 六向迁移矩阵（`l2_matrix.py` → `l2_matrix.csv` / `l2_matrix.png`）

协议：六个有序方向 训练域→测试域；训练集 = 训练域全域 + 测试域 20 锚点（仅进
训练折）；第三域不参与（纯跨库）。10 种子均值：

| train → test | MAE (eV) | Spearman ρ |
|---|---|---|
| Mamun → EqV2   | 0.257 | 0.169 |
| Mamun → CatHub | 0.115 | **0.850** |
| EqV2 → Mamun   | **0.878** | 0.203 |
| EqV2 → CatHub  | **0.601** | 0.337 |
| CatHub → Mamun | 0.233 | **0.838** |
| CatHub → EqV2  | 0.179 | 0.218 |

失败格如实呈现：**EqV2 作为训练域的两个方向显著失败**（MAE 0.60–0.88 eV）——
EqV2 单域能量范围宽但组成覆盖窄（115 唯一组成），支撑不起外推；Mamun↔CatHub
两个方向排序极佳（ρ≈0.84–0.85）但 Mamun→CatHub 仍有 0.115 eV 系统性残差。
矩阵明显不对称：迁移难度由训练域覆盖度与测试域能量尺度共同决定。

## L3 机制分析（`l3_mechanism.py` → `l3_interactions.csv` / `l3_b3_check.json`）

SHAP 交互值（xgboost `pred_interactions=True`，全量三域池化 B4 模型，
mean |Φ_ij|）"域指示 × 描述符" Top5：

| rank | 交互对 | mean |Φ| (eV) |
|---|---|---|
| 1 | EqV2 指示 × **group_wmean** | 0.0323 |
| 2 | EqV2 指示 × **en_wmean** | 0.0204 |
| 3 | EqV2 指示 × melting_point_max | 0.0080 |
| 4 | Mamun 指示 × group_wmean | 0.0075 |
| 5 | EqV2 指示 × en_max | 0.0067 |

交互强度集中在 EqV2 指示与**族（group）/电负性（en）加权均值**之间——与 Phase 3
i2 中"电负性族跨域符号翻转最严重"的证据互洽：树模型正是通过域×电负性族交互
吸收跨泛函斜率翻转。

**关键对照 B3**（域指示 + 线性模型，无交互；同协议 pooled KFold(5) × 10 种子，
EqV2 OOF，折内均值填补 NaN）：

- B3 线性：MAE **0.2314 ± 0.0019 eV**，ρ 0.234
- B4 GBDT（同折重跑）：MAE **0.1226 ± 0.0050 eV**，ρ 0.485（与 Phase 3 记录一致）
- Δ(B3−B4) = +0.109 eV ≫ 0，B3 远未达 ~0.13 eV

**机制结论：未坍塌**（`mechanism_collapsed=false`）。无交互的线性域偏移模型只能
拿到 0.231 eV；B4 相对 B3 的 0.109 eV 增益必须来自域指示×描述符的非加性交互，
其载体即上表的 group/en 族。机制链闭环：i2 符号翻转（单轴失效）→ L3 交互定位
（高维联合映射）→ B3 对照排除纯加性解释。
