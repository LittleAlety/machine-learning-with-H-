# H* 吸附能模型 — Web 端特征规范（feature_spec）

本文件是 `web/assets/predict.js` 与 `scripts/15_export_web_assets.py` 共同遵守的
**唯一事实来源**（single source of truth）。任何一端修改特征逻辑，两端必须同步。
来源代码：`scripts/06_hstar_features.py`（特征化）、`scripts/05_fetch_hstar.py`
（structure/facet 推断）、`scripts/07_hstar_models.py`（建模特征集与参数）。

## 1. 输入与组成规范化

输入：化学式字符串，如 `Pt3Ti`、`CuPt3`、`Ag`。

- 解析正则：`([A-Z][a-z]?)(\d*)`，数字缺省 = 1。
- 规范化（canonicalization）：
  1. 按元素符号字典序（ASCII）排序；
  2. 所有计量数除以它们的最大公约数（gcd）；
  3. 计量数 1 省略数字。
- 例：`TiPt3 → Pt3Ti`；`Cu2Pt2 → CuPt`；`Pt2Ti6 → PtTi3`。

## 2. 元素属性（6 项，来源 mendeleev 1.2.0）

| 键 | mendeleev 字段 | 含义 |
|---|---|---|
| `en` | `en_pauling` | Pauling 电负性 |
| `radius` | `atomic_radius` | 原子半径（pm） |
| `group` | `group_id` | 族号 |
| `period` | `period` | 周期 |
| `ie1` | `ionenergies[1]` | 第一电离能（eV） |
| `d_el` | 由 `econf` 解析 | 价层 d 电子数 |

`d_el` 解析规则：取电子组态字符串最后一个 `]` 之后的尾部，用正则 `(\d)d(\d*)`
匹配所有 `nd^x` 项，x 缺省按 1 计，求和。例：Pt `[Xe] 4f14 5d9 6s` → 9；
Sc `[Ar] 3d 4s2` → 1；Zn `[Ar] 3d10 4s2` → 10。

v2 数据集 37 种元素的全部属性 mendeleev 均齐全，**未启用**手工兜底
（HAND_LOOKUP 仅含 Tc，且实际未触发）。完整数值表见 `elements.json`。

## 3. 36 个建模特征（顺序 = 模型 feature_names 顺序）

组成解析得 `{元素: 计量数}`，元素列表 `els`（规范化后的字母序），
计数 `c_i`，权重 `w_i = c_i / Σc`（浮点逐步归一化）。

| # | 特征 | 定义 |
|---|---|---|
| 1 | `n_elements` | 元素个数 |
| 2–31 | `{prop}_{stat}` | 6 属性 × 5 统计（见下），属性顺序 en→radius→group→period→ie1→d_el，统计顺序 wmean→max→min→range→wstd |
| 32–34 | `structure_A1 / structure_L10 / structure_L12` | 结构 one-hot |
| 35–36 | `facet_101 / facet_111` | 晶面 one-hot |

5 种统计（对属性值向量 v，权重 w，n ≤ 3）：

- `wmean = Σ(v_i·w_i) / Σw_i`（即 numpy `np.average(v, weights=w)`）
- `max / min = 组元最大/最小值`
- `range = max − min`
- `wstd = sqrt( Σ(w_i·(v_i−μ)²) / Σw_i )`，其中 μ = wmean（加权，**非**无偏）

**泄漏列**：`site_spread`（同表面不同吸附构型的能量极差，含目标值本身）
**禁止入模**，仅作分析列。ID 列：`comp, structure, facet, energy_eV, n_raw,
site_spread`（42 列 = 6 ID + 36 特征）。

## 4. structure / facet 推断规则（源自 05 的 12 原子超胞计量模式）

由规范化组成的计量模式唯一确定：

| 计量模式 | structure | facet | one-hot |
|---|---|---|---|
| 单元素（M12） | `A1` | `111` | structure_A1=1, facet_111=1 |
| 双元素 3:1（A9B3） | `L12` | `111` | structure_L12=1, facet_111=1 |
| 双元素 1:1（A6B6） | `L10` | `101` | structure_L10=1, facet_101=1 |
| 其他（如 2:1、三元素） | 未知 | 未知 | 全部 one-hot = 0，`in_domain=false` |

已知恒等关系：`structure_L10 ≡ facet_101`，`structure_A1 + structure_L12 ≡ facet_111`。

## 5. 模型与推理语义

- 模型：XGBoost 3.4.1 `XGBRegressor`，参数
  `learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8,
  random_state=42`（其余默认，`objective=reg:squarederror`）。
- 训练数据：`hstar_dataset_v2.csv` 全量 1836 行 × 36 特征。
- 目标：`energy_eV`（最稳定位点 H* 吸附能 E_ads，eV，负值=放热）。
- 派生：**ΔG_H\* = E_ads + 0.24 eV**（标准 HER 自由能校正）。
- 推理（与 C++ 实现对齐，纯前端可复现到 0 误差）：
  1. `base_score` 取 `booster.save_config()` 的
     `learner.learner_model_param.base_score`（3.x 为带方括号字符串，如
     `"[-3.919383E-1]"`，去括号取 float32）；
  2. 树遍历：内部节点比较 `float32(feature) < float32(split_condition)`，
     成立走 `yes`，否则走 `no`；缺失值（NaN/null）走 `missing`
     指定的 nodeid（本模型特征无缺失，逻辑仍完整实现）；
  3. 累加：`acc = float32(acc + float32(leaf))`，从 `acc=float32(base_score)`
     开始按树顺序逐棵累加（**float32 逐步舍入是关键**，与 XGBoost C++ 预测器
     的 bst_float 累加一致；用 float64 累加会产生 ~1e-7 偏差）；
  4. `reg:squarederror` 无 logistic/link 变换，acc 即为 E_ads 预测值。
- JSON dump 中所有 `split_condition` 与 `leaf` 均已验证为 float32 精确可表示。

## 6. in_domain 判定（页面提示用）

`in_domain = true` 当且仅当：
1. 组成可解析且所有元素都在训练集 37 元素之内（见 `elements.json` 的
   `training_elements`）；
2. 计量模式可推断结构（§4 前三种之一）。

否则仍输出预测值，但 `in_domain=false` 且 `warnings` 说明原因（外推，仅供参考）。
