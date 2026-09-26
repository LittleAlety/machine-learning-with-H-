# H* 吸附能模型 v3（84 特征）— Web 端特征规范（feature_spec）

本文件是 `web/assets/predict.js` 与 `scripts/26_export_web_assets_v2.py` 共同遵守的**唯一事实来源**。任何一端修改特征逻辑，两端必须同步。
来源代码：`scripts/06_hstar_features.py`（基线 36 特征化）、
`scripts/23_descriptor_refine.py`（新 48 特征，文献 4 MagpieData 复刻）、
`scripts/05_fetch_hstar.py`（structure/facet 推断）。
v2（36 特征）版规范见 git 历史。

## 1. 输入与组成规范化

输入：化学式字符串，如 `Pt3Ti`、`CuPt3`、`Ag`。

- 解析正则：`([A-Z][a-z]?)(\d*)`，数字缺省 = 1。
- 规范化：① 元素符号按 ASCII 字典序排序；② 计量数除以最大公约数；③ 计量数 1 省略。
- 例：`TiPt3 → Pt3Ti`；`Cu2Pt2 → CuPt`。

## 2. 元素属性（13 项，来源 mendeleev 1.3.0；完整数值见 elements.json）

| 键 | 定义 | 单位 |
|---|---|---|
| `en` | `en_pauling`，Pauling 电负性 | Pauling 标度，无量纲 |
| `radius` | `atomic_radius`，原子半径 | pm |
| `group` | `group_id`，族号 | 族序数，无量纲 |
| `period` | `period`，周期 | 周期序数，无量纲 |
| `ie1` | `ionenergies[1]`，第一电离能 | eV |
| `d_el` | 电子组态尾部（最后一个 `]` 之后）正则 `(\d)d(\d*)` 全部 d 占据求和（缺省=1） | 电子数 |
| `melting_point` | `melting_point`，熔点。**Sn 因同素异形体 mendeleev 缺值，按 Magpie/文献手工值 505.08 K 补齐** | K |
| `mendeleev_number` | `mendeleev_number`，门捷列夫数 | 序数，无量纲 |
| `covalent_radius` | `covalent_radius`（= Pyykkö 双键半径），共价半径 | pm（Pyykkö 双键半径） |
| `n_valence` | `nvalence()`，价电子总数 | 电子数 |
| `nd_valence` | 最外层 d 亚层电子数（econf 解析；单占据写作 `5d` 缺省=1） | 电子数 |
| `n_unfilled` | Σ(亚层容量−占据)，对 econf 中部分占据亚层求和（容量 s=2/p=6/d=10/f=14） | 电子数 |
| `nd_unfilled` | 仅 d 亚层的 (10−占据) 求和 | 电子数 |

## 3. 84 个建模特征（顺序 = model.json feature_names 顺序）

组成解析得 `{元素: 计量数}`，元素列表 `els`（规范化后字母序），权重 `w_i = c_i / Σc`。

| # | 特征 | 定义 |
|---|---|---|
| 1 | `n_elements` | 元素个数 |
| 2–31 | `{prop}_{stat}` | 6 基线属性（en→radius→group→period→ie1→d_el）× 5 统计（wmean→max→min→range→wstd） |
| 32–34 | `structure_A1/L10/L12` | 结构 one-hot（推断规则见 §4） |
| 35–36 | `facet_101/111` | 晶面 one-hot |
| 37–71 | `{prop}_{stat}` | 7 新属性（melting_point→mendeleev_number→covalent_radius→n_valence→nd_valence→n_unfilled→nd_unfilled）× 5 统计（同 wmean→max→min→range→wstd 顺序） |
| 72–84 | `{prop}_mode` | 13 属性（6 基线 + 7 新，按 en→radius→group→period→ie1→d_el→melting_point→mendeleev_number→covalent_radius→n_valence→nd_valence→n_unfilled→nd_unfilled 顺序）的 mode 统计 |

5 种加权统计（对属性值向量 v，权重 w）：

- `wmean = Σ(v_i·w_i)/Σw_i`（numpy `np.average(v, weights=w)`）
- `max / min = 组元最大/最小值`；`range = max − min`
- `wstd = sqrt(Σ(w_i·(v_i−μ)²)/Σw_i)`，μ=wmean（加权，非无偏）

**mode 统计（众数，逐位定义供 JS 复现）**：取**计量数最大的组元**的该属性原始值；
若计量数并列（如 L1₀ 型 `CuPt` 1:1），取 `els` 中**字母序（ASCII）最靠前**的元素
（即规范化化学式中的首元素；例：`CuPt` 的 `en_mode = en(Cu) = 1.90`）。
单元素组成的 mode = 该元素属性值。

**泄漏列**：`site_spread` 禁止入模。ID 列：`comp, structure, facet, energy_eV, n_raw, site_spread`。

## 4. structure / facet 推断规则（与 v2 一致）

| 计量模式 | structure | facet |
|---|---|---|
| 单元素 | A1 | 111 |
| 双元素 3:1 | L12 | 111 |
| 双元素 1:1 | L10 | 101 |
| 其他 | 未知（one-hot 全 0，in_domain=false） | 未知 |

## 5. 模型与推理语义

- XGBoost 3.4.1 `XGBRegressor`，`learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8, random_state=42`，`objective=reg:squarederror`。
- 训练数据：`hstar_dataset_v2.csv` 全量 1836 行 × 84 特征（快照 `data_processed/hstar_features_v3_84feat.csv`）。
- 目标：`energy_eV`；派生 **ΔG_H\* = E_ads + 0.24 eV**。
- 推理（f32 语义，与 C++ 预测器位级一致）：
  1. `base_score` 取 `save_config()` 的 `learner_model_param.base_score`（带方括号字符串，去括号取 float32）；
  2. 内部节点 `float32(feature) < float32(split_condition)` 走 yes 否则 no，缺失走 missing；
  3. `acc = float32(acc + float32(leaf))`，自 `float32(base_score)` 逐树累加；
  4. 无 link 变换，acc 即 E_ads。

## 6. in_domain 判定

`in_domain = true` 当且仅当：① 全部元素在训练集 37 元素内（elements.json `training_elements`）；② 计量模式命中 §4 前三种。
否则仍输出预测，`in_domain=false` 并给 warnings（外推，仅供参考）。
