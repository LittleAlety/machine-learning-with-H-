# 特征描述符表 — dataset_v2

输入：dataset_v1.csv（946 行）；输出：dataset_v2.csv（946 行 × 46 列）

## 标识列（非特征）

| 列 | 含义 |
|---|---|
| comp | 表面组成化学式（数字为下标计量数，如 Pt3Co） |
| facet | 晶面米勒指数（111/211 等） |
| term | L1_2 合金表面终止层标签（AA/AB/BB；none=无标签） |
| adsorbate | 吸附物种（CO / O） |
| energy_eV | 目标值：CatApp reaction_energy（eV，**数值越大=结合越强**，≈ −E_ads+常数） |
| dataset_src | 数据来源子库 |
| n_raw | 去重前原始条数 |

## 数值特征（组成统计）

6 种元素性质 × 5 种组合统计 = 30 个特征；权重 = 化学计量分数。

### 元素性质来源

- 主源：mendeleev（en_pauling / atomic_radius / group_id / period / ionenergies[1] / 电子组态）
- d_el：由电子组态解析稀有气体核心以外的 d 电子数（无角标按 1 计，如 Sc/Y/La 的 nd¹）
- 手工 lookup 兜底启用情况：**未启用**（全部属性 mendeleev 齐全）

### 特征清单

| 特征 | 化学含义 |
|---|---|
| n_elements | 组成元素个数（1=纯金属，2+=合金） |
| en_wmean | mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）；按化学计量分数加权平均（合金整体性质） |
| en_max | mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）；组元最大值（最极端组元的贡献） |
| en_min | mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）；组元最小值 |
| en_range | mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）；组元极差（max-min，合金性质差异/错配度） |
| en_wstd | mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）；按化学计量分数加权标准差（组元性质离散度） |
| radius_wmean | 原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）；按化学计量分数加权平均（合金整体性质） |
| radius_max | 原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）；组元最大值（最极端组元的贡献） |
| radius_min | 原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）；组元最小值 |
| radius_range | 原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）；组元极差（max-min，合金性质差异/错配度） |
| radius_wstd | 原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）；按化学计量分数加权标准差（组元性质离散度） |
| group_wmean | 元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）；按化学计量分数加权平均（合金整体性质） |
| group_max | 元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）；组元最大值（最极端组元的贡献） |
| group_min | 元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）；组元最小值 |
| group_range | 元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）；组元极差（max-min，合金性质差异/错配度） |
| group_wstd | 元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）；按化学计量分数加权标准差（组元性质离散度） |
| period_wmean | 元素周期（原子尺寸与 d 轨道延展性）；按化学计量分数加权平均（合金整体性质） |
| period_max | 元素周期（原子尺寸与 d 轨道延展性）；组元最大值（最极端组元的贡献） |
| period_min | 元素周期（原子尺寸与 d 轨道延展性）；组元最小值 |
| period_range | 元素周期（原子尺寸与 d 轨道延展性）；组元极差（max-min，合金性质差异/错配度） |
| period_wstd | 元素周期（原子尺寸与 d 轨道延展性）；按化学计量分数加权标准差（组元性质离散度） |
| ie1_wmean | 第一电离能 eV（表面向吸附物种给电子能力）；按化学计量分数加权平均（合金整体性质） |
| ie1_max | 第一电离能 eV（表面向吸附物种给电子能力）；组元最大值（最极端组元的贡献） |
| ie1_min | 第一电离能 eV（表面向吸附物种给电子能力）；组元最小值 |
| ie1_range | 第一电离能 eV（表面向吸附物种给电子能力）；组元极差（max-min，合金性质差异/错配度） |
| ie1_wstd | 第一电离能 eV（表面向吸附物种给电子能力）；按化学计量分数加权标准差（组元性质离散度） |
| d_el_wmean | 价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）；按化学计量分数加权平均（合金整体性质） |
| d_el_max | 价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）；组元最大值（最极端组元的贡献） |
| d_el_min | 价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）；组元最小值 |
| d_el_range | 价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）；组元极差（max-min，合金性质差异/错配度） |
| d_el_wstd | 价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）；按化学计量分数加权标准差（组元性质离散度） |

## 结构/类别特征（one-hot）

| 特征 | 化学含义 |
|---|---|
| facet_111 | 晶面（表面原子排列/配位数，(211) 为台阶面更活泼）：= 111 |
| facet_211 | 晶面（表面原子排列/配位数，(211) 为台阶面更活泼）：= 211 |
| term_AA | 表面终止层（L1_2 合金表层为纯 A、混合 AB 或纯 B，决定吸附位点局域组成）：= AA |
| term_AB | 表面终止层（L1_2 合金表层为纯 A、混合 AB 或纯 B，决定吸附位点局域组成）：= AB |
| term_BB | 表面终止层（L1_2 合金表层为纯 A、混合 AB 或纯 B，决定吸附位点局域组成）：= BB |
| term_none | 表面终止层（L1_2 合金表层为纯 A、混合 AB 或纯 B，决定吸附位点局域组成）：= none |
| ads_CO | 吸附物种（CO 或 O，化学亲和性不同）：= CO |
| ads_O | 吸附物种（CO 或 O，化学亲和性不同）：= O |

数值特征 30 个 + one-hot 8 个 + n_elements = 39 个模型输入特征。

## 已知共线性局限（facet / term × adsorbate）

结构 one-hot 与吸附种高度共线：
- facet：CO 共 334 行（(111) 322 / (211) 12，后者全部为纯金属）；O 共 612 行，全部位于 (211)。
- term：CO 共 334 行全部 term=none；O 的 term=none 行为 0（其余全部带 AA/AB/BB 标签）。
因此 `term_none` 与 `ads_CO` 完全共线、`facet_111` 近乎共线，模型/SHAP 中 term_none / facet_111 的重要性**不能解读为纯终止层/晶面效应**（混杂吸附种效应）。若要晶面/终止层结论需补 CO(211)（尤其合金）/O(111) 数据。