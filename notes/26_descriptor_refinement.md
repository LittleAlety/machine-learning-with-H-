# 26 描述符精准化：文献 4 MagpieData 高 SHAP 组成描述符的复刻与诚实检验

> 脚本：`scripts/23_descriptor_refine.py`（可重跑，`/usr/local/bin/python3`）；
> 数据：`data_processed/hstar_dataset_v2.csv`（1,836 行，1,836 个 comp 组，37 元素）；
> 产出：`outputs/desrefine_*.csv`；运行日志全程打印（含覆盖率、逐折 MAE、判定）。
> 公平契约：新特征只加在特征侧；fold 划分不变（GroupKFold(5, comp) 同一批 fold）；
> 嵌套 CV 外层 5 折 / 内层 4 折随机搜索 30 组拟合超参；改善 >0.005 eV 才采纳。

## 结论速览（最终判定）

| 判定 | 配置 | 依据 |
|---|---|---|
| **采纳** | 全量合并（+48 特征，84 维） | 嵌套 CV OOF MAE 0.1139，Δ=+0.0059 eV vs 锁定基线 0.1198（>0.005），且 vs 嵌套基线 Δ=+0.0079，5/5 外折逐折配对全部改善 |
| 边界（不单独采纳） | +门捷列夫数族 | 嵌套 Δ vs 嵌套基线 +0.0051（踩线），但 vs 0.1198 仅 +0.0031，且该族与既有 group/period 高度共线（\|ρ\|≤0.98），不主张单独采纳 |
| 阴性 | +熔点族 / +未填满电子族 / +价电子与共价半径族 / +mode 系列 | 单独加入均未过阈（详见下表） |

一句话机理：**熔点是金属内聚能（M–M 键强度）的独立代理，与既有 6 性质低共线（max |ρ|=0.79），携带真实增量信息；而门捷列夫数、未填满电子数、d 价电子数本质上是既有族号/周期/d 电子数的重排（|ρ|=0.95–1.00），单独加入无增量——组成信息并非完全饱和，但可挖的"新物理维度"只剩内聚能这一类。**

---

## 1 方法与新增描述符（文献 4 §4.3，notes/25 对照）

文献 4（HER/OER 手稿）经 SHAP 验证的 MagpieData 组成特征，用 mendeleev 零成本复刻为 7 种新元素性质：

| 新性质 | 定义 | Magpie 对应 | 覆盖率 |
|---|---|---|---|
| melting_point | 熔点 K（mendeleev） | MeltingT | 37/37（Sn 因同素异形体 mendeleev 缺值，按 Magpie/文献值 505.08 K 手工补齐并记录） |
| mendeleev_number | 门捷列夫数 | MendeleevNumber | 37/37 |
| covalent_radius | 共价半径 pm（Pyykkö） | CovalentRadius | 37/37 |
| n_valence | 价电子总数（mendeleev nvalence()） | NValence | 37/37 |
| nd_valence | 最外层 d 亚层电子数（econf 解析） | NdValence | 37/37 |
| n_unfilled | Σ(亚层容量−占据)，部分占据亚层 | NUnfilled | 37/37 |
| nd_unfilled | 仅 d 亚层的未填满数 | NdUnfilled | 37/37 |

实现细节（诚实记录）：
- mendeleev 把单占据亚层写作 "5d"（省略 1），初版正则漏配导致 La/Sc/Y/Lu 的 nd_valence/n_unfilled 错误，已修复（`(\d)([spdf])(\d*)`，缺省占据=1）并复测。
- 统计量：沿用现有加权方案 wmean/wstd/min/max/range × 7 性质 = 35；另加 **mode**（原子分数最大组元的属性值，并列取化学式字母序首元素）× 13 个性质（既有 6 + 新 7）= 13。合计 **48 个新特征，1,836 行 NaN 单元格 0**（见 `desrefine_coverage.csv`、`desrefine_new_features.csv`）。

## 2 逐配置结果

### 2.1 固定参数消融（XGBoost_tuned lr=0.03/depth=7/n=400/sub=0.8，seed=42，同一批 fold）

| 配置 | n_feat | OOF MAE (eV) | Δ vs 0.1198 |
|---|---|---|---|
| baseline_36 | 36 | 0.1198±0.0087（精确复现锁定值） | 0 |
| +熔点族 | 41 | 0.1160±0.0096 | +0.0038 |
| +门捷列夫数族 | 41 | 0.1167±0.0094 | +0.0031 |
| +未填满电子族 | 46 | 0.1195±0.0090 | +0.0003 |
| +价电子与共价半径族 | 51 | 0.1200±0.0087 | −0.0002 |
| +mode系列 | 49 | 0.1209±0.0104 | −0.0011 |
| **全量合并** | **84** | **0.1134±0.0102** | **+0.0064** |

### 2.2 嵌套 CV（外层 GroupKFold5 / 内层 GroupKFold4 随机搜索 30 组；fold 与基线一致）

| 配置 | 嵌套 OOF MAE (eV) | Δ vs 嵌套基线 | Δ vs 0.1198 | 过阈判定 |
|---|---|---|---|---|
| baseline_36 | 0.1218±0.0123 | 0 | −0.0020 | — |
| +熔点族 | 0.1174±0.0109 | +0.0044 | +0.0024 | 否 |
| +门捷列夫数族 | 0.1167±0.0106 | +0.0051 | +0.0031 | 边界（仅 vs 嵌套基线踩线，4/5 折改善） |
| +未填满电子族 | 0.1182±0.0128 | +0.0036 | +0.0016 | 否 |
| +价电子与共价半径族 | 0.1191±0.0099 | +0.0026 | +0.0007 | 否 |
| +mode系列 | 0.1215±0.0113 | +0.0002 | −0.0017 | 否 |
| **全量合并** | **0.1139±0.0113** | **+0.0079** | **+0.0059** | **是（双参照均过阈）** |

- 全量合并逐折配对 Δ（baseline−full）：[+0.0132, +0.0132, +0.0013, +0.0056, +0.0063]，**5/5 折全部为正**，非单折侥幸。
- 注意嵌套基线（0.1218）略高于锁定值 0.1198：内层重搜超参是另一种协议，因此同时报告两个参照系的 Δ；全量合并在两个参照系下均 >0.005 eV，判定稳健。
- 明细见 `desrefine_ablation.csv`、`desrefine_nested_cv.csv`、`desrefine_config_summary.csv`。

## 3 共线性诊断（新 48 特征 vs 既有 36 特征，Pearson）

|ρ|>0.9 的**新-旧**特征对共 15 个（`desrefine_collinearity.csv`），三类完全/近完全冗余：

| 新特征族 | 冗余对象 | 证据 |
|---|---|---|
| nd_valence × 5 统计 | d_el × 5 统计 | **ρ=1.0000**（5/5 统计完全相等；两者定义同源=最外层 d 电子数） |
| mendeleev_number | group/period | max ρ=0.983（group_max）；wmean ρ=0.946 —— 门捷列夫数≈族号+周期的重排序编码 |
| nd_unfilled | group | ρ=−0.967~−0.983（d 未填满数=10−d 电子数，过渡金属区与族号近似线性反相关） |

新信息真实存在的族：**熔点族与既有特征最大 |ρ| 仅 0.79**（melting_point_min vs en_min）、mode 系列最大 0.88（period_mode vs period_wmean）、covalent_radius 最大 0.77——这解释了为何只有含熔点/mode 的配置在消融中见效，而门捷列夫数族单独过阈不稳（其信息基线已含）。

## 4 SHAP 族分解（信息论视角；全量合并 84 特征全量训练，TreeExplainer）

| 族 | n_feat | mean\|SHAP\| 份额 | 族内最强特征（mean\|SHAP\|） |
|---|---|---|---|
| 基线-组成元素性质族（6 性质×5 统计） | 30 | 53.38% | group_wmean（0.2851） |
| 新-熔点族 | 5 | **20.80%** | melting_point_max（0.1056） |
| 新-mode系列 | 13 | 11.01% | group_mode（0.0480） |
| 新-未填满电子族 | 10 | 6.30% | nd_unfilled_wmean（0.0372） |
| 新-价电子与共价半径族 | 15 | 4.82% | covalent_radius_wmean（0.0102） |
| 新-门捷列夫数族 | 5 | 2.79% | mendeleev_number_wmean（0.0126） |
| 基线-结构/晶面族 | 5 | 0.88% | structure_L10 |
| 基线-n_elements | 1 | 0.02% | n_elements |

解读：**并非"组成信息已饱和"**——新特征合计分到 45.7% 的 mean|SHAP| 份额，其中熔点族一族即占 20.8%，超过结构/晶面族一个数量级以上，与文献 4"mean MeltingT 为 HER 最重要特征"的发现互证。机理上熔点是金属内聚能/体模量相关量，H* 吸附本质是 M–H 键与 M–M 键的竞争，故携带独立于电负性/半径/电离能的物理信息。但注意：份额来自全量拟合（含记忆成分），真实增量以嵌套 CV 的 +0.0059~0.0079 eV 为准——即新信息真实但**边际收益已小**（相对 0.12 eV 基线仅 ~5–7%）。

## 5 局限与后续

1. **增益幅度**：0.006–0.008 eV 的绝对改善小于 fold 间 std（~0.011），虽逐折配对 5/5 为正、双参照过阈，仍属边际改善；若项目其他管线（如 17 轮位点特征 +0.0175 eV）竞争特征预算，位点特征优先级更高。
2. **Sn 熔点手工补齐**（505.08 K）与 Tc 基线兜底值同为外部来源，已记录；删除 Sn 组成重跑的影响未评估。
3. **mode 的平局规则**（50/50 组成取字母序首元素）是任意约定；L1₀ 体系 mode 退化为"字母序首元素性质"，可能弱化其区分力。
4. SHAP 族份额基于全量训练，偏好高基数/高方差族，仅作机理解读，不作采纳依据。
5. 未做新特征间的逐步前向选择；全量合并的增益可能主要由熔点族+mode 系列贡献（两者单独 Δ 分别为 +0.0038/−0.0011，合并 +0.0064，存在非加和交互），后续可用前向选择定位最小有效子集。

---

## 6 采纳落地（scripts/26_export_web_assets_v2.py，2025-09-11 重跑可复现）

全量合并 84 特征配置已过公平契约（§2.2：嵌套 Δ=+0.0059 eV vs 0.1198、+0.0079 vs 嵌套基线、5/5 外折配对改善），落成正式模型 v3-84feat 并导出全部终端资产。

### 6.1 最终模型与精度

- 模型：XGBRegressor（lr=0.03, depth=7, n=400, sub=0.8, seed=42），全量 1,836 行 × 84 特征训练。
- **OOF MAE = 0.1134 eV**（GroupKFold(5, comp) 同一批 fold、固定 tuned 参数），精确复现 §2.1 消融值；对比基线 36 特征的 0.1198 eV，绝对改善 0.0064 eV（−5.3%）。
- 数据快照：`data_processed/hstar_features_v3_84feat.csv`（1,836 行 × 89 列 = 4 ID + n_raw + 84 特征 + 目标）。
- 自洽性：Python f32 树遍历与 booster.predict 位级一致（1,836/1,836）；`featurize84` 重建特征与训练侧 max|diff|=4.6e-13（浮点求和顺序级）。

### 6.2 资产清单（web/assets/，全部覆盖更新）

| 文件 | 内容 | 大小 |
|---|---|---|
| model.json | 400 棵树 + base_score=−0.39193830（f32，括号字符串已处理）+ 84 特征名 | 4.06 MB |
| elements.json | 39 元素（37 训练 + Ge/Sb）× 13 属性 + units/manual_overrides 说明；Sn.melting_point=505.08 K 手工值 | 13.3 KB |
| feature_spec.md | 84 特征完整规范；mode 平局规则：计量数并列取规范化化学式（ASCII 字母序）首元素 | 5.2 KB |
| predictions_all.json | 1,836 行 OOF 预测（**注意：旧版为全量拟合值，本版改为 OOF**） | 415 KB |
| candidates_top.json | 按 OOF \|ΔG_H*\| 重排 top-50；统计不可区分阈值改用新 OOF MAE 0.1134 | 11.1 KB |
| shap_global.json | 全量 SHAP top-15；top1=group_wmean(0.2851)，melting_point_max 升至第 2（0.1056） | 1.6 KB |
| descriptor_distributions.json | 新 top-12（group_wmean, melting_point_max, group_mode, nd_unfilled_wmean, melting_point_wmean, en_wmean, d_el_wmean, melting_point_wstd, mendeleev_number_wmean, en_min, en_wstd, covalent_radius_wmean）32-bin 分布 + 中文标签 | 8.4 KB |
| descriptor_stats.json | 同步扩到 84 特征（旧版 36，前端输入定位面板依赖，一并更新） | 9.4 KB |
| parity_cases.json | 20 用例（域内 15 + 域外 5：CrSb/GePt3/AuSb3 含域外元素、Cu2Pt 与 CoNiPt 计量不可推断），含逐位 84 特征向量 + Python 预测值，供 JS 奇偶校验 | 53.8 KB |

### 6.3 候选榜首变化（OOF 口径，旧 top-5 → 新 top-5）

| rank | 旧（36 特征，KFold OOF） | 新（84 特征，GroupKFold OOF） |
|---|---|---|
| 1 | CuPt3 (pred ΔG −0.0002) | **IrOs3 (+0.0000)**，true ΔG −0.0245 |
| 2 | CrPt3 (−0.0002) | Au3Rh (−0.0004) |
| 3 | AgNi (−0.0005) | BiZr (−0.0010，红旗：含 LOEO 高误差元素 Bi) |
| 4 | Zn3Zr (−0.0009) | CuPd3 (−0.0013，true ΔG +0.0140) |
| 5 | CdPd3 (−0.0018) | Hg3Ir (−0.0024) |

旧榜首 CuPt3、CrPt3、AgNi 在新模型 OOF 下**全部跌出 top-50**（新模型对 Pt 系近最优区给出了更保守的 ΔG）；新榜首 IrOs3 的 DFT 真值 ΔG=−0.0245 eV 也确在近最优区。注意旧榜为 KFold OOF、新榜为 GroupKFold OOF，口径差异与特征增益共同贡献排名变化。

### 6.4 遗留事项

- `parity_report.json` 仍为旧 36 特征模型的报告；JS 侧按 parity_cases.json 重跑奇偶校验由前端代理负责（predict.js/index.html 本轮未动）。
- SHAP 族结构印证 §4：熔点族（melting_point_max/wmean/wstd 均入 top-12）是 v3 的主要增量来源。
