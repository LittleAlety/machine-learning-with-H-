# 12 — H* 主线模型深度优化报告（SPEC 第 8 节，scripts/12_optimize_hstar.py）

> 本轮为 GroupKFold 口径重做版：搜索/权重/stacking/嵌套 CV 全部统一
> GroupKFold（groups=规范化 comp），替代旧版随机 KFold 草稿
>（旧随机 KFold 草稿产物已归档 deprecated/opt_*，本轮未引用、未覆盖）。

## 1. 设置与口径

- 数据：hstar_dataset_v2.csv（1,836 行 × 36 特征；site_spread 泄漏列已排除，仅作分析列），种子 42
- 每个 comp 在数据集中仅 1 行，GroupKFold(comp) 等价于组大小为 1 的划分——与 07 的 GroupKFold 口径完全一致，仍按组协议执行
- 搜索：随机采样 150 组（≥150），空间：max_depth[3-10]、min_child_weight[1-20]、gamma[0-0.5]、subsample/colsample_bytree/colsample_bylevel[0.5-1.0]、reg_alpha/reg_lambda[1e-3-10 log均匀]、learning_rate[0.01-0.3 log均匀]；n_estimators 上限 2000 + early stopping（50 轮，验证折用 GroupShuffleSplit 按组切 15%，组级零泄漏）。因 sklearn SearchCV 无法逐折传 eval_set，采用同分布手写采样循环（等价 RandomizedSearchCV）

## 2. 超参搜索（figures/opt_search_trajectory.png）

- 150 组全程 GroupKFold(5) 评估；最终最优 MAE = 0.1309 eV（第 134 组），基线为 0.1198 eV
- 前 50 组累计最优 0.1354；后 100 组仅再降 0.0045 eV，搜索已收敛（收益递减）
- **口径注意**：搜索阶段为兼容 early stopping，评估模型只在每折的 85% 子折上训练，数值相对『满折训练』的 07 基线系统性偏高约 0.005-0.01 eV，二者不能直接比较；协议对齐的比较见 §4（XGB_opt，满折+固定迭代数）与 §5（嵌套 CV 同折对照）
- 最优配置（GroupKFold MAE=0.1309±0.0089，平均最优迭代数 406，全量收尾 449）：

| 参数 | 值 |
|---|---|
| max_depth | 9 |
| min_child_weight | 10.812729 |
| gamma | 0.0476 |
| subsample | 0.863922 |
| colsample_bytree | 0.988332 |
| colsample_bylevel | 0.659163 |
| reg_alpha | 0.069958 |
| reg_lambda | 0.050016 |
| learning_rate | 0.011911 |

## 3. 样本权重实验（同一最优超参 + 同一 GroupKFold，评估不加权）

| 权重方案 | GroupKFold MAE (eV) |
|---|---|
| uniform | 0.1309±0.0089 |
| n_raw | 0.1250±0.0081 |
| inv_comp_freq | 0.1313±0.0095 |

- 三方案展开差 0.0063 eV
- 说明：v2 每个 comp 仅 1 行，按 comp 行频逆加权退化为 uniform，故 inv_comp_freq 实现为**组成元素频次逆加权**（w = 1/组内元素在全库的平均出现频次，归一化），用于抑制热门元素化学的过度代表
- 最优方案 = **n_raw**，用于最终模型训练

## 4. Stacking 融合（XGB + RF + Ridge 元学习器，组感知手写两层）

| 模型 | GroupKFold OOF MAE (eV) |
|---|---|
| XGB_opt | 0.1260 |
| RandomForest | 0.1222 |
| Stacking | 0.1233 |

- Stacking 相对单 XGB：Δ = -0.0027 eV。融合增益 ≤0.005，**最终模型取单模型 XGB**：更简单、SHAP 管线兼容。XGB 与 RF 误差高度相关，Ridge 元学习器可提取的互补信息有限。
- 组感知实现：元特征由外层训练折内部的 GroupKFold(4) OOF 生成，杜绝元特征泄漏

## 5. 嵌套 CV 无偏估计（外层 GroupKFold(5) × 内层搜索 50 组）

| 外层折 | 内层最优MAE | 外层MAE(优化) | 外层MAE(基线同折) |
|---|---|---|---|
| 1 | 0.1319 | 0.1330 | 0.1297 |
| 2 | 0.1367 | 0.1211 | 0.1152 |
| 3 | 0.1409 | 0.1235 | 0.1127 |
| 4 | 0.1290 | 0.1422 | 0.1309 |
| 5 | 0.1388 | 0.1208 | 0.1105 |

- **优化流程无偏 MAE = 0.1281±0.0093 eV**；基线（07 固定参数）同外层折对照 = 0.1198±0.0097 eV
- LOEO（最优配置）汇集 MAE = 0.1539 eV（基线 0.1547）；最差 3 元素：Mn 0.484、Fe 0.281、Bi 0.257

## 6. 外部验证（EqV2，零重训，复用 11 全流程）

- ALL：MAE = 0.3318 eV，系统偏差 mean(pred−true) = +0.1423 eV，去偏差平移后 MAE = 0.2909 eV
- 基线（07 模型，11 报告）：MAE = 0.3470 / bias +0.158 / 去偏差 0.313
- 系统偏差仍显著为正，与 11 结论一致：主要源于泛函/参考态差异（BEEF-vdW vs RPBE）与域外晶面/元素（Sb）——**超参优化不能消除跨库系统偏差**，这是数据分布问题而非模型容量问题

## 7. 汇总对比（outputs/opt_comparison.csv）

| 阶段 | 设置 | 协议 | MAE (eV) |
|---|---|---|---|
| 基线 | XGBoost_tuned（07 小网格） | GroupKFold(规范化comp) | 0.1198 |
| 超参搜索 | XGB 最优配置（150组搜索） | GroupKFold(规范化comp) | 0.1309 |
| 样本权重 | uniform | GroupKFold(规范化comp) | 0.1309 |
| 样本权重 | n_raw | GroupKFold(规范化comp) | 0.1250 |
| 样本权重 | inv_comp_freq | GroupKFold(规范化comp) | 0.1313 |
| Stacking对比 | XGB_opt | GroupKFold(规范化comp) OOF | 0.1260 |
| Stacking对比 | RandomForest | GroupKFold(规范化comp) OOF | 0.1222 |
| Stacking对比 | Stacking | GroupKFold(规范化comp) OOF | 0.1233 |
| 嵌套CV无偏估计 | 优化流程（内层搜索50组） | 外层GroupKFold(5)+内层GroupKFold(4) | 0.1281 |
| 嵌套CV无偏估计 | 基线（07固定参数，同折） | 外层GroupKFold(5) | 0.1198 |
| LOEO | XGB 最优配置 | 留一元素外测（按n_test加权） | 0.1539 |
| 外部验证 | 最终模型（零重训） | EqV2 ALL | 0.3318 |

## 8. 与基线的诚实结论

- 基线 GroupKFold MAE = 0.1198 eV；嵌套 CV 无偏 MAE = 0.1281 eV；差值 = -0.0083 eV（正=优化有效）
- 嵌套 CV 下优化流程反而差 0.0083 eV：全数据搜索选出的最优配置存在搜索过拟合，嵌套评估予以纠正——这正是诚实评估铁律的意义。

## 9. 候选刷新

- 差值 -0.0083 eV 未超过 0.005 阈值，**保留 08 原候选排序**（candidate_rankings_hstar.csv 与 figures/hstar_candidates.png 未改动）。原因：优化收益边际/为负，用无显著差异的模型重排候选只会引入噪声，不提供新信息。

## 10. 产物清单

- outputs/opt_search_results.csv（150 组全记录）/ opt_sample_weights.csv / opt_stacking_compare.csv / opt_nested_cv.csv / opt_loeo_results.csv / opt_comparison.csv / opt_extval_results.csv / opt_extval_metrics.csv / opt_best_params.json
- figures/opt_search_trajectory.png / opt_extval_parity.png
- 未触碰任何 01-11 产物