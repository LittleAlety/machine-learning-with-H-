# 08 — H* 主线模型深度优化报告（SPEC 第 8 节，scripts/12_optimize_hstar.py）

## 1. 设置与口径

- 数据：hstar_dataset_v2.csv（1,836 行 × 41 特征，已去泄漏，site_spread 禁入），种子 42
- 搜索：随机采样 160 组（空间同 SPEC：max_depth[3-12]、min_child_weight[1-20]、gamma[0-0.5]、subsample/colsample_bytree/colsample_bylevel[0.5-1.0]、reg_alpha/reg_lambda[1e-3-10 log]、learning_rate[0.01-0.2]），n_estimators 上限 2000 + early stopping（50 轮，内层 15% 验证折）；因 sklearn SearchCV 无法逐折传 eval_set，采用同分布手写采样循环（等价 RandomizedSearchCV）
- 评估协议：所有搜索在内层 5-fold（种子 42）内进行；最终无偏估计用嵌套 CV（§5）

## 2. 超参搜索轨迹（figures/opt_search_trajectory.png）

- 前 20 组累计最优 MAE = 0.1342 eV；80 组后 0.1330；最终 0.1288 eV（第 134 组）
- 后 80 组仅再降 0.0042 eV，搜索已收敛（收益递减明显）
- 最优配置（内层 CV MAE=0.1288±0.0087，平均最优迭代数 341）：

| 参数 | 值 |
|---|---|
| max_depth | 11 |
| min_child_weight | 10.812729 |
| gamma | 0.0476 |
| subsample | 0.863922 |
| colsample_bytree | 0.988332 |
| colsample_bylevel | 0.659163 |
| reg_alpha | 0.069958 |
| reg_lambda | 0.050016 |
| learning_rate | 0.01977 |
| n_estimators（全量收尾，early stopping 选定） | 306 |

## 3. 样本权重实验（同一最优超参 + 同一随机 5-fold，评估 MAE 不加权）

| 权重方案 | 随机CV MAE (eV) |
|---|---|
| uniform | 0.1289±0.0086 |
| n_raw | 0.1225±0.0103 |
| inv_comp_freq | 0.1289±0.0086 |

- 结论：三方案差异展开仅 0.0064 eV
- 最优方案 = **n_raw**；n_raw 加权假设『构型数多→均值更可靠』，但本数据集每行已是单表面最低能量（非均值），n_raw 与噪声水平无直接对应；组成频次逆加权（w = 1/该 comp 组内样本数）抑制被过多表面（多晶面/多结构）过度代表的组成，理论上利于稀有化学的权重均衡，实测未带来超出折间波动的改善

## 4. Stacking 融合 vs 单模型（同折 OOF）

| 模型 | OOF MAE (eV) |
|---|---|
| XGB_opt | 0.1207 |
| RandomForest | 0.1222 |
| Stacking | 0.1227 |

- Stacking（XGB* + RF + LinearRegression 元学习器）相对单 XGB：Δ = +0.0020 eV。融合无增益甚至略差：XGB 与 RF 误差高度相关，线性元学习器无可提取的互补信息，且元特征仅 2 维、XGB 本身已占主导权重。
- 最终模型取**单模型 XGB（最优配置）**：更简单、可解释（SHAP 管线兼容）、嵌套 CV 已对其做无偏验证

## 5. 诚实评估：四口径最终对比（基线 → 优化）

### 5a. 嵌套 CV（外层 5-fold，内层各折随机搜索 60 组，搜索全程不接触外层测试折）

| 外层折 | 内层最优MAE | 外层MAE(优化) | 外层MAE(基线同折) |
|---|---|---|---|
| 1 | 0.1352 | 0.1219 | 0.1157 |
| 2 | 0.1349 | 0.1123 | 0.1041 |
| 3 | 0.1306 | 0.1285 | 0.1198 |
| 4 | 0.1319 | 0.1369 | 0.1263 |
| 5 | 0.1309 | 0.1377 | 0.1355 |

- **嵌套 CV 汇集 MAE（无偏估计）= 0.1274 eV**；基线（07 固定参数）同外层折对照 = 0.1203 eV

### 5b. 四口径汇总

| 评估口径 | 基线 MAE (eV) | 优化模型 MAE (eV) | Δ |
|---|---|---|---|
| 嵌套CV（外层5-fold无偏估计） | 0.120 | 0.127 | +0.007 |
| 随机5-fold CV | 0.120 | 0.121 | +0.001 |
| GroupKFold(规范化comp) | 0.120 | 0.127 | +0.007 |
| LOEO（按n_test加权汇集） | 0.155 | 0.153 | -0.002 |
| 外部验证 EqV2 ALL | 0.347 | 0.330 | -0.017 |

- GroupKFold 明细 MAE=0.1265±0.0092；LOEO 最差 3 元素：Mn 0.480、Fe 0.283、Bi 0.262
- 外部验证（EqV2，零重训，复用 11 全流程）：MAE=0.3297 eV，系统偏差 mean(pred−true)=+0.1397 eV，去偏差平移后 MAE=0.2861 eV（baseline：0.347 / +0.158 / 0.313）
- 系统偏差仍为正（整体高估吸附能 ~0.1-0.16 eV），与 11 结论一致：主要源于泛函/参考态差异（BEEF-vdW vs RPBE）与域外晶面/元素（Sb），**超参优化不能消除跨库系统偏差**——它是数据分布问题，不是模型容量问题

## 6. 与基线的诚实结论

- 嵌套 CV：0.1274 vs 基线同折 0.1203（Δ = +0.0072 eV）；对 07 报告口径（随机CV 0.120）Δ = -0.0074 eV
- 嵌套 CV 下优化模型反而略差（-0.0074 eV）：全数据搜索的最优配置存在搜索过拟合，嵌套评估予以纠正——这正是铁律 (a) 的意义。

## 7. 候选刷新

- 嵌套 CV MAE 0.1274，较基线 0.120 提升 -0.0074 eV（未超过 0.003 阈值），**保留 08 原候选清单**（candidate_rankings_hstar.csv 与 hstar_candidates.png 未改动）。

## 8. 产物清单

- data_processed/opt_search_results.csv（160 组全记录）/ opt_sample_weights.csv / opt_stacking_compare.csv / opt_nested_cv.csv / opt_loeo_results.csv / opt_final_metrics.csv / opt_extval_results.csv / opt_extval_metrics.csv / opt_best_params.json
- figures/opt_search_trajectory.png / opt_extval_parity.png
- 未触碰任何 01-11 产物