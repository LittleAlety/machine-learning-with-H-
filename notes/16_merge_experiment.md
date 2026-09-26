# 16 — CatHub 同质金属 HER 增量合并重训实验（Round 4 Agent E）

> 核心问题：把 extval2_homogeneous.csv（Catalysis-Hub 同质金属 HER，
> 355 表面）并入 Mamun 主线训练，能否改善 H* 主线模型？
> 公平性契约（plan_round4.md）：新数据只进训练集不进测试集；
> 采用阈值 0.005 eV；种子 42；site_spread 禁入；不改 01-15 产物。

## 1. 重叠分析（outputs/merge_overlap_analysis.csv）

- 355 表面按 comp 聚合（min E_true，最稳定位点口径，同主线）→ **67 个唯一组成**：与 Mamun 1,836 重叠 **47** 个，纯新增 **20** 个
- 能量分布：Mamun mean=-0.392±0.551 eV（range [-2.31, 2.74]）；纯新增 mean=-0.614±0.481 eV（range [-1.13, 0.44]）——新增整体偏负（强结合侧），落在 Mamun 分布范围内
- 重叠组成系统偏差：E_CatHub − E_Mamun mean=-0.183 eV（median|Δ|=0.240 eV）→ 泛函/参考态差异显著，**重叠组成 Mamun 优先**的合并策略合理（保持域内单源口径）
- 纯新增 20 组成来源：AlonsoStrain2023 11、HansenFirst2018 8、KaiData-driven2022 1；泛函分箱：{'PBE': 11, 'BEEF-vdW': 8, 'MS2': 1}
- 新增元素：Mg/Nd/Sm/Si 不在 Mamun 37 元素内（Nd/Sm 的 group 用 HAND_LOOKUP 兜底，同 14；Mg/Si 由 mendeleev 齐全覆盖）

## 2. 合并变体

- **(a) 朴素合并**：沿用 36 特征；新增组成 structure/facet one-hot 按 06/14 推断规则（单元素→structure_A1=1，合金→structure 全 0；facet 精确 111/101→对应 one-hot，应变/台阶/hcp0001 等→全 0）
- **(b) 合并 + 泛函 one-hot**：dftFunctional 分箱 BEEF-vdW/PBE/RPBE/MS2/其他（前缀匹配：PBE+D3/PBEsol→PBE，RPBE-D3/RPBE_*VSHE→RPBE，BEEF-vdW_*VSHE→BEEF-vdW）；Mamun 行全部标 BEEF-vdW

## 3. 评估结果（outputs/merge_comparison.csv）

主口径：fold 仅由原始 1,836 组成的 GroupKFold(5, comp) 决定，新增组成永远加入各折训练集、从不进测试集；固定出厂参数 {'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 400, 'subsample': 0.8}，种子 42。

| 变体 | 口径 | MAE (eV) | Δ vs 基线 |
|---|---|---|---|
| baseline_mamun_only | 主口径 | 0.1198±0.0087 | +0.0000 |
| a_naive_merge | 主口径 | 0.1209±0.0100 | -0.0011 |
| b_merge_func_onehot | 主口径 | 0.1206±0.0100 | -0.0009 |
| a_naive_merge | 次口径_全合并GCV | 0.1238±0.0111 | -0.0040 |
| b_merge_func_onehot | 次口径_全合并GCV | 0.1243±0.0110 | -0.0045 |

- 基线（纯 Mamun，主口径）MAE = 0.1198±0.0087 eV，与契约基线 0.1198±0.0087 完全一致（协议复现）

## 4. 深度检查

所有变体主口径改善均未超过 0.005 eV 阈值，按契约**不触发**嵌套 CV / LOEO / EqV2 / 文献锚点复测。

## 5. 结论

**不采纳合并**：
- 变体 a_naive_merge 主口径 Δ = -0.0011 eV，未达 0.005 阈值
- 变体 b_merge_func_onehot 主口径 Δ = -0.0009 eV，未达 0.005 阈值
- 判读：20 个纯新增组成相对 1,836 行的 Mamun 主线体量过小（约 1%），且多为应变/非密排面/域外元素（Mg/Nd/Sm/Si）表面，与原域（A1/L1₀/L1₂ 密排面）分布差异大；重叠组成的系统性泛函偏差（mean -0.183 eV）也说明跨源标签不可直接混用。
- 主线模型与候选排序保持不变（未触碰任何 01-15 产物）。

## 6. 产物清单

- outputs/merge_overlap_analysis.csv（67 组成逐条：重叠判定、能量对比、合并决策）
- outputs/merge_comparison.csv（基线/变体(a)/(b) × 主/次口径 fold 级 MAE 与 Δ）
- 本报告：notes/16_merge_experiment.md