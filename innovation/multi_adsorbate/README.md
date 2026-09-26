# Workstream C-1：多吸附种统一吸附能引擎（plan-hb v2.0）

统一清洗口径下 15 个吸附种 × 16,499 条组成级吸附能记录，逐吸附种建模 + 跨吸附种迁移实验 + HER×ORR 双判据演示。规模对比：Chem. Sci. 2026, 17, 783（doi 对照条目）覆盖约数千条单吸附种记录，本引擎为 15 吸附种 1.65 万条的统一口径数据集，规模约为其数倍。

## 产物
| 文件 | 内容 |
|---|---|
| clean_report.md | 数据体检：每吸附种记录数/唯一组成/能量范围；min 聚合口径；2H/2N/2O 单位式说明 |
| per_adsorbate_models.py | 逐吸附种 XGBoost（v3 同超参 lr=0.03/depth=7/n=400/sub=0.8） |
| per_adsorbate_mae.csv | GroupKFold(5, seed42) × 3 种子 OOF MAE±std |
| per_adsorbate_parity.png | 大档 8 吸附种 2×4 parity（300dpi 暖色系） |
| transfer_experiment.py | 硬共享 MLP(128-64, 逐吸附种头, one-hot 身份) vs 单任务 GBDT，同折同 10 种子 |
| transfer_raw.csv / transfer_verdict.json | 原始 MAE 与公平契约裁决 |
| dual_volcano.py / dual_volcano.png / dual_candidates.csv | HER×ORR 双判据演示（**演示口径，非定量结论**） |
| multi_adsorbate_summary.json | 机器可读汇总 |

## 大档 8 吸附种 OOF MAE（eV，3 种子均值±std）
| 吸附种 | n | MAE |
|---|---|---|
| H | 1836 | 0.121±0.000 |
| S | 1806 | 0.164±0.000 |
| O | 1795 | 0.223±0.001 |
| 2H | 1802 | 0.236±0.001 |
| N | 1796 | 0.278±0.001 |
| C | 1776 | 0.307±0.001 |
| 2O | 1680 | 0.441±0.002 |
| 2N | 1761 | 0.549±0.003 |

小档 7 个（n=243–352，小样本警示）：H2O 0.077 / CH3 0.156 / NH 0.206 / OH 0.210 / CH 0.222 / SH 0.224 / CH2 0.280。
阴性记录：2N/2O MAE 显著偏高（0.44–0.55 eV），与二者能量跨度大（std≈3.1 eV）一致，如实保留。

## 迁移裁决（公平契约：改善>0.005 eV 且 10/10 种子方向一致）
**跨吸附种迁移红利在数据共享不在表示共享。** 硬共享 MLP 联合训练在全部 15 个吸附种上 10/10 种子一致**劣于**单任务 GBDT（小档平均差 −0.048 eV，大档 −0.081 eV，即 NN 更差），契约未满足、方向为负。该阴性结果与"NN 第二身份输给 GBDT"预期一致，仍成文。小档吸附种并未从联合表示中获得显著红利；统一描述符与统一清洗口径本身（数据共享）才是多吸附种引擎的价值所在。

## 双判据演示（演示口径，非定量假设）
- ΔG_H* = E_ads(H) + 0.24 eV；ΔG_OH* = E_ads(OH) + 3.2 eV；ORR 最优锚 0.9 eV（文献火山口径）。
- 双优区 |ΔG_H*|≤0.1 且 |ΔG_OH*−0.9|≤0.3：HER 窗口 247 个组成、ORR 窗口 250 个组成，**交集为 0**——H 与 OH 结合强度此消彼长，当前组成池中无双优候选，如实记录；dual_candidates.csv 列出最接近双优区的 Top-20（score = max(|ΔG_H|/0.1, |ΔG_OH−0.9|/0.3)）。
- 前 5 名：LaMn(L10-101)、Cr3Hf(L12-111)、MnY(L10-101)、RhY3(L12-111)、FeLa(L10-101)。
- 注：OH* 训练集仅 243 组成，池外组成的 ΔG_OH* 为模型外推（csv 中 oh_in_training 标记），演示性质。

## 复现
```
cd innovation/multi_adsorbate
/usr/local/bin/python3 per_adsorbate_models.py   # ~12 min (3 workers)
/usr/local/bin/python3 transfer_experiment.py    # ~60 min (3 workers)
/usr/local/bin/python3 dual_volcano.py           # ~2 min
```
脚本幂等，覆盖写同名产物。
