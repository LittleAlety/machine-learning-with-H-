# Workstream P：误差解剖探针（P1–P4）人读版结论

> 纪律声明：所有探针结果仅用于**诊断**，未用于选择交付配置；判读规则均预注册于各脚本头部。
> 折协议 GroupKFold(5, shuffle=True, random_state=42) 按规范化组成分组；基线 v3 XGBoost
> (lr=0.03, depth=7, n_est=400, subsample=0.8)。

## P1 容量扫描（`p1_capacity_sweep.py` → `p1_capacity_curves.png`）

- GBDT 高容量区（depth≥7 且 n_est≥400）OOF MAE 极差 = **0.0023 eV**
  （判据 ≤ ±0.005）→ 判定：**capacity saturated**
- GBDT 最优点：depth=7, n_est=2000,
  OOF MAE = 0.1169 eV
- MLP 各宽度最优 OOF MAE：{'16': 0.1169, '64': 0.1144, '256': 0.1138}；
  触发 A-4 深挖：**False**（只记录，不行动）

## P2 学习曲线（`p2_learning_curve.py` → `p2_learning_curve.png`）

- 25/50/75/100% 组级子采样 × 10 种子；均值 OOF MAE =
  {'0.25': 0.1374, '0.5': 0.1256, '0.75': 0.1214, '1.0': 0.1173} eV
- 末端斜率 = **0.0099 eV/倍数据**（判据 <0.01）
  → 判定：**data-volume saturated**

## P3 噪声地板（`p3_noise_floor.py` → `p3_noise_floor.md`）——锚点

- 路径 a（原始记录 (comp×structure) 组内 MAD/2 中位数，7048 条记录）：
  **floor_a = 0.0366 eV**
  （对照：min-vs-rest 中位偏差 0.2209 eV）
- 路径 b（10 种子 OOF 方差分解）：种子均值预测残差 **floor_b = 0.1163 eV**；
  种子间 std = 0.0225 eV（可约估计方差）
- 10 种子 OOF MAE = **0.1173 eV**
- **架构理论收益上限 = OOF MAE − floor_a = 0.0807 eV**
  （路径 b 口径 0.0010 eV）→ **headroom remains**

## P4 信息天花板 oracle（`p4_info_ceiling.py` → `p4_oracle_bar.png`）

| 作弊特征 | Δ_A (eV) | 保留率 r | 闸门判定 |
|---|---|---|---|
| site_onehot | 0.0009 | -5.397 | **NO_GAIN** |
| config_mean | 0.0391 | -0.068 | **LEAK** |
| config_count | 0.0096 | -0.151 | **LEAK** |

- 信息侧最大 Δ_A = **0.0391 eV** vs 容量侧最大改善（P1 GBDT 段）
  = **0.0027 eV** —— 差距即"缺信息不缺容量"的直接量化
- 预期全判 LEAK（反向验证闸门灵敏度），实际判定见上表，边界结果如实记录

## 组合图

`p_error_anatomy.png`：四面板（A 容量曲线 / B 学习曲线 / C 噪声地板分解 / D 信息 vs 容量），
300 dpi 暖色系，入正文用。

## 复现

```
cd innovation/probes
/usr/local/bin/python3 p1_capacity_sweep.py   # GBDT+MLP 网格（约数十分钟）
/usr/local/bin/python3 p2_learning_curve.py
/usr/local/bin/python3 p3_noise_floor.py
/usr/local/bin/python3 p4_info_ceiling.py     # 依赖 P1 verdict
/usr/local/bin/python3 p5_combo_figure.py
/usr/local/bin/python3 build_summary.py
```
