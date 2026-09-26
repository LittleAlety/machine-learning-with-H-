# Phase 5 Route 1 — 物理瓶颈模型（comp → 低维潜空间 → E_ads）

预注册：`config/preregistered_phase5.yaml` → `route1_bottleneck`（候选维度 [1,2,3] 锁死，双结局判读锁死）。

## 方法
- 数据：`data_processed/hstar_features_v3_84feat.csv`（1836 样本，84 特征），GroupKFold(5, groups=comp) OOF，10 种子 [0,1,2,7,13,42,99,123,2024,31337]。基线 v3 OOF = 0.1134 eV。
- 臂 (a) 线性瓶颈：StandardScaler → PLSRegression(n_components=k)（低秩线性回归，PLS 实现，锁死）。
  注：GroupKFold 为确定性划分，线性臂无随机性，故 10 种子 std=0（种子退化，如实记录）。
- 臂 (b) MLP 瓶颈：84→64-32→k→32-16→1，ReLU；标准化仅在内层训练折拟合；early stopping 用内层 GroupShuffleSplit(15%) 验证折（patience=100，Adam lr=1e-3，wd=1e-4，最多 3000 epochs）。

## 结果（OOF MAE, eV；10 种子 mean±std；gap = mean − 0.1134）

| k | 线性臂 (PLS) | gap | MLP 臂 | gap |
|---|---|---|---|---|
| 1 | 0.2699 ± 0.0000 | +0.1565 | 0.1503 ± 0.0135 | +0.0369 |
| 2 | 0.2179 ± 0.0000 | +0.1045 | 0.1443 ± 0.0026 | +0.0309 |
| 3 | 0.2040 ± 0.0000 | +0.0906 | 0.1470 ± 0.0029 | +0.0336 |

## 判定（bottleneck_verdict.json）
- 线性臂 k=2 gap = +0.1045 eV → gap 结局；MLP 臂 k=2 gap = +0.0309 eV → gap 结局。
- 两臂一致（无冲突）：**final_verdict = gap** —— 2 维瓶颈显著掉精度，"group_wmean 型单一低维管道是充分统计量"不成立；存在 SHAP 单特征排名未暴露的额外独立维度（MLP 臂从 k=1→2 提升 −0.006 eV，k=2→3 无进一步提升，提示缺口不在瓶颈宽度而在"非线性组成空间→能量"映射本身）。

## 潜变量解读（latent_corr.png，k=2，seed 31337 OOF 潜变量）
- 潜维度 1 与 2 高度共线（|r| = 0.986），二者是同一主导方向的镜像：与 group_wmean（|r|≈0.86–0.89）、mendeleev_number_wmean（≈0.76–0.80）、d_el_wmean / nd_valence_wmean（≈0.72–0.76）、melting_point_wmean（≈0.69–0.70）、n_unfilled_wmean（≈0.64–0.68）强相关。
- 即模型自发学到的潜空间确实压缩为"d 带中心代理族"一维方向，但正是这个压缩丢掉了 v3 需要的 ~0.03 eV 信息 → 与 gap 判定自洽。

## 复现
```bash
python innovation/bottleneck/run_bottleneck.py   # 幂等，覆盖写 results/verdict/latent/png
```
产物：`run_bottleneck.py`、`bottleneck_results.json`、`bottleneck_verdict.json`、`latent_corr.png`（300 dpi）、`latent_k2_seedlast.npy`。
