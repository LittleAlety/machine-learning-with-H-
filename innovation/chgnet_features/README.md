# Workstream H: CHGNet 特征工厂（Phase 3）

红线遵守：11（CHGNet 仅作特征生成器，不替代交付引擎）、14（含 CHGNet 特征的 lockbox 开工前冻结、SHA256 留证、单次裁决）。
全程未读取/打印任何 .env 或密钥。

## 检查点结论一览

| 检查点 | 规则 | 结果 | 判定 |
|---|---|---|---|
| H0 | chgnet 可装、预训练权重可得 | chgnet 0.4.2 + CHGNet v0.3.0 权重 OK，CPU 推理 ~0.6 s/slab | **PASS** |
| H1 | 1836 组成全部可建 slab | 1836/1836，失败 0（A1/L12/L10 + facet 111/101，32–128 原子） | **PASS** |
| H6 | 推理/建模前冻结 lockbox | 沿用 splits/lockbox_v1.json（276 组成），freeze SHA256=`689a4ea9…31cbb` | **DONE** |
| H2 | 可移植性闸门 | Δ_A=0.00250 eV ≤ 0.005 → **NO_GAIN**（保留率 r=0.516，未及判定线） | **NO_GAIN** |
| H3 | OOF MAE 改善 >0.005 且 10/10 | Δ=0.00250±0.00057 eV，10/10 种子改善但幅度不足 | **FAIL（契约未过）** |
| H4 | 3d 磁性子层 Δ MAE ≥ 0.02 eV | 子集 n=398，Δ=0.01729±0.00197 eV，10/10 改善 | **FAIL（差 0.0027 eV 未达线，接近）** |
| H5 | Mn/Fe/Co/Ni LOEO 退化 <0.02 eV | 四元素全部为负退化（改善）：Fe −0.059、Mn −0.020、Ni −0.019、Co −0.014 eV | **PASS** |

## 关键数字

- 基线（v3 36 描述符，XGB，GroupKFold(5)×10 种子）OOF MAE = 0.11960 eV
- +CHGNet 9 特征后 OOF MAE = 0.11710 eV（Δ = 0.00250 eV，10/10 种子为正，方向稳健）
- 3d 磁性子集（含 Fe/Co/Ni/Mn/Cr，n=398）：0.20506 → 0.18777 eV（Δ = 0.01729 eV）
- LOEO（seed 42，含该元素组成留出）：Fe 0.2795→0.2206；Mn 0.4823→0.4626；Co 0.1307→0.1171；Ni 0.1346→0.1155
- 全量推理成本：1836 slab ≈ 13 min（3 分片并行，实测峰值内存触发过 cgroup OOM，单进程顺序可稳定完成）；远低于 4 h 预算，未触发降级路径

## 判读

CHGNet 表面磁矩特征对**含 3d 磁性元素的组成**有真实、方向一致的增益（H4 子层 Δ≈0.017 eV 且 LOEO 大幅改善 Fe/Mn），
但在全组成公平契约上幅度不足（H3 未过 0.005 eV 线），按预注册闸门 H2 判 **NO_GAIN**，不进入主交付模型。
特征文件与脚本保留作后续机制分析（磁性表面子课题）使用。按红线 11，CHGNet 仅用于特征生成。

## 已知限制（如实记录）

1. CHGNet v0.3.0 不输出原子电荷 → 预注册"电荷转移统计"为 N/A（模型能力所限，未伪造）。
2. 表面层定义 v2 修正：初版 z≥zmax−2Å 在 SlabGenerator 重定向胞上仅捕获 1–2 原子（离散度退化）；
   修正为 z 坐标顶层 25% 原子。修正发生在任何模型拟合之前，lockbox 未被触碰。
3. 晶格常数为元素原子半径加权估计（确定性、无目标泄漏），非弛豫结构；slab 未做几何弛豫（单点推理）。
4. H5 的 CatHub 跨域线（不劣于 0.177）需主线交付引擎复核，本工作流未动用交付引擎。

## 产物清单

- `h0_env_report.json` / `h1_slab_build.json` / `h6_lockbox_freeze.json` / `h2_h5_eval.json` / `chgnet_summary.json`
- `lockbox_chgnet_frozen.csv`（SHA256 留证）
- `chgnet_features.csv`（1836 × 9 特征 + n_atoms/n_surf_atoms）
- `slabs/`（1836 CIF）、`cache/`（逐组成推理缓存，含原始 m/z 向量，可离线再聚合）
- 脚本：`h1_build_slabs.py`、`h6_freeze_lockbox.py`、`run_chgnet.py`（幂等，支持 --shard/--subset/--limit）、`collect_features.py`、`h2_h5_eval.py`

## 复现

```bash
python h1_build_slabs.py          # H1
python h6_freeze_lockbox.py       # H6（建模前）
python run_chgnet.py --subset all # 幂等；可 --shard k --nshards K 分片
python collect_features.py
python h2_h5_eval.py              # H2/H3/H4/H5
```

---

# Phase 4 Workstream M：磁性线收口（最后一次预注册尝试）

预注册：`config/preregistered_phase4.yaml` workstream_M_magnetic（候选 ≤5，先注册后运行，红线 18 一次性强制关闭）。
脚本 `m_closure.py`（幂等）→ `m_closure.json`。协议与公平契约同口径：v3 84 维基线 +1 候选，GroupKFold(5, groups=comp) 固定折 × 10 种子；基线复现 0.11374 eV（与 fair_contract.yaml 登记值一致）。

## 候选 → 列映射（一次登记，未新增候选）

| 候选 | 映射 |
|---|---|
| m_max | chg_surf_mag_max（直接） |
| m_surface_mean | chg_surf_mag_mean（直接） |
| m_mean_wstd_ratio | chg_surf_mag_mean / (chg_surf_mag_std + 1e-6)（构造） |
| m_max_over_mean | chg_surf_mag_max / (chg_surf_mag_mean + 1e-6)（构造） |
| m_gini | 表面层 \|磁矩\| 的 Gini（由 cache m_raw/z_raw 按 top-z-quartile 构造） |

## 5 候选逐一结果（Δ MAE，10 种子均值，正值=改善）

| 候选 | 全局 Δ（胜种子数） | 3d 子层 Δ（全量 OOF） | 3d 子层 Δ（H4 式重训） | 判定 |
|---|---|---|---|---|
| m_max | −0.00001 (4/10) | −0.00010 | +0.01037 | FAIL |
| m_mean_wstd_ratio | +0.00009 (5/10) | +0.00057 | +0.00400 | FAIL |
| m_surface_mean | +0.00124 (10/10) | +0.00638 | +0.01088 | FAIL |
| m_max_over_mean | +0.00060 (9/10) | +0.00083 | +0.01177 | FAIL |
| m_gini | −0.00010 (4/10) | +0.00147 | +0.00234 | FAIL |

过线条件：3d 子层 Δ≥0.02 eV **且** 全局公平契约（Δ>0.005 eV 且 10/10）同时满足；无候选过线，嵌套 CV 不适用（跳过）。

## 最终判定：**CLOSED（关闭）**

5 个单特征候选均未过线（最优 m_surface_mean 的 H4 式 3d 增益仅 +0.0109 eV，仍低于 0.02 eV 线；全局增益 ≤0.00124 eV，远低于 0.005 eV 契约）。
按红线 18，磁性特征线**一次性强制关闭**：此后该线只接受真实新数据（新 DFT/实验磁矩测量等）重启，禁止在既有数据上继续挖掘派生统计量。
