# 批次 C 原子级直接预裁 (Phase 6, 零核时)

裁决问题: 冻结清单 batch_C_relax_audit_30 (25 体系) 的 CHGNet 弛豫均显示剧烈重构
(表面位移 3.0–6.7 Å, 能降 -3.4 ~ -9.2 eV/atom)。这是真实物理还是 CHGNet 代理势
在这些体系上的伪影? 本目录用 `_structures` 中真实 DFT 弛豫构型做原子级直接对照。

## 口径选择: 口径 A (直接对照可行)

结构审计所称"构型覆盖 100%"成立且更强: `_structures` 每条稀释 H* 记录同时引用
`Hstar` (吸附态) 与 `star` (**DFT 弛豫清洁 slab**) —— 清洁 DFT 弛豫构型直接可得,
无需退到口径 B (剥离 H 的金属骨架)。

DFT 侧"原型→弛豫态"位移的算法: db 的 star 底 2 层被 FixAtoms 固定在理想截断
体相位置 (实测 25/25 体系成立, 层间平移矢量残差 t_err < 5e-7 Å)。理想截断点阵
由固定层平移外延 (fcc 衍生 A1/L1₀/L1₂ 均为纯平移堆垛), 每个非固定原子与同物种
理想点 Hungarian 配对 (面内最小像, z 直取) 得弛豫位移。

## 三量对照口径

- (a) 原型→CHGNet: `chgnet_features/slabs/*.cif` → `relaxed_subset/*.json` 末态,
  主口径为 r2_features 同序笛卡尔位移 (**25/25 复现冻结值, 容差 1e-3**);
  Hungarian 最小像口径另报 (CHGNet 重构下原子跨周期像, 数值略小, 结论不变)。
  CHGNet 弛豫含晶胞自由度 (晶格变化百分之几, `chg_dlat_pct` 列另报), 配对与
  最小像均在原型晶格分数坐标下进行。
- (b) 原型→DFT: 固定层外延理想截断 → star 弛豫态 (上述算法)。
- (c) CHGNet vs DFT 直接对照: 两侧晶胞不同 (原型 2×2 超胞 128/32 原子, 半径估算
  晶格常数; DFT db 为 12 原子小胞), 跨胞逐原子 RMSD 无定义。直接对照以"相对各自
  理想截断的弛豫位移场"的同物理量进行: 顶层层均位移、分物种顶层位移、顶层层距
  弛豫 % (d12_relax_pct, 晶胞无关)。**此为口径 A 的如实局限**。

## 裁决规则 (预注册)

DFT 顶层层均位移 < 0.5 Å 且 CHGNet 表面均位移 > 1 Å → **ARTIFACT**;
DFT 顶层层均 ≥ 0.5 Å (同量级大位移) → **REAL_RECONSTRUCTION**; 其他 → MIXED。

## 结果: 25/25 ARTIFACT

| 量 | 中位 | 范围 |
|---|---|---|
| DFT 顶层层均位移 | 0.117 Å | 0.022–0.403 Å |
| CHGNet 表面均位移 | 3.503 Å | 3.011–6.710 Å |
| CHGNet/DFT 比值 | 40× | 7.8–136× |

- 与间接预裁 (CHGNet 3.503 Å vs DFT 吸附诱导 0.210 Å, 22×) **方向一致且更强**:
  直接对照分离了吸附贡献 (DFT 侧为清洁表面原型→弛豫), 中位比值 40×。
- 冻结值交叉核对 25/25 在容差内。
- 开发过程中的伪阳性教训: 初版层聚类 (tol=0.5 Å) 把 CrTl/Cr3Pb 顶层起伏
  (~0.55 Å) 劈成子层, 理想外推错位一层, 曾误判 2 例 REAL (2.4–2.6 Å ≈ 恰为层距)。
  修复为自适应 tol (0.35×理想层距) 后二者 DFT 位移 0.26/0.40 Å, 判 ARTIFACT。
  此教训本身值得记录: 任何基于层外推的审计必须防"劈层"。

## 产物

- `preadjudicate.py` — 幂等 (无 --force 时跳过计算仅刷新 SHA256 清单)
- `batch_c_direct_comparison.csv` — 25 行逐体系三量对照 + 裁决
- `verdict.json` — 总裁决/逐体系/口径声明/局限/哨兵建议/多 star ref 全记录
- `disp_scatter.png` — DFT vs CHGNet 位移散点 (对数轴)
- `sha256_manifest.json` — 产物 SHA256

复跑: `python3 innovation/batch_c_preadjudication/preadjudicate.py [--force]`
