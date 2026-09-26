# innovation/continuous_local — Phase 5 路线 2: 基座势弛豫 + 连续局域特征

预注册: `config/preregistered_phase5.yaml` → `route2_foundation_relax`（规则先于实验冻结）。

## 流程

1. **r1_relax.py** — CHGNet(v0.3.0, CPU) 对 `innovation/chgnet_features/slabs/` 原型 slab
   做结构弛豫（FIRE, fmax=0.05 eV/Å, max 100 步）。
   - 预算: CPU 4 h。单点/步 ~0.5-2 s（128 原子 slab），全量 1836 需 ~20-40 h →
     按预注册降级为 **3d 磁性 + 逃逸角相关子集**（组成含 Fe/Co/Ni/Mn/Cr/La/Y/Sc, n=663）。
   - 幂等: `relaxed_subset/<name>.json` 存在即跳过；失败写 `<name>.fail.json`。
   - OOM 防护: 单进程（沿 Phase 3 经验），md5 伪随机顺序保证限时内覆盖无偏。
   - 用法: `RELAX_THREADS=3 python r1_relax.py --subset mag3d_escape`
2. **r2_features.py** — 弛豫后连续局域描述符（预注册统计清单，未事后挑）：
   表面层(z 上分位 25%) CN 均值/最小值/标准差；GCN（CN_max=12 fcc 参考）均值/标准差/最小值；
   近邻加权电负性与 d 电子数均值/标准差；d12 层间距及相对弛豫%、表面位移统计、
   表面起伏、弛豫能降/原子。输出 `continuous_local_features.csv`。
3. **r3_eval.py** — 可移植性闸门（线A/线B, Ridge 推断保留率）+ 公平契约
   （84+N vs v3, GroupKFold(5, comp), 同折同 10 种子, Δ>0.005 且 10/10）+
   嵌套 CV 复核 + 3d 磁性子层 Δ（对照 Phase 3/4 的 0.0173/0.0109）。
   输出 `gate_contract_report.json`。

## 诚实条款（预注册 anchor_gate）

本会话无 DFT 能力：**DFT 复弛豫锚定为待办**。CHGNet 弛豫平均能降 ~6.5 eV/atom、
71% slab 表面均位移 >1 Å（d12 变化最大 −53%），说明原型 slab 经基座势发生了显著重构——
全部结论均为**条件性**（伪影风险已披露），禁止无条件精度宣称。

## 最终结果（256/663 子集覆盖, 13.9% 全表）

- 覆盖子集契约: Δ=+0.0129 eV, 10/10 种子, 嵌套 CV +0.0179 (4/5 折) —— 过阈
- 可移植性闸门: **实测 LEAK**（Δ_A=0.0129, Δ_B=−0.0145, retention=−1.12）——
  36 个组成描述符无法在未见组成上推断弛豫几何特征
- 全表 1836（未覆盖 NaN）口径: Δ=−0.0015, 0/10 —— 无增益
- 3d 磁性子层 (n=155): Δ=−0.0082（反向; Phase 3/4 磁矩特征曾 +0.0173/+0.0109）
- 位点级目标 OOF ≤0.090 未达（覆盖子集 gold 0.186）
- **判定: subset_signal_gate_failed（条件性、不可采纳）**；oracle 缺口 ~0.033 eV 未关闭

## 产物

- `relaxed_subset/*.json` — 弛豫轨迹摘要 + 终态结构（含 fmax/converged/能降）
- `continuous_local_features.csv` — 连续局域特征（18 个新特征 + 元数据）
- `gate_contract_report.json` — 闸门/契约/嵌套CV/3d 子层数字
- `summary.json` — 会话总结（覆盖率、耗时、判定）
