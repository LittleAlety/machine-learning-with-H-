# C-3 v2 Stage-1 闸门迭代状态

_生成时间（UTC）：2026-09-26T08:40:20+00:00_

> Iteration driver executed on a Windows workstation; no HPC allocation or compiled QE stack is present here, so DFT gates are PENDING, not fabricated.

## 闸门矩阵

| 步骤 | 内容 | 状态 | 说明 |
|---|---|---|---|
| 1.1 compile | Source archives verify (SHA256) but no compiled pw.x/cp2k/gp… | ⛔ BLOCKED | Source archives verify (SHA256) but no compiled pw.x/cp2k/gpaw on PATH. Build per 1.1. |
| 1.2 anchor gate | No BEEF-vdW recomputation yet. Run 1.2 on the cluster and wr… | ⏳ PENDING | No BEEF-vdW recomputation yet. Run 1.2 on the cluster and write recomputed_dG_eV into anchor_gate_sheet.csv; this script |
| 1.3 probe core-hours | Three probe tasks have not been measured. The plan forbids c… | ⏳ PENDING | Three probe tasks have not been measured. The plan forbids carrying over the 199,875 estimate as an approved allocation: |
| 1.4 MLIP-assisted path (optional) | MACE-MPA-0 pre-relax -> DFT refine may only accelerate produ… | ⏳ PENDING | MACE-MPA-0 pre-relax -> DFT refine may only accelerate production after it reproduces |dE| <= 0.1 eV on the SAME anchor  |

## 出口判定
- Stage 1 出口硬条件：锚定闸门 **PASS** + 探针核时落档。
- 当前 Stage 1：**CONDITIONAL GO (env pending)**（仅可做环境/锚定准备，不得开生产）。
- 批次生产（Stage 2/3）放行：**False**。

## 红线
- 闸门不追溯放宽；现场参数变更先写修正案再执行；阴性与失败结果全部保留。
