# S2 静态几何特征分流 (Phase 6 stage_S2_static)

预注册: `config/preregistered_phase6.yaml stage_S2_static` + `config/preregistered_phase5.yaml fair_contract`
（min_gain 0.005 eV / 10/10 种子 / 嵌套 CV 复核）。

## 特征清单（锁死，复用 `continuous_local/r2_features.py` 静态实现，10 个）

从未弛豫原型 slab（`innovation/chgnet_features/slabs/*.cif`, 1836/1836, 0 失败）确定性生成，
表面层定义 z > quantile(z, 0.75)，CN 截断 = 共价半径和 ×1.2，GCN 参考 CN_max=12 (fcc)：

- 表面 CN: `surf_cn_mean / surf_cn_min / surf_cn_std`
- GCN: `surf_gcn_mean / surf_gcn_std / surf_gcn_min`
- A/B 近邻加权电负性 (Pauling): `surf_nn_en_mean / surf_nn_en_std`
- A/B 近邻加权 d 电子: `surf_nn_d_mean / surf_nn_d_std`

产物: `static_geom_features.csv`（1836 行 × 10 特征 + comp/structure/facet 键）。
弛豫依赖组（层距/位移/弛豫能降，另 8 个）不在本目录，见 `continuous_local/`。

## 评估口径

基线 = v3 84 特征（`hstar_features_v3_84feat.csv`），金标准 = 84+10。
XGB lr0.03/depth7/n400/sub0.8，GroupKFold(5, comp)，10 种子 [0,1,2,7,13,42,99,123,2024,31337]。
基线复现 OOF 0.1137 eV（参考 0.1134，口径一致）。覆盖率 100%（静态组天然全覆盖，无 NaN 稀释）。

## 结果（详见 gate_contract.json）

| 项目 | 数值 | 判定 |
|---|---|---|
| 闸门 Δ_A / Δ_B / 保留率 | +0.0003 / -0.0005 / -1.94 | **NO_GAIN**（Δ_A 未过 0.005，构造性可移植无从谈起泄漏） |
| 公平契约 Δ | +0.0003 eV，7/10 种子 | 未过（需 >0.005 且 10/10） |
| 嵌套 CV | gold 0.1139 vs base 0.1135，Δ=-0.0004，2/5 折 | 未过 |
| 3d 磁性子层 (n=398) | Δ=+0.0033 eV，10/10 种子 | 方向一致但低于阈值（对照 Phase3 +0.0173 / Phase4 +0.0109 / Phase5 弛豫组 -0.0082） |

## 判定

**negative（阴性入档）**：静态组闸门 NO_GAIN、契约未过阈、嵌套 CV 未复核 → 不满足
`stage_S2_static.adopt`（PASS 且 Δ>0.005 + 10/10），不进入 v5.1。

口径对比（一句）：Phase 5 弛豫依赖组在 14% 覆盖子集上 Δ_A=0.0129 过阈但线 B 崩塌
（保留率 -1.12）判 **LEAK**（`innovation/continuous_local/gate_contract_report.json`）；
静态组虽构造性可移植、全覆盖，但增益 ~0.0003 eV 远低于采纳阈值——两组一为"有信号不可移植"，一为"可移植无信号"，均不可采纳。
唯一值得记录的残留信号：3d 磁性子层 Δ=+0.0033 且 10/10 种子方向一致，可作后续候选方向参考（未达采纳标准，不作宣称）。

## 复现

```bash
python innovation/static_geom/s1_static_features.py   # 幂等, 全量重算覆盖写
python innovation/static_geom/s2_gate_contract.py     # ~6 min, 覆盖写 gate_contract.json
```
