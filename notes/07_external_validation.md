# 07 — 跨库外部验证报告（SPEC 第 7 节，scripts/11_external_validation.py）

## 1. 数据源

| 项 | 训练侧（内部） | 外部验证侧 |
|---|---|---|
| 数据集 | Mamun et al., Sci. Data 6:76 (2019)，CatBench Zenodo 镜像 | EqV2-HER-Discovery：Oguz et al., *ACS Catal.* 2025, 15, 19461（GitHub ergroup） |
| DFT 泛函 | BEEF-vdW | **RPBE**（VASP；README 明确记载） |
| 目标 | ref_ads_eng，H* 吸附能（0.5H₂(g)+*→H*） | `Full_optimized_DFT_H_ads_energy`（eqv2_361，全弛豫）/ `single_point_DFT_H_ads_E`（eqv2_500，EqV2 弛豫后单点） |
| 条数 | 1,836 唯一表面 | 原始 861 行 → 去缺失 857 行 → 唯一 (src, comp, facet) 表面 **491** 个（361 集 219 + 500 集 272） |

### 能量参考态对齐论证
- OC20/AdsorbML 框架（EqV2 数据的生产管线）的吸附能约定为
  E_ads = E(slab+ads) − E(slab) − E(气相参考)，**H 的气相参考为 ½H₂(g)**，
  与 Mamun 的 ref_ads_eng（0.5H₂+*→H*）同口径，无需常数平移。
- 自洽性佐证：361 集是论文按 HER 最优筛选出的最终候选，其 E_ads 全部落在
  **−0.40 ~ −0.10 eV**、区间中点 ≈ −0.25 eV（均值 −0.265 / 中位 −0.273）——
  恰为 ΔG_H* = E_ads + 0.24 ≈ 0
  的 Nørskov 最优窗口，直接印证 ½H₂ 参考态。
- 泛函差异（BEEF-vdW vs RPBE）不表现为常数平移，而以系统性偏差形式进入
  残差（见 §3 bias），故正文同时报告原始 MAE 与去偏差平移后 MAE。

## 2. 方法（零重训）

1. 出厂模型：`hstar_best_params.json` 的 XGBoost_tuned
   （lr=0.03, depth=7, n=400, subsample=0.8）在全量 hstar_dataset_v2
   （1,836×36 特征）上重训，种子 42；特征集与 07 完全一致（排除 site_spread 等 ID 列）。
2. 外部特征化复用 06 的函数（importlib 同源加载 `parse_comp` /
   `build_element_table` / `wmean` / `wstd`），comp 规范化同 05
   （元素字母序+gcd 约简，如 Sn14Ir6→Ir3Sn7）；mendeleev 兜底未启用。
3. **结构 one-hot 处理规则**（EqV2 无 A1/L1₀/L1₂ 结构类型标注）：
   - `structure_A1/L10/L12` 全部置 0（=**缺失**，而非属于任何一类；EqV2 体相为
     任意晶系的金属间化合物，本就不在这三种有序结构分类内）；
   - facet：Surface=111 → `facet_111=1`；其余晶面（100/110/210/211/221，共 382/491）
     两个 facet one-hot 均置 0（训练域外，模型对该维度只能外推）。
   - 因此"facet=111 子集"是唯一严格的域内比较，其余分层为域外压力测试。
4. 同一 (src, comp, facet) 的多条记录（不同吸附位点/终端，185 组，位点极差最大
   1.80 eV）取**最低 E_ads**，与 H* 线"每表面最稳定位点"口径一致。
   注：361（全弛豫）与 500（单点）的 219 个 (comp,facet) 完全重叠但 DFT 协议不同，
   故按来源分层报告、不互相覆盖。

## 3. 跨库指标（残差 = pred − true）

内部对照：Mamun 随机 5-fold OOF MAE = **0.120 eV**（复算与 07 记录一致）。

| 分层 | n | MAE | RMSE | R² | bias (pred−true) | 去偏差 MAE |
|---|---|---|---|---|---|---|
| **全部** | 491 | **0.347** | 0.603 | −0.94 | **+0.158** | 0.313 |
| eqv2_361（全弛豫，近最优窗口） | 219 | **0.285** | 0.364 | −20.0 | +0.131 | 0.255 |
| eqv2_500（单点，全区间） | 272 | 0.397 | 0.741 | −0.65 | +0.180 | 0.358 |
| facet=111（域内晶面） | 109 | 0.341 | 0.430 | −11.3 | +0.113 | 0.309 |
| 不含 Sb（元素域内） | 364 | 0.335 | 0.639 | −0.71 | +0.093 | 0.312 |
| facet=111 且不含 Sb（最严格域内） | 86 | 0.337 | 0.431 | −19.2 | +0.064 | 0.317 |
| 含 Sb（元素域外，Sb∉Mamun 37 元素） | 127 | 0.382 | 0.486 | −4.9 | +0.343 | 0.250 |

按元素分层（n≥5，节录）：含 Ir 0.136 / 含 Pt 0.189 / 含 Ni 0.198（贵/半贵金属，MAE 最低）；
含 Nb 0.263 / 含 Co 0.289 / 含 Cu 0.365 / 含 Mo 0.380 / 含 Sn 0.424 / 含 Mn 0.463 /
含 Ta 0.463 / 含 Ti 0.508 / 含 Fe 0.619（3d 前段+主族，MAE 最高）。
完整表见 `data_processed/extval_metrics.csv`。

补充稳健性读数：
- 4 条极端真值（|E_ads|>2 eV，全部在 500 集，如 FeTi4(221) +7.31 eV，
  疑似单点协议下未重构/物理异常表面）贡献 RMSE 主要尾部；剔除后 MAE=0.316。
- **R² 全为负且不可用于横向解读**：EqV2 真值区间被筛选压缩（361 集全距仅
  0.30 eV），R² 的分母（真值方差）远小于误差方差，属区间受限伪象，
  不代表模型反向。排序能力：Spearman ρ = 0.17（全部）/ 0.29（361 集），弱正相关。

## 4. 系统性偏差讨论

- 全库 bias = **+0.158 eV**（pred − true > 0：模型预测的吸附比 RPBE 真值**偏弱**）。
  方向与泛函定性预期一致：BEEF-vdW 含色散修正、对 H* 吸附总体略强于 RPBE，
  以其训练的模型映射到 RPBE 能量面时整体上移（预测值偏正）。
- 偏差高度组成依赖：含 Sb +0.343 / 含 Sn +0.347 / 含 Mn +0.424 / 含 Fe +0.357
  （训练集中这些元素的 d 电子/电负性组合样本少或为域外元素 Sb），
  而含 Ir −0.043 / 含 Nb −0.004 几乎无偏——说明偏差并非纯泛函常数项，
  而是"泛函差 × 组成域外度"的叠加；故全局常数平移只能把 MAE 从 0.347
  降到 0.313（−10%），无法根治。
- 结构/晶面信息缺失（one-hot 全 0）是第二大失真源：训练集中
  structure/facet 恒非零，外部样本在该子空间全部落在训练流形之外；
  facet=111 子集 MAE（0.341）并未优于全体，说明误差主因是组成域外度
  而非晶面编码。

## 5. GASdb 支线（闸门触发，已放弃）

- 下载成功（docs.pkl 1.25 GB，协议 3，顶层 list of gaspy mongo 文档，
  已补装 ase 3.29.0 / pymongo）。
- 解析尝试 #1：`pickle.load` → MemoryError（本机 3 GB 内存，对象图展开超限）。
- 解析尝试 #2：自定义 Unpickler 流式拦截顶层 APPENDS、只留 H 条目轻量字段 →
  仍 MemoryError（pickle memo 表对全流对象持续引用，内存随流单调增长）。
- 容器内 swapon 被拒（无法加 swap）。按预设闸门放弃（放弃原因为内存不足，不变），详见
  `data_raw/gasdb_attempt_log.md`；为控制交付体积，该 1.25 GB 原始 pkl 已由主控删除，
  不在交付包内——可用原 URL（保留在 attempt_log 中）重新下载后在大内存环境复用。

## 6. 结论（可信度声明）

模型在完全独立、不同泛函（RPBE vs BEEF-vdW）、不同化学空间
（金属间化合物 vs 纯金属/有序合金）的 EqV2 数据上零重训预测达到
**MAE ≈ 0.35 eV（近最优 HER 窗口的 361 集为 0.285 eV）、系统偏差 +0.16 eV**，
约为内部 OOF 误差（0.12 eV）的 3 倍，排序能力弱（Spearman ρ≈0.2–0.3）：
**该组成描述符模型可作为 Mamun 域内（贵金属基 (111)/(101) 合金）的定量工具
（≈0.12 eV 级），对域外组成/晶面仅能作 ±0.3–0.4 eV 精度的粗筛，且对
含 Sb/Sn/Fe/Mn/Ti 体系的预测需按上述正偏差谨慎解读，不宜用于跨库精确排名。**

## 产物清单
- `data_processed/extval_results.csv`（491 行：src/comp/facet/真值/预测/残差/域标记）
- `data_processed/extval_metrics.csv`（分层指标表）
- `figures/extval_parity.png`（parity + 残差分布，含 Mamun OOF 对照）
- `data_raw/eqv2_361.csv`、`data_raw/eqv2_500.csv`、`data_raw/gasdb_attempt_log.md`
  （注：gasdb_docs.pkl 1.25 GB 已由主控删除以控制交付体积，可按 log 中原 URL 重下）
- `scripts/11_external_validation.py`（可独立复跑，种子 42）
