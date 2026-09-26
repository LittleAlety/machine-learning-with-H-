# Workstream Q (site_continuous) — 可行性探测结果：搁置

## 判定
**SHELVED（先决条件不满足）**。预注册先决条件为「EqV2 491 表面自带结构文件可用」，实际不可用。

## 证据
- `data_raw/eqv2_361.csv`（361 行）与 `data_raw/eqv2_500.csv`（500 行）均只有：
  组成（Ametal/Bmetal/symbol/浓度）、晶系/点群、晶面、H 吸附能与表面能等标量字段。
  **无原子坐标、无晶胞、无 cif/traj 路径、无配位/位点描述**。
  （注：任务描述中的 `data_processed/eqv2_*.csv` 不存在，文件位于 `data_raw/`。）
- 全库 `find` 搜索 cif/xyz/traj：唯一结构集合为 `innovation/chgnet_features/slabs/` 下
  1836 个**自建原型 slab**（CHGNet 特征线 H1 产物），命名 `A3B_Prototype_111.cif`，
  与 EqV2 按 `Bulk_id=mp-*` 弛豫的真实表面不是同一结构；松匹配仅 20/500 行，不可用。
- 结论：不存在可用于 A/B 配位数与 GCN 几何描述符的真实 EqV2 表面结构，
  若强行用原型结构会制造伪特征，违背预注册设计，故搁置，不训练。

## 对照基线（重启时）
离散 one-hot 位点编码的红利兑现率 = **14.7%**；Q 需显著超过该值方判定兑现。

## 重启所需最小数据清单
1. 每个 EqV2 key（`mp-*_H_*_*`）对应的弛豫 slab 结构文件（cif/traj/extxyz），
   或可由 Bulk_id + Surface + 位点索引唯一定位的结构库；
2. 位点标注：H 吸附位点原子索引/坐标与 A/B 元素身份；
3. 结构文件自带 positions / cell / pbc / chemical_symbols（ASE 可读）；
4. （可选）未弛豫结构与表面能，用于弛豫变化量特征。

## 重启协议（按预注册）
纯几何计算 A/B 配位数 + GCN 类连续局域描述符（无 QM 成本）→ 位点级/表面级模型
→ 10 种子同协议 → 对照 14.7% 显著性检验 → 产出写入 `innovation/site_continuous/`。

详见 `q_feasibility.json`。
