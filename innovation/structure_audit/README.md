# Phase 6 Stage S1 — 弛豫结构来源审计（structure_audit）

判定结果：**分支 α**（真实弛豫结构 100% 覆盖 → 连续局域特征直接在 DFT 几何上计算，无需 β 双腿）。

## 三处探测结论

### 1. 项目内镜像（决定性发现）—— 覆盖率 100%

`data_raw/MamunHighT2019_adsorption.json.gz` 顶层除 45,130 条反应记录外，还含一个此前
未利用的 **`_structures` 块**：42,667 个 ASE-db 风格结构行（cell / numbers / positions /
constraints / unique_id），每条反应记录 `raw[*].ref` 的 MD5 unique_id 即索引。

| 指标 | 数字 |
|---|---|
| 反应记录引用的 distinct ref | 42,667 |
| ref → `_structures` 命中 | **42,667 / 42,667 = 100%** |
| H* 主线构型（含 `_1/_2/...` 变体） | 7,048 |
| H* 构型结构全覆盖 | **7,048 / 7,048 = 100%** |
| H* 组成 | 1,836 |
| H* 组成结构全覆盖 | **1,836 / 1,836 = 100%** |

几何合理性抽查（随机 30 构型）：30/30 可解析；H 高出顶层金属层 0.49–1.88 Å
（mean 1.19 Å）且逐构型不同 → 为**弛豫后终态几何**（非理想原型）；约束为
FixAtoms 底层固定（与 Mamun 2019 "deposited" 约束口径一致）。

### 2. Catalysis-Hub GraphQL API —— 结构字段存在，作为备份通道

- schema 内省（公开、无需 Key）证实 `System` 类型含完整几何字段：
  `positions / cell / numbers / constraints / Cifdata / Trajdata / forces / fmax / InputFile / uniqueId`。
- 认证小样本探测本会话跳过：`CATHUB_API_KEY` 不在环境变量中（安全红线：仅从
  `os.environ` 读取，本会话不触碰 `.env`）；历史会话（notes/09）已用同一 API + Key
  成功拉取 Mamun 元数据 8,663 条，通道可用性有先例。若镜像结构出现缺损，可经
  `systems(first:.., pubId:"MamunHighT2019")` 按 uniqueId 回补（限速 ≤9 req/min）。

### 3. Materials Cloud / 原始论文数据仓储 —— 确认发布结构文件

Mamun et al., Sci. Data 6:76 (2019)（doi:10.1038/s41597-019-0080-z，PMC6538633）
Data Records 明确声明：
(i) Catalysis-Hub 永久链接 `https://www.catalysis-hub.org/publications/MamunHighT2019`，
网页/API 可下载 CIF / JSON / POSCAR / QE 输入；
(ii) **全部 QE 原始输出已上传 Materials Cloud archive**，可用 ASE 直接读出 Atoms
（含弛豫终态结构与计算结果）。
→ 两路公开来源互证，即使镜像失效亦有权威备份。

## 结构可得性分档图谱

| 层级 | 组成覆盖 | 构型覆盖 | 来源 |
|---|---|---|---|
| H* 主线（1,836 组成 / 7,048 构型） | **100%** | **100%** | 镜像 `_structures`（弛豫终态） |
| 全镜像（45,130 条反应，C/H/O/N/S 多吸附质） | 100% | 100%（42,667/42,667 ref） | 同上 |
| 权威备份 | — | — | CatHub API（Cifdata/Trajdata）+ Materials Cloud QE 原始输出 |

## 分支判定

**α**：真实弛豫结构组成覆盖率 100% ≥ 50% → Phase 6 连续局域特征（层距、位移、
弛豫依赖组）**直接在 Mamun 原 DFT 弛豫几何上计算**；CHGNet 重构仅作对照臂，
DFT 自弛豫双腿（β）不启动，省下对应 ~40% 工作量。预注册中 stage_S2_static 的
"弛豫依赖组"特征（层距/位移）现可直接从 `_structures` 几何提取，无需原型近似。

## 复现

```
python innovation/structure_audit/probe_structure_sources.py
# 输出 innovation/structure_audit/audit_report.json
```

安全：脚本不读取/打印任何 `.env` 或 Key；API 认证仅 `os.environ["CATHUB_API_KEY"]`，
缺失时自动降级为 schema 内省并记录环境限制。
