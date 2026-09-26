# 09 — Catalysis-Hub 官方库整合：权威交叉验证、facet/sites 校验、增量池盘点

脚本：`scripts/13_cathub_fetch.py`（GraphQL API，`https://api.catalysis-hub.org/graphql`）。
目的：H* 主线镜像（CatBench 版 MamunHighT2019）缺 facet/sites 元数据、structure/facet
靠计量模式推断；本模块直连官方库完成三件事——**(A)** 能量权威交叉核对、
**(B)** facet/sites 元数据校验与补齐、**(C)** 非 Mamun H 吸附增量池盘点（合并重训决策依据）。

## 0. API 工程与请求纪律

- **Key 管理**：仅从环境变量 `CATHUB_API_KEY` 读取（本地回退项目根 `.env`，
  `.env` 不入库、分发前删除——README 已加警示），脚本中无任何硬编码密钥。
- **限速/预算**：6.7 s/请求（≤9 req/min，限额 10/min），日预算硬上限 500 次
  （`data_raw/cathub_cache/req_state.json` 持久计数，跨日自动重置，超限即收尾）。
- **断点续传**：每页原始 JSON 增量落盘 `data_raw/cathub_cache/pages/{pass}/`，
  游标存 `{pass}_state.json`，中断后重跑自动跳过已拉页。
- **分页稳定性缺陷（实测发现，重要）**：该 API 游标分页**顺序不稳定**
  （疑似 OFFSET 分页无 ORDER BY，跨请求行序漂移）：单遍全量拉取 26,477 条边
  去重后仅 20,363 行（~23% 行被重复返回、等量行被漏掉）。已用 `id` 字段验证
  库内行内容基本唯一（首 200 行 0 重复），即重复是分页漂移而非库内冗余。
- **对策**：同一查询跑 3 个独立 pass 取并集（按内容去重）。实测覆盖率：

| pass | 原始边 | 累计并集 | 占 totalCount=26,477 |
|---|---|---|---|
| 1 | 26,477 | 20,363 | 76.9% |
| 1+2 | 26,477 | 22,854 | 86.3% |
| 1+2+3 | 26,477 | **23,990** | **90.6%** |

- **请求账目**：3 pass × 133 页 + publications 1 + 探测 3 ≈ **404/500**，预算内。
  残余 ~9% 未捕获（非 Mamun 87.0%、Mamun 97.8%）；下文计数均为**下界**，
  相对构成比例可靠。若需补齐可改日再跑 pass（缓存自动续传并集）。

## 1. 任务A：Mamun 官方子集 vs 本地镜像（权威交叉验证）

产物：`data_processed/cathub_mamun.csv`（8,663 行；products~H* 且 reactants~H2，
覆盖探测值 8,856 的 97.8%）。方程构成：稀释 H*（`0.5H2(g) + * -> H*`）6,894 行 +
双倍计量写法（`H2(g) + 2* -> 2H*`）1,769 行。

与镜像 H* 子集（7,048 构型 / 1,836 组成）逐条核对（组成规范化同 05，容差 1e-3 eV）：

- **组成匹配率 1,821/1,836 = 99.2%**；仅镜像 15 个（AgFe3, AgTc, CdFe3, CrV, CuFe3,
  CuHg, Fe3Pb, Fe3Zn, FeNb, FeRh3, LaNi3, Tc, TlZn, V, Zn）、仅官方 0 个。
  仅镜像组成疑为 CatBench 整理版与官方库的版本差（或 2.2% 未捕获的极端情形），
  占比 0.8%，不影响主线结论。
- **能量逐条匹配 6,893/7,005 = 98.4%**（消耗式一对一匹配）；匹配对偏差
  **max = 5.2e-4 eV < 容差**，mean = 1.4e-7 eV，p50/p90/p99 均为 0
  —— 镜像能量值与官方库在浮点精度内一致，镜像可信度获权威确认。
- **表面级核对**（v1 每表面最低能 vs 官方最低能）：1,821/1,836 表面可比，
  **98.96% 在容差内**；超容差 19 个（max 1.07 eV：Nb、HgNb、CdW、CuIn、FeIn 等），
  主因应为官方侧 ~2% 未捕获行导致的最小值偏移，非系统性偏差（mean |Δ| = 2.9e-3 eV）。

## 2. 任务B：官方 facet/sites vs 推断标签

产物：`data_processed/hstar_facet_check.csv`（1,836 表面；零额外请求，复用任务A数据）。
v1/v2 组成集合一致（1,836 ≡ 1,836），join 以 v1 为准。

- **facet 一致 1,821/1,821 = 100%（0 冲突）**，15 个官方缺数据（即仅镜像组成）。
  我们从 12 原子超胞计量数推断的 A1(111)/L1₂(111)/L1₀(101) 标签被官方数据
  **完全证实**——主线的 structure/facet one-hot 特征不再是"推断"，而是经权威验证。
- 官方 facet 分布：111 1,216 个、101 605 个（与推断口径一致）。
- **sites（H 吸附位点）覆盖率 99.2%**（镜像完全缺失的元数据，现已存档于
  `cathub_mamun.csv` 的 `sites` 列）：top|B 956、hollow|A_A_A|HCP 860、
  hollow|A_A_A|FCC 806、hollow|A_A_B|HCP 668、hollow|A_A_B|FCC 643、
  bridge|A_A|B 573、top|A 524 等。为后续位点级分析（如 hollow vs top 能量分层）
  提供数据基础。

## 3. 任务C：非 Mamun 增量池盘点（合并重训决策依据）

产物：`data_processed/cathub_hstar_extra.csv`（15,327 行，覆盖探测值 17,621 的 87.0%）
+ `data_processed/cathub_hstar_extra_by_pub.csv`（45 个来源分组计数/能量范围/题名年份）。

### 3.1 来源构成（Top，行数为下界）

| pubId | 行 | 组成数 | E 范围 (eV) | 体系 |
|---|---|---|---|---|
| ZhangOff-equilibrium2025 | 9,205 | 3 | −5.74~3.13 | 富硼金属二硼化物（高覆盖 H） |
| YohannesCombined2023 | 2,590 | 48 | −11.63~22.06 | 过渡金属**氮化物** ML 筛选 |
| BoesAdsorption2018 | 1,589 | 7 | −2.23~22.34 | fcc(111) 纯金属多吸附质（含共吸附） |
| ZhangDopant-dependent2025 | 538 | 51 | −0.75~1.95 | 金属硼化物掺杂 |
| AlonsoStrain2023 | 347 | 46 | −1.24~1.23 | 金属间化合物 HER（ML 发现） |
| DickensElectronic2018 | 213 | 111 | −3.36~1.55 | 氧化物（钙钛矿/铝酸钙） |
| ZhangUnlocking2025 | 168 | 168 | −1.98~−0.28 | MBene |
| PasumarthiFacetDependence2023 | 132 | 1 | −0.02~0.75 | 纯金属 facet 依赖 |
| TangModeling2020 | 94 | 73 | −0.93~1.70 | 显式水 HER 动力学 |
| 其余 36 个来源 | <50 each | — | — | 金属/合金小集 + 分子/酶簇/单原子等 |

### 3.2 字段分布

- **泛函**：PBE_D3 9,911、PBE 2,962、RPBE 1,633、BEEF-vdW 311、其余为
  电位标注变体（`*_-x.xxVSHE`）、+U、SCAN/MS2/r2SCAN 等 50 余种 —— 与主线
  BEEF-vdW 同泛函的仅 ~2%；
- **DFT 代码**：VASP 12,548、QE 系 2,588、QMCPACK/DACAPO/GPAW/ORCA 零星；
- **方程**：稀释 `0.5H2+*->H*` 4,370 行，其余多为高覆盖 H（硼化物 `4.5H2->9H*` 等）
  或共吸附（N2/CH4/H2O/H2S/CO*）方程；
- **能量范围**：[−15.2, 22.3] eV，mean −1.69 eV，|E|>3 eV 离群 5,750 行
  （几乎全部是氮化物/共吸附/高覆盖体系）；
- **晶面**：硼化物终端标记（B03Hn 等）9,200+，常规金属晶面 111 2,016、
  100 1,359、110 1,321、001 802。

### 3.3 决策（用户授权 lead 定）

**主线保持 Mamun 单源纯净，不合并重训**：增量池 ~80% 为硼化物/氮化物/MBene/
氧化物/分子体系（异质，组成域外且含非金属），直接合并会污染主线的"同质金属
稀释 H*"数据域；泛函混杂（PBE_D3 为主 vs 主线 BEEF-vdW）引入系统性偏移。
同质金属 HER 子集（~843 行）改作**第二外部验证集**——见 `notes/14_extval2.md`。

## 4. 产物清单与局限

| 产物 | 内容 |
|---|---|
| `data_processed/cathub_mamun.csv` | 官方 Mamun H 子集 8,663 行（含 facet/sites/泛函） |
| `data_processed/hstar_facet_check.csv` | 1,836 表面 facet/sites 校验表 |
| `data_processed/cathub_hstar_extra.csv` | 非 Mamun 增量 15,327 行 |
| `data_processed/cathub_hstar_extra_by_pub.csv` | 45 来源盘点表 |
| `data_raw/cathub_cache/` | 3 pass 分页缓存 + 游标 + 请求计数（可续传） |

局限：① 分页漂移导致 ~9% 行未捕获（Mamun 仅 2.2%），计数为下界；
② 去重按"全字段内容"（id 仅 pass2/3 有），库内若存在内容完全相同的合法重复行
会被合并（已抽查首 200 行未见）；③ `sites`/`facet` 字段沿用官方库标注，
未做原子结构级复核。
