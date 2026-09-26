# 99 — 终审验证报告（独立核验，二值结论）

核验人：独立终审验证员。核验日期：2026-09-09。
方法：不复用被验脚本的函数，独立重读原始数据（`data_raw/MamunHighT2019_adsorption.json.gz`、
各 CSV）复算；脚本逻辑以源码行号佐证；指标以 sklearn 独立复算。
工具：shell + /usr/local/bin/python3（pandas/numpy/PIL/sklearn）。

---

## A. CatHub 数据线

### 项1 cathub_mamun.csv 行数 + 镜像交叉核对 + 20 组成抽查 —— **PASS**

- `wc -l data_processed/cathub_mamun.csv` = 8664（含表头）→ **8,663 数据行** ✓
  （其中稀释方程 `0.5H2(g) + * -> H*` 6,894 行、双倍计量 `H2(g) + 2* -> 2H*` 1,769 行，与 notes/09 一致）
- 独立复算（自写 canon_comp 规范化 + 消耗式一对一匹配，容差 1e-3 eV，不复用 scripts/13）：
  - 镜像 H* 子集：7,048 构型 / 1,836 组成 ✓
  - 组成匹配：**1,821/1,836 = 99.2%** ✓；仅镜像 15 个组成（AgFe3, AgTc, CdFe3, CrV,
    CuFe3, CuHg, Fe3Pb, Fe3Zn, FeNb, FeRh3, LaNi3, Tc, TlZn, V, Zn），与 notes/09 所列完全一致
  - 能量逐条匹配：**6,893/7,005 = 98.4%** ✓
  - 匹配对偏差：**max = 5.22e-4 eV，mean = 1.44e-7 eV** ✓（notes 声称 5.2e-4 / 1.4e-7）
- 随机抽查 20 个共有组成（seed=20240909：Re3Zr, AuPb3, CoNi3, RhTa, AlFe3, LaTa3, Cr3Zn,
  CoIr, HgTl3, CrHg3, GaLa, CrOs3, Al3Au, NbNi3, CoRe, LaRe3, PdPt, HgW3, TaTi, Co3Ta）：
  **匹配对最大偏差 1.11e-16 eV < 1e-3** ✓。
  对抗性说明：若对每条镜像能量不做消耗匹配而直接取最近官方值，AlFe3 表面出现 1.71 eV 偏差
  ——经查系镜像 3 构型 vs 官方仅 1 行（官方 ~2.2% 未捕获行），非能量值不一致；消耗式匹配
  下该组成 1 对匹配偏差为 0、2 行未匹配。此与 notes/09 自述的"分页漂移残余未捕获"局限一致，
  不构成对声明的否证。

### 项2 hstar_facet_check.csv facet 一致性 —— **PASS**

- 直接统计 `facet_match` 列：True=1,821、False=0、NaN=15（官方缺数据行），共 1,836 行。
  "1,821/1,821 一致、0 冲突"属实（分母不含 NaN 的 15 行）。

### 项3 cathub_hstar_extra.csv 行数与 by_pub 盘点 —— **PASS**

- extra：15,327 数据行 ✓；`pubId` 唯一计数 = **45** ✓
- by_pub 文件 45 行，`n` 列合计 = 15,327，与 extra 行数闭合 ✓

### 项4 extval2_metrics.csv ALL 行 + extval2_homogeneous.csv 能量范围 —— **PASS**

- 从 `extval2_homogeneous.csv`（**355 数据行** ✓）独立复算 ALL 指标：
  MAE=0.17682、RMSE=0.24501、R²=0.74157、bias=+0.08782 —— 与
  `outputs/extval2_metrics.csv` ALL 行逐位一致，且与 notes/14 的 0.177/0.742/+0.088 一致 ✓
- 能量范围：E_true ∈ [−1.239, +1.749] eV，全部 |E|<5 ✓

### 项5 scripts/14 清洗漏斗 15,327→13,812→843→355 —— **PASS**

- 源码：`scripts/14_extval2_homogeneous.py:159`（步骤1 反应身份）、`:166`（2a 组成可解析）、
  `:171`（2b 同质金属纯度）、`:176`（3a 物理范围）、`:184`（3b MAD 离群），每步 log 记录剔除数。
- `data_raw/extval2_run.log` 实测：15327 −(步骤1 剔1515)→ 13812 −(2b 剔12969)→ 843
  （2a/3a/3b 剔 0）→ 聚合 **355 表面**。漏斗数字与声明逐步吻合 ✓

## B. 模型优化线

### 项6 opt_comparison.csv vs notes/12 —— **PASS**

- CSV 实读：基线 GroupKFold **0.1198**±0.0087、嵌套CV 优化 **0.1281**±0.0093 vs 基线同折
  **0.1198**±0.0097、Stacking **0.1233**、外部验证 EqV2 ALL **0.3318** —— 与 notes/12 §7 全表一致 ✓
- 交叉验证 `outputs/opt_nested_cv.csv` 五折外层 MAE（优化）均值 = 0.12813，
  基线同折均值 = 0.11980，独立闭合 ✓

### 项7 opt_search_results.csv ≥150 组 —— **PASS**

- 150 数据行 ✓；min(mean_MAE)=0.1309 出现在 config_id=134（与 notes/12 §2 "第 134 组"一致）✓

### 项8 scripts/12 泄漏列排除 / 种子 / 分组 —— **PASS**

- `scripts/12_optimize_hstar.py:63-64`：`site_spread` 列入 ID_COLS（注释明标"泄漏特征…禁止入模"）；
  `:226-227` 双重断言 `site_spread not in feats`；运行日志 `[读入] ... 36 特征；site_spread 已排除` ✓
- 独立验证：v2 数据集 42 列剔除 6 个 ID 列后恰为 36 特征，不含 site_spread；
  v2 的 comp 列 1,836 值全部通过规范化恒等检验（canon(c)==c）✓
- `:61` SEED=42；`:229` groups = df["comp"]；GroupKFold 用于搜索/权重/stacking/嵌套 CV 全程
  （`:126,:129,:144-151,:194-201`）✓

### 项9 "未改动候选排序"声明 —— **PASS**

- `stat`：data_processed/candidate_rankings_hstar.csv mtime = 09-08 18:15:25；
  outputs/12_run_log.txt = 09-09 04:01:29、scripts/12 本身 = 09-09 04:03:56。
  候选文件早于 scripts/12 运行约 10 小时，未被改动 ✓；run log 末行亦记录
  "未达阈值，保留原候选"。

## C. 安全与工程

### 项10 硬编码 Key 检查 —— **PASS**

- `grep -rn "<Key前6位>" .`（全树，含 notebooks/thesis/deprecated/data_raw/outputs/figures）：
  除 `.env` 外**零命中** ✓
- scripts/13 仅经 `os.environ.get("CATHUB_API_KEY")` + `.env` 回退读取
  （`scripts/13_cathub_fetch.py:70-80`），无硬编码 ✓

### 项11 .env 与分发排除 —— **PASS**

- `.env` 存在（59 字节，权限 600），含 `CATHUB_API_KEY=<已脱敏>…` ✓
- 项目中不存在任何 zip 打包脚本或 zip 产物；README.md:42 有明确分发说明：
  "项目根 `.env` 含 Catalysis-Hub API Key（私密），勿提交入库，分发前务必删除" ✓
  （以"说明排除"形式满足；无 .gitignore，但项目未见 git 仓库，不构成泄漏面）

### 项12 三张新增 PNG 有效性 —— **PASS**

- PIL `verify()` + 重新打开全部通过：
  - extval2_parity.png：3360×1500 PNG，288,951 B
  - opt_search_trajectory.png：2160×1260 PNG，153,708 B
  - opt_extval_parity.png：3360×1500 PNG，247,482 B

---

## 总结论：**ALL PASS（12/12 项全部通过）**

备注（非否决项，供 lead 参考）：
1. 项1 的"能量偏差"声明仅在消耗式匹配对口径下成立；官方侧 ~2.2% 未捕获行导致个别组成
   （如 AlFe3）镜像行数多于官方行，notes/09 已如实披露该局限。
2. 项2 的"1,821/1,821"分母不含 15 个官方缺数据组成，表述准确但需读者注意口径。
3. notes/08_optimization.md 为旧版（随机 KFold 草稿）报告，notes/12 已声明本轮为重做版
   且旧产物未引用未覆盖；两份文件并存不构成矛盾，但建议后续清理命名以避免混淆。
