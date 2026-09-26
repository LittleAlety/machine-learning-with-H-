# 15 — 收尾优化记录（polish 轮，2025-09-09）

范围：仅工程/交付质量小项；未重跑任何管线脚本，未改动任何数据产物数值。

## 1. 本轮修复清单

### 1.1 陈旧产物归档（data_processed/opt_* 与 notes/08）

| 文件 | 迁移前 | 迁移后 |
|---|---|---|
| opt_search_results.csv、opt_sample_weights.csv、opt_stacking_compare.csv、opt_nested_cv.csv、opt_loeo_results.csv、opt_final_metrics.csv、opt_extval_results.csv、opt_extval_metrics.csv、opt_best_params.json（共 9 个，旧随机 KFold 草稿） | `data_processed/opt_*` | `deprecated/opt_*`（本轮开始前已就位，本轮补齐引用清理与归档说明） |
| `08_optimization.md`（旧版优化报告，随机 KFold 口径） | `notes/08_optimization.md` | `deprecated/08_optimization.md` |
| `deprecated/README.md`（新建） | — | 记录迁移日期、废弃原因、旧→现行产物对应关系，并警示 deprecated/ 与 outputs/ 存在**同名不同值**文件 |

引用核查与同步修改：
- `grep -rn` 确认 scripts/、thesis/、其他 notes 均无对 `data_processed/opt_*` 的读取依赖
  （scripts/12 仅注释提及，notes/12 为免责声明提及）。
- `SPEC.md` 第 8 节产物行的 `notes/08_optimization.md` → `notes/12_optimization.md`（重做版为实际产物），
  并加注"旧草稿已归档 deprecated/"。
- `scripts/12_optimize_hstar.py` 两处注释/docstring 中的 `data_processed/opt_*` → `deprecated/opt_*`
  （其中一处为生成 notes/12 的模板字符串，已同步改 `notes/12_optimization.md` 第 5 行保持一致）。
- `notes/99_final_audit.md`（审计日志）保留不动：其对 notes/08 的陈述仍为事实，
  仅路径前缀已变为 deprecated/（见 §3 建议）。

### 1.2 requirements.txt 补全

- 通读 scripts/01–14 全部 import：第三方依赖为 pandas / numpy / scipy / scikit-learn /
  xgboost / shap / mendeleev / matplotlib；网络（05/13/14）与 .env 读取均用标准库
  urllib/os，**未发现 requests、python-dotenv 的实际使用**。
- 变更：`xgboost==3.4.1`→`>=3.4`、`shap==0.52.0`→`>=0.52`、`mendeleev>=1.0`→`>=1.2`；
  **新增 `scipy>=1.11`**（scripts/10 直接 import）；**移除未被使用的 `requests`**；
  加注释说明网络/.env 走标准库。已核对已装版本全部满足（scipy 1.16.2、pandas 2.3.2、
  numpy 2.2.5、sklearn 1.7.2、matplotlib 3.10.3、xgboost 3.4.1、shap 0.52.0、mendeleev 1.2.0）。

### 1.3 README 翻新

- 去除"本科毕设项目""对应原毕设题目"框架表述，改为中立项目描述，并补扩展线一句话定位。
- 目录结构补：`outputs/`、`deprecated/`、`thesis/`、`requirements.txt`、`web/`
  （终端展示页面，即将交付，见 web/ 目录）；scripts 导航扩展为 01–14 全量。
- 管线阶段表补 9–14 六个阶段（多吸附种、跨库外部验证、深度优化、官方库核对、第二外部验证），
  并补 notes/10（文献锚点）与 notes/99（终审核验）的导航说明。
- 复现步骤补 scripts/09–14 命令；环境行改为指向 requirements.txt 并补 scipy。
- 保留并原样强化 .env 私密 Key 分发前删除警示。
- 新增数据权威性结论：官方库交叉核对组成匹配率 99.2%（1,821/1,836）、
  能量逐条匹配率 98.4%（6,893/7,005）、匹配对最大偏差 5.2e-4 eV。

### 1.4 代码小修（scripts/13、14）

- `scripts/13_cathub_fetch.py`：
  - `API_KEY` 由模块导入时加载改为**惰性加载**（首次 `gql()` 时才读 Key）——修复前
     analyze 模式即使全部命中本地缓存也会因无 Key 直接 `sys.exit`；已实测 import 不再触发 Key 检查。
  - 删除 `merge_passes(where)` 的死参数 `where`（函数体从未使用），同步更新调用点。
- `scripts/14_extval2_homogeneous.py`：删除未使用的调色板常量 `C_DEEP`。
- 新增 `.gitignore`：`.env`、`__pycache__/`、`*.pyc`、`data_raw/cathub_cache/`、`.DS_Store`。
- `SPEC.md` 目录契约同步现状（补 outputs/、deprecated/、thesis/，scripts 导航扩至 01–14）。
- 验证：`py_compile` 通过（12/13/14）；13 导入测试通过（API_KEY=None、签名正确）；
  未运行任何会写数据产物的脚本。

## 2. 毕设痕迹盘点（grep -rni "毕设|毕业设计|毕业论文|原方案|导师|开题"，排除 deprecated/）

### 公开文件

| 文件 | 命中数 | 处置 |
|---|---|---|
| `README.md` | 2（"本科毕设项目""对应原毕设题目"） | ✅ 本轮已清理 |
| `SPEC.md` | 2（"本科毕设落地项目""对应原毕设题目"） | ✅ 本轮已清理（仅措辞中立化，契约内容不变） |
| `thesis/成果汇总报告.md` | 5（含"毕设方案""原方案设想"等） | ⏸ 不动，标记待主代理处理（主代理将另写交付文档取代） |
| `thesis/成果汇总报告.converted.md` | 5（同上，为 .md 的转换副本） | ⏸ 不动，随主文件一并由主代理处置 |

### 内部日志

| 位置 | 命中数 | 处置 |
|---|---|---|
| `notes/*.md` | 0 | 无需处理 |
| `plan*.md` | 文件不存在 | — |
| `scripts/*.py` | 0 | 无需处理 |

## 3. 留给未来的建议（按优先级）

1. **【高】thesis/ 两份报告的去毕设化/替代**：公开交付前由主代理以新交付文档取代；
   取代后建议将 `成果汇总报告.converted.md` 一并移除（纯转换副本，避免双源漂移）。
2. **【中】deprecated/ 与 outputs/ 同名文件风险**：`opt_best_params.json` 等 9 个文件名
   两侧相同但数值不同（旧 max_depth=11 vs 现行 9）。已在 deprecated/README 警示；
   若仍担心误引，可给 deprecated/ 文件加 `_randomkfold_v1` 后缀。
3. **【中】.env 分发纪律**：警示已保留，但 .env 仍物理存在于项目根。建议分发流程使用
   显式排除清单（或 `.gitignore` + `git archive`），不要依赖人工记得删除。
4. **【低】notes/99_final_audit.md 路径时效**：其第 114 行提及的 notes/08 现位于
   deprecated/；审计日志按惯例不回改，此处仅备忘。
5. **【低】scripts/13 覆盖率分母硬编码**：`main()` 中 `totalCount 应为 26,477` 为探测期
   快照值，官方库随时间增长后会失真；可改为读取分页返回的实时 totalCount。
6. **【已解决】notes/08 命名混淆**：旧版报告已归档 deprecated/08_optimization.md，
   SPEC 产物指针已改 notes/12_optimization.md，deprecated/README 记录迁移原因。
