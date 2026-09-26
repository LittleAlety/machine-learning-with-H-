# 14 — 第二外部验证：Catalysis-Hub 同质金属 HER 子集（零重训）

脚本：`scripts/14_extval2_homogeneous.py`。
定位：在 EqV2 跨库验证（`notes/07`，MAE 0.347 / bias +0.158）之外，用 Catalysis-Hub
非 Mamun 池中的**同质金属 HER 条目**构建第二个外部验证集。与 EqV2（不同库、
VASP/RPBE、任意晶系金属间化合物、单点协议）不同，本子集与主线同为表面科学
口径的 DFT 吸附能，可分离"泛函/参考态差异"与"模型本身外推能力"两个误差来源。
决策前提（用户授权）：主线保持 Mamun 单源纯净，**不合并重训**，本子集仅作验证。

## 1. 子集构建与清洗（每步剔除记录）

数据源 `data_processed/cathub_hstar_extra.csv`（15,327 行，87% 覆盖下界）：

| 步骤 | 规则 | 剔除 | 余 |
|---|---|---|---|
| 原始 | 非 Mamun 全池 | — | 15,327 |
| 1 反应身份 | 05 同规则的计量推广：仅保留 k·(0.5H₂(g)+*)→k·H* 纯 H 吸附（两侧物种仅 H₂(g)/*/H* 且 H 配平），共吸附（N₂/CH₄/H₂O/H₂S/CO*）与不平衡方程（`H2(g) -> H*`）剔除 | 1,515 | 13,812 |
| 2a 组成可解析 | gcd 约简规范化（同 05） | 0 | 13,812 |
| 2b 同质金属纯度 | 含 H,B,C,N,O,F,P,S,Cl,Se,Br,I 的组成一律排除 → 硼化物/氮化物/氧化物/M-N-C/分子催化剂/H 预覆盖表面 | 12,969 | 843 |
| 3a 物理范围 | \|E_perH\| ≤ 5 eV（Boes +22 eV 类异常实际已在步骤1随共吸附方程剔除，此步兜底） | 0 | 843 |
| 3b 稳健离群 | median ± 8×1.4826·MAD（median=−0.277, MAD=0.540 eV） | 0 | 843 |
| 4 聚合 | 按 (pubId, comp, facet) 取最低 E_perH（最稳定位点口径，同 11/v1） | — | **355 表面** |

能量归一：Cathub `reactionEnergy` 为按方程式书写的总反应能（Mamun 内实测
2H 方程 ≈ 2×1H 方程，中位偏差 0.21 eV），故 k>1 的行 E_perH = E/k
（HansenFirst2018、WangAchieving2021 等全部为 k=2，靠此规则得以纳入）；
60/355 表面含 k>1 构型，k=1 严格稀释口径单独分层（敏感性检验）。

纳入来源（22 个，行级 843）：BoesAdsorption2018（97）、AlonsoStrain2023（347）、
PasumarthiFacetDependence2023（132）、MontoyaThe2015（47）、KaiData-driven2022（45）、
WangAchieving2021（31）、FantaQMCcopperCO2024（28）、TangFrom2020（22）、
HansenFirst2018（17）、PengRole2020（15）、Gauthierrole2021（12）、ClarkInfluence2018（11）、
TangModeling2020（7）、SchumannSelectivity2018（6）及 8 个 ≤4 行小集。
逐项排除：Zhang 系硼化物/MBene（9,911 行）、Yohannes 氮化物（2,590）、
Dickens 氧化物（213）、Dheer 酶簇、Jung/Hossain M-N-C、E.Molecular 分子催化剂、
Bukas C₈O₃、WangUniversal/Logadottir/Medford 的 `H2(g)->H*` 不平衡方程等。

特征化与 11 完全同口径（复用 06 函数）；structure one-hot：单元素组成 → A1=1
（训练集 M12 超胞 ≡ A1(111) 模式），合金 → 全 0（表面计量模式未知不臆造）；
facet one-hot：111/101 在域内置 1，其余晶面置 0（外推）。镧系 Nd/Sm 的
group_id mendeleev 缺失，本地兜底记 3 族（已在日志记录，且这些组成本就在
Mamun 37 元素域外）。

## 2. 零重训验证结果

出厂模型：XGBoost_tuned（lr=0.03 / depth=7 / n=400 / sub=0.8，种子 42，
主线 1,836 行全量训练；内部 OOF MAE=0.120 eV 复算一致）。**零重训**直接预测：

| 分层 | n | MAE | RMSE | R² | bias (pred−true) | 去偏差 MAE |
|---|---|---|---|---|---|---|
| **ALL** | 355 | **0.177** | 0.245 | 0.742 | **+0.088** | 0.160 |
| 仅 k=1（严格稀释） | 295 | 0.172 | 0.235 | 0.772 | +0.091 | 0.151 |
| 域内（单元素 + facet=111） | 75 | **0.141** | 0.211 | 0.629 | +0.022 | 0.147 |
| 元素全在 Mamun 37 内 | 326 | 0.181 | 0.247 | 0.742 | +0.088 | 0.162 |

按泛函分层（偏差高度泛函依赖，定位了系统性偏移来源）：

| 泛函 | n | MAE | bias |
|---|---|---|---|
| BEEF-vdW（与主线同泛函） | 114 | 0.142 | **+0.001** |
| RPBE | 8 | 0.056 | +0.030 |
| PBE（AlonsoStrain 为主） | 203 | 0.184 | +0.146 |
| MS2（KaiData-driven，meta-GGA） | 5 | 0.353 | +0.353 |
| RPBE_-0.413VSHE（Gauthier，电位标注） | 12 | 0.365 | **−0.150** |

按来源节录：Boes 0.060（n=7）、Montoya 0.090（12）、TangModeling 0.059（7）、
Schumann 0.081（6）、PengRole 0.077（15）为最优组；AlonsoStrain 0.185（202，
金属间化合物为主）、Hansen 0.221（17，k=2 归一）、WangAchieving 0.201（26）居中；
Gauthier（VSHE 电位标注）与 Kai（MS2 泛函）偏差最大且方向相反。

产物：`data_processed/extval2_homogeneous.csv`（355 表面 + 预测/残差/域内标记）、
`outputs/extval2_metrics.csv`、`figures/extval2_parity.png`。

## 3. 结论（诚实版）

1. **同源 DFT 验证显著优于跨库验证**：同质金属子集全体 MAE 0.177 eV、
   域内 0.141 eV、同泛函（BEEF-vdW）0.142 eV 且 bias≈0——逼近内部 OOF
   （0.120 eV）；而 EqV2 跨库 MAE 0.347 eV、bias +0.158 eV。
   **精度落差的主因是库间协议差（泛函/参考态/弛豫协议）与组成域外度，
   而非模型本身缺陷**：在与训练分布同族的体系上，出厂模型几乎无系统偏差。
2. **系统性偏移可按泛函定位**：PBE 系 +0.15、MS2 +0.35、VSHE 电位标注 −0.15，
   BEEF-vdW/RPBE ≈ 0。去全局常数偏差后 MAE 仅从 0.177 → 0.160，
   因为残差是"泛函差 × 体系"的叠加而非单一平移（与 notes/07 对 EqV2 的结论一致）。
3. **k>1 覆盖度归一可信**：k=1 严格稀释口径（0.172）与全体（0.177）差异 <0.01 eV，
   E/k 标度律在交叉库上再次自洽。
4. 局限：① 增量池 87% 覆盖下界，355 表面为保守估计；② 合金 structure 全 0、
   非 111/101 晶面 one-hot 全 0 属域外外推（与 11 同规则，已分层报告）；
   ③ Gauthier/Kai 等电位标注/杂化泛函小集（n≤12）的偏差解读受样本量限制；
   ④ 本验证为"同库不同文献"，仍共享 Cathub 的收录协议，严格意义上介于
   内部验证与 EqV2 式跨库验证之间。
