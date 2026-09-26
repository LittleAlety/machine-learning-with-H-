# 24. 头部候选的实验文献交叉核对（HER）

**任务**：检验 XGBoost 模型（1,836 个 DFT 组成，GroupKFold OOF MAE=0.1198 eV）给出的 Top-15 HER 候选（按 Sabatier 原则 ΔG_H*≈0 排序）与已发表实验电催化 HER 证据是否一致。
**方法**：以英文关键词（组成式 + "hydrogen evolution / HER / overpotential / exchange current"）检索期刊论文、会议摘要与学位论文；逐候选记录定量数据（η@10 mA cm⁻²、j0、Tafel 斜率）或定性报道。检索时间：2026 年。
**判定标准**：
- **支持**：同组成或紧邻组成（同族合金）有实验 HER 证据，且实测活性较高（与模型"近零 ΔG=优良候选"一致）；
- **矛盾**：有实验证据且实测活性差（与预测相反）；
- **间接**：无该组成记录，但同族（同元素对/同构型）体系有定量 HER 数据；
- **无记录**：未检索到任何实验 HER 记录（可能意味着预测的新颖性）。

---

## 1. 锚点对照：模型排名 vs 经典实验火山曲线

| 锚点 | 出处 | 与本项目的相关性 |
|---|---|---|
| Trasatti 实验火山：j0 vs M–H 键强度，Pt/Rh/Ir 居顶，W/Mo/Ta/Nb 结合过强，Ag/Au/Cu/Bi/Tl 结合过弱 | Trasatti, *J. Electroanal. Chem.* 1972, 39, 163–184 | 模型把 "强结合贵金属 × 弱结合主族/3d 金属" 的合金（CuPt3、CdPd3、RhTl、BiY…）预测为 ΔG≈0，正是 Trasatti 火山两侧元素"调平"的物理图像，方向自洽 |
| Nørskov DFT 火山：j0 vs 计算 ΔG_H*，ΔG≈0 处 j0 最大，Pt 为最优 | Nørskov et al., *J. Electrochem. Soc.* 2005, 152, J23–J26, DOI 10.1149/1.1856988 | 本项目筛选原理（ΔG_H*→0）与该文完全一致；该文已知 Pt、Pd、Ir、Rh 在峰顶附近 |
| DFT 高通量筛选→实验验证的先例：700+ 二元表面合金中预测 BiPt 优于 Pt，实验合成证实 | Greeley et al., *Nat. Mater.* 2006, 5, 909–913, DOI 10.1038/nmat1752 | 证明"计算/ML 筛选近零 ΔG 合金→实验命中"的范式可行；BiPt 与本清单中 RhTl、BiY 等含弱结合主族元素的候选同属一类设计思想 |
| 碱性 HBE 火山（实验） | Sheng et al., *Energy Environ. Sci.* 2013, 6, 1509–1512 | 提醒：碱性介质中 ΔG_H* 之外还有水解离（Volmer）势垒，含亲氧元素（V、Cr、Zr…）的合金在碱性下可能获得额外促进——Ir3V/Pt3V 实验正是如此 |

**小结**：模型预测的近零 ΔG 集合在组成类型上（Pt/Pd/Rh/Ir/Os/Ru 与弱结合金属配对）与经典火山曲线的预期吻合，未发现"把已知很差元素单独排入头部"的系统性错误。

---

## 2. 总表：Top-15 候选的实验证据交叉核对

| # | 候选（模型 ΔG, eV） | 实验证据（定量优先） | 判定 | 主要来源 |
|---|---|---|---|---|
| 1 | CuPt3 (L12,111), ≈0.000 | CuPt 合金纳米粒子（Cu:Pt≈44:56）：酸性 η₁₀=39 mV，优于纯 Pt NP（61 mV）；中性稳定性约为 Pt 的 100 倍；碱性下 Cu 溶出重构后性能提升。CuPt/rGO 在 8 M KOH 中 j0=0.09–0.89 mA cm⁻²（298–338 K）。CuPt3 本身（L12/L1₃ 有序相在相图研究中存在）无直接 HER 记录 | **支持**（同族紧邻组成） | Liu et al., *RSC Adv.* 2021, DOI 10.1039/D0RA09386F（PMC8696985）; *Catalysts* 2026, 16(3), 236（mdpi.com/2073-4344/16/3/236） |
| 2 | CrPt3 (L12,111), ≈0.000 | 未检索到 CrPt3 或 Cr-Pt 合金的 HER 实验记录。Cr-Pt 体系有相图/结构研究（Preußner et al. 2009），Pt-Cr 主要是 ORR 合金；单质 Cr 在 Trasatti 火山中 HER 很弱（j0≈1.25×10⁻⁷ A cm⁻²）。注意 Cr 的强亲氧/钝化可能带来表面结构问题（模型未标红旗，但值得人工复核） | **无记录** | Preußner et al., *Mater. Sci. Eng. A* 2009, DOI 10.1016/j.msea.2008.11.015；Trasatti 1972 |
| 3 | AgNi (L10,101), ≈−0.001 | Ni–Ag 体相合金（电子束共蒸镀克服互不相溶）：Ni₀.₇₅Ag₀.₂₅ 的碱性 HER 活性（几何面积归一）约为纯 Ni 的 2 倍，稳定性相当；DFT 证实合金产生近最优氢结合能位点。另有 AgNi 合金纳米粒子/碳量子点体系用于全解水的报道 | **支持** | Tang, Hahn, Klobuchar, Ng, Wellendorff, Bligaard, Jaramillo, *Phys. Chem. Chem. Phys.* 2014, 16, 19250–19257；Sayed et al., *J. Alloys Compd.* 2021（S0925838820328565） |
| 4 | Zn3Zr (L12,111), ≈−0.001 | 未检索到任何 HER 记录。Zr-Zn 金属间化合物（ZrZn2 等）仅有超导/结构研究；Zr 合金的氢吸收研究属于腐蚀科学语境 | **无记录** | 检索 "Zn3Zr / ZrZn2 hydrogen adsorption OR catalysis" 无结果 |
| 5 | CdPd3 (L12,111), ≈−0.002 | 未检索到 HER 记录。Pd-Cd 仅见于热催化（加氢/脱氢选择性）与相图文献；Cd 为高挥发性有毒元素，电催化界基本回避 | **无记录** | 检索 "CdPd3 / Pd-Cd hydrogen evolution" 无结果 |
| 6 | Au3Ir (L12,111), ≈+0.002 | 无 Au3Ir 金属间化合物的 HER 记录。间接证据：(i) Au(111) 上自发沉积 Ir 显著提升 HER（Štrbac et al., *J. Electrochem. Soc.* 2018, 165, DOI 10.1149/2.0441815jes）；(ii) AuIr/C 复合物（未成合金）作 ORR/OER 双功能催化剂（Yuan et al., *J. Energy Chem.* 2016）。Au 结合 H 过弱、Ir 偏强，合金化调平方向与模型一致 | **间接**（方向支持） | iopscience.iop.org/article/10.1149/2.0441815jes；sciencedirect.com/science/article/pii/S209549561630050X |
| 7 | CoOs3 (L12,111), ≈+0.002 | 无 Co-Os 合金 HER 记录。间接：Os 基 HER 研究近年兴起——单原子 Os（Os-SA@SNC）1 M KOH 中 η₁₀=13 mV，优于 Pt/C（19 mV）；但文献明确指出**纯 Os 对中间体吸附过强**、活性受限。模型给出 CoOs3 ΔG≈+0.002（略弱结合）与之定性自洽：Co 稀释/调弱 Os–H | **间接** | "Sulfur-mediated transformation from osmium nanocrystals to single atoms…"，PMC12260528（2025）；Cao et al., *Energy Reviews* 2023, 2, 100053（Os 基催化综述） |
| 8 | Ag3Zr (L12,111), ≈+0.003 | 未检索到 HER 记录。Ag-Zr 仅有非晶/结构研究；近邻 CuZr 金属玻璃粉体经 HF 脱合金后有 HER/OER 活性报道（Xie et al., *Catalysts* 2022, 12, 1378），提示 Zr 组分表面易氧化、需脱合金重构 | **无记录** | mdpi.com/2073-4344/12/11/1378（旁证） |
| 9 | RhTl (L10,101), ≈+0.004 | 未检索到 Rh-Tl 合金的 HER（乃至任何催化）记录。Rh 本身处于 Trasatti 火山顶部（优良 HER 金属），Tl 极弱结合 H，"Rh×Tl 调平至 ΔG≈0"在物理上合理，但 Tl 的毒性/稳定性是实际障碍 | **无记录** | 检索 "RhTl / Rh-Tl alloy catalyst" 无结果 |
| 10 | BiY (L10,101), ≈+0.004 ⚑ | 未检索到 Bi-Y 体系的任何催化记录。红旗佐证充分：Bi 位于 Trasatti 火山最左翼（H 结合极弱，HER 惰性元素）；Y 为强亲氧稀土，Pt-RE 合金文献证实 Y 在电化学环境中会向表面迁移并氧化/溶解（见 §4）。两元素结合出现近零 ΔG 更可能是数据外推假象，实验可实现性存疑 | **无记录**（红旗佐证：成立） | PMC9804461（Pt-RE 综述）；Trasatti 1972 |
| 11 | Pd3V (L12,111), ≈+0.004 | Pd3V 本身无 HER 记录（仅有 DFT 结构/弹性研究与 Pd-V 氢分离膜研究——氢会显著加速 Pd-V 金属间互扩散，提示 H 环境下的相稳定性问题）。但同构 L12/A15 族的 **Ir3V** 与 **Pt3V** 有强实验记录：有序 Ir3V 在 1 M KOH 中 η₁₀=9.0 mV（质量活性为 Pt/C 的 6.7 倍）；Pt3V 在 0.5 M H₂SO₄ 中 η₁₀≈20 mV、Tafel 36 mV dec⁻¹、500 mA cm⁻² 下 100 h 无衰减，均超越商业 Pt/C | **间接（强支持族）** | Chen et al., *Nano Energy* 2021, 81, 105636；Da et al., *Adv. Energy Mater.* 2023, 13, 2300127, DOI 10.1002/aenm.202300127；Pd-V 扩散：OSTI 2371623 |
| 12 | LaSn (L10,101), ≈+0.005 ⚑ | 未检索到 La-Sn 合金的 HER 记录。红旗佐证：La 为最强亲氧元素之一，LaNi₅ 类 La 基金属间化合物以**储氢/吸氢**著称（H 进入体相而非表面脱附），La-Sn 在空气中自发氧化；La 基 IMC 的电催化报道仅限 NO₃RR（LaCoSi/LaCuSi），其中 HER 是竞争副反应 | **无记录**（红旗佐证：成立） | Nesterenko et al., *Eng. Proc.* 2026（LaCoSi NO3RR/HER 竞争） |
| 13 | Cd3Ti (L12,111), ≈+0.005 | 未检索到 HER 记录。Cd-Ti 互溶性差、Cd 毒性大，电催化界无此体系 | **无记录** | 检索无结果 |
| 14 | Pt3Sc (L12,111), ≈+0.005 ⚑ | **HER 会议报告**（Baik, Lee, Pak，第 235 届 ECS 会议，Dallas 2019）：电子束辐照法合成 Pt3Sc/C 与 Pt3La/C，Pt3La/C 在 η=100 mV 处质量活性最高但易腐蚀；**Pt3Sc/C 稳定性显著提高，3000 次 HER 循环后无明显衰减**。ORR 证据充分：多晶 Pt3Sc 的 ORR 活性约为 Pt 的 1.5–3 倍（Chorkendorff 组），Pt3Sc/PECNTs 单电池功率密度 760 mW cm⁻²（Garapati & Sundara, *Int. J. Hydrogen Energy* 2019）。DFT  segregation 能数据（*Energies* 2021, 14, 7814）：洁净表面 Sc 偏析能 +1.05 eV（Pt 皮稳定），但**有氧吸附时降至 −0.02 eV（Sc 偏析到表面）**——直接支持"强亲氧 Sc"红旗 | **部分支持**（活性证据少、稳定性证据有；红旗佐证成立） | scholar.gist.ac.kr/handle/local/22891；sciencedirect S0360319919307840；mdpi.com/1996-1073/14/22/7814 |
| 15 | Co3Ru (L12,111), ≈−0.005 | **强实验记录（同族 CoRu 合金）**：CoRu@N-CNTs 在 1 M KOH 中 η₁₀=19 mV、Tafel 26.2 mV dec⁻¹，优于 Pt/C（Gao et al., *J. Electrochem.* 2024, 30, 2403081）；RuCo 纳米合金 η₁₀=12.8 mV（1 M KOH）、18 mV（碱性海水）（Meng et al., *J. Catal.* 2025）；CoRu/CNB 全 pH：21/33/56 mV（碱/酸/中性）（*Catalysts* 2025, 15, 1106）；RuCo@N-石墨烯壳 η₁₀=28 mV（*Nano Energy* 2017 起多篇）；DFT 均归因于 Co–Ru 协同优化 ΔG_H* 至近零——与模型机制解释完全一致 | **支持（最强案例之一）** | jelectrochem.xmu.edu.cn/journal/vol30/iss9/7；sciencedirect S0021979725007003；mdpi.com/2073-4344/15/12/1106 |

---

## 3. 深挖案例

### 3.1 Cu–Pt 体系（对应候选 #1 CuPt3）
- **实验事实**：Liu et al.（*RSC Adv.* 2021, DOI 10.1039/D0RA09386F）水热法合成 20–30 nm 单分散 CuPt 合金（Cu:Pt≈44:56，接近 CuPt 计量比）：0.05 M H₂SO₄ 中 η₁₀=39 mV，比同法纯 Pt NP（61 mV）低 22 mV；中性（pH 7.2）下 1.2×10⁵ s 恒流测试过电位衰减率仅为 Pt 的 ~1%（稳定性约 100 倍）；碱性（1 M KOH）初始 η₁₀=319 mV 反而差于 Pt（128 mV），但 Cu 溶出表面重构后性能反超且稳定。
- **与模型对照**：模型把 CuPt3(111) 排在 ΔG≈0 的第 1 位；实验证实 Cu–Pt 合金在酸/中性确实优于 Pt，方向一致。**但两点警示**：(i) 实验最优组成是近 CuPt 而非 CuPt3，且碱性初始活性差、性能提升依赖 Cu 溶出重构——说明真实工作表面偏离理想 L12(111) 终止面；(ii) *Catalysts* 2026（16, 236）报道 CuPt/rGO 在浓碱中 j0（298 K 仅 0.09 mA cm⁻²）低于 CoPt、NiPt，与"Cu 系在碱性初始较差"互洽。
- **结论**：支持模型排序的方向性，但理想有序表面与真实（重构后）表面的差异不可忽视——这正是表面吸附能模型未覆盖的部分。

### 3.2 Pt₃Sc / Pt–稀土体系（对应红旗候选 #14 Pt3Sc，兼及 #10 BiY、#12 LaSn 的红旗逻辑）
- **HER 实验**：Baik, Lee & Pak（235th ECS Meeting, Dallas, 2019；GIST 机构库）首次用电子束辐照合成 Pt₃Sc/C 与 Pt₃La/C：Pt₃La/C 在 η=100 mV 处质量活性最高但"低稳定、易腐蚀"；**Pt₃Sc/C 3000 次 HER 循环无可见衰减**——与模型把 Pt₃Sc 预测为 ΔG≈+0.005 的头部候选定性一致（活性好），且实验上它是 Pt–稀土中较稳定者。
- **ORR/稳定性旁证**：Chorkendorff 组多晶 Pt₃Y/Pt₃Sc ORR 活性为 Pt 的 1.5–6 倍；Pt₃Sc/PECNTs 燃料电池单电池 760 mW cm⁻²（Garapati & Sundara, *IJHE* 2019）。Pt–稀土综述（PMC9804461）明确指出："Y 或 Sc 原子从合金内部向表面迁移，最终溶解或氧化"，其能垒与化合物生成焓相关——即**强亲氧稀土的表面偏析/流失是实验上已被观测的真实失效通道**。
- **DFT 佐证红旗**：*Energies* 2021（14, 7814）计算 Pt₃Sc 洁净表面 Sc 偏析能 +1.05 eV（稳定 Pt 皮），但**氧吸附后降至 −0.02 eV**——氧化环境下 Sc 会翻出表面。这直接支持模型对"强亲氧 Sc"的红旗标注，也解释了为什么 Pt₃Sc 的 HER 证据只见于精心控制的会议级样品。
- **结论**：Pt₃Sc 是"活性预测可信、但表面稳定性需红旗"的典型——模型的 ΔG 排序与红旗机制在实验文献中各找到独立佐证。

### 3.3 Co–Ru 合金（对应候选 #15 Co3Ru）
- **实验事实（多篇独立重复）**：
  - CoRu@N-CNTs（Gao et al., *J. Electrochem.* 2024, 30, 2403081）：1 M KOH 中 η₁₀=19 mV，Tafel 26.19 mV dec⁻¹，优于 Pt/C；归因于 Co–Ru 位点电子通信。
  - RuCo 合金纳米粒子（Meng et al., *J. Catal.* 2025）：1 M KOH η₁₀=12.8 mV，模拟碱性海水 18 mV。
  - CoRu/CNB（*Catalysts* 2025, 15, 1106）：碱/酸/中性 η₁₀=21/33/56 mV，Tafel 33/39/74 mV dec⁻¹，三介质均优于商业 Pt/C。
  - RuCo@N 掺杂石墨烯壳（*Nano Energy* 2017 等）：η₁₀=28 mV，10,000 循环稳定；DFT 指出 Ru 掺入 Co 核降低 ΔG_H*。
  - RuCo-PBA 衍生合金（*ChemCatChem* 2025, cctc.202500178）：η₁₀=129 mV（较差的一支，载体/结构不同）。
- **与模型对照**：模型给 Co₃Ru(111) ΔG≈−0.005（近零）；各篇实验的 DFT 均独立得出"Co–Ru 协同把 Ru–H/Co–H 结合调至接近热中性"。**机理层面的一致性是三个深挖案例中最强的**。
- **差异点**：实验体系均为纳米合金/碳复合物、近 1:1 计量比（hcp 固溶体或合金核），而非化学计量 L12 型 Co₃Ru 有序表面；模型预测的具体表面结构尚无直接实验对应物。

---

## 4. 红旗项的实验侧面佐证

| 红旗 | 实验佐证 | 来源 |
|---|---|---|
| #10 BiY（Bi 为 LOEO 高误差元素；Y 强亲氧） | Bi 位于 Trasatti 实验火山最左翼，是 HER 惰性元素——Bi 作主元出现"ΔG≈0"本身就提示外推风险；Y 的强亲氧性在 Pt–Y 文献中被反复证实（Y 向表面迁移、氧化/溶解；Pt₅Y 初始活性高但首 600 循环即快速衰减） | Trasatti 1972；PMC9804461 |
| #12 LaSn（La 强亲氧） | La 基金属间化合物（LaNi₅ 类）以体相储氢著称，表面极易氧化；唯一检索到的 La-IMC 电催化工作（LaCoSi/LaCuSi，NO₃RR）中 HER 只是副反应；Pt₃La/C 的 HER 实验（ECS 2019）明确报道其"易腐蚀、稳定性低" | Baik et al., ECS 2019；Eng. Proc. 2026 |
| #14 Pt₃Sc（Sc 强亲氧） | 见 §3.2：氧吸附使 Sc 偏析能从 +1.05 eV 降到 −0.02 eV（*Energies* 2021）；Pt–RE 综述记录 Sc/Y 表面迁移-溶解通道；但 Pt₃Sc 是三者中唯一有正面 HER 稳定性实验（3000 循环）者 | *Energies* 2021, 14, 7814；PMC9804461；ECS 2019 |

**红旗机制小结**：三个红旗项的担忧（强亲氧元素在电化学/氧化环境下表面偏析、氧化、流失）全部能在 Pt–稀土实验文献中找到独立佐证；Pt₃Sc 因生成焓大、Sc 迁移能垒高而最稳定，这与"红旗程度"的排序（BiY、LaSn 更可疑）一致。

---

## 5. 一致性统计与总体可信度结论

**统计（15 个候选）**：
- 有实验证据（含直接同族记录）：**7/15** —— CuPt3（支持）、AgNi（支持）、Au3Ir（间接）、CoOs3（间接）、Pd3V（间接强支持：Ir3V/Pt3V）、Pt3Sc（部分支持）、Co3Ru（支持）。
- 其中明确**支持**模型预测（实测高活性/近零 ΔG 方向一致）：4 例（CuPt3、AgNi、Co3Ru、Pd3V 族）；间接/部分支持 3 例。
- **矛盾**：0 例。
- **完全无实验记录**：8/15（CrPt3、Zn3Zr、CdPd3、Ag3Zr、RhTl、BiY、LaSn、Cd3Ti）——这部分体现了预测的新颖性，但也意味着无法用实验检验；其中含全部 2 个主族-稀土红旗项（BiY、LaSn）。

**总体可信度评级：中等偏上（B+）**
1. **方向性可信**：模型把"强结合贵金属 × 弱结合金属"的配对预测为近零 ΔG，与 Trasatti/Nørskov 火山的物理图像一致；4 个有定量实验的体系（CuPt、Ni-Ag、CoRu、Ir3V/Pt3V 族）全部实测为高活性 HER 催化剂，无矛盾案例。模型头部清单"富集真实 HER 活性合金"的能力得到交叉验证。
2. **主要保留意见**：
   - 实验证据几乎全部来自**纳米颗粒/无序合金/载体复合体系**，与模型假设的化学计量有序表面（L12(111)、L10(101) 特定终止面）存在结构鸿沟；CuPt 案例显示真实活性表面在工况下重构（Cu 溶出），理想表面预测只能视为"起点"排序。
   - 8/15 无记录候选中混有物理上可疑的组合（BiY、LaSn、RhTl、Cd3Ti、CdPd3、Ag3Zr、Zn3Zr）——含 Cd/Tl 的毒性元素组合与稀土-主族组合大概率是 DFT 训练集外推产物，红旗机制（强亲氧/高误差元素）的标注与实验文献佐证吻合，建议在最终推荐清单中降权或剔除。
   - CrPt3 虽无红旗，但 Cr 的钝化/亲氧与 Trasatti 火山上 Cr 的极低 j0 提示应人工复核其表面态假设。
3. **推荐优先验证序**：CoRu 系（实验已多独立证实，可做结构对照）> Cu-Pt 系（关注工况重构）> Pd₃V/Pt₃V/Ir₃V 族（L12 有序性的直接靶点）> Pt₃Sc（稳定性红旗与活性并存，适合做表面偏析对照实验）。

---

## 6. 参考文献（含 URL）

1. Liu X., Yang C., Li X. et al. A monodispersed CuPt alloy: synthesis and its superior catalytic performance in the hydrogen evolution reaction over a full pH range. *RSC Advances* 2021, DOI 10.1039/D0RA09386F. https://pmc.ncbi.nlm.nih.gov/articles/PMC8696985/
2. Bimetallic M–Pt (M = Co, Ni, Cu) Alloy Nanoparticles on Reduced Graphene Oxide for Alkaline HER. *Catalysts* 2026, 16(3), 236. https://www.mdpi.com/2073-4344/16/3/236
3. Tang M.H., Hahn C., Klobuchar A.J., Ng J.W.D., Wellendorff J., Bligaard T., Jaramillo T.F. Nickel–silver alloy electrocatalysts for hydrogen evolution and oxidation in an alkaline electrolyte. *Phys. Chem. Chem. Phys.* 2014, 16, 19250–19257. https://suncat.stanford.edu/publications/nickel-silver-alloy-electrocatalysts-hydrogen-evolution-and-oxidation-alkaline
4. Sayed E.T. et al. Nitrogen doped carbon quantum dots conjugated with AgNi alloy nanoparticles as potential electrocatalyst for efficient water splitting. *J. Alloys Compd.* 2021. https://www.sciencedirect.com/science/article/pii/S0925838820328565
5. Gao M.-T., Wei Y. et al. Electronic Communication between Co and Ru Sites Decorated on Nitrogen-doped Carbon Nanotubes Boost the Alkaline HER. *Journal of Electrochemistry* 2024, 30(9), 2403081. https://jelectrochem.xmu.edu.cn/journal/vol30/iss9/7/
6. Meng W. et al. Ruthenium-cobalt alloy nanoparticles uniformly dispersed … (HER, η₁₀=12.8 mV in 1 M KOH). *Journal of Catalysis* 2025. https://www.sciencedirect.com/science/article/pii/S0021979725007003
7. CoRu Alloy/Ru Nanoparticles: A Synergistic Catalyst for Efficient pH-Universal Hydrogen Evolution. *Catalysts* 2025, 15(12), 1106. https://www.mdpi.com/2073-4344/15/12/1106
8. Ruthenium Cobalt Nanoalloy Derived from Its Prussian Blue Analogue for Efficient Hydrogen Evolution Electrocatalysis. *ChemCatChem* 2025, cctc.202500178. https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cctc.202500178
9. Da Y., Jiang R. et al. Development of a Novel Pt₃V Alloy Electrocatalyst for Highly Efficient and Durable Industrial Hydrogen Evolution Reaction in Acid Environment. *Adv. Energy Mater.* 2023, 13, 2300127. DOI 10.1002/aenm.202300127. https://www.x-mol.com/paper/1636164414290325504
10. Chen L.-W., Guo X. et al. Structurally ordered intermetallic Ir₃V electrocatalysts for alkaline hydrogen evolution reaction. *Nano Energy* 2021, 81, 105636. DOI 10.1016/j.nanoen.2020.105636. https://inis.iaea.org/records/c7dp6-anx61
11. Baik C., Lee S.W., Pak C. Investigation of New Pt Alloy Catalyst with Rare Earth Elements for Hydrogen Evolution Reaction. 235th ECS Meeting, Dallas, 2019 (conference abstract). https://scholar.gist.ac.kr/handle/local/22891
12. Garapati M.S., Sundara R. Highly efficient and ORR active platinum-scandium alloy-partially exfoliated carbon nanotubes electrocatalyst for PEMFC. *Int. J. Hydrogen Energy* 2019. https://www.sciencedirect.com/science/article/pii/S0360319919307840
13. Platinum–Rare Earth Alloy Electrocatalysts for the Oxygen Reduction Reaction: A Brief Overview (Pt₃Y/Pt₃Sc 表面迁移与氧化失效讨论). https://pmc.ncbi.nlm.nih.gov/articles/PMC9804461/
14. First-Principles Study of Pt-Based Bifunctional OER/ORR Electrocatalyst（Pt₃Sc 偏析能：洁净 +1.05 eV，氧吸附 −0.02 eV）. *Energies* 2021, 14, 7814. https://www.mdpi.com/1996-1073/14/22/7814
15. Santos D.M.F. et al. Platinum–rare earth electrodes for hydrogen evolution in alkaline water electrolysis. *Int. J. Hydrogen Energy* 2013, 38, 3137–3145. https://www.sciencedirect.com/science/article/abs/pii/S0360319912028376
16. Štrbac S. et al. Electrocatalysis of Hydrogen Evolution Reaction on Au(111) Modified by Spontaneously Deposited Ir. *J. Electrochem. Soc.* 2018, 165. https://iopscience.iop.org/article/10.1149/2.0441815jes
17. Yuan L. et al. Gold-iridium bifunctional electrocatalyst for ORR and OER. *J. Energy Chem.* 2016. https://www.sciencedirect.com/science/article/pii/S209549561630050X
18. Sulfur-mediated transformation from osmium nanocrystals to single atoms for efficient alkaline HER（Os-SA@SNC η₁₀=13 mV；纯 Os 吸附过强）. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12260528/
19. Cao et al. Revitalizing osmium-based catalysts for energy conversion. *Energy Reviews* 2023, 2, 100053. https://www.sciencedirect.com/science/article/pii/S2772970223000408
20. Acceleration of Pd–V intermetallic diffusion by hydrogen（Pd-V 膜 H 加速互扩散）. OSTI 2371623. https://www.osti.gov/servlets/purl/2371623
21. Greeley J., Jaramillo T.F., Bonde J., Chorkendorff I., Nørskov J.K. Computational high-throughput screening of electrocatalytic materials for hydrogen evolution (BiPt). *Nat. Mater.* 2006, 5, 909–913. DOI 10.1038/nmat1752. https://pubmed.ncbi.nlm.nih.gov/17041585/
22. Nørskov J.K. et al. Trends in the Exchange Current for Hydrogen Evolution. *J. Electrochem. Soc.* 2005, 152, J23–J26. DOI 10.1149/1.1856988.
23. Trasatti S. Work function, electronegativity, and electrochemical behaviour of metals: III. Electrolytic hydrogen evolution in acid solutions. *J. Electroanal. Chem.* 1972, 39, 163–184.
24. Sheng W., Myint M., Chen J.G., Yan Y. Correlating the HER activity in alkaline electrolytes with the hydrogen binding energy on monometallic surfaces. *Energy Environ. Sci.* 2013, 6, 1509–1512.
25. Preußner J. et al. Determination of phases in the system chromium–platinum. *Mater. Sci. Eng. A* 2009. https://www.sciencedirect.com/science/article/abs/pii/S0921509308014561
26. Xie Z. et al. CuZr Metal Glass Powder as Electrocatalysts for HER and OER. *Catalysts* 2022, 12, 1378. https://www.mdpi.com/2073-4344/12/11/1378
27. Nesterenko S. et al. La-based intermetallic compounds (LaCoSi/LaCuSi) as catalyst in electrochemical ammonia synthesis（LaCoSi 上 HER 为竞争副反应）. *Eng. Proc.* 2026, 117, 71. https://www.mdpi.com/2673-4591/117/1/71
