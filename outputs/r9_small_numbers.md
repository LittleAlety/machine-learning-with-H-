# Round 9 小数字核实（全部本地重算，scripts/29）

## 3a_OOF残差占比 — v2(36feat)

- 结果：|res|≤0.2eV: 1579/1836=0.8600；|res|≤0.1eV: 1247/1836=0.6792
- 细节：MAE=0.1198；median|r|=0.0561

## 3a_OOF残差占比 — v3(84feat)

- 结果：|res|≤0.2eV: 1610/1836=0.8769；|res|≤0.1eV: 1307/1836=0.7119
- 细节：MAE=0.1134；median|r|=0.0509

## 3b_0.180降幅 — 随机5-fold 模型对比（deprecated 成果汇总报告 §4.2 场景）

- 结果：Ridge 0.180159 → XGB_tuned 0.120263：降幅 0.3325 (33.25%)；vs Linear 0.180746：33.46%；按舍入值 0.180→0.120：33.33%
- 细节：文档写 34% 有误；精确 33.25%（Ridge 基线），舍入口径 33.3%。注意此为随机5-fold MAE（非 GroupKFold OOF 0.1198）。

## 3c_EqV2_Spearman分层 — ALL(491)

- 结果：ρ=0.1735
- 细节：逐层从 extval_results.csv 重算

## 3c_EqV2_Spearman分层 — eqv2_361_relaxed(219)

- 结果：ρ=0.2915
- 细节：逐层从 extval_results.csv 重算

## 3c_EqV2_Spearman分层 — eqv2_500_sp(272)

- 结果：ρ=0.0918
- 细节：逐层从 extval_results.csv 重算

## 3c_EqV2_Spearman分层 — facet=111(109)

- 结果：ρ=0.0030
- 细节：逐层从 extval_results.csv 重算

## 3c_EqV2_Spearman分层 — 不含Sb(364)

- 结果：ρ=0.1923
- 细节：逐层从 extval_results.csv 重算

## 3d_7005vs7048 — Cathub 任务A 能量逐条核对

- 结果：43 = 15 个仅镜像组成的全部构型（43 条）；7048-43=7005
- 细节：镜像 H* 构型 7048（=7,048）；官方稀释方程行 6894（=6,894）；镜像组成 1836，仅镜像 15 个：['AgFe3', 'AgTc', 'CdFe3', 'CrV', 'CuFe3', 'CuHg', 'Fe3Pb', 'Fe3Zn', 'FeNb', 'FeRh3', 'LaNi3', 'Tc', 'TlZn', 'V', 'Zn']；这 15 个组成占镜像构型 43 条 → 核对分母 n_total=7005（=7,005）；差值 43=7048-7005。仅镜像组成中出现在官方非稀释方程（H2+2*->2H*）行的：['AgFe3', 'AgTc', 'CdFe3', 'CrV', 'CuFe3', 'CuHg', 'Fe3Pb', 'Fe3Zn', 'LaNi3', 'Tc', 'TlZn', 'V', 'Zn']

