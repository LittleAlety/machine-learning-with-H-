# 新维度扩展筛选报告（文献驱动 + 公平契约）

基线（84 特征）GroupKFold MAE = **0.1166 eV**（3 种子）。

| 配置 | 特征数 | MAE (eV) | Δ vs 基线 |
|---|---:|---:|---:|
| baseline | 84 | 0.1166 | +0.0000 |
| +ea | 89 | 0.11773 | +0.0011 |
| +aw | 89 | 0.11719 | +0.0006 |
| +rho | 89 | 0.1162 | -0.0004 |
| +avol | 89 | 0.11622 | -0.0004 |
| +pol | 89 | 0.11824 | +0.0016 |
| +tc | 89 | 0.1169 | +0.0003 |
| +shc | 89 | 0.1167 | +0.0001 |
| +engh | 89 | 0.11822 | +0.0016 |
| +all | 124 | 0.11799 | +0.0014 |

- 单独加入即改善（Δ<0）的维度：**['rho', 'avol']**
- 全量加入（+all）Δ = **+0.0014 eV**
- 候选扩展特征文件：`hstar_features_expanded.csv`（仅保留获胜维度，需嵌套 CV/10 种子复核后才上线）

## 仍需 DFT/弛豫才能加入的维度（C-3 v2 回灌后）
功函数 Φ（需 curated 表或 slab 计算）、内聚能、d 带中心/宽度、Fermi softness、广义配位数 GCN、slab 图神经网络（AdsorbML/OC20）、磁矩。

## 文献依据
- Li, Ma, Xin, *Catal. Today* (feature engineering: EN + IP + EA).
- Noh et al., *Chem. Sci.* (d-band width + EN, active learning).
- Kirkvold et al., *J. Phys. Chem. Lett.* 2024 (CatEmbed categorical embeddings).
- Huang & Zhuang (Fermi softness); Calle-Vallejo et al. (GCN).
- Lan et al., AdsorbML; Price et al., *Sci. Adv.* (strain GNN).
- Liasi et al. 2026 (work function HER descriptor); Trasatti 1972.
- Tshitoyan et al., *Nature* 2019 (mat2vec); Zhou et al., Atom2Vec.