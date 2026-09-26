# OMat24 元素体相参考能量描述符筛选

基线（84 特征）GroupKFold MAE = **0.1166 eV**（3 种子）。

| 配置 | 特征数 | MAE (eV) | Δ vs 基线 |
|---|---:|---:|---:|
| baseline | 84 | 0.1166 | +0.0000 |
| +ebulk_ref | 89 | 0.11683 | +0.0002 |

- OMat24 参考文件中缺失、按均值插补的元素：[]
- 该量为元素参考相的 VASP(PBE) 体相总能/原子（eV），不是内聚能（缺孤立原子能量），也不是吸附标签。
- OMat24 主体为体相晶体、不含表面 H 吸附；其对本吸附模型的增量预期有限，真正增益需表面弛豫返回的几何/电子结构（C-3 v2 回灌）。
- 数据源：facebook/OMAT24 references/omat-elemental-reference-compounds.json.gz；Barroso-Luque et al., arXiv:2410.12771。