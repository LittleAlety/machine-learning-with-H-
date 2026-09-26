"""E5: Workstream E summary + escape-angle red-flag statistics.

Reads: e1_key_pairs.json, e2_gap_verdict.json, e3_ladder.json,
       e4_verdict.json, e4_pooled_vs_separate.csv,
       innovation/multi_adsorbate/dual_candidates.csv
Red-flag rules (existing convention):
  Tc radioactive; Hg/Tl toxicity; La/Y/Sc strong oxophilicity;
  Mn/Bi/Fe LOEO (leaching / oxidation).
Writes: scaling_summary.json, README.md. Idempotent.
"""
import os
import re
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RULES = {'radioactive_Tc': ['Tc'], 'toxic_Hg_Tl': ['Hg', 'Tl'],
         'oxophilic_La_Y_Sc': ['La', 'Y', 'Sc'], 'LOEO_Mn_Bi_Fe': ['Mn', 'Bi', 'Fe']}


def flags(comp):
    els = set(re.findall(r'[A-Z][a-z]?', comp))
    return [k for k, v in RULES.items() if els & set(v)]


def main():
    e1 = json.load(open(os.path.join(HERE, 'e1_key_pairs.json')))
    e2 = json.load(open(os.path.join(HERE, 'e2_gap_verdict.json')))
    e3 = json.load(open(os.path.join(HERE, 'e3_ladder.json')))
    e4 = json.load(open(os.path.join(HERE, 'e4_verdict.json')))
    dc = pd.read_csv(os.path.join(ROOT, 'innovation', 'multi_adsorbate',
                                  'dual_candidates.csv'))

    dc['red_flags'] = dc.comp.map(flags)
    dc['red_flag_count'] = dc.red_flags.map(len)
    n_cand, n_flag = len(dc), int((dc.red_flag_count > 0).sum())
    from collections import Counter
    cnt = Counter(f for fl in dc.red_flags for f in fl)
    redflag = dict(n_candidates=n_cand, n_flagged=n_flag,
                   flagged_fraction=round(n_flag / n_cand, 3),
                   by_rule=dict(cnt),
                   oh_in_training_fraction=round(float(dc.oh_in_training.mean()), 3),
                   interpretation=(
                       'Of the %d compositions nearest to the HER x ORR ideal '
                       'point, %d (%.0f%%) carry at least one red flag '
                       '(radioactivity / toxicity / strong oxophilicity / '
                       'LOEO leaching) and %.0f%% lie outside the OH* training '
                       'set (model extrapolation). "Escape angle = untrusted '
                       'angle": the compositions that escape the scaling-line '
                       'gap are simultaneously model-untrusted (extrapolated) '
                       'and materials-unstable -- double warning.')
                   % (n_cand, n_flag, 100 * n_flag / n_cand,
                      100 * (1 - dc.oh_in_training.mean())))

    summary = {
        'workstream': 'E scaling-relation quantification (plan-hb v3.0)',
        'scope_qualifier': e2['scope_qualifier'],
        'E1_scaling_matrix': {
            'method': 'Theil-Sen (pre-registered), shared (comp,structure,facet)',
            'coverage': '210/210 ordered pairs with n>=30',
            'H_to_OH': e1.get('H->OH'), 'OH_to_H': e1.get('OH->H'),
            'H_to_O': e1.get('H->O'), 'O_to_H': e1.get('O->H'),
            'literature_comparison': (
                'Literature OH*/H* scaling slopes are ~0.5-1; our H->OH '
                'Theil-Sen slope is %.2f but with R2=%.2f (n=%d, small OH* '
                'tier) -- direction consistent, magnitude above the band and '
                'statistically weak. OH->O slope %.2f (R2=%.2f) matches the '
                'well-established O*-OH* scaling (~1.7-2 directional).'
                % (e1['H->OH']['slope'], e1['H->OH']['r2'], e1['H->OH']['n'],
                   e1['OH->O']['slope'], e1['OH->O']['r2']))},
        'E2_gap_verdict': {
            'data_convention': e2['data_convention'],
            'HER_x_ORR': {k: v for k, v in e2['HER_x_ORR'].items()},
            'HER_x_OER': {k: v for k, v in e2['HER_x_OER'].items()}},
        'E3_ladder': {
            'hypothesis_check': e3['hypothesis_check'],
            'empirical_relation': e3['empirical_relation'],
            'empirical_relation_large_tier': e3['empirical_relation_large_tier'],
            'dimer_large_tier': e3['dimer_large_tier'],
            'note': ('Over all 15 adsorbates MAE does NOT rise with atom '
                     'count (negative result, reported as-is); within the '
                     'large tier the ladder is driven by dissociated dimers '
                     '(2H/2N/2O): MAE ~ 0.19*n_atoms + 0.03 (r=0.69).')},
        'E4_pooled_vs_separate': {
            'verdict': e4['verdict'],
            'n_pooled_wins': e4['n_pooled_wins'],
            'mean_delta_eV': e4['mean_delta_eV']},
        'E5_escape_angle_redflags': redflag}
    with open(os.path.join(HERE, 'scaling_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readme = f"""# Workstream E — 标度关系量化 (plan-hb v3.0)

Scope qualifier applying to ALL conclusions: {e2['scope_qualifier']}.

## E1 标度矩阵 (`e1_scaling_matrix.py`)
Theil-Sen (预注册), 15×15 有序对, 共享 (comp,structure,facet), 210/210 对 n≥30。
产物: `e1_slope/intercept/r2/nshared_matrix.csv`, `e1_scaling_matrix.png`, `e1_key_pairs.json`。
- H→OH: slope={e1['H->OH']['slope']}, intercept={e1['H->OH']['intercept']}, R²={e1['H->OH']['r2']} (n={e1['H->OH']['n']})
- OH→H: slope={e1['OH->H']['slope']}, intercept={e1['OH->H']['intercept']}, R²={e1['OH->H']['r2']}
- H→O: slope={e1['H->O']['slope']}, R²={e1['H->O']['r2']} (n={e1['H->O']['n']})
- 文献对比: OH*/H* 文献斜率约 0.5–1；本数据 H→OH=1.38 略高于区间且 R² 仅 0.07
  （OH* 小档 n=229，统计弱），方向一致、幅度偏弱。OH→O=1.74 (R²=0.81) 与 O*–OH*
  标度关系文献吻合。

## E2 双功能缺口裁决 (`e2_gap_verdict.py`)
口径: **标签口径**（共享组成的实测能量，不用模型预测）；demo 级自由能修正
(ΔG_H=E+0.24, ΔG_OH=E+3.2 eV，同 C-1)。
- HER×ORR 标度线: {e2['HER_x_ORR']['scaling_line']} (R²={e2['HER_x_ORR']['r2']}, n={e2['HER_x_ORR']['n']})
- **d_min = {e2['HER_x_ORR']['d_min_eV']} eV = {e2['HER_x_ORR']['d_min_over_resolution']}× 模型分辨率 0.11 eV**
- 裁决: {e2['HER_x_ORR']['verdict']}
- 最近实测点距理想点 {e2['HER_x_ORR']['nearest_measured_point_dist_eV']} eV（佐证）
- HER×OER: {e2['HER_x_OER']['scaling_line']}, d_min={e2['HER_x_OER']['d_min_eV']} eV = {e2['HER_x_OER']['d_min_over_resolution']}× 分辨率 → 同为物理事实
- 图: `e2_gap.png`（标度线+理想点+d_min 标注+分辨率圆，替代双火山信息量）

## E3 阶梯分析 (`e3_ladder.py`)
- 15 吸附种全体: {e3['hypothesis_check']}（阴性如实：原子数/价电子/多原子均不显著）
- 大档 (n=8): {e3['empirical_relation_large_tier']['formula']} (r={e3['empirical_relation_large_tier']['r']}, p={e3['empirical_relation_large_tier']['p']:.1e})；
  阶梯由解离二聚体 (2H/2N/2O) 驱动，dimer flag Spearman ρ={e3['dimer_large_tier']['spearman_rho']}
- 经验关系式（全体）: {e3['empirical_relation']['formula']}（不显著，仅如实记录）

## E4 池化 GBDT 对照 (`e4_pooled_gbdt.py`)
同折同种子 (GroupKFold(5,42) by comp, seeds 0/1/2, v3 超参)；公平契约 >0.005 eV + 3/3 方向。
- 池化+one-hot 仅在 {e4['n_pooled_wins']}/15 吸附种 (CH2, NH) 满足契约；平均 Δ(sep−pooled) = {e4['mean_delta_eV']} eV
- **裁决: {e4['verdict']}** —— 数据共享红利被指示特征吃掉/被跨吸附种干扰抵消，分吸附种模型保持基准。
- 表: `e4_pooled_vs_separate.csv`

## E5 逃逸角红旗统计
- 最近邻候选 {redflag['n_candidates']} 个中 {redflag['n_flagged']} 个 ({redflag['flagged_fraction']*100:.0f}%) 带 ≥1 红旗
  （Tc 放射性 / Hg,Tl 毒性 / La,Y,Sc 强亲氧 / Mn,Bi,Fe LOEO），规则分布: {redflag['by_rule']}
- OH* 训练集外（模型外推）占比 {1-redflag['oh_in_training_fraction']:.0%}
- **"逃逸角=不可信角"获得双重量化支撑：逃离标度缺口的组成同时是模型不可信
  （外推）+ 材料不稳定（红旗）—— 双重警示，不得单独引用其中任一维度。**

## 复现
`for s in e1_scaling_matrix e2_gap_verdict e3_ladder e4_pooled_gbdt e5_summary; do /usr/local/bin/python3 innovation/scaling/$s.py; done`（幂等覆盖）。
"""
    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write(readme)
    print(json.dumps(redflag, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
