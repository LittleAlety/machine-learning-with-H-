"""E3: error-ladder analysis (Workstream E, task 3).

Hypothesis: per-adsorbate OOF MAE rises monotonically with the degree of
localization / chemical complexity of the adsorbate.

Descriptors per adsorbate (parsed from the adsorbate label, n_ads_units,
ads_atom in multiads_dataset.csv):
  n_atoms_total   = atoms in the adsorbed unit(s) (2H -> 2, CH3 -> 4, ...)
  valence_e       = total valence electrons of the adsorbed unit
  multiatom       = 1 if the unit contains >1 element type or >1 heavy atom
                    (H2O, OH, NH, CH, CH2, CH3, SH are multi-element)
Inputs : innovation/multi_adsorbate/per_adsorbate_mae.csv (seeds 0,1,2)
Outputs: e3_ladder.png (scatter + linear trend, 300 dpi),
         e3_ladder.json (correlations + empirical relation).
Idempotent: overwrites outputs.
"""
import os
import re
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
MAE_CSV = os.path.join(ROOT, 'innovation', 'multi_adsorbate',
                       'per_adsorbate_mae.csv')
VALENCE = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'S': 6}
COV = {'H': '#d9a441', 'S': '#8a7b4f', '2H': '#d9a441', 'N': '#DFB27E',
       'O': '#b5522d', 'C': '#7A4A2B', '2N': '#DFB27E', '2O': '#b5522d',
       'NH': '#B35C24', 'CH': '#B35C24', 'SH': '#B35C24', 'H2O': '#B35C24',
       'CH2': '#B35C24', 'CH3': '#B35C24', 'OH': '#B35C24'}


def parse_unit(ads):
    """'2H'->[('H',2)], 'H2O'->[('H',2),('O',1)], 'CH3'->[('C',1),('H',3)]"""
    s = ads
    lead = re.match(r'^(\d+)([A-Z].*)$', s)
    mult, rest = (int(lead.group(1)), lead.group(2)) if lead else (1, s)
    atoms = [(el, int(num) if num else 1)
             for el, num in re.findall(r'([A-Z][a-z]?)(\d*)', rest)]
    return [(el, k * mult) for el, k in atoms]


def main():
    mae = pd.read_csv(MAE_CSV)
    rows = []
    for _, r in mae.iterrows():
        atoms = parse_unit(r.adsorbate)
        n_at = sum(k for _, k in atoms)
        ve = sum(VALENCE[el] * k for el, k in atoms)
        multi = int(len({el for el, _ in atoms}) > 1)
        rows.append(dict(adsorbate=r.adsorbate, mae=r.mae_mean, n=r.n,
                         tier=r.tier, n_atoms=n_at, valence_e=ve,
                         multiatom=multi,
                         dimer=int(bool(re.match(r'^2', r.adsorbate)))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, 'e3_ladder_data.csv'), index=False)

    res = {'descriptors': d.to_dict('records')}
    d['dimer'] = d.adsorbate.str.match(r'^2').astype(int)
    for prop in ('n_atoms', 'valence_e', 'multiatom', 'dimer'):
        sp, spp = spearmanr(d[prop], d.mae)
        pe, pep = pearsonr(d[prop], d.mae)
        res[prop] = dict(spearman_rho=round(float(sp), 3),
                         spearman_p=float('%.2e' % spp),
                         pearson_r=round(float(pe), 3),
                         pearson_p=float('%.2e' % pep))

    # empirical relation: MAE = a*n_atoms + b (all 15 adsorbates)
    lr = linregress(d.n_atoms, d.mae)
    res['empirical_relation'] = dict(
        formula='MAE[eV] = %.4f * n_atoms + %.4f' % (lr.slope, lr.intercept),
        slope=round(float(lr.slope), 4), intercept=round(float(lr.intercept), 4),
        r=round(float(lr.rvalue), 3), p=float('%.2e' % lr.pvalue))
    # large-tier only (small tier has n<500 warning)
    dl = d[d.tier == 'large']
    lr2 = linregress(dl.n_atoms, dl.mae)
    res['empirical_relation_large_tier'] = dict(
        formula='MAE[eV] = %.4f * n_atoms + %.4f' % (lr2.slope, lr2.intercept),
        slope=round(float(lr2.slope), 4), intercept=round(float(lr2.intercept), 4),
        r=round(float(lr2.rvalue), 3), p=float('%.2e' % lr2.pvalue), n=len(dl))
    dl2 = d[d.tier == 'large']
    sp_d, spp_d = spearmanr(dl2.dimer, dl2.mae)
    res['dimer_large_tier'] = dict(spearman_rho=round(float(sp_d), 3),
                                   spearman_p=float('%.2e' % spp_d),
                                   note='dissociated-dimer flag (2H/2N/2O), '
                                        'large tier only (n=8)')
    rho_atoms = res['n_atoms']['spearman_rho']
    res['hypothesis_check'] = (
        'SUPPORTED' if rho_atoms > 0.7 and res['n_atoms']['spearman_p'] < 0.05
        else 'PARTIALLY SUPPORTED' if rho_atoms > 0.4
        else 'NOT SUPPORTED')
    res['hypothesis_check'] += (': Spearman rho(MAE, n_atoms)=%.2f over 15 '
                                'adsorbates' % rho_atoms)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(d.n_atoms[d.multiatom == 0], d.mae[d.multiatom == 0], s=55,
               color='#DFB27E', edgecolors='#7A4A2B', linewidths=0.6,
               label='single-element unit', zorder=3)
    ax.scatter(d.n_atoms[d.multiatom == 1], d.mae[d.multiatom == 1], s=55,
               color='#B35C24', edgecolors='#7A4A2B', linewidths=0.6,
               label='multi-element unit', zorder=3)
    xs = np.linspace(0.5, d.n_atoms.max() + 0.5, 10)
    ax.plot(xs, lr.slope * xs + lr.intercept, color='#b5522d', lw=1.8,
            label='fit: %s (r=%.2f)' % (res['empirical_relation']['formula'],
                                        lr.rvalue))
    offsets = {'2H': (5, 2), 'CH': (5, -10), 'OH': (5, 8), 'NH': (-22, 6),
               'SH': (-22, -8), '2N': (-24, 4), '2O': (5, 4)}
    for _, r in d.iterrows():
        ax.annotate(r.adsorbate, (r.n_atoms, r.mae),
                    textcoords='offset points',
                    xytext=offsets.get(r.adsorbate, (5, 4)), fontsize=8,
                    color='#7A4A2B')
    ax.set_xlabel('Number of atoms in adsorbed unit')
    ax.set_ylabel('Per-adsorbate OOF MAE (eV)')
    ax.set_title('E3 error ladder: MAE vs adsorbate complexity\n'
                 '(Spearman $\\rho$=%.2f, p=%.1e)' % (rho_atoms,
                                                      res['n_atoms']['spearman_p']))
    ax.set_ylim(top=0.68)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    for spn in ('top', 'right'):
        ax.spines[spn].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'e3_ladder.png'), dpi=300)

    with open(os.path.join(HERE, 'e3_ladder.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'descriptors'},
                     indent=2))


if __name__ == '__main__':
    main()
