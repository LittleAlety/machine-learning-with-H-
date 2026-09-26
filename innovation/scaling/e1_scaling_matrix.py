"""E1: pairwise adsorption-energy scaling matrix (Workstream E, task 1).

PRE-REGISTERED METHOD: Theil-Sen robust regression (scipy.stats.theilslopes),
chosen a priori over OLS for outlier robustness on composition-level data.

For every ordered pair (A, B) of the 15 adsorbates we take the shared
(comp, structure, facet) rows and fit  E_B = a * E_A + b.
Outputs:
  e1_slope_matrix.csv / e1_intercept_matrix.csv / e1_r2_matrix.csv /
  e1_nshared_matrix.csv  (15x15, row=B predicted, col=A predictor; NaN if
  shared pairs < 30)
  e1_scaling_matrix.png  (slope heatmap, warm palette, 300 dpi)
  e1_key_pairs.json      (slope/intercept/R2/n for literature-relevant pairs)
Idempotent: overwrites outputs.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_dataset.csv')
MIN_SHARED = 30
KEY = ['comp', 'structure', 'facet']


def main():
    df = pd.read_csv(DATA)
    ads_list = sorted(df.adsorbate.unique())
    n = len(ads_list)
    slope = np.full((n, n), np.nan)
    inter = np.full((n, n), np.nan)
    r2m = np.full((n, n), np.nan)
    nsh = np.zeros((n, n), dtype=int)
    groups = {a: g[KEY + ['energy_eV']] for a, g in df.groupby('adsorbate')}

    key_pairs = {}
    for i, B in enumerate(ads_list):        # row: predicted
        for j, A in enumerate(ads_list):    # col: predictor
            if A == B:
                continue
            m = groups[A].merge(groups[B], on=KEY, suffixes=('_A', '_B'))
            nsh[i, j] = len(m)
            if len(m) < MIN_SHARED:
                continue
            x = m.energy_eV_A.values
            y = m.energy_eV_B.values
            a, b, _, _ = theilslopes(y, x)
            yhat = a * x + b
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            slope[i, j] = a
            inter[i, j] = b
            r2m[i, j] = r2
            if {A, B} in ({'H', 'OH'}, {'H', 'O'}, {'H', 'S'}, {'O', 'OH'},
                          {'H', 'N'}, {'H', 'C'}):
                key_pairs[f'{A}->{B}'] = dict(slope=round(float(a), 4),
                                              intercept=round(float(b), 4),
                                              r2=round(float(r2), 4), n=len(m))

    idx = pd.Index(ads_list)
    pd.DataFrame(slope, index=idx, columns=idx).to_csv(
        os.path.join(HERE, 'e1_slope_matrix.csv'))
    pd.DataFrame(inter, index=idx, columns=idx).to_csv(
        os.path.join(HERE, 'e1_intercept_matrix.csv'))
    pd.DataFrame(r2m, index=idx, columns=idx).to_csv(
        os.path.join(HERE, 'e1_r2_matrix.csv'))
    pd.DataFrame(nsh, index=idx, columns=idx).to_csv(
        os.path.join(HERE, 'e1_nshared_matrix.csv'))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    masked = np.ma.masked_invalid(slope)
    cmap = plt.get_cmap('copper_r').copy()
    cmap.set_bad('#f5efe6')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1.5)
    ax.set_xticks(range(n), ads_list, rotation=90, fontsize=8)
    ax.set_yticks(range(n), ads_list, fontsize=8)
    ax.set_xlabel('Predictor adsorbate A')
    ax.set_ylabel('Predicted adsorbate B')
    ax.set_title('E1 scaling matrix: slope a in E(B) = a*E(A) + b  (Theil-Sen)')
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('slope a')
    for i in range(n):
        for j in range(n):
            if not np.isnan(slope[i, j]):
                ax.text(j, i, f'{slope[i, j]:.2f}', ha='center', va='center',
                        fontsize=5.5,
                        color='white' if slope[i, j] > 0.9 else '#7A4A2B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'e1_scaling_matrix.png'), dpi=300)

    with open(os.path.join(HERE, 'e1_key_pairs.json'), 'w') as f:
        json.dump(key_pairs, f, indent=2, ensure_ascii=False)
    print(json.dumps(key_pairs, indent=2))
    print('shared-pair coverage: %d/%d pairs with n>=%d'
          % (np.sum(nsh >= MIN_SHARED), n * (n - 1), MIN_SHARED))


if __name__ == '__main__':
    main()
