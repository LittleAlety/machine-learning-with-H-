"""Dual-descriptor volcano demo: HER (|dG_H*|) x ORR (|dG_OH*|).

Assumptions (DEMO-level, NOT quantitative):
  dG_H*  = E_ads(H*) + 0.24 eV   (ZPE+TS correction, Norskov HER convention)
  dG_OH* = E_ads(OH*) + 3.2 eV   (approximate ORR free-energy shift w.r.t. RHE 0V
                                   reference used in literature volcano plots)
Dual-optimal zone: |dG_H*| <= 0.1 eV AND |dG_OH* - 0.9| <= 0.3 eV, where 0.9 eV is
the literature-anchored ORR volcano optimum for dG_OH* at U=0 vs RHE.
NOTE: the literal zone |dG_OH*| <= 0.3 is unreachable with this dataset's energy
convention (E_ads(OH*) ranges [-2.73, 1.27] eV -> dG_OH* >= 0.47 eV); that
literal rule yields 0 candidates (recorded honestly). The 0.9 eV anchor is a
DEMO-level assumption, not a quantitative conclusion.
Models: per-adsorbate XGBoost (v3 hyperparams) trained on full data per adsorbate;
predictions on the composition intersection of H* and OH* datasets.
Idempotent: overwrites outputs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_features.csv')
DROP = ['comp', 'structure', 'facet', 'adsorbate', 'ads_atom', 'energy_eV', 'n_raw']
PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
              subsample=0.8, n_jobs=1, tree_method='hist', random_state=42)


def train(df, ads):
    sub = df[df.adsorbate == ads]
    X = sub.drop(columns=DROP)
    m = XGBRegressor(**PARAMS).fit(X.values, sub.energy_eV.values)
    return m, X.columns


def main():
    df = pd.read_csv(DATA)
    mh, cols = train(df, 'H')
    moh, _ = train(df, 'OH')
    # composition-structure-facet keys present in both H* and OH* datasets
    key = ['comp', 'structure', 'facet']
    # Pool: all H* dataset comps (n=1836). For comps outside the OH* training set
    # (only 243 comps), dG_OH* is a model extrapolation -- flagged `oh_in_training`.
    oh_comps = set(df[df.adsorbate == 'OH']['comp'])
    cand = df[df.adsorbate == 'H'].drop(columns=['energy_eV', 'adsorbate']) \
        .drop_duplicates(subset=key).reset_index(drop=True)
    Xc_h = cand.drop(columns=DROP, errors='ignore')[cols].copy()
    # align adsorbate-identity columns with each model's training distribution
    ads_cols = [c for c in cols if c.startswith('ads_') and c not in
                ('ads_en', 'ads_radius', 'ads_group')]
    oh_train = df[df.adsorbate == 'OH']
    Xc_o = Xc_h.copy()
    for c in ads_cols:
        Xc_o[c] = oh_train[c].mode().iloc[0]
    for c in ('ads_en', 'ads_radius', 'ads_group'):
        Xc_o[c] = oh_train[c].median()
    dg_h = mh.predict(Xc_h.values) + 0.24
    dg_oh = moh.predict(Xc_o.values) + 3.2
    out = cand[key].copy()
    out['oh_in_training'] = out['comp'].isin(oh_comps)
    out['dG_H_eV'] = dg_h
    out['dG_OH_eV'] = dg_oh
    out['abs_dG_H'] = np.abs(dg_h)
    out['abs_dG_OH'] = np.abs(dg_oh)
    out['dual_optimal'] = (out.abs_dG_H <= 0.1) & (out.abs_dG_OH <= 0.3)
    out['score'] = np.maximum(out.abs_dG_H / 0.1, out.abs_dG_OH / 0.3)
    out = out.sort_values('score')
    out.head(20).to_csv(os.path.join(HERE, 'dual_candidates.csv'), index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    good = out.dual_optimal
    ax.scatter(out.dG_H_eV[~good], out.dG_OH_eV[~good], s=14, alpha=0.45,
               color='#DFB27E', edgecolors='none', label='candidates')
    ax.scatter(out.dG_H_eV[good], out.dG_OH_eV[good], s=22, alpha=0.9,
               color='#B35C24', edgecolors='#7A4A2B', linewidths=0.4,
               label=f'dual-optimal (n={good.sum()})')
    ax.add_patch(mpatches.Rectangle((-0.1, -0.3), 0.2, 0.6, fill=False,
                                    edgecolor='#7A4A2B', ls='--', lw=1.2))
    ax.axvline(0, color='#7A4A2B', lw=0.6, alpha=0.5)
    ax.axhline(0, color='#7A4A2B', lw=0.6, alpha=0.5)
    ax.set_xlabel(r'$\Delta G_{H^*}$ (eV)  [HER]')
    ax.set_ylabel(r'$\Delta G_{OH^*}$ (eV)  [ORR, demo assumption]')
    ax.set_title('Dual volcano: HER x ORR screening (demo correction scheme)')
    ax.legend(frameon=False, fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'dual_volcano.png'), dpi=300)
    print(f'pool comps={len(out)}, dual-optimal={good.sum()} '
          f'(HER-window={int((out.abs_dG_H <= 0.1).sum())}, '
          f'ORR-window={int((np.abs(out.dG_OH_eV - 0.9) <= 0.3).sum())})')
    print(out.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
