"""Per-adsorbate composition-level XGBoost models (Workstream C-1, task 2).
Same hyperparams as v3: lr=0.03, max_depth=7, n_estimators=400, subsample=0.8.
CV: GroupKFold(5, shuffle=True, random_state=42) grouped by normalized comp.
Seeds (0,1,2) -> OOF MAE mean±std. Idempotent: overwrites outputs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_features.csv')

BIG = ['H', 'S', '2H', 'N', 'O', 'C', '2N', '2O']        # unique comp >= 500
SMALL = ['NH', 'CH', 'SH', 'H2O', 'CH2', 'CH3', 'OH']   # < 500
SEEDS = (0, 1, 2)
DROP = ['comp', 'structure', 'facet', 'adsorbate', 'ads_atom', 'energy_eV', 'n_raw']
PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
              subsample=0.8, n_jobs=-1, tree_method='hist')


def oof_pred(df, seed):
    X = df.drop(columns=DROP).values
    y = df['energy_eV'].values
    groups = df['comp'].astype(str).str.upper().str.strip()
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=seed, **PARAMS)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def main():
    df = pd.read_csv(DATA)
    rows, panels = [], {}
    for ads in BIG + SMALL:
        sub = df[df.adsorbate == ads].reset_index(drop=True)
        maes, oofs = [], []
        for s in SEEDS:
            o = oof_pred(sub, s)
            maes.append(mean_absolute_error(sub.energy_eV, o))
            oofs.append(o)
        tier = 'large' if ads in BIG else 'small'
        warn = '' if tier == 'large' else 'SMALL-SAMPLE WARNING (n<500)'
        rows.append(dict(adsorbate=ads, n=len(sub), tier=tier,
                         mae_mean=np.mean(maes), mae_std=np.std(maes), warning=warn))
        panels[ads] = (sub.energy_eV.values, np.mean(oofs, axis=0))
        print(f"{ads:5s} n={len(sub):5d} MAE={np.mean(maes):.4f}±{np.std(maes):.4f} {warn}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, 'per_adsorbate_mae.csv'), index=False)

    # parity 2x4 for large tier
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    C = '#B35C24'
    for ax, ads in zip(axes.ravel(), BIG):
        y, p = panels[ads]
        ax.scatter(y, p, s=6, alpha=0.4, color=C, edgecolors='none')
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], '--', color='#7A4A2B', lw=1)
        mae = res.loc[res.adsorbate == ads, 'mae_mean'].iloc[0]
        ax.set_title(f"{ads}  MAE={mae:.3f} eV", fontsize=10)
        ax.set_xlabel('DFT energy (eV)', fontsize=8)
        ax.set_ylabel('Predicted (eV)', fontsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('Per-adsorbate XGBoost parity (OOF, seeds 0/1/2 averaged)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'per_adsorbate_parity.png'), dpi=300)
    print('saved per_adsorbate_mae.csv, per_adsorbate_parity.png')


if __name__ == '__main__':
    main()
