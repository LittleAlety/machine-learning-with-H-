"""E4: pooled GBDT vs per-adsorbate GBDT (Workstream E, task 4).

Question: does pooling all 15 adsorbates with a one-hot adsorbate indicator
beat per-adsorbate models, i.e. is the data-sharing benefit eaten by the
indicator feature?

Protocol (identical for both arms):
  Features : 45-dim multiads_features (pooled arm adds adsorbate one-hot;
             per-adsorbate arm drops constant identity columns as in C-1)
  Folds    : GroupKFold(5, shuffle=True, random_state=42) grouped by
             normalized comp -> a composition never crosses folds, even
             across adsorbates
  Seeds    : (0, 1, 2) -- same as Workstream C-1 per_adsorbate_models.py
  Model    : XGBRegressor, v3 hyperparams (lr=0.03, depth=7, n=400, sub=0.8)
Fair contract: pooled wins only if mean OOF-MAE improvement > 0.005 eV AND
the improvement direction is consistent across 3/3 seeds for that adsorbate.
Outputs: e4_pooled_vs_separate.csv, e4_verdict.json. Idempotent.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_features.csv')
DROP = ['comp', 'structure', 'facet', 'adsorbate', 'ads_atom', 'energy_eV',
        'n_raw']
PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
              subsample=0.8, n_jobs=-1, tree_method='hist')
SEEDS = (0, 1, 2)
FAIR_DELTA = 0.005  # eV


def oof(X, y, groups, seed):
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros_like(y, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=seed, **PARAMS)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    df = pd.read_csv(DATA)
    groups_all = df['comp'].astype(str).str.upper().str.strip()
    ads_list = list(df.adsorbate.unique())

    # pooled design matrix: numeric features + adsorbate one-hot
    Xp = pd.concat([df.drop(columns=DROP, errors='ignore'),
                    pd.get_dummies(df.adsorbate, prefix='ads')], axis=1)
    Xp = Xp.astype(float).values
    y_all = df.energy_eV.values

    pooled_mae = {a: [] for a in ads_list}
    sep_mae = {a: [] for a in ads_list}
    for s in SEEDS:
        pred_pool = oof(Xp, y_all, groups_all, s)
        df['_pred_pool'] = pred_pool
        for a in ads_list:
            mask = (df.adsorbate == a).values
            pooled_mae[a].append(mean_absolute_error(y_all[mask],
                                                     pred_pool[mask]))
            sub = df[df.adsorbate == a]
            Xs = sub.drop(columns=DROP + ['_pred_pool'], errors='ignore') \
                    .values
            pred_sep = oof(Xs, sub.energy_eV.values,
                           sub['comp'].astype(str).str.upper().str.strip(), s)
            sep_mae[a].append(mean_absolute_error(sub.energy_eV, pred_sep))
        print('seed %d done' % s, flush=True)
    df.drop(columns=['_pred_pool'], inplace=True)

    rows = []
    for a in ads_list:
        pm, sm = np.mean(pooled_mae[a]), np.mean(sep_mae[a])
        delta = sm - pm  # >0 means pooled better
        consistent = int(np.all(np.array(sep_mae[a]) - np.array(pooled_mae[a])
                                > 0)) or int(np.all(np.array(sep_mae[a])
                                                    - np.array(pooled_mae[a]) < 0))
        wins = delta > FAIR_DELTA and np.all(
            np.array(sep_mae[a]) - np.array(pooled_mae[a]) > 0)
        rows.append(dict(adsorbate=a, n=int((df.adsorbate == a).sum()),
                         separate_mae=round(sm, 4),
                         pooled_mae=round(pm, 4),
                         delta_sep_minus_pooled=round(delta, 4),
                         direction_consistent_3of3=bool(consistent),
                         pooled_wins_fair=bool(wins)))
    res = pd.DataFrame(rows).sort_values('delta_sep_minus_pooled',
                                         ascending=False)
    res.to_csv(os.path.join(HERE, 'e4_pooled_vs_separate.csv'), index=False)

    n_win = int(res.pooled_wins_fair.sum())
    mean_delta = float(res.delta_sep_minus_pooled.mean())
    verdict = ('POOLED WINS' if n_win >= 12 else
               'SEPARATE WINS' if (res.delta_sep_minus_pooled < -FAIR_DELTA).sum() >= 12
               else 'NO CLEAR WINNER')
    vtxt = ('%s: pooled-with-one-hot wins fair contract on %d/15 adsorbates; '
            'mean delta(sep-pooled)=%.4f eV. ') % (verdict, n_win, mean_delta)
    if verdict != 'POOLED WINS':
        vtxt += ('The data-sharing benefit is largely absorbed by the '
                 'adsorbate indicator feature (or offset by cross-adsorbate '
                 'interference); per-adsorbate models remain the reference.')
    out = dict(protocol=dict(folds='GroupKFold(5,shuffle=True,random_state=42) '
                             'by comp; seeds (0,1,2); v3 XGB hyperparams'),
               fair_contract='improvement>0.005 eV AND 3/3 seed direction',
               n_pooled_wins=n_win, mean_delta_eV=round(mean_delta, 4),
               verdict=vtxt, per_adsorbate=res.to_dict('records'))
    with open(os.path.join(HERE, 'e4_verdict.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(vtxt)
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
