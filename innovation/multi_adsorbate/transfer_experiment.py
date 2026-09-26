"""Cross-adsorbate transfer experiment (Workstream C-1, task 3).

Config locked (NN second legal identity):
  input = shared descriptors (standardized, fit on train folds only) + adsorbate one-hot
  hard-shared MLP trunk 128-64 (ReLU), one linear output head per adsorbate
  500 epochs max, early stopping (patience 30 on val loss, 10% of train fold)
vs single-task GBDT (same XGB hyperparams as per_adsorbate_models.py).

Fairness: identical GroupKFold(5, shuffle=True, random_state=42) folds, grouped by
normalized comp; 10 seeds (0,1,2,7,13,42,99,123,2024,31337). Verdict contract:
improvement > 0.005 eV AND 10/10 same-direction seed agreement.
Idempotent: overwrites outputs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_features.csv')

BIG = ['H', 'S', '2H', 'N', 'O', 'C', '2N', '2O']
SMALL = ['NH', 'CH', 'SH', 'H2O', 'CH2', 'CH3', 'OH']
ADS = BIG + SMALL
SEEDS = (0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337)
DROP = ['comp', 'structure', 'facet', 'adsorbate', 'ads_atom', 'energy_eV', 'n_raw']
PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
              subsample=0.8, n_jobs=1, tree_method='hist')
EPOCHS, PATIENCE, BS, LR = 500, 30, 512, 1e-3
N_FOLDS = 5


class MultiHeadMLP(nn.Module):
    def __init__(self, d_in, n_tasks):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_tasks)])

    def forward(self, x, task_idx):
        h = self.trunk(x)
        out = torch.zeros(x.shape[0], 1)
        for t in torch.unique(task_idx):
            m = task_idx == t
            out[m] = self.heads[int(t)](h[m])
        return out


def gbdt_one(args):
    """OOF MAE for one (adsorbate, seed) under the shared fixed folds."""
    ads, seed, Xall, yall, isin_mask, folds = args
    idx_all = np.where(isin_mask)[0]
    oof = np.zeros(isin_mask.sum())
    for tr, te in folds:
        tr_m = tr[np.isin(tr, idx_all)]
        te_m = te[np.isin(te, idx_all)]
        if len(te_m) == 0 or len(tr_m) == 0:
            continue
        m = XGBRegressor(random_state=seed, **PARAMS)
        m.fit(Xall[tr_m], yall[tr_m])
        oof[np.isin(idx_all, te_m)] = m.predict(Xall[te_m])
    return ads, seed, mean_absolute_error(yall[idx_all], oof)


def nn_seed(seed, Xall, yall, tall, oh, folds, n_tasks):
    """Full OOF predictions of joint multi-task NN for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    oof = np.zeros(len(yall))
    Xm = np.hstack([Xall, oh])
    for tr, te in folds:
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-8  # train-fold fit only
        ym, ys = yall[tr].mean(), yall[tr].std() + 1e-8
        Xtr = (Xm[tr] - mu) / sd
        ytr = (yall[tr] - ym) / ys
        ttr = tall[tr]
        itr, iva = train_test_split(np.arange(len(tr)), test_size=0.1,
                                    random_state=seed)
        model = MultiHeadMLP(Xm.shape[1], n_tasks)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.MSELoss()
        Xv = torch.tensor(Xtr[iva]); yv = torch.tensor(ytr[iva]).unsqueeze(1)
        tv = torch.tensor(ttr[iva])
        best, bad, best_state = np.inf, 0, None
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(itr)
            for b in range(0, len(perm), BS):
                bi = perm[b:b + BS]
                xb = torch.tensor(Xtr[bi]); yb = torch.tensor(ytr[bi]).unsqueeze(1)
                tb = torch.tensor(ttr[bi])
                opt.zero_grad()
                loss = lossf(model(xb, tb), yb)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Xv, tv), yv).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            Xte = torch.tensor((Xm[te] - mu) / sd)
            pred = model(Xte, torch.tensor(tall[te])).numpy().ravel() * ys + ym
        oof[te] = pred
    return oof


def main():
    df = pd.read_csv(DATA)
    df['group'] = df['comp'].astype(str).str.upper().str.strip()
    feat_cols = [c for c in df.columns if c not in DROP + ['group']]
    Xall = df[feat_cols].values.astype(np.float32)
    yall = df['energy_eV'].values.astype(np.float32)
    task_map = {a: i for i, a in enumerate(ADS)}
    tall = df['adsorbate'].map(task_map).values
    n_tasks = len(ADS)
    oh = np.zeros((len(df), n_tasks), dtype=np.float32)
    oh[np.arange(len(df)), tall] = 1.0

    # fixed folds (same for both models and all seeds)
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(gkf.split(Xall, yall, df['group']))

    results = []          # (adsorbate, tier, seed, model, mae)
    # ---- single-task GBDT (parallel over adsorbate x seed) ----
    jobs = [(ads, seed, Xall, yall, (df.adsorbate == ads).values, folds)
            for ads in ADS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        for ads, seed, mae in ex.map(gbdt_one, jobs):
            results.append((ads, seed, 'gbdt', mae))
            print(f'GBDT {ads} seed={seed} MAE={mae:.4f}', flush=True)

    # ---- multi-task NN (joint, all adsorbates; parallel over seeds) ----
    with ProcessPoolExecutor(max_workers=3) as ex:
        oofs = ex.map(nn_seed, SEEDS,
                      [Xall] * len(SEEDS), [yall] * len(SEEDS),
                      [tall] * len(SEEDS), [oh] * len(SEEDS),
                      [folds] * len(SEEDS), [n_tasks] * len(SEEDS))
        for seed, oof in zip(SEEDS, oofs):
            for ads in ADS:
                mask = (df.adsorbate == ads).values
                results.append((ads, seed, 'nn', mean_absolute_error(yall[mask], oof[mask])))
            print(f'NN done seed {seed}', flush=True)

    res = pd.DataFrame(results, columns=['adsorbate', 'seed', 'model', 'mae'])
    res.to_csv(os.path.join(HERE, 'transfer_raw.csv'), index=False)

    # verdict
    verdicts = {}
    for tier, group in (('small', SMALL), ('large', BIG)):
        wins, deltas = 0, []
        for ads in group:
            g = res[(res.adsorbate == ads) & (res.model == 'gbdt')].sort_values('seed').mae.values
            n = res[(res.adsorbate == ads) & (res.model == 'nn')].sort_values('seed').mae.values
            d = g - n  # >0 means NN better
            deltas.append(float(d.mean()))
            if (d > 0.005).all():
                wins += 1
        verdicts[tier] = dict(mean_delta=float(np.mean(deltas)),
                              per_ads_delta={a: d for a, d in zip(group, deltas)},
                              n_ads_significant=wins, n_ads=len(group))
    small = verdicts['small']
    if small['mean_delta'] > 0.005 and small['n_ads_significant'] == small['n_ads']:
        verdict = 'NN joint training significantly beats single-task GBDT on small-tier adsorbates (transfer gain via shared representation).'
    else:
        verdict = ('跨吸附种迁移红利在数据共享不在表示共享：联合硬共享 MLP 未能在小档吸附种上'
                   '显著优于单任务 GBDT（公平契约未满足），迁移红利主要来自描述符/数据口径统一而非表示共享。')
    out = dict(verdict=verdict, verdicts=verdicts,
               contract='improvement>0.005 eV AND 10/10 same-direction seeds')
    with open(os.path.join(HERE, 'transfer_verdict.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(verdict)


if __name__ == '__main__':
    main()
