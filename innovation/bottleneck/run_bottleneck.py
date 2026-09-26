#!/usr/bin/env python
"""
Phase 5 Route 1: physical bottleneck model (comp -> low-dim latent -> E_ads).

Pre-registered rules (config/preregistered_phase5.yaml, route1_bottleneck, LOCKED):
  - candidate bottleneck dims: [1, 2, 3]  (locked, no expansion)
  - two arms (locked):
      (a) linear bottleneck: 84 -> linear projection to k -> linear readout,
          implemented as PLSRegression (locked implementation: PLS,
          StandardScaler -> PLSRegression(n_components=k)).
      (b) MLP bottleneck: 84 -> 64 - 32 -> k -> 32 - 16 -> 1 (ReLU),
          standardization fit on inner training fold only,
          early stopping on inner validation fold (GroupShuffleSplit by comp).
  - evaluation: GroupKFold(5, groups=comp), OOF MAE, 10 seeds mean +/- std.
  - dual-outcome verdict:
      sufficiency: k=2 gap vs v3 OOF 0.1134 < 0.005 eV -> low-dim group_wmean-type
                   pipeline is a sufficient statistic.
      gap: bottleneck loses accuracy significantly -> a second independent
           dimension beyond SHAP single-feature ranking exists.
      on conflict between arms, the MLP arm governs (recorded verbatim).

Idempotent: reruns overwrite bottleneck_results.json / latent_corr.png.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
DIMS = [1, 2, 3]
V3_OOF = 0.1134

df = pd.read_csv(os.path.join(ROOT, 'data_processed/hstar_features_v3_84feat.csv'))
FEATURES = df.columns[5:].tolist()          # 84 features
X = df[FEATURES].to_numpy(float)
y = df['energy_eV'].to_numpy(float)
groups = df['comp'].to_numpy()
print(f'data: {X.shape}, groups={len(np.unique(groups))}')

# ---------------------------------------------------------------- arm (a): linear bottleneck (PLS)
def linear_arm_oof(k, seed):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        sc, pls = StandardScaler(), PLSRegression(n_components=k)
        sc.fit(X[tr]); pls.fit(sc.transform(X[tr]), y[tr])
        oof[te] = pls.predict(sc.transform(X[te])).ravel()
    return mean_absolute_error(y, oof)

# ---------------------------------------------------------------- arm (b): MLP bottleneck
class BottleneckMLP(nn.Module):
    def __init__(self, d_in, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, k))
        self.decoder = nn.Sequential(nn.Linear(k, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, 1))
    def forward(self, x):
        return self.decoder(self.encoder(x)).squeeze(-1)

def train_mlp(Xtr, ytr, Xval, yval, k, seed, max_epochs=3000, patience=100):
    torch.manual_seed(seed)
    xsc, ysc = StandardScaler(), StandardScaler()      # fit on inner train only
    Xtr_s = xsc.fit_transform(Xtr); ytr_s = ysc.fit_transform(ytr.reshape(-1, 1)).ravel()
    Xval_s = xsc.transform(Xval)
    Xt = torch.tensor(Xtr_s, dtype=torch.float32); yt = torch.tensor(ytr_s, dtype=torch.float32)
    Xv = torch.tensor(Xval_s, dtype=torch.float32); yv = torch.tensor(ysc.transform(yval.reshape(-1,1)).ravel(), dtype=torch.float32)
    model = BottleneckMLP(Xtr.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best_val, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xv), yv).item()
        if v < best_val - 1e-6:
            best_val, best_state, wait = v, {kk: vv.clone() for kk, vv in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model, xsc, ysc

def mlp_arm_oof(k, seed, return_latent=False):
    oof = np.full(len(y), np.nan)
    lat = np.full((len(y), k), np.nan) if return_latent else None
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed * 10 + fold)
        itr, ival = next(gss.split(X[tr], y[tr], groups[tr]))
        model, xsc, ysc = train_mlp(X[tr][itr], y[tr][itr], X[tr][ival], y[tr][ival], k, seed)
        model.eval()
        with torch.no_grad():
            Xte = torch.tensor(xsc.transform(X[te]), dtype=torch.float32)
            z = model.encoder(Xte).numpy()
            pred = model.decoder(torch.tensor(z, dtype=torch.float32)).numpy()
        oof[te] = ysc.inverse_transform(pred.reshape(-1, 1)).ravel()
        if return_latent: lat[te] = z
    mae = mean_absolute_error(y, oof)
    return (mae, lat) if return_latent else mae

results = {'config': {'seeds': SEEDS, 'dims': DIMS, 'v3_oof': V3_OOF,
                      'protocol': 'GroupKFold(5, groups=comp)',
                      'linear_impl': 'StandardScaler -> PLSRegression(n_components=k)',
                      'mlp_arch': '84-64-32-k-32-16-1 ReLU, inner-fold standardization, early stopping on inner GroupShuffleSplit val'},
           'linear': {}, 'mlp': {}}

latent_k2 = None
for k in DIMS:
    lin = [linear_arm_oof(k, s) for s in SEEDS]
    results['linear'][k] = {'seed_maes': lin, 'mean': float(np.mean(lin)), 'std': float(np.std(lin)),
                            'gap_vs_v3': float(np.mean(lin) - V3_OOF)}
    print(f'linear k={k}: {np.mean(lin):.4f} +/- {np.std(lin):.4f}  gap={np.mean(lin)-V3_OOF:+.4f}')
    mlp = []
    for s in SEEDS:
        if k == 2:
            m, latent_k2 = mlp_arm_oof(k, s, return_latent=True)
        else:
            m = mlp_arm_oof(k, s)
        mlp.append(m)
    results['mlp'][k] = {'seed_maes': mlp, 'mean': float(np.mean(mlp)), 'std': float(np.std(mlp)),
                         'gap_vs_v3': float(np.mean(mlp) - V3_OOF)}
    print(f'mlp    k={k}: {np.mean(mlp):.4f} +/- {np.std(mlp):.4f}  gap={np.mean(mlp)-V3_OOF:+.4f}')

np.save(os.path.join(OUT, 'latent_k2_seedlast.npy'), latent_k2)
with open(os.path.join(OUT, 'bottleneck_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- verdict (dual-outcome, pre-registered)
lin2 = results['linear']['2']['gap_vs_v3']; mlp2 = results['mlp']['2']['gap_vs_v3']
def verdict_of(gap): return 'sufficiency' if gap < 0.005 else 'gap'
v_lin, v_mlp = verdict_of(lin2), verdict_of(mlp2)
conflict = v_lin != v_mlp
final = v_mlp if conflict else v_lin
verdict = {'linear_arm_k2_gap_eV': lin2, 'mlp_arm_k2_gap_eV': mlp2,
           'linear_arm_verdict': v_lin, 'mlp_arm_verdict': v_mlp,
           'arms_conflict': conflict,
           'final_verdict': final,
           'rule': 'gap < 0.005 eV -> sufficiency; else gap; conflict -> MLP arm governs'}
with open(os.path.join(OUT, 'bottleneck_verdict.json'), 'w') as f:
    json.dump(verdict, f, indent=2)
print('VERDICT:', verdict)

# ---------------------------------------------------------------- latent interpretation (k=2)
INTERP_FEATURES = ['group_wmean', 'en_wmean', 'melting_point_wmean', 'd_el_wmean',
                   'ie1_wmean', 'mendeleev_number_wmean', 'n_valence_wmean',
                   'nd_valence_wmean', 'radius_wmean', 'covalent_radius_wmean',
                   'period_wmean', 'n_unfilled_wmean']
Z = latent_k2
C = np.zeros((2, len(INTERP_FEATURES)))
for i in range(2):
    for j, f in enumerate(INTERP_FEATURES):
        C[i, j] = np.corrcoef(Z[:, i], df[f].to_numpy(float))[0, 1]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 2.6))
im = ax.imshow(np.abs(C), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(INTERP_FEATURES)))
ax.set_xticklabels(INTERP_FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(['Latent dim 1', 'Latent dim 2'], fontsize=10)
for i in range(2):
    for j in range(len(INTERP_FEATURES)):
        ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=8,
                color='black' if C[i, j] < 0.6 else 'white')
ax.set_title('|Pearson r| between k=2 MLP bottleneck latents and top SHAP-family features (OOF, seed 31337)',
             fontsize=10)
fig.colorbar(im, ax=ax, label='|r|', shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'latent_corr.png'), dpi=300)
print('saved latent_corr.png')
for i in range(2):
    order = np.argsort(-np.abs(C[i]))[:5]
    print(f'latent dim {i+1} top corr:', [(INTERP_FEATURES[j], round(C[i, j], 3)) for j in order])
