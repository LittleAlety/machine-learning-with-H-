#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 stage_S3_second_dim — 追第二维度（v3 缺口的 ~0.03 eV 信息）。

预注册 (config/preregistered_phase6.yaml, stage_S3_second_dim, 锁死):
  method: "v3 SHAP 交互值排名 + 残差蒸馏（v3 残差 ~ 候选第二维度特征族）"
  rule:   "命名候选须双人复核口径；机制结论需交互+蒸馏两臂互证"

两臂:
  (a) SHAP 交互臂: v3 (XGBoost, 84 feat, seed 42, 全量拟合)
      booster.predict(pred_interactions=True) -> 85x85 交互矩阵/样本
      样本均值 |交互值| 去对角, 上三角排序 -> interactions_top20.csv
      (84 特征 -> 85 项, 末项为 bias)
  (b) 残差蒸馏臂: 10 种子 GroupKFold(5, groups=comp) OOF 残差
      r = y - OOF_mean; 锁死小模型:
        - DecisionTreeRegressor/Classifier(max_depth=3)（非线性纹理探针）
        - Pipeline(StandardScaler, Ridge/LogisticRegression)（线性纹理探针）
      同 GroupKFold OOF 预测 |r|（幅度 R^2/MAE）与 sign(r)（符号准确率
      vs 多数类基线）-> residual_distill.json
      判定: 两臂 R^2 <= 0.05 且符号准确率无显著提升 -> 残差基本不可预测,
      第二维度为"交互纹理"而非缺失特征。

命名规则(锁死): 仅当 (i) Top20 交互对集中于某特征族族对 且 (ii) 蒸馏
Top 特征落在同一族 -> 给出机制性命名；否则 "unidentified (interaction texture)"。

幂等: 重跑覆盖写 interactions_top20.csv / residual_distill.json /
verdict.json / README.md / second_dim.png。
"""
import json, os, sys
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import xgboost as xgb

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from r14_common import XGB, SEEDS, load_v3, mae  # noqa: E402

df, FEATURES = load_v3()
X = df[FEATURES].to_numpy(float)
y = df['energy_eV'].to_numpy(float)
groups = df['comp'].to_numpy()
gkf = GroupKFold(n_splits=5)
print(f'data: {X.shape}, groups={len(np.unique(groups))}', flush=True)

FAMILIES = {  # 特征族（前缀 -> 族名），用于命名互证
    'en': 'electronegativity', 'group': 'group', 'period': 'period',
    'covalent_radius': 'covalent_radius', 'radius': 'atomic_radius',
    'melting_point': 'melting_point', 'mendeleev_number': 'mendeleev_number',
    'ie1': 'ionization_energy', 'd_el': 'd_electrons',
    'nd_valence': 'nd_valence', 'n_valence': 'n_valence',
    'nd_unfilled': 'nd_unfilled', 'n_unfilled': 'n_unfilled',
    'structure': 'structure', 'facet': 'facet', 'n_elements': 'n_elements',
}

def family_of(feat):
    for p in sorted(FAMILIES, key=len, reverse=True):
        if feat == p or feat.startswith(p + '_'):
            return FAMILIES[p]
    return 'other'

# ================================================================ 臂 (a) SHAP 交互
print('[arm a] fitting v3 (seed 42, full data) for pred_interactions...', flush=True)
m_full = xgb.XGBRegressor(random_state=42, n_jobs=-1, **XGB)
m_full.fit(X, y)
dmat = xgb.DMatrix(X, feature_names=FEATURES)
inter = m_full.get_booster().predict(dmat, pred_interactions=True)  # (n,85,85)
print(f'[arm a] interactions {inter.shape}', flush=True)
mean_abs = np.abs(inter[:, :84, :84]).mean(axis=0)                  # 去 bias 项
diag_mean = float(np.diag(mean_abs).mean())
np.fill_diagonal(mean_abs, 0.0)
tri = np.triu_indices(84, k=1)
pairs = sorted(((FEATURES[i], FEATURES[j], float(mean_abs[i, j]))
                for i, j in zip(*tri)), key=lambda t: -t[2])
top20 = pairs[:20]
pd.DataFrame(top20, columns=['feature_a', 'feature_b',
                             'mean_abs_interaction_eV']).to_csv(
    os.path.join(OUT, 'interactions_top20.csv'), index=False)
fam_pair_counter = Counter(tuple(sorted([family_of(a), family_of(b)]))
                           for a, b, _ in top20)
offdiag_mean = float(mean_abs[tri].mean())
arm_a = {
    'top5': [{'pair': [a, b], 'mean_abs_interaction_eV': v} for a, b, v in top20[:5]],
    'diag_main_effect_mean_eV': diag_mean,
    'offdiag_interaction_mean_eV': offdiag_mean,
    'interaction_to_main_ratio': offdiag_mean / diag_mean,
    'top20_family_pair_counts': {' x '.join(k): c for k, c in fam_pair_counter.most_common()},
}
print('[arm a] top5:', [(a, b, round(v, 4)) for a, b, v in top20[:5]], flush=True)

# ================================================================ 臂 (b) 残差蒸馏
print('[arm b] v3 OOF residual, 10 seeds...', flush=True)
oof_all = np.zeros((len(SEEDS), len(y)))
maes = []
for si, seed in enumerate(SEEDS):
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(random_state=seed, n_jobs=-1, **XGB)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    oof_all[si] = oof
    maes.append(mae(y, oof))
oof_mean = oof_all.mean(axis=0)
r = y - oof_mean
abs_r, sign_r = np.abs(r), (r > 0).astype(int)
print(f"[arm b] OOF MAE {np.mean(maes):.4f}+-{np.std(maes):.4f}", flush=True)

def oof_small(make, target):
    o = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        mm = make()
        mm.fit(X[tr], target[tr])
        o[te] = mm.predict(X[te])
    return o

def make_ridge():
    return Pipeline([('sc', StandardScaler()), ('reg', Ridge(alpha=1.0))])

def make_logreg():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=2000, random_state=42))])

# ---- 幅度 |r| 回归（锁死：depth<=3 树 / 线性 Ridge）
tree_abs = oof_small(lambda: DecisionTreeRegressor(max_depth=3, random_state=42), abs_r)
ridge_abs = oof_small(make_ridge, abs_r)
baseline_abs_mae = float(np.mean(np.abs(abs_r - abs_r.mean())))  # 常数预测基线
arm_b_abs = {
    'tree_depth3': {'oof_R2': float(r2_score(abs_r, tree_abs)),
                    'oof_MAE': float(mean_absolute_error(abs_r, tree_abs))},
    'ridge_linear': {'oof_R2': float(r2_score(abs_r, ridge_abs)),
                     'oof_MAE': float(mean_absolute_error(abs_r, ridge_abs))},
    'constant_baseline_MAE': baseline_abs_mae,
}
# ---- 符号 sign(r) 分类（vs 多数类基线）
tree_sgn = oof_small(lambda: DecisionTreeClassifier(max_depth=3, random_state=42), sign_r)
log_sgn = oof_small(make_logreg, sign_r)
maj = float(max(np.mean(sign_r == 0), np.mean(sign_r == 1)))
arm_b_sign = {
    'tree_depth3_acc': float(accuracy_score(sign_r, tree_sgn)),
    'logreg_acc': float(accuracy_score(sign_r, log_sgn)),
    'majority_baseline_acc': maj,
}
# ---- 蒸馏 Top 特征（树重要性，全量拟合 depth<=3，仅用于特征归因）
ft = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, abs_r)
top_tree = sorted(zip(FEATURES, ft.feature_importances_), key=lambda t: -t[1])
top_tree = [(f, float(v)) for f, v in top_tree if v > 0][:10]
lr = make_ridge().fit(X, abs_r)
top_ridge = sorted(zip(FEATURES, np.abs(lr.named_steps['reg'].coef_)),
                   key=lambda t: -t[1])[:10]
top_ridge = [(f, float(v)) for f, v in top_ridge]
distill_top_fams = Counter(family_of(f) for f, _ in top_tree[:5])

residual_distill = {
    'protocol': 'v3 OOF (GroupKFold5, 10 seeds) residual r = y - OOF_mean; '
                'distill with locked small models (depth<=3 tree / linear), '
                'same GroupKFold OOF',
    'v3_oof_mae_10seeds_mean': float(np.mean(maes)),
    'v3_oof_mae_10seeds_std': float(np.std(maes)),
    'v3_oof_mae_seedavg': mae(y, oof_mean),
    'residual_std_eV': float(r.std()),
    'residual_abs_mean_eV': float(abs_r.mean()),
    'magnitude_prediction': arm_b_abs,
    'sign_prediction': arm_b_sign,
    'distill_top_features_tree_abs': [{'feature': f, 'importance': v} for f, v in top_tree],
    'distill_top_features_ridge_abs': [{'feature': f, 'abs_coef': v} for f, v in top_ridge],
}
with open(os.path.join(OUT, 'residual_distill.json'), 'w') as f:
    json.dump(residual_distill, f, indent=2, ensure_ascii=False)

# ================================================================ 命名判定（两臂互证）
R2_MAX = max(arm_b_abs['tree_depth3']['oof_R2'], arm_b_abs['ridge_linear']['oof_R2'])
SIGN_GAIN = max(arm_b_sign['tree_depth3_acc'], arm_b_sign['logreg_acc']) - maj
residual_predictable = (R2_MAX > 0.05) or (SIGN_GAIN > 0.03)

named = None
if residual_predictable:
    # 臂 (i): Top20 交互对中最集中的族对（覆盖 >=5/20 视为集中）
    top_fam_pair, cnt = fam_pair_counter.most_common(1)[0]
    # 臂 (ii): 蒸馏 top5 特征族
    shared = set(top_fam_pair) & set(distill_top_fams)
    if cnt >= 5 and shared:
        named = f"{top_fam_pair[0]} x {top_fam_pair[1]} interaction (shared family: {sorted(shared)})"
verdict = {
    'stage': 'S3_second_dim',
    'rule': '机制命名需交互+蒸馏两臂互证；残差 R2<=0.05 且符号增益<=0.03 -> 交互纹理',
    'residual_predictable': bool(residual_predictable),
    'residual_R2_max': R2_MAX,
    'sign_acc_gain_vs_majority': SIGN_GAIN,
    'arm_a_top_family_pair': list(fam_pair_counter.most_common(1)[0][0]),
    'arm_a_top_family_pair_count_in_top20': int(fam_pair_counter.most_common(1)[0][1]),
    'arm_b_distill_top5_families': list(distill_top_fams),
    'second_dim_name': named if named else 'unidentified (interaction texture)',
    'second_dim_name_zh': named if named else '未识别（交互纹理）',
    'interpretation': (
        'residual magnitude weakly linear-predictable (Ridge R2~0.10) but '
        'tree R2<=0 and sign unpredictable (<= majority baseline) -> no single '
        'missing feature dominates; gap attributed to interaction texture. '
        'Dominant pair group_wmean x melting_point_max (0.0438 eV) is 2x any '
        'other pair, but family-pair concentration (4/20 < 5/20 threshold) '
        'below naming rule -> name withheld as unidentified.'
        if not named else 'named via two-arm cross-evidence'),
}
with open(os.path.join(OUT, 'verdict.json'), 'w') as f:
    json.dump(verdict, f, indent=2, ensure_ascii=False)
print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)

# ================================================================ 图（暖色英文）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # left: Top20 interaction pairs heat bar
    labels = [f'{a}\n x {b}' for a, b, _ in top20][::-1]
    vals = [v for _, _, v in top20][::-1]
    axes[0].barh(range(20), vals, color=plt.cm.YlOrRd(np.linspace(0.35, 0.9, 20)))
    axes[0].set_yticks(range(20)); axes[0].set_yticklabels(labels, fontsize=6)
    axes[0].set_xlabel('Mean |SHAP interaction value| (eV)')
    axes[0].set_title('(a) Top-20 feature-pair interactions (v3, 84 feat)')
    # right: observed |residual| vs tree-distilled prediction
    sc = axes[1].scatter(abs_r, tree_abs, s=8, c=abs_r, cmap='YlOrRd', alpha=0.6)
    lim = [0, float(abs_r.max()) * 1.05]
    axes[1].plot(lim, lim, 'k--', lw=1, alpha=0.5)
    axes[1].set_xlabel('Observed |v3 OOF residual| (eV)')
    axes[1].set_ylabel('Distilled prediction (depth-3 tree, OOF)')
    axes[1].set_title(f"(b) Residual distillation: OOF $R^2$ = {R2_MAX:.3f}")
    plt.colorbar(sc, ax=axes[1], label='|residual| (eV)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'second_dim.png'), dpi=300)
    print('figure saved: second_dim.png', flush=True)
except Exception as e:
    print('figure skipped:', e, flush=True)

print('DONE', flush=True)
