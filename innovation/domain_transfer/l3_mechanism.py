# -*- coding: utf-8 -*-
"""
Phase 4 Workstream L — L3 机制分析（预注册: config/preregistered_phase4.yaml
workstream_L.L3_mechanism）。

两部分：
(1) B4 域指示 GBDT（全量三域池化 + 2 域哑变量，锁死出厂 tuned 参数）上计算
    SHAP 交互值（xgboost pred_contribs_interactions），量化
    "域指示 × 各描述符" 交互强度 = 全样本 mean |Phi_ij|，排序输出
    l3_interactions.csv（含域×描述符、域×域、以及 Top 描述符×描述符对照）。
(2) 关键对照 B3：域指示 + 线性模型（无交互项）同协议重跑
    —— 协议 = Phase 3 i3 pooled KFold(5, shuffle, seed) × 10 种子，
    EqV2 OOF MAE / Spearman ρ；同折同种子对照 B4。
    判读（预注册）：若 B3 也达 ~0.13 eV，"树模型交互修正"的机制解释坍塌，
    如实记录。结论写入 l3_b3_check.json。
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "innovation" / "domain_transfer"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
XGB_PARAMS = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]
B4_REF_MAE = 0.1226  # Phase 3 i3 domain_indicator 10 种子均值（innovation/functional_invariance/README.md）
COLLAPSE_THRESHOLD = 0.13  # 预注册判读线：B3 也达 ~0.13 eV 即机制解释坍塌


def load_domains():
    spec = importlib.util.spec_from_file_location(
        "run_invariance", ROOT / "innovation" / "functional_invariance" / "run_invariance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_domains()


def build_design(domains, feat_cols):
    names = list(domains)
    X = pd.concat([domains[k]["X"] for k in names], ignore_index=True)
    y = np.concatenate([domains[k]["y"] for k in names])
    dom = np.concatenate([[k] * len(domains[k]["y"]) for k in names])
    dum = pd.get_dummies(dom, drop_first=True)
    X_ind = np.hstack([X[feat_cols].values.astype(float), dum.values.astype(float)])
    ind_names = [f"domain::{c}" for c in dum.columns]
    return X_ind, y, dom, feat_cols + ind_names, ind_names


# ------------------------------------------------------------ SHAP 交互
def step_shap(X_ind, y, all_names, ind_names):
    m = XGBRegressor(random_state=0, n_jobs=-1, **XGB_PARAMS)
    m.fit(X_ind, y)
    booster = m.get_booster()
    booster.feature_names = all_names
    import xgboost as xgb
    dm = xgb.DMatrix(X_ind, feature_names=all_names)
    inter = booster.predict(dm, pred_interactions=True)  # (n, p+1, p+1) float32
    n_feat = len(all_names)
    inter = inter[:, :n_feat, :n_feat]
    mean_abs = np.abs(inter).mean(axis=0)
    np.fill_diagonal(mean_abs, np.nan)  # 主效应不进交互排序
    ind_idx = [all_names.index(c) for c in ind_names]
    rows = []
    for ii in ind_idx:
        for j, fname in enumerate(all_names):
            if j in ind_idx or np.isnan(mean_abs[ii, j]):
                continue
            rows.append({"domain_indicator": all_names[ii], "partner_feature": fname,
                         "pair_type": "domain_x_descriptor",
                         "mean_abs_interaction": float(mean_abs[ii, j])})
    for a in range(len(ind_idx)):
        for b in range(a + 1, len(ind_idx)):
            rows.append({"domain_indicator": all_names[ind_idx[a]],
                         "partner_feature": all_names[ind_idx[b]],
                         "pair_type": "domain_x_domain",
                         "mean_abs_interaction": float(mean_abs[ind_idx[a], ind_idx[b]])})
    # 对照：全特征对 Top（描述符×描述符）
    desc_idx = [j for j in range(n_feat) if j not in ind_idx]
    desc_pairs = []
    for a in range(len(desc_idx)):
        ia = desc_idx[a]
        for b in range(a + 1, len(desc_idx)):
            ib = desc_idx[b]
            desc_pairs.append((float(mean_abs[ia, ib]), all_names[ia], all_names[ib]))
    desc_pairs.sort(reverse=True)
    for v, fa, fb in desc_pairs[:20]:
        rows.append({"domain_indicator": fa, "partner_feature": fb,
                     "pair_type": "descriptor_x_descriptor_top20",
                     "mean_abs_interaction": v})
    df = pd.DataFrame(rows)
    dx = df[df["pair_type"] == "domain_x_descriptor"].copy()
    dx["rank_within_domain_indicator"] = dx.groupby("domain_indicator")[
        "mean_abs_interaction"].rank(ascending=False)
    df = df.merge(dx[["domain_indicator", "partner_feature",
                      "rank_within_domain_indicator"]],
                  on=["domain_indicator", "partner_feature"], how="left")
    df = df.sort_values(["pair_type", "mean_abs_interaction"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT / "l3_interactions.csv", index=False)
    top5 = dx.sort_values("mean_abs_interaction", ascending=False).head(5)
    print("[L3] 域指示×描述符 交互 Top5:")
    print(top5.to_string(index=False))
    return df, top5


# ------------------------------------------------------------ B3 对照
def step_b3(X_ind, y, dom):
    is_eq = dom == "EqV2_RPBE"
    rows = []
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_b3 = np.full(len(y), np.nan)
        oof_b4 = np.full(len(y), np.nan)
        for tr, te in kf.split(X_ind):
            mu = np.nanmean(X_ind[tr], axis=0)  # 折内均值填补（线性模型不吃 NaN）
            Xtr_i = np.where(np.isnan(X_ind[tr]), mu, X_ind[tr])
            Xte_i = np.where(np.isnan(X_ind[te]), mu, X_ind[te])
            lr = LinearRegression()
            lr.fit(Xtr_i, y[tr])
            oof_b3[te] = lr.predict(Xte_i)
            xg = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
            xg.fit(X_ind[tr], y[tr])
            oof_b4[te] = xg.predict(X_ind[te])
        rows.append({"seed": seed,
                     "B3_linear_EqV2_MAE": float(mean_absolute_error(y[is_eq], oof_b3[is_eq])),
                     "B3_linear_EqV2_rho": float(spearmanr(y[is_eq], oof_b3[is_eq])[0]),
                     "B4_gbdt_EqV2_MAE": float(mean_absolute_error(y[is_eq], oof_b4[is_eq])),
                     "B4_gbdt_EqV2_rho": float(spearmanr(y[is_eq], oof_b4[is_eq])[0])})
        print(f"[L3-B3] seed={seed:5d} B3 MAE={rows[-1]['B3_linear_EqV2_MAE']:.4f} "
              f"B4 MAE={rows[-1]['B4_gbdt_EqV2_MAE']:.4f}")
    return pd.DataFrame(rows)


def main():
    domains, feat_cols = load_domains()
    X_ind, y, dom, all_names, ind_names = build_design(domains, feat_cols)
    inter_df, top5 = step_shap(X_ind, y, all_names, ind_names)

    b3 = step_b3(X_ind, y, dom)
    b3_mae_mean = float(b3["B3_linear_EqV2_MAE"].mean())
    b4_mae_mean = float(b3["B4_gbdt_EqV2_MAE"].mean())
    collapse = b3_mae_mean <= COLLAPSE_THRESHOLD
    conclusion = (
        "MECHANISM COLLAPSED: B3 (domain indicator + linear model, no interactions) "
        "also reaches ~0.13 eV; the 'tree-model interaction correction' narrative is NOT "
        "supported — gains are attributable to additive domain shifts alone."
        if collapse else
        "MECHANISM SUPPORTED: B3 (no interactions) is clearly worse than B4; "
        "domain-indicator x descriptor interactions (quantified in l3_interactions.csv) "
        "carry a real, non-additive share of the B4 gain.")
    doc = {
        "protocol": "pooled KFold(5, shuffle, seed) x 10 seeds, same folds for B3/B4; "
                    "features = 84 descriptors + 2 domain dummies; EqV2 OOF metrics",
        "preregistered_rule": "if B3 also reaches ~0.13 eV -> mechanism explanation collapses, report honestly",
        "collapse_threshold_eV": COLLAPSE_THRESHOLD,
        "B4_reference_MAE_phase3": B4_REF_MAE,
        "B3_linear": {"EqV2_MAE_mean": b3_mae_mean,
                      "EqV2_MAE_sd": float(b3["B3_linear_EqV2_MAE"].std()),
                      "EqV2_rho_mean": float(b3["B3_linear_EqV2_rho"].mean())},
        "B4_gbdt_rerun": {"EqV2_MAE_mean": b4_mae_mean,
                          "EqV2_MAE_sd": float(b3["B4_gbdt_EqV2_MAE"].std()),
                          "EqV2_rho_mean": float(b3["B4_gbdt_EqV2_rho"].mean())},
        "delta_B3_minus_B4_MAE": b3_mae_mean - b4_mae_mean,
        "mechanism_collapsed": bool(collapse),
        "conclusion": conclusion,
        "per_seed": b3.to_dict("records"),
        "shap_interaction_top5": top5.to_dict("records"),
    }
    (OUT / "l3_b3_check.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[L3] B3 MAE={b3_mae_mean:.4f} vs B4 MAE={b4_mae_mean:.4f} "
          f"-> collapsed={collapse}")
    print("[写出] l3_interactions.csv / l3_b3_check.json")


if __name__ == "__main__":
    main()
