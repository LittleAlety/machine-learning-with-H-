# -*- coding: utf-8 -*-
"""
33_monotonic.py — A-2：单调性约束实验（XGBoost monotone_constraints）

预注册单调对（方向待 SHAP 复核）：
  group_wmean↑ → E_ads↑；en_wmean↑ → E_ads↑；d_el_wmean↑ → E_ads↑
三档：无约束 / Top-3（预注册三对按 SHAP 复核后的方向）/ Top-6（三对 + SHAP 主效应
  最强且符号稳定的另 3 个特征）
协议：10 种子同折 OOF 公平契约（增益>0.005 且 10/10 一致）+ 嵌套 CV 复核
  （外层 GroupKFold(5)，内层 GroupKFold(4) 在三档间选择）。
输出：outputs/r14_a2_monotonic.json；outputs/r14_a2_monotonic_shap.csv
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, DP, XGB, SEEDS, load_v3, mae

OUT = ROOT / "outputs"
PREREG = ["group_wmean", "en_wmean", "d_el_wmean"]  # 预注册↑方向（SHAP 复核）


def constraints_vector(feats, dirs):
    return tuple(int(dirs.get(f, 0)) for f in feats)


def shap_main_effects(X, y, feats, seed=42):
    """全数据 XGB + SHAP：每特征 mean shap 与特征值的秩相关符号 → 单调方向复核。"""
    import shap
    from scipy.stats import spearmanr
    m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB).fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    rec = {}
    for j, f in enumerate(feats):
        rho = spearmanr(X[:, j], sv[:, j]).statistic
        rec[f] = {"mean_abs_shap": float(np.mean(np.abs(sv[:, j]))),
                  "spearman_rho": float(rho),
                  "direction": int(np.sign(rho)) if abs(rho) > 0.1 else 0}
    return rec


def oof(X, y, groups, seed, cons):
    p = np.full(len(y), np.nan)
    kw = dict(random_state=seed, n_jobs=-1, **XGB)
    if cons is not None:
        kw["monotone_constraints"] = cons
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = XGBRegressor(**kw).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return mae(p, y)


def main():
    df, feats = load_v3()
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values

    shap_rec = shap_main_effects(X, y, feats)
    pd.DataFrame(shap_rec).T.to_csv(OUT / "r14_a2_monotonic_shap.csv")
    prereg_check = {f: {"prereg": "+1(↑E_ads)", "shap_dir": shap_rec[f]["direction"]}
                    for f in PREREG}
    print("[SHAP 复核]", prereg_check)

    # 三档约束
    tiers = {}
    tiers["none"] = None
    d3 = {f: (shap_rec[f]["direction"] or 1) for f in PREREG}  # 方向以 SHAP 复核为准
    tiers["top3"] = constraints_vector(feats, d3)
    # Top-6：三对之外按 mean|shap| 排序取方向显著的前 3 个
    rest = sorted((f for f in feats if f not in PREREG),
                  key=lambda f: -shap_rec[f]["mean_abs_shap"])
    extra = [f for f in rest if shap_rec[f]["direction"] != 0][:3]
    d6 = dict(d3); d6.update({f: shap_rec[f]["direction"] for f in extra})
    tiers["top6"] = constraints_vector(feats, d6)
    print(f"[档位] top3={list(d3.items())} top6_extra={[(f, d6[f]) for f in extra]}")

    # 10 种子公平契约
    per = {t: [oof(X, y, groups, s, c) for s in SEEDS] for t, c in tiers.items()}
    base = np.array(per["none"])
    summ = {}
    for t in ("top3", "top6"):
        diffs = base - np.array(per[t])
        summ[t] = {"per_seed_mae": per[t], "mean": float(np.mean(per[t])),
                   "gain_vs_none": float(diffs.mean()),
                   "direction_consistent": int((diffs > 0).sum()),
                   "contract_gain_ok": bool(diffs.mean() > 0.005),
                   "contract_direction_ok": bool((diffs > 0).all())}
        print(f"[{t}] {np.mean(per[t]):.4f}  增益 {diffs.mean():+.4f}  "
              f"方向 {(diffs>0).sum()}/10", flush=True)
    summ["none"] = {"per_seed_mae": per["none"], "mean": float(base.mean())}

    # 嵌套 CV：内层三档选择
    nested = np.full(len(y), np.nan)
    picks = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        im = {}
        for t, c in tiers.items():
            p = np.full(len(tr), np.nan)
            kw = dict(random_state=42, n_jobs=-1, **XGB)
            if c is not None:
                kw["monotone_constraints"] = c
            for itr, ite in GroupKFold(4).split(X[tr], y[tr], groups[tr]):
                m = XGBRegressor(**kw).fit(X[tr][itr], y[tr][itr])
                p[ite] = m.predict(X[tr][ite])
            im[t] = mae(p, y[tr])
        best = min(im, key=im.get)
        picks.append(best)
        kw = dict(random_state=42, n_jobs=-1, **XGB)
        if tiers[best] is not None:
            kw["monotone_constraints"] = tiers[best]
        nested[te] = XGBRegressor(**kw).fit(X[tr], y[tr]).predict(X[te])
        print(f"  外层折内层选择 {best} {im}", flush=True)
    nested_mae = mae(nested, y)

    best_tier = min(("top3", "top6"), key=lambda t: summ[t]["mean"])
    adopt = (summ[best_tier]["contract_gain_ok"]
             and summ[best_tier]["contract_direction_ok"]
             and nested_mae < base.mean() - 0.005)
    res = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_pairs": {f: "+1" for f in PREREG},
        "shap_check": prereg_check, "top6_extra": {f: d6[f] for f in extra},
        "summary": summ, "nested_cv": {"mae": nested_mae, "picks": picks},
        "verdict": {
            "adopt": bool(adopt), "best_tier": best_tier if adopt else None,
            "reason": (f"{'采纳 '+best_tier if adopt else '不采纳'}："
                       f"best={best_tier} 增益 {summ[best_tier]['gain_vs_none']:+.4f}，"
                       f"方向 {summ[best_tier]['direction_consistent']}/10，"
                       f"嵌套 CV {nested_mae:.4f} vs 基线 {base.mean():.4f}")}}
    json.dump(res, open(OUT / "r14_a2_monotonic.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"[判定] {res['verdict']['reason']}")
    print("[写出] outputs/r14_a2_monotonic.json, r14_a2_monotonic_shap.csv")


if __name__ == "__main__":
    main()
