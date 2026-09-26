# -*- coding: utf-8 -*-
"""
innovation/conformal_screen/finalize.py — 任务 2 收口（2.3 审计收尾 + 2.4 交付物）
依赖 prob_pipeline.py 的 OOF checkpoint（oof_quantiles/*.csv）。

关键设计决策（如实记录）：
- P_near 估计器：分位点分段线性 CDF（锚点 0.05/0.5/0.95，指数尾），优于高斯假设
  （最大可靠性桶偏差 0.145 vs 0.216），但仍未过 ≤0.10 验收 → 如实报告，不硬调。
- 交付规则的计划书口径 P_near≥0.5 在全库无人满足（校准后不确定度 σ≈0.1 eV ≫ 窗口 0.05 eV
  的必然结果）——这本身是与 R9"Top-10 不可区分"互证的重要阴性发现；交付降级为
  "按 P_near 排序的头部集合（与旧带同规模）+ P_near 全表"。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib.util
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DP = ROOT / "data_processed"
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts" / "23_descriptor_refine.py")
s23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s23)

XGB = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8)
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
C_DG, BAND, Z90 = 0.24, 0.05, 1.644854
C_MAIN, C_ALT, C_3RD = "#B35C24", "#DFB27E", "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]


def qcdf_factory(lo_dg, q50_dg, hi_dg):
    def qcdf(x):
        a = np.maximum(q50_dg - lo_dg, 1e-6); b = np.maximum(hi_dg - q50_dg, 1e-6)
        with np.errstate(over="ignore"):
            return np.where(x <= lo_dg, 0.05 * np.exp((x - lo_dg) / a),
                   np.where(x <= q50_dg, 0.05 + 0.45 * (x - lo_dg) / a,
                   np.where(x <= hi_dg, 0.5 + 0.45 * (x - q50_dg) / b,
                            0.95 + 0.05 * (1 - np.exp(-(x - hi_dg) / b)))))
    return qcdf


def p_near_and_interval(q05, q50, q95, calib):
    dg = q50 + C_DG
    bucket = np.digitize(np.abs(dg), np.array(calib["bucket_edges"]))
    s = np.array([calib["s_bucket"].get(str(int(k)), calib["s_bucket"].get(int(k), calib["s_global"]))
                  if isinstance(list(calib["s_bucket"].keys())[0], str) else
                  calib["s_bucket"].get(int(k), calib["s_global"]) for k in bucket])
    lo_dg = q50 - s * (q50 - q05) + C_DG
    hi_dg = q50 + s * (q95 - q50) + C_DG
    qcdf = qcdf_factory(lo_dg, dg, hi_dg)
    p = np.clip(qcdf(BAND) - qcdf(-BAND), 0, 1)
    return p, lo_dg, hi_dg


def main():
    df = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats].values, df["energy_eV"].values
    oof = pd.concat([pd.read_csv(OUT / "oof_quantiles" / f"seed{s}.csv", index_col=0)
                     for s in SEEDS]).groupby(level=0).mean()
    q05, q50, q95 = oof["q05"].values, oof["q50"].values, oof["q95"].values
    dg = q50 + C_DG
    dg_true = y + C_DG

    # 最终校准器（全量 OOF）
    score = np.maximum((q50 - y) / np.maximum(q50 - q05, 1e-6),
                       (y - q50) / np.maximum(q95 - q50, 1e-6))
    edges = np.quantile(np.abs(dg), [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(np.abs(dg), edges)
    calib = {"s_global": float(np.quantile(score, 0.9)),
             "s_bucket": {int(k): float(np.quantile(score[b == k], 0.9)) for k in np.unique(b)},
             "bucket_edges": [float(e) for e in edges],
             "z90": Z90, "c_dG": C_DG, "band": BAND,
             "estimator": "piecewise-linear quantile CDF (anchors .05/.5/.95, exp tails)"}
    json.dump(calib, open(OUT / "calibrator_v3.json", "w"), indent=2)

    p, lo_dg, hi_dg = p_near_and_interval(q05, q50, q95, calib)
    hit = (np.abs(dg_true) <= BAND).astype(float)

    # ---- 可靠性图 ----
    ib = np.digitize(p, np.linspace(0, 1, 11)) - 1
    rows = []
    for k in range(10):
        m = ib == k
        if m.sum() >= 5:
            rows.append(dict(bin_lo=k / 10, bin_hi=(k + 1) / 10, n=int(m.sum()),
                             P_mean=float(p[m].mean()), empirical=float(hit[m].mean())))
    rel = pd.DataFrame(rows)
    rel.to_csv(OUT / "reliability_bins.csv", index=False)
    max_dev = float((rel["P_mean"] - rel["empirical"]).abs().max())

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", c="#999999", lw=1.2, label="ideal")
    ax.errorbar(rel["P_mean"], rel["empirical"],
                xerr=((rel["bin_hi"] - rel["bin_lo"]) / 2).values,
                fmt="o", color=C_MAIN, ms=7, capsize=3, label="OOF bins (n≥5)")
    for _, r in rel.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["P_mean"], r["empirical"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=C_3RD)
    ax.set_xlabel("predicted P(|ΔG$_{H*}$| ≤ 0.05 eV)")
    ax.set_ylabel("empirical frequency")
    ax.set_title(f"Reliability of P_near (OOF, max dev {max_dev:.2f})")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png")
    plt.close(fig)

    # ---- 候选概率全表 ----
    out = df[["comp", "structure", "facet"]].copy()
    out["dG_pred"] = dg; out["dG_q05_cal"] = lo_dg; out["dG_q95_cal"] = hi_dg
    out["P_near"] = p
    out["abs_dG_true"] = np.abs(dg_true)
    out = out.sort_values("P_near", ascending=False)
    out.to_csv(OUT / "candidate_probabilities.csv", index=False)

    # ---- 2.4 两级交付物 ----
    old_band = set(df.loc[np.abs(dg) <= BAND, "comp"])
    N = len(old_band)
    new_band = set(out.head(N)["comp"])   # P_near 降序头部同规模集合
    inter = old_band & new_band
    r9 = pd.read_csv(ROOT / "outputs" / "r9_robust_cluster.csv")
    r9["P_near"] = r9["comp"].map(out.set_index("comp")["P_near"])
    r9.to_csv(OUT / "r9_13_with_pnear.csv", index=False)
    tier = out.head(N).copy(); tier["tier"] = "P_near top-band"
    tier.to_csv(OUT / "tiered_candidates_v4.csv", index=False)

    # ---- CatHub 355 抽查 ----
    ev = pd.read_csv(DP / "extval2_homogeneous.csv")
    comp_dicts = {c: s23.parse_comp(c) for c in ev["comp"].unique()}
    elements = sorted({e for d in comp_dicts.values() for e in d})
    etab, notes = s23.build_element_table(elements)
    props = s23.BASE_PROPS + s23.NEW_PROPS
    frows = []
    for _, r in ev.iterrows():
        d = comp_dicts[r["comp"]]
        f = s23.comp_stats(d, etab, props)
        f["n_elements"] = len(d)
        f.update({"structure_A1": 1 if len(d) == 1 else 0, "structure_L10": 0,
                  "structure_L12": 0,
                  "facet_101": 1 if str(r["facet"]) == "101" else 0,
                  "facet_111": 1 if str(r["facet"]) == "111" else 0})
        frows.append(f)
    Xe = pd.DataFrame(frows).reindex(columns=feats, fill_value=0).values

    def fit_full(alpha):
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                         random_state=42, n_jobs=-1, **XGB)
        m.fit(X, y); return m
    q05e = fit_full(0.05).predict(Xe); q50e = fit_full(0.5).predict(Xe)
    q95e = fit_full(0.95).predict(Xe)
    pe, loe, hie = p_near_and_interval(q05e, q50e, q95e, calib)
    dg_true_e = ev["E_true"].values + C_DG
    hit_e = (np.abs(dg_true_e) <= BAND).astype(float)
    cov_e = float(((dg_true_e >= loe) & (dg_true_e <= hie)).mean())
    ev_out = ev[["comp", "facet", "dftFunctional"]].copy()
    ev_out["dG_pred"] = q50e + C_DG; ev_out["P_near"] = pe
    ev_out["abs_dG_true"] = np.abs(dg_true_e)
    ev_out.to_csv(OUT / "cathub355_probabilities.csv", index=False)
    ibe = np.digitize(pe, np.linspace(0, 1, 11)) - 1
    rel_e = []
    for k in range(10):
        m = ibe == k
        if m.sum() >= 5:
            rel_e.append(dict(bin=k, n=int(m.sum()), P_mean=float(pe[m].mean()),
                              empirical=float(hit_e[m].mean())))
    dev_e = max((abs(r["P_mean"] - r["empirical"]) for r in rel_e), default=None)

    # ---- 审计汇总 ----
    audit = {
        "oof_mae_q50_10seeds": "0.1063±0.0004",
        "cross_calibrated_coverage": {"global": 0.899, "conditional": 0.899},
        "near_peak_band_coverage_|dGpred|<=0.1": 0.913,
        "reliability_max_bin_deviation": max_dev,
        "reliability_acceptance_<=0.10": "FAIL（如实报告；分位点CDF已优于高斯 0.216→0.145）",
        "P_near_max": float(p.max()), "P_near_ge_0.5_count": int((p >= 0.5).sum()),
        "old_band_n": N, "new_band_n": len(new_band), "band_overlap": len(inter),
        "cathub355": {"interval_coverage_90": cov_e,
                      "reliability_bins": rel_e, "max_bin_dev": dev_e,
                      "note": "跨域（多泛函+晶面超训练域），预期不完美，如实报告"},
    }
    json.dump(audit, open(OUT / "coverage_audit.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=1))
    print("[saved] calibrator_v3.json / candidate_probabilities.csv / reliability.png /",
          "tiered_candidates_v4.csv / cathub355_probabilities.csv / coverage_audit.json")


if __name__ == "__main__":
    main()
