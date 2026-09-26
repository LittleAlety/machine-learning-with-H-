# -*- coding: utf-8 -*-
"""
innovation/conformal_screen/prob_pipeline.py — 校准化热中性概率（任务 2 一体化管线）

2.1 v3 模型 + 10 种子 × GroupKFold(5) OOF 分位数预测（q05/q50/q95，种子间取均值）
2.2 分组条件 conformal 校准（全局 + |ΔG_pred| 五桶）→ P(|ΔG_H*|≤0.05 eV)
2.3 覆盖率审计：交叉校准（5 折外推）库内可靠性图 + 近峰带专项 + CatHub 355 抽查
2.4 两级交付物升级：P_near≥0.5 且校准区间跨 0.05 的新候选带 vs 旧口径对比

断点续跑：OOF 分位数按种子 checkpoint 到 oof_quantiles/ 目录。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DP = ROOT / "data_processed"
OUT = Path(__file__).resolve().parent
CKPT = OUT / "oof_quantiles"
CKPT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts" / "23_descriptor_refine.py")
s23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s23)  # 有 __main__ 保护，安全

XGB = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8)
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
C_DG = 0.24          # ΔG 修正常数
BAND = 0.05          # 近热中性窗口
Z90 = 1.644854       # Φ⁻¹(0.95)：90% 区间半宽系数

C_MAIN, C_ALT, C_3RD = "#B35C24", "#DFB27E", "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]


def fit_q(Xtr, ytr, alpha, seed):
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                     random_state=seed, n_jobs=-1, **XGB)
    m.fit(Xtr, ytr)
    return m


def step1_oof_quantiles(df, X, y, groups):
    """10 种子 OOF q05/q50/q95，逐种子 checkpoint；返回种子均值 DataFrame。"""
    per_seed = []
    for s in SEEDS:
        f = CKPT / f"seed{s}.csv"
        if f.exists():
            per_seed.append(pd.read_csv(f, index_col=0))
            continue
        res = pd.DataFrame(index=df.index,
                           columns=["q05", "q50", "q95"], dtype=float)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, y, groups):
            res.loc[te, "q05"] = fit_q(X[tr], y[tr], 0.05, s).predict(X[te])
            res.loc[te, "q50"] = fit_q(X[tr], y[tr], 0.50, s).predict(X[te])
            res.loc[te, "q95"] = fit_q(X[tr], y[tr], 0.95, s).predict(X[te])
        res.to_csv(f)
        mae = float(np.mean(np.abs(res["q50"] - y)))
        print(f"  [seed {s}] OOF MAE(q50) = {mae:.4f}", flush=True)
        per_seed.append(res)
    avg = pd.concat(per_seed).groupby(level=0).mean()
    maes = [float(np.mean(np.abs(r["q50"] - y))) for r in per_seed]
    print(f"[step1] 10 种子 OOF MAE(q50): {np.mean(maes):.4f}±{np.std(maes):.4f}")
    return avg, maes


def _score(yv, q05, q50, q95):
    """双侧非对称 conformal 分数：max(下侧偏差/下臂, 上侧偏差/上臂)。"""
    lo_arm = np.maximum(q50 - q05, 1e-6)
    hi_arm = np.maximum(q95 - q50, 1e-6)
    return np.maximum((q50 - yv) / lo_arm, (yv - q50) / hi_arm)


def calibrate(oof, y, bucket_edges):
    """条件 conformal：非对称 score，全局与 |ΔG_pred| 分桶的 90% 分位数缩放。"""
    q05, q50, q95 = oof["q05"].values, oof["q50"].values, oof["q95"].values
    score = _score(y, q05, q50, q95)
    dg_pred = q50 + C_DG
    bucket = np.digitize(np.abs(dg_pred), bucket_edges)
    s_global = float(np.quantile(score, 0.9))
    s_bucket = {int(b): float(np.quantile(score[bucket == b], 0.9))
                for b in np.unique(bucket)}
    return {"s_global": s_global, "s_bucket": s_bucket,
            "bucket_edges": list(map(float, bucket_edges))}


def apply_calib(q05, q50, q95, calib):
    """返回 (P_near, dg_q05_cal, dg_q95_cal, bucket)"""
    hw = np.maximum((q95 - q05) / 2, 1e-6)
    dg = q50 + C_DG
    bucket = np.digitize(np.abs(dg), np.array(calib["bucket_edges"]))
    s = np.array([calib["s_bucket"].get(int(b), calib["s_global"]) for b in bucket])
    lo, hi = q50 - s * (q50 - q05), q50 + s * (q95 - q50)
    sigma = (hi - lo) / 2 / Z90
    mu = dg
    p = norm.cdf((BAND - mu) / sigma) - norm.cdf((-BAND - mu) / sigma)
    return p, lo + C_DG, hi + C_DG, bucket


def coverage(y, q05, q50, q95, calib):
    lo, hi = q50 - 0, q50 + 0  # placeholder
    p, lo_dg, hi_dg, _ = apply_calib(q05, q50, q95, calib)
    dg_true = y + C_DG
    cov = ((dg_true >= lo_dg) & (dg_true <= hi_dg)).mean()
    return float(cov), p


def main():
    df = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    assert len(feats) == 84, len(feats)
    X, y, groups = df[feats].values, df["energy_eV"].values, df["comp"].values
    print(f"[load] v3 {X.shape}")

    # ---- 2.1 OOF 分位数 ----
    oof, seed_maes = step1_oof_quantiles(df, X, y, groups)

    # ---- 2.2 校准（全量 OOF 上拟合最终校准器）----
    dg_pred = oof["q50"].values + C_DG
    edges = np.quantile(np.abs(dg_pred), [0.2, 0.4, 0.6, 0.8])
    calib = calibrate(oof, y, edges)
    print(f"[step2] 全局 s90={calib['s_global']:.3f}；分桶 s90="
          f"{ {k: round(v,3) for k,v in calib['s_bucket'].items()} }")

    # ---- 2.3a 交叉校准审计（诚实的库内覆盖率：校准与评估不同数据）----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    covs_g, covs_c, p_all = [], [], np.full(len(y), np.nan)
    for trc, tee in kf.split(oof):
        cal_g = {"s_global": float(np.quantile(
            _score(y[trc], oof["q05"].values[trc], oof["q50"].values[trc],
                   oof["q95"].values[trc]), 0.9)),
            "s_bucket": {}, "bucket_edges": calib["bucket_edges"]}
        cal_c = calibrate(oof.iloc[trc], y[trc], calib["bucket_edges"])
        cg, _ = coverage(y[tee], oof["q05"].values[tee], oof["q50"].values[tee],
                         oof["q95"].values[tee], cal_g)
        cc, p_te = coverage(y[tee], oof["q05"].values[tee], oof["q50"].values[tee],
                            oof["q95"].values[tee], cal_c)
        covs_g.append(cg); covs_c.append(cc); p_all[tee] = p_te
    print(f"[audit] 交叉校准覆盖率 全局={np.mean(covs_g):.3f} 条件={np.mean(covs_c):.3f}")

    # 可靠性分桶（用最终校准器的 P_near 与全量 OOF 真值；同时报交叉校准的覆盖率）
    p_near, lo_dg, hi_dg, bucket = apply_calib(
        oof["q05"].values, oof["q50"].values, oof["q95"].values, calib)
    dg_true = y + C_DG
    hit = (np.abs(dg_true) <= BAND).astype(float)
    bins = np.linspace(0, 1, 11)
    ib = np.digitize(p_near, bins) - 1
    rel_rows, max_dev = [], 0.0
    for b in range(10):
        m = ib == b
        if m.sum() < 5:
            continue
        pm, hm = float(p_near[m].mean()), float(hit[m].mean())
        max_dev = max(max_dev, abs(pm - hm))
        rel_rows.append({"bin": b, "n": int(m.sum()), "P_mean": pm, "empirical": hm})
    rel = pd.DataFrame(rel_rows)
    print(f"[audit] 可靠性最大桶偏差 = {max_dev:.3f}（验收 ≤0.10）")

    # 近峰带专项（交叉校准覆盖率）
    near = np.abs(dg_pred) <= 0.1
    covs_near = []
    for trc, tee in kf.split(oof):
        tee2 = tee[np.abs(dg_pred[tee]) <= 0.1]
        if len(tee2) == 0:
            continue
        cal_c = calibrate(oof.iloc[trc], y[trc], calib["bucket_edges"])
        cc, _ = coverage(y[tee2], oof["q05"].values[tee2], oof["q50"].values[tee2],
                         oof["q95"].values[tee2], cal_c)
        covs_near.append(cc)
    near_cov = float(np.mean(covs_near))
    print(f"[audit] 近峰带(|ΔG_pred|≤0.1, n={near.sum()}) 交叉校准覆盖率 = {near_cov:.3f}（验收 ≥0.80）")

    np.save(OUT / "_p_all.npy", p_all)
    return df, feats, X, y, groups, oof, calib, rel, max_dev, near_cov, seed_maes


if __name__ == "__main__":
    main()
