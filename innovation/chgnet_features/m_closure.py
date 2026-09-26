# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/m_closure.py — Phase 4 Workstream M: 磁性线最后一次预注册尝试（收口）

预注册来源: config/preregistered_phase4.yaml workstream_M_magnetic
  candidates_locked = ["m_max", "m_mean_wstd_ratio", "m_surface_mean", "m_max_over_mean", "m_gini"]  (≤5, 禁止扩候选)
  rule: "公平契约 + 嵌套 CV；过线 → v5 磁性定向版；不过 → 关闭，仅真实新数据可重启"

协议（与 scripts/r14_common.py / 公平契约完全同口径）:
  - 基线: v3 84 维特征集 (hstar_features_v3_84feat.csv 去掉 ID 列)
  - 每候选单独 +1 维 → 85 维
  - GroupKFold(5, groups=comp) 固定折, 种子 [0,1,2,7,13,42,99,123,2024,31337]
  - XGBRegressor(lr=0.03, depth=7, n=400, subsample=0.8)

判定（须同时满足才过线，任一候选过线即 v5 磁性定向版）:
  A. 3d 磁性子层（comp 含 Fe/Co/Ni/Mn/Cr, n=398）OOF MAE 改善 ≥ 0.02 eV（10 种子均值, 定向线）
  B. 全局公平契约: OOF MAE 改善 > 0.005 eV 且 10/10 种子方向一致
  过线候选另跑嵌套 CV（外层 GroupKFold(5)/内层 GroupKFold(4) 二选一）复核；无过线候选则记为不适用。

候选 → 实际列映射（候选为聚合统计概念；一次性构造并登记，禁止事后新增候选）:
  m_max             = chg_surf_mag_max                                    [直接]
  m_surface_mean    = chg_surf_mag_mean                                   [直接]
  m_mean_wstd_ratio = chg_surf_mag_mean / (chg_surf_mag_std + 1e-6)       [构造: csv 无加权 std, 用表面磁矩总体 std 近似]
  m_max_over_mean   = chg_surf_mag_max / (chg_surf_mag_mean + 1e-6)       [构造]
  m_gini            = 表面层 |磁矩| 的 Gini 系数                            [构造: 由 cache/*.json 的 m_raw/z_raw
                      按 run_chgnet.py 同口径 (z > 75 分位) 取表面层后计算]

幂等: 重复运行覆盖同一 m_closure.json, 数字确定（固定折/种子）。
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from r14_common import XGB, SEEDS, load_v3, oof_predict, mae  # noqa: E402

OUT = Path(__file__).resolve().parent
CHG_CSV = OUT / "chgnet_features.csv"
CACHE = OUT / "cache"
EPS = 1e-6
MAG3D = ["Fe", "Co", "Ni", "Mn", "Cr"]
CANDIDATES = ["m_max", "m_mean_wstd_ratio", "m_surface_mean", "m_max_over_mean", "m_gini"]
D_THRESH, G_THRESH = 0.02, 0.005

MAPPING_DOC = {
    "m_max": "chg_surf_mag_max (直接)",
    "m_surface_mean": "chg_surf_mag_mean (直接)",
    "m_mean_wstd_ratio": "chg_surf_mag_mean/(chg_surf_mag_std+1e-6) (构造)",
    "m_max_over_mean": "chg_surf_mag_max/(chg_surf_mag_mean+1e-6) (构造)",
    "m_gini": "Gini(|m_surf|), m_raw/z_raw top-z-quartile (构造)",
}


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def build_candidates(df):
    """按登记映射构造 5 个候选特征（一次性, 顺序 = candidates_locked）。"""
    cols = {}
    cols["m_max"] = df["chg_surf_mag_max"].values
    cols["m_surface_mean"] = df["chg_surf_mag_mean"].values
    cols["m_mean_wstd_ratio"] = (df["chg_surf_mag_mean"] / (df["chg_surf_mag_std"] + EPS)).values
    cols["m_max_over_mean"] = (df["chg_surf_mag_max"] / (df["chg_surf_mag_mean"] + EPS)).values
    g = []
    for _, r in df.iterrows():
        d = json.loads((CACHE / f"{r.comp}_{r.structure}_{r.facet}.json").read_text())
        m, z = np.array(d["m_raw"]), np.array(d["z_raw"])
        g.append(gini(m[z > np.quantile(z, 0.75)]))
    cols["m_gini"] = np.array(g)
    return cols


def fit_subset(X, y, groups, seed):
    return oof_predict(X, y, groups, seed)


def main():
    df, feats = load_v3()
    assert len(feats) == 84
    chg = pd.read_csv(CHG_CSV)
    df = df.merge(chg[["comp", "structure", "facet", "chg_surf_mag_mean",
                       "chg_surf_mag_max", "chg_surf_mag_std"]],
                  on=["comp", "structure", "facet"], how="inner")
    assert len(df) == 1836, len(df)
    Xb = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values
    has3d = df["comp"].apply(lambda c: bool(set(MAG3D) & set(re.findall(r"[A-Z][a-z]?", c)))).values
    cand = build_candidates(df)

    # 基线 OOF（全量 + 3d 子层重训, 与 H4 双口径）
    base_full = {s: oof_predict(Xb, y, groups, s) for s in SEEDS}
    base3d = {s: fit_subset(Xb[has3d], y[has3d], groups[has3d], s) for s in SEEDS}
    base = {"global_mae": {str(s): mae(base_full[s], y) for s in SEEDS},
            "subset3d_mae_from_full": {str(s): mae(base_full[s][has3d], y[has3d]) for s in SEEDS},
            "subset3d_mae_refit": {str(s): mae(base3d[s], y[has3d]) for s in SEEDS}}
    print(f"[base] global {np.mean(list(base['global_mae'].values())):.5f} "
          f"3d(full) {np.mean(list(base['subset3d_mae_from_full'].values())):.5f} "
          f"3d(refit) {np.mean(list(base['subset3d_mae_refit'].values())):.5f}", flush=True)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"baseline": "v3 84feat", "model": XGB, "seeds": SEEDS,
                           "folds": "GroupKFold(5, groups=comp)",
                           "n_rows": int(len(df)), "n_3d_subset": int(has3d.sum())},
              "candidate_mapping": MAPPING_DOC, "baseline": base, "candidates": {}}

    for name in CANDIDATES:
        Xg = np.hstack([Xb, cand[name].reshape(-1, 1)])
        gold_full = {s: oof_predict(Xg, y, groups, s) for s in SEEDS}
        gold3d = {s: fit_subset(Xg[has3d], y[has3d], groups[has3d], s) for s in SEEDS}
        d_glob = [base["global_mae"][str(s)] - mae(gold_full[s], y) for s in SEEDS]
        d_3df = [base["subset3d_mae_from_full"][str(s)] - mae(gold_full[s][has3d], y[has3d]) for s in SEEDS]
        d_3dr = [base["subset3d_mae_refit"][str(s)] - mae(gold3d[s], y[has3d]) for s in SEEDS]
        res = {
            "global": {"delta_mean": float(np.mean(d_glob)), "delta_std": float(np.std(d_glob)),
                       "seeds_win": int(sum(d > 0 for d in d_glob)),
                       "per_seed_delta": {str(s): float(d) for s, d in zip(SEEDS, d_glob)},
                       "mae_mean": float(np.mean([mae(gold_full[s], y) for s in SEEDS])),
                       "contract_passed": bool(np.mean(d_glob) > G_THRESH and all(d > 0 for d in d_glob))},
            "subset3d_from_full_oof": {"delta_mean": float(np.mean(d_3df)),
                                       "delta_std": float(np.std(d_3df)),
                                       "seeds_win": int(sum(d > 0 for d in d_3df)),
                                       "thresh": D_THRESH,
                                       "passed": bool(np.mean(d_3df) >= D_THRESH)},
            "subset3d_refit_h4style": {"delta_mean": float(np.mean(d_3dr)),
                                       "delta_std": float(np.std(d_3dr)),
                                       "seeds_win": int(sum(d > 0 for d in d_3dr)),
                                       "thresh": D_THRESH,
                                       "passed": bool(np.mean(d_3dr) >= D_THRESH)},
        }
        res["targeted_passed"] = bool(res["subset3d_from_full_oof"]["passed"]
                                      and res["subset3d_refit_h4style"]["passed"])
        res["overall_passed"] = bool(res["targeted_passed"] and res["global"]["contract_passed"])
        report["candidates"][name] = res
        print(f"[{name}] Δglob {np.mean(d_glob):+.5f} ({sum(d>0 for d in d_glob)}/10) | "
              f"Δ3d(full) {np.mean(d_3df):+.5f} | Δ3d(refit) {np.mean(d_3dr):+.5f} | "
              f"overall {'PASS' if res['overall_passed'] else 'fail'}", flush=True)

    winners = [n for n in CANDIDATES if report["candidates"][n]["overall_passed"]]
    # 嵌套 CV 复核（仅过线候选；外层 5 折内层 4 折在 base/base+cand 间二选一）
    nested = {}
    for name in winners:
        Xg = np.hstack([Xb, cand[name].reshape(-1, 1)])
        pred = np.full(len(y), np.nan)
        picks = []
        for tr, te in GroupKFold(5).split(Xb, y, groups):
            inner_mae = {}
            for tag, X in [("base", Xb), ("cand", Xg)]:
                p = np.full(len(tr), np.nan)
                for itr, ite in GroupKFold(4).split(X[tr], y[tr], groups[tr]):
                    m = XGBRegressor(random_state=42, n_jobs=-1, **XGB)
                    m.fit(X[tr][itr], y[tr][itr])
                    p[ite] = m.predict(X[tr][ite])
                inner_mae[tag] = mae(p, y[tr])
            best = min(inner_mae, key=inner_mae.get)
            picks.append(best)
            X = Xb if best == "base" else Xg
            m = XGBRegressor(random_state=42, n_jobs=-1, **XGB).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        nested[name] = {"inner_picks": picks, "nested_mae": mae(pred, y),
                        "baseline_nested_mae_expected": 0.1134,
                        "confirmed": bool(mae(pred, y) < 0.1134 - G_THRESH)}
    report["nested_cv"] = nested if nested else {"skipped": "无过线候选，嵌套 CV 不适用"}

    if winners and all(nested[w]["confirmed"] for w in winners):
        verdict = "V5_MAGNETIC_TARGETED"
    else:
        verdict = "CLOSED"
    report["verdict"] = verdict
    report["closure_statement"] = (
        "Phase 4 Workstream M 为磁性特征线最后一次预注册尝试（候选 ≤5，先注册后运行）。"
        + ("判定: 过线 → v5 磁性定向版成文。" if verdict == "V5_MAGNETIC_TARGETED"
           else "判定: 5 候选均未同时满足 3d 子层 Δ≥0.02 eV 与全局公平契约（>0.005 且 10/10），"
                "磁性特征线一次性强制关闭。此后该线只接受真实新数据（新 DFT/实验磁矩测量等）重启，"
                "禁止在既有数据上继续挖掘派生统计量。"))
    (OUT / "m_closure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"verdict": verdict, "winners": winners}, ensure_ascii=False))


if __name__ == "__main__":
    main()
