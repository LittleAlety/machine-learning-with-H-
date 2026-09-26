# -*- coding: utf-8 -*-
"""
innovation/continuous_local/r3_eval.py — 闸门 + 公平契约 + 嵌套CV + 3d 子层 (Phase5 route2)

协议与 innovation/portability_gate/gate.py, chgnet_features/h2_h5_eval.py 完全同口径:
  XGB_PARAMS = lr0.03/depth7/n400/sub0.8, GroupKFold(5, comp), 10 种子
  契约: Δ OOF MAE > 0.005 且 10/10 种子改善; 嵌套 CV 外层 GroupKFold(5) 复核
覆盖率: 弛豫限时降级 -> 仅 relaxed_subset 覆盖行参与主评估 (same folds/seeds, 行内公平);
  次级口径: 全 1836 行, 未覆盖行特征 NaN (XGBoost 原生处理), 仅作稀释对照披露。
诚实条款: 本会话无 DFT -> 全部结论条件性 ("弛豫质量未经 DFT 锚定")。
"""
import json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portability_gate"))
from gate import V2_COLS, XGB_PARAMS, SEEDS, THRESH, RETAIN, oof_mae, ridge_infer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
NEW = OUT / "continuous_local_features.csv"

MAG3D = ["Fe", "Co", "Ni", "Mn", "Cr"]
META = {"name", "n_atoms", "relax_n_steps", "relax_converged", "relax_fmax_final",
        "comp", "structure", "facet"}


def fit_pred(Xtr, ytr, Xte, seed, params=None):
    m = XGBRegressor(random_state=seed, n_jobs=-1, **(params or XGB_PARAMS))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def oof(X, y, g, seed):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = fit_pred(X[tr], y[tr], X[te], seed)
    return pred


def contract_block(Xb, Xg, y, g, tag):
    base_p = {s: oof(Xb, y, g, s) for s in SEEDS}
    gold_p = {s: oof(Xg, y, g, s) for s in SEEDS}
    base = {s: float(np.mean(np.abs(base_p[s] - y))) for s in SEEDS}
    gold = {s: float(np.mean(np.abs(gold_p[s] - y))) for s in SEEDS}
    d = {s: base[s] - gold[s] for s in SEEDS}
    dm = float(np.mean(list(d.values())))
    wins = int(sum(v > 0 for v in d.values()))
    return {"tag": tag, "n": len(y),
            "base_mae_mean": float(np.mean(list(base.values()))),
            "gold_mae_mean": float(np.mean(list(gold.values()))),
            "delta_mean": dm, "delta_std": float(np.std(list(d.values()))),
            "seeds_win": wins, "thresh": THRESH,
            "passed": bool(dm > THRESH and wins == 10),
            "per_seed_delta": {str(s): d[s] for s in SEEDS}}


def nested_cv(Xb, Xg, y, g, seed0=42):
    """外层 GroupKFold(5); 内层 GroupKFold(3) 小网格选参 (depth/lr), 外层复核。
    基线臂用固定 XGB_PARAMS 同折对照 (与 scripts/12 口径一致)。"""
    grid = [dict(learning_rate=lr, max_depth=md, n_estimators=400, subsample=0.8)
            for lr in (0.03, 0.05, 0.08) for md in (5, 7)]
    rows = []
    for f, (tr, te) in enumerate(GroupKFold(5).split(Xg, y, g)):
        best, best_mae = None, np.inf
        for p in grid:
            pr = np.full(len(tr), np.nan)
            for itr, ite in GroupKFold(3).split(Xg[tr], y[tr], g[tr]):
                pr[ite] = fit_pred(Xg[tr][itr], y[tr][itr], Xg[tr][ite], seed0 + f, p)
            mae = float(np.mean(np.abs(pr - y[tr])))
            if mae < best_mae:
                best, best_mae = p, mae
        pg = fit_pred(Xg[tr], y[tr], Xg[te], seed0 + f, best)
        pb = fit_pred(Xb[tr], y[tr], Xb[te], seed0 + f)  # 固定参数基线同折
        rows.append({"outer_fold": f + 1, "best_params": best,
                     "outer_mae_gold": float(np.mean(np.abs(pg - y[te]))),
                     "outer_mae_base": float(np.mean(np.abs(pb - y[te])))})
    df = pd.DataFrame(rows)
    return {"per_fold": rows,
            "gold_mae_mean": float(df.outer_mae_gold.mean()),
            "base_mae_mean": float(df.outer_mae_base.mean()),
            "delta_mean": float((df.outer_mae_base - df.outer_mae_gold).mean()),
            "folds_win": int((df.outer_mae_gold < df.outer_mae_base).sum())}


def main():
    t0 = time.time()
    df = pd.read_csv(FEAT)
    new = pd.read_csv(NEW)
    ncol = [c for c in new.columns if c not in META]
    m = df.merge(new[["comp", "structure", "facet"] + ncol],
                 on=["comp", "structure", "facet"], how="inner")
    print(f"covered rows: {len(m)}/{len(df)} ({len(m)/len(df)*100:.1f}%), "
          f"covered comps: {m['comp'].nunique()}/{df['comp'].nunique()}", flush=True)

    report = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(),
              "coverage": {"n_covered": len(m), "n_total": len(df),
                           "frac": len(m) / len(df),
                           "comps_covered": int(m["comp"].nunique()),
                           "comps_total": int(df["comp"].nunique())},
              "features": ncol,
              "honesty": "条件性: 弛豫质量未经 DFT 锚定 (本会话无 DFT 能力); "
                         "评估限于弛豫覆盖子集 (行内同折同种子公平)"}

    y, g = m["energy_eV"].values, m["comp"].values
    Xb = m[V2_COLS].values
    F = m[ncol].values
    Xg = np.hstack([Xb, F])

    # ---- 闸门 (线A/线B, 覆盖子集) ----
    gate_base = {s: oof_mae(Xb, y, g, seed=s) for s in SEEDS}
    gate_gold = {s: oof_mae(Xg, y, g, seed=s) for s in SEEDS}
    dA = float(np.mean([gate_base[s] - gate_gold[s] for s in SEEDS]))
    Fi = ridge_infer(F, Xb, g)
    dB = float(np.mean([gate_base[s] - oof_mae(np.hstack([Xb, Fi]), y, g, seed=s)
                        for s in SEEDS]))
    r = dB / dA if dA > 0 else float("nan")
    verdict = "NO_GAIN" if dA <= THRESH else ("PASS" if r >= RETAIN else "LEAK")
    report["portability_gate"] = {"delta_A": dA, "delta_B": dB, "retention": r,
                                  "retain_thresh": RETAIN, "verdict": verdict,
                                  "note": "物理可即时计算性: 全部特征仅需结构弛豫后的几何+"
                                          "元素周期表量, 无需目标值/DFT -> 预期 PASS, 实测如上"}
    print("gate:", verdict, "dA=%.4f dB=%.4f r=%.2f" % (dA, dB, r), flush=True)

    # ---- 公平契约 (覆盖子集) ----
    report["contract_covered"] = contract_block(Xb, Xg, y, g, "covered_subset")
    print("contract(covered): Δ=%.4f wins=%d/10" % (
        report["contract_covered"]["delta_mean"],
        report["contract_covered"]["seeds_win"]), flush=True)

    # ---- 次级口径: 全表 NaN 填充 (稀释对照) ----
    mfull = df.merge(new[["comp", "structure", "facet"] + ncol],
                     on=["comp", "structure", "facet"], how="left")
    yf, gf = mfull["energy_eV"].values, mfull["comp"].values
    Xbf = mfull[V2_COLS].values
    Xgf = np.hstack([Xbf, mfull[ncol].values.astype(float)])
    report["contract_full_nan"] = contract_block(Xbf, Xgf, yf, gf, "full1836_nanfill")
    print("contract(full NaN): Δ=%.4f" % report["contract_full_nan"]["delta_mean"], flush=True)

    # ---- 3d 磁性子层 Δ (覆盖行内) ----
    has3d = m["comp"].apply(lambda c: bool(set(MAG3D) & set(re.findall(r"[A-Z][a-z]?", c)))).values
    if has3d.sum() > 30:
        report["mag3d_sublayer"] = contract_block(Xb[has3d], Xg[has3d], y[has3d],
                                                  g[has3d], "mag3d_covered")
        report["mag3d_sublayer"]["phase3_ref"] = 0.0173
        report["mag3d_sublayer"]["phase4_ref"] = 0.0109
        print("3d sublayer: n=%d Δ=%.4f" % (has3d.sum(),
              report["mag3d_sublayer"]["delta_mean"]), flush=True)
    else:
        report["mag3d_sublayer"] = {"skipped": True, "n": int(has3d.sum())}

    # ---- 嵌套 CV 复核 (覆盖子集) ----
    report["nested_cv_covered"] = nested_cv(Xb, Xg, y, g)
    print("nested: gold=%.4f base=%.4f folds_win=%d/5" % (
        report["nested_cv_covered"]["gold_mae_mean"],
        report["nested_cv_covered"]["base_mae_mean"],
        report["nested_cv_covered"]["folds_win"]), flush=True)

    # ---- 综合判定 ----
    c = report["contract_covered"]
    gate_ok = report["portability_gate"]["verdict"] == "PASS"
    nested_ok = report["nested_cv_covered"]["delta_mean"] > 0 and \
        report["nested_cv_covered"]["folds_win"] >= 4
    site_target = c["gold_mae_mean"] <= 0.090
    if c["passed"] and gate_ok and nested_ok:
        verdict_final = "v5_candidate_conditional"
    elif c["passed"] and nested_ok and not gate_ok:
        # 子集契约+嵌套过阈但闸门未过: 信号不可采纳, 记条件性子集信号 (非阳性)
        verdict_final = "subset_signal_gate_failed"
    elif (c["delta_mean"] > 0 and c["seeds_win"] >= 8) or site_target:
        verdict_final = "conditional_positive"
    else:
        verdict_final = "negative"
    if report["contract_full_nan"]["delta_mean"] <= 0 and verdict_final in (
            "v5_candidate_conditional", "conditional_positive"):
        verdict_final = "negative_full_table"  # 全表口径无增益 -> 不可宣称全局阳性
    report["verdict"] = {"label": verdict_final,
                         "conditional": True,
                         "condition": "弛豫质量未经 DFT 锚定; 覆盖率 %.0f%%" % (100 * len(m) / len(df)),
                         "site_oof_target_0.090": bool(site_target)}
    report["runtime_min"] = (time.time() - t0) / 60

    (OUT / "gate_contract_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
