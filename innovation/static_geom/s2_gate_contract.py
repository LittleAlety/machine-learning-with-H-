# -*- coding: utf-8 -*-
"""
innovation/static_geom/s2_gate_contract.py — S2 静态组: 闸门 + 公平契约 + 嵌套CV + 3d 子层

协议与 innovation/portability_gate/gate.py, continuous_local/r3_eval.py 完全同口径:
  XGB_PARAMS = lr0.03/depth7/n400/sub0.8, GroupKFold(5, comp), 10 种子
  基线 = v3 84 特征 (hstar_features_v3_84feat.csv, OOF 0.1134 eV 参考);
  金标准 = 84 + 10 静态特征。
  契约 (preregistered_phase5.yaml fair_contract): Δ OOF MAE > 0.005 且 10/10 种子改善,
  嵌套 CV 外层 GroupKFold(5) 复核。
覆盖率: 原型 slab 1836/1836 -> 静态组天然全覆盖, 主口径即全表 (无 NaN 稀释问题)。
判定 (preregistered_phase6.yaml stage_S2_static.adopt):
  闸门 PASS 且 Δ>0.005 + 10/10 -> v5.1 定向采纳; 否则阴性入档。
幂等: 重跑覆盖写 gate_contract.json。
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portability_gate"))
from gate import XGB_PARAMS, SEEDS, THRESH, RETAIN, ridge_infer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
NEW = OUT / "static_geom_features.csv"

MAG3D = ["Fe", "Co", "Ni", "Mn", "Cr"]
META = {"name", "comp", "structure", "facet"}


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
    base = {s: float(np.mean(np.abs(oof(Xb, y, g, s) - y))) for s in SEEDS}
    gold = {s: float(np.mean(np.abs(oof(Xg, y, g, s) - y))) for s in SEEDS}
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
    """外层 GroupKFold(5); 内层 GroupKFold(3) 小网格选参, 外层复核 (与 r3_eval 同口径)。"""
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
        pb = fit_pred(Xb[tr], y[tr], Xb[te], seed0 + f)
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
    v3_cols = df.columns[5:].tolist()  # 84 features
    assert len(v3_cols) == 84
    m = df.merge(new[["comp", "structure", "facet"] + ncol],
                 on=["comp", "structure", "facet"], how="inner")
    print(f"covered rows: {len(m)}/{len(df)} ({len(m)/len(df)*100:.1f}%), "
          f"comps: {m['comp'].nunique()}/{df['comp'].nunique()}", flush=True)

    report = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(),
              "stage": "phase6_S2_static_geom",
              "coverage": {"n_covered": len(m), "n_total": len(df),
                           "frac": len(m) / len(df),
                           "note": "静态组由未弛豫原型几何确定性生成, 天然全覆盖"},
              "features": ncol,
              "baseline": "v3 84 features, XGB lr0.03/depth7/n400/sub0.8, "
                          "GroupKFold(5,comp), 10 seeds",
              "prereg": ["config/preregistered_phase6.yaml stage_S2_static",
                         "config/preregistered_phase5.yaml fair_contract"]}

    y, g = m["energy_eV"].values, m["comp"].values
    Xb = m[v3_cols].values.astype(float)
    F = m[ncol].values.astype(float)
    Xg = np.hstack([Xb, F])

    # ---- 闸门 (全表; 构造性可移植, 仍实测) ----
    gate_base = {s: float(np.mean(np.abs(oof(Xb, y, g, s) - y))) for s in SEEDS}
    gate_gold = {s: float(np.mean(np.abs(oof(Xg, y, g, s) - y))) for s in SEEDS}
    dA = float(np.mean([gate_base[s] - gate_gold[s] for s in SEEDS]))
    Fi = ridge_infer(F, Xb, g)
    dB = float(np.mean([gate_base[s] - float(np.mean(np.abs(
        oof(np.hstack([Xb, Fi]), y, g, s) - y))) for s in SEEDS]))
    r = dB / dA if dA > 0 else float("nan")
    verdict = "NO_GAIN" if dA <= THRESH else ("PASS" if r >= RETAIN else "LEAK")
    report["portability_gate"] = {"delta_A": dA, "delta_B": dB, "retention": r,
                                  "retain_thresh": RETAIN, "verdict": verdict,
                                  "note": "静态组仅需未弛豫原型几何+周期表量 -> 构造性可移植, "
                                          "线B Ridge 推断仅为形式检验; 实测如上"}
    print("gate:", verdict, "dA=%.4f dB=%.4f r=%.2f" % (dA, dB, r), flush=True)

    # ---- 公平契约 (全表 84+10 vs 84) ----
    report["contract_full"] = contract_block(Xb, Xg, y, g, "full1836")
    print("contract: Δ=%.4f wins=%d/10" % (
        report["contract_full"]["delta_mean"],
        report["contract_full"]["seeds_win"]), flush=True)

    # ---- 3d 磁性子层 Δ ----
    has3d = m["comp"].apply(lambda c: bool(set(MAG3D) & set(re.findall(r"[A-Z][a-z]?", c)))).values
    if has3d.sum() > 30:
        report["mag3d_sublayer"] = contract_block(Xb[has3d], Xg[has3d], y[has3d],
                                                  g[has3d], "mag3d_full")
        report["mag3d_sublayer"]["phase3_ref"] = 0.0173
        report["mag3d_sublayer"]["phase4_ref"] = 0.0109
        report["mag3d_sublayer"]["phase5_relax_group_ref"] = -0.0082
        print("3d sublayer: n=%d Δ=%.4f" % (has3d.sum(),
              report["mag3d_sublayer"]["delta_mean"]), flush=True)
    else:
        report["mag3d_sublayer"] = {"skipped": True, "n": int(has3d.sum())}

    # ---- 嵌套 CV 复核 ----
    report["nested_cv"] = nested_cv(Xb, Xg, y, g)
    print("nested: gold=%.4f base=%.4f folds_win=%d/5" % (
        report["nested_cv"]["gold_mae_mean"],
        report["nested_cv"]["base_mae_mean"],
        report["nested_cv"]["folds_win"]), flush=True)

    # ---- 判定 ----
    c = report["contract_full"]
    gate_ok = report["portability_gate"]["verdict"] == "PASS"
    nested_ok = report["nested_cv"]["delta_mean"] > 0 and \
        report["nested_cv"]["folds_win"] >= 4
    adopt = c["passed"] and gate_ok and nested_ok
    report["verdict"] = {
        "label": "adopt_v5.1" if adopt else "negative",
        "gate_pass": bool(gate_ok),
        "contract_passed": bool(c["passed"]),
        "nested_ok": bool(nested_ok),
        "rule": "stage_S2_static.adopt: 静态组 PASS 且 Δ>0.005 + 10/10 (+嵌套CV复核) -> v5.1 定向采纳",
        "phase5_relax_group_contrast": (
            "对照: Phase 5 弛豫依赖组 (层距/位移/弛豫能降, 14% 覆盖子集) "
            "闸门 LEAK (Δ_A=0.0129, Δ_B=-0.0145, 保留率 -1.12, "
            "innovation/continuous_local/gate_contract_report.json); "
            "静态组为构造性可移植, 两者口径分离记录。")}
    report["runtime_min"] = (time.time() - t0) / 60

    (OUT / "gate_contract.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
