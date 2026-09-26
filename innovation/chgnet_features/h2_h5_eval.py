# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/h2_h5_eval.py — 检查点 H2/H3/H4/H5 评估
协议: 与 innovation/portability_gate/gate.py 及 scripts/07 完全同口径
  - 基线: v3 36 个描述符 (V2_COLS)
  - 模型: XGBRegressor(lr=0.03, depth=7, n=400, subsample=0.8), GroupKFold(5) OOF
  - 种子: [0,1,2,7,13,42,99,123,2024,31337] (同折同种子, 公平契约)
判定:
  H3_contract: Δ OOF MAE > 0.005 eV 且 10/10 种子改善
  H2_gate    : 可移植性闸门 (线A=金标准改善 Δ_A, 线B=Ridge推断改善 Δ_B, r=Δ_B/Δ_A≥0.8 PASS)
  H4_targeted: 3d 磁性子层 (含 Fe/Co/Ni/Mn/Cr) Δ MAE ≥ 0.02 eV 即单独成论
  H5_loeo    : 逐元素留出 Mn/Fe/Co/Ni, 加特征后退化 < 0.02 eV
"""
import json, re, sys
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
CHG = OUT / "chgnet_features.csv"
MAG3D = ["Fe", "Co", "Ni", "Mn", "Cr"]

CHG_COLS = ["chg_surf_mag_mean", "chg_surf_mag_absmean", "chg_surf_mag_max",
            "chg_surf_mag_min", "chg_surf_mag_std", "chg_surf_mag_range",
            "chg_bulk_mag_absmean", "chg_surf_bulk_absmag_diff",
            "chg_energy_per_atom"]


def fit_predict(Xtr, ytr, Xte, seed):
    m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    df = pd.read_csv(FEAT)
    chg = pd.read_csv(CHG)
    df = df.merge(chg[["comp", "structure", "facet"] + CHG_COLS],
                  on=["comp", "structure", "facet"], how="inner")
    assert len(df) == 1836, len(df)
    Xb = df[V2_COLS].values
    F = df[CHG_COLS].values
    y = df["energy_eV"].values
    groups = df["comp"].values
    Xg = np.hstack([Xb, F])
    report = {}

    # ---- H3 公平契约 + H2 线A ----
    base = {s: oof_mae(Xb, y, groups, seed=s) for s in SEEDS}
    gold = {s: oof_mae(Xg, y, groups, seed=s) for s in SEEDS}
    deltas = {s: base[s] - gold[s] for s in SEEDS}
    dA = float(np.mean(list(deltas.values())))
    wins = int(sum(d > 0 for d in deltas.values()))
    report["H3_contract"] = {
        "base_mae_mean": float(np.mean(list(base.values()))),
        "gold_mae_mean": float(np.mean(list(gold.values()))),
        "delta_mean": dA, "delta_std": float(np.std(list(deltas.values()))),
        "seeds_win": wins, "thresh": THRESH,
        "passed": bool(dA > THRESH and wins == 10),
        "per_seed": {str(s): {"base": base[s], "gold": gold[s], "delta": deltas[s]} for s in SEEDS},
    }

    # ---- H2 线B 可移植性 ----
    Fi = ridge_infer(F, Xb, groups)
    dB_seeds = {s: base[s] - oof_mae(np.hstack([Xb, Fi]), y, groups, seed=s) for s in SEEDS}
    dB = float(np.mean(list(dB_seeds.values())))
    r = dB / dA if dA > 0 else float("nan")
    verdict = "NO_GAIN" if dA <= THRESH else ("PASS" if r >= RETAIN else "LEAK")
    report["H2_gate"] = {"delta_A": dA, "delta_B": dB, "retention": r,
                         "retain_thresh": RETAIN, "verdict": verdict}

    # ---- H4 3d 磁性子层 ----
    has3d = df["comp"].apply(lambda c: bool(set(MAG3D) & set(re.findall(r"[A-Z][a-z]?", c)))).values
    sub = df[has3d]
    Xb_s, Xg_s = Xb[has3d], Xg[has3d]
    y_s, g_s = y[has3d], groups[has3d]
    b_s = {s: oof_mae(Xb_s, y_s, g_s, seed=s) for s in SEEDS}
    g_s_ = {s: oof_mae(Xg_s, y_s, g_s, seed=s) for s in SEEDS}
    d4 = {s: b_s[s] - g_s_[s] for s in SEEDS}
    report["H4_targeted"] = {
        "n_subset": int(has3d.sum()),
        "base_mae_mean": float(np.mean(list(b_s.values()))),
        "gold_mae_mean": float(np.mean(list(g_s_.values()))),
        "delta_mean": float(np.mean(list(d4.values()))),
        "delta_std": float(np.std(list(d4.values()))),
        "seeds_win": int(sum(v > 0 for v in d4.values())),
        "thresh": 0.02,
        "passed": bool(np.mean(list(d4.values())) >= 0.02),
    }

    # ---- H5 LOEO (Mn/Fe/Co/Ni) ----
    loeo = {}
    for el in ["Mn", "Fe", "Co", "Ni"]:
        mask = df["comp"].apply(lambda c: el in re.findall(r"[A-Z][a-z]?", c)).values
        if mask.sum() == 0 or (~mask).sum() == 0:
            loeo[el] = {"skipped": True}
            continue
        for tag, X in [("base", Xb), ("gold", Xg)]:
            pred = fit_predict(X[~mask], y[~mask], X[mask], seed=42)
            loeo.setdefault(el, {})[tag] = float(np.mean(np.abs(pred - y[mask])))
        loeo[el]["n_test"] = int(mask.sum())
        loeo[el]["degradation"] = loeo[el]["gold"] - loeo[el]["base"]
    h5_pass = all(v.get("degradation", 0) < 0.02 for v in loeo.values() if not v.get("skipped"))
    report["H5_loeo"] = {"per_element": loeo, "degradation_thresh": 0.02,
                         "passed": bool(h5_pass),
                         "note": "CatHub 跨域线待主线引擎复核; 此处裁决组成级 LOEO"}

    (OUT / "h2_h5_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in report.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
