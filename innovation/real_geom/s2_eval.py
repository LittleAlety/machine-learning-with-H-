# -*- coding: utf-8 -*-
"""
innovation/real_geom/s2_eval.py — S4 评估：公平契约 + 嵌套CV + 3d 子层 + 位点级 + 闸门

协议（与 portability_gate/gate.py、continuous_local/r3_eval.py、site_aware/f2 同口径）：
  XGBoost lr=0.03/depth=7/n=400/sub=0.8；10 种子 [0,1,2,7,13,42,99,123,2024,31337]
  表面级：GroupKFold(5, comp)，84 维 v3 基线（fair_contract baseline 0.11374）；
          候选 = 18 维位点真实几何特征的两口径聚合
            mean18/min18（全位点均值/逐特征 min，部署可得，无标签耦合）
            mins18（min-energy 位点，金标准口径——仅用于闸门线 A）
          契约：Δ>0.005 且 10/10；嵌套 CV 外层 GroupKFold(5) 复核；3d 磁性子层 Δ
  位点级：开发集（site lockbox 276 comps 冻结，不触碰），
          GroupKFold(5, shuffle, rs=42) 同 F2/F3；对照 F3=0.1156（84+onehot），
          目标 ≤0.090；臂：84 / 84+onehot / 84+geom18 / 84+onehot+geom18
闸门：
  G1 特征可移植性（r3 线A/线B）：mean+min 36 维 ridge_infer 保留率 ≥0.8
  G2 最稳定位点选择耦合（F4 教训）：mins18 的 Δ_A vs 训练折 Ridge 预测能量选位点的 Δ_B，
     保留率 ≥0.8 方 PASS；判 LEAK 则 mins 口径不可采纳、仅 mean/min 口径可宣称
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portability_gate"))
from gate import XGB_PARAMS, SEEDS, THRESH, RETAIN  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
SITE = OUT / "real_geom_site_features.csv"
F0_JSON = ROOT / "innovation" / "site_aware" / "f0_site_lockbox.json"

MAG3D = ["Fe", "Co", "Ni", "Mn", "Cr"]
V3_ID = ["comp", "structure", "facet", "n_raw", "energy_eV"]
GEOM18 = ["site_cn", "site_gcn", "h_height_A",
          "nn_cn_mean", "nn_cn_min", "nn_cn_std",
          "nn_gcn_mean", "nn_gcn_min", "nn_gcn_std",
          "nn_en_mean", "nn_en_std", "nn_d_mean", "nn_d_std",
          "d12_relaxed_A", "d12_relax_pct",
          "surf_disp_mean_A", "surf_disp_max_A", "surf_rumpling_A"]


def fit_pred(Xtr, ytr, Xte, seed, params=None):
    m = XGBRegressor(random_state=seed, n_jobs=-1, **(params or XGB_PARAMS))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def oof(X, y, g, seed, shuffle=False):
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(5)
    if shuffle:  # 位点级口径同 F2/F3：GroupKFold(5, shuffle=True, rs=42)
        from sklearn.model_selection import GroupKFold as G
        gkf = G(n_splits=5, shuffle=True, random_state=42)
    for tr, te in gkf.split(X, y, g):
        pred[te] = fit_pred(X[tr], y[tr], X[te], seed)
    assert not np.isnan(pred).any()
    return pred


def mae10(X, y, g, shuffle=False, ret_preds=False):
    ps = {s: oof(X, y, g, s, shuffle) for s in SEEDS}
    maes = {s: float(np.mean(np.abs(ps[s] - y))) for s in SEEDS}
    if ret_preds:
        return maes, ps
    return maes


def contract_block(Xb, Xg, y, g, tag, shuffle=False):
    base = mae10(Xb, y, g, shuffle)
    gold = mae10(Xg, y, g, shuffle)
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
    grid = [dict(learning_rate=lr, max_depth=md, n_estimators=400, subsample=0.8)
            for lr in (0.03, 0.05, 0.08) for md in (5, 7)]
    rows = []
    for f, (tr, te) in enumerate(GroupKFold(5).split(Xg, y, g)):
        best, best_mae = None, np.inf
        for p in grid:
            pr = np.full(len(tr), np.nan)
            for itr, ite in GroupKFold(3).split(Xg[tr], y[tr], g[tr]):
                pr[ite] = fit_pred(Xg[tr][itr], y[tr][itr], Xg[tr][ite], seed0 + f, p)
            m = float(np.mean(np.abs(pr - y[tr])))
            if m < best_mae:
                best, best_mae = p, m
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


def ridge_infer(F, Xb, g):
    Fi = np.full_like(F, np.nan, dtype=float)
    for tr, te in GroupKFold(5).split(Xb, np.zeros(len(Xb)), g):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(Xb[tr], F[tr])
        Fi[te] = m.predict(Xb[te]).reshape(len(te), -1)
    assert not np.isnan(Fi).any()
    return Fi


def main():
    t0 = time.time()
    v3 = pd.read_csv(FEAT)
    site = pd.read_csv(SITE)
    lb = set(json.loads(F0_JSON.read_text())["lockbox_comps"])
    v3cols = [c for c in v3.columns if c not in V3_ID]
    assert len(v3cols) == 84

    # ---------------- 表面级聚合（两口径） ----------------
    g = site.groupby("comp")
    mean18 = g[GEOM18].mean().add_prefix("rg_mean_")
    min18 = g[GEOM18].min().add_prefix("rg_min_")
    idx_min_e = g["energy_eV"].idxmin()
    mins18 = site.loc[idx_min_e].set_index("comp")[GEOM18].add_prefix("rg_mins_")

    surf = v3.set_index("comp").join([mean18, min18, mins18])
    cov = surf["rg_mean_site_cn"].notna()
    print(f"surface coverage: {int(cov.sum())}/{len(surf)} comps", flush=True)
    surfc = surf[cov].reset_index()     # 行内公平：覆盖子集（1832/1836）
    y, grp = surfc["energy_eV"].values, surfc["comp"].values
    Xb = surfc[v3cols].values.astype(float)
    F_mean = surfc[[c for c in surfc.columns if c.startswith("rg_mean_")]].values
    F_min = surfc[[c for c in surfc.columns if c.startswith("rg_min_")]].values
    F_mins = surfc[[c for c in surfc.columns if c.startswith("rg_mins_")]].values
    F_mm = np.hstack([F_mean, F_min])   # 主候选：均值+min 36 维（部署可得）

    report = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(),
              "protocol": {"xgb": XGB_PARAMS, "seeds": SEEDS,
                           "folds_surface": "GroupKFold(5, comp)",
                           "folds_site": "GroupKFold(5, shuffle, rs=42)",
                           "baseline": "84 v3, fair_contract baseline 0.11374"},
              "coverage": {"surface_comps_covered": int(cov.sum()),
                           "surface_comps_total": int(len(surf)),
                           "site_records_parsed": int(len(site))},
              "features_18": GEOM18}

    print("== surface contracts ==", flush=True)
    report["contract_mean18"] = contract_block(Xb, np.hstack([Xb, F_mean]), y, grp, "84+mean18")
    print("mean18 Δ=%.4f wins=%d" % (report["contract_mean18"]["delta_mean"],
          report["contract_mean18"]["seeds_win"]), flush=True)
    report["contract_min18"] = contract_block(Xb, np.hstack([Xb, F_min]), y, grp, "84+min18")
    print("min18  Δ=%.4f wins=%d" % (report["contract_min18"]["delta_mean"],
          report["contract_min18"]["seeds_win"]), flush=True)
    report["contract_meanmin36"] = contract_block(Xb, np.hstack([Xb, F_mm]), y, grp,
                                                  "84+mean18+min18 (main)")
    print("mm36   Δ=%.4f wins=%d" % (report["contract_meanmin36"]["delta_mean"],
          report["contract_meanmin36"]["seeds_win"]), flush=True)

    # ---- 3d 磁性子层 ----
    has3d = surfc["comp"].apply(
        lambda c: bool(set(MAG3D) & set(re.findall(r"[A-Z][a-z]?", c)))).values
    if has3d.sum() > 30:
        report["mag3d_sublayer"] = contract_block(
            Xb[has3d], np.hstack([Xb, F_mm])[has3d], y[has3d], grp[has3d], "mag3d_mm36")
        report["mag3d_sublayer"]["phase3_magnetic_moment_ref"] = 0.0173
        print("3d sublayer n=%d Δ=%.4f wins=%d" % (has3d.sum(),
              report["mag3d_sublayer"]["delta_mean"],
              report["mag3d_sublayer"]["seeds_win"]), flush=True)

    # ---- 嵌套 CV（主候选） ----
    report["nested_cv_meanmin36"] = nested_cv(Xb, np.hstack([Xb, F_mm]), y, grp)
    print("nested: gold=%.4f base=%.4f folds_win=%d/5" % (
        report["nested_cv_meanmin36"]["gold_mae_mean"],
        report["nested_cv_meanmin36"]["base_mae_mean"],
        report["nested_cv_meanmin36"]["folds_win"]), flush=True)

    # ---- 闸门 G1：特征可移植性（mean+min 36 维 ridge 推断保留率） ----
    gate_base = mae10(Xb, y, grp)
    gate_gold = mae10(np.hstack([Xb, F_mm]), y, grp)
    dA1 = float(np.mean([gate_base[s] - gate_gold[s] for s in SEEDS]))
    Fi = ridge_infer(F_mm, Xb, grp)
    gate_inf = mae10(np.hstack([Xb, Fi]), y, grp)
    dB1 = float(np.mean([gate_base[s] - gate_inf[s] for s in SEEDS]))
    r1 = dB1 / dA1 if dA1 > 0 else float("nan")
    report["gate_G1_feature_portability"] = {
        "delta_A": dA1, "delta_B": dB1, "retention": r1, "retain_thresh": RETAIN,
        "verdict": "NO_GAIN" if dA1 <= THRESH else ("PASS" if r1 >= RETAIN else "LEAK")}
    print("G1: %s dA=%.4f dB=%.4f r=%.2f" % (
        report["gate_G1_feature_portability"]["verdict"], dA1, dB1, r1), flush=True)

    # ---- 闸门 G2：最稳定位点选择耦合（F4 教训：min-energy 选位点用标签 → 须可移植检验） ----
    # 线 A：标签选 min-energy 位点的 mins18
    dA2_gold = mae10(np.hstack([Xb, F_mins]), y, grp)
    dA2 = float(np.mean([gate_base[s] - dA2_gold[s] for s in SEEDS]))
    # 线 B：逐外折用训练折 Ridge(84+geom18 → 位点能量) 预测测试折各位点能量，
    #        取预测能量 min 位点的特征（部署时标签不可见的可移植口径）
    dev_site = site[site["comp"].isin(set(surfc["comp"]))].copy()
    dev_site = dev_site.merge(v3[["comp"] + v3cols], on="comp", how="left",
                              validate="many_to_one")
    Xs = np.hstack([dev_site[v3cols].values.astype(float),
                    dev_site[GEOM18].values.astype(float)])
    ys = dev_site["energy_eV"].values
    gs = dev_site["comp"].values
    comps = surfc["comp"].values
    comp_pos = {c: i for i, c in enumerate(comps)}
    best_e = np.full(len(comps), np.inf)
    sel_feat = np.full((len(comps), 18), np.nan)
    for tr_c, te_c in GroupKFold(5).split(Xb, y, grp):
        te_comps = set(comps[te_c])
        te_mask = np.array([c in te_comps for c in gs])
        rmod = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        rmod.fit(Xs[~te_mask], ys[~te_mask])
        pred_e = rmod.predict(Xs[te_mask])
        rows_te = np.where(te_mask)[0]
        for j, p in zip(rows_te, pred_e):
            ci = comp_pos[gs[j]]
            if p < best_e[ci]:
                best_e[ci] = p
                sel_feat[ci] = dev_site[GEOM18].values[j]
    assert not np.isnan(sel_feat).any(), "有 comp 未被任何测试折覆盖"
    dB2_inf = mae10(np.hstack([Xb, sel_feat]), y, grp)
    dB2 = float(np.mean([gate_base[s] - dB2_inf[s] for s in SEEDS]))
    r2 = dB2 / dA2 if dA2 > 0 else float("nan")
    report["gate_G2_minsite_selection"] = {
        "delta_A_label_selected": dA2, "delta_B_ridge_selected": dB2,
        "retention": r2, "retain_thresh": RETAIN,
        "verdict": "NO_GAIN" if dA2 <= THRESH else ("PASS" if r2 >= RETAIN else "LEAK"),
        "note": "F4 教训：min-energy 位点选择是标签耦合操作；线 B 用训练折 Ridge "
                "预测位点能量选位点，模拟部署可得性。判 LEAK 则 mins 口径不可采纳。"}
    print("G2: %s dA=%.4f dB=%.4f r=%.2f" % (
        report["gate_G2_minsite_selection"]["verdict"], dA2, dB2, r2), flush=True)

    # ---------------- 位点级评估（开发集，lockbox 冻结不触碰） ----------------
    print("== site-level ==", flush=True)
    sdev = site[~site["comp"].isin(lb)].copy()
    sdev = sdev.merge(v3[["comp"] + v3cols], on="comp", how="left",
                      validate="many_to_one")
    ys_, gs_ = sdev["energy_eV"].values, sdev["comp"].values
    # 与 F3 同口径对照臂：join F1 的 site_full 元数据（record 级，97.8% 覆盖；
    # 未匹配记录 one-hot 全 0，行内公平）
    f1 = pd.read_csv(ROOT / "innovation" / "site_aware" / "site_dataset_dev.csv",
                     usecols=["record_id", "site_full"])
    sdev = sdev.merge(f1, on="record_id", how="left")
    cats = sorted(f1["site_full"].unique())
    cat_pos = {c: i for i, c in enumerate(cats)}
    OH = np.zeros((len(sdev), len(cats)))
    for i, sf in enumerate(sdev["site_full"]):
        if isinstance(sf, str):
            OH[i, cat_pos[sf]] = 1.0
    X84 = sdev[v3cols].values.astype(float)
    Xg18 = sdev[GEOM18].values.astype(float)
    X_site_arms = {
        "base84": X84,
        "onehot28_F3replica": np.hstack([X84, OH]),
        "geom18": np.hstack([X84, Xg18]),
        "geom18_plus_onehot": np.hstack([X84, OH, Xg18]),
    }
    site_rep = {"n_dev_records": int(len(sdev)), "n_dev_comps": int(len(set(gs_))),
                "F3_ref_site_oof_mae": 0.1155946, "F3_ref_onehot_gain": 0.1667604,
                "F3_target": 0.090, "realization_rate_ref_F5": 0.1466,
                "arms": {}}
    for tag, Xa in X_site_arms.items():
        maes, preds = mae10(Xa, ys_, gs_, shuffle=True, ret_preds=True)
        site_rep["arms"][tag] = {
            "oof_mae_10seed_mean": float(np.mean(list(maes.values()))),
            "oof_mae_per_seed": {str(s): maes[s] for s in SEEDS},
            "target_0.090_met": bool(np.mean(list(maes.values())) <= 0.090)}
        print("site arm %s: OOF=%.4f" % (tag, site_rep["arms"][tag]["oof_mae_10seed_mean"]),
              flush=True)
    # 位点→表面 min 聚合兑现率（开发集）：geom18 臂逐位点 OOF 预测 min 聚合 vs v3 标签
    _, preds_g = mae10(np.hstack([X84, Xg18]), ys_, gs_, shuffle=True, ret_preds=True)
    v3dev = v3[v3["comp"].isin(set(gs_))].set_index("comp")
    mins_pred = pd.DataFrame({"comp": gs_, "pred": preds_g[SEEDS[0]]}).groupby("comp")["pred"].min()
    common = mins_pred.index.intersection(v3dev.index)
    surf_min_mae = float(np.mean(np.abs(mins_pred[common].values
                                        - v3dev.loc[common, "energy_eV"].values)))
    site_rep["min_agg_surface_mae_seed0"] = surf_min_mae
    site_rep["F5_ref_site_min_surface_mae"] = 0.1093577
    report["site_level"] = site_rep

    # ---------------- 综合判定 ----------------
    c = report["contract_meanmin36"]
    g1 = report["gate_G1_feature_portability"]["verdict"] == "PASS"
    nested_ok = report["nested_cv_meanmin36"]["delta_mean"] > 0 and \
        report["nested_cv_meanmin36"]["folds_win"] >= 4
    if c["passed"] and g1 and nested_ok:
        verdict = "v5_candidate"
        cond = "契约+G1闸门+嵌套CV 全过（真实 DFT 几何，无 CHGNet 伪影条款）"
    elif c["passed"] and nested_ok:
        verdict = "contract_passed_gate_failed"
        cond = "契约与嵌套过阈但 G1 闸门未过——信号不可移植，不可采纳"
    elif c["delta_mean"] > 0 and c["seeds_win"] >= 8:
        verdict = "conditional_positive"
        cond = "方向一致但未达契约全条款"
    elif c["delta_mean"] > 0:
        verdict = "weak_positive"
        cond = "均值改善但种子一致性不足"
    else:
        verdict = "negative"
        cond = "无增益"
    report["verdict"] = {"label": verdict, "condition": cond,
                         "mins18_usable": report["gate_G2_minsite_selection"]["verdict"] == "PASS",
                         "site_target_met": site_rep["arms"]["geom18"]["target_0.090_met"]}
    report["runtime_min"] = (time.time() - t0) / 60
    (OUT / "real_geom_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
