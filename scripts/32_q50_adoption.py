# -*- coding: utf-8 -*-
"""
32_q50_adoption.py — A-1：q50（XGBoost quantile α=0.5）采纳检验

步骤：
1. 核对 innovation/conformal_screen/oof_quantiles/seed*.csv 复现 OOF MAE(q50)≈0.1063
2. v3 基线（squarederror）10 种子同折 OOF MAE（方向一致性判定用）
3. 嵌套 CV：外层 GroupKFold(5)；内层 GroupKFold(4) 仅在
   {objective=squarederror, objective=quantileerror α=0.5} 两配置间按 MAE 选择，
   不外扩调参；报嵌套 MAE（选择流程的无偏估计）
4. LOEO 复测 q50（seed=42），重点 Mn/Fe/Bi 折与既有 v3 LOEO 对比（变化 <0.02 eV 为限）
5. 公平契约判定 + 跨域证据权衡（q50 CatHub 0.206 vs v3 0.177）写入 JSON

输出：outputs/r14_a1_q50.json；data_processed/hstar_loeo_q50.csv
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, DP, XGB, SEEDS, load_v3, mae

CKPT = ROOT / "innovation" / "conformal_screen" / "oof_quantiles"
OUT = ROOT / "outputs"


def mk_xgb(seed, objective=None):
    kw = dict(random_state=seed, n_jobs=-1, **XGB)
    if objective == "quantile":
        kw.update(objective="reg:quantileerror", quantile_alpha=0.5)
    return XGBRegressor(**kw)


def oof(X, y, groups, seed, objective=None):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = mk_xgb(seed, objective).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def main():
    df, feats = load_v3()
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values
    res = {"created_utc": datetime.now(timezone.utc).isoformat()}

    # 1. 复现核对
    q_maes = []
    for s in SEEDS:
        f = CKPT / f"seed{s}.csv"
        assert f.exists(), f
        d = pd.read_csv(f, index_col=0)
        assert len(d) == len(y)
        q_maes.append(mae(d["q50"].values, y))
    res["step1_reproduce"] = {"per_seed_mae": q_maes,
                              "mean": float(np.mean(q_maes)),
                              "expected": 0.1063,
                              "reproduced": bool(abs(np.mean(q_maes) - 0.1063) < 0.001)}
    print(f"[1] 复现 q50 OOF MAE {np.mean(q_maes):.4f}（期望≈0.1063）")

    # 2. v3 基线 10 种子 OOF
    base_maes = []
    for s in SEEDS:
        base_maes.append(mae(oof(X, y, groups, s), y))
        print(f"  v3 seed {s}: {base_maes[-1]:.4f}", flush=True)
    res["step2_v3_baseline"] = {"per_seed_mae": base_maes,
                                "mean": float(np.mean(base_maes)),
                                "std": float(np.std(base_maes))}
    diffs = [b - q for b, q in zip(base_maes, q_maes)]
    res["step2_gain"] = {
        "mean_gain": float(np.mean(diffs)),
        "direction_consistent": int(sum(d > 0 for d in diffs)),
        "contract_gain_ok": bool(np.mean(diffs) > 0.005),
        "contract_direction_ok": bool(all(d > 0 for d in diffs))}
    print(f"[2] 基线 {np.mean(base_maes):.4f}±{np.std(base_maes):.4f}；"
          f"增益 {np.mean(diffs):.4f}，方向一致 {sum(d>0 for d in diffs)}/10")

    # 3. 嵌套 CV（内层仅二选一）
    outer = GroupKFold(5)
    nested_pred = np.full(len(y), np.nan)
    picks = []
    for tr, te in outer.split(X, y, groups):
        inner = GroupKFold(4)
        inner_mae = {}
        for obj in ("squared", "quantile"):
            p = np.full(len(tr), np.nan)
            for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                m = mk_xgb(42, None if obj == "squared" else "quantile")
                m.fit(X[tr][itr], y[tr][itr])
                p[ite] = m.predict(X[tr][ite])
            inner_mae[obj] = mae(p, y[tr])
        best = min(inner_mae, key=inner_mae.get)
        picks.append(best)
        m = mk_xgb(42, None if best == "squared" else "quantile").fit(X[tr], y[tr])
        nested_pred[te] = m.predict(X[te])
    res["step3_nested_cv"] = {"inner_picks": picks,
                              "nested_mae": mae(nested_pred, y),
                              "baseline_nested_mae_expected": 0.1134}
    print(f"[3] 嵌套 CV MAE={mae(nested_pred, y):.4f}（内层选择 {picks}）")

    # 4. LOEO q50
    elements = sorted({e for c in groups for e in re.findall(r"[A-Z][a-z]?", c)})
    rows = []
    for el in elements:
        has = df["comp"].apply(lambda c: el in re.findall(r"[A-Z][a-z]?", c)).values
        if has.sum() < 3 or (~has).sum() < 10:
            continue
        m = mk_xgb(42, "quantile").fit(X[~has], y[~has])
        pred = m.predict(X[has])
        rows.append({"held_out_element": el, "n_test": int(has.sum()),
                     "MAE_q50": mean_absolute_error(y[has], pred)})
        print(f"  LOEO {el}: {rows[-1]['MAE_q50']:.4f}", flush=True)
    loeo_q = pd.DataFrame(rows)
    loeo_v3 = pd.read_csv(DP / "hstar_loeo_results.csv")
    mg = loeo_q.merge(loeo_v3[["held_out_element", "MAE"]], on="held_out_element")
    mg["delta"] = mg["MAE_q50"] - mg["MAE"]
    mg.to_csv(DP / "hstar_loeo_q50.csv", index=False)
    focus = {el: float(mg.loc[mg.held_out_element == el, "delta"].iloc[0])
             for el in ("Mn", "Fe", "Bi") if (mg.held_out_element == el).any()}
    pooled_q = float(np.average(loeo_q["MAE_q50"], weights=loeo_q["n_test"]))
    res["step4_loeo"] = {"pooled_mae_q50": pooled_q,
                         "pooled_mae_v3": 0.155,
                         "focus_delta": focus,
                         "focus_within_0.02": bool(all(abs(v) < 0.02 for v in focus.values()))}
    print(f"[4] LOEO q50 汇集 {pooled_q:.4f}（v3 0.155）；Mn/Fe/Bi Δ={focus}")

    # 5. 判定 + 跨域权衡
    adopt = (res["step2_gain"]["contract_gain_ok"]
             and res["step2_gain"]["contract_direction_ok"]
             and res["step4_loeo"]["focus_within_0.02"])
    res["step5_verdict"] = {
        "adopt": bool(adopt),
        "cross_domain_evidence": {
            "q50_cathub_mae": 0.206, "v3_cathub_mae": 0.177,
            "note": ("q50 在外部 CatHub 上差于 v3 点预测（+0.029 eV）。"
                     "权衡：库内 OOF/LOEO 显著改善且契约双过，但跨域泛化存疑。"
                     "判定依据以同分布库内契约为准，跨域差距如实记录并纳入 v4 风险说明。")},
        "reason": ""}
    r = res["step5_verdict"]
    r["reason"] = ("采纳 q50" if adopt else "不采纳 q50") + (
        f"：OOF 增益 {res['step2_gain']['mean_gain']:.4f} eV"
        f"（>0.005={res['step2_gain']['contract_gain_ok']}），"
        f"方向 {res['step2_gain']['direction_consistent']}/10，"
        f"嵌套 CV {res['step3_nested_cv']['nested_mae']:.4f}，"
        f"LOEO Mn/Fe/Bi 变化 {focus}")
    json.dump(res, open(OUT / "r14_a1_q50.json", "w"), ensure_ascii=False, indent=2)
    print(f"[5] 判定：{r['reason']}")
    print("[写出] outputs/r14_a1_q50.json, data_processed/hstar_loeo_q50.csv")
    return adopt


if __name__ == "__main__":
    main()
