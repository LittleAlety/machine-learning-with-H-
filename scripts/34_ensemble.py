# -*- coding: utf-8 -*-
"""
34_ensemble.py — A-3：三族简单平均（XGBoost + LightGBM + CatBoost）

- 同折同种子 OOF 简单平均（禁元学习器/stacking）
- v3 84 特征全为数值列（facet/structure 为 ID 列不入模），无原生 categorical 可用；
  CatBoost 以数值接口训练（如实记录）。
- 公平契约：10 种子 OOF MAE 相对 v3 基线增益 >0.005 且 10/10 方向一致；嵌套 CV 复核
  （内层在 {XGB 单模, 三族平均} 二选一）。
输出：outputs/r14_a3_ensemble.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, XGB, SEEDS, load_v3, mae

OUT = ROOT / "outputs"
LGBM = dict(n_estimators=400, learning_rate=0.03, max_depth=7, subsample=0.8)
CB = dict(iterations=400, learning_rate=0.03, depth=7, subsample=0.8, verbose=0)


def fam_models(seed):
    return [XGBRegressor(random_state=seed, n_jobs=-1, **XGB),
            LGBMRegressor(random_state=seed, n_jobs=-1, **LGBM),
            CatBoostRegressor(random_state=seed, allow_writing_files=False, **CB)]


def oof_avg(X, y, groups, seed, ensemble=True):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        ms = fam_models(seed) if ensemble else fam_models(seed)[:1]
        preds = [m.fit(X[tr], y[tr]).predict(X[te]) for m in ms]
        p[te] = np.mean(preds, axis=0)
    return p


def main():
    df, feats = load_v3()
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values

    base_maes, ens_maes = [], []
    for s in SEEDS:
        base_maes.append(mae(oof_avg(X, y, groups, s, False), y))
        print(f"  seed {s} xgb {base_maes[-1]:.4f}", flush=True)
        ens_maes.append(mae(oof_avg(X, y, groups, s, True), y))
        print(f"  seed {s} ens {ens_maes[-1]:.4f}", flush=True)
    diffs = np.array(base_maes) - np.array(ens_maes)

    # 嵌套 CV：内层 {XGB 单模, 三族平均} 二选一
    nested = np.full(len(y), np.nan)
    picks = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        im = {}
        for ens in (False, True):
            p = np.full(len(tr), np.nan)
            for itr, ite in GroupKFold(4).split(X[tr], y[tr], groups[tr]):
                ms = fam_models(42) if ens else fam_models(42)[:1]
                p[ite] = np.mean([m.fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite])
                                  for m in ms], axis=0)
            im[ens] = mae(p, y[tr])
        best = min(im, key=im.get)
        picks.append("ens" if best else "xgb")
        ms = fam_models(42) if best else fam_models(42)[:1]
        nested[te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in ms], axis=0)
        print(f"  外层折内层选择 {picks[-1]} {im}", flush=True)
    nested_mae = mae(nested, y)

    adopt = bool(diffs.mean() > 0.005 and (diffs > 0).all())
    res = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "categorical_note": "v3 84 特征全数值，无原生 categorical；CatBoost 数值接口训练",
        "xgb_per_seed": base_maes, "ensemble_per_seed": ens_maes,
        "xgb_mean": float(np.mean(base_maes)),
        "ensemble_mean": float(np.mean(ens_maes)),
        "ensemble_std": float(np.std(ens_maes)),
        "gain": float(diffs.mean()), "direction_consistent": int((diffs > 0).sum()),
        "nested_cv": {"mae": nested_mae, "picks": picks},
        "verdict": {"adopt": adopt,
                    "reason": (f"{'采纳' if adopt else '不采纳'}三族平均："
                               f"增益 {diffs.mean():+.4f}（契约>0.005），"
                               f"方向 {(diffs>0).sum()}/10，嵌套 CV {nested_mae:.4f}")}}
    json.dump(res, open(OUT / "r14_a3_ensemble.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"[判定] {res['verdict']['reason']}")
    print("[写出] outputs/r14_a3_ensemble.json")


if __name__ == "__main__":
    main()
