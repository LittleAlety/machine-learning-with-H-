# -*- coding: utf-8 -*-
"""
31_multisplit_eval.py — B-2：多划分稳健性（lockbox 口径，4 个划分种子）

- 划分种子 42（=lockbox_v1）+ 1/2/3，同协议（结构分层 × comp 分组 15%）
- v3 全特征：每个划分上，用训练部分（85%）全量训练 XGB(seed=42)，在 15% 上评 MAE
- 报 4 划分 MAE 分布 mean±std + bootstrap CI（10,000 次，种子 42）
- 与分组 CV（10 种子 OOF 0.11374）对照；若 lockbox 口径显著差（>0.125 eV）记录 OOF 乐观偏差
输出：splits/lockbox_multi_eval.json / .csv；splits/lockbox_seed{1,2,3}.json
注：本步只用 lockbox 行做"划分口径"评估，不在其上调任何参/选任何型（纪律保持）。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, DP, XGB, load_v3
from importlib import import_module
m30 = import_module("30_lockbox_split") if False else None


def make_split(df, split_seed, frac=0.15):
    rng = np.random.RandomState(split_seed)
    lock = []
    for st, sub in df.groupby("structure"):
        comps = np.sort(sub["comp"].unique())
        n = max(1, int(round(len(comps) * frac)))
        lock.extend(rng.choice(comps, size=n, replace=False).tolist())
    return sorted(lock)


def main():
    df, feats = load_v3()
    X, y = df[feats].values.astype(float), df["energy_eV"].values
    sdir = ROOT / "splits"
    sdir.mkdir(exist_ok=True)

    rows = []
    for seed in [42, 1, 2, 3]:
        f = sdir / f"lockbox_seed{seed}.json" if seed != 42 else sdir / "lockbox_v1.json"
        if f.exists():
            lock = json.load(open(f))["lockbox_comps"]
        else:
            lock = make_split(df, seed)
            payload = {"version": f"lockbox_seed{seed}",
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "protocol": {"stratify": "structure", "group": "comp",
                                    "fraction": 0.15, "split_seed": seed},
                       "n_lockbox_comps": len(lock), "lockbox_comps": lock,
                       "sha256_of_comp_list": hashlib.sha256(
                           json.dumps(lock, sort_keys=True).encode()).hexdigest()}
            json.dump(payload, open(f, "w"), ensure_ascii=False, indent=2)
        is_lock = df["comp"].isin(lock).values
        mdl = XGBRegressor(random_state=42, n_jobs=-1, **XGB)
        mdl.fit(X[~is_lock], y[~is_lock])
        pred = mdl.predict(X[is_lock])
        mae = mean_absolute_error(y[is_lock], pred)
        rows.append({"split_seed": seed, "n_lockbox": int(is_lock.sum()),
                     "MAE": mae})
        print(f"[seed {seed}] lockbox n={is_lock.sum()}  MAE={mae:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(sdir / "lockbox_multi_eval.csv", index=False)
    maes = res["MAE"].values
    rng = np.random.RandomState(42)
    boots = np.array([rng.choice(maes, size=len(maes), replace=True).mean()
                      for _ in range(10000)])
    summ = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "v3 84feat XGB(seed42) 训练于 85%，评于 15% lockbox；仅评估，不调参选型",
        "per_split": rows,
        "mae_mean": float(maes.mean()), "mae_std": float(maes.std(ddof=1)),
        "bootstrap_ci95_mean": [float(np.quantile(boots, 0.025)),
                                float(np.quantile(boots, 0.975))],
        "group_cv_10seed_oof_mae": 0.11374,
    }
    flag = bool(maes.mean() > 0.125)
    summ["oof_optimism_flag"] = flag
    if flag:
        summ["note"] = ("lockbox 口径显著差于分组 CV（>0.125 eV）："
                        "OOF 存在乐观偏差，幅度 %.4f eV" % (maes.mean() - 0.11374))
    json.dump(summ, open(sdir / "lockbox_multi_eval.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"[汇总] 4 划分 MAE {maes.mean():.4f}±{maes.std(ddof=1):.4f}  "
          f"CI95 {summ['bootstrap_ci95_mean']}  oof_optimism_flag={flag}")
    print("[写出] splits/lockbox_multi_eval.{csv,json}")


if __name__ == "__main__":
    main()
