# -*- coding: utf-8 -*-
"""
innovation/portability_gate/gate.py — 可移植性闸门（Portability Gate）v1
把 3.9 节的内部泄漏审查做法形式化为通用协议：
任何候选特征须通过"未见组成可计算性检验"方可进入模型。

判定逻辑（三态）：
  线 A（泄漏线）：金标准特征入模，OOF MAE 改善 Δ_A 是否过公平契约阈值（0.005 eV）
  线 B（可移植线）：特征改为"仅用训练折可见信息推断"（Ridge: 特征 ~ 既有描述符），
                   OOF 改善 Δ_B 的保留率 r = Δ_B/Δ_A 是否 ≥ 0.8
  PASS    : Δ_A 过阈 且 r ≥ 0.8
  LEAK    : Δ_A 过阈 且 r < 0.8（改善在不可移植口径下崩塌 → 泄漏/元数据耦合）
  NO_GAIN : Δ_A 未过阈
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
FEAT_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
V1_CSV = ROOT / "data_processed" / "hstar_dataset_v1.csv"
OUT = Path(__file__).resolve().parent

XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8)
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
THRESH = 0.005      # 公平契约采纳阈值 (eV)
RETAIN = 0.8        # 可移植保留率阈值

V2_COLS = ([f"{p}_{s}" for p in ["en", "radius", "group", "period", "ie1", "d_el"]
            for s in ["wmean", "max", "min", "range", "wstd"]]
           + ["structure_A1", "structure_L10", "structure_L12", "facet_101", "facet_111",
              "n_elements"])
assert len(V2_COLS) == 36


def load_base():
    df = pd.read_csv(FEAT_CSV)
    X = df[V2_COLS].values
    y = df["energy_eV"].values
    groups = df["comp"].values
    return df, X, y, groups


def oof_mae(X, y, groups, seed=42):
    """GroupKFold(5) OOF MAE，与 scripts/07 同协议。"""
    gkf = GroupKFold(n_splits=5)
    pred = np.full_like(y, np.nan, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(np.mean(np.abs(pred - y)))


def ridge_infer(F, X_base, groups):
    """逐折：仅用训练折拟合 特征~既有描述符 的 Ridge，在测试折生成推断值（模拟部署可得性）。"""
    gkf = GroupKFold(n_splits=5)
    Fi = np.full_like(F, np.nan, dtype=float)
    for tr, te in gkf.split(X_base, np.zeros(len(X_base)), groups):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X_base[tr], F[tr])
        Fi[te] = m.predict(X_base[te]).reshape(len(te), -1)
    assert not np.isnan(Fi).any()
    return Fi


def portability_check(name, F, X_base, y, groups, seeds=SEEDS, thresh=THRESH, retain=RETAIN):
    """F: (n, k) 候选特征（金标准口径）。返回 GateReport dict。"""
    base = {s: oof_mae(X_base, y, groups, seed=s) for s in seeds}
    gold = {s: oof_mae(np.hstack([X_base, F]), y, groups, seed=s) for s in seeds}
    dA = float(np.mean([base[s] - gold[s] for s in seeds]))
    dA_std = float(np.std([base[s] - gold[s] for s in seeds]))
    seeds_win = int(sum(base[s] - gold[s] > 0 for s in seeds))
    Fi = ridge_infer(F, X_base, groups)
    inf_mae = oof_mae(np.hstack([X_base, Fi]), y, groups, seed=42)
    dB = float(base[42] - inf_mae)
    r = float(dB / dA) if abs(dA) > 1e-12 else 0.0
    if dA <= thresh:
        verdict = "NO_GAIN"
    else:
        verdict = "PASS" if r >= retain else "LEAK"
    return {
        "feature": name, "k_dims": int(F.shape[1]),
        "mae_base_10seeds": float(np.mean(list(base.values()))),
        "mae_gold_10seeds": float(np.mean(list(gold.values()))),
        "delta_A": dA, "delta_A_std": dA_std, "seeds_win": f"{seeds_win}/10",
        "delta_B_inferred": dB, "retention": r,
        "verdict": verdict,
    }


if __name__ == "__main__":
    df, X, y, groups = load_base()
    print(f"[gate] 基线装载: {X.shape}, 36 维 v2 描述符")
