# -*- coding: utf-8 -*-
"""innovation/probes/_common.py — Workstream P 误差解剖探针公共模块

协议（预注册，禁止改动）：
- 特征表 data_processed/hstar_features_v3_84feat.csv；ID 列 = comp/structure/facet/n_raw/energy_eV，
  其余 84 列为特征；目标 = energy_eV（同表面多构型取最稳定位点 E_ads 最低值）
- 折协议：GroupKFold(n_splits=5, shuffle=True, random_state=42)，groups=规范化 comp，禁止重切
- 基线 XGBoost：lr=0.03, max_depth=7, n_estimators=400, subsample=0.8（OOF MAE≈0.1134 eV）
- 种子表：SEEDS10 = (0,1,2,7,13,42,99,123,2024,31337)
- 图表：300dpi、暖色系 #B35C24/#DFB27E/#7A4A2B、去右上 spine、英文标注
- 探针结果仅用于诊断，禁止用于选择交付配置
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DP = ROOT / "data_processed"
PROBES = ROOT / "innovation" / "probes"
FEAT_CSV = DP / "hstar_features_v3_84feat.csv"
RAW_GZ = ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz"

ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]
XGB_BASE = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8)
SEEDS10 = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
SEEDS3 = [0, 1, 2]

C_MAIN, C_ALT, C_3RD = "#B35C24", "#DFB27E", "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_data():
    df = pd.read_csv(FEAT_CSV)
    feats = [c for c in df.columns if c not in ID_COLS]
    assert len(feats) == 84, f"特征数 {len(feats)} != 84"
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values.astype(float)
    groups = df["comp"].values
    return df, X, y, groups, feats


def gkf_splits(X, y, groups):
    """固定折协议：GroupKFold(5, shuffle=True, random_state=42)。"""
    return list(GroupKFold(n_splits=5, shuffle=True,
                           random_state=42).split(X, y, groups))


def oof_pred(X, y, groups, seed, xgb_params=None):
    """固定折 OOF 预测 + 逐折 train 预测（用于 train MAE）。"""
    params = dict(XGB_BASE)
    if xgb_params:
        params.update(xgb_params)
    pred = np.full(len(y), np.nan)
    tr_pred = np.full(len(y), np.nan)
    for tr, te in gkf_splits(X, y, groups):
        m = XGBRegressor(random_state=seed, n_jobs=-1, **params)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        tr_pred[tr] = m.predict(X[tr])
    return pred, tr_pred


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
