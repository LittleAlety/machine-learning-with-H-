# -*- coding: utf-8 -*-
"""r14_common.py — plan-hb v2.0 r14 共享工具：数据加载、折协议、种子表、XGB 基线。"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"

ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]
XGB = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8)
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]


def load_v3():
    df = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    assert len(feats) == 84, len(feats)
    return df, feats


def oof_predict(X, y, groups, seed, make_model=None):
    """固定协议 GroupKFold(5, groups=comp) OOF 预测。"""
    if make_model is None:
        make_model = lambda: XGBRegressor(random_state=seed, n_jobs=-1, **XGB)
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
