# -*- coding: utf-8 -*-
"""F2/F3 位点级 XGBoost 模型 — plan-hb v3.0 Workstream F

预注册检查点（写死，不得中途解释）：
  F3 精度目标：位点级 OOF MAE ≤ 0.090 eV
    （锚点：v3 表面级 0.1134 eV − 信息侧红利 0.0391 eV → 合理区间 0.077–0.090；
      若位点级 OOF MAE < 0.077 eV 视为超出红利区间的"改善"，记泄漏嫌疑并分析）
  F2 口径一致性：
    (a) 数据级：各表面 mean(位点能量) − min(位点能量) 的中位数 ≈ 0.21 eV
        （可接受区间 0.15–0.30；P3 min-vs-rest 中位 0.2209 为同源基准）
    (b) 模型级：位点预测 min 聚合回表面后与 v3 标签（min 口径）差中位数

协议（与 probes/_common.py 同，禁止改动）：
  特征 = 84 维 v3 组成描述符 + site_full one-hot（开发集类别，28 类）
  XGBoost lr=0.03, max_depth=7, n_estimators=400, subsample=0.8
  GroupKFold(5, shuffle=True, random_state=42)，groups=comp；10 种子
  仅用开发集（lockbox 冻结，F5 占优前不触碰）
公平契约：改善 > 0.005 eV 且 10/10 种子方向一致。

输出：site_oof_preds_dev.csv（record 级 OOF 预测×10 种子，供 F5 复用）
      f2f3_report.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEV_CSV = OUT / "site_dataset_dev.csv"
FEAT_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"

XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
                  subsample=0.8)
SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
ID_COLS = ["record_id", "comp", "structure", "facet", "energy_eV",
           "site_full", "site_coarse"]
# 预注册锚点
F3_TARGET = 0.090
BOUNTY_LOW, BOUNTY_HIGH = 0.077, 0.090
F2_RANGE = (0.15, 0.30)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def oof(X, y, groups, seed):
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    assert not np.isnan(pred).any()
    return pred


def main():
    df = pd.read_csv(DEV_CSV)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    assert len(feat_cols) == 84
    site_dummies = pd.get_dummies(df["site_full"], prefix="site").astype(float)
    X_site = np.hstack([df[feat_cols].values, site_dummies.values])
    X_base = df[feat_cols].values
    y = df["energy_eV"].values
    groups = df["comp"].values

    # ---- F2(a) 数据级口径自检：mean−min 口径差中位数 ----
    g = df.groupby("comp")["energy_eV"]
    gap = (g.mean() - g.min())
    f2a = float(gap.median())

    # ---- F3 位点级 OOF（84+site one-hot，10 种子）----
    preds, maes_site = {}, []
    for s in SEEDS:
        p = oof(X_site, y, groups, s)
        preds[s] = p
        maes_site.append(mae(p, y))
        print(f"[F3] seed={s} site-level OOF MAE={maes_site[-1]:.4f}", flush=True)
    # 对照：同折同种子，无位点 one-hot（量化 one-hot 的位点级贡献）
    maes_base = []
    for s in SEEDS:
        p = oof(X_base, y, groups, s)
        maes_base.append(mae(p, y))
        print(f"[F3-base] seed={s} no-site OOF MAE={maes_base[-1]:.4f}",
              flush=True)

    oof_df = df[ID_COLS].copy()
    for s in SEEDS:
        oof_df[f"oof_pred_seed{s}"] = preds[s]
    oof_df.to_csv(OUT / "site_oof_preds_dev.csv", index=False)

    # ---- F2(b) 模型级口径自检：min 聚合回表面 vs v3 标签 ----
    v3 = pd.read_csv(FEAT_CSV)[["comp", "energy_eV"]].rename(
        columns={"energy_eV": "v3_label"})
    pred_mean = oof_df.copy()
    pred_mean["oof_mean"] = np.mean([preds[s] for s in SEEDS], axis=0)
    surf_pred = pred_mean.groupby("comp")["oof_mean"].min().rename("site_min_pred")
    chk = v3.merge(surf_pred, on="comp", how="inner")
    f2b_median = float((chk["site_min_pred"] - chk["v3_label"]).abs().median())
    f2b_mae = float((chk["site_min_pred"] - chk["v3_label"]).abs().mean())

    m_site = float(np.mean(maes_site))
    m_base = float(np.mean(maes_base))
    site_gain = m_base - m_site
    wins = sum(b > s for b, s in zip(maes_base, maes_site))

    f3_verdict = ("PASS" if m_site <= F3_TARGET else "FAIL")
    leak_suspect = bool(m_site < BOUNTY_LOW)
    report = {
        "protocol": {"features": f"84 v3 + {site_dummies.shape[1]} site one-hot",
                     "xgb": XGB_PARAMS, "folds": "GroupKFold(5,shuffle,rs=42)",
                     "seeds": SEEDS, "n_dev_records": int(len(df)),
                     "n_dev_surfaces": int(df['comp'].nunique())},
        "F2a_mean_minus_min_median": f2a,
        "F2a_accept_range": list(F2_RANGE),
        "F2a_verdict": "PASS" if F2_RANGE[0] <= f2a <= F2_RANGE[1] else "FAIL",
        "F2b_minagg_pred_vs_v3label_median_abs": f2b_median,
        "F2b_minagg_pred_vs_v3label_mae": f2b_mae,
        "F2b_note": ("模型级口径差：位点模型逐位点预测后 min 聚合，与 v3 min 标签"
                     "的差反映模型误差而非聚合口径差；数据级口径差见 F2a"),
        "F3_site_oof_mae_per_seed": dict(zip(map(str, SEEDS), maes_site)),
        "F3_site_oof_mae_10seed_mean": m_site,
        "F3_site_oof_mae_std": float(np.std(maes_site)),
        "F3_target": F3_TARGET,
        "F3_verdict": f3_verdict,
        "bounty_interval": [BOUNTY_LOW, BOUNTY_HIGH],
        "beyond_bounty_leak_suspect": leak_suspect,
        "site_oof_mae_base_no_onehot_10seed_mean": m_base,
        "site_level_gain_from_onehot": site_gain,
        "site_level_gain_seeds_win": f"{wins}/10",
    }
    (OUT / "f2f3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in
                      ["F2a_mean_minus_min_median", "F2a_verdict",
                       "F2b_minagg_pred_vs_v3label_median_abs",
                       "F3_site_oof_mae_10seed_mean", "F3_verdict",
                       "beyond_bounty_leak_suspect",
                       "site_level_gain_from_onehot",
                       "site_level_gain_seeds_win"]}, indent=2))


if __name__ == "__main__":
    main()
