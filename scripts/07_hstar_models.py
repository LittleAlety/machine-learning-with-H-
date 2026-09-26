# -*- coding: utf-8 -*-
"""
07_hstar_models.py — H* 主线建模与交叉验证（同 03 流程，hstar_ 前缀产物）

- 模型：LinearRegression / Ridge / Lasso（StandardScaler+Pipeline）、RandomForest、XGBoost
- XGBoost 小网格（n_estimators, max_depth, learning_rate, subsample）→ hstar_xgb_grid_results.csv
- 随机 5-fold CV（种子 42）→ hstar_model_compare.csv
- GroupKFold(groups=comp 规范化组成) → hstar_group_cv_results.csv
  （comp 已元素排序+约简，无同素异写，规避 CO 线的隐性泄漏）
- 图：hstar_model_cv_compare.png、hstar_pred_vs_true.png（最优模型 OOF）
- 最优参数 → hstar_best_params.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw",
           "site_spread"]  # site_spread: 泄漏特征（含目标值，新表面不可得），禁止入模
SCORING = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
XGB_GRID = {"n_estimators": [200, 400], "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]}

C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def cv_table(models, X, y, cv, groups=None):
    rows = []
    for name, mdl in models.items():
        r = cross_validate(mdl, X, y, cv=cv, groups=groups, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for m in SCORING:
            v = r[f"test_{m}"]
            if m in ("MAE", "RMSE"):
                v = -v
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std()
        rows.append(row)
        print(f"  {name:16s} MAE {row['MAE_mean']:.3f}±{row['MAE_std']:.3f}  "
              f"RMSE {row['RMSE_mean']:.3f}  R2 {row['R2_mean']:.3f}")
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values  # 规范化组成（元素已排序+约简）
    print(f"[读入] hstar_dataset_v2.csv  {X.shape[0]} 行 × {X.shape[1]} 特征；"
          f"{len(set(groups))} 个 comp 组")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=5)

    print("[XGBoost 网格]", XGB_GRID)
    gs = GridSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1), XGB_GRID,
                      cv=kf, scoring="neg_mean_absolute_error")
    gs.fit(X, y)
    best_xgb_params = gs.best_params_
    print("[XGBoost 最优参数]", best_xgb_params, f"MAE={-gs.best_score_:.3f}")
    grid_df = pd.DataFrame(gs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    grid_df["mean_MAE"] = -grid_df.pop("mean_test_score")
    grid_df = grid_df.sort_values("rank_test_score")
    grid_df.to_csv(DP / "hstar_xgb_grid_results.csv", index=False)

    models = {
        "Linear": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        "Lasso": Pipeline([("sc", StandardScaler()),
                           ("m", Lasso(max_iter=20000, random_state=SEED))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        "XGBoost_default": XGBRegressor(random_state=SEED, n_jobs=-1),
        "XGBoost_tuned": XGBRegressor(random_state=SEED, n_jobs=-1, **best_xgb_params),
    }

    print("\n[随机 5-fold CV]")
    cmp_df = cv_table(models, X, y, kf)
    cmp_df.to_csv(DP / "hstar_model_compare.csv", index=False)
    print("\n[GroupKFold (groups=规范化 comp)]")
    gcv_df = cv_table(models, X, y, gkf, groups=groups)
    gcv_df.to_csv(DP / "hstar_group_cv_results.csv", index=False)

    best_name = cmp_df.loc[cmp_df["MAE_mean"].idxmin(), "model"]
    best_mae = float(cmp_df["MAE_mean"].min())
    with open(DP / "hstar_best_params.json", "w") as f:
        json.dump({"best_model": best_name, "random_cv_MAE": best_mae,
                   "xgb_best_params": best_xgb_params}, f, ensure_ascii=False, indent=2)
    print(f"\n[最优模型] {best_name}  随机CV MAE={best_mae:.3f}")

    # ---- LOEO（留一元素外测，最优模型）----
    import re as _re
    from sklearn.metrics import mean_absolute_error, r2_score
    best_model_loeo = (XGBRegressor(random_state=SEED, n_jobs=-1, **best_xgb_params)
                       if best_name == "XGBoost_tuned" else models[best_name])
    elements = sorted({e for c in groups for e in _re.findall(r"[A-Z][a-z]?", c)})
    loeo_rows = []
    for el in elements:
        has = df["comp"].apply(lambda c: el in _re.findall(r"[A-Z][a-z]?", c)).values
        if has.sum() < 3 or (~has).sum() < 10:
            continue
        mdl = best_model_loeo.__class__(**best_model_loeo.get_params())
        mdl.fit(X[~has], y[~has])
        pred = mdl.predict(X[has])
        loeo_rows.append({"held_out_element": el, "n_test": int(has.sum()),
                          "MAE": mean_absolute_error(y[has], pred),
                          "RMSE": float(np.sqrt(np.mean((y[has] - pred) ** 2))),
                          "R2": r2_score(y[has], pred)})
    loeo = pd.DataFrame(loeo_rows).sort_values("MAE", ascending=False)
    pooled_mae = np.average(loeo["MAE"], weights=loeo["n_test"])
    loeo.to_csv(DP / "hstar_loeo_results.csv", index=False)
    print(f"\n[LOEO] 汇集 MAE={pooled_mae:.3f} eV（按测试折大小加权）；"
          f"最差 3 元素：{', '.join(f'{r.held_out_element} {r.MAE:.3f}' for r in loeo.head(3).itertuples())}")
    print("[写出] data_processed/hstar_loeo_results.csv")

    # 图 1：模型对比
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xpos = np.arange(len(cmp_df))
    ax.bar(xpos - 0.2, cmp_df["MAE_mean"], width=0.4, yerr=cmp_df["MAE_std"],
           color=C_MAIN, label="Random 5-fold", capsize=3, error_kw=dict(lw=1))
    ax.bar(xpos + 0.2, gcv_df["MAE_mean"], width=0.4, yerr=gcv_df["MAE_std"],
           color=C_ALT, label="GroupKFold (by comp)", capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(xpos)
    ax.set_xticklabels(cmp_df["model"], rotation=20, ha="right")
    ax.set_ylabel("MAE (eV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hstar_model_cv_compare.png")
    plt.close(fig)

    # 图 2：pred vs true（最优模型 OOF，按结构类型着色）
    best_model = models[best_name]
    oof = cross_val_predict(best_model, X, y, cv=kf, n_jobs=-1)
    from sklearn.metrics import mean_absolute_error, r2_score
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for st, col, mk in [("A1", "#B35C24", "o"), ("L12", "#DFB27E", "s"),
                        ("L10", "#7A4A2B", "^")]:
        m = df["structure"] == st
        ax.scatter(y[m], oof[m], s=14, alpha=0.6, color=col, marker=mk,
                   label=st, edgecolors="none")
    lim = [min(y.min(), oof.min()) - 0.2, max(y.max(), oof.max()) + 0.2]
    ax.plot(lim, lim, "--", color="#8c8c8c", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"True $E_{ads}$(H*) (eV)")
    ax.set_ylabel("Predicted (eV, out-of-fold)")
    ax.set_title(f"{best_name}:  MAE={mean_absolute_error(y, oof):.3f} eV, "
                 f"$R^2$={r2_score(y, oof):.3f}", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hstar_pred_vs_true.png")
    plt.close(fig)
    print("[写出] figures/hstar_model_cv_compare.png, hstar_pred_vs_true.png")

    merged = cmp_df.merge(gcv_df, on="model", suffixes=("_rand", "_group"))
    print("\n===== H* 模型对比（MAE, eV）=====")
    print(merged[["model", "MAE_mean_rand", "MAE_std_rand",
                  "MAE_mean_group", "MAE_std_group",
                  "R2_mean_rand", "R2_mean_group"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
