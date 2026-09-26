# -*- coding: utf-8 -*-
"""
03_models.py — dataset_v2 建模与交叉验证

- 模型：LinearRegression / Ridge / Lasso（StandardScaler+Pipeline）、RandomForest、XGBoost
- XGBoost 只调 4 参数（n_estimators, max_depth, learning_rate, subsample），小网格，
  网格明细 → data_processed/xgb_grid_results.csv，最优参数 → data_processed/best_params.json
- 随机 5-fold CV（种子 42）：MAE/RMSE/R²（均值±标准差）→ data_processed/model_compare.csv
- GroupKFold（groups=规范化 comp，留一体系外测）→ data_processed/group_cv_results.csv
- LOEO（Leave-One-Element-Out，逐元素留出真实外推测试）→ data_processed/loeo_results.csv
  （groups=组成中任一元素：每次留出所有含某元素的样本；参考基线 XGB LOEO MAE≈0.27-0.55）
- 三口径汇总对照（随机CV / 分组CV / LOEO）→ data_processed/cv_protocol_summary.csv
- 图：figures/model_cv_compare.png（随机 vs 分组 CV 的 MAE）、
      figures/pred_vs_true.png（最优模型的 out-of-fold 预测）
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data_processed" / "dataset_v2.csv"
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"
SEED = 42

C_MAIN = "#B35C24"   # 暖赭
C_ALT = "#DFB27E"    # 浅杏
C_PT = "#A85838"
plt.rcParams.update({
    "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

ID_COLS = ["comp", "facet", "term", "adsorbate", "energy_eV", "dataset_src", "n_raw"]
SCORING = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

import re as _re
from functools import reduce as _reduce
from math import gcd as _gcd


def normalize_comp(comp):
    """元素排序 + 计量数约简（如 Au3Ag → AgAu3），用于 GroupKFold 分组键，
    避免化学等价组成（不同终止层写法）泄漏到训练/测试两侧。"""
    toks = _re.findall(r"([A-Z][a-z]?)(\d*)", comp)
    cd = {el: int(n) if n else 1 for el, n in toks}
    g = _reduce(_gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))

XGB_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
}


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


def loeo_table(models, X, y, comps):
    """留一元素外测（LOEO）：每次留出组成中含某元素的全部样本，在其余样本上训练。
    groups = 组成中任一元素（一个样本可含多个元素，因此每个元素折独立留出）。
    返回长表：model, element, n_test, MAE, RMSE, R2。"""
    elems = sorted({el for c in comps for el in re.findall(r"[A-Z][a-z]?", c)})
    # 预计算每个元素的命中掩码
    comp_elems = {c: set(re.findall(r"[A-Z][a-z]?", c)) for c in comps}
    rows = []
    for name, mdl in models.items():
        for el in elems:
            mask = np.array([el in comp_elems[c] for c in comps])
            n_test = int(mask.sum())
            m = clone(mdl)
            m.fit(X[~mask], y[~mask])
            pred = m.predict(X[mask])
            mae = mean_absolute_error(y[mask], pred)
            rmse = float(np.sqrt(mean_squared_error(y[mask], pred)))
            r2 = float(r2_score(y[mask], pred)) if n_test >= 2 else float("nan")
            rows.append({"model": name, "element": el, "n_test": n_test,
                         "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"  {name:16s} LOEO 完成（{len(elems)} 元素折）")
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_CSV)
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values
    print(f"[读入] {IN_CSV}  {X.shape[0]} 行 × {X.shape[1]} 特征；{len(set(groups))} 个 comp 组")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=5)

    # ---- XGBoost 小网格调参（随机 5-fold，MAE 选优）----
    print("[XGBoost 网格]", XGB_GRID)
    gs = GridSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1), XGB_GRID,
                      cv=kf, scoring="neg_mean_absolute_error",
                      return_train_score=False)
    gs.fit(X, y)
    best_xgb_params = gs.best_params_
    print("[XGBoost 最优参数]", best_xgb_params, f"MAE={-gs.best_score_:.3f}")
    grid_df = pd.DataFrame(gs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    grid_df["mean_MAE"] = -grid_df["mean_test_score"]
    grid_df = grid_df.drop(columns="mean_test_score").sort_values("rank_test_score")
    grid_df.to_csv(DP / "xgb_grid_results.csv", index=False)

    # ---- 模型集合 ----
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
    cmp_df.to_csv(DP / "model_compare.csv", index=False)
    print("[写出] data_processed/model_compare.csv")

    print("\n[GroupKFold (groups=规范化 comp)]")
    gcv_df = cv_table(models, X, y, gkf, groups=groups)
    gcv_df.to_csv(DP / "group_cv_results.csv", index=False)
    print("[写出] data_processed/group_cv_results.csv")

    # ---- LOEO：留一元素外测（真实外推指标）----
    print("\n[LOEO 留一元素外测]")
    loeo_df = loeo_table(models, X, y, df["comp"])
    loeo_df.to_csv(DP / "loeo_results.csv", index=False)
    print("[写出] data_processed/loeo_results.csv")

    # ---- 三口径汇总对照 ----
    # LOEO 汇总用样本数加权平均（等价于汇集所有留出折的预测后统一算 MAE）
    loeo_pool = (loeo_df.groupby("model")
                        .apply(lambda g: np.average(g["MAE"], weights=g["n_test"]),
                               include_groups=False)
                        .rename("MAE_LOEO_pooled"))
    summary = (cmp_df[["model", "MAE_mean", "MAE_std"]]
               .rename(columns={"MAE_mean": "MAE_randomCV", "MAE_std": "MAE_randomCV_std"})
               .merge(gcv_df[["model", "MAE_mean"]].rename(
                   columns={"MAE_mean": "MAE_groupCV_comp"}), on="model")
               .merge(loeo_pool.reset_index(), on="model"))
    summary["LOEO_minus_random"] = summary["MAE_LOEO_pooled"] - summary["MAE_randomCV"]
    summary = summary.round(4)
    summary.to_csv(DP / "cv_protocol_summary.csv", index=False)
    print("[写出] data_processed/cv_protocol_summary.csv")
    print("\n===== 三口径对照（MAE, eV）：随机CV < 分组CV(comp) < LOEO(元素) 为预期递增 =====")
    print(summary.to_string(index=False))
    xgb_loeo = loeo_df[loeo_df["model"] == "XGBoost_tuned"]
    print("\n[LOEO 明细] XGBoost_tuned 代表性元素折 MAE：")
    for el in ["Pt", "Au", "Ag", "Ni"]:
        r = xgb_loeo[xgb_loeo["element"] == el]
        if len(r):
            print(f"  {el}: MAE={r['MAE'].iloc[0]:.3f} (n_test={int(r['n_test'].iloc[0])})")

    # ---- 最优模型（随机 CV MAE 最小）----
    best_name = cmp_df.loc[cmp_df["MAE_mean"].idxmin(), "model"]
    best_model = models[best_name]
    best_mae = float(cmp_df["MAE_mean"].min())
    with open(DP / "best_params.json", "w") as f:
        json.dump({"best_model": best_name, "random_cv_MAE": best_mae,
                   "xgb_best_params": best_xgb_params}, f, ensure_ascii=False, indent=2)
    print(f"\n[最优模型] {best_name}  随机CV MAE={best_mae:.3f}")

    # ---- 图 1：模型对比（随机 vs 分组 CV 的 MAE）----
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xpos = np.arange(len(cmp_df))
    ax.bar(xpos - 0.2, cmp_df["MAE_mean"], width=0.4, yerr=cmp_df["MAE_std"],
           color=C_MAIN, label="Random 5-fold", capsize=3, error_kw=dict(lw=1))
    ax.bar(xpos + 0.2, gcv_df["MAE_mean"], width=0.4, yerr=gcv_df["MAE_std"],
           color=C_ALT, label="GroupKFold (normalized comp)", capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(xpos)
    ax.set_xticklabels(cmp_df["model"], rotation=20, ha="right")
    ax.set_ylabel("MAE (eV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_cv_compare.png")
    plt.close(fig)

    # ---- 图 2：最优模型 pred vs true（out-of-fold）----
    oof = cross_val_predict(best_model, X, y, cv=kf, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for ads, col, mk in [("CO", "#D98E5F", "o"), ("O", "#7A4A2B", "s")]:
        m = df["adsorbate"] == ads
        ax.scatter(y[m], oof[m], s=16, alpha=0.65, color=col, marker=mk,
                   label=ads, edgecolors="none")
    lim = [min(y.min(), oof.min()) - 0.2, max(y.max(), oof.max()) + 0.2]
    ax.plot(lim, lim, "--", color="#8c8c8c", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True $E$ (eV)"); ax.set_ylabel("Predicted $E$ (eV, out-of-fold)")
    from sklearn.metrics import mean_absolute_error, r2_score
    ax.set_title(f"{best_name}:  MAE={mean_absolute_error(y, oof):.3f} eV, "
                 f"$R^2$={r2_score(y, oof):.3f}", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "pred_vs_true.png")
    plt.close(fig)
    print("[写出] figures/model_cv_compare.png, figures/pred_vs_true.png")

    # 终端汇总表
    merged = cmp_df.merge(gcv_df, on="model", suffixes=("_rand", "_group"))
    print("\n===== 模型对比（MAE, eV）=====")
    print(merged[["model", "MAE_mean_rand", "MAE_std_rand",
                  "MAE_mean_group", "MAE_std_group",
                  "R2_mean_rand", "R2_mean_group"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
