# -*- coding: utf-8 -*-
"""
10_multiads_model.py — 多吸附种统一模型 + 标度关系归纳

输入：data_processed/multiads_features.csv（09 生成，16,499 行）
建模（XGBoost 为主，复用 07 的调参策略与小网格）：
  - XGBoost 小网格调参（随机 5-fold，种子 42）→ multiads_xgb_grid_results.csv
  - 随机 5-fold CV + GroupKFold(groups=规范化 comp) → multiads_model_compare.csv
  - OOF 逐吸附种 MAE → multiads_per_adsorbate_mae.csv（与 H* 单吸附种模型对比）
SHAP：TreeExplainer（种子 42 抽样子集），np.random.seed(42) 后 summary_plot
  → figures/multiads_shap_summary.png
标度关系（归纳分析核心）：以 H* 为锚
  - multiads_dataset.csv 透视为 (comp, structure) × adsorbate 的最稳定位点能量
  - 原子吸附种 {H,C,N,O,S} 两两 Spearman/Pearson 相关矩阵热图
    → figures/multiads_scaling_matrix.png
  - 关键散点（X* vs H*：O, C, N, S）→ figures/multiads_scaling_examples.png
  - 相关系数表 → multiads_scaling_corr.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "adsorbate", "ads_atom",
           "energy_eV", "n_raw"]  # n_ads_units 是结构信息（非标签派生），保留为特征
SCORING = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
XGB_GRID = {"n_estimators": [200, 400], "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]}  # 同 07

C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
C_3RD = "#7A4A2B"
C_4TH = "#E8D3B0"
ADS_COLORS = {"O": C_MAIN, "C": C_ALT, "N": C_3RD, "S": "#C98B5E"}
from matplotlib.colors import LinearSegmentedColormap
WARM_DIV = LinearSegmentedColormap.from_list(  # 同 08：替代 shap 默认红蓝
    "warm_div", ["#5C3A10", "#F5E6C8", "#B3401E"])
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def cv_table(models, X, y, cv, groups=None):
    rows = []
    for name, mdl in models.items():
        r = cross_validate(mdl, X, y, cv=cv, groups=groups, scoring=SCORING,
                           n_jobs=-1)
        row = {"model": name, "cv": "group" if groups is not None else "random"}
        for m in SCORING:
            v = r[f"test_{m}"]
            if m in ("MAE", "RMSE"):
                v = -v
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std()
        rows.append(row)
        print(f"  [{row['cv']:6s}] {name:16s} MAE {row['MAE_mean']:.3f}±"
              f"{row['MAE_std']:.3f}  RMSE {row['RMSE_mean']:.3f}  "
              f"R2 {row['R2_mean']:.3f}")
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DP / "multiads_features.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values  # 规范化组成（同 07，规避隐性泄漏）
    print(f"[读入] multiads_features.csv  {X.shape[0]} 行 × {X.shape[1]} 特征；"
          f"{len(set(groups))} 个 comp 组，{df['adsorbate'].nunique()} 种吸附种")

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
    grid_df.sort_values("rank_test_score").to_csv(
        DP / "multiads_xgb_grid_results.csv", index=False)

    models = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        "XGBoost_default": XGBRegressor(random_state=SEED, n_jobs=-1),
        "XGBoost_tuned": XGBRegressor(random_state=SEED, n_jobs=-1,
                                      **best_xgb_params),
    }
    print("\n[随机 5-fold CV]")
    cmp_rand = cv_table(models, X, y, kf)
    print("\n[GroupKFold (groups=规范化 comp)]")
    cmp_grp = cv_table(models, X, y, gkf, groups=groups)
    cmp_df = pd.concat([cmp_rand, cmp_grp], ignore_index=True)
    cmp_df.to_csv(DP / "multiads_model_compare.csv", index=False)
    with open(DP / "multiads_best_params.json", "w") as f:
        json.dump({"best_model": "XGBoost_tuned",
                   "xgb_best_params": best_xgb_params,
                   "random_cv_MAE": float(
                       cmp_rand.loc[cmp_rand["model"] == "XGBoost_tuned",
                                    "MAE_mean"].iloc[0])},
                  f, ensure_ascii=False, indent=2)

    # ---- OOF 逐吸附种 MAE（与 H* 单吸附种模型对比）----
    tuned = models["XGBoost_tuned"]
    oof = cross_val_predict(tuned, X, y, cv=kf, n_jobs=-1)
    df["_oof"] = oof
    per_ads = (df.groupby("adsorbate")
               .apply(lambda d: pd.Series({
                   "n": len(d),
                   "MAE": float(np.mean(np.abs(d["energy_eV"] - d["_oof"]))),
                   "RMSE": float(np.sqrt(np.mean((d["energy_eV"] - d["_oof"]) ** 2))),
                   "E_range": float(d["energy_eV"].max() - d["energy_eV"].min()),
               }), include_groups=False)
               .sort_values("MAE"))
    per_ads.to_csv(DP / "multiads_per_adsorbate_mae.csv")
    print("\n[OOF 逐吸附种 MAE]\n", per_ads.round(3).to_string())

    # H* 单吸附种模型锚点（07 产物）
    hstar_bp = json.loads((DP / "hstar_best_params.json").read_text())
    print(f"\n[对比] H* 单吸附种模型随机CV MAE={hstar_bp['random_cv_MAE']:.3f} eV；"
          f"统一模型在 H* 子集 OOF MAE={per_ads.loc['H', 'MAE']:.3f} eV")

    # ---- SHAP（种子 42 抽样子集）----
    import shap
    np.random.seed(SEED)
    tuned.fit(X, y)
    n_shap = min(5000, len(X))
    idx = np.random.choice(len(X), n_shap, replace=False)
    Xs = X.iloc[idx]
    explainer = shap.TreeExplainer(tuned)
    sv = explainer.shap_values(Xs)
    mean_abs = np.abs(sv).mean(axis=0)
    shap_rank = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
    shap_rank.to_csv(DP / "multiads_shap_importance.csv")
    print("\n[SHAP top-10]\n", shap_rank.head(10).round(4).to_string())
    np.random.seed(SEED)  # summary_plot 前固定种子（SPEC 要求）
    plt.figure()
    # shap 0.52 的 cmap 为定义期绑定的默认参数，须显式传入暖色系（替代 08 的事后重着色）
    shap.summary_plot(sv, Xs, show=False, max_display=15, cmap=WARM_DIV)
    fig = plt.gcf()
    fig.set_size_inches(7.6, 5.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "multiads_shap_summary.png", bbox_inches="tight")
    plt.close("all")
    print("[写出] figures/multiads_shap_summary.png")

    # ================= 标度关系归纳（以 H* 为锚）=================
    ds = pd.read_csv(DP / "multiads_dataset.csv")
    piv = (ds.pivot_table(index=["comp", "structure"], columns="adsorbate",
                          values="energy_eV", aggfunc="min"))
    atomic = ["H", "C", "N", "O", "S"]
    piv_a = piv[atomic].dropna()
    print(f"\n[标度关系] 五种原子吸附种齐备的表面数 n={len(piv_a)}")

    def corr_mat(func):
        m = pd.DataFrame(np.nan, index=atomic, columns=atomic)
        for a in atomic:
            for b in atomic:
                m.loc[a, b] = func(piv_a[a], piv_a[b])[0]
        return m.astype(float)

    sp_mat, pe_mat = corr_mat(spearmanr), corr_mat(pearsonr)
    corr_out = pd.concat({"spearman": sp_mat, "pearson": pe_mat})
    corr_out.to_csv(DP / "multiads_scaling_corr.csv")
    print("[Spearman]\n", sp_mat.round(3).to_string())
    print("[Pearson]\n", pe_mat.round(3).to_string())

    # 热图（Spearman / Pearson 双面板，低饱和暖色系）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    im = None
    for ax, (mat, ttl) in zip(axes, [(sp_mat, "Spearman"),
                                     (pe_mat, "Pearson")]):
        im = ax.imshow(mat.values, cmap="OrRd", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(5), [f"{a}*" for a in atomic])
        ax.set_yticks(range(5), [f"{a}*" for a in atomic])
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=10,
                        color="white" if mat.values[i, j] > 0.85 else "#4a2c14")
        ax.set_title(f"{ttl} correlation, $E_{{ads}}$(X*) pairs (n={len(piv_a)})",
                     fontsize=11)
    # colorbar 放入独立坐标区（tight_layout 预留右缘），避免压盖右侧 Pearson 热图数值
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    cax = fig.add_axes([0.925, 0.16, 0.016, 0.68])
    fig.colorbar(im, cax=cax, label="correlation coefficient")
    fig.savefig(FIGDIR / "multiads_scaling_matrix.png")
    plt.close(fig)
    print("[写出] figures/multiads_scaling_matrix.png")

    # 关键散点：X* vs H*（O, C, N, S）
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.8), sharex=True)
    for ax, a in zip(axes.ravel(), ["O", "C", "N", "S"]):
        sub = piv[[a, "H"]].dropna()
        r_sp = spearmanr(sub[a], sub["H"])[0]
        r_pe = pearsonr(sub[a], sub["H"])[0]
        ax.scatter(sub["H"], sub[a], s=10, alpha=0.45, color=ADS_COLORS[a],
                   edgecolors="none")
        # 最小二乘标度线
        coef = np.polyfit(sub["H"], sub[a], 1)
        xl = np.array([sub["H"].min(), sub["H"].max()])
        ax.plot(xl, np.polyval(coef, xl), "-", color=C_3RD, lw=1.4,
                label=f"slope={coef[0]:.2f}")
        ax.set_xlabel(r"$E_{ads}$(H*) (eV)")
        ax.set_ylabel(rf"$E_{{ads}}$({a}*) (eV)")
        ax.set_title(f"{a}* vs H*:  Spearman={r_sp:.3f}, Pearson={r_pe:.3f}, "
                     f"n={len(sub)}", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "multiads_scaling_examples.png")
    plt.close(fig)
    print("[写出] figures/multiads_scaling_examples.png")

    # 标度线斜率/截距表（供报告引用）
    scale_rows = []
    for a in ["O", "C", "N", "S"]:
        sub = piv[[a, "H"]].dropna()
        coef = np.polyfit(sub["H"], sub[a], 1)
        scale_rows.append({
            "pair": f"{a}* vs H*", "n": len(sub),
            "spearman": spearmanr(sub[a], sub["H"])[0],
            "pearson": pearsonr(sub[a], sub["H"])[0],
            "slope": coef[0], "intercept": coef[1]})
    pd.DataFrame(scale_rows).to_csv(DP / "multiads_scaling_vs_hstar.csv",
                                    index=False)

    print(json.dumps({
        "best_xgb_params": best_xgb_params,
        "random_cv": cmp_rand.to_dict("records"),
        "group_cv": cmp_grp.to_dict("records"),
        "shap_top5": shap_rank.head(5).round(4).to_dict(),
        "scaling_pairs": scale_rows,
        "per_ads_mae": per_ads["MAE"].round(3).to_dict(),
        "hstar_single_mae": hstar_bp["random_cv_MAE"],
    }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
