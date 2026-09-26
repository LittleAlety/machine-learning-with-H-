# -*- coding: utf-8 -*-
"""
26b_redraw_final_figures.py — 从既有数据 csv 重绘 4 张最终版图（零重训）

重建 /tmp 中丢失的最终排版代码；只读既有产物。
产物（--outdir，默认 figures/）：
  1. hstar_pred_vs_true.png   v2 模型 OOF pred vs true（三结构双通道编码 + 残差 inset）
  2. hstar_candidates.png     全 1,836 表面 OOF |ΔG_H*| 分布（全图 + 0–0.3 eV zoom）
  3. extval_parity.png        EqV2-HER 外部验证 parity + 残差分布（轴外点箭头标注）
  4. alval_learning_curve.png 回溯 AL 学习曲线（random ±1 std 带 + 两条 AL 曲线）

数据源：
  - data_processed/candidate_rankings_hstar.csv  （1,836 行 OOF：energy_eV / pred_E_ads
    / pred_dG / abs_pred_dG；MAE=0.120 eV、R²=0.816、|ΔG|≤0.1 eV 窗口 n=253 均由此复现）
  - data_processed/extval_results.csv            （11_external_validation.py 产物）
  - outputs/alval_A_learning_curve.csv           （24_al_validation.py 产物）

用法：python3 scripts/26b_redraw_final_figures.py [--outdir figures/]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"

# ---- 暖色系调色板（禁蓝紫渐变）----
TERRA = "#b5522d"        # 陶土
OLIVE = "#8a7b4f"        # 橄榄
SAND = "#d9a441"         # 沙金
SAND_LIGHT = "#d9b98a"   # 浅沙金（直方图 / ±1 std 带基色）
DEEP = "#7a4a2b"         # 深棕
WINDOW = "#f4e6e1"       # excellent window 浅填充
MAMUN_BG = "#e8d9c6"     # Mamun OOF 浅暖灰背景
GREY = "#8c8c8c"         # 对角线 / 0 虚线

plt.rcParams.update({"savefig.dpi": 300, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- fig 1
def fig_pred_vs_true(outdir):
    df = pd.read_csv(DP / "candidate_rankings_hstar.csv")
    y = df["energy_eV"].values
    p = df["pred_E_ads"].values
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)

    with plt.rc_context({"font.size": 10}):
        _fig_pred_vs_true_draw(outdir, y, p, df, mae, r2)


def _fig_pred_vs_true_draw(outdir, y, p, df, mae, r2):
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_axes([0.086, 0.13, 0.79, 0.79])
    for st, mk, col in [("A1", "o", TERRA), ("L12", "s", OLIVE), ("L10", "^", SAND)]:
        m = (df["structure"] == st).values
        ax.scatter(y[m], p[m], s=12, alpha=0.45, marker=mk, color=col,
                   edgecolors="none", label=f"{st} (n={int(m.sum())})")
    lim = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
    ax.plot(lim, lim, "--", color=GREY, lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.set_xlabel(r"True $E_{ads}$(H*) (eV)")
    ax.set_ylabel("Predicted (eV, out-of-fold)")
    ax.set_title(f"XGBoost_tuned:  MAE = {mae:.3f} eV,  $R^2$ = {r2:.3f}",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99),
              fontsize=12, borderaxespad=0.0, handletextpad=0.2)

    # 残差直方图 inset：主 axes 盒外右侧
    axins = fig.add_axes([0.72, 0.30, 0.235, 0.42])
    res = p - y
    axins.hist(res, bins=24, range=(-1.1, 1.1), color=SAND_LIGHT,
               edgecolor="white", linewidth=0.5)
    axins.axvline(0, color=GREY, ls="--", lw=1.2)
    axins.set_xlim(-1.1, 1.1)
    axins.set_xticks(np.arange(-1.0, 1.01, 0.5))
    axins.set_xticklabels([f"{v:.1f}" for v in np.arange(-1.0, 1.01, 0.5)])
    axins.set_title("OOF residuals", fontsize=12)
    axins.set_xlabel("residual (eV)", fontsize=11)
    axins.set_ylabel("count", fontsize=11)
    axins.tick_params(labelsize=10)

    fig.savefig(outdir / "hstar_pred_vs_true.png")
    plt.close(fig)
    print(f"[写出] {outdir / 'hstar_pred_vs_true.png'}  "
          f"(MAE={mae:.3f}, R2={r2:.3f})")


# ---------------------------------------------------------------- fig 2
def fig_candidates(outdir):
    df = pd.read_csv(DP / "candidate_rankings_hstar.csv")
    adg = df["abs_pred_dG"].values          # |ΔG_H*| = |pred_E_ads + 0.24|
    n_win = int((adg <= 0.1).sum())
    top10 = df["abs_pred_dG"].head(10).values

    with plt.rc_context({"font.size": 10.5}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.6),
                                       gridspec_kw=dict(width_ratios=[2.05, 1]))
        for ax, rng, nb in [(ax1, (0.0, 2.0), 60), (ax2, (0.0, 0.3), 30)]:
            ax.axvspan(0, 0.1, color=WINDOW, zorder=0)
            ax.hist(adg, bins=nb, range=rng, color=SAND_LIGHT,
                    edgecolor="white", linewidth=0.6, zorder=2)
            ax.axvline(0.12, color=DEEP, ls="--", lw=2, zorder=3)
            ax.axhline(0, color="#3a3a3a", lw=1.2, zorder=3)
            for v in top10:  # rug：轴线下方短竖线
                ax.axvline(v, ymin=0.012, ymax=0.075, color=TERRA, lw=2.5,
                           zorder=4)

        ax1.set_xlim(-0.1, 2.1)
        ax1.set_ylim(-13, 112)
        ax1.set_xticks(np.arange(0, 2.01, 0.25))
        ax1.set_xticklabels([f"{v:.2f}" for v in np.arange(0, 2.01, 0.25)])
        ax1.set_xlabel(r"$|\Delta G_{H*}|$ (eV, out-of-fold prediction)")
        ax1.set_ylabel("Count")
        t = ax1.text(0.32, 105, f"n = {n_win} in window", color=TERRA,
                     fontsize=11)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        ax2.set_xlim(0, 0.3)
        ax2.set_ylim(-4.3, 33.5)
        ax2.set_xticks(np.arange(0, 0.31, 0.05))
        ax2.set_xticklabels([f"{v:.2f}" for v in np.arange(0, 0.31, 0.05)])
        ax2.set_xlabel(r"$|\Delta G_{H*}|$ (eV)")
        ax2.set_ylabel("Count")
        ax2.set_title("zoom: 0-0.3 eV", fontsize=13)

        handles = [Patch(facecolor=SAND_LIGHT, edgecolor="white"),
                   Patch(facecolor=WINDOW, edgecolor="#e3cdc5"),
                   Line2D([0], [0], color=DEEP, ls="--", lw=2),
                   Line2D([0], [0], color=TERRA, lw=2.5)]
        labels = [f"all surfaces (n={len(adg):,})",
                  r"excellent window $|\Delta G_{H*}| \leq$ 0.1 eV",
                  "model MAE = 0.12 eV",
                  "top-10 candidates (rug)"]
        ax1.legend(handles, labels, frameon=False, loc="upper left",
                   bbox_to_anchor=(0.44, 0.99), fontsize=12)

        fig.tight_layout()
        fig.savefig(outdir / "hstar_candidates.png")
        plt.close(fig)
    print(f"[写出] {outdir / 'hstar_candidates.png'}  (n={len(adg)}, "
          f"window n={n_win})")


# ---------------------------------------------------------------- fig 3
def fig_extval_parity(outdir):
    oof = pd.read_csv(DP / "candidate_rankings_hstar.csv")   # Mamun in-house OOF
    ext = pd.read_csv(DP / "extval_results.csv")             # EqV2 外部验证明细
    y0, p0 = oof["energy_eV"].values, oof["pred_E_ads"].values
    oof_mae = mean_absolute_error(y0, p0)
    yt, yp = ext["E_true"].values, ext["E_pred"].values
    res = yp - yt
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    bias = float(np.mean(res))
    ind = ext["in_domain_facet"].values

    lim = [-2.85, 1.85]
    off = (yt > lim[1]) | (yt < lim[0]) | (yp > lim[1]) | (yp < lim[0])
    n_off = int(off.sum())
    max_true = yt.max()
    n_below = int((res < -2).sum())
    min_res = res.min()

    fig = plt.figure(figsize=(11.4, 5.2))
    # ---- 左：pred vs true ----
    ax = fig.add_axes([0.0925, 0.122, 0.362, 0.80])
    ax.scatter(y0, p0, s=8, alpha=0.3, color=MAMUN_BG, edgecolors="none",
               label=f"Mamun in-house OOF (MAE={oof_mae:.3f} eV)")
    ax.scatter(yt[ind], yp[ind], s=22, alpha=0.7, marker="o", color=TERRA,
               edgecolors="none", label="EqV2 facet=111 (in-domain)")
    ax.scatter(yt[~ind], yp[~ind], s=22, alpha=0.55, marker="s", color=SAND,
               edgecolors="none", label="EqV2 other facets (OOD)")
    ax.plot(lim, lim, "--", color=GREY, lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"True $E_{ads}$(H*) (eV, EqV2 / Mamun)")
    ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"EqV2 external: MAE = {mae:.3f} eV, $R^2$ = {r2:.3f}, "
                 f"bias = {bias:+.3f} eV", fontsize=13)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99),
              fontsize=11, borderaxespad=0.0)
    ax.text(-1.0, -2.55, f"n={n_off} off-axis (max true {max_true:.1f} eV)",
            va="center", color=DEEP, fontsize=10)
    ax.annotate("", xy=(1.80, -2.55), xytext=(1.15, -2.55),
                arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.8))

    # ---- 右：残差分布 ----
    ax2 = fig.add_axes([0.540, 0.122, 0.442, 0.80])
    ax2.hist(p0 - y0, bins=60, density=True, alpha=0.6, color=MAMUN_BG,
             label="Mamun OOF residuals")
    ax2.hist(res[ind], bins=30, density=True, alpha=0.7, color=TERRA,
             label="EqV2 facet=111 residuals")
    ax2.hist(res[~ind], bins=30, density=True, alpha=0.6, color=SAND,
             label="EqV2 other-facet residuals")
    ax2.axvline(0, color=GREY, ls="--", lw=1.2)
    ax2.set_xlim(-2, 2)
    ax2.set_xticks(np.arange(-2.0, 2.01, 0.5))
    ax2.set_xticklabels([f"{v:.1f}" for v in np.arange(-2.0, 2.01, 0.5)])
    ax2.set_ylim(0, None)
    ax2.set_xlabel(r"Residual = pred $-$ true (eV)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual distribution (systematic bias)", fontsize=13)
    ax2.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               fontsize=11)
    ax2.annotate(f"n={n_below} below axis\n(min \u2212{abs(min_res):.1f} eV)",
                 xy=(-1.94, 1.9), xytext=(-1.08, 1.9),
                 ha="left", va="center", color=DEEP, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.8))

    fig.savefig(outdir / "extval_parity.png")
    plt.close(fig)
    print(f"[写出] {outdir / 'extval_parity.png'}  (MAE={mae:.3f}, "
          f"R2={r2:.3f}, bias={bias:+.3f}, off-axis n={n_off}, "
          f"below-axis n={n_below})")


# ---------------------------------------------------------------- fig 4
def fig_alval_curve(outdir):
    curve = pd.read_csv(OUT / "alval_A_learning_curve.csv")
    k = curve["k"].values
    rmean = curve["rand_mae_mean"].values
    rstd = curve["rand_mae_std"].values

    fig, ax = plt.subplots(figsize=(7.0, 5.19))
    ax.fill_between(k, rmean - rstd, rmean + rstd, color=SAND_LIGHT, alpha=0.35,
                    lw=0, label="random \u00b11 std (n=10 seeds)")
    ax.plot(k, rmean, "--", color=SAND_LIGHT, lw=1.8, marker="s", ms=8,
            mfc=SAND_LIGHT, mec=DEEP, mew=1.5, label="random acquisition (mean)")
    ax.plot(k, curve["al_mae"].values, "-", color=TERRA, lw=2.5, marker="o",
            ms=10, mec=DEEP, mew=1.2,
            label="AL top-k (width \u00d7 volcano kernel)")
    ax.plot(k, curve["alw_mae"].values, ":", color=DEEP, lw=2.2, marker="^",
            ms=9, label="AL top-k (width only, ablation)")
    for kk, v in zip(k, curve["al_mae"].values):
        if kk == 0:
            continue
        t = ax.annotate(f"{v:.3f}", (kk, v), textcoords="offset points",
                        xytext=(0, -6), ha="center", va="top",
                        color=TERRA, fontsize=12)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xticks(k)
    ax.set_ylim(0.095, 0.167)
    ax.set_xlabel("k = number of revealed (pseudo-DFT) points added")
    ax.set_ylabel("MAE on remaining pool (eV)")
    fig.suptitle("Retrospective AL validation: 200-composition hold-out pool",
                 fontsize=13.5, y=0.975)
    ax.set_title("(initial train 1,636; uncertainty = q95$-$q05, score = "
                 r"width$\times$exp($-|\Delta G_{H*}|$/0.1))", fontsize=11,
                 pad=4)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.895, bottom=0.30)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2,
               frameon=False, fontsize=11.5)
    fig.savefig(outdir / "alval_learning_curve.png", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"[写出] {outdir / 'alval_learning_curve.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "figures"),
                    help="输出目录（默认 figures/）")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_pred_vs_true(outdir)
    fig_candidates(outdir)
    fig_extval_parity(outdir)
    fig_alval_curve(outdir)


if __name__ == "__main__":
    main()
