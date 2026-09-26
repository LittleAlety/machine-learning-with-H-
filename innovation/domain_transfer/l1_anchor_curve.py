# -*- coding: utf-8 -*-
"""
Phase 4 Workstream L — L1 锚点效率曲线（预注册: config/preregistered_phase4.yaml
workstream_L.L1_anchor_curve；红线 16：锚点只进训练折，禁止进入任何测试评估面）。

协议：
  测试域 = EqV2_RPBE（跨库主目标，与 Phase 3 i3 / 0.347 eV 基线同口径）。
  锚点数梯度 n ∈ {0(参考), 5, 10, 20, 50}：从测试域随机抽 n 行进训练折，
  其余测试域行作为评估面（锚点绝不参与评估）。
  训练集 = Mamun_BEEFvdW 全域 + CatHub_PBE 全域 + n 个锚点。
  模型 = B4 域指示 GBDT（84 描述符 + 2 域哑变量，XGBoost 锁死
  hstar_best_params.json 出厂 tuned 参数）。
  10 种子重复（预注册 seeds）。
产物：l1_curve.csv（逐种子）、l1_curve.png（MAE 与 Spearman ρ 双曲线，
300dpi、暖色系 #B35C24/#DFB27E/#7A4A2B、英文标注、去右上 spine）。
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "innovation" / "domain_transfer"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
ANCHOR_GRID = [0, 5, 10, 20, 50]  # 0 为零锚点参考点
TEST_DOMAIN = "EqV2_RPBE"
XGB_PARAMS = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]

C_MAIN = "#B35C24"
C_FILL = "#DFB27E"
C_DARK = "#7A4A2B"


def load_domains():
    """复用 Phase 3 run_invariance 的三域加载与 84 维特征对齐代码。"""
    spec = importlib.util.spec_from_file_location(
        "run_invariance", ROOT / "innovation" / "functional_invariance" / "run_invariance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_domains()


def build_design(domains, feat_cols):
    names = list(domains)
    X = pd.concat([domains[k]["X"] for k in names], ignore_index=True)
    y = np.concatenate([domains[k]["y"] for k in names])
    dom = np.concatenate([[k] * len(domains[k]["y"]) for k in names])
    dummies = pd.get_dummies(dom, drop_first=True).values.astype(float)
    X_ind = np.hstack([X[feat_cols].values.astype(float), dummies])
    return X_ind, y, dom


def run_l1(X_ind, y, dom):
    is_te = dom == TEST_DOMAIN
    idx_te = np.where(is_te)[0]
    idx_other = np.where(~is_te)[0]
    rows = []
    for n_anchor in ANCHOR_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(idx_te)
            anchors = perm[:n_anchor]
            eval_idx = perm[n_anchor:]  # 红线16：锚点不进评估面
            tr = np.concatenate([idx_other, anchors])
            m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
            m.fit(X_ind[tr], y[tr])
            pred = m.predict(X_ind[eval_idx])
            mae = float(mean_absolute_error(y[eval_idx], pred))
            rho = float(spearmanr(y[eval_idx], pred)[0])
            rows.append({"n_anchor": n_anchor, "seed": seed,
                         "n_train": len(tr), "n_eval": len(eval_idx),
                         "MAE_eV": mae, "spearman_rho": rho})
            print(f"[L1] n_anchor={n_anchor:3d} seed={seed:5d} "
                  f"MAE={mae:.4f} rho={rho:.3f} (n_eval={len(eval_idx)})")
    return pd.DataFrame(rows)


def plot_l1(df):
    summ = (df.groupby("n_anchor")
            .agg(MAE_mean=("MAE_eV", "mean"), MAE_sd=("MAE_eV", "std"),
                 rho_mean=("spearman_rho", "mean"), rho_sd=("spearman_rho", "std"))
            .reset_index().sort_values("n_anchor"))
    x = summ["n_anchor"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mcol, scol, ylab in [
            (axes[0], "MAE_mean", "MAE_sd", "Cross-library MAE on EqV2 (eV)"),
            (axes[1], "rho_mean", "rho_sd", "Spearman rho on EqV2")]:
        ax.plot(x, summ[mcol], "-o", color=C_MAIN, lw=2, ms=6,
                markeredgecolor=C_DARK, label="B4 domain-indicator GBDT")
        ax.fill_between(x, summ[mcol] - summ[scol], summ[mcol] + summ[scol],
                        color=C_FILL, alpha=0.45, label="±1 SD (10 seeds)")
        for xi, yi in zip(x, summ[mcol]):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=C_DARK)
        ax.set_xlabel("Number of anchor points from test domain (train fold only)")
        ax.set_ylabel(ylab)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].axhline(0.347, color=C_DARK, ls="--", lw=1.2,
                    label="Zero-retrain baseline 0.347 eV")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("L1 anchor efficiency curve — MAE", color=C_DARK)
    axes[1].set_title("L1 anchor efficiency curve — ranking", color=C_DARK)
    fig.suptitle("Phase 4 L1: anchor efficiency of B4 (domain-indicator GBDT), "
                 "test domain = EqV2 (RPBE)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "l1_curve.png", dpi=300)
    plt.close(fig)
    return summ


def main():
    domains, feat_cols = load_domains()
    X_ind, y, dom = build_design(domains, feat_cols)
    df = run_l1(X_ind, y, dom)
    df.to_csv(OUT / "l1_curve.csv", index=False)
    summ = plot_l1(df)
    print(summ.round(4).to_string(index=False))
    print("[写出] l1_curve.csv / l1_curve.png")


if __name__ == "__main__":
    main()
