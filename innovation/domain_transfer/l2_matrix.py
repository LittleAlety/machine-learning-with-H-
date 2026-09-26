# -*- coding: utf-8 -*-
"""
Phase 4 Workstream L — L2 六向迁移矩阵（预注册: config/preregistered_phase4.yaml
workstream_L.L2_matrix；红线 16：锚点只进训练折）。

协议（与 L1 完全一致）：
  三域 Mamun_BEEFvdW / EqV2_RPBE / CatHub_PBE 的六个有序方向 训练域→测试域：
  训练集 = 训练域全域 + 从测试域随机抽 20 个锚点（红线16：仅进训练折）；
  评估面 = 测试域其余全部行；第三域不参与（纯跨库迁移）。
  模型 = B4 域指示 GBDT（84 描述符 + 2 域哑变量，锁死出厂 tuned 参数），
  10 种子重复，报告 MAE 与 Spearman ρ 的逐种子值与均值。
  失败格（高 MAE / 低 ρ）如实呈现，不做任何剔除。
产物：l2_matrix.csv（逐种子长表 + 均值列透视）、l2_matrix.png
（MAE / ρ 双热图，300dpi、暖色系、英文标注、去右上 spine）。
"""
import importlib.util
import json
import warnings
from itertools import permutations
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
N_ANCHOR = 20
XGB_PARAMS = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]

C_MAIN = "#B35C24"
C_DARK = "#7A4A2B"
SHORT = {"Mamun_BEEFvdW": "Mamun", "EqV2_RPBE": "EqV2", "CatHub_PBE": "CatHub"}


def load_domains():
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
    return X_ind, y, dom, names


def run_l2(X_ind, y, dom, names):
    rows = []
    for tr_dom, te_dom in permutations(names, 2):
        idx_tr0 = np.where(dom == tr_dom)[0]
        idx_te = np.where(dom == te_dom)[0]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(idx_te)
            anchors = perm[:N_ANCHOR]
            eval_idx = perm[N_ANCHOR:]
            tr = np.concatenate([idx_tr0, anchors])
            m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
            m.fit(X_ind[tr], y[tr])
            pred = m.predict(X_ind[eval_idx])
            mae = float(mean_absolute_error(y[eval_idx], pred))
            rho = float(spearmanr(y[eval_idx], pred)[0])
            rows.append({"train_domain": tr_dom, "test_domain": te_dom,
                         "n_anchor": N_ANCHOR, "seed": seed,
                         "n_train": len(tr), "n_eval": len(eval_idx),
                         "MAE_eV": mae, "spearman_rho": rho})
            print(f"[L2] {SHORT[tr_dom]:6s} -> {SHORT[te_dom]:6s} "
                  f"seed={seed:5d} MAE={mae:.4f} rho={rho:.3f}")
    return pd.DataFrame(rows)


def plot_l2(df, names):
    mae_piv = df.pivot_table(index="train_domain", columns="test_domain",
                             values="MAE_eV", aggfunc="mean").reindex(index=names, columns=names)
    rho_piv = df.pivot_table(index="train_domain", columns="test_domain",
                             values="spearman_rho", aggfunc="mean").reindex(index=names, columns=names)
    labels = [SHORT[n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, piv, title, cbar_lab in [
            (axes[0], mae_piv, "MAE (eV), mean over 10 seeds", "MAE (eV)"),
            (axes[1], rho_piv, "Spearman rho, mean over 10 seeds", "Spearman rho")]:
        M = piv.values
        im = ax.imshow(M, cmap="YlOrBr")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(labels)
        ax.set_xlabel("Test domain (anchors drawn from here, train fold only)")
        ax.set_ylabel("Train domain")
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color=C_DARK)
                else:
                    v = M[i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if v > np.nanmax(M) * 0.65 else "black")
        ax.set_title(title, color=C_DARK)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.75, label=cbar_lab)
    fig.suptitle("Phase 4 L2: six-direction transfer matrix of B4 "
                 "(domain-indicator GBDT, 20 anchors from test domain into "
                 "train fold)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "l2_matrix.png", dpi=300)
    plt.close(fig)
    return mae_piv, rho_piv


def main():
    domains, feat_cols = load_domains()
    X_ind, y, dom, names = build_design(domains, feat_cols)
    df = run_l2(X_ind, y, dom, names)
    df.to_csv(OUT / "l2_matrix.csv", index=False)
    mae_piv, rho_piv = plot_l2(df, names)
    print("MAE mean matrix:\n", mae_piv.round(4).to_string())
    print("rho mean matrix:\n", rho_piv.round(4).to_string())
    print("[写出] l2_matrix.csv / l2_matrix.png")


if __name__ == "__main__":
    main()
