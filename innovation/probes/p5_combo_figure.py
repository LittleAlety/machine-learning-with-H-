# -*- coding: utf-8 -*-
"""P5 组合图 p_error_anatomy.png — 单图四面板（入正文用）
A: P1 GBDT 容量曲线（OOF vs train）；B: P2 学习曲线；
C: P3 噪声地板示意（OOF MAE 分解为 floor + headroom）；
D: P4 信息侧 Δ_A vs 容量侧最大改善。
读取各探针已存盘的中间结果（幂等）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROBES, C_MAIN, C_ALT, C_3RD, plt


def main():
    gbdt = pd.read_csv(PROBES / "p1_gbdt_results.csv")
    p2 = pd.read_csv(PROBES / "p2_results.csv")
    p3 = json.loads((PROBES / "p3_results.json").read_text())
    p4 = json.loads((PROBES / "p4_results.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # A: P1 GBDT capacity
    ax = axes[0, 0]
    g = (gbdt[gbdt.model == "GBDT"]
         .groupby(["max_depth", "n_estimators"])[["oof_mae", "train_mae"]]
         .mean().reset_index())
    colors = {3: C_ALT, 5: C_3RD, 7: C_MAIN, 10: "#8C3B12", 16: "#4A2A12"}
    for d in [3, 5, 7, 10, 16]:
        sub = g[g.max_depth == d].sort_values("n_estimators")
        ax.plot(sub.n_estimators, sub.oof_mae, "o-", color=colors[d],
                label=f"OOF d={d}", markersize=4)
        ax.plot(sub.n_estimators, sub.train_mae, "s--", color=colors[d],
                alpha=0.5, markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel("n_estimators (log scale)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("A  Capacity sweep (GBDT)", loc="left")
    ax.legend(fontsize=7, frameon=False)

    # B: P2 learning curve
    ax = axes[0, 1]
    mean = p2.groupby("fraction").oof_mae.mean()
    std = p2.groupby("fraction").oof_mae.std()
    for seed, sub in p2.groupby("seed"):
        sub = sub.sort_values("fraction")
        ax.plot(sub.fraction * 100, sub.oof_mae, "-", color=C_ALT, alpha=0.3,
                linewidth=0.8)
    ax.errorbar(mean.index * 100, mean.values, yerr=std.values, fmt="o-",
                color=C_MAIN, capsize=3)
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("OOF MAE (eV)")
    ax.set_title("B  Learning curve", loc="left")

    # C: P3 noise floor decomposition
    ax = axes[1, 0]
    oof = p3["path_b"]["oof_mae_10seed_mean"]
    floor = p3["floor_headline"]
    ax.bar(["v3 XGBoost\nOOF MAE"], [floor], color=C_3RD, width=0.45,
           label=f"noise floor (path a) = {floor:.3f} eV")
    ax.bar(["v3 XGBoost\nOOF MAE"], [oof - floor], bottom=[floor], color=C_ALT,
           width=0.45, label=f"reducible headroom = {oof - floor:.3f} eV")
    ax.axhline(oof, color=C_MAIN, linestyle="--", linewidth=1,
               label=f"OOF MAE = {oof:.3f} eV")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("C  Noise floor decomposition", loc="left")
    ax.legend(fontsize=8, frameon=False)

    # D: P4 info vs capacity
    ax = axes[1, 1]
    cap = p4["capacity_side_gain"]
    info = p4["info_side_max_deltaA"]
    bars = ax.bar(["Capacity side\n(P1 GBDT best)", "Information side\n(P4 oracle max)"],
                  [cap, info], color=[C_MAIN, C_3RD], width=0.5)
    for b, v in zip(bars, [cap, info]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("OOF MAE improvement (eV)")
    ax.set_title("D  Information ceiling vs capacity headroom", loc="left")

    fig.tight_layout()
    fig.savefig(PROBES / "p_error_anatomy.png")
    plt.close(fig)
    print("[写出]", PROBES / "p_error_anatomy.png")


if __name__ == "__main__":
    main()
