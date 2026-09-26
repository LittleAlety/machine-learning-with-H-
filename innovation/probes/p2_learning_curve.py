# -*- coding: utf-8 -*-
"""P2 学习曲线（learning curve）— 预注册判读规则

设计（预注册）：
- 训练量 25% / 50% / 75% / 100%，按**组级**（规范化 comp）子采样；
  每个 (fraction, seed) 用 RandomState(seed) 无放回抽取组子集，
  然后在子集上跑固定折协议 GroupKFold(5, shuffle=True, random_state=42) 的 OOF MAE
- 模型：基线 XGBoost（lr=0.03, depth=7, n_est=400, subsample=0.8），10 种子

判读规则（预注册）：
- 末端斜率 = (MAE_75% − MAE_100%) / (log2(1.0) − log2(0.75))（单位 eV/倍数据，
  以均值曲线计算）
- 末端斜率 < 0.01 eV/倍数据 → 判定数据量饱和（"data-volume saturated"）
诊断专用，禁止用于选择交付配置。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PROBES, C_MAIN, C_ALT, SEEDS10, load_data, oof_pred, mae, plt)

FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def run(X, y, groups, csv):
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    have = set(zip(done.get("fraction", []), done.get("seed", [])))
    uniq = np.unique(groups)
    for frac in FRACTIONS:
        for seed in SEEDS10:
            if (frac, seed) in have:
                continue
            if frac >= 1.0:
                keep = np.ones(len(y), bool)
            else:
                rng = np.random.RandomState(seed)
                n_keep = max(5, int(round(len(uniq) * frac)))
                gkeep = set(rng.choice(uniq, size=n_keep, replace=False))
                keep = np.array([g in gkeep for g in groups])
            pred, _ = oof_pred(X[keep], y[keep], groups[keep], seed)
            rows.append(dict(fraction=frac, seed=seed, n_rows=int(keep.sum()),
                             oof_mae=mae(pred, y[keep])))
            out = pd.DataFrame(rows)
            out.to_csv(csv, index=False)
            out.to_csv("/tmp/p2_results.csv", index=False)
        print(f"[P2] fraction={frac} done", flush=True)
    return pd.DataFrame(rows)


def make_figure(res):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for seed in SEEDS10:
        sub = res[res.seed == seed].sort_values("fraction")
        ax.plot(sub.fraction * 100, sub.oof_mae, "-", color=C_ALT, alpha=0.35,
                linewidth=0.9)
    mean = res.groupby("fraction").oof_mae.mean()
    std = res.groupby("fraction").oof_mae.std()
    x = mean.index * 100
    ax.errorbar(x, mean.values, yerr=std.values, fmt="o-", color=C_MAIN,
                capsize=3, label="mean ± std (10 seeds)")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("OOF MAE (eV)")
    ax.set_title("P2 learning curve (group-level subsampling)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PROBES / "p2_learning_curve.png")
    plt.close(fig)


def main():
    PROBES.mkdir(exist_ok=True)
    df, X, y, groups, feats = load_data()
    csv = PROBES / "p2_results.csv"
    res = run(X, y, groups, csv)
    make_figure(res)
    mean = res.groupby("fraction").oof_mae.mean()
    slope = float((mean[0.75] - mean[1.00]) / (np.log2(1.00) - np.log2(0.75)))
    verdict = {
        "mean_oof_by_fraction": {str(f): float(mean[f]) for f in FRACTIONS},
        "std_oof_by_fraction": {str(f): float(res[res.fraction == f].oof_mae.std())
                                for f in FRACTIONS},
        "slope_tail_eV_per doubling": slope,
        "verdict": "data-volume saturated" if slope < 0.01 else "NOT saturated",
    }
    (PROBES / "p2_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
