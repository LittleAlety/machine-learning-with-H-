# -*- coding: utf-8 -*-
"""
27_seed_robustness.py — Round 9 子任务 1A：v2/v3 多种子稳健性 + Top-10 跨种子
稳定性 + v2 vs v3 配对显著性检验

协议与主训练脚本严格一致（scripts/23_descriptor_refine.py 口径）：
  - GroupKFold(n_splits=5, groups=comp)，同一批 fold 对所有种子/模型一致；
  - XGBRegressor(learning_rate=0.03, max_depth=7, n_estimators=400,
    subsample=0.8, random_state=<seed>)，固定 tuned 参数，仅 random_state 变化；
  - v2 = hstar_dataset_v2.csv 的 36 特征；v3 = hstar_features_v3_84feat.csv 的
    84 特征（已核验两表行序/能量/v2 特征列逐位一致）。

任务：
  1) v2/v3 各 10 种子（0,1,2,7,13,42,99,123,2024,31337）OOF MAE；
  2) v3 各种子 OOF → ΔG=E_ads+0.24 → |ΔG| 最小 Top-10 组成，两两 Jaccard；
  3) 配对显著性：(a) 嵌套 CV 5 外折逐折 MAE（desrefine_nested_cv.csv，
     baseline_36 vs full_merged）Wilcoxon 符号秩 + 配对 t（n=5，功效受限）；
     (b) 组成级 OOF 残差绝对值（seed=42）配对 Wilcoxon（n=1836，注意伪重复：
     同 fold 内残差不独立，p 值偏乐观，需与 (a) 联合解读）。

产物：
  outputs/r9_seed_sensitivity.csv    模型×种子×OOF_MAE（含逐折）
  outputs/r9_top10_seed_jaccard.csv  两两种子 Jaccard + ≥8/10 稳定候选名单
  outputs/r9_paired_test.csv         检验方法/统计量/p/效应量

注意：/mnt/agents/output 挂载对原子 rename Permission denied，统一 open(...,'w')。
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"

SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
                  subsample=0.8)
DG_SHIFT = 0.24
ID2 = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
ID3 = ["comp", "structure", "facet", "n_raw", "energy_eV"]


def oof_predict(X, y, groups, seed):
    """固定 tuned 参数 + GroupKFold(5, comp) OOF 预测（与 23 号脚本同 fold）。"""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    fold_maes = []
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        fold_maes.append(mean_absolute_error(y[te], pred))
    return oof, fold_maes


def main():
    v2 = pd.read_csv(DP / "hstar_dataset_v2.csv")
    v3 = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
    f2 = [c for c in v2.columns if c not in ID2]
    f3 = [c for c in v3.columns if c not in ID3]
    assert len(f2) == 36 and len(f3) == 84
    assert (v2["comp"].values == v3["comp"].values).all()
    assert f2 == f3[:36]
    y = v3["energy_eV"].values
    groups = v3["comp"].values
    print(f"[读入] v2 {v2.shape} / v3 {v3.shape}；{v3['comp'].nunique()} 组成",
          flush=True)

    # ================= 任务 1：多种子敏感性 =================
    rows, oof_store = [], {}
    for tag, df, feats in [("v2_36feat", v2, f2), ("v3_84feat", v3, f3)]:
        X = df[feats]
        for sd in SEEDS:
            oof, fm = oof_predict(X, y, groups, sd)
            mae = mean_absolute_error(y, oof)
            oof_store[(tag, sd)] = oof
            rows.append({"model": tag, "seed": sd, "oof_mae": mae,
                         "fold_maes": "|".join(f"{v:.5f}" for v in fm)})
            print(f"  [{tag} seed={sd:5d}] OOF MAE = {mae:.5f}", flush=True)
    sens = pd.DataFrame(rows)
    sens.to_csv(open(OUT / "r9_seed_sensitivity.csv", "w"), index=False,
                float_format="%.5f")

    print("\n===== 任务 1 汇总 =====", flush=True)
    summ = (sens.groupby("model")["oof_mae"]
            .agg(["mean", "std", "min", "max"]).round(5))
    print(summ.to_string(), flush=True)
    piv = sens.pivot(index="seed", columns="model", values="oof_mae")
    piv["diff_v3_minus_v2"] = piv["v3_84feat"] - piv["v2_36feat"]
    d = piv["diff_v3_minus_v2"]
    print("\n逐种子差值 (v3−v2, 负=v3 更优):", flush=True)
    print(piv.round(5).to_string(), flush=True)
    print(f"差值 mean={d.mean():.5f} std={d.std(ddof=1):.5f} "
          f"median={d.median():.5f} min={d.min():.5f} max={d.max():.5f}; "
          f"v3 胜出种子数 {(d < 0).sum()}/10", flush=True)

    # ================= 任务 2：Top-10 跨种子稳定性（v3） =================
    top10 = {}
    for sd in SEEDS:
        dG = oof_store[("v3_84feat", sd)] + DG_SHIFT
        order = np.argsort(np.abs(dG))
        top10[sd] = list(v3["comp"].values[order[:10]])
    jac_rows = []
    for a, b in combinations(SEEDS, 2):
        A, B = set(top10[a]), set(top10[b])
        j = len(A & B) / len(A | B)
        jac_rows.append({"record": "pairwise_jaccard", "seed_a": a,
                         "seed_b": b, "jaccard": j, "comp": "",
                         "seeds_present": ""})
    jvals = np.array([r["jaccard"] for r in jac_rows])
    from collections import Counter
    cnt = Counter(c for sd in SEEDS for c in top10[sd])
    stable = sorted(((c, n) for c, n in cnt.items() if n >= 8),
                    key=lambda t: (-t[1], t[0]))
    for c, n in stable:
        jac_rows.append({"record": "stable_candidate_>=8of10", "seed_a": "",
                         "seed_b": "", "jaccard": "", "comp": c,
                         "seeds_present": n})
    jac_df = pd.DataFrame(jac_rows)
    jac_df.to_csv(open(OUT / "r9_top10_seed_jaccard.csv", "w"), index=False,
                  float_format="%.4f")
    print("\n===== 任务 2：Top-10 跨种子 Jaccard =====", flush=True)
    print(f"  两两 Jaccard（45 对）mean={jvals.mean():.4f} "
          f"std={jvals.std(ddof=1):.4f} min={jvals.min():.4f} "
          f"max={jvals.max():.4f}", flush=True)
    print(f"  ≥8/10 种子出现的稳定候选 {len(stable)} 个: "
          f"{[(c, n) for c, n in stable]}", flush=True)

    # ================= 任务 3：配对显著性 =================
    pt_rows = []
    # (a) 嵌套 CV 5 外折逐折配对
    ncv = pd.read_csv(OUT / "desrefine_nested_cv.csv")
    b36 = (ncv[ncv.config == "baseline_36"].sort_values("outer_fold")
           ["MAE"].values)
    f84 = (ncv[ncv.config == "full_merged"].sort_values("outer_fold")
           ["MAE"].values)
    diff_fold = b36 - f84  # 正 = v3 改善
    w_a = stats.wilcoxon(diff_fold)
    t_a = stats.ttest_rel(b36, f84)
    pt_rows.append({"test": "nested_cv_fold_wilcoxon", "n": 5,
                    "statistic": w_a.statistic, "p_value": w_a.pvalue,
                    "effect_median": np.median(diff_fold),
                    "effect_mean": diff_fold.mean(),
                    "note": "逐外折 MAE 差 (baseline_36−full_merged)；n=5 功效极有限，"
                            "Wilcoxon 双侧最小可达 p=0.0625；精确法"})
    pt_rows.append({"test": "nested_cv_fold_paired_t", "n": 5,
                    "statistic": t_a.statistic, "p_value": t_a.pvalue,
                    "effect_median": np.median(diff_fold),
                    "effect_mean": diff_fold.mean(),
                    "note": "同上的配对 t；n=5 小样本正态性不可验，仅供参考"})
    # (b) 组成级 OOF 残差绝对值配对（seed=42）
    r2 = np.abs(y - oof_store[("v2_36feat", 42)])
    r3 = np.abs(y - oof_store[("v3_84feat", 42)])
    diff_res = r2 - r3  # 正 = v3 残差更小
    w_b = stats.wilcoxon(diff_res)
    pt_rows.append({"test": "oof_absresidual_wilcoxon_seed42", "n": len(y),
                    "statistic": w_b.statistic, "p_value": w_b.pvalue,
                    "effect_median": np.median(diff_res),
                    "effect_mean": diff_res.mean(),
                    "note": "组成级 OOF|残差|差 (v2−v3)，同一组成在两模型各一 OOF "
                            "残差故配对成立；但同 fold 样本共享训练集（伪重复），"
                            "p 值偏乐观，应结合 n=5 折级检验解读"})
    pt = pd.DataFrame(pt_rows)
    pt.to_csv(open(OUT / "r9_paired_test.csv", "w"), index=False,
              float_format="%.6g")
    print("\n===== 任务 3：配对检验 =====", flush=True)
    print(pt.to_string(index=False), flush=True)
    print(f"  折级逐折差: {[round(v,5) for v in diff_fold]}", flush=True)
    print(f"  残差级: v3 残差更小占 {(diff_res > 0).mean():.3f}，"
          f"中位差 {np.median(diff_res):+.5f} eV，"
          f"均值差 {diff_res.mean():+.5f} eV", flush=True)

    print("\n[完成] r9_seed_sensitivity.csv / r9_top10_seed_jaccard.csv / "
          "r9_paired_test.csv", flush=True)


if __name__ == "__main__":
    main()
