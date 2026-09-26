# -*- coding: utf-8 -*-
"""
28_al_seed_variance.py — Round 9 子任务 1A④：AL 纯不确定度消融策略的模型种子方差

背景：scripts/24_al_validation.py 任务 A 中，纯不确定度消融（width-only top-k）
在 k=20 时报剩余池 MAE 0.1081 vs 随机 0.1407±0.0069（−23%，4.7σ），
但该策略自身只跑了单次确定性运行（模型 seed=42），跨种子方差未报告。

本脚本复用 24 号脚本任务 A 的全部逻辑（同一留池：POOL_SEED=123 取 200 组成；
同一 36 特征；同一 XGB tuned 参数；同一分位数不确定度管线 q05/q95→width），
仅把 width-only 策略在 5 个模型种子（0,1,7,42,123）下各跑一遍
（base 与 q05/q95 分位数模型同时换种子，即整条策略管线的种子），
k∈{5,10,15,20}，报 mean±std；随机基线按 24 号原样重算（模型 seed=42，
RAND_SEEDS=1000..1009，n=10）用于 Welch t 比较与复现校验。

显著性口径：
  - σ_naive     = (rand_mean − alw_mean) / rand_std        （24 号原口径）
  - σ_combined  = (rand_mean − alw_mean) / sqrt(rand_std² + alw_std²)
  - Welch t（alw n=5 vs rand n=10，不等方差）+ p 值

产物：outputs/r9_al_seed_variance.csv
注意：/mnt/agents/output 挂载对原子 rename Permission denied，统一 open(...,'w')。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"

POOL_SEED = 123          # 同 24：留池种子（固定）
N_POOL = 200
K_LIST = [5, 10, 15, 20]
N_RAND = 10
RAND_SEEDS = [1000 + i for i in range(N_RAND)]
MODEL_SEEDS = [0, 1, 7, 42, 123]
DG_SHIFT = 0.24
VOLCANO_SIGMA = 0.1
ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
                  subsample=0.8)


def fit_base(X, y, seed):
    m = XGBRegressor(**{**XGB_PARAMS, "random_state": seed}, n_jobs=-1)
    m.fit(X, y)
    return m


def fit_quantile(X, y, alpha, seed):
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                     **{**XGB_PARAMS, "random_state": seed}, n_jobs=-1)
    m.fit(X, y)
    return m


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    # ---- 与 24 号完全一致的留池划分 ----
    comps = np.array(sorted(df["comp"].unique()))
    rng = np.random.RandomState(POOL_SEED)
    perm = rng.permutation(len(comps))
    pool_comps = set(comps[perm[:N_POOL]])
    train_df = df[~df["comp"].isin(pool_comps)].reset_index(drop=True)
    pool_df = df[df["comp"].isin(pool_comps)].reset_index(drop=True)
    Xtr, ytr = train_df[feats], train_df["energy_eV"].values
    print(f"[留池] 训练 {len(train_df)} 行 / 池 {len(pool_df)} 行 "
          f"(POOL_SEED={POOL_SEED})，特征 {len(feats)} 个", flush=True)

    # ---- width-only 策略 × 5 模型种子 ----
    rows = []
    for sd in MODEL_SEEDS:
        base0 = fit_base(Xtr, ytr, sd)
        q05 = fit_quantile(Xtr, ytr, 0.05, sd).predict(pool_df[feats])
        q95 = fit_quantile(Xtr, ytr, 0.95, sd).predict(pool_df[feats])
        width = q95 - q05
        width_order = pool_df.assign(width=width).sort_values(
            "width", ascending=False)
        for k in K_LIST:
            sel_w = width_order.head(k)
            Xw = pd.concat([Xtr, sel_w[feats]])
            yw = np.concatenate([ytr, sel_w["energy_eV"].values])
            rest_w = pool_df.drop(index=sel_w.index)
            m = mae(rest_w["energy_eV"].values,
                    fit_base(Xw, yw, sd).predict(rest_w[feats]))
            rows.append({"record": "alw_seed", "k": k, "seed": sd,
                         "n_remaining": N_POOL - k, "mae": m})
            print(f"  [width-only seed={sd:3d} k={k:2d}] 剩余池 MAE = {m:.5f}",
                  flush=True)

    # ---- 随机基线（24 号原样：模型 seed=42，RAND_SEEDS=1000..1009）----
    rand = {}
    for k in K_LIST:
        maes = []
        for sd in RAND_SEEDS:
            rsel = pool_df.sample(n=k, random_state=sd)
            Xr = pd.concat([Xtr, rsel[feats]])
            yr = np.concatenate([ytr, rsel["energy_eV"].values])
            rrest = pool_df.drop(index=rsel.index)
            maes.append(mae(rrest["energy_eV"].values,
                            fit_base(Xr, yr, 42).predict(rrest[feats])))
        rand[k] = np.array(maes)
        print(f"  [随机 k={k:2d}] MAE = {rand[k].mean():.5f}±"
              f"{rand[k].std(ddof=1):.5f}（24 号复现校验）", flush=True)

    # ---- 汇总 + Welch t ----
    alw = pd.DataFrame(rows)
    print("\n===== 汇总（width-only 跨种子 vs 随机基线） =====", flush=True)
    for k in K_LIST:
        a = alw.loc[alw.k == k, "mae"].values
        r = rand[k]
        am, asd = a.mean(), a.std(ddof=1)
        rm, rsd = r.mean(), r.std(ddof=1)
        diff = rm - am
        s_naive = diff / rsd
        s_comb = diff / np.sqrt(rsd**2 + asd**2)
        wt = stats.ttest_ind(a, r, equal_var=False)
        rows.append({"record": "summary", "k": k, "seed": "mean_std",
                     "n_remaining": N_POOL - k,
                     "mae": f"{am:.5f}±{asd:.5f}"})
        print(f"  [k={k:2d}] AL(width-only) {am:.5f}±{asd:.5f} "
              f"(n=5, min {a.min():.5f}/max {a.max():.5f}) | "
              f"随机 {rm:.5f}±{rsd:.5f} (n=10) | diff={diff:+.5f} "
              f"= {s_naive:.2f}σ_naive / {s_comb:.2f}σ_combined | "
              f"Welch t={wt.statistic:.2f}, p={wt.pvalue:.4g}", flush=True)
        rows[-1].update({
            "rand_mean": rm, "rand_std": rsd, "alw_mean": am, "alw_std": asd,
            "diff_rand_minus_alw": diff, "sigma_naive": s_naive,
            "sigma_combined": s_comb, "welch_t": wt.statistic,
            "welch_p": wt.pvalue,
            "welch_df": (asd**2/5 + rsd**2/10)**2 /
                        ((asd**2/5)**2/4 + (rsd**2/10)**2/9)})

    pd.DataFrame(rows).to_csv(open(OUT / "r9_al_seed_variance.csv", "w"),
                              index=False, float_format="%.6g")
    print("\n[完成] outputs/r9_al_seed_variance.csv", flush=True)


if __name__ == "__main__":
    main()
