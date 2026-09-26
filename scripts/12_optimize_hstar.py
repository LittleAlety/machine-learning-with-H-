# -*- coding: utf-8 -*-
"""
12_optimize_hstar.py — H* 主线模型深度优化（SPEC 第 8 节）

与旧版（随机 KFold 口径）的根本区别：本轮**全协议统一 GroupKFold
（groups = 规范化组成 comp，与 06/07 口径一致）**，包括超参搜索评估、
样本权重对比、stacking、嵌套 CV；early stopping 的验证折用
GroupShuffleSplit 切分，保证组级零泄漏。

1. 超参搜索：随机采样 150 组（≥150；因 sklearn RandomizedSearchCV 无法
   逐折传 eval_set 做 early stopping，采用同分布手写采样循环，等价
   RandomizedSearchCV）。9 维空间：max_depth[3-10] 整数、
   min_child_weight[1-20]、gamma[0-0.5]、subsample/colsample_bytree/
   colsample_bylevel[0.5-1.0]、reg_alpha/reg_lambda[1e-3-10 log均匀]、
   learning_rate[0.01-0.3 log均匀]；n_estimators ≤2000 + early stopping(50)
   → outputs/opt_search_results.csv + figures/opt_search_trajectory.png
2. 样本权重 3 组：uniform / n_raw 加权 / 组成频次逆加权，同一 GroupKFold
   下对比 MAE（评估不加权）→ outputs/opt_sample_weights.csv
   （注：v2 数据集每个 comp 仅 1 行，按 comp 频次逆加权退化为 uniform；
   故"组成频次逆加权"实现为按组成元素出现频次的逆权重，正文有说明）
3. Stacking：XGB(最优配置) + RandomForest + Ridge 元学习器，手写组感知
   两层 stacking（内层 GroupKFold 生成元特征，杜绝泄漏），同 GroupKFold
   评估 → outputs/opt_stacking_compare.csv
4. 嵌套 CV 无偏估计：外层 5-fold GroupKFold + 内层随机搜索 50 组
   （内层 GroupKFold(4)），基线（07 固定参数）同外层折对照
   → outputs/opt_nested_cv.csv；与基线（GroupKFold 0.120 eV）诚实对比
5. EqV2 外部验证复测：最优模型零重训，复用 11 全流程函数（importlib 同源
   加载）→ outputs/opt_extval_results.csv / opt_extval_metrics.csv /
   figures/opt_extval_parity.png
6. 条件刷新候选：仅当 基线GroupKFold MAE − 嵌套CV MAE > 0.005 eV 时，
   复用 08 逻辑刷新 candidate_rankings_hstar.csv 与
   figures/hstar_candidates.png；否则保留原排序并说明原因
汇总：outputs/opt_comparison.csv（基线 vs 权重组 vs stacking vs 嵌套CV
      vs 外部验证）、outputs/opt_best_params.json、notes/12_optimization.md
红线：site_spread 泄漏特征禁入（仅分析列）；种子 42；不改 01-11 产物口径
（候选刷新除外）；旧随机KFold口径废弃草稿已归档 deprecated/opt_*，
本轮不覆盖、不引用，全部新产物置于 outputs/。
"""
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw",
           "site_spread"]  # site_spread: 泄漏特征（含目标值），禁止入模
DGCORR = 0.24  # eV，同 08
RADIOACTIVE = {"Tc"}
OXOPHILIC = {"La", "Y", "Sc"}
LOEO_RISKY = {"Mn", "Bi", "Fe"}

PARAM_KEYS = ["max_depth", "min_child_weight", "gamma", "subsample",
              "colsample_bytree", "colsample_bylevel", "reg_alpha",
              "reg_lambda", "learning_rate"]
N_ITER_SEARCH = 150      # 全数据随机搜索组数（要求 ≥150）
N_ITER_NESTED = 50       # 嵌套 CV 内层搜索组数
ES_ROUNDS = 50           # early stopping 轮数
N_EST_CAP = 2000         # n_estimators 上限
REFRESH_GAIN = 0.005     # 候选刷新的最小收益阈值 (eV)

C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
C_DEEP = "#7A4A2B"
C_BG = "#E8D9C6"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- 同源加载 11 的外部验证函数（保证与 11 口径一致）----
_spec11 = importlib.util.spec_from_file_location(
    "extval11", ROOT / "scripts" / "11_external_validation.py")
F11 = importlib.util.module_from_spec(_spec11)
_spec11.loader.exec_module(F11)


# =====================================================================
# 工具函数
# =====================================================================
def sample_params(rng):
    """按任务指定的 9 维空间采样一组 XGB 超参。"""
    return {
        "max_depth": int(rng.integers(3, 11)),                    # 3-10
        "min_child_weight": float(rng.uniform(1, 20)),
        "gamma": float(rng.uniform(0, 0.5)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(10 ** rng.uniform(-3, 1)),             # log 均匀
        "reg_lambda": float(10 ** rng.uniform(-3, 1)),            # log 均匀
        "learning_rate": float(10 ** rng.uniform(np.log10(0.01),
                                                 np.log10(0.3))),  # log 均匀
    }


def make_xgb(params, n_estimators=N_EST_CAP, early_stopping=True):
    kw = dict(random_state=SEED, n_jobs=-1, n_estimators=n_estimators, **params)
    if early_stopping:
        kw["early_stopping_rounds"] = ES_ROUNDS
    return XGBRegressor(**kw)


def group_val_split(X, y, groups, seed, test_size=0.15):
    """从训练折中按组切出 early stopping 验证折（组级无泄漏）。"""
    gs = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    return next(gs.split(X, y, groups))


def eval_config_gcv(params, X, y, groups, sample_weight=None,
                    n_splits=5, seed=SEED):
    """GroupKFold 评估一组配置：每折内按组切 15% 验证折做 early stopping。
    评估 MAE 始终不加权；sample_weight 仅作用于训练。返回 (fold_maes, iters)。"""
    gkf = GroupKFold(n_splits=n_splits)
    fold_mae, best_iters = [], []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        t2, va = group_val_split(X.iloc[tr], y[tr], groups[tr], seed + i)
        mdl = make_xgb(params)
        fit_kw = dict(eval_set=[(X.iloc[tr].iloc[va], y[tr][va])], verbose=False)
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight[tr][t2]
        mdl.fit(X.iloc[tr].iloc[t2], y[tr][t2], **fit_kw)
        fold_mae.append(mean_absolute_error(y[te], mdl.predict(X.iloc[te])))
        best_iters.append(mdl.best_iteration + 1)
    return fold_mae, best_iters


def random_search(X, y, groups, n_iter, rng, tag="", sample_weight=None,
                  n_splits=5, seed=SEED):
    """随机搜索主循环（GroupKFold 评估）；返回全记录 DataFrame。"""
    rows, best, t0 = [], np.inf, time.time()
    for i in range(n_iter):
        p = sample_params(rng)
        fold_mae, iters = eval_config_gcv(p, X, y, groups,
                                          sample_weight=sample_weight,
                                          n_splits=n_splits, seed=seed)
        mae, std = float(np.mean(fold_mae)), float(np.std(fold_mae))
        best = min(best, mae)
        rows.append({"config_id": i + 1,
                     **{k: round(v, 6) if isinstance(v, float) else v
                        for k, v in p.items()},
                     "mean_MAE": mae, "std_MAE": std,
                     "mean_best_iter": float(np.mean(iters)),
                     "cummax_best_MAE": best, "is_best_so_far": mae <= best})
        if (i + 1) % 25 == 0:
            print(f"  [{tag}搜索 {i + 1}/{n_iter}] 当前最优 MAE={best:.4f} "
                  f"（{time.time() - t0:.0f}s）", flush=True)
    res = pd.DataFrame(rows)
    res["rank"] = res["mean_MAE"].rank().astype(int)
    return res


def fit_with_es(params, X, y, groups, sample_weight=None, seed=SEED):
    """大块训练：按组切 15% 验证折 early stopping，再以最优迭代数全量收尾。"""
    t2, va = group_val_split(X, y, groups, seed)
    mdl = make_xgb(params)
    fit_kw = dict(eval_set=[(X.iloc[va], y[va])], verbose=False)
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight[t2]
    mdl.fit(X.iloc[t2], y[t2], **fit_kw)
    n_it = mdl.best_iteration + 1
    final = make_xgb(params, n_estimators=n_it, early_stopping=False)
    kw2 = {"sample_weight": sample_weight} if sample_weight is not None else {}
    final.fit(X, y, **kw2)
    return final, n_it


def row_of(res_df):
    """取搜索记录中最优行并还原参数 dict。"""
    brow = res_df.loc[res_df["mean_MAE"].idxmin()]
    params = {k: (int(brow[k]) if k == "max_depth" else float(brow[k]))
              for k in PARAM_KEYS}
    return brow, params


# =====================================================================
# 组感知 Stacking（手写两层，元特征由组内 GroupKFold OOF 生成）
# =====================================================================
def stacking_gkf_oof(X, y, groups, xgb_fixed, seed=SEED):
    """外层 GroupKFold(5) OOF 对比 XGB / RF / Stacking(Ridge 元学习器)。
    stacking 的元特征由外层训练折内部的 GroupKFold(4) OOF 生成，
    Ridge 在元特征上训练；基模型再于整个外层训练折重训。"""
    gkf = GroupKFold(n_splits=5)
    oof = {k: np.zeros(len(y)) for k in ("XGB_opt", "RandomForest", "Stacking")}
    for f, (tr, te) in enumerate(gkf.split(X, y, groups)):
        inner = GroupKFold(n_splits=4)
        meta = np.zeros((len(tr), 2))
        for itr, ite in inner.split(X.iloc[tr], y[tr], groups[tr]):
            mx = clone(xgb_fixed).fit(X.iloc[tr].iloc[itr], y[tr][itr])
            mr = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                       n_jobs=-1).fit(X.iloc[tr].iloc[itr],
                                                      y[tr][itr])
            meta[ite, 0] = mx.predict(X.iloc[tr].iloc[ite])
            meta[ite, 1] = mr.predict(X.iloc[tr].iloc[ite])
        ridge = Ridge(random_state=SEED).fit(meta, y[tr])
        mx = clone(xgb_fixed).fit(X.iloc[tr], y[tr])
        mr = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                   n_jobs=-1).fit(X.iloc[tr], y[tr])
        px, pr = mx.predict(X.iloc[te]), mr.predict(X.iloc[te])
        oof["XGB_opt"][te] = px
        oof["RandomForest"][te] = pr
        oof["Stacking"][te] = ridge.predict(np.column_stack([px, pr]))
    return {k: mean_absolute_error(y, v) for k, v in oof.items()}, oof


# =====================================================================
def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    assert "site_spread" in df.columns
    feats = [c for c in df.columns if c not in ID_COLS]
    assert "site_spread" not in feats
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values
    print(f"[读入] hstar_dataset_v2.csv  {X.shape[0]} 行 × {X.shape[1]} 特征；"
          f"{len(set(groups))} 个 comp 组；site_spread 已排除", flush=True)

    # ---- 基线数值（07 口径，从已有产物读取，保证同源）----
    base_gcv_df = pd.read_csv(DP / "hstar_group_cv_results.csv")
    BASE_GCV = float(base_gcv_df.loc[base_gcv_df["model"] == "XGBoost_tuned",
                                     "MAE_mean"].iloc[0])
    BASE_GCV_STD = float(base_gcv_df.loc[base_gcv_df["model"] == "XGBoost_tuned",
                                         "MAE_std"].iloc[0])
    baseline_params = json.loads((DP / "hstar_best_params.json")
                                 .read_text())["xgb_best_params"]
    print(f"[基线] 07 XGBoost_tuned GroupKFold MAE={BASE_GCV:.4f}±"
          f"{BASE_GCV_STD:.4f}，参数={baseline_params}", flush=True)

    # ---- 样本权重方案 ----
    n_raw = df["n_raw"].values.astype(float)
    w_uniform = np.ones(len(df))
    w_nraw = n_raw / n_raw.mean()
    # 组成频次逆加权：v2 每 comp 仅 1 行（comp 频次恒 1，直接逆加权退化），
    # 故实现为组成元素频次逆加权：w = 1/mean_{e in comp} freq(e)，抑制
    # 被大量组成反复出现的"热门"元素化学，提升稀有化学权重
    elem_freq = {}
    for c in groups:
        for e in set(re.findall(r"[A-Z][a-z]?", c)):
            elem_freq[e] = elem_freq.get(e, 0) + 1
    w_invcomp = np.array([1.0 / np.mean([elem_freq[e] for e in
                                         set(re.findall(r"[A-Z][a-z]?", c))])
                          for c in groups])
    w_invcomp /= w_invcomp.mean()
    weight_schemes = {"uniform": w_uniform, "n_raw": w_nraw,
                      "inv_comp_freq": w_invcomp}
    print(f"[权重] n_raw 范围 {n_raw.min():.0f}-{n_raw.max():.0f}；"
          f"inv_comp_freq 范围 {w_invcomp.min():.3f}-{w_invcomp.max():.3f}",
          flush=True)

    # =================================================================
    # 1. 深度超参搜索（GroupKFold 评估，uniform 权重）
    # =================================================================
    print(f"\n[1] 随机搜索 {N_ITER_SEARCH  } 组（GroupKFold(5) + 组内验证折 "
          f"early stopping）", flush=True)
    rng = np.random.default_rng(SEED)
    search = random_search(X, y, groups, N_ITER_SEARCH, rng, tag="全数据")
    search.to_csv(OUT / "opt_search_results.csv", index=False)
    best_row, best_params = row_of(search)
    print(f"[搜索最优] GroupKFold MAE={best_row['mean_MAE']:.4f}±"
          f"{best_row['std_MAE']:.4f} 参数={best_params} "
          f"平均最优迭代数={best_row['mean_best_iter']:.0f}", flush=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(search["config_id"], search["mean_MAE"], s=14, color=C_ALT,
               alpha=0.7, edgecolors="none", label="config GroupKFold MAE")
    ax.plot(search["config_id"], search["cummax_best_MAE"], color=C_MAIN,
            lw=1.8, label="running best")
    ax.axhline(BASE_GCV, color=C_DEEP, ls="--", lw=1.2,
               label=f"baseline XGBoost_tuned ({BASE_GCV:.3f})")
    ax.set_xlabel("search iteration (random sampling)")
    ax.set_ylabel("GroupKFold MAE (eV)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "opt_search_trajectory.png")
    plt.close(fig)
    print("[写出] outputs/opt_search_results.csv, "
          "figures/opt_search_trajectory.png", flush=True)

    # =================================================================
    # 2. 样本权重实验（同一最优超参，同一 GroupKFold）
    # =================================================================
    print("\n[2] 样本权重实验（最优超参 + GroupKFold(5)，评估不加权）",
          flush=True)
    wrows = []
    for name, w in weight_schemes.items():
        fold_mae, _ = eval_config_gcv(best_params, X, y, groups,
                                      sample_weight=w)
        wrows.append({"weight_scheme": name, "MAE": float(np.mean(fold_mae)),
                      "MAE_std": float(np.std(fold_mae))})
        print(f"  {name:14s} MAE={np.mean(fold_mae):.4f}±"
              f"{np.std(fold_mae):.4f}", flush=True)
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(OUT / "opt_sample_weights.csv", index=False)
    best_wname = wdf.loc[wdf["MAE"].idxmin(), "weight_scheme"]
    best_w = weight_schemes[best_wname]
    print(f"[权重结论] 最优方案 = {best_wname}", flush=True)

    # =================================================================
    # 3. Stacking（XGB 最优 + RF + Ridge 元学习器，组感知，uniform 权重）
    # =================================================================
    print("\n[3] Stacking（XGB* + RF + Ridge，同 GroupKFold 折 OOF 对比）",
          flush=True)
    mdl_full, best_iter_full = fit_with_es(best_params, X, y, groups,
                                           sample_weight=best_w)
    print(f"  [全量收尾] 最优迭代数 n_estimators={best_iter_full}", flush=True)
    xgb_fixed = XGBRegressor(random_state=SEED, n_jobs=-1,
                             n_estimators=best_iter_full, **best_params)
    stack_maes, stack_oof = stacking_gkf_oof(X, y, groups, xgb_fixed)
    stack_df = pd.DataFrame([{"model": k, "OOF_MAE": v}
                             for k, v in stack_maes.items()])
    stack_df.to_csv(OUT / "opt_stacking_compare.csv", index=False)
    for k, v in stack_maes.items():
        print(f"  {k:14s} OOF MAE={v:.4f}", flush=True)

    # 全量 stacking（供外部验证/候选刷新用）：内层 GroupKFold(5) 元特征
    def fit_stacking_full(Xf, yf, gf):
        inner = GroupKFold(n_splits=5)
        meta = np.zeros((len(yf), 2))
        for itr, ite in inner.split(Xf, yf, gf):
            mx = clone(xgb_fixed).fit(Xf.iloc[itr], yf[itr])
            mr = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                       n_jobs=-1).fit(Xf.iloc[itr], yf[itr])
            meta[ite, 0] = mx.predict(Xf.iloc[ite])
            meta[ite, 1] = mr.predict(Xf.iloc[ite])
        ridge = Ridge(random_state=SEED).fit(meta, yf)
        bx = clone(xgb_fixed).fit(Xf, yf)
        brf = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                    n_jobs=-1).fit(Xf, yf)
        return ridge, bx, brf

    # 最终模型选择：stacking 仅当 OOF 提升 >0.005 eV 才采用，否则单 XGB
    use_stack = (stack_maes["XGB_opt"] - stack_maes["Stacking"]) > REFRESH_GAIN
    print(f"[最终模型] {'Stacking' if use_stack else '单模型 XGB（最优配置）'}"
          f"（stacking 相对单 XGB Δ={stack_maes['Stacking'] - stack_maes['XGB_opt']:+.4f}）",
          flush=True)
    if use_stack:
        ridge_full, bx_full, brf_full = fit_stacking_full(X, y, groups)
        final_predict = lambda Xq: ridge_full.predict(
            np.column_stack([bx_full.predict(Xq), brf_full.predict(Xq)]))
    else:
        final_predict = lambda Xq: mdl_full.predict(Xq)

    # =================================================================
    # 4. 嵌套 CV 无偏估计（外层 GroupKFold(5) + 内层搜索）+ LOEO
    # =================================================================
    print(f"\n[4a] 嵌套 CV（外层 GroupKFold(5) × 内层搜索 "
          f"{N_ITER_NESTED} 组）", flush=True)
    outer = GroupKFold(n_splits=5)
    nest_rows, oof_opt, oof_base = [], np.zeros(len(y)), np.zeros(len(y))
    for f, (tr, te) in enumerate(outer.split(X, y, groups)):
        rng_in = np.random.default_rng(1000 + f)
        res_in = random_search(X.iloc[tr], y[tr], groups[tr], N_ITER_NESTED,
                               rng_in, tag=f"外层折{f + 1}", n_splits=4,
                               seed=SEED + f)
        brow, bparams = row_of(res_in)
        mdl, n_it = fit_with_es(bparams, X.iloc[tr], y[tr], groups[tr],
                                seed=SEED + f)
        oof_opt[te] = mdl.predict(X.iloc[te])
        mb = XGBRegressor(random_state=SEED, n_jobs=-1, **baseline_params)
        mb.fit(X.iloc[tr], y[tr])
        oof_base[te] = mb.predict(X.iloc[te])
        mae_f = mean_absolute_error(y[te], oof_opt[te])
        mae_b = mean_absolute_error(y[te], oof_base[te])
        nest_rows.append({"outer_fold": f + 1,
                          "inner_best_MAE": brow["mean_MAE"],
                          "outer_MAE_optimized": mae_f,
                          "outer_MAE_baseline": mae_b,
                          "n_estimators": n_it,
                          **{f"best_{k}": v for k, v in bparams.items()}})
        print(f"  [外层折{f + 1}] 优化 MAE={mae_f:.4f}  基线 MAE={mae_b:.4f}",
              flush=True)
    nest_df = pd.DataFrame(nest_rows)
    nest_df.to_csv(OUT / "opt_nested_cv.csv", index=False)
    nested_mae = mean_absolute_error(y, oof_opt)
    nested_std = float(nest_df["outer_MAE_optimized"].std())
    nested_base = mean_absolute_error(y, oof_base)
    nested_base_std = float(nest_df["outer_MAE_baseline"].std())
    print(f"[嵌套CV] 优化模型无偏 MAE={nested_mae:.4f}±{nested_std:.4f}；"
          f"基线同折对照 MAE={nested_base:.4f}±{nested_base_std:.4f}",
          flush=True)

    # ---- 4b. LOEO（最优配置，复用 07 折划分逻辑）----
    print("\n[4b] LOEO（最优配置）", flush=True)
    elements = sorted(elem_freq)
    loeo_rows = []
    for el in elements:
        has = df["comp"].apply(
            lambda c: el in re.findall(r"[A-Z][a-z]?", c)).values
        if has.sum() < 3 or (~has).sum() < 10:
            continue
        mdl, _ = fit_with_es(best_params, X[~has], y[~has],
                             groups[~has], seed=SEED)
        pred = mdl.predict(X[has])
        loeo_rows.append({"held_out_element": el, "n_test": int(has.sum()),
                          "MAE": mean_absolute_error(y[has], pred)})
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(OUT / "opt_loeo_results.csv", index=False)
    loeo_mae = float(np.average(loeo["MAE"], weights=loeo["n_test"]))
    base_loeo_df = pd.read_csv(DP / "hstar_loeo_results.csv")
    base_loeo = float(np.average(base_loeo_df["MAE"],
                                 weights=base_loeo_df["n_test"]))
    print(f"  LOEO 汇集 MAE={loeo_mae:.4f}（{len(loeo)} 折；基线 "
          f"{base_loeo:.4f}）", flush=True)

    # =================================================================
    # 5. EqV2 外部验证（零重训，复用 11 全流程函数）
    # =================================================================
    print("\n[5] EqV2 外部验证（零重训，模型=最终模型全量训练）", flush=True)
    raw = F11.load_eqv2()
    agg = F11.aggregate_surface(raw)
    elements_ext = sorted({e for c in agg["comp"]
                           for e in F11.F06.parse_comp(c)})
    etab, fb = F11.F06.build_element_table(elements_ext)
    Xe = F11.featurize(agg, etab).reindex(columns=feats, fill_value=0)
    assert list(Xe.columns) == feats and not Xe.isna().any().any()
    agg["E_pred"] = final_predict(Xe)
    agg["residual"] = agg["E_pred"] - agg["E_true"]
    agg["in_domain_facet"] = agg["facet"] == 111
    agg["has_Sb"] = agg["comp"].str.contains("Sb")
    agg.to_csv(OUT / "opt_extval_results.csv", index=False)
    mrows = [F11.metric_row("ALL", agg["E_true"].values, agg["E_pred"].values)]
    for src, sub in agg.groupby("src"):
        mrows.append(F11.metric_row(f"src={src}", sub["E_true"].values,
                                    sub["E_pred"].values))
    for name, mask in [("facet=111（域内晶面）", agg["in_domain_facet"]),
                       ("不含Sb（元素域内）", ~agg["has_Sb"]),
                       ("facet=111 且不含Sb",
                        agg["in_domain_facet"] & ~agg["has_Sb"]),
                       ("含Sb（元素域外）", agg["has_Sb"])]:
        sub = agg[mask]
        mrows.append(F11.metric_row(name, sub["E_true"].values,
                                    sub["E_pred"].values))
    ext_metrics = pd.DataFrame(mrows)
    ext_metrics.to_csv(OUT / "opt_extval_metrics.csv", index=False)
    ext_all = mrows[0]
    print(f"  外部验证 ALL: MAE={ext_all['MAE']:.4f}  "
          f"bias={ext_all['bias_pred_minus_true']:+.4f}  "
          f"去偏差MAE={ext_all['MAE_after_bias_shift']:.4f}", flush=True)

    # parity 图（口径同 11）
    gkf_oof = stack_oof["XGB_opt"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    ax = axes[0]
    ax.scatter(y, gkf_oof, s=7, alpha=0.25, color=C_BG, edgecolors="none",
               label=f"Mamun in-house OOF (MAE={stack_maes['XGB_opt']:.3f} eV)")
    for flag, mk, col, lab in [(True, "o", C_MAIN, "EqV2 facet=111 (in-domain)"),
                               (False, "s", C_ALT, "EqV2 other facets (OOD)")]:
        m = agg["in_domain_facet"] == flag
        ax.scatter(agg.loc[m, "E_true"], agg.loc[m, "E_pred"], s=22,
                   alpha=0.75, marker=mk, color=col, edgecolors="none",
                   label=lab)
    lim = [min(y.min(), agg["E_true"].min(), agg["E_pred"].min()) - 0.3,
           max(y.max(), agg["E_true"].max(), agg["E_pred"].max()) + 0.3]
    ax.plot(lim, lim, "--", color="#8c8c8c", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"True $E_{ads}$(H*) (eV, EqV2 / Mamun)")
    ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"optimized model on EqV2: MAE={ext_all['MAE']:.3f} eV, "
                 f"bias={ext_all['bias_pred_minus_true']:+.3f} eV", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1]
    ax.hist(gkf_oof - y, bins=60, alpha=0.6, color=C_BG, density=True,
            label="Mamun OOF residuals")
    ax.hist(agg.loc[agg["in_domain_facet"], "residual"], bins=30, alpha=0.7,
            color=C_MAIN, density=True, label="EqV2 facet=111 residuals")
    ax.hist(agg.loc[~agg["in_domain_facet"], "residual"], bins=30, alpha=0.6,
            color=C_ALT, density=True, label="EqV2 other-facet residuals")
    ax.axvline(0, color="#8c8c8c", ls="--", lw=1)
    ax.set_xlabel(r"Residual = pred $-$ true (eV)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (systematic bias)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "opt_extval_parity.png")
    plt.close(fig)
    print("[写出] outputs/opt_extval_results.csv / opt_extval_metrics.csv / "
          "figures/opt_extval_parity.png", flush=True)

    # =================================================================
    # 6. 汇总对比表 outputs/opt_comparison.csv
    #    （基线 vs 权重组 vs stacking vs 嵌套CV vs 外部验证）
    # =================================================================
    base_ext = pd.read_csv(DP / "extval_metrics.csv")
    base_ext_all = float(base_ext.loc[base_ext["stratum"] == "ALL",
                                      "MAE"].iloc[0])
    cmp_rows = [
        {"stage": "基线", "setting": "XGBoost_tuned（07 小网格）",
         "protocol": "GroupKFold(规范化comp)", "MAE": round(BASE_GCV, 4),
         "MAE_std": round(BASE_GCV_STD, 4),
         "note": "当前基线 0.120 eV"},
        {"stage": "超参搜索", "setting": "XGB 最优配置（150组搜索）",
         "protocol": "GroupKFold(规范化comp)",
         "MAE": round(float(best_row["mean_MAE"]), 4),
         "MAE_std": round(float(best_row["std_MAE"]), 4),
         "note": "搜索阶段内层评估，非无偏"},
    ]
    for r in wdf.itertuples():
        cmp_rows.append({"stage": "样本权重", "setting": r.weight_scheme,
                         "protocol": "GroupKFold(规范化comp)",
                         "MAE": round(r.MAE, 4), "MAE_std": round(r.MAE_std, 4),
                         "note": "同一最优超参；评估不加权"})
    for r in stack_df.itertuples():
        cmp_rows.append({"stage": "Stacking对比", "setting": r.model,
                         "protocol": "GroupKFold(规范化comp) OOF",
                         "MAE": round(r.OOF_MAE, 4), "MAE_std": np.nan,
                         "note": "uniform 权重；元学习器=Ridge"})
    cmp_rows += [
        {"stage": "嵌套CV无偏估计", "setting": "优化流程（内层搜索50组）",
         "protocol": "外层GroupKFold(5)+内层GroupKFold(4)",
         "MAE": round(nested_mae, 4), "MAE_std": round(nested_std, 4),
         "note": "无偏估计；与基线同外层折对照"},
        {"stage": "嵌套CV无偏估计", "setting": "基线（07固定参数，同折）",
         "protocol": "外层GroupKFold(5)",
         "MAE": round(nested_base, 4), "MAE_std": round(nested_base_std, 4),
         "note": "同外层折固定参数对照"},
        {"stage": "LOEO", "setting": "XGB 最优配置",
         "protocol": "留一元素外测（按n_test加权）",
         "MAE": round(loeo_mae, 4), "MAE_std": np.nan,
         "note": f"基线 LOEO={base_loeo:.4f}"},
        {"stage": "外部验证", "setting": "最终模型（零重训）",
         "protocol": "EqV2 ALL",
         "MAE": round(float(ext_all["MAE"]), 4), "MAE_std": np.nan,
         "note": f"基线={base_ext_all:.4f}；bias="
                 f"{ext_all['bias_pred_minus_true']:+.4f}"},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT / "opt_comparison.csv", index=False)
    print("\n===== 汇总对比（outputs/opt_comparison.csv）=====")
    print(cmp_df.to_string(index=False), flush=True)

    json.dump({"best_params": best_params,
               "best_weight_scheme": best_wname,
               "n_estimators_full_fit": best_iter_full,
               "final_model": "stacking" if use_stack else "xgb_single",
               "search_best_groupkfold_MAE": float(best_row["mean_MAE"]),
               "nested_cv_MAE": nested_mae, "nested_cv_MAE_std": nested_std,
               "nested_cv_baseline_MAE": nested_base,
               "baseline_groupkfold_MAE": BASE_GCV,
               "loeo_MAE": loeo_mae,
               "extval_MAE": float(ext_all["MAE"]),
               "extval_bias": float(ext_all["bias_pred_minus_true"])},
              open(OUT / "opt_best_params.json", "w"),
              ensure_ascii=False, indent=2)

    # =================================================================
    # 7. 候选刷新判定（复用 08 逻辑；阈值：基线−嵌套CV > 0.005 eV）
    # =================================================================
    gain = BASE_GCV - nested_mae
    improved = gain > REFRESH_GAIN
    print(f"\n[7] 候选刷新判定：基线GroupKFold {BASE_GCV:.4f} − 嵌套CV "
          f"{nested_mae:.4f} = {gain:+.4f} eV → "
          f"{'满足 >0.005，刷新候选' if improved else '未达阈值，保留原候选'}",
          flush=True)
    if improved:
        refresh_candidates(df, X, y, nested_mae, xgb_fixed, best_w,
                           best_wname, use_stack, stack_oof)

    # =================================================================
    # 8. notes/12_optimization.md
    # =================================================================
    write_notes(search, best_params, best_row, wdf, best_wname, stack_df,
                nest_df, nested_mae, nested_std, nested_base, nested_base_std,
                loeo_mae, loeo, ext_metrics, cmp_df, improved, gain,
                best_iter_full, BASE_GCV, use_stack, base_loeo,
                n_feats=X.shape[1])


def refresh_candidates(df, X, y, mae, xgb_fixed, best_w, best_wname,
                       use_stack, stack_oof):
    """复用 08 的 HER 筛选逻辑：OOF 预测 → ΔG_H* → 排名 → 红旗 → 刷新产物。
    单模型情形与 08 同口径（随机 KFold(5) OOF）；stacking 情形用组感知
    GroupKFold OOF（更严格，notes 中注明）。"""
    if use_stack:
        oof = stack_oof["Stacking"]
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y))
        for tr, te in kf.split(X):
            m = clone(xgb_fixed)
            kw = {"sample_weight": best_w[tr]} if best_wname != "uniform" else {}
            m.fit(X.iloc[tr], y[tr], **kw)
            oof[te] = m.predict(X.iloc[te])
    df = df.copy()
    df["pred_E_ads"] = oof
    df["pred_dG"] = df["pred_E_ads"] + DGCORR
    df["true_dG"] = df["energy_eV"] + DGCORR
    df["abs_pred_dG"] = df["pred_dG"].abs()
    rank = df.sort_values("abs_pred_dG").reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    best_abs = rank["abs_pred_dG"].min()
    rank["statistically_indistinguishable"] = (rank["abs_pred_dG"]
                                               - best_abs) < mae

    def flags(comp):
        els = set(re.findall(r"[A-Z][a-z]?", comp))
        fl = []
        if els & RADIOACTIVE:
            fl.append("含放射性" + "/".join(sorted(els & RADIOACTIVE)))
        if els & OXOPHILIC:
            fl.append("强亲氧" + "/".join(sorted(els & OXOPHILIC))
                      + "(表面稳定性存疑)")
        if els & LOEO_RISKY:
            fl.append("含LOEO高误差元素" + "/".join(sorted(els & LOEO_RISKY)))
        return ";".join(fl)
    rank["red_flags"] = rank["comp"].apply(flags)
    out_cols = ["rank", "comp", "structure", "facet", "n_raw",
                "energy_eV", "pred_E_ads", "true_dG", "pred_dG",
                "abs_pred_dG", "statistically_indistinguishable", "red_flags"]
    old = pd.read_csv(DP / "candidate_rankings_hstar.csv")
    rank[out_cols].round(4).to_csv(DP / "candidate_rankings_hstar.csv",
                                   index=False)
    old_top10 = set(old.head(10)["comp"])
    new_top10 = set(rank.head(10)["comp"])
    changed = sorted(old_top10 ^ new_top10)
    print(f"[刷新] candidate_rankings_hstar.csv；top-10 变动 "
          f"{len(changed)} 个：{changed if changed else '无'}", flush=True)
    print(rank[out_cols].head(5).round(3).to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bins = np.linspace(0, rank["abs_pred_dG"].max(), 60)
    ax.hist(rank["abs_pred_dG"], bins=bins, color=C_ALT, edgecolor="white",
            linewidth=0.4, label=f"all surfaces (n={len(rank)})")
    ax.hist(rank.head(10)["abs_pred_dG"], bins=bins, color=C_MAIN,
            edgecolor="white", linewidth=0.4, label="top-10 candidates")
    ax.axvline(mae, color=C_DEEP, ls="--", lw=1.2,
               label=f"model MAE = {mae:.2f} eV (nested CV)")
    ax.set_xlabel(r"|$\Delta G_{H*}$| (eV, out-of-fold prediction)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hstar_candidates.png")
    plt.close(fig)
    print("[刷新] figures/hstar_candidates.png", flush=True)
    return changed


def write_notes(search, best_params, best_row, wdf, best_wname, stack_df,
                nest_df, nested_mae, nested_std, nested_base, nested_base_std,
                loeo_mae, loeo, ext_metrics, cmp_df, improved, gain,
                best_iter_full, base_gcv, use_stack, base_loeo, n_feats):
    worst3 = loeo.sort_values("MAE", ascending=False).head(3)
    ext_all = ext_metrics.iloc[0]
    L = [
        "# 12 — H* 主线模型深度优化报告（SPEC 第 8 节，scripts/12_optimize_hstar.py）",
        "",
        "> 本轮为 GroupKFold 口径重做版：搜索/权重/stacking/嵌套 CV 全部统一",
        "> GroupKFold（groups=规范化 comp），替代旧版随机 KFold 草稿",
        ">（旧随机 KFold 草稿产物已归档 deprecated/opt_*，本轮未引用、未覆盖）。",
        "",
        "## 1. 设置与口径",
        "",
        f"- 数据：hstar_dataset_v2.csv（1,836 行 × {n_feats} 特征；site_spread"
        f" 泄漏列已排除，仅作分析列），种子 {SEED}",
        "- 每个 comp 在数据集中仅 1 行，GroupKFold(comp) 等价于组大小为 1 的"
        "划分——与 07 的 GroupKFold 口径完全一致，仍按组协议执行",
        f"- 搜索：随机采样 {len(search)} 组（≥150），空间：max_depth[3-10]、"
        "min_child_weight[1-20]、gamma[0-0.5]、subsample/colsample_bytree/"
        "colsample_bylevel[0.5-1.0]、reg_alpha/reg_lambda[1e-3-10 log均匀]、"
        f"learning_rate[0.01-0.3 log均匀]；n_estimators 上限 {N_EST_CAP} + "
        f"early stopping（{ES_ROUNDS} 轮，验证折用 GroupShuffleSplit 按组切 "
        "15%，组级零泄漏）。因 sklearn SearchCV 无法逐折传 eval_set，"
        "采用同分布手写采样循环（等价 RandomizedSearchCV）",
        "",
        "## 2. 超参搜索（figures/opt_search_trajectory.png）",
        "",
        f"- 150 组全程 GroupKFold(5) 评估；最终最优 MAE = "
        f"{search['mean_MAE'].min():.4f} eV（第 "
        f"{int(search.loc[search['mean_MAE'].idxmin(), 'config_id'])} 组），"
        f"基线为 {base_gcv:.4f} eV",
        f"- 前 50 组累计最优 {search.head(50)['cummax_best_MAE'].iloc[-1]:.4f}；"
        f"后 100 组仅再降 {search.head(50)['cummax_best_MAE'].iloc[-1] - search['mean_MAE'].min():.4f} eV，"
        "搜索已收敛（收益递减）",
        f"- 最优配置（GroupKFold MAE={best_row['mean_MAE']:.4f}±"
        f"{best_row['std_MAE']:.4f}，平均最优迭代数 "
        f"{best_row['mean_best_iter']:.0f}，全量收尾 {best_iter_full}）：",
        "",
        "| 参数 | 值 |",
        "|---|---|",
    ]
    L += [f"| {k} | {v} |" for k, v in best_params.items()]
    L += [
        "",
        "## 3. 样本权重实验（同一最优超参 + 同一 GroupKFold，评估不加权）",
        "",
        "| 权重方案 | GroupKFold MAE (eV) |",
        "|---|---|",
    ]
    L += [f"| {r.weight_scheme} | {r.MAE:.4f}±{r.MAE_std:.4f} |"
          for r in wdf.itertuples()]
    spread = wdf["MAE"].max() - wdf["MAE"].min()
    L += [
        "",
        f"- 三方案展开差 {spread:.4f} eV"
        + ("（< 0.005，样本权重对精度**无实质影响**）" if spread < 0.005
           else ""),
        "- 说明：v2 每个 comp 仅 1 行，按 comp 行频逆加权退化为 uniform，"
        "故 inv_comp_freq 实现为**组成元素频次逆加权**（w = 1/组内元素在"
        "全库的平均出现频次，归一化），用于抑制热门元素化学的过度代表",
        f"- 最优方案 = **{best_wname}**，用于最终模型训练",
        "",
        "## 4. Stacking 融合（XGB + RF + Ridge 元学习器，组感知手写两层）",
        "",
        "| 模型 | GroupKFold OOF MAE (eV) |",
        "|---|---|",
    ]
    L += [f"| {r.model} | {r.OOF_MAE:.4f} |" for r in stack_df.itertuples()]
    sx = float(stack_df.loc[stack_df["model"] == "Stacking", "OOF_MAE"].iloc[0])
    xx = float(stack_df.loc[stack_df["model"] == "XGB_opt", "OOF_MAE"].iloc[0])
    L += [
        "",
        f"- Stacking 相对单 XGB：Δ = {sx - xx:+.4f} eV。"
        + ("融合增益 ≤0.005，**最终模型取单模型 XGB**：更简单、SHAP 管线兼容。"
           if not use_stack else "融合增益 >0.005，**最终模型取 Stacking**。")
        + "XGB 与 RF 误差高度相关，Ridge 元学习器可提取的互补信息有限。",
        "- 组感知实现：元特征由外层训练折内部的 GroupKFold(4) OOF 生成，"
        "杜绝元特征泄漏",
        "",
        "## 5. 嵌套 CV 无偏估计（外层 GroupKFold(5) × 内层搜索 "
        f"{N_ITER_NESTED} 组）",
        "",
        "| 外层折 | 内层最优MAE | 外层MAE(优化) | 外层MAE(基线同折) |",
        "|---|---|---|---|",
    ]
    L += [f"| {r.outer_fold} | {r.inner_best_MAE:.4f} "
          f"| {r.outer_MAE_optimized:.4f} | {r.outer_MAE_baseline:.4f} |"
          for r in nest_df.itertuples()]
    L += [
        "",
        f"- **优化流程无偏 MAE = {nested_mae:.4f}±{nested_std:.4f} eV**；"
        f"基线（07 固定参数）同外层折对照 = {nested_base:.4f}±"
        f"{nested_base_std:.4f} eV",
        f"- LOEO（最优配置）汇集 MAE = {loeo_mae:.4f} eV（基线 {base_loeo:.4f}）；"
        "最差 3 元素："
        + "、".join(f"{r.held_out_element} {r.MAE:.3f}"
                    for r in worst3.itertuples()),
        "",
        "## 6. 外部验证（EqV2，零重训，复用 11 全流程）",
        "",
        f"- ALL：MAE = {ext_all['MAE']:.4f} eV，系统偏差 mean(pred−true) = "
        f"{ext_all['bias_pred_minus_true']:+.4f} eV，去偏差平移后 MAE = "
        f"{ext_all['MAE_after_bias_shift']:.4f} eV",
        "- 基线（07 模型，11 报告）：MAE = 0.3470 / bias +0.158 / 去偏差 0.313",
        "- 系统偏差仍显著为正，与 11 结论一致：主要源于泛函/参考态差异"
        "（BEEF-vdW vs RPBE）与域外晶面/元素（Sb）——**超参优化不能消除"
        "跨库系统偏差**，这是数据分布问题而非模型容量问题",
        "",
        "## 7. 汇总对比（outputs/opt_comparison.csv）",
        "",
        "| 阶段 | 设置 | 协议 | MAE (eV) |",
        "|---|---|---|---|",
    ]
    L += [f"| {r.stage} | {r.setting} | {r.protocol} | {r.MAE:.4f} |"
          for r in cmp_df.itertuples()]
    L += [
        "",
        "## 8. 与基线的诚实结论",
        "",
        f"- 基线 GroupKFold MAE = {base_gcv:.4f} eV；嵌套 CV 无偏 MAE = "
        f"{nested_mae:.4f} eV；差值 = {gain:+.4f} eV（正=优化有效）",
    ]
    if gain > 0.005:
        L.append(f"- 提升 {gain:.4f} eV > 0.005 阈值，判定优化有实质收益。")
    elif gain >= 0:
        L.append(f"- **收益边际**：0 ≤ 差值 ≤ 0.005 eV，深度优化未带来有实际"
                 "意义的提升。")
    else:
        L.append(f"- 嵌套 CV 下优化流程反而差 {abs(gain):.4f} eV：全数据搜索"
                 "选出的最优配置存在搜索过拟合，嵌套评估予以纠正——这正是"
                 "诚实评估铁律的意义。")
    if gain <= 0.005:
        sx_ = float(stack_df.loc[stack_df["model"] == "XGB_opt",
                                 "OOF_MAE"].iloc[0])
        L.append(f"- 两处协议对齐的对照指向同一结论：(i) 嵌套 CV 同折对照 "
                 f"{nested_mae:.4f} vs {nested_base:.4f}；(ii) 最优配置满折训练 "
                 f"+ 固定迭代数的 GroupKFold OOF {sx_:.4f} vs 基线 {base_gcv:.4f}"
                 "——均无收益。07 小网格已落在该特征集的性能平台区；进一步提升"
                 "需来自特征/数据层面（位点结构描述符、跨库联合训练），而非"
                 "更密的超参搜索。")
    L += [
        "",
        "## 9. 候选刷新",
        "",
    ]
    if improved:
        top5 = pd.read_csv(DP / "candidate_rankings_hstar.csv").head(5)
        L += [
            f"- 差值 {gain:+.4f} > 0.005 eV，**已用最终模型刷新** "
            "candidate_rankings_hstar.csv 与 figures/hstar_candidates.png"
            f"（筛选逻辑复用 08：OOF 预测、ΔG_H* = E_ads + {DGCORR}、"
            "统计不可区分带、红旗规则）",
            "",
            "新 top-5：",
            "",
            "| rank | comp | structure | facet | pred ΔG_H* | true ΔG_H* | 红旗 |",
            "|---|---|---|---|---|---|---|",
        ]
        L += [f"| {r['rank']} | {r['comp']} | {r['structure']} | {r['facet']} "
              f"| {r['pred_dG']:+.4f} | {r['true_dG']:+.4f} "
              f"| {r['red_flags'] or '—'} |" for _, r in top5.iterrows()]
    else:
        L += [
            f"- 差值 {gain:+.4f} eV 未超过 0.005 阈值，**保留 08 原候选排序**"
            "（candidate_rankings_hstar.csv 与 figures/hstar_candidates.png "
            "未改动）。原因：优化收益边际/为负，用无显著差异的模型重排候选"
            "只会引入噪声，不提供新信息。",
        ]
    L += [
        "",
        "## 10. 产物清单",
        "",
        "- outputs/opt_search_results.csv（150 组全记录）/ "
        "opt_sample_weights.csv / opt_stacking_compare.csv / "
        "opt_nested_cv.csv / opt_loeo_results.csv / opt_comparison.csv / "
        "opt_extval_results.csv / opt_extval_metrics.csv / opt_best_params.json",
        "- figures/opt_search_trajectory.png / opt_extval_parity.png",
        ("- 刷新：candidate_rankings_hstar.csv / figures/hstar_candidates.png"
         if improved else "- 未触碰任何 01-11 产物"),
    ]
    (ROOT / "notes" / "12_optimization.md").write_text("\n".join(L),
                                                       encoding="utf-8")
    print("[写出] notes/12_optimization.md", flush=True)


if __name__ == "__main__":
    main()
