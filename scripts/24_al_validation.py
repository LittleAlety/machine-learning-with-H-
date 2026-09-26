# -*- coding: utf-8 -*-
"""
24_al_validation.py — 主动学习补点价值的双重验证（回溯式内部模拟 + 外部真实点代理）

任务 A（回溯式 AL 验证，内部模拟"补算 DFT 点"）：
  - 从 1,836 组成中留出 200 个组成（固定种子，按 comp 分组）作"假想未算池"，
    其余 1,636 为初始训练集；
  - 基线 XGBoost 在 1,636 上训练；用 21 号脚本的分位数不确定度管线
    （q05/q95 → width）对池中 200 点计算采集分 = width × exp(−|pred_ΔG|/0.1)；
  - 补点策略对比：AL top-k vs 随机补点（10 个随机种子取均值±std），k∈{5,10,15,20}；
  - 每步把选中点"揭示"加入训练集重训，在池中剩余点上测 MAE；
  - 判定：AL 在 k=10/20 时若 rand_mean − al_mae > rand_std 则判显著优于随机。

任务 B（外部代理点验证，extval2 同泛函子集充当"新补算点"）：
  - 全量基线（1,836 训练）对 extval2 同泛函 BEEF-vdW 子集计算采集分；
  - top-10 / top-20 加入训练重训，在剩余外部点上测 MAE（补点前 vs +10 vs +20）；
  - 附随机补点对照（10 种子均值±std），诚实报告升/降与被选点化学身份。

产物：
  outputs/alval_A_pool_acquisition.csv    池 200 点的预测/区间/采集分
  outputs/alval_A_learning_curve.csv      k × (AL MAE, 随机均值±std, 显著性)
  outputs/alval_A_selected.csv            各 k 被 AL 选中的池点
  outputs/alval_B_extval_acquisition.csv  extval2-BEEF 115 点的预测/区间/采集分
  outputs/alval_B_metrics.csv             补点前后外部 MAE（含随机对照）
  outputs/alval_B_selected.csv            top-10/20 被选点明细（化学身份）
  outputs/alval_summary.json              关键指标汇总
  figures/alval_learning_curve.png        学习曲线（300dpi 暖色系英文标注）

注意：/mnt/agents/output 挂载对 rename/原子写可能 Permission denied，统一用普通 open(...,'w')。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
FIGDIR = ROOT / "figures"
SEED = 42
POOL_SEED = 123        # 任务 A 留池种子（固定）
N_POOL = 200
K_LIST = [5, 10, 15, 20]
N_RAND = 10            # 随机补点种子数（≥10）
RAND_SEEDS = [1000 + i for i in range(N_RAND)]
DG_SHIFT = 0.24        # ΔG_H* = E_ads + 0.24 eV
VOLCANO_SIGMA = 0.1

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
                  subsample=0.8, random_state=SEED)

C_MAIN = "#B35C24"   # burnt orange
C_ALT = "#DFB27E"    # tan
C_GRAY = "#9b9187"   # warm gray
C_DARK = "#5c4a38"   # warm dark brown
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


# ---------------------------------------------------------------- 模型封装（同 21）
def fit_base(X, y, seed=SEED):
    m = XGBRegressor(**{**XGB_PARAMS, "random_state": seed}, n_jobs=-1)
    m.fit(X, y)
    return m


def fit_quantile(X, y, alpha):
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                     **XGB_PARAMS, n_jobs=-1)
    m.fit(X, y)
    return m


def acquisition(pred, width):
    """采集分 = width × exp(−|pred_ΔG|/0.1)，ΔG = E_ads + 0.24 eV（同 21）。"""
    dG = pred + DG_SHIFT
    return dG, width * np.exp(-np.abs(dG) / VOLCANO_SIGMA)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------- extval2 特征化（同 14 口径）
_spec = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
F06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F06)
F06.HAND_LOOKUP.update({
    "Nd": dict(en=1.14, radius=181.0, group=3, period=6, ie1=5.469, d_el=0),
    "Sm": dict(en=1.17, radius=180.0, group=3, period=6, ie1=5.631, d_el=0),
})


def featurize_ext(agg, feats):
    """与 scripts/14 的 featurize 完全同口径：单元素→A1=1，合金→structure 全 0；
    facet 仅精确 '101'/'111' 置 1，其余晶面/应变标注全 0（域外外推）。"""
    elements = sorted({e for c in agg["comp"] for e in F06.parse_comp(c)})
    etab, fb = F06.build_element_table(elements)
    if fb:
        print(f"  [兜底] mendeleev 缺失启用 HAND_LOOKUP: {fb}")
    comp_dicts = {c: F06.parse_comp(c) for c in agg["comp"].unique()}
    rows = []
    for _, row in agg.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        f = {"n_elements": len(els)}
        for p in F06.PROPS:
            v = np.array([etab[e][p] for e in els])
            f[f"{p}_wmean"] = F06.wmean(v, w)
            f[f"{p}_max"] = float(v.max())
            f[f"{p}_min"] = float(v.min())
            f[f"{p}_range"] = float(v.max() - v.min())
            f[f"{p}_wstd"] = F06.wstd(v, w)
        f.update({"structure_A1": 1 if len(els) == 1 else 0,
                  "structure_L10": 0, "structure_L12": 0,
                  "facet_101": 1 if row["facet"] == "101" else 0,
                  "facet_111": 1 if row["facet"] == "111" else 0})
        rows.append(f)
    Xe = pd.DataFrame(rows, index=agg.index)
    return Xe.reindex(columns=feats, fill_value=0)


# ---------------------------------------------------------------- 任务 A
def task_A(df, feats):
    print("=" * 70)
    print("[任务 A] 回溯式主动学习验证（留出 200 组成作假想未算池）")
    comps = np.array(sorted(df["comp"].unique()))
    rng = np.random.RandomState(POOL_SEED)
    perm = rng.permutation(len(comps))
    pool_comps = set(comps[perm[:N_POOL]])
    train_df = df[~df["comp"].isin(pool_comps)].reset_index(drop=True)
    pool_df = df[df["comp"].isin(pool_comps)].reset_index(drop=True)
    print(f"  初始训练 {len(train_df)} 行（{train_df['comp'].nunique()} 组成），"
          f"池 {len(pool_df)} 行（{pool_df['comp'].nunique()} 组成），种子 {POOL_SEED}")

    Xtr, ytr = train_df[feats], train_df["energy_eV"].values
    Xp, yp = pool_df[feats], pool_df["energy_eV"].values

    base0 = fit_base(Xtr, ytr)
    q05 = fit_quantile(Xtr, ytr, 0.05).predict(Xp)
    q95 = fit_quantile(Xtr, ytr, 0.95).predict(Xp)
    pred0 = base0.predict(Xp)
    width = q95 - q05
    dG, score = acquisition(pred0, width)
    mae0 = mae(yp, pred0)
    print(f"  [k=0] 池 MAE = {mae0:.4f} eV（补点前）")

    pool_df = pool_df.assign(pred=pred0, q05=q05, q95=q95, width=width,
                             pred_dG=dG, acquisition_score=score)
    pool_df[["comp", "structure", "facet", "energy_eV", "pred", "q05", "q95",
             "width", "pred_dG", "acquisition_score"]].to_csv(
        open(OUT / "alval_A_pool_acquisition.csv", "w"),
        index=False, float_format="%.5f")

    al_order = pool_df.sort_values("acquisition_score", ascending=False)
    width_order = pool_df.sort_values("width", ascending=False)  # 消融：纯不确定度
    curve_rows, sel_rows = [{"k": 0, "n_remaining": N_POOL, "al_mae": mae0,
                             "alw_mae": mae0,
                             "rand_mae_mean": mae0, "rand_mae_std": 0.0,
                             "rand_mae_min": mae0, "rand_mae_max": mae0,
                             "diff_rand_minus_al": 0.0,
                             "diff_rand_minus_alw": 0.0,
                             "significant": "", "significant_width": ""}], []

    for k in K_LIST:
        # ---- AL top-k（主协议：采集分 = width × 火山邻近核）----
        sel_al = al_order.head(k)
        for _, r in sel_al.iterrows():
            sel_rows.append({"k": k, "strategy": "AL", "comp": r["comp"],
                             "structure": r["structure"], "facet": r["facet"],
                             "energy_eV": r["energy_eV"], "pred": r["pred"],
                             "width": r["width"], "pred_dG": r["pred_dG"],
                             "acquisition_score": r["acquisition_score"],
                             "seed": ""})
        Xaug = pd.concat([Xtr, sel_al[feats]])
        yaug = np.concatenate([ytr, sel_al["energy_eV"].values])
        rest_al = pool_df.drop(index=sel_al.index)
        m_al = mae(rest_al["energy_eV"].values,
                   fit_base(Xaug, yaug).predict(rest_al[feats]))

        # ---- 消融：纯不确定度 top-k（width only，无火山核）----
        sel_w = width_order.head(k)
        Xw = pd.concat([Xtr, sel_w[feats]])
        yw = np.concatenate([ytr, sel_w["energy_eV"].values])
        rest_w = pool_df.drop(index=sel_w.index)
        m_alw = mae(rest_w["energy_eV"].values,
                    fit_base(Xw, yw).predict(rest_w[feats]))

        # ---- 随机 k × N_RAND 种子 ----
        rand_maes = []
        for sd in RAND_SEEDS:
            rsel = pool_df.sample(n=k, random_state=sd)
            Xr = pd.concat([Xtr, rsel[feats]])
            yr = np.concatenate([ytr, rsel["energy_eV"].values])
            rrest = pool_df.drop(index=rsel.index)
            rand_maes.append(mae(rrest["energy_eV"].values,
                                 fit_base(Xr, yr).predict(rrest[feats])))
        rand_maes = np.array(rand_maes)
        rmean, rstd = rand_maes.mean(), rand_maes.std(ddof=1)
        diff = rmean - m_al
        diff_w = rmean - m_alw
        sig = "YES" if diff > rstd else "no"
        sig_w = "YES" if diff_w > rstd else "no"
        curve_rows.append({"k": k, "n_remaining": N_POOL - k, "al_mae": m_al,
                           "alw_mae": m_alw,
                           "rand_mae_mean": rmean, "rand_mae_std": rstd,
                           "rand_mae_min": rand_maes.min(),
                           "rand_mae_max": rand_maes.max(),
                           "diff_rand_minus_al": diff,
                           "diff_rand_minus_alw": diff_w,
                           "significant": sig, "significant_width": sig_w})
        print(f"  [k={k:2d}] 剩余 {N_POOL-k:3d} | AL MAE={m_al:.4f} | "
              f"AL(width-only) MAE={m_alw:.4f} | "
              f"随机 MAE={rmean:.4f}±{rstd:.4f} (min {rand_maes.min():.4f} / "
              f"max {rand_maes.max():.4f}) | diff={diff:+.4f} 显著={sig} | "
              f"diff_w={diff_w:+.4f} 显著={sig_w}")

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(open(OUT / "alval_A_learning_curve.csv", "w"),
                 index=False, float_format="%.5f")
    pd.DataFrame(sel_rows).to_csv(open(OUT / "alval_A_selected.csv", "w"),
                                  index=False, float_format="%.5f")
    print(f"  [输出] alval_A_pool_acquisition.csv / alval_A_learning_curve.csv / "
          f"alval_A_selected.csv")
    return curve, pool_df


# ---------------------------------------------------------------- 任务 B
def task_B(df, feats):
    print("=" * 70)
    print("[任务 B] 外部代理点验证（extval2 同泛函 BEEF-vdW 子集）")
    ev = pd.read_csv(DP / "extval2_homogeneous.csv")
    # 同泛函筛选口径：dftFunctional 统一小写后恰为 'beef-vdw'
    # ——剔除 4 个带 VSHE 电位偏移标注的 BEEF-vdW 行（PasumarthiFacetDependence2023，
    # 参考态不同）；比 notes/14 的精确匹配（114）多 1 行：SongStrongly2026 Pt(111)
    # 原始标记为小写 'beef-vdW'。
    beef = ev[ev["dftFunctional"].str.lower() == "beef-vdw"].reset_index(drop=True)
    n_excl_vshe = int(ev["dftFunctional"].str.contains("VSHE").sum() -
                      (ev["dftFunctional"].str.contains("VSHE") &
                       (ev["dftFunctional"].str.lower() != "beef-vdw")).sum())
    print(f"  extval2 共 {len(ev)} 行 → 同泛函 BEEF-vdW 子集 {len(beef)} 行"
          f"（剔除 VSHE 电位偏移标注 4 行，含小写标记 1 行）")

    Xf, yf = df[feats], df["energy_eV"].values
    base_full = fit_base(Xf, yf)
    Xe = featurize_ext(beef, feats)
    assert list(Xe.columns) == feats and not Xe.isna().any().any()

    pred0 = base_full.predict(Xe)
    q05 = fit_quantile(Xf, yf, 0.05).predict(Xe)
    q95 = fit_quantile(Xf, yf, 0.95).predict(Xe)
    width = q95 - q05
    dG, score = acquisition(pred0, width)
    yt = beef["E_true"].values
    mae_before = mae(yt, pred0)
    bias_before = float((pred0 - yt).mean())
    print(f"  [补点前] 外部 MAE = {mae_before:.4f} eV, bias = {bias_before:+.4f} eV")

    beef = beef.assign(pred=pred0, q05=q05, q95=q95, width=width,
                       pred_dG=dG, acquisition_score=score)
    beef[["pubId", "comp", "facet", "E_true", "pred", "q05", "q95", "width",
          "pred_dG", "acquisition_score"]].to_csv(
        open(OUT / "alval_B_extval_acquisition.csv", "w"),
        index=False, float_format="%.5f")

    al_order = beef.sort_values("acquisition_score", ascending=False)
    # 同组成多晶面/多文献行特征几乎相同（facet one-hot 仅 101/111 两档），
    # 采集分高度并列 → 另做"按组成去重"变体（每组成保留采集分最高的 1 行再取 top-k）
    al_dedup = al_order.drop_duplicates("comp", keep="first")
    met_rows = [{"stage": "before", "k": 0, "n_eval": len(beef),
                 "al_mae": mae_before, "al_bias": bias_before,
                 "aldedup_mae": "", "aldedup_bias": "",
                 "rand_mae_mean": "", "rand_mae_std": ""}]
    sel_rows = []
    for k in [10, 20]:
        sel = al_order.head(k)
        for _, r in sel.iterrows():
            sel_rows.append({"k": k, "variant": "raw_topk",
                             "pubId": r["pubId"], "comp": r["comp"],
                             "facet": r["facet"], "E_true": r["E_true"],
                             "pred": r["pred"], "residual": r["pred"] - r["E_true"],
                             "width": r["width"], "pred_dG": r["pred_dG"],
                             "acquisition_score": r["acquisition_score"]})
        sel_d = al_dedup.head(k)
        for _, r in sel_d.iterrows():
            sel_rows.append({"k": k, "variant": "comp_dedup_topk",
                             "pubId": r["pubId"], "comp": r["comp"],
                             "facet": r["facet"], "E_true": r["E_true"],
                             "pred": r["pred"], "residual": r["pred"] - r["E_true"],
                             "width": r["width"], "pred_dG": r["pred_dG"],
                             "acquisition_score": r["acquisition_score"]})

        def add_eval(sel_df_):
            Xaug = pd.concat([Xf, Xe.loc[sel_df_.index]])
            yaug = np.concatenate([yf, sel_df_["E_true"].values])
            m_aug = fit_base(Xaug, yaug)
            rest = beef.drop(index=sel_df_.index)
            pred_rest = m_aug.predict(Xe.loc[rest.index])
            return (mae(rest["E_true"].values, pred_rest),
                    float((pred_rest - rest["E_true"].values).mean()))

        m_al, b_al = add_eval(sel)
        m_ald, b_ald = add_eval(sel_d)

        rand_maes = []
        for sd in RAND_SEEDS:
            rsel = beef.sample(n=k, random_state=sd)
            Xr = pd.concat([Xf, Xe.loc[rsel.index]])
            yr = np.concatenate([yf, rsel["E_true"].values])
            rrest = beef.drop(index=rsel.index)
            rand_maes.append(mae(rrest["E_true"].values,
                                 fit_base(Xr, yr).predict(Xe.loc[rrest.index])))
        rand_maes = np.array(rand_maes)
        met_rows.append({"stage": f"after_top{k}", "k": k,
                         "n_eval": len(beef) - k, "al_mae": m_al,
                         "al_bias": b_al, "aldedup_mae": m_ald,
                         "aldedup_bias": b_ald,
                         "rand_mae_mean": rand_maes.mean(),
                         "rand_mae_std": rand_maes.std(ddof=1)})
        print(f"  [+{k:2d} 点] 剩余 {len(beef)-k:3d} | AL MAE={m_al:.4f} "
              f"(bias {b_al:+.4f}) | AL(组成去重) MAE={m_ald:.4f} "
              f"(bias {b_ald:+.4f}) | 随机 MAE={rand_maes.mean():.4f}±"
              f"{rand_maes.std(ddof=1):.4f}")

    pd.DataFrame(met_rows).to_csv(open(OUT / "alval_B_metrics.csv", "w"),
                                  index=False, float_format="%.5f")
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(open(OUT / "alval_B_selected.csv", "w"),
                  index=False, float_format="%.5f")
    print(f"  [输出] alval_B_extval_acquisition.csv / alval_B_metrics.csv / "
          f"alval_B_selected.csv")

    print("  [Top-20 被选点化学身份]")
    for i, r in enumerate(al_order.head(20).itertuples(), 1):
        print(f"    {i:2d}. {r.pubId:24s} {r.comp:8s} {r.facet:18s} "
              f"E_true={r.E_true:+.3f} pred={r.pred:+.3f} width={r.width:.3f} "
              f"score={r.acquisition_score:.4f}")
    return beef, pd.DataFrame(met_rows), sel_df


# ---------------------------------------------------------------- 图
def plot_curve(curve, outpath):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.fill_between(curve["k"], curve["rand_mae_mean"] - curve["rand_mae_std"],
                    curve["rand_mae_mean"] + curve["rand_mae_std"],
                    color=C_ALT, alpha=0.35, linewidths=0,
                    label=f"random ±1 std (n={N_RAND} seeds)")
    ax.plot(curve["k"], curve["rand_mae_mean"], "s--", color=C_ALT,
            markeredgecolor=C_DARK, ms=6, lw=1.4, label="random acquisition (mean)")
    ax.plot(curve["k"], curve["al_mae"], "o-", color=C_MAIN,
            markeredgecolor=C_DARK, ms=7, lw=2.0,
            label="AL top-k (width × volcano kernel)")
    ax.plot(curve["k"], curve["alw_mae"], "^:", color=C_DARK, ms=6, lw=1.4,
            label="AL top-k (width only, ablation)")
    for _, r in curve.iloc[1:].iterrows():
        ax.annotate(f"{r['al_mae']:.3f}", (r["k"], r["al_mae"]),
                    textcoords="offset points", xytext=(6, -14),
                    fontsize=9, color=C_MAIN)
    ax.set_xlabel("k = number of revealed (pseudo-DFT) points added")
    ax.set_ylabel("MAE on remaining pool (eV)")
    ax.set_title("Retrospective AL validation: 200-composition hold-out pool\n"
                 "(initial train 1,636; uncertainty = q95$-$q05, "
                 "score = width$\\times$exp($-|\\Delta G_{H^*}|/0.1$))", fontsize=10)
    ax.set_xticks(curve["k"])
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {outpath}")


# ---------------------------------------------------------------- 主流程
def main():
    OUT.mkdir(exist_ok=True)
    FIGDIR.mkdir(exist_ok=True)

    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    print(f"[读入] {len(df)} 行 × {len(feats)} 特征，"
          f"{df['comp'].nunique()} 个 comp 组")

    curve, pool = task_A(df, feats)
    beef, bmet, bsel = task_B(df, feats)
    plot_curve(curve, FIGDIR / "alval_learning_curve.png")

    summary = {
        "taskA": {
            "pool_seed": POOL_SEED, "n_pool": N_POOL, "n_train0": len(df) - N_POOL,
            "curve": curve.round(5).to_dict("records"),
            "verdict_k10": curve.loc[curve["k"] == 10, "significant"].iloc[0],
            "verdict_k20": curve.loc[curve["k"] == 20, "significant"].iloc[0],
        },
        "taskB": {
            "n_beef_subset": len(beef),
            "filter": "dftFunctional.str.lower() == 'beef-vdw'（剔除 4 行 VSHE 电位偏移标注）",
            "metrics": bmet.to_dict("records"),
        },
    }
    json.dump(summary, open(OUT / "alval_summary.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print("[汇总]", json.dumps(summary["taskA"]["verdict_k10"],
                               ensure_ascii=False),
          json.dumps(summary["taskA"]["verdict_k20"], ensure_ascii=False))


if __name__ == "__main__":
    main()
