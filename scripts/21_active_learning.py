# -*- coding: utf-8 -*-
"""
21_active_learning.py — H* 吸附能模型的主动学习闭环（不确定性量化 + 候选枚举 + 采集排序）

流程（全部数字来自真实运行，GroupKFold(5) 按 comp 分组，种子 42）：
  1. 基线 XGBoost（tuned 参数）GroupKFold OOF，复核 MAE；
  2. 不确定性量化两套方案（同一 GroupKFold 划分，OOF 口径）：
       A. 分位数 XGBoost（reg:quantileerror, alpha=0.05/0.95）→ OOF 90% 区间宽度；
       B. Bootstrap 集成（每折内 20 个有放回重采样模型）→ OOF 预测标准差；
     校准检验：width vs |OOF 残差| 的 Spearman 相关；90% 区间经验覆盖率（分桶）；
     按「覆盖率接近 90% 优先、Spearman 次之」选校准更好的方案用于候选排序。
  3. 组成空间枚举：39 元素（elements.json）× 数据集中出现的化学计量原型
     （纯金属 A1/111、AB L10/101、A3B L12/111），排除训练已有组成，
     用与训练完全相同的管线生成 36 特征，标记 in_domain；
  4. 采集分 = width × exp(-|pred_dG|/0.1)（不确定度 × 火山峰邻近度），
     输出 Top-50 总榜 + 利用型 Top-20（近峰低不确定度）+ 探索型 Top-20（高不确定度）；
  5. 图：figures/active_learning_calibration.png、figures/active_learning_map.png；
  6. 明细 CSV：outputs/active_learning_proposals.csv（Top-50）、
     outputs/active_learning_exploit_top20.csv、outputs/active_learning_explore_top20.csv、
     outputs/active_learning_candidates_all.csv（全部候选预测）、
     outputs/active_learning_calibration_bins.csv（校准分桶表）。

注意：/mnt/agents/output 挂载对 rename/原子写可能 Permission denied，统一用普通 open(...,'w')。
"""
import json
import re
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
FIGDIR = ROOT / "figures"
ASSETS = ROOT / "web" / "assets"
SEED = 42
N_BOOT = 20          # bootstrap 集成规模（≥20）
DG_SHIFT = 0.24      # ΔG_H* = E_ads + 0.24 eV
VOLCANO_SIGMA = 0.1  # 火山峰邻近度核宽度（eV）

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
XGB_PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
                  subsample=0.8, random_state=SEED)
PROPS = ["en", "radius", "group", "period", "ie1", "d_el"]
STATS = ["wmean", "max", "min", "range", "wstd"]

# 低饱和暖色系（禁蓝紫渐变）
C_MAIN = "#B35C24"   #  burnt orange
C_ALT = "#DFB27E"    #  tan
C_GRAY = "#9b9187"   #  warm gray
C_DARK = "#5c4a38"   #  warm dark brown
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


# ---------------------------------------------------------------- 特征化（与 web/feature_spec.md 完全一致）
def parse_formula(formula):
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts or "".join(e + (n or "") for e, n in parts) != formula:
        raise ValueError(f"无法解析化学式: {formula}")
    return {e: int(n) if n else 1 for e, n in parts}


def canonical_formula(comp_dict):
    g = 0
    for n in comp_dict.values():
        g = gcd(g, n)
    items = sorted(comp_dict.items())
    return "".join(e + ("" if n // g == 1 else str(n // g)) for e, n in items)


def stoich_pattern(comp_dict):
    ns = sorted(comp_dict.values())
    return ":".join(map(str, ns))


def compute_features(comp_dict, structure, facet, elem_props):
    """36 特征，顺序与训练集列一致。"""
    els = sorted(comp_dict.keys())
    counts = np.array([comp_dict[e] for e in els], dtype=float)
    w = counts / counts.sum()
    feat = {"n_elements": len(els)}
    for p in PROPS:
        v = np.array([elem_props[e][p] for e in els], dtype=float)
        mu = np.average(v, weights=w)
        feat[f"{p}_wmean"] = mu
        feat[f"{p}_max"] = v.max()
        feat[f"{p}_min"] = v.min()
        feat[f"{p}_range"] = v.max() - v.min()
        feat[f"{p}_wstd"] = np.sqrt(np.sum(w * (v - mu) ** 2) / np.sum(w))
    for s in ["A1", "L10", "L12"]:
        feat[f"structure_{s}"] = 1 if structure == s else 0
    for f in ["101", "111"]:
        feat[f"facet_{f}"] = 1 if str(facet) == f else 0
    return feat


# ---------------------------------------------------------------- 模型封装
def fit_base(X, y):
    m = XGBRegressor(**XGB_PARAMS, n_jobs=-1)
    m.fit(X, y)
    return m


def fit_quantile(X, y, alpha):
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                     **XGB_PARAMS, n_jobs=-1)
    m.fit(X, y)
    return m


def oof_predictions(X, y, groups):
    """GroupKFold(5) OOF：基线 + 方案A(分位数) + 方案B(bootstrap)。
    返回 DataFrame(y, pred, q05, q95, width_A, boot_std, width_B)。"""
    gkf = GroupKFold(n_splits=5)
    n = len(y)
    res = pd.DataFrame({"y": y,
                        "pred": np.full(n, np.nan),
                        "q05": np.full(n, np.nan),
                        "q95": np.full(n, np.nan),
                        "boot_std": np.full(n, np.nan)})
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y[tr]
        res.loc[te, "pred"] = fit_base(Xtr, ytr).predict(Xte)
        res.loc[te, "q05"] = fit_quantile(Xtr, ytr, 0.05).predict(Xte)
        res.loc[te, "q95"] = fit_quantile(Xtr, ytr, 0.95).predict(Xte)
        rng = np.random.RandomState(SEED + fold)
        boot_preds = np.empty((N_BOOT, len(te)))
        for b in range(N_BOOT):
            idx = rng.randint(0, len(tr), size=len(tr))
            bm = XGBRegressor(random_state=SEED + 1000 * fold + b,
                              **{k: v for k, v in XGB_PARAMS.items()
                                 if k != "random_state"}, n_jobs=-1)
            bm.fit(Xtr.iloc[idx], ytr[idx])
            boot_preds[b] = bm.predict(Xte)
        res.loc[te, "boot_std"] = boot_preds.std(axis=0, ddof=1)
        print(f"  [fold {fold}] 训练 {len(tr)} / 测试 {len(te)} 完成")
    res["width_A"] = res["q95"] - res["q05"]
    # 方案 B 的 90% 区间（正态近似）：pred ± 1.645 * std
    res["width_B"] = 2 * 1.6448536269514722 * res["boot_std"]
    return res


def calibration_report(res, width_col, label):
    """Spearman(width, |残差|) + 90% 区间覆盖率（总体与按宽度五分桶）。"""
    abs_res = (res["y"] - res["pred"]).abs()
    rho, pval = spearmanr(res[width_col], abs_res)
    if width_col == "width_A":
        lo, hi = res["q05"], res["q95"]
    else:
        half = res[width_col] / 2
        lo, hi = res["pred"] - half, res["pred"] + half
    cover = ((res["y"] >= lo) & (res["y"] <= hi)).mean()
    print(f"[{label}] Spearman(width, |resid|) = {rho:.4f} (p={pval:.2e})  "
          f"90%区间经验覆盖率 = {cover:.4f}  平均宽度 = {res[width_col].mean():.4f} eV")
    bins = pd.qcut(res[width_col], 5, duplicates="drop")
    tab = pd.DataFrame({"bin": bins, "width": res[width_col],
                        "abs_resid": abs_res,
                        "covered": ((res["y"] >= lo) & (res["y"] <= hi)).astype(float)})
    g = tab.groupby("bin", observed=True)
    out = pd.DataFrame({"method": label, "n": g.size(),
                        "width_mean": g["width"].mean(),
                        "abs_resid_mean": g["abs_resid"].mean(),
                        "abs_resid_std": g["abs_resid"].std(),
                        "coverage": g["covered"].mean()}).reset_index()
    out["bin_lo"] = out["bin"].map(lambda b: b.left)
    out["bin_hi"] = out["bin"].map(lambda b: b.right)
    return rho, cover, res[width_col].mean(), out


# ---------------------------------------------------------------- 枚举候选
def enumerate_candidates(elem_props, training_elements, existing_comps):
    """按数据集中出现的化学计量原型枚举：
       纯金属→A1/111；AB(1:1)→L10/101；A3B(3:1)→L12/111（A≠B，有序，含 AB3）。
       排除训练集中已有组成。"""
    all_elements = sorted(elem_props.keys())
    train_el_set = set(training_elements)
    rows = []
    # (原型, structure, facet, 生成器)
    # 纯金属
    for a in all_elements:
        comp = canonical_formula({a: 1})
        if comp in existing_comps:
            continue
        rows.append((comp, {a: 1}, "A1", "111"))
    # AB (1:1)
    for i, a in enumerate(all_elements):
        for b in all_elements[i + 1:]:
            comp = canonical_formula({a: 1, b: 1})
            if comp in existing_comps:
                continue
            rows.append((comp, {a: 1, b: 1}, "L10", "101"))
    # A3B / AB3 (3:1, 有序)
    for a in all_elements:
        for b in all_elements:
            if a == b:
                continue
            comp = canonical_formula({a: 3, b: 1})
            if comp in existing_comps:
                continue
            rows.append((comp, {a: 3, b: 1}, "L12", "111"))
    feats = [compute_features(cd, s, f, elem_props)
             for _, cd, s, f in rows]
    df = pd.DataFrame(feats)
    df.insert(0, "facet", [f for _, _, _, f in rows])
    df.insert(0, "structure", [s for _, _, s, _ in rows])
    df.insert(0, "comp", [c for c, _, _, _ in rows])
    df["in_domain"] = [
        all(e in train_el_set for e in cd)  # 元素均在训练集 37 元素内
        for _, cd, _, _ in rows
    ]  # 化学计量原型只枚举了数据集中出现的三种，恒为已见原型
    return df


# ---------------------------------------------------------------- 图
def plot_calibration(bins_all, best_label, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, marker, color in [("A:quantile", "o", C_MAIN),
                                 ("B:bootstrap", "s", C_ALT)]:
        sub = bins_all[bins_all["method"] == label]
        axes[0].errorbar(sub["width_mean"], sub["abs_resid_mean"],
                         yerr=sub["abs_resid_std"], fmt=marker, color=color,
                         ecolor=color, elinewidth=1, capsize=3, ms=6,
                         label=f"{label}", alpha=0.9)
        axes[1].plot(np.arange(len(sub)) + (0 if label == "A:quantile" else 0.22),
                     sub["coverage"], marker, color=color, ms=7, label=label)
    axes[0].set_xlabel("OOF interval width (eV, bin mean)")
    axes[0].set_ylabel("|OOF residual| (eV, bin mean ± std)")
    axes[0].set_title("(a) Width vs error (width quintiles)")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.90, color=C_DARK, ls="--", lw=1.2, label="nominal 90%")
    axes[1].set_xlabel("width quintile bin (low → high)")
    axes[1].set_ylabel("empirical coverage of 90% interval")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_title("(b) Coverage by width bin")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Calibration of OOF uncertainty (GroupKFold); selected: {best_label}",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {outpath}")


def plot_map(cand, top50, width_label, outpath):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(cand["abs_dG"], cand["width"], s=6, c=C_GRAY, alpha=0.25,
               linewidths=0, label=f"all candidates (n={len(cand)})")
    ax.scatter(top50["abs_dG"], top50["width"], s=26, c=C_MAIN,
               edgecolors=C_DARK, linewidths=0.4, alpha=0.95,
               label="Top-50 acquisition")
    ax.set_xlabel("|predicted ΔG$_{H^*}$| (eV)")
    ax.set_ylabel(f"uncertainty ({width_label}, eV)")
    ax.set_title("Acquisition map: uncertainty vs volcano-peak proximity\n"
                 "(ΔG$_{H^*}$ = 0 is the HER volcano peak)")
    ax.legend(frameon=False)
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
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values
    print(f"[读入] {X.shape[0]} 行 × {X.shape[1]} 特征，{len(set(groups))} 个 comp 组")

    # ---------- 1+2. OOF 不确定性量化与校准 ----------
    print("[OOF] GroupKFold(5) 基线 + 分位数 + bootstrap ...")
    res = oof_predictions(X, y, groups)
    mae = (res["y"] - res["pred"]).abs().mean()
    print(f"[基线] OOF MAE = {mae:.4f} eV（文献口径 0.1198 eV）")

    rhoA, covA, wA, binsA = calibration_report(res, "width_A", "A:quantile")
    rhoB, covB, wB, binsB = calibration_report(res, "width_B", "B:bootstrap")
    bins_all = pd.concat([binsA, binsB], ignore_index=True)
    bins_all.drop(columns=["bin"]).to_csv(
        open(OUT / "active_learning_calibration_bins.csv", "w"), index=False)

    # 选择：覆盖率偏离 0.90 小者优先，Spearman 高者次之
    scoreA = (abs(covA - 0.90), -rhoA)
    scoreB = (abs(covB - 0.90), -rhoB)
    best = "A" if scoreA <= scoreB else "B"
    best_label = "A:quantile" if best == "A" else "B:bootstrap"
    print(f"[选择] 校准更优方案 = {best_label}  "
          f"(A: |cov-0.9|={abs(covA-0.90):.3f}, rho={rhoA:.3f} | "
          f"B: |cov-0.9|={abs(covB-0.90):.3f}, rho={rhoB:.3f})")

    # ---------- 3. 枚举候选并特征化 ----------
    elem_json = json.load(open(ASSETS / "elements.json"))
    elem_props = elem_json["elements"]
    training_elements = elem_json["meta"]["training_elements"]
    existing = set(df["comp"].unique())
    cand = enumerate_candidates(elem_props, training_elements, existing)
    print(f"[枚举] 候选 {len(cand)} 个（纯金属/AB/A3B，排除训练已有 {len(existing)} 组成）；"
          f"in_domain = {int(cand['in_domain'].sum())}")

    # ---------- 全量训练模型用于候选预测 ----------
    print("[全量训练] 基线 + 分位数(0.05/0.95) + bootstrap 集成 ...")
    Xc = cand[feats]
    base_full = fit_base(X, y)
    pred = base_full.predict(Xc)
    q05_full = fit_quantile(X, y, 0.05).predict(Xc)
    q95_full = fit_quantile(X, y, 0.95).predict(Xc)
    rng = np.random.RandomState(SEED)
    boot = np.empty((N_BOOT, len(cand)))
    for b in range(N_BOOT):
        idx = rng.randint(0, len(X), size=len(X))
        bm = XGBRegressor(random_state=SEED + 5000 + b,
                          **{k: v for k, v in XGB_PARAMS.items()
                             if k != "random_state"}, n_jobs=-1)
        bm.fit(X.iloc[idx], y[idx])
        boot[b] = bm.predict(Xc)
    boot_std = boot.std(axis=0, ddof=1)

    cand["pred_E_ads"] = pred
    cand["pred_dG"] = pred + DG_SHIFT
    cand["abs_dG"] = np.abs(cand["pred_dG"])
    # 区间列：两法都给，width 用选中方案
    if best == "A":
        cand["q05"], cand["q95"] = q05_full, q95_full
    else:
        half = 1.6448536269514722 * boot_std
        cand["q05"], cand["q95"] = pred - half, pred + half
    cand["width"] = cand["q95"] - cand["q05"]
    cand["boot_std"] = boot_std
    cand["width_A"] = q95_full - q05_full
    cand["width_B"] = 2 * 1.6448536269514722 * boot_std

    # ---------- 4. 采集排序 ----------
    # 采集分 = 不确定度 × 火山峰邻近度：width * exp(-|dG|/0.1)
    # 理由：宽度代表新信息（探索），exp 核把采集预算聚焦在 HER 火山峰附近
    # （|dG|≲0.3 eV 才有催化价值），连续可乘、无需调权重，尺度即 eV。
    cand["acquisition_score"] = cand["width"] * np.exp(-cand["abs_dG"] / VOLCANO_SIGMA)
    cand_sorted = cand.sort_values("acquisition_score", ascending=False)

    cols = ["comp", "structure", "facet", "pred_E_ads", "pred_dG",
            "q05", "q95", "width", "in_domain", "acquisition_score"]
    top50 = cand_sorted.head(50)
    top50[cols].to_csv(open(OUT / "active_learning_proposals.csv", "w"),
                       index=False, float_format="%.5f")
    # 利用型：近峰（|dG|≤0.1 eV）中不确定度最低
    exploit = (cand[cand["abs_dG"] <= 0.1]
               .sort_values("width").head(20))
    exploit[cols].to_csv(open(OUT / "active_learning_exploit_top20.csv", "w"),
                         index=False, float_format="%.5f")
    # 探索型：不确定度最高
    explore = cand.sort_values("width", ascending=False).head(20)
    explore[cols].to_csv(open(OUT / "active_learning_explore_top20.csv", "w"),
                         index=False, float_format="%.5f")
    cand_sorted[cols + ["boot_std", "width_A", "width_B", "abs_dG"]].to_csv(
        open(OUT / "active_learning_candidates_all.csv", "w"),
        index=False, float_format="%.5f")
    print(f"[输出] proposals Top-50 / exploit Top-20 / explore Top-20 / "
          f"candidates_all ({len(cand_sorted)}) 已写入 outputs/")

    print("\n[Top-10 采集建议]")
    for i, r in enumerate(top50.head(10).itertuples(), 1):
        print(f"  {i:2d}. {r.comp:10s} {r.structure:3s}/{r.facet}  "
              f"dG={r.pred_dG:+.3f} eV  width={r.width:.3f} eV  "
              f"in_domain={r.in_domain}  score={r.acquisition_score:.4f}")
    print(f"[统计] Top-50 中 in_domain=True: {int(top50['in_domain'].sum())}/50；"
          f"|dG|≤0.2 eV: {int((top50['abs_dG']<=0.2).sum())}/50")
    print(f"[统计] 利用型Top20 width 中位数 {exploit['width'].median():.3f} eV；"
          f"探索型Top20 width 中位数 {explore['width'].median():.3f} eV；"
          f"探索型 in_domain=True: {int(explore['in_domain'].sum())}/20")

    # ---------- 5. 图 ----------
    plot_calibration(bins_all, best_label, FIGDIR / "active_learning_calibration.png")
    width_label = "q95-q05" if best == "A" else "2×1.645×bootstrap std"
    plot_map(cand, top50, width_label, FIGDIR / "active_learning_map.png")

    # ---------- 6. 汇总指标（供报告引用） ----------
    summary = {
        "oof_mae": round(float(mae), 4),
        "A_quantile": {"spearman": round(float(rhoA), 4),
                       "coverage90": round(float(covA), 4),
                       "mean_width_eV": round(float(wA), 4)},
        "B_bootstrap": {"spearman": round(float(rhoB), 4),
                        "coverage90": round(float(covB), 4),
                        "mean_width_eV": round(float(wB), 4),
                        "n_boot": N_BOOT},
        "selected": best_label,
        "n_candidates": int(len(cand)),
        "n_in_domain": int(cand["in_domain"].sum()),
        "top50_in_domain": int(top50["in_domain"].sum()),
        "top50_absdG_le_0.2": int((top50["abs_dG"] <= 0.2).sum()),
        "exploit_top20_median_width": round(float(exploit["width"].median()), 4),
        "explore_top20_median_width": round(float(explore["width"].median()), 4),
    }
    json.dump(summary, open(OUT / "active_learning_summary.json", "w"),
              indent=2, ensure_ascii=False)
    print("[汇总]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
