# -*- coding: utf-8 -*-
"""
19_error_analysis.py — 主线 H* 模型分层误差分析（补 W10 缺口）

- OOF 预测：XGBoost_tuned（hstar_best_params.json）+ GroupKFold(5, groups=规范化 comp)，
  与 07_hstar_models.py 的 GroupKFold 口径一致，得到逐样本残差（residual = pred - true）
- 分层 MAE / bias（均值残差）/ RMSE（每层报 n）：
  facet(111/101)、structure(A1/L1₀/L1₂)、纯度（纯金属/合金）、
  周期类别（含3d/含4d/含5d/含镧系，可叠加）、贵金属（含/不含）
- 最差个案：|残差| top-20，结合 hstar_loeo_results.csv 高误差元素与
  candidate_rankings_hstar.csv 的 red_flags 逻辑（强亲氧/放射性/LOEO 高误差）逐条诊断
- 图：figures/errana_stratified_box.png（五维分层残差箱线图，300dpi 低饱和暖色系）
- 产出：outputs/errana_by_facet.csv、errana_by_structure.csv、errana_by_category.csv、
  outputs/errana_worst20.csv、notes/20_error_analysis.md
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
FIGDIR = ROOT / "figures"
NOTES = ROOT / "notes"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]

# 周期类别口径：标准 d 区划分；镧系单列（数据集内仅 La 一个镧系元素）。
# La 属镧系，不计入"含 5d"；非 d 区元素（Al/Ga/In/Sn/Pb/Bi/Tl）不进任何周期层。
D3 = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
D4 = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
D5 = {"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
LANTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
         "Ho", "Er", "Tm", "Yb", "Lu"}
NOBLE = {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au"}

# red_flags 逻辑（与 08_hstar_interpret.py 一致）
RADIOACTIVE = {"Tc"}
OXOPHILIC = {"La", "Y", "Sc"}
LOEO_RISKY = {"Mn", "Bi", "Fe"}  # hstar_loeo_results.csv 中 MAE 最高的 3 个元素

STRUCT_LABEL = {"A1": "A1", "L10": "L1$_0$", "L12": "L1$_2$"}

# 低饱和暖色系（与 07 一致基调，无蓝紫渐变）
C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
C_DEEP = "#7A4A2B"
C_SOFT = "#E8D3B5"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def elements_of(comp):
    return set(re.findall(r"[A-Z][a-z]?", comp))


def flags_of(comp):
    els = elements_of(comp)
    fl = []
    if els & RADIOACTIVE:
        fl.append("含放射性" + "/".join(sorted(els & RADIOACTIVE)))
    if els & OXOPHILIC:
        fl.append("强亲氧" + "/".join(sorted(els & OXOPHILIC)) + "(表面稳定性存疑)")
    if els & LOEO_RISKY:
        fl.append("含LOEO高误差元素" + "/".join(sorted(els & LOEO_RISKY)))
    return ";".join(fl)


def strat_table(mask_map, resid, y):
    """mask_map: {层名: bool array} → 每层 n/MAE/bias/RMSE"""
    rows = []
    for name, m in mask_map.items():
        m = np.asarray(m)
        if m.sum() == 0:
            rows.append({"group": name, "n": 0, "MAE": np.nan,
                         "bias": np.nan, "RMSE": np.nan})
            continue
        r = resid[m]
        rows.append({"group": name, "n": int(m.sum()),
                     "MAE": float(np.mean(np.abs(r))),
                     "bias": float(np.mean(r)),
                     "RMSE": float(np.sqrt(np.mean(r ** 2)))})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    groups = df["comp"].values

    with open(DP / "hstar_best_params.json") as f:
        xgb_params = json.load(f)["xgb_best_params"]
    print(f"[读入] {X.shape[0]} 行 × {X.shape[1]} 特征；tuned 参数 {xgb_params}")

    # ---- OOF 预测（GroupKFold(5), groups=规范化 comp，同 07 口径）----
    model = XGBRegressor(random_state=SEED, n_jobs=-1, **xgb_params)
    gkf = GroupKFold(n_splits=5)
    oof = cross_val_predict(model, X, y, cv=gkf, groups=groups, n_jobs=-1)
    resid = oof - y  # 残差定义：pred - true；bias>0 表示系统性高估（吸附能偏正/吸附偏弱）
    df["pred_eV"] = oof
    df["residual"] = resid
    df["abs_res"] = np.abs(resid)
    print(f"[OOF] 全体 MAE={np.mean(np.abs(resid)):.4f} eV  "
          f"bias={np.mean(resid):+.4f} eV  RMSE={np.sqrt(np.mean(resid**2)):.4f} eV")

    els_list = df["comp"].apply(elements_of)
    facet_s = df["facet"].astype(str)  # facet 列在 csv 中为整型，统一按字符串比较

    # ---- 分层 1：晶面 ----
    t_facet = strat_table({f"facet_{f}": facet_s == f
                           for f in ["111", "101"]}, resid, y)
    t_facet.to_csv(OUT / "errana_by_facet.csv", index=False)

    # ---- 分层 2：结构 ----
    t_struct = strat_table({s: df["structure"] == s
                            for s in ["A1", "L10", "L12"]}, resid, y)
    t_struct.to_csv(OUT / "errana_by_structure.csv", index=False)

    # ---- 分层 3：纯度 + 周期类别 + 贵金属（合入一张 category 表）----
    cat_masks = {
        "纯金属 (n_elements=1)": df["n_elements"] == 1,
        "合金 (n_elements>1)": df["n_elements"] > 1,
        "含3d元素": els_list.apply(lambda e: bool(e & D3)),
        "含4d元素": els_list.apply(lambda e: bool(e & D4)),
        "含5d元素": els_list.apply(lambda e: bool(e & D5)),
        "含镧系元素": els_list.apply(lambda e: bool(e & LANTH)),
        "含贵金属": els_list.apply(lambda e: bool(e & NOBLE)),
        "不含贵金属": els_list.apply(lambda e: not (e & NOBLE)),
    }
    t_cat = strat_table(cat_masks, resid, y)
    t_cat.insert(0, "dimension", (["纯度"] * 2 + ["周期类别(可叠加)"] * 4
                                  + ["贵金属"] * 2))
    t_cat = t_cat[["dimension", "group", "n", "MAE", "bias", "RMSE"]]
    t_cat.to_csv(OUT / "errana_by_category.csv", index=False)

    print("\n[按晶面]\n", t_facet.round(4).to_string(index=False))
    print("\n[按结构]\n", t_struct.round(4).to_string(index=False))
    print("\n[按类别]\n", t_cat.round(4).to_string(index=False))

    # ---- 最差个案 top-20 ----
    loeo = pd.read_csv(DP / "hstar_loeo_results.csv")
    loeo_mae = dict(zip(loeo["held_out_element"], loeo["MAE"]))
    worst = df.nlargest(20, "abs_res").copy()
    worst["elements"] = worst["comp"].apply(lambda c: "/".join(sorted(elements_of(c))))
    worst["red_flags"] = worst["comp"].apply(flags_of)
    worst["loeo_mae_max"] = worst["comp"].apply(
        lambda c: max((loeo_mae.get(e, 0.0) for e in elements_of(c)), default=0.0))
    worst = worst[["comp", "structure", "facet", "n_elements", "elements",
                   "energy_eV", "pred_eV", "residual", "abs_res",
                   "loeo_mae_max", "red_flags"]]
    worst.to_csv(OUT / "errana_worst20.csv", index=False)
    print("\n[|残差| top-20]")
    print(worst.round(3).to_string(index=False))

    # ---- 图：五维分层残差箱线图 ----
    dims = [
        ("facet", [("111", facet_s == "111"),
                   ("101", facet_s == "101")]),
        ("structure", [(STRUCT_LABEL[s], df["structure"] == s)
                       for s in ["A1", "L10", "L12"]]),
        ("purity", [("pure metal", df["n_elements"] == 1),
                    ("alloy", df["n_elements"] > 1)]),
        ("d-block (overlap)", [("has 3d", cat_masks["含3d元素"]),
                               ("has 4d", cat_masks["含4d元素"]),
                               ("has 5d", cat_masks["含5d元素"]),
                               ("has Ln", cat_masks["含镧系元素"])]),
        ("noble metal", [("yes", cat_masks["含贵金属"]),
                         ("no", cat_masks["不含贵金属"])]),
    ]
    box_colors = [C_MAIN, C_ALT, C_DEEP, C_SOFT]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.6), sharey=True)
    for ax, (title, layer) in zip(axes, dims):
        data = [resid[np.asarray(m)] for _, m in layer]
        labels = [f"{name}\n(n={int(np.asarray(m).sum())})" for name, m in layer]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False, medianprops=dict(color="#3d2b1f", lw=1.4),
                        whiskerprops=dict(color=C_DEEP), capprops=dict(color=C_DEEP))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(box_colors[i % len(box_colors)])
            patch.set_alpha(0.75)
        # 抖动散点（抽样避免过密）
        rng = np.random.default_rng(SEED)
        for i, d in enumerate(data):
            s = rng.choice(len(d), size=min(len(d), 250), replace=False)
            ax.scatter(rng.normal(i + 1, 0.05, len(s)), d[s],
                       s=3, alpha=0.25, color=C_DEEP, edgecolors="none")
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("residual (pred - true, eV)")
    fig.suptitle("Stratified OOF residuals of H* model (XGBoost_tuned, GroupKFold x5)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "errana_stratified_box.png", bbox_inches="tight")
    plt.close(fig)
    print("\n[写出] figures/errana_stratified_box.png")

    # ---- 诊断摘要（供 notes 使用）----
    summary = {
        "overall": {"MAE": float(np.mean(np.abs(resid))),
                    "bias": float(np.mean(resid)),
                    "RMSE": float(np.sqrt(np.mean(resid ** 2)))},
        "by_facet": t_facet.to_dict("records"),
        "by_structure": t_struct.to_dict("records"),
        "by_category": t_cat.to_dict("records"),
    }
    with open(OUT / "errana_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 最差个案共性统计
    n_flag = int((worst["red_flags"] != "").sum())
    n_loeo_risky = int(worst["comp"].apply(
        lambda c: bool(elements_of(c) & LOEO_RISKY)).sum())
    print(f"[top-20 共性] 带红旗 {n_flag}/20；含 LOEO 高误差元素(Mn/Bi/Fe) {n_loeo_risky}/20")
    print("[写出] outputs/errana_by_facet.csv / by_structure.csv / by_category.csv / "
          "errana_worst20.csv / errana_summary.json")


if __name__ == "__main__":
    main()
