# -*- coding: utf-8 -*-
"""
04_interpret_screen.py — SHAP 解释 + Sabatier 候选筛选

- 最优树模型（读 data_processed/best_params.json；RF 或 XGBoost_tuned 中随机CV MAE 更低者）
- SHAP TreeExplainer（全数据拟合）：
    figures/shap_summary.png（beeswarm，暖色系 colormap，无蓝紫渐变）
    figures/shap_dependence_top.png（top-3 特征 dependence）
- 筛选（主筛 CO*，O* 在 notes 作对照讨论）：
    目标窗口 E ∈ [0.7, 1.3] eV（CatApp 标度，中心 1.0 eV；化学论证见 notes/04_screening.md）
    预测值 = out-of-fold（KFold 5, 种子 42）；按 |pred - 中心| 排序；
    与最优候选距离差 < 模型 MAE 者标注"统计不可区分"
  → data_processed/candidate_rankings.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
import shap

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "facet", "term", "adsorbate", "energy_eV", "dataset_src", "n_raw"]
WINDOW_LO, WINDOW_HI = 0.7, 1.3   # eV，CO* 目标窗口（CatApp 标度）
WINDOW_CENTER = (WINDOW_LO + WINDOW_HI) / 2

# 暖色发散 colormap（深棕→米白→橙红），替代 shap 默认红蓝
WARM_DIV = LinearSegmentedColormap.from_list(
    "warm_div", ["#5C3A10", "#F5E6C8", "#B3401E"])
C_BAR = "#C98A5A"

plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def build_best_model(info):
    """最优树模型：XGBoost_tuned 或 RandomForest（按随机 CV MAE）。"""
    cmp_df = pd.read_csv(DP / "model_compare.csv")
    tree = cmp_df[cmp_df["model"].isin(["RandomForest", "XGBoost_tuned"])]
    best_tree = tree.loc[tree["MAE_mean"].idxmin(), "model"]
    if best_tree == "XGBoost_tuned":
        return best_tree, XGBRegressor(random_state=SEED, n_jobs=-1,
                                       **info["xgb_best_params"])
    return best_tree, RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)


def main():
    df = pd.read_csv(DP / "dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values

    info = json.load(open(DP / "best_params.json"))
    best_name, model = build_best_model(info)
    mae = float(info["random_cv_MAE"])
    print(f"[最优树模型] {best_name}（随机CV MAE={mae:.3f} eV）")

    # ---- SHAP ----
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1]
    top5 = [(feats[i], float(mean_abs[i])) for i in top_idx[:5]]
    print("[SHAP top-5]", top5)

    # beeswarm（事后重着色为暖色 colormap：shap 内部硬编码红蓝，monkeypatch 不可靠）
    # shap summary_plot 的蜂群抖动用到 numpy 随机数，显式播种保证图字节级可复现
    np.random.seed(SEED)
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    fig = plt.gcf()
    for ax in fig.axes:
        for coll in ax.collections:  # 散点与 colorbar 网格
            try:
                coll.set_cmap(WARM_DIV)
            except Exception:
                pass
    fig.set_size_inches(7.6, 5.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "shap_summary.png")
    plt.close("all")
    print("[写出] figures/shap_summary.png")

    # dependence：top-3
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, i in zip(axes, top_idx[:3]):
        v = X.iloc[:, i].values
        sc = ax.scatter(v, shap_values[:, i], c=v, cmap="YlOrBr",
                        s=12, alpha=0.7, edgecolors="none")
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls="--")
        ax.set_xlabel(feats[i]); ax.set_ylabel("SHAP value (eV)")
        plt.colorbar(sc, ax=ax, label="feature value", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGDIR / "shap_dependence_top.png")
    plt.close(fig)
    print("[写出] figures/shap_dependence_top.png")

    # ---- Sabatier 筛选（主筛 CO*，out-of-fold 预测）----
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    df["pred_E_eV"] = oof

    co = df[df["adsorbate"] == "CO"].copy()
    co["dist_to_target"] = (co["pred_E_eV"] - WINDOW_CENTER).abs()
    co["in_window"] = co["pred_E_eV"].between(WINDOW_LO, WINDOW_HI)
    co = co.sort_values("dist_to_target").reset_index(drop=True)
    co.insert(0, "rank", co.index + 1)
    d_min = co["dist_to_target"].min()
    co["statistically_indistinguishable"] = (co["dist_to_target"] - d_min) < mae
    out_cols = ["rank", "comp", "facet", "term", "adsorbate",
                "energy_eV", "pred_E_eV", "dist_to_target",
                "in_window", "statistically_indistinguishable"]
    co = co.rename(columns={"energy_eV": "true_E_eV"})
    out_cols[5] = "true_E_eV"
    co[out_cols].round(4).to_csv(DP / "candidate_rankings.csv", index=False)
    print(f"[写出] data_processed/candidate_rankings.csv  ({len(co)} 个 CO 候选)")
    print(co[out_cols].head(10).round(3).to_string(index=False))

    # ---- notes/04_screening.md ----
    import re as _re
    # 纯金属锚点（动态从数据读取，CO 有 (111)/(211)，O 全部为合金）
    n_el = df["comp"].apply(lambda c: len(_re.findall(r"[A-Z][a-z]?", c)))
    pm = df[n_el == 1]
    co111 = dict(zip(pm[(pm["facet"] == 111) & (pm["adsorbate"] == "CO")]["comp"],
                     pm[(pm["facet"] == 111) & (pm["adsorbate"] == "CO")]["energy_eV"]))
    co211 = dict(zip(pm[(pm["facet"] == 211) & (pm["adsorbate"] == "CO")]["comp"],
                     pm[(pm["facet"] == 211) & (pm["adsorbate"] == "CO")]["energy_eV"]))
    co111_sorted = sorted(co111.items(), key=lambda kv: -kv[1])
    co111_str = " > ".join(f"{k}({v:.2f})" for k, v in co111_sorted)
    # O* 锚点：按"含某元素的所有行"的平均能量排序（O 数据全为合金，无纯金属）
    o_rows = df[df["adsorbate"] == "O"]
    el_o = {}
    for _, r in o_rows.iterrows():
        for el in set(_re.findall(r"[A-Z][a-z]?", r["comp"])):
            el_o.setdefault(el, []).append(r["energy_eV"])
    el_o_mean = sorted(((el, float(np.mean(v))) for el, v in el_o.items()), key=lambda kv: -kv[1])
    o_top_str = "/".join(f"{el}" for el, _ in el_o_mean[:4])
    o_top_rng = f"{el_o_mean[3][1]:.1f}–{el_o_mean[0][1]:.1f}"
    o_bot4 = [el for el, _ in el_o_mean[-4:]][::-1]  # 由弱到次弱
    o_bot_str = "/".join(o_bot4)
    o_bot_hi = el_o_mean[-4][1]

    # facet × adsorbate 共线性结构（动态）
    o_min, o_max = float(o_rows["energy_eV"].min()), float(o_rows["energy_eV"].max())
    ct_t = pd.crosstab(df["term"], df["adsorbate"])
    n_co_term_none = int(ct_t.loc["none", "CO"]) if ("none" in ct_t.index and "CO" in ct_t.columns) else 0
    n_o_term_none = int(ct_t.loc["none", "O"]) if ("none" in ct_t.index and "O" in ct_t.columns) else 0
    n_o_term_lbl = int((o_rows["term"] != "none").sum())
    ct = pd.crosstab(df["facet"], df["adsorbate"])
    n_co111 = int(ct.loc[111, "CO"]) if (111 in ct.index and "CO" in ct.columns) else 0
    n_co211 = int(ct.loc[211, "CO"]) if (211 in ct.index and "CO" in ct.columns) else 0
    n_o111 = int(ct.loc[111, "O"]) if (111 in ct.index and "O" in ct.columns) else 0
    n_o211 = int(ct.loc[211, "O"]) if (211 in ct.index and "O" in ct.columns) else 0

    # 假阳性率：预测落入窗口的候选中，真值在窗口外的比例（经验值 + 理论量级）
    in_win = co[co["in_window"]]
    fp_emp = float((~in_win["true_E_eV"].between(WINDOW_LO, WINDOW_HI)).mean()) if len(in_win) else float("nan")
    half_w = (WINDOW_HI - WINDOW_LO) / 2
    sigma = mae / 0.7979  # 正态误差假设下 σ ≈ MAE / E|N(0,1)|
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    fp_center = 2 * (1 - phi(half_w / sigma))  # 预测恰在窗口中心时真值出窗概率
    grid = np.linspace(WINDOW_LO, WINDOW_HI, 601)
    fp_unif = float(np.mean([2 * (1 - phi(min(p - WINDOW_LO, WINDOW_HI - p) / sigma)) for p in grid]))

    top10 = co.head(10)
    # 三口径指标（从原始结果文件读全精度，避免中间舍入不一致）
    cmp_raw = pd.read_csv(DP / "model_compare.csv").set_index("model")
    gcv_raw = pd.read_csv(DP / "group_cv_results.csv").set_index("model")
    loeo_raw = pd.read_csv(DP / "loeo_results.csv")
    loeo_bm = loeo_raw[loeo_raw["model"] == best_name]
    mae_rand = float(cmp_raw.loc[best_name, "MAE_mean"])
    mae_grp = float(gcv_raw.loc[best_name, "MAE_mean"])
    mae_loeo = float(np.average(loeo_bm["MAE"], weights=loeo_bm["n_test"]))
    lines = [
        "# 04 SHAP 解释与 Sabatier 候选筛选",
        "",
        "## 能量标度的重要说明（与 SPEC 第 2 节的出入）",
        "",
        "SPEC 写『负值=放热吸附』，但对数据的经验检验表明**数值越大 = 结合越强**（≈ −E_ads + 常数）：",
        f"纯金属 CO* 锚点（(111) 面，动态取自 dataset_v2）：{co111_str}；",
        "O*（全部为合金数据，按含该元素行的平均能量排序）最强端为 "
        f"{o_top_str} 类（约 {o_top_rng} eV），最弱端为 {o_bot_str} 类（≤ {o_bot_hi:.1f} eV），"
        "与已知的化学吸附强弱趋势完全一致（亲氧的早期过渡金属最强，惰性后过渡/贵金属最弱）。",
        "因此本数据能量实际是『结合强度』方向（源于 CatApp 的气相参考态校正方案）；",
        "所有『强/弱结合』的解读均以此方向为准，这不影响模型与筛选的相对排序结论。",
        "",
        "## SHAP 结果（最优树模型：%s）" % best_name,
        "",
        "mean(|SHAP|) top-5：",
        "",
        "| 特征 | mean(|SHAP|) / eV |",
        "|---|---|",
    ]
    lines += [f"| {n} | {v:.3f} |" for n, v in top5]
    lines += [
        "",
        "- `figures/shap_summary.png`：beeswarm（暖色系 colormap，低→高 = 深棕→橙红；"
        "summary_plot 前已设 np.random.seed(42)，字节级可复现）",
        "- `figures/shap_dependence_top.png`：top-3 特征 dependence",
        "",
        "### 共线性局限：facet_111 / term_none 不能解读为纯结构效应",
        "",
        f"本数据集中结构 one-hot 与 adsorbate 高度共线：",
        f"- facet：CO 共 {n_co111 + n_co211} 行，其中 (111) {n_co111} 行、(211) {n_co211} 行"
        f"（后者全部为修复标签污染后恢复的纯金属）；O 共 {n_o111 + n_o211} 行，**全部位于 (211)**，"
        f"(111) 上为 {n_o111} 行。",
        f"- term：CO 共 {n_co_term_none} 行**全部 term=none**；O 的 term=none 行为 {n_o_term_none}，"
        f"其余 {n_o_term_lbl} 行全部带 AA/AB/BB 终止层标签。",
        "因此 `term_none` 与 `ads_CO` **完全共线**、`facet_111` 近乎共线，SHAP 中 term_none / facet_111 的"
        "高重要性实质是吸附种（CO vs O）指示变量，**不能解读为纯终止层/晶面效应**。",
        "若要得到晶面/终止层结论，需补充 CO(211)（尤其合金）、O(111) 与带终止层标签的 CO 数据。",
        "",
        "## 筛选窗口的化学论证（主筛 CO*）",
        "",
        "**Sabatier 原则**：好的催化剂对关键中间体结合『适中』——过强则毒化表面，过弱则无法活化。",
        "",
        "以 CO 氧化为例（CatApp 标度，数值大=结合强）：",
        "",
        f"- 纯金属锚点（(111) 面）：Au({co111['Au']:.2f})/Ag({co111['Ag']:.2f}) 结合过弱，CO 难以吸附活化；",
        f"  Rh({co111['Rh']:.2f})/Ni({co111['Ni']:.2f}) 结合过强，CO 毒化表面占据位点；",
        f"  Pt({co111['Pt']:.2f})/Pd({co111['Pd']:.2f}) 是工业基准但仍有 CO 自毒化问题。",
        "- 文献共识（Nørskov 等的 CO 氧化火山曲线）：最优点的 CO 结合**略弱于 Pt(111)**，"
        "位于 Cu(111) 与 Pt(111) 之间。",
        f"- 据此取窗口 **[{WINDOW_LO}, {WINDOW_HI}] eV（中心 {WINDOW_CENTER} eV）**："
        f"下界高于 Cu(111)={co111['Cu']:.2f} 保证足够活化能力；上界 1.3 略高于 Pt(111)={co111['Pt']:.2f}，"
        "把 Pt 近邻候选（火山最优在 Pt 附近略弱侧）纳入考察，同时明显低于 "
        f"Rh(111)={co111['Rh']:.2f}/Ni(111)={co111['Ni']:.2f} 的毒化端。",
        "",
        "**假设与局限**：(1) 仅用单一描述符 E(CO) 定位火山，未考虑 O2 活化/反应能垒；"
        "(2) CatApp 参考标度与常用吸附能标度有平移，窗口只在本数据集内部有效；"
        "(3) 预测为 out-of-fold 值，候选按 |pred − 中心| 排序。",
        "",
        "### O* 对照讨论",
        "",
        f"O* 能量全为正值（{o_min:.2f}–{o_max:.2f} eV）且标度与 CO* 不同（气相参考态不同），两种吸附种的能量不能直接比较。"
        "且 O* 数据全部位于 (211) 面（见上『共线性局限』），没有任何 (111) 参照。",
        "O 的 Sabatier 窗口应对应具体反应（如 CO 氧化中 O2 解离或 O* 加氢），且本数据 O* 的强结合端"
        f"（{o_top_str} 类，>3.5 eV）对应易氧化/烧结体系；由于缺乏明确的 O* 火山锚点反应，"
        "本工作不给出 O* 定量窗口，仅在主筛中以 CO* 为准，O* 留待后续（可平移管线至 H*/O* 火山）。",
        "",
        "## 候选排名（CO*，按距窗口中心距离升序；OOF 预测）",
        "",
        f"模型 MAE = {mae:.3f} eV；`statistically_indistinguishable=True` 表示该候选与第 1 名的"
        "距离差小于 MAE，统计上无法区分。",
        "",
        f"三口径指标（汇总见 data_processed/cv_protocol_summary.csv，{best_name}）："
        f"随机CV MAE={mae_rand:.3f}，分组CV(comp) MAE={mae_grp:.3f}，"
        f"LOEO(元素) MAE={mae_loeo:.3f}——"
        "对新元素体系的真实外推误差明显大于随机 CV，候选排名解读应以 LOEO 口径为保守上限。",
        "",
        f"**假阳性率披露**：窗口半宽 {half_w:.1f} eV 与模型 MAE {mae:.2f} eV 同量级，"
        f"预测落入窗口的候选中，真值实际在窗口外的比例不可忽略——本次 OOF 经验值为 "
        f"{fp_emp:.0%}（{int((~in_win['true_E_eV'].between(WINDOW_LO, WINDOW_HI)).sum())}/{len(in_win)}）；"
        f"理论估算（误差正态，σ≈MAE/0.8≈{sigma:.2f}）：预测恰在窗口中心时约 {fp_center:.0%}，"
        f"预测均匀分布于窗口时约 {fp_unif:.0%}。"
        "约 40% 量级的假阳性属统计预期，因此窗口内候选应视为『优先实验/计算清单』而非确定命中。",
        "",
        "| rank | comp | facet | term | true E | pred E | 距中心 | MAE 内不可区分 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in top10.iterrows():
        lines.append(f"| {r['rank']} | {r['comp']} | {int(r['facet'])} | {r['term']} | "
                     f"{r['true_E_eV']:.3f} | {r['pred_E_eV']:.3f} | {r['dist_to_target']:.3f} | "
                     f"{'✓' if r['statistically_indistinguishable'] else ''} |")
    lines += [
        "",
        f"完整清单见 `data_processed/candidate_rankings.csv`（{len(co)} 个 CO 候选，"
        f"其中 in_window=True 共 {int(co['in_window'].sum())} 个）。",
    ]
    (ROOT / "notes" / "04_screening.md").write_text("\n".join(lines), encoding="utf-8")
    print("[写出] notes/04_screening.md")


if __name__ == "__main__":
    main()
