# -*- coding: utf-8 -*-
"""
29_r9_robust_cluster_ci.py — Round 9 计算补证（子任务 1B）

内容：
 1. 稳健候选簇：c ∈ {0.20,...,0.28}（9 值）× {v2 OOF, v3 OOF} 共 18 个名单，
    各取 |ΔG| Top-50（口径同 22 号脚本 §3 现行口径 (i)：组成得分 = min|ΔG|；
    实证上 v2/v3 数据集每组成恰 1 行，去重平凡成立，仍按 groupby-min 实现）。
    统计每个候选在 18 名单中出现次数，分级 A≥16 / B≥9 / C<9。
    另报 v2∩v3 在默认 c=0.24 下 Top-50 交集大小。
 2. 外部验证 bootstrap CI：EqV2-HER（491，data_processed/extval_results.csv）
    全体 MAE/R²/Spearman；CatHub 同质子集（data_processed/extval2_homogeneous.csv）
    全体 355 / 域内单元素+111 n=75 / 同泛函 BEEF-vdW n=114 的 MAE/R²。
    10,000 次样本级 bootstrap（种子 42），95% percentile CI。
 3. 小数字核实：
    a. v2/v3 OOF |residual| ≤ 0.2 eV 占比；
    b. 0.180→0.120 降幅（Ridge/XGB 随机 5-fold，data_processed/hstar_model_compare.csv）；
    c. EqV2 Spearman ρ=0.17–0.29 的分层还原；
    d. Cathub 7,005 vs 7,048 的 43 条去向（仅用本地 cathub_mamun.csv 与镜像 json）。
 4. min-聚合极值偏置：data_processed/hstar_group_spread.csv（comp×structure，
    n_raw 构型，E_min/E_mean/E_median），量化标签取 min 的系统性偏低幅度。

产物：
  outputs/r9_robust_cluster.csv / r9_extval_ci.csv / r9_small_numbers.md /
  r9_min_aggregation_bias.csv；notes/28b_cluster_ci.md（由本脚本数据生成要点，
  报告正文手写）。
红线：种子 42；不重训模型；不改任何既有文件；写出用非原子 open/to_csv。
"""
import gzip
import json
import re
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
ASSETS = ROOT / "web" / "assets"
SEED = 42
C_GRID = [round(0.20 + 0.01 * i, 2) for i in range(9)]   # 0.20..0.28
TOPN = 50
N_BOOT = 10_000
TOL = 1e-3
DILUTE_EQ = "0.5H2(g) + * -> H*"


def canon_comp(formula):
    """与 scripts/13_cathub_fetch.py 完全一致：元素字母序 + 计量数约简。"""
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", formula or "")
    if not toks or "".join(a + b for a, b in toks) != formula:
        return None
    d = {}
    for el, n in toks:
        d[el] = d.get(el, 0) + int(n or 1)
    g = reduce(gcd, d.values())
    return "".join(f"{el}{d[el]//g if d[el]//g > 1 else ''}" for el in sorted(d))


# =====================================================================
# 1. 稳健候选簇
# =====================================================================
print("=" * 60, "\n[任务1] 稳健候选簇")
v2 = pd.read_csv(OUT / "calib_oof_predictions.csv")[
    ["comp", "structure", "facet", "y_true", "pred_base"]]
v2 = v2.rename(columns={"pred_base": "pred"})
with open(ASSETS / "predictions_all.json") as f:
    v3 = pd.DataFrame(json.load(f))[["comp", "structure", "facet", "pred_E_ads"]]
v3["facet"] = v3["facet"].astype(str)
v2["facet"] = v2["facet"].astype(str)
v3 = v3.rename(columns={"pred_E_ads": "pred"})
print(f"[1] v2 OOF {len(v2)} 行（唯一 comp {v2['comp'].nunique()}）；"
      f"v3 OOF {len(v3)} 行（唯一 comp {v3['comp'].nunique()}）")

models = {"v2": v2, "v3": v3}
# 每个 (model, c) 的 Top-50（按 comp 去重取 min|ΔG|，与 22 号脚本口径 (i) 一致）
lists = {}          # (model, c) -> DataFrame(Top-50: comp, structure, facet, abs_dG)
per_setting = {}    # (model, c) -> 全量 comp 级表（供 |ΔG| 统计）
for mname, mdf in models.items():
    for c in C_GRID:
        d = mdf.copy()
        d["abs_dG"] = (d["pred"] + c).abs()
        # 组成级去重：每 comp 取 min|ΔG| 的行
        idx = d.groupby("comp")["abs_dG"].idxmin()
        comp_lvl = d.loc[idx].sort_values("abs_dG").reset_index(drop=True)
        per_setting[(mname, c)] = comp_lvl
        lists[(mname, c)] = comp_lvl.head(TOPN)

# 出现次数统计（成员键 = comp|structure|facet；comp 唯一时等价于 comp）
from collections import Counter
cnt = Counter()
cnt_by_model = {"v2": Counter(), "v3": Counter()}
member_info = {}
absdg_vals = {}
for (mname, c), top in lists.items():
    for _, r in top.iterrows():
        key = f"{r['comp']}|{r['structure']}|{r['facet']}"
        cnt[key] += 1
        cnt_by_model[mname][key] += 1
        member_info[key] = (r["comp"], r["structure"], r["facet"])
for key in cnt:
    comp = member_info[key][0]
    vals = []
    for (mname, c), cl in per_setting.items():
        row = cl[cl["comp"] == comp]
        if len(row):
            vals.append(float(row["abs_dG"].iloc[0]))
    absdg_vals[key] = vals


def grade(n):
    return "A" if n >= 16 else ("B" if n >= 9 else "C")


rows = []
for key, n in cnt.items():
    comp, structure, facet = member_info[key]
    vals = absdg_vals[key]
    rows.append({
        "comp": comp, "structure": structure, "facet": facet,
        "n_lists_of_18": n,
        "n_v2_lists_of_9": cnt_by_model["v2"][key],
        "n_v3_lists_of_9": cnt_by_model["v3"][key],
        "grade": grade(n),
        "abs_dG_median": float(np.median(vals)),
        "abs_dG_min": float(np.min(vals)),
        "abs_dG_max": float(np.max(vals)),
        "in_v2_c0.24_top50": key in set(
            lists[("v2", 0.24)].assign(
                k=lambda d: d["comp"] + "|" + d["structure"] + "|" + d["facet"])["k"]),
        "in_v3_c0.24_top50": key in set(
            lists[("v3", 0.24)].assign(
                k=lambda d: d["comp"] + "|" + d["structure"] + "|" + d["facet"])["k"]),
    })
clus = pd.DataFrame(rows).sort_values(
    ["n_lists_of_18", "abs_dG_median"], ascending=[False, True]).reset_index(drop=True)
clus.to_csv(OUT / "r9_robust_cluster.csv", index=False)

nA = (clus.grade == "A").sum()
nB = (clus.grade == "B").sum()
nC = (clus.grade == "C").sum()
print(f"[1] 进入过任一名单的候选 {len(clus)} 个：A 级 {nA} / B 级 {nB} / C 级 {nC}")
print(f"[1] 出现次数分布: {clus['n_lists_of_18'].value_counts().sort_index(ascending=False).to_dict()}")
print(f"[1] 模型内稳定性（9 个 c 名单全勤）：v2 内 9/9 的候选 "
      f"{int((clus.n_v2_lists_of_9 == 9).sum())} 个；v3 内 9/9 的候选 "
      f"{int((clus.n_v3_lists_of_9 == 9).sum())} 个")
print(clus.head(25).to_string(index=False))

# v2∩v3 在 c=0.24 的 Top-50 交集
k24 = {m: set(lists[(m, 0.24)]["comp"]) for m in models}
inter24 = k24["v2"] & k24["v3"]
print(f"[1] c=0.24 Top-50 交集 |v2∩v3| = {len(inter24)}")
print(f"[1] v3 c=0.24 Top-3: {lists[('v3',0.24)]['comp'].head(3).tolist()} "
      f"（应复现 IrOs3/Au3Rh/BiZr）；CuPt3 ∈ v3 Top-50: {'CuPt3' in k24['v3']}（应 False）")
print(f"[1] v2 c=0.24 Top-1: {lists[('v2',0.24)]['comp'].iloc[0]}（应复现 CuPt3）")

# =====================================================================
# 2. 外部验证 bootstrap CI
# =====================================================================
print("=" * 60, "\n[任务2] bootstrap CI")
rng = np.random.default_rng(SEED)


def boot_ci(y_true, y_pred, fn, n_boot=N_BOOT):
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        vals[b] = fn(y_true[ii], y_pred[ii])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae_fn(yt, yp):
    return mean_absolute_error(yt, yp)


def r2_fn(yt, yp):
    return r2_score(yt, yp)


def spear_fn(yt, yp):
    return spearmanr(yt, yp).statistic


ci_rows = []

# ---- EqV2-HER 491（全体）----
eq = pd.read_csv(DP / "extval_results.csv")
assert len(eq) == 491
yt, yp = eq["E_true"].values, eq["E_pred"].values
for mname, fn in [("MAE", mae_fn), ("R2", r2_fn), ("Spearman_rho", spear_fn)]:
    pt = fn(yt, yp)
    lo, hi = boot_ci(yt, yp, fn)
    ci_rows.append({"dataset": "EqV2-HER", "stratum": "ALL", "n": 491,
                    "metric": mname, "point": pt, "ci_lo": lo, "ci_hi": hi})
    print(f"[2] EqV2 ALL {mname}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")
# 附：EqV2 去偏差 MAE 的 CI（文档引用 0.313，一并给区间）
resid = yp - yt
deb_fn = lambda a, b: np.mean(np.abs((b - a) - np.mean(b - a)))
pt = deb_fn(yt, yp)
lo, hi = boot_ci(yt, yp, deb_fn)
ci_rows.append({"dataset": "EqV2-HER", "stratum": "ALL", "n": 491,
                "metric": "MAE_debiased", "point": pt, "ci_lo": lo, "ci_hi": hi})
print(f"[2] EqV2 ALL MAE_debiased: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

# ---- CatHub 同质子集 ----
ch = pd.read_csv(DP / "extval2_homogeneous.csv")
assert len(ch) == 355
strata = {
    "ALL": ch,
    "in_domain(单元素+111)": ch[ch["in_domain"]],
    "func=BEEF-vdW": ch[ch["dftFunctional"] == "BEEF-vdW"],
}
for sname, sdf in strata.items():
    yt, yp = sdf["E_true"].values, sdf["E_pred"].values
    for mname, fn in [("MAE", mae_fn), ("R2", r2_fn)]:
        pt = fn(yt, yp)
        lo, hi = boot_ci(yt, yp, fn)
        ci_rows.append({"dataset": "CatHub同质子集", "stratum": sname,
                        "n": len(sdf), "metric": mname, "point": pt,
                        "ci_lo": lo, "ci_hi": hi})
        print(f"[2] CatHub {sname} (n={len(sdf)}) {mname}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

cidf = pd.DataFrame(ci_rows)
cidf.to_csv(OUT / "r9_extval_ci.csv", index=False)

# =====================================================================
# 3. 小数字核实
# =====================================================================
print("=" * 60, "\n[任务3] 小数字核实")
small = []

# ---- 3a. OOF |residual| ≤ 0.2 eV 占比 ----
r_v2 = (v2["y_true"] - v2["pred"]).abs()
v3j = v3.merge(v2[["comp", "y_true"]], on="comp", how="left")
r_v3 = (v3j["y_true"] - v3j["pred"]).abs()
for name, r in [("v2(36feat)", r_v2), ("v3(84feat)", r_v3)]:
    f02 = float((r <= 0.2).mean())
    f01 = float((r <= 0.1).mean())
    small.append({"item": "3a_OOF残差占比", "scope": name,
                  "value": f"|res|≤0.2eV: {(r<=0.2).sum()}/1836={f02:.4f}；"
                           f"|res|≤0.1eV: {(r<=0.1).sum()}/1836={f01:.4f}",
                  "detail": f"MAE={r.mean():.4f}；median|r|={r.median():.4f}"})
    print(f"[3a] {name}: |r|≤0.2 {f02:.2%}，|r|≤0.1 {f01:.2%}，MAE {r.mean():.4f}")

# ---- 3b. 0.180→0.120 降幅 ----
mc = pd.read_csv(DP / "hstar_model_compare.csv")
mae_ridge = mc.loc[mc.model == "Ridge", "MAE_mean"].iloc[0]
mae_lin = mc.loc[mc.model == "Linear", "MAE_mean"].iloc[0]
mae_xgb = mc.loc[mc.model == "XGBoost_tuned", "MAE_mean"].iloc[0]
red_ridge = (mae_ridge - mae_xgb) / mae_ridge
red_lin = (mae_lin - mae_xgb) / mae_lin
red_round = (0.180 - 0.120) / 0.180
small.append({"item": "3b_0.180降幅", "scope": "随机5-fold 模型对比（deprecated 成果汇总报告 §4.2 场景）",
              "value": f"Ridge {mae_ridge:.6f} → XGB_tuned {mae_xgb:.6f}：降幅 {red_ridge:.4f} "
                       f"({red_ridge:.2%})；vs Linear {mae_lin:.6f}：{red_lin:.2%}；"
                       f"按舍入值 0.180→0.120：{red_round:.2%}",
              "detail": "文档写 34% 有误；精确 33.25%（Ridge 基线），舍入口径 33.3%。"
                        "注意此为随机5-fold MAE（非 GroupKFold OOF 0.1198）。"})
print(f"[3b] Ridge→XGB: {red_ridge:.4f}；Linear→XGB: {red_lin:.4f}；舍入口径 {red_round:.4f}")

# ---- 3c. EqV2 Spearman 分层还原 ----
for sname, sdf in [("ALL(491)", eq),
                   ("eqv2_361_relaxed(219)", eq[eq.src == "eqv2_361_relaxed"]),
                   ("eqv2_500_sp(272)", eq[eq.src == "eqv2_500_sp"]),
                   ("facet=111(109)", eq[eq.facet.astype(str) == "111"]),
                   ("不含Sb(364)", eq[~eq.has_Sb])]:
    rho = spearmanr(sdf["E_true"], sdf["E_pred"]).statistic
    small.append({"item": "3c_EqV2_Spearman分层", "scope": sname,
                  "value": f"ρ={rho:.4f}", "detail": "逐层从 extval_results.csv 重算"})
    print(f"[3c] {sname}: ρ={rho:.4f}")

# ---- 3d. 7,005 vs 7,048 的 43 条 ----
hub = pd.read_csv(DP / "cathub_mamun.csv")
dil = hub[hub["Equation"] == DILUTE_EQ].copy()
dil["comp"] = dil["chemicalComposition"].map(canon_comp)
nodil = hub[hub["Equation"] != DILUTE_EQ].copy()
nodil["comp"] = nodil["chemicalComposition"].map(canon_comp)
with gzip.open(ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz", "rt") as f:
    mirror = json.load(f)
mrows = []
for k, v in mirror.items():
    if "-> H*" not in k or "0.5H2" not in k:
        continue
    prefix = k.split("_", 1)[0]
    mrows.append({"comp": canon_comp(prefix)})
mir = pd.DataFrame(mrows)
hub_comps = set(dil["comp"].dropna().unique())
mir_comps = set(mir["comp"].dropna().unique())
only_mir = sorted(mir_comps - hub_comps)
n_total = int(mir["comp"].isin(hub_comps).sum())   # 消耗式核对的分母
n_43 = int(mir["comp"].isin(only_mir).sum())
nodil_comps = set(nodil["comp"].dropna().unique())
covered_by_nodil = sorted(set(only_mir) & nodil_comps)
det = (f"镜像 H* 构型 {len(mir)}（=7,048）；官方稀释方程行 {len(dil)}（=6,894）；"
       f"镜像组成 {len(mir_comps)}，仅镜像 {len(only_mir)} 个：{only_mir}；"
       f"这 {len(only_mir)} 个组成占镜像构型 {n_43} 条 → 核对分母 n_total={n_total}（=7,005）；"
       f"差值 {len(mir)-n_total}={len(mir)}-{n_total}。"
       f"仅镜像组成中出现在官方非稀释方程（H2+2*->2H*）行的：{covered_by_nodil}")
small.append({"item": "3d_7005vs7048", "scope": "Cathub 任务A 能量逐条核对",
              "value": f"43 = 15 个仅镜像组成的全部构型（{n_43} 条）；{len(mir)}-{n_43}={n_total}",
              "detail": det})
print(f"[3d] {det}")

# ---- 写出 3 ----
sdf = pd.DataFrame(small)
with open(OUT / "r9_small_numbers.md", "w") as f:
    f.write("# Round 9 小数字核实（全部本地重算，scripts/29）\n\n")
    for _, r in sdf.iterrows():
        f.write(f"## {r['item']} — {r['scope']}\n\n- 结果：{r['value']}\n- 细节：{r['detail']}\n\n")

# =====================================================================
# 4. min-聚合极值偏置
# =====================================================================
print("=" * 60, "\n[任务4] min-聚合偏置")
sp = pd.read_csv(DP / "hstar_group_spread.csv")
print(f"[4] {len(sp)} 个 comp×structure 组；n_raw 分布：")
print(sp["n_raw"].value_counts().sort_index().to_string())
multi = sp[sp["n_raw"] >= 2]
diff_mean = multi["E_mean"] - multi["E_min"]     # ≥0，min 相对均值的下移幅度
diff_med = multi["E_median"] - multi["E_min"]
qs = np.percentile(diff_mean, [5, 25, 50, 75, 95])
bias_rows = [
    {"stat": "n_groups_total", "value": len(sp)},
    {"stat": "n_groups_nraw_eq1", "value": int((sp.n_raw == 1).sum())},
    {"stat": "n_groups_nraw_ge2", "value": len(multi)},
    {"stat": "share_label_from_min_of_ge2_configs",
     "value": len(multi) / len(sp)},
    {"stat": "n_raw_median", "value": float(sp.n_raw.median())},
    {"stat": "n_raw_max", "value": int(sp.n_raw.max())},
    {"stat": "E_mean_minus_E_min_p05(n>=2)", "value": qs[0]},
    {"stat": "E_mean_minus_E_min_p25(n>=2)", "value": qs[1]},
    {"stat": "E_mean_minus_E_min_p50(n>=2)", "value": qs[2]},
    {"stat": "E_mean_minus_E_min_p75(n>=2)", "value": qs[3]},
    {"stat": "E_mean_minus_E_min_p95(n>=2)", "value": qs[4]},
    {"stat": "E_mean_minus_E_min_mean(n>=2)", "value": float(diff_mean.mean())},
    {"stat": "E_median_minus_E_min_mean(n>=2)", "value": float(diff_med.mean())},
    {"stat": "E_median_minus_E_min_p50(n>=2)", "value": float(diff_med.median())},
]
# 对 |ΔG| 排名的实际影响：min vs mean 口径下 |ΔG|≤0.1 eV 的表面数
sp["dg_min"] = (sp["E_min"] + 0.24).abs()
sp["dg_mean"] = (sp["E_mean"] + 0.24).abs()
bias_rows.append({"stat": "n_|dG|<=0.1_min口径", "value": int((sp.dg_min <= 0.1).sum())})
bias_rows.append({"stat": "n_|dG|<=0.1_mean口径", "value": int((sp.dg_mean <= 0.1).sum())})
bias_rows.append({"stat": "spearman_dg_min_vs_mean",
                  "value": float(spearmanr(sp.dg_min, sp.dg_mean).statistic)})
# |dG|≤0.1 集合（min 口径）的标签下移幅度（该簇受 min 偏置最相关）
near = sp[sp.dg_min <= 0.1]
if len(near):
    nm = near[near.n_raw >= 2]
    bias_rows.append({"stat": "near_peak(n_min口径|dG|<=0.1)_n",
                      "value": len(near)})
    bias_rows.append({"stat": "near_peak_中_多构型占比",
                      "value": len(nm) / len(near)})
    bias_rows.append({"stat": "near_peak_E_mean_minus_E_min_median",
                      "value": float((nm.E_mean - nm.E_min).median()) if len(nm) else np.nan})
bdf = pd.DataFrame(bias_rows)
bdf.to_csv(OUT / "r9_min_aggregation_bias.csv", index=False)
print(bdf.to_string(index=False))
print("\n[完成] r9_robust_cluster.csv / r9_extval_ci.csv / r9_small_numbers.md / "
      "r9_min_aggregation_bias.csv")
