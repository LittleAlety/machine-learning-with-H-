# -*- coding: utf-8 -*-
"""
23_descriptor_refine.py — 描述符精准化：文献 4（HER/OER 手稿）MagpieData 高 SHAP
组成描述符的 mendeleev 复刻 + 诚实检验（阴性同样交付）

文献依据（notes/25_literature_digest.md §4.3/§4.6）：
  文献 4 经 SHAP 验证有效的组成特征，全部可用 mendeleev 零成本实现：
  熔点 MeltingT、门捷列夫数 MendeleevNumber、未填满 d/总电子数 (NdUnfilled/NUnfilled)、
  价电子数 (NdValence/NValence)、共价半径 CovalentRadius；统计量补 mode（众数）。

公平契约（与 17_feature_expand.py 一致）：
- 新特征只加在特征侧；fold 划分不变（GroupKFold(5, comp)，同一批 fold 对所有配置）；
- 嵌套 CV：外层 GroupKFold(5) / 内层 GroupKFold(4) 随机搜索 30 组拟合超参；
- 改善 >0.005 eV 才采纳；阴性结果同样交付。
- 基线：v2 原 36 特征 + XGBoost_tuned（hstar_best_params.json：lr=0.03 depth=7
  n=400 sub=0.8, seed=42），GroupKFold OOF MAE=0.1198 eV。

新增元素性质（7 种）：
  melting_point     熔点 K（mendeleev；Sn 因同素异形体缺失，按 Magpie/文献值
                    505.08 K 手工补齐并记录）
  mendeleev_number  门捷列夫数（周期表位置的另一种排序编码）
  covalent_radius   共价半径 pm（mendeleev covalent_radius = Pyykkö 双键半径）
  n_valence         价电子总数（mendeleev nvalence()；Magpie NValence 对应）
  nd_valence        最外层 d 亚层电子数（Magpie NdValence 对应；注意与既有 d_el
                    定义同源——d_el=电子组态尾部的 d 电子数，预期高度冗余）
  n_unfilled        未填满电子数：对电子组态中每个部分占据亚层求 (容量-占据) 之和
                    （Magpie NUnfilled 定义；容量 s=2/p=6/d=10/f=14）
  nd_unfilled       未填满 d 电子数（仅 d 亚层；Magpie NdUnfilled 对应）

统计量：沿用现有方案 wmean/wstd/min/max/range × 7 新性质 = 35；
另加 mode 系列 = 13 个性质（既有 6 + 新增 7）的"原子分数最大组元"属性值
（并列时取化学式字母序首元素，确定性）。合计新增 48 个特征。

配置（7 个）：
  baseline_36 / +熔点族(5) / +门捷列夫数族(5) / +未填满电子族(10) /
  +价电子与共价半径族(15) / +mode系列(13) / 全量合并(48)

产出：
  outputs/desrefine_coverage.csv        新性质元素级覆盖率
  outputs/desrefine_new_features.csv    1836 行 × 48 新特征
  outputs/desrefine_ablation.csv        固定参数 GroupKFold 消融（7 配置）
  outputs/desrefine_nested_cv.csv       嵌套 CV 逐外折明细（7 配置）
  outputs/desrefine_config_summary.csv  逐配置汇总 + 采纳判定
  outputs/desrefine_collinearity.csv    新特征 vs 既有 36 特征 |ρ|>0.9 相关对
  outputs/desrefine_shap_family.csv     全量合并模型的 SHAP 族分解
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
SEED = 42
ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
ADOPT_THRESHOLD = 0.005
BASELINE_REF = 0.1198  # 项目锁定基线 OOF MAE (eV)

COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
SUBSHELL_RE = re.compile(r"(\d)([spdf])(\d*)")  # 单占据写作 "5d"（省略 1）
SUBSHELL_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}

BASE_PROPS = ["en", "radius", "group", "period", "ie1", "d_el"]
NEW_PROPS = ["melting_point", "mendeleev_number", "covalent_radius",
             "n_valence", "nd_valence", "n_unfilled", "nd_unfilled"]
STATS = ["wmean", "max", "min", "range", "wstd"]

# 手工兜底：Tc（同 06 脚本）；Sn 熔点（Magpie/文献 505.08 K，mendeleev 缺）
HAND_LOOKUP = {
    "Tc": dict(en=2.10, radius=135.0, group=7, period=5, ie1=7.11938, d_el=5),
    "Sn": dict(melting_point=505.08),
}

# 特征族分组（配置定义）
FAMILIES = {
    "熔点族": ["melting_point"],
    "门捷列夫数族": ["mendeleev_number"],
    "未填满电子族": ["n_unfilled", "nd_unfilled"],
    "价电子与共价半径族": ["n_valence", "nd_valence", "covalent_radius"],
}


# ---------------------------------------------------------------- 元素表
def parse_comp(comp):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(comp)}


def parse_subshells(econf):
    """电子组态串 → {(n, l): occ}（惰性气体芯为简写，天然跳过全满芯层）。"""
    out = {}
    for m in SUBSHELL_RE.finditer(str(econf)):
        n, l = int(m.group(1)), m.group(2)
        occ = int(m.group(3)) if m.group(3) else 1
        out[(n, l)] = out.get((n, l), 0) + occ
    return out


def d_electron_count(econf):
    """基线 d_el 定义（同 06 脚本）：组态尾部全部 d 占据之和。"""
    tail = str(econf).split("]")[-1]
    return sum(int(n) if n else 1
               for _, n in re.findall(r"(\d)d(\d*)", tail))


def build_element_table(elements):
    """→ {sym: {prop: float}}，覆盖 6 既有 + 7 新性质；记录缺失/兜底。"""
    from mendeleev import element as md_element
    table, notes = {}, []
    for sym in elements:
        e = md_element(sym)
        conf = parse_subshells(e.econf)
        n_max = max(n for n, _ in conf) if conf else None
        nd_val = sum(o for (n, l), o in conf.items()
                     if l == "d" and n == max(nn for (nn, ll) in conf if ll == "d")) \
            if any(l == "d" for _, l in conf) else 0
        n_unf = sum(SUBSHELL_CAP[l] - o for (n, l), o in conf.items()
                    if SUBSHELL_CAP[l] > o)
        nd_unf = sum(SUBSHELL_CAP["d"] - o for (n, l), o in conf.items()
                     if l == "d" and SUBSHELL_CAP["d"] > o)
        try:
            n_val = e.nvalence()
        except Exception:
            n_val = None
        vals = {
            "en": e.en_pauling, "radius": e.atomic_radius, "group": e.group_id,
            "period": e.period, "ie1": (e.ionenergies or {}).get(1),
            "d_el": d_electron_count(e.econf),
            "melting_point": e.melting_point,
            "mendeleev_number": e.mendeleev_number,
            "covalent_radius": e.covalent_radius,
            "n_valence": n_val,
            "nd_valence": float(nd_val),
            "n_unfilled": float(n_unf),
            "nd_unfilled": float(nd_unf),
        }
        for k, v in list(vals.items()):
            if v is None:
                fallback = HAND_LOOKUP.get(sym, {}).get(k)
                if fallback is not None:
                    vals[k] = fallback
                    notes.append(f"{sym}.{k} mendeleev 缺失→手工兜底 {fallback}")
                else:
                    notes.append(f"{sym}.{k} 缺失且无兜底→NaN")
                    vals[k] = np.nan
        table[sym] = {k: float(v) for k, v in vals.items()}
    return table, notes


# ---------------------------------------------------------------- 统计展开
def comp_stats(d, etab, props, prefix=""):
    els = list(d)
    w = np.array([d[e] for e in els], dtype=float)
    w /= w.sum()
    feats = {}
    for p in props:
        vals = [etab[e][p] for e in els]
        if any(np.isnan(v) for v in vals):
            feats.update({f"{prefix}{p}_{s}": np.nan for s in STATS})
            feats[f"{prefix}{p}_mode"] = np.nan
            continue
        v = np.array(vals)
        mu = np.average(v, weights=w)
        feats[f"{prefix}{p}_wmean"] = float(mu)
        feats[f"{prefix}{p}_max"] = float(v.max())
        feats[f"{prefix}{p}_min"] = float(v.min())
        feats[f"{prefix}{p}_range"] = float(v.max() - v.min())
        feats[f"{prefix}{p}_wstd"] = float(
            np.sqrt(np.average((v - mu) ** 2, weights=w)))
        # mode：原子分数最大组元的属性值；并列取化学式（字母序）首元素
        i_mode = int(np.argmax([d[e] for e in els]))
        feats[f"{prefix}{p}_mode"] = float(v[i_mode])
    return feats


# ---------------------------------------------------------------- 评估
def fixed_ablation(X, y, groups, params):
    gkf = GroupKFold(n_splits=5)
    maes = []
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
        m.fit(X.iloc[tr], y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(X.iloc[te])))
    return np.array(maes)


def nested_cv(X, y, groups, tag):
    param_dist = {
        "n_estimators": [200, 300, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    rows = []
    gkf_out = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(gkf_out.split(X, y, groups)):
        inner = GroupKFold(n_splits=4)
        rs = RandomizedSearchCV(
            XGBRegressor(random_state=SEED, n_jobs=-1), param_dist,
            n_iter=30, cv=inner.split(X.iloc[tr], y[tr], groups[tr]),
            scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
        rs.fit(X.iloc[tr], y[tr])
        mae = mean_absolute_error(y[te], rs.best_estimator_.predict(X.iloc[te]))
        rows.append({"config": tag, "outer_fold": k, "MAE": mae,
                     "best_params": json.dumps(rs.best_params_)})
        print(f"    [nested/{tag}] 外折 {k}: MAE={mae:.4f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    y = df["energy_eV"].values
    groups = df["comp"].values
    best_params = json.loads((DP / "hstar_best_params.json").read_text()
                             )["xgb_best_params"]
    base_feats = [c for c in df.columns if c not in ID_COLS]
    print(f"[读入] v2 {df.shape[0]} 行 × {len(base_feats)} 基线特征；"
          f"{df['comp'].nunique()} 个 comp 组", flush=True)
    print(f"[基线参数] {best_params}", flush=True)

    comp_dicts = {c: parse_comp(c) for c in df["comp"].unique()}
    elements = sorted({el for d in comp_dicts.values() for el in d})
    print(f"[元素] {len(elements)} 个: {','.join(elements)}", flush=True)
    etab, notes = build_element_table(elements)
    print(f"[元素表] 兜底/缺失记录 {len(notes)} 条:", flush=True)
    for n in notes:
        print(f"  - {n}", flush=True)

    # 覆盖率表
    cov = pd.DataFrame([{"property": p,
                         "n_missing": sum(np.isnan(etab[s][p]) for s in elements),
                         "coverage": 1 - np.mean([np.isnan(etab[s][p])
                                                  for s in elements])}
                        for p in NEW_PROPS])
    cov.to_csv(OUT / "desrefine_coverage.csv", index=False)
    print(cov.to_string(index=False), flush=True)

    # 展开新特征（每个组成一行）
    rows = {c: comp_stats(d, etab, NEW_PROPS) for c, d in comp_dicts.items()}
    new_df = pd.DataFrame.from_dict(rows, orient="index")
    # mode 系列 = 13 个性质（6 既有 + 7 新）的众数统计
    mode_rows = {c: {f"{p}_mode": comp_stats(d, etab, [p])[f"{p}_mode"]
                     for p in BASE_PROPS + NEW_PROPS}
                 for c, d in comp_dicts.items()}
    mode_df = pd.DataFrame.from_dict(mode_rows, orient="index")
    new_df = new_df.drop(columns=[c for c in new_df.columns
                                  if c.endswith("_mode")])
    new_df = pd.concat([new_df, mode_df], axis=1)
    new_df = new_df.loc[df["comp"]].reset_index(drop=True)
    print(f"[新特征] {new_df.shape[1]} 个；NaN 单元格 "
          f"{int(new_df.isna().sum().sum())}", flush=True)
    na_cols = new_df.columns[new_df.isna().any()].tolist()
    print(f"[新特征] 含 NaN 的列: {na_cols}", flush=True)
    new_df.to_csv(OUT / "desrefine_new_features.csv", index=False)

    # 族 → 列
    fam_cols = {fam: [c for c in new_df.columns
                      if not c.endswith("_mode")
                      and any(c.startswith(p + "_") for p in props)]
                for fam, props in FAMILIES.items()}  # mode 单列一族，族间互斥
    mode_cols = [c for c in new_df.columns if c.endswith("_mode")]
    all_new = list(new_df.columns)

    configs = {
        "baseline_36": df[base_feats],
        "plus_melting": pd.concat([df[base_feats], new_df[fam_cols["熔点族"]]], axis=1),
        "plus_mendeleevnum": pd.concat([df[base_feats], new_df[fam_cols["门捷列夫数族"]]], axis=1),
        "plus_unfilled": pd.concat([df[base_feats], new_df[fam_cols["未填满电子族"]]], axis=1),
        "plus_valence_covalent": pd.concat([df[base_feats], new_df[fam_cols["价电子与共价半径族"]]], axis=1),
        "plus_mode": pd.concat([df[base_feats], new_df[mode_cols]], axis=1),
        "full_merged": pd.concat([df[base_feats], new_df[all_new]], axis=1),
    }
    for tag, X in configs.items():
        print(f"  配置 {tag:22s} n_feat={X.shape[1]}", flush=True)

    # ---- (1) 固定参数消融（同一批 fold）
    print("\n===== (1) 固定参数 GroupKFold(5, comp) 消融 =====", flush=True)
    abl_rows, fold_maes = [], {}
    for tag, X in configs.items():
        maes = fixed_ablation(X, y, groups, best_params)
        fold_maes[tag] = maes
        abl_rows.append({"config": tag, "n_features": X.shape[1],
                         "MAE_mean": maes.mean(), "MAE_std": maes.std(),
                         "fold_MAEs": "|".join(f"{m:.4f}" for m in maes)})
        print(f"  {tag:22s} MAE {maes.mean():.4f}±{maes.std():.4f}", flush=True)
    abl = pd.DataFrame(abl_rows)
    base_mae = abl.loc[abl.config == "baseline_36", "MAE_mean"].iloc[0]
    abl["delta_vs_baseline36"] = base_mae - abl["MAE_mean"]
    abl["delta_vs_0.1198"] = BASELINE_REF - abl["MAE_mean"]
    abl.to_csv(OUT / "desrefine_ablation.csv", index=False)
    print(f"[基线复现] baseline_36 固定参数 MAE={base_mae:.4f}（锁定值 0.1198）",
          flush=True)

    # ---- (2) 嵌套 CV（所有配置；内层拟合超参）
    print("\n===== (2) 嵌套 CV（外5/内4，内层随机搜索30组） =====", flush=True)
    nested = pd.concat([nested_cv(X, y, groups, tag)
                        for tag, X in configs.items()])
    nested.to_csv(OUT / "desrefine_nested_cv.csv", index=False)
    nsum = nested.groupby("config")["MAE"].agg(["mean", "std"]).reset_index()
    nbase = nsum.loc[nsum.config == "baseline_36", "mean"].iloc[0]
    print(nsum.round(4).to_string(index=False), flush=True)

    # ---- 汇总 + 判定
    summ = abl.merge(nsum.rename(columns={"mean": "nested_MAE_mean",
                                          "std": "nested_MAE_std"}),
                     on="config")
    summ["nested_delta_vs_baseline36"] = nbase - summ["nested_MAE_mean"]
    summ["nested_delta_vs_0.1198"] = BASELINE_REF - summ["nested_MAE_mean"]
    summ["adopted"] = (summ["config"] != "baseline_36") & (
        summ["nested_delta_vs_baseline36"] > ADOPT_THRESHOLD)
    summ.to_csv(OUT / "desrefine_config_summary.csv", index=False)
    print("\n[汇总]", flush=True)
    print(summ.round(4).to_string(index=False), flush=True)
    print(f"[判定] 采纳阈值 Δ>{ADOPT_THRESHOLD} eV（嵌套CV vs 嵌套基线）；"
          f"采纳配置: {summ.loc[summ.adopted, 'config'].tolist() or '无'}", flush=True)

    # ---- (3) 共线性诊断：新 48 特征 vs 既有 36 特征
    print("\n===== (3) 共线性诊断（Pearson） =====", flush=True)
    base_X = df[base_feats]
    corr = pd.concat([base_X.reset_index(drop=True), new_df], axis=1).corr()
    sub = corr.loc[all_new, base_feats]
    pairs = []
    for c in all_new:
        for b in base_feats:
            r = sub.loc[c, b]
            if abs(r) > 0.9:
                pairs.append({"new_feature": c, "base_feature": b,
                              "pearson_r": round(float(r), 4)})
    col_df = pd.DataFrame(pairs).sort_values(
        "pearson_r", key=abs, ascending=False)
    col_df.to_csv(OUT / "desrefine_collinearity.csv", index=False)
    print(f"[共线性] |ρ|>0.9 的新-旧特征对 {len(col_df)} 个", flush=True)
    print(col_df.to_string(index=False), flush=True)
    maxcorr = sub.abs().max(axis=1).sort_values(ascending=False)
    print("[共线性] 每个新特征与既有特征的最大 |ρ|：", flush=True)
    print(maxcorr.round(3).to_string(), flush=True)

    # ---- (4) SHAP 族分解（全量合并模型，信息论视角）
    print("\n===== (4) SHAP 族分解（full_merged 全量训练） =====", flush=True)
    import shap
    X_full = configs["full_merged"]
    m = XGBRegressor(random_state=SEED, n_jobs=-1, **best_params)
    m.fit(X_full, y)
    sv = shap.TreeExplainer(m).shap_values(X_full)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X_full.columns)
    shap_family = {
        "基线-组成元素性质族(6性质×5统计)": [c for c in base_feats
                                        if any(c.startswith(p + "_")
                                               for p in BASE_PROPS)],
        "基线-n_elements": ["n_elements"],
        "基线-结构/晶面族": [c for c in base_feats
                          if c.startswith(("structure_", "facet_"))],
        "新-熔点族": fam_cols["熔点族"],
        "新-门捷列夫数族": fam_cols["门捷列夫数族"],
        "新-未填满电子族": fam_cols["未填满电子族"],
        "新-价电子与共价半径族": fam_cols["价电子与共价半径族"],
        "新-mode系列": mode_cols,
    }
    total = imp.sum()
    fam_rows = []
    for fam, cols in shap_family.items():
        cols = [c for c in cols if c in imp.index]
        fam_rows.append({"family": fam, "n_features": len(cols),
                         "mean_abs_shap_sum": float(imp[cols].sum()),
                         "share_pct": float(100 * imp[cols].sum() / total),
                         "top_feature": imp[cols].idxmax() if cols else None,
                         "top_feature_shap": float(imp[cols].max()) if cols else None})
    fam_df = pd.DataFrame(fam_rows).sort_values("mean_abs_shap_sum",
                                                ascending=False)
    fam_df.to_csv(OUT / "desrefine_shap_family.csv", index=False)
    print(fam_df.round(4).to_string(index=False), flush=True)
    new_share = fam_df[fam_df.family.str.startswith("新")]["share_pct"].sum()
    print(f"[SHAP] 全部新特征族合计 mean|SHAP| 份额 {new_share:.2f}%", flush=True)

    print("\n[完成] 产出写入 outputs/desrefine_*.csv", flush=True)


if __name__ == "__main__":
    main()
