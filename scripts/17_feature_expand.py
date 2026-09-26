# -*- coding: utf-8 -*-
"""
17_feature_expand.py — Round 4 Agent G：特征扩展消融实验

公平性契约（plan_round4.md）：
- 基线：XGBoost_tuned（hstar_best_params.json：lr=0.03, depth=7, n=400, sub=0.8）
  + v2 原 36 特征 + GroupKFold(5, 规范化 comp)，应复现 MAE=0.1198±0.0087 eV
  （注：任务书所谓"出厂参数"实为 tuned 参数；XGBoost_default 的 GroupKFold MAE=0.1296，
   0.1198 只能由 tuned 参数复现，故基线采用 tuned 参数）
- 种子 42；site_spread 禁入；一次只加一组特征；改善 >0.005 eV 才判"有效"
- 任何"有效"扩展 → 嵌套 CV（外 GroupKFold5 / 内 GroupKFold4，内层随机搜索 30 组）+ SHAP 验证

特征组：
(a) 位点特征（hstar_facet_check.csv 的 site_H_official，CatHub 官方校验位点元数据）：
    解析为 6 列多热编码 site_{top,bridge,hollow_fcc,hollow_hcp,other,missing}；
    top-tilt/bridge-tilt/hollow-tilt 归入对应基础类型；4fold 及其他归入 other；
    缺失行 5 个类型列置 NaN（XGBoost 原生处理），missing=1。
(b) mendeleev 扩展属性：在现有 6 属性之外枚举剩余数值型属性，
    37 个相关元素中缺失率 >10% 的属性直接弃用；保留属性按
    wmean/max/min/range/wstd 五统计展开；组元缺失 → 该组成的统计置 NaN（XGBoost 原生）。

产出：outputs/featexp_ablation.csv（+ 若触发验证：featexp_nested_cv.csv、
featexp_shap_importance.csv）；特征明细打印到 stdout 供报告引用。
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
ADOPT_THRESHOLD = 0.005  # 改善需 > 0.005 eV

SITE_KEYS = {"top", "top-tilt", "bridge", "bridge-tilt",
             "hollow", "hollow-tilt", "4fold"}
SITE_COLS = ["site_top", "site_bridge", "site_hollow_fcc",
             "site_hollow_hcp", "site_other", "site_missing"]

# mendeleev 1.2.0 中除现有 6 属性（en_pauling/atomic_radius/group_id/period/
# ionenergies[1]/econf→d_el）外，枚举到的全部数值型候选属性
# （density/fusion_heat/evaporation_heat 在 mendeleev 1.2.0 中不存在，已确认）
MENDELEEV_CANDIDATES = [
    "electron_affinity", "boiling_point", "melting_point",
    "specific_heat_capacity", "thermal_conductivity",
    "covalent_radius", "covalent_radius_cordero", "covalent_radius_pyykko",
    "covalent_radius_bragg", "atomic_volume", "dipole_polarizability",
    "lattice_constant", "gas_basicity", "proton_affinity",
    "en_ghosh", "en_allen", "atomic_weight",
    "vdw_radius", "vdw_radius_alvarez",
    "metallic_radius", "metallic_radius_c12",
    "abundance_crust", "atomic_radius_rahm",
]
MISS_RATE_MAX = 0.10  # 元素缺失率 >10% 弃用
STATS = ["wmean", "max", "min", "range", "wstd"]
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


# ---------------------------------------------------------------- 位点特征
def parse_sites(s):
    """site_H_official（'|' 连接的位点注解集合）→ 条目列表，如 ['bridge','A_B','A']。"""
    entries, cur = [], None
    for tok in s.split("|"):
        if tok in SITE_KEYS:
            cur = [tok]
            entries.append(cur)
        else:
            cur.append(tok)
    return entries


def site_features(s):
    """→ 6 列多热；缺失行类型列 NaN + missing=1。"""
    if pd.isna(s):
        return dict.fromkeys(SITE_COLS[:5], np.nan) | {"site_missing": 1}
    es = parse_sites(s)
    base = [e[0].replace("-tilt", "") for e in es]
    hfcc = any(b == "hollow" and e[-1] == "FCC" for b, e in zip(base, es))
    hhcp = any(b == "hollow" and e[-1] == "HCP" for b, e in zip(base, es))
    return {
        "site_top": int("top" in base),
        "site_bridge": int("bridge" in base),
        "site_hollow_fcc": int(hfcc),
        "site_hollow_hcp": int(hhcp),
        "site_other": int(any(b not in ("top", "bridge", "hollow") for b in base)),
        "site_missing": 0,
    }


# ---------------------------------------------------------------- mendeleev 扩展属性
def parse_comp(comp):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(comp)}


def build_extended_element_table(elements):
    """→ (保留属性表 {el: {prop: float|None}}, 保留属性列表, 弃用记录)。"""
    from mendeleev import element as md_element
    raw = {sym: {p: getattr(md_element(sym), p, None)
                 for p in MENDELEEV_CANDIDATES} for sym in elements}
    kept, dropped = [], []
    for p in MENDELEEV_CANDIDATES:
        miss = [s for s in elements if raw[s][p] is None]
        rate = len(miss) / len(elements)
        if rate > MISS_RATE_MAX:
            dropped.append({"property": p, "n_missing": len(miss),
                            "miss_rate": round(rate, 4),
                            "missing_elements": ",".join(miss)})
        else:
            kept.append(p)
            if miss:
                print(f"  [保留但含缺失] {p}: {len(miss)}/{len(elements)} "
                      f"({rate:.1%}) 缺 {miss}")
    print(f"[mendeleev] 候选 {len(MENDELEEV_CANDIDATES)} 属性 → 保留 {len(kept)}，"
          f"弃用 {len(dropped)}（缺失率 >{MISS_RATE_MAX:.0%}）")
    for d in dropped:
        print(f"  [弃用] {d['property']}: {d['n_missing']}/{len(elements)} "
              f"({d['miss_rate']:.1%}) 缺 [{d['missing_elements']}]")
    table = {sym: {p: (float(raw[sym][p]) if raw[sym][p] is not None else None)
                   for p in kept} for sym in elements}
    return table, kept, dropped


def expand_stats(comp_dicts, etab, kept):
    """按现有五统计展开；组元缺失 → 该组成该属性全部统计置 NaN。"""
    rows = {}
    for comp, d in comp_dicts.items():
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        feats = {}
        for p in kept:
            vals = [etab[e][p] for e in els]
            if any(v is None for v in vals):
                for s in STATS:
                    feats[f"{p}_{s}"] = np.nan
                continue
            v = np.array(vals)
            mu = np.average(v, weights=w)
            feats[f"{p}_wmean"] = float(mu)
            feats[f"{p}_max"] = float(v.max())
            feats[f"{p}_min"] = float(v.min())
            feats[f"{p}_range"] = float(v.max() - v.min())
            feats[f"{p}_wstd"] = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
        rows[comp] = feats
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- 评估
def groupkfold_mae(X, y, groups, params):
    """固定 GroupKFold(5, comp)（同一批 fold 对所有变体一致），返回逐折 MAE。"""
    gkf = GroupKFold(n_splits=5)
    maes = []
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
        m.fit(X.iloc[tr], y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(X.iloc[te])))
    return np.array(maes)


def nested_cv(X, y, groups, tag):
    """外 GroupKFold(5) / 内 GroupKFold(4)，内层随机搜索 30 组。"""
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
        pred = rs.best_estimator_.predict(X.iloc[te])
        mae = mean_absolute_error(y[te], pred)
        rows.append({"variant": tag, "outer_fold": k, "MAE": mae,
                     "best_params": json.dumps(rs.best_params_)})
        print(f"    [{tag}] 外折 {k}: MAE={mae:.4f}  best={rs.best_params_}")
    return pd.DataFrame(rows)


def shap_check(X, y, params, new_cols, tag):
    """全量训练 + TreeExplainer，报告新特征在 mean|SHAP| 中的排位与份额。"""
    import shap
    m = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    m.fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = pd.DataFrame({"feature": X.columns,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    imp["is_new"] = imp["feature"].isin(new_cols)
    total = imp["mean_abs_shap"].sum()
    new_share = imp.loc[imp.is_new, "mean_abs_shap"].sum() / total
    n_top20 = (imp.head(20).is_new).sum()
    print(f"  [SHAP/{tag}] 新特征 mean|SHAP| 总份额 {new_share:.2%}；"
          f"Top20 中新特征 {n_top20} 个")
    print(imp[imp.is_new].head(10).to_string(index=False))
    imp.to_csv(OUT / f"featexp_shap_importance_{tag}.csv", index=False)
    return new_share, n_top20


# ---------------------------------------------------------------- 主流程
def main():
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    fc = pd.read_csv(DP / "hstar_facet_check.csv")
    y = df["energy_eV"].values
    groups = df["comp"].values
    best_params = json.loads((DP / "hstar_best_params.json").read_text()
                             )["xgb_best_params"]
    print(f"[基线参数] XGBoost_tuned {best_params}（hstar_best_params.json）")

    base_feats = [c for c in df.columns if c not in ID_COLS]
    print(f"[读入] v2 {df.shape[0]} 行 × {len(base_feats)} 基线特征；"
          f"{len(set(groups))} 个 comp 组")

    # ---- (a) 位点特征
    na_rate = fc["site_H_official"].isna().mean()
    print(f"\n[位点特征] site_H_official 缺失 {fc['site_H_official'].isna().sum()}"
          f"/{len(fc)} = {na_rate:.2%}（阈值 20%）→ {'保留' if na_rate <= 0.20 else '整组放弃'}")
    assert set(fc["comp"]) == set(df["comp"]), "comp 集合不一致"
    site_df = pd.DataFrame([site_features(s) for s in
                            fc.set_index("comp").loc[df["comp"], "site_H_official"]],
                           index=df.index)
    print("[位点特征] 均值（非缺失行）:")
    print(site_df.mean().round(4).to_string())

    # ---- (b) mendeleev 扩展属性
    comp_dicts = {c: parse_comp(c) for c in df["comp"].unique()}
    elements = sorted({el for d in comp_dicts.values() for el in d})
    print(f"\n[mendeleev] {len(elements)} 个相关元素")
    etab, kept, dropped = build_extended_element_table(elements)
    prop_df = expand_stats(comp_dicts, etab, kept)
    prop_df = prop_df.loc[df["comp"]].reset_index(drop=True)
    prop_cols = list(prop_df.columns)
    n_nan_cells = int(prop_df.isna().sum().sum())
    print(f"[mendeleev] 展开 {len(kept)} 属性 × 5 统计 = {len(prop_cols)} 特征；"
          f"NaN 单元格 {n_nan_cells}（组元缺失传播，XGBoost 原生处理）")
    pd.DataFrame(dropped).to_csv(OUT / "featexp_dropped_properties.csv", index=False)

    # ---- 消融（同一批 GroupKFold fold）
    variants = {
        "baseline_v2_36f": df[base_feats],
        "plus_site": pd.concat([df[base_feats], site_df], axis=1),
        "plus_mendeleev": pd.concat([df[base_feats], prop_df], axis=1),
        "plus_site_mendeleev": pd.concat([df[base_feats], site_df, prop_df], axis=1),
    }
    print("\n===== 消融：GroupKFold(5, comp)，XGBoost_tuned，seed=42 =====")
    rows, fold_maes = [], {}
    for tag, X in variants.items():
        maes = groupkfold_mae(X, y, groups, best_params)
        fold_maes[tag] = maes
        rows.append({"variant": tag, "n_features": X.shape[1],
                     "MAE_mean": maes.mean(), "MAE_std": maes.std(),
                     "fold_MAEs": "|".join(f"{m:.4f}" for m in maes)})
        print(f"  {tag:22s} n_feat={X.shape[1]:4d}  MAE {maes.mean():.4f}±{maes.std():.4f}")
    abl = pd.DataFrame(rows)
    base_mae = abl.loc[abl.variant == "baseline_v2_36f", "MAE_mean"].iloc[0]
    abl["delta_vs_baseline"] = base_mae - abl["MAE_mean"]  # 正=改善
    # 逐折配对差（同一批 fold，直接可比）
    for tag in variants:
        if tag != "baseline_v2_36f":
            d = fold_maes["baseline_v2_36f"] - fold_maes[tag]
            abl.loc[abl.variant == tag, "paired_fold_delta_mean"] = d.mean()
            abl.loc[abl.variant == tag, "paired_fold_delta_min"] = d.min()
    abl["effective"] = abl["delta_vs_baseline"] > ADOPT_THRESHOLD
    abl.to_csv(OUT / "featexp_ablation.csv", index=False)
    print(f"\n[写出] outputs/featexp_ablation.csv（基线 MAE={base_mae:.4f}，"
          f"改善阈值 >{ADOPT_THRESHOLD} eV）")
    print(abl.round(4).to_string(index=False))

    # ---- 防假阳性：对"有效"变体补嵌套 CV + SHAP
    winners = abl[(abl.effective) & (abl.variant != "baseline_v2_36f")]["variant"].tolist()
    if not winners:
        print("\n[结论] 无变体改善超过 0.005 eV → 不触发嵌套 CV/SHAP，"
              "建议保留现有 36 特征模型")
        return
    print(f"\n[防假阳性] 有效变体: {winners} → 嵌套 CV（外5内4，内层随机搜索30组）+ SHAP")
    nested_frames = []
    for tag in ["baseline_v2_36f"] + winners:  # 基线同样跑嵌套 CV 作对照
        nested_frames.append(nested_cv(variants[tag], y, groups, tag))
    nested = pd.concat(nested_frames)
    nested.to_csv(OUT / "featexp_nested_cv.csv", index=False)
    print(nested.groupby("variant")["MAE"].agg(["mean", "std"]).round(4).to_string())
    new_cols_map = {"plus_site": SITE_COLS, "plus_mendeleev": prop_cols,
                    "plus_site_mendeleev": SITE_COLS + prop_cols}
    for tag in winners:
        shap_check(variants[tag], y, best_params, new_cols_map[tag], tag)


if __name__ == "__main__":
    main()
