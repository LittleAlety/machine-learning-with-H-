# -*- coding: utf-8 -*-
"""38_feature_dimension_expansion.py — literature-grounded new feature screen.

The academic survey (doubao-academic-researcher) flagged several elemental
dimensions the current 84-feature model lacks. We add only values sourced
programmatically from mendeleev (no hand-typed constants):

    ea    electron affinity (eV)      Li/Ma/Xin feature engineering uses EN + IP + EA
    aw    atomic weight               basic identity
    rho   density                     Gong Li-adsorption: density highly correlated
    avol  atomic volume               size / packing
    pol   dipole polarizability       adsorbate polarisation response
    tc    thermal conductivity        (weak prior; tested, kept only if it helps)
    shc   specific heat capacity      lattice response
    engh  Ghosh electronegativity     alternative EN scale

Each property is expanded into composition-weighted stats (wmean/max/min/range/
wstd) using the SAME convention as 06_hstar_features.py. We then run GroupKFold
(canonical composition) with the tuned XGBoost over 3 seeds and compare:
baseline vs +each property vs +all. Only properties that genuinely reduce MAE
under this fair contract are promoted to a candidate enriched feature file.

Outputs:
    dimension_expansion_screen.csv   per-config MAE / delta
    dimension_expansion_report.md
    hstar_features_expanded.csv      baseline + properties that pass (candidate)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
OUT_SCREEN = ROOT / "outputs" / "dimension_expansion_screen.csv"
OUT_REPORT = ROOT / "notes" / "dimension_expansion_report.md"
OUT_FEATURES = ROOT / "data_processed" / "hstar_features_expanded.csv"

PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8,
              n_jobs=2, verbosity=0)
SEEDS = (0, 2024, 7)
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

NEW_PROPS = {
    "ea": ("electron_affinity", "电子亲和能 eV"),
    "aw": ("atomic_weight", "原子量"),
    "rho": ("density", "密度 g/cm3"),
    "avol": ("atomic_volume", "原子体积 cm3/mol"),
    "pol": ("dipole_polarizability", "偶极极化率 a0^3"),
    "tc": ("thermal_conductivity", "热导率 W/mK"),
    "shc": ("specific_heat_capacity", "比热容 J/gK"),
    "engh": ("en_ghosh", "Ghosh 电负性"),
}
STATS = ["wmean", "max", "min", "range", "wstd"]


def parse_comp(comp):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(comp)}


def build_new_table(elements):
    from mendeleev import element as md_el
    raw = {p: {} for p in NEW_PROPS}
    for sym in elements:
        e = md_el(sym)
        for p, (attr, _) in NEW_PROPS.items():
            raw[p][sym] = getattr(e, attr, None)
    # Impute per property across elements; record gaps honestly.
    table, gaps = {}, []
    for p in NEW_PROPS:
        vals = {s: (float(v) if v is not None else np.nan) for s, v in raw[p].items()}
        present = [v for v in vals.values() if not np.isnan(v)]
        fill = float(np.mean(present)) if present else 0.0
        for s, v in vals.items():
            if np.isnan(v):
                vals[s] = fill
                gaps.append(f"{s}.{p}")
        table[p] = vals
    return table, gaps


def wmean(v, w):
    return float(np.average(v, weights=w))


def wstd(v, w):
    mu = np.average(v, weights=w)
    return float(np.sqrt(np.average((np.asarray(v) - mu) ** 2, weights=w)))


def expand(df, table):
    comp_dicts = {c: parse_comp(c) for c in df["comp"].unique()}
    rows = []
    for _, r in df.iterrows():
        d = comp_dicts[r["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        f = {}
        for p in NEW_PROPS:
            v = np.array([table[p][e] for e in els])
            f[f"{p}_wmean"] = wmean(v, w)
            f[f"{p}_max"] = float(v.max())
            f[f"{p}_min"] = float(v.min())
            f[f"{p}_range"] = float(v.max() - v.min())
            f[f"{p}_wstd"] = wstd(v, w)
        rows.append(f)
    return pd.DataFrame(rows, index=df.index)


def cv_mae(X, y, groups, seed):
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = XGBRegressor(random_state=seed, **PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return float(np.mean(np.abs(oof - y)))


def main():
    df = pd.read_csv(IN_CSV)
    y = df["energy_eV"]
    groups = df["comp"]
    id_cols = {"comp", "structure", "facet", "n_raw", "energy_eV"}
    base_cols = [c for c in df.columns if c not in id_cols]

    elements = sorted({el for c in df["comp"] for el in parse_comp(c)})
    table, gaps = build_new_table(elements)
    newX = expand(df, table)
    print(f"[elements] {len(elements)}; imputed gaps: {len(gaps)} -> {gaps[:8]}")

    configs = {"baseline": []}
    for p in NEW_PROPS:
        configs[f"+{p}"] = [c for c in newX.columns if c.startswith(p + "_")]
    configs["+all"] = list(newX.columns)

    # Resumable: reload finished configs so interruptions never lose work.
    results, done = [], set()
    if OUT_SCREEN.exists():
        prev = pd.read_csv(OUT_SCREEN)
        results = prev.to_dict("records")
        done = set(prev["config"])
    base_mae = None
    Xbase = df[base_cols].reset_index(drop=True)
    newX = newX.reset_index(drop=True)
    for name, add in configs.items():
        if name in done:
            print(f"{name:10s} (cached)")
            if name == "baseline":
                base_mae = next(r["groupkfold_MAE"] for r in results if r["config"] == "baseline")
            continue
        X = pd.concat([Xbase, newX[add]], axis=1) if add else Xbase
        maes = [cv_mae(X, y.reset_index(drop=True), groups.reset_index(drop=True), s)
                for s in SEEDS]
        mean, sd = float(np.mean(maes)), float(np.std(maes))
        if name == "baseline":
            base_mae = mean
        ref = base_mae if base_mae is not None else mean
        results.append({"config": name, "n_features": X.shape[1],
                        "groupkfold_MAE": round(mean, 5), "sd": round(sd, 5),
                        "delta_vs_baseline": round(mean - ref, 5)})
        print(f"{name:10s} n={X.shape[1]:3d} MAE={mean:.4f} Δ={mean-ref:+.4f}", flush=True)
        pd.DataFrame(results).to_csv(OUT_SCREEN, index=False, encoding="utf-8-sig")  # checkpoint

    base_mae = next(r["groupkfold_MAE"] for r in results if r["config"] == "baseline")
    for r in results:
        r["delta_vs_baseline"] = round(r["groupkfold_MAE"] - base_mae, 5)
    scr = pd.DataFrame(results)
    scr.to_csv(OUT_SCREEN, index=False, encoding="utf-8-sig")

    # Promote properties whose mean delta < 0 (improve) to candidate file.
    winners = [r["config"][1:] for r in results
               if r["config"].startswith("+") and r["config"] != "+all"
               and r["delta_vs_baseline"] < 0]
    keep_cols = [c for c in newX.columns if any(c.startswith(p + "_") for p in winners)]
    out = pd.concat([df.reset_index(drop=True),
                     newX[keep_cols].reset_index(drop=True)], axis=1)
    out.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

    allrow = next(r for r in results if r["config"] == "+all")
    md = ["# 新维度扩展筛选报告（文献驱动 + 公平契约）", "",
          f"基线（84 特征）GroupKFold MAE = **{base_mae:.4f} eV**（3 种子）。",
          "", "| 配置 | 特征数 | MAE (eV) | Δ vs 基线 |", "|---|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['config']} | {r['n_features']} | {r['groupkfold_MAE']} | "
                  f"{r['delta_vs_baseline']:+.4f} |")
    md += ["", f"- 单独加入即改善（Δ<0）的维度：**{winners if winners else '无'}**",
           f"- 全量加入（+all）Δ = **{allrow['delta_vs_baseline']:+.4f} eV**",
           f"- 候选扩展特征文件：`{OUT_FEATURES.name}`（仅保留获胜维度，需嵌套 CV/10 种子复核后才上线）",
           "", "## 仍需 DFT/弛豫才能加入的维度（C-3 v2 回灌后）",
           "功函数 Φ（需 curated 表或 slab 计算）、内聚能、d 带中心/宽度、Fermi softness、"
           "广义配位数 GCN、slab 图神经网络（AdsorbML/OC20）、磁矩。",
           "", "## 文献依据",
           "- Li, Ma, Xin, *Catal. Today* (feature engineering: EN + IP + EA).",
           "- Noh et al., *Chem. Sci.* (d-band width + EN, active learning).",
           "- Kirkvold et al., *J. Phys. Chem. Lett.* 2024 (CatEmbed categorical embeddings).",
           "- Huang & Zhuang (Fermi softness); Calle-Vallejo et al. (GCN).",
           "- Lan et al., AdsorbML; Price et al., *Sci. Adv.* (strain GNN).",
           "- Liasi et al. 2026 (work function HER descriptor); Trasatti 1972.",
           "- Tshitoyan et al., *Nature* 2019 (mat2vec); Zhou et al., Atom2Vec."]
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"[winners] {winners}")
    print(f"Wrote {OUT_SCREEN.name}, {OUT_REPORT.name}, {OUT_FEATURES.name}")


if __name__ == "__main__":
    main()
