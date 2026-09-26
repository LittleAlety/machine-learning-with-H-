# -*- coding: utf-8 -*-
"""41_omat_reference_dimension.py - OMat24 elemental bulk reference energy descriptor.

The full OMat24 corpus (100M+ bulk PBE frames) is multi-GB and its Meta CDN host is
not reliably reachable here. The dataset DOES ship a small, authoritative file of
elemental reference compounds (VASP PBE ComputedEntry total energies of the
reference elemental phases). We extract the per-element reference bulk energy per
atom (e_bulk_ref), expand it with the same composition-weighted convention, and
screen it under the same GroupKFold / 3-seed fair contract.

NOTE: this is a bulk energy reference, not a cohesive energy (isolated-atom energies
are not in the file) and not an adsorption label. Outputs:
    outputs/omat_reference_dimension_screen.csv
    notes/omat_reference_dimension.md
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
REF_GZ = ROOT / "data_raw" / "omat_elemental_references.json.gz"
OUT_SCREEN = ROOT / "outputs" / "omat_reference_dimension_screen.csv"
OUT_MD = ROOT / "notes" / "omat_reference_dimension.md"

PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8,
              n_jobs=2, verbosity=0)
SEEDS = (0, 2024, 7)
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
STATS = ["wmean", "max", "min", "range", "wstd"]


def parse_comp(c):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(c)}


def load_refs(elements):
    with gzip.open(REF_GZ, "rt") as f:
        raw = json.load(f)
    table, present, missing = {}, [], []
    for el in elements:
        e = raw.get(el)
        if e and e.get("energy") is not None:
            nat = float(sum(e["composition"].values()))
            table[el] = float(e["energy"]) / nat
            present.append(el)
        else:
            missing.append(el)
    fill = float(np.mean(list(table.values())))
    for el in missing:
        table[el] = fill
    return table, missing, fill


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
        w = np.array([d[e] for e in els], float)
        w /= w.sum()
        v = np.array([table[e] for e in els])
        rows.append({"eb_wmean": wmean(v, w), "eb_max": float(v.max()),
                     "eb_min": float(v.min()), "eb_range": float(v.max() - v.min()),
                     "eb_wstd": wstd(v, w)})
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
    y = df["energy_eV"].reset_index(drop=True)
    groups = df["comp"].reset_index(drop=True)
    id_cols = {"comp", "structure", "facet", "n_raw", "energy_eV"}
    base_cols = [c for c in df.columns if c not in id_cols]

    elements = sorted({el for c in df["comp"] for el in parse_comp(c)})
    table, missing, fill = load_refs(elements)
    print(f"[refs] present={len(elements)-len(missing)} missing={missing} (mean-fill {fill:.3f})")

    newX = expand(df, table).reset_index(drop=True)
    Xbase = df[base_cols].reset_index(drop=True)
    configs = {"baseline": [], "+ebulk_ref": list(newX.columns)}
    results = []
    base_mae = None
    for name, add in configs.items():
        X = pd.concat([Xbase, newX[add]], axis=1) if add else Xbase
        maes = [cv_mae(X, y, groups, s) for s in SEEDS]
        mean, sd = float(np.mean(maes)), float(np.std(maes))
        if name == "baseline":
            base_mae = mean
        results.append({"config": name, "n_features": X.shape[1],
                        "groupkfold_MAE": round(mean, 5), "sd": round(sd, 5),
                        "delta_vs_baseline": round(mean - base_mae, 5)})
        print(f"{name:12s} n={X.shape[1]:3d} MAE={mean:.4f} Δ={mean-base_mae:+.4f}", flush=True)
    pd.DataFrame(results).to_csv(OUT_SCREEN, index=False, encoding="utf-8-sig")

    md = ["# OMat24 元素体相参考能量描述符筛选", "",
          f"基线（84 特征）GroupKFold MAE = **{base_mae:.4f} eV**（3 种子）。", "",
          "| 配置 | 特征数 | MAE (eV) | Δ vs 基线 |", "|---|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['config']} | {r['n_features']} | {r['groupkfold_MAE']} | "
                  f"{r['delta_vs_baseline']:+.4f} |")
    md += ["", f"- OMat24 参考文件中缺失、按均值插补的元素：{missing}",
           "- 该量为元素参考相的 VASP(PBE) 体相总能/原子（eV），不是内聚能（缺孤立原子能量），"
           "也不是吸附标签。",
           "- OMat24 主体为体相晶体、不含表面 H 吸附；其对本吸附模型的增量预期有限，"
           "真正增益需表面弛豫返回的几何/电子结构（C-3 v2 回灌）。",
           "- 数据源：facebook/OMAT24 references/omat-elemental-reference-compounds.json.gz；"
           "Barroso-Luque et al., arXiv:2410.12771。"]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("Wrote", OUT_SCREEN.name, OUT_MD.name)


if __name__ == "__main__":
    main()
