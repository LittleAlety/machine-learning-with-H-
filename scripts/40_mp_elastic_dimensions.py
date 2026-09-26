# -*- coding: utf-8 -*-
"""40_mp_elastic_dimensions.py - DFT bulk descriptors from Materials Project.

The cheap mendeleev elemental stats (script 38) did not improve the model. A more
physically grounded, DFT-relaxation-derived set of bulk descriptors is available
from Materials Project elasticity: bulk modulus K_VRH, shear modulus G_VRH and
universal anisotropy index for each element's stable phase. We pull these via the
official API, expand them with the same composition-weighted convention, and
screen under the same GroupKFold / 3-seed fair contract.

Key read ONLY from MP_API_KEY env var. Outputs:
    outputs/mp_elastic_dimensions_screen.csv
    notes/mp_elastic_dimensions.md
    data_processed/hstar_features_mp_elastic.csv (candidate, only if it helps)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://api.materialsproject.org"
IN_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
CACHE = ROOT / "data_raw" / "mp_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_SCREEN = ROOT / "outputs" / "mp_elastic_dimensions_screen.csv"
OUT_MD = ROOT / "notes" / "mp_elastic_dimensions.md"
OUT_FEATURES = ROOT / "data_processed" / "hstar_features_mp_elastic.csv"

PARAMS = dict(learning_rate=0.03, max_depth=7, n_estimators=400, subsample=0.8,
              n_jobs=2, verbosity=0)
SEEDS = (0, 2024, 7)
import re
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
DESCS = ["kv", "gv", "ani"]   # bulk modulus, shear modulus, anisotropy
STATS = ["wmean", "max", "min", "range", "wstd"]


def parse_comp(c):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(c)}


def api(path, key, params):
    url = f"{BASE}/{path}?{urlencode(params, doseq=True)}"
    req = Request(url, headers={"X-API-Key": key,
                                "User-Agent": "Mozilla/5.0 ai-surface-cat"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def stable_ids(elements, key):
    """element -> stable-phase material_id (cached)."""
    cf = CACHE / "stable_ids.json"
    out = json.loads(cf.read_text()) if cf.exists() else {}
    for el in elements:
        if el in out:
            continue
        d = api("materials/summary/", key, {
            "formula": el, "_limit": 50,
            "_fields": "material_id,is_stable,energy_above_hull"})
        stable = [x["material_id"] for x in d["data"] if x.get("is_stable")]
        out[el] = sorted(stable)[0] if stable else None
        cf.write_text(json.dumps(out))
        time.sleep(0.4)
    return out


def elasticity(material_ids, key):
    """material_id -> {kv,gv,ani} (batched)."""
    cf = CACHE / "elasticity.json"
    out = json.loads(cf.read_text()) if cf.exists() else {}
    todo = [m for m in material_ids if m and m not in out]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        d = api("materials/elasticity/", key, {
            "material_ids": ",".join(chunk),
            "_fields": "material_id,bulk_modulus,shear_modulus,universal_anisotropy"})
        for x in d["data"]:
            out[x["material_id"]] = {
                "kv": (x.get("bulk_modulus") or {}).get("vrh"),
                "gv": (x.get("shear_modulus") or {}).get("vrh"),
                "ani": x.get("universal_anisotropy")}
        cf.write_text(json.dumps(out))
        time.sleep(0.5)
    return out


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
        f = {}
        for desc in DESCS:
            v = np.array([table[desc][e] for e in els], float)
            f[f"{desc}_wmean"] = wmean(v, w)
            f[f"{desc}_max"] = float(v.max())
            f[f"{desc}_min"] = float(v.min())
            f[f"{desc}_range"] = float(v.max() - v.min())
            f[f"{desc}_wstd"] = wstd(v, w)
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
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("[fatal] set MP_API_KEY env var")
    df = pd.read_csv(IN_CSV)
    y = df["energy_eV"].reset_index(drop=True)
    groups = df["comp"].reset_index(drop=True)
    id_cols = {"comp", "structure", "facet", "n_raw", "energy_eV"}
    base_cols = [c for c in df.columns if c not in id_cols]

    elements = sorted({el for c in df["comp"] for el in parse_comp(c)})
    sid = stable_ids(elements, key)
    missing_el = [e for e in elements if not sid[e]]
    elx = elasticity([sid[e] for e in elements if sid[e]], key)

    # per-element table with mean imputation; record gaps
    table, gaps = {}, []
    for desc in DESCS:
        vals = {}
        for e in elements:
            m = sid.get(e)
            v = elx.get(m, {}).get(desc) if m else None
            vals[e] = float(v) if v is not None else np.nan
        present = [x for x in vals.values() if not np.isnan(x)]
        fill = float(np.mean(present))
        for e, v in vals.items():
            if np.isnan(v):
                vals[e] = fill
                gaps.append(f"{e}.{desc}")
        table[desc] = vals
    print(f"[elements] {len(elements)}; no stable id: {missing_el}; imputed: {gaps}")

    newX = expand(df, table).reset_index(drop=True)
    Xbase = df[base_cols].reset_index(drop=True)
    configs = {"baseline": []}
    for desc in DESCS:
        configs[f"+{desc}"] = [c for c in newX.columns if c.startswith(desc + "_")]
    configs["+all"] = list(newX.columns)

    results, done = [], set()
    if OUT_SCREEN.exists():
        results = pd.read_csv(OUT_SCREEN).to_dict("records")
        done = {r["config"] for r in results}
    base_mae = None
    for name, add in configs.items():
        if name in done:
            if name == "baseline":
                base_mae = next(r["groupkfold_MAE"] for r in results if r["config"] == "baseline")
            continue
        X = pd.concat([Xbase, newX[add]], axis=1) if add else Xbase
        maes = [cv_mae(X, y, groups, s) for s in SEEDS]
        mean, sd = float(np.mean(maes)), float(np.std(maes))
        if name == "baseline":
            base_mae = mean
        ref = base_mae if base_mae is not None else mean
        results.append({"config": name, "n_features": X.shape[1],
                        "groupkfold_MAE": round(mean, 5), "sd": round(sd, 5),
                        "delta_vs_baseline": round(mean - ref, 5)})
        print(f"{name:10s} n={X.shape[1]:3d} MAE={mean:.4f} Δ={mean-ref:+.4f}", flush=True)
        pd.DataFrame(results).to_csv(OUT_SCREEN, index=False, encoding="utf-8-sig")

    base_mae = next(r["groupkfold_MAE"] for r in results if r["config"] == "baseline")
    for r in results:
        r["delta_vs_baseline"] = round(r["groupkfold_MAE"] - base_mae, 5)
    pd.DataFrame(results).to_csv(OUT_SCREEN, index=False, encoding="utf-8-sig")

    winners = [r["config"][1:] for r in results
               if r["config"].startswith("+") and r["config"] != "+all"
               and r["delta_vs_baseline"] < 0]
    keep = [c for c in newX.columns if any(c.startswith(d + "_") for d in winners)]
    pd.concat([df.reset_index(drop=True), newX[keep]], axis=1).to_csv(
        OUT_FEATURES, index=False, encoding="utf-8-sig")

    md = ["# Materials Project 弹性体相描述符筛选", "",
          f"基线（84 特征）GroupKFold MAE = **{base_mae:.4f} eV**（3 种子）。", "",
          "| 配置 | 特征数 | MAE (eV) | Δ vs 基线 |", "|---|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['config']} | {r['n_features']} | {r['groupkfold_MAE']} | "
                  f"{r['delta_vs_baseline']:+.4f} |")
    md += ["", f"- 改善（Δ<0）的描述符：**{winners if winners else '无'}**",
           f"- 无稳定相 ID 的元素：{missing_el}",
           f"- 均值插补：{gaps}",
           "- 数据为 MP 稳定相 VASP(PBE) 弹性常数（K_VRH/G_VRH，GPa；各向异性无量纲）。",
           "- 候选文件需嵌套 CV/10 种子复核后才上线；体相模量本质仍为成分代理，"
           "真正增益预期来自弛豫返回的表面几何/电子结构（C-3 v2 回灌）。"]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[winners] {winners}")
    print("Wrote", OUT_SCREEN.name, OUT_MD.name, OUT_FEATURES.name)


if __name__ == "__main__":
    main()
