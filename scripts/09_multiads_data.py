# -*- coding: utf-8 -*-
"""
09_multiads_data.py — Mamun 2019 全量 45,130 条吸附反应 → 多吸附种统一数据集

解析 data_raw/MamunHighT2019_adsorption.json.gz 全部吸附反应（H/C/N/O/S 及
CHx/NH/OH/SH/2H/2N/2O/H2O），吸附种从反应键名右端 "-> X*" 提取。
与 H* 线（05）口径一致：
  - 结构/晶面由 12 原子超胞计量模式推断（M12→A1(111)，A9B3→L12(111)，A6B6→L10(101)）
  - comp 规范化（元素字母序 + gcd 约简）
  - 同 (comp, structure, adsorbate) 的 _N 后缀条目为不同吸附构型，取最低 E_ads
    （最稳定位点），n_raw 记录构型数

特征化（复用 06_hstar_features.py 的函数，保证口径一致；importlib 按路径加载）：
  - 组成描述符：n_elements + 6 元素性质 × 5 加权统计（同 02/06）
  - 结构描述符：structure/facet one-hot
  - 吸附种描述符（新增）：
      ads_atom：与表面成键的原子（CHx→C, NH→N, OH/H2O→O, SH→S, 2H→H 等）
      ads_atom one-hot（H/C/N/O/S）+ 吸附原子元素性质（电负性/半径/族号，mendeleev）
      n_ads_units：吸附单元数（"2X*" 解离吸附为 2，其余为 1；结构信息，非标签派生）
  ⚠ 绝不引入任何由标签能量派生的特征（site_spread 泄漏教训，见 notes/05）。

输出：data_processed/multiads_dataset.csv（去重后数据集）
      data_processed/multiads_features.csv（特征化后，建模输入）
      data_processed/multiads_adsorbate_counts.csv（各吸附种条数统计：原始/去重后）
      figures/multiads_adsorbate_counts.png
"""
import gzip
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz"
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"

# ---- 复用 06 的解析与特征化函数（按路径加载，保证口径一致）----
_spec = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hf)
parse_comp, build_element_table = _hf.parse_comp, _hf.build_element_table
wmean, wstd = _hf.wmean, _hf.wstd
PROPS, STATS = _hf.PROPS, _hf.STATS
STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}

# 反应键名："{prefix}_{lhs} -> {ads}*" 或 "{prefix}_{lhs} -> {ads}*_{n}"
KEY_RE = re.compile(r"^([A-Za-z0-9]+)_(.*?)\s*->\s*([A-Za-z0-9]+)\*(_\d+)?$")
# 吸附原子（与表面成键的原子）：复合吸附种取键合原子
ADS_ATOM = {
    "H": "H", "2H": "H",
    "C": "C", "CH": "C", "CH2": "C", "CH3": "C",
    "N": "N", "2N": "N", "NH": "N",
    "O": "O", "2O": "O", "OH": "O", "H2O": "O",
    "S": "S", "SH": "S",
}

C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
C_3RD = "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def normalize_comp(cd):
    """元素按字母排序 + 计量数 gcd 约简（同 05）。"""
    from functools import reduce
    from math import gcd
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def parse_prefix(p):
    toks = re.findall(r"([A-Z][a-z]?)(\d+)", p)
    rebuilt = "".join(a + b for a, b in toks)
    if rebuilt != p:
        raise ValueError(f"[前缀解析异常] {p!r}")
    return {el: int(n) for el, n in toks}


def main():
    DP.mkdir(exist_ok=True)
    FIGDIR.mkdir(exist_ok=True)

    data = json.loads(gzip.decompress(RAW.read_bytes()))
    print(f"[读入] {RAW.name}: {len(data)} 个键（含 _structures 元数据）")

    # ---- 全量解析 ----
    rows, bad = [], []
    for k, rec in data.items():
        if k == "_structures":
            continue
        m = KEY_RE.match(k)
        if not m:
            bad.append(k)
            continue
        prefix, _, ads, _ = m.groups()
        cd = parse_prefix(prefix)
        structure, facet = STRUCT_RULE[tuple(sorted(cd.values()))]
        rows.append(dict(comp=normalize_comp(cd), structure=structure, facet=facet,
                         adsorbate=ads, energy_eV=float(rec["ref_ads_eng"])))
    assert not bad, f"[键名解析失败] {bad[:5]}"
    df = pd.DataFrame(rows)
    print(f"[解析] {len(df)} 条吸附反应，{df['adsorbate'].nunique()} 种吸附种标签，"
          f"{df['comp'].nunique()} 个唯一规范化组成")

    raw_counts = df["adsorbate"].value_counts().rename("n_raw_records")

    # ---- 去重：同 (comp, structure, adsorbate) 多构型取最低 E_ads（与 H* 线口径一致）----
    g = df.groupby(["comp", "structure", "adsorbate"])
    out = (g["energy_eV"].min().rename("energy_eV").reset_index()
           .merge(g.size().rename("n_raw").reset_index(),
                  on=["comp", "structure", "adsorbate"]))
    out["facet"] = out["structure"].map({"A1": 111, "L12": 111, "L10": 101})
    out["ads_atom"] = out["adsorbate"].map(ADS_ATOM)
    assert out["ads_atom"].notna().all(), "存在未映射的吸附种"
    out["n_ads_units"] = out["adsorbate"].str.startswith("2").astype(int) + 1
    out = out[["comp", "structure", "facet", "adsorbate", "ads_atom",
               "n_ads_units", "energy_eV", "n_raw"]]
    out = out.sort_values(["adsorbate", "structure", "comp"]).reset_index(drop=True)
    out.to_csv(DP / "multiads_dataset.csv", index=False)
    print(f"[写出] {DP/'multiads_dataset.csv'}  ({len(out)} 行, "
          f"去重前 {len(df)} → 去重后 {len(out)})")

    # ---- 各吸附种条数统计 ----
    dedup_counts = out["adsorbate"].value_counts().rename("n_dedup")
    surf_counts = (out.groupby("adsorbate")["comp"].nunique()
                   .rename("n_unique_comp"))
    counts = pd.concat([raw_counts, dedup_counts, surf_counts], axis=1)
    counts.index.name = "adsorbate"
    counts["ads_atom"] = counts.index.map(ADS_ATOM)
    counts = counts.sort_values("n_dedup", ascending=False)
    counts.to_csv(DP / "multiads_adsorbate_counts.csv")
    print(f"[写出] {DP/'multiads_adsorbate_counts.csv'}")
    print(counts.to_string())

    # ---- 图：各吸附种条数 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xpos = np.arange(len(counts))
    ax.bar(xpos - 0.2, counts["n_raw_records"], width=0.4, color=C_ALT,
           label="Raw records")
    ax.bar(xpos + 0.2, counts["n_dedup"], width=0.4, color=C_MAIN,
           label=r"After dedup (lowest $E_{ads}$)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "multiads_adsorbate_counts.png")
    plt.close(fig)
    print("[写出] figures/multiads_adsorbate_counts.png")

    # ================= 特征化 =================
    comp_dicts = {c: parse_comp(c) for c in out["comp"].unique()}
    elements = sorted({el for d in comp_dicts.values() for el in d})
    etab, fallback_used = build_element_table(elements)
    print(f"[comp] {len(comp_dicts)} 个唯一组成，{len(elements)} 种元素；"
          f"兜底 {'未启用' if not fallback_used else fallback_used}")

    # 吸附原子元素性质（mendeleev）：电负性/半径/族号
    ads_elements = sorted(out["ads_atom"].unique())
    ads_tab, ads_fb = build_element_table(ads_elements)
    print(f"[吸附原子性质] {ads_elements}；兜底 {'未启用' if not ads_fb else ads_fb}")

    feat_rows = []
    for _, row in out.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        # 注意：out 已含 n_ads_units 列（merge/concat 前不可重复携带，否则
        # 生成 n_ads_units/n_ads_units.1 重复列），此处不再并入 feats。
        feats = {"n_elements": len(els)}
        for p in PROPS:
            v = np.array([etab[e][p] for e in els])
            feats[f"{p}_wmean"] = wmean(v, w)
            feats[f"{p}_max"] = float(v.max())
            feats[f"{p}_min"] = float(v.min())
            feats[f"{p}_range"] = float(v.max() - v.min())
            feats[f"{p}_wstd"] = wstd(v, w)
        a = ads_tab[row["ads_atom"]]
        feats["ads_en"] = a["en"]          # 吸附原子 Pauling 电负性
        feats["ads_radius"] = a["radius"]  # 吸附原子半径 pm
        feats["ads_group"] = a["group"]    # 吸附原子族号
        feat_rows.append(feats)
    feats_df = pd.DataFrame(feat_rows, index=out.index)

    oh = pd.get_dummies(out[["structure", "facet", "ads_atom"]].astype(str),
                        dtype=int)
    feat_out = pd.concat([out.reset_index(drop=True),
                          feats_df.reset_index(drop=True),
                          oh.reset_index(drop=True)], axis=1)
    feat_out.to_csv(DP / "multiads_features.csv", index=False)
    print(f"[写出] {DP/'multiads_features.csv'}  ({len(feat_out)} 行 × "
          f"{feat_out.shape[1]} 列)")

    print(json.dumps({
        "n_raw_records": int(len(df)), "n_dedup": int(len(out)),
        "n_adsorbates": int(out["adsorbate"].nunique()),
        "n_unique_comp": int(out["comp"].nunique()),
        "counts_dedup": counts["n_dedup"].to_dict(),
        "energy_range": [float(out["energy_eV"].min()),
                         float(out["energy_eV"].max())],
        "fallback_used": fallback_used, "ads_fallback": ads_fb,
        "n_feature_cols": int(feat_out.shape[1]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
