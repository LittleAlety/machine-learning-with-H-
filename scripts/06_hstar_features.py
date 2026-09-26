# -*- coding: utf-8 -*-
"""
06_hstar_features.py — hstar_dataset_v1 → hstar_dataset_v2（H* 主线特征化）

复用 02 的元素描述符体系（Pauling 电负性、原子半径、族号、周期、第一电离能、d 电子数；
mendeleev 优先，缺失时 HAND_LOOKUP 兜底并记录）+ 化学计量加权统计 wmean/max/min/range/wstd。
结构描述符：structure one-hot（A1/L10/L12）、facet one-hot。
派生特征：site_spread（同表面不同吸附构型的能量极差，位点能量景观多样性，来自 05）。
输出：data_processed/hstar_dataset_v2.csv + notes/hstar_descriptor_table.md
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data_processed" / "hstar_dataset_v1.csv"
OUT_CSV = ROOT / "data_processed" / "hstar_dataset_v2.csv"
TABLE_MD = ROOT / "notes" / "hstar_descriptor_table.md"

# 手工兜底 lookup（同 02，另补 Tc；仅在 mendeleev 缺失时启用并记录）
HAND_LOOKUP = {
    "Tc": dict(en=2.10, radius=135.0, group=7, period=5, ie1=7.11938, d_el=5),
}

PROPS = ["en", "radius", "group", "period", "ie1", "d_el"]
STATS = ["wmean", "max", "min", "range", "wstd"]
PROP_CN = {
    "en": "Pauling 电负性（M–H 键极性与电荷转移趋势）",
    "radius": "原子半径 pm（表面晶格尺寸/应变与吸附位几何）",
    "group": "族号（仅对过渡金属组元是 d 带填充度的近似代理；对主族组元不适用，"
             "此时该特征仅代表周期表位置的粗粒度编码）",
    "period": "周期（原子尺寸与 d 轨道延展性）",
    "ie1": "第一电离能 eV（表面给电子能力，与 H 得电子成键相关）",
    "d_el": "价层 d 电子数（d 带填充度，HER 结合强度经典描述符维度）",
}
STAT_CN = {
    "wmean": "按化学计量分数加权平均（合金整体性质）",
    "max": "组元最大值",
    "min": "组元最小值",
    "range": "组元极差（合金性质错配度）",
    "wstd": "按化学计量分数加权标准差（组元性质离散度）",
}

COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
D_ELEC_RE = re.compile(r"(\d)d(\d*)")


def parse_comp(comp):
    """规范化 comp（元素已按字母排序，如 Pt3Ti、Ag）→ {元素: 计量数}。"""
    tokens = COMP_TOKEN.findall(comp)
    rebuilt = "".join(el + num for el, num in tokens)
    if rebuilt != comp or not tokens:
        raise ValueError(f"[comp 解析异常] {comp!r}，请人工检查")
    return {el: int(num) if num else 1 for el, num in tokens}


def d_electron_count(econf):
    tail = str(econf).split("]")[-1]
    return sum(int(n) if n else 1 for _, n in D_ELEC_RE.findall(tail))


def build_element_table(elements):
    from mendeleev import element as md_element
    table, fallback_used = {}, []
    for sym in elements:
        e = md_element(sym)
        vals = {
            "en": e.en_pauling, "radius": e.atomic_radius, "group": e.group_id,
            "period": e.period, "ie1": (e.ionenergies or {}).get(1),
            "d_el": d_electron_count(e.econf) if str(e.econf) not in ("None", "") else None,
        }
        for k in PROPS:
            if vals[k] is None:
                vals[k] = HAND_LOOKUP[sym][k]
                fallback_used.append(f"{sym}.{k}")
        table[sym] = {k: float(vals[k]) for k in PROPS}
    return table, fallback_used


def wmean(v, w):
    return float(np.average(v, weights=w))


def wstd(v, w):
    mu = np.average(v, weights=w)
    return float(np.sqrt(np.average((np.asarray(v) - mu) ** 2, weights=w)))


def main():
    df = pd.read_csv(IN_CSV)
    print(f"[读入] {IN_CSV}  ({len(df)} 行)")

    comp_dicts = {c: parse_comp(c) for c in df["comp"].unique()}
    elements = sorted({el for d in comp_dicts.values() for el in d})
    print(f"[comp] {len(comp_dicts)} 个唯一组成，{len(elements)} 种元素，解析全部通过")

    etab, fallback_used = build_element_table(elements)
    print(f"[兜底] {'未启用' if not fallback_used else fallback_used}")

    feat_rows = []
    for _, row in df.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        feats = {"n_elements": len(els)}  # site_spread 已在 v1 中，直接作为特征列
        for p in PROPS:
            v = np.array([etab[e][p] for e in els])
            feats[f"{p}_wmean"] = wmean(v, w)
            feats[f"{p}_max"] = float(v.max())
            feats[f"{p}_min"] = float(v.min())
            feats[f"{p}_range"] = float(v.max() - v.min())
            feats[f"{p}_wstd"] = wstd(v, w)
        feat_rows.append(feats)
    feats_df = pd.DataFrame(feat_rows, index=df.index)

    oh = pd.get_dummies(df[["structure", "facet"]].astype(str), dtype=int)
    out = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True),
                     oh.reset_index(drop=True)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    print(f"[写出] {OUT_CSV}  ({len(out)} 行, {out.shape[1]} 列)")

    num_feats = [f"{p}_{s}" for p in PROPS for s in STATS]
    lines = [
        "# 特征描述符表 — hstar_dataset_v2（HER 主线）",
        "",
        f"输入：hstar_dataset_v1.csv（{len(df)} 行）；输出：hstar_dataset_v2.csv"
        f"（{len(out)} 行 × {out.shape[1]} 列）",
        "",
        "## 标识列（非特征）",
        "",
        "| 列 | 含义 |",
        "|---|---|",
        "| comp | 规范化组成（元素字母序+计量约简，如 Pt3Ti） |",
        "| structure | 晶体结构类型（A1 纯金属 / L1₂ 3:1 / L1₀ 1:1） |",
        "| facet | 晶面（A1/L1₂=(111)，L1₀=(101)） |",
        "| energy_eV | 目标值：最稳定位点 H* 吸附能 E_ads（eV，负值=放热） |",
        "| n_raw | 该表面的吸附构型数（去重前） |",
        "| site_spread | ⚠ 分析列（泄漏特征，禁止入模）：同表面各构型能量极差，仅用于分析 |",
        "",
        "## 数值特征",
        "",
        "| 特征 | 化学含义 |",
        "|---|---|",
        "| n_elements | 组成元素个数（1=纯金属，2=双金属） |",
    ]
    for p in PROPS:
        for s in STATS:
            lines.append(f"| {p}_{s} | {PROP_CN[p]}；{STAT_CN[s]} |")
    lines += [
        "",
        "注：`site_spread`（v1/v2 中的分析列，同表面各构型能量极差）为**目标泄漏特征**"
        "（含目标值本身，对新表面不可得），已从建模特征集排除（07/08 的 ID 列处理），"
        "仅用于位点能量景观分析；v2 中保留该列仅为溯源。",
        "",
        "## 结构特征（one-hot）",
        "",
        "| 特征 | 化学含义 |",
        "|---|---|",
        "| structure_A1 | 纯金属 fcc (111) 表面 |",
        "| structure_L10 | L1₀ 有序双金属 (101) 面（层状交替，1:1） |",
        "| structure_L12 | L1₂ 有序双金属 (111) 面（3:1，Cu₃Au 原型） |",
        "| facet_101 / facet_111 | 晶面（表面原子配位环境） |",
        "",
        "⚠ 恒等特征对：`structure_L10 ≡ facet_101`、`structure_A1 + structure_L12 ≡ facet_111`"
        "（结构类型与晶面一一绑定，one-hot 后完全共线）。树模型对此不敏感故保留，"
        "但解释 SHAP 时应注意这两组特征的重要性不可分割。",
        "",
        "## 元素属性来源",
        "",
        "- mendeleev（en_pauling / atomic_radius / group_id / period / ionenergies[1] / 电子组态）",
        "- d_el 由电子组态解析，无角标按 1 计（Sc/Y/La 的 nd¹）",
        f"- 手工 lookup 兜底：{'**未启用**（全部属性 mendeleev 齐全，含 Tc）' if not fallback_used else ', '.join(fallback_used)}",
        "",
        f"模型输入特征共 {1 + len(num_feats) + oh.shape[1]} 个"
        f"（n_elements + {len(num_feats)} 组成统计 + {oh.shape[1]} one-hot；site_spread 已排除）。",
    ]
    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[写出] {TABLE_MD}")
    print(json.dumps({"fallback_used": fallback_used, "n_rows": len(out),
                      "n_cols": int(out.shape[1])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
