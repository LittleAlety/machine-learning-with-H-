# -*- coding: utf-8 -*-
"""
02_features.py — dataset_v1 → dataset_v2（组成/结构描述符）

- comp 解析：元素 + 下标计量数（"Pt3Co" → {Pt:3, Co:1}；"NiAl" → {Ni:1, Al:1}）；
  解析重建字符串与原式不符时打印并抛错（不静默丢弃）
- 元素描述符（mendeleev 优先，缺失时 HAND_LOOKUP 兜底并记录）：
  en_pauling 电负性 en（mendeleev 值，与 Pauling 原值略有出入）、
  原子半径 radius(pm, mendeleev atomic_radius 经验半径)、族号 group、周期 period、
  第一电离能 ie1(eV)、价层 d 电子数 d_el（由电子组态解析）
- 组合统计（按化学计量分数加权）：wmean / max / min / range / wstd
- 结构描述符：facet one-hot、term one-hot（AA/AB/BB/none）、adsorbate one-hot
- 输出 notes/descriptor_table.md：每个特征一句化学含义 + 兜底记录
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data_processed" / "dataset_v1.csv"
OUT_CSV = ROOT / "data_processed" / "dataset_v2.csv"
TABLE_MD = ROOT / "notes" / "descriptor_table.md"

# 手工兜底 lookup（值取自 mendeleev 1.2.0 / 标准物理化学手册；radius 单位 pm，ie1 单位 eV）
# 仅在 mendeleev 对应属性缺失时启用，启用情况打印并写入 descriptor_table.md
HAND_LOOKUP = {
    "Ag": dict(en=1.93, radius=160.0, group=11, period=5, ie1=7.57623, d_el=10),
    "Al": dict(en=1.61, radius=125.0, group=13, period=3, ie1=5.98577, d_el=0),
    "As": dict(en=2.18, radius=115.0, group=15, period=4, ie1=9.78855, d_el=10),
    "Au": dict(en=2.40, radius=135.0, group=11, period=6, ie1=9.22555, d_el=10),
    "Bi": dict(en=1.90, radius=160.0, group=15, period=6, ie1=7.28552, d_el=10),
    "Cd": dict(en=1.69, radius=155.0, group=12, period=5, ie1=8.99382, d_el=10),
    "Co": dict(en=1.88, radius=135.0, group=9,  period=4, ie1=7.88101, d_el=7),
    "Cr": dict(en=1.66, radius=140.0, group=6,  period=4, ie1=6.76651, d_el=5),
    "Cu": dict(en=1.90, radius=135.0, group=11, period=4, ie1=7.72638, d_el=10),
    "Fe": dict(en=1.83, radius=140.0, group=8,  period=4, ie1=7.90247, d_el=6),
    "Ga": dict(en=1.81, radius=130.0, group=13, period=4, ie1=5.99930, d_el=10),
    "Ge": dict(en=2.01, radius=125.0, group=14, period=4, ie1=7.89944, d_el=10),
    "Hf": dict(en=1.30, radius=155.0, group=4,  period=6, ie1=6.82507, d_el=2),
    "Hg": dict(en=1.90, radius=150.0, group=12, period=6, ie1=10.43750, d_el=10),
    "In": dict(en=1.78, radius=155.0, group=13, period=5, ie1=5.78636, d_el=10),
    "Ir": dict(en=2.20, radius=135.0, group=9,  period=6, ie1=8.96702, d_el=7),
    "La": dict(en=1.10, radius=195.0, group=3,  period=6, ie1=5.57690, d_el=1),
    "Mn": dict(en=1.55, radius=140.0, group=7,  period=4, ie1=7.43404, d_el=5),
    "Mo": dict(en=2.16, radius=145.0, group=6,  period=5, ie1=7.09243, d_el=5),
    "Nb": dict(en=1.60, radius=145.0, group=5,  period=5, ie1=6.75885, d_el=4),
    "Ni": dict(en=1.91, radius=135.0, group=10, period=4, ie1=7.63988, d_el=8),
    "Os": dict(en=2.20, radius=130.0, group=8,  period=6, ie1=8.43823, d_el=6),
    "Pb": dict(en=1.80, radius=180.0, group=14, period=6, ie1=7.41668, d_el=10),
    "Pd": dict(en=2.20, radius=140.0, group=10, period=5, ie1=8.33684, d_el=10),
    "Pt": dict(en=2.20, radius=135.0, group=10, period=6, ie1=8.95883, d_el=9),
    "Re": dict(en=1.90, radius=135.0, group=7,  period=6, ie1=7.83352, d_el=5),
    "Rh": dict(en=2.28, radius=135.0, group=9,  period=5, ie1=7.45890, d_el=8),
    "Ru": dict(en=2.20, radius=130.0, group=8,  period=5, ie1=7.36050, d_el=7),
    "Sb": dict(en=2.05, radius=145.0, group=15, period=5, ie1=8.60839, d_el=10),
    "Sc": dict(en=1.36, radius=160.0, group=3,  period=4, ie1=6.56149, d_el=1),
    "Si": dict(en=1.90, radius=110.0, group=14, period=3, ie1=8.15168, d_el=0),
    "Sn": dict(en=1.96, radius=145.0, group=14, period=5, ie1=7.34392, d_el=10),
    "Ta": dict(en=1.50, radius=145.0, group=5,  period=6, ie1=7.54957, d_el=3),
    "Ti": dict(en=1.54, radius=140.0, group=4,  period=4, ie1=6.82812, d_el=2),
    "Tl": dict(en=1.80, radius=190.0, group=13, period=6, ie1=6.10829, d_el=10),
    "V":  dict(en=1.63, radius=135.0, group=5,  period=4, ie1=6.74619, d_el=3),
    "W":  dict(en=1.70, radius=135.0, group=6,  period=6, ie1=7.86403, d_el=4),
    "Y":  dict(en=1.22, radius=180.0, group=3,  period=5, ie1=6.21726, d_el=1),
    "Zn": dict(en=1.65, radius=135.0, group=12, period=4, ie1=9.39420, d_el=10),
    "Zr": dict(en=1.33, radius=155.0, group=4,  period=5, ie1=6.63413, d_el=2),
}

PROPS = ["en", "radius", "group", "period", "ie1", "d_el"]
STATS = ["wmean", "max", "min", "range", "wstd"]
PROP_CN = {
    "en": "mendeleev en_pauling 电负性（数值与 Pauling 原值有出入，如 Pt 2.20 vs 2.28；吸附键极性与电荷转移趋势）",
    "radius": "原子半径 pm（mendeleev `atomic_radius` 属性，经验原子半径——该库元数据仅标注 'Atomic radius'，"
              "未区分共价/金属半径定义，与同库 covalent_radius_cordero / metallic_radius 数值不同，"
              "如 Pt 分别为 135 / 136 / 130 pm；表面晶格尺寸/应变与吸附位几何）",
    "group": "元素周期表族号（价电子结构大类，过渡金属 d 带位置代理）",
    "period": "元素周期（原子尺寸与 d 轨道延展性）",
    "ie1": "第一电离能 eV（表面向吸附物种给电子能力）",
    "d_el": "价层 d 电子数（d 带填充度，吸附强度的经典描述符维度）",
}
STAT_CN = {
    "wmean": "按化学计量分数加权平均（合金整体性质）",
    "max": "组元最大值（最极端组元的贡献）",
    "min": "组元最小值",
    "range": "组元极差（max-min，合金性质差异/错配度）",
    "wstd": "按化学计量分数加权标准差（组元性质离散度）",
}

COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
D_ELEC_RE = re.compile(r"(\d)d(\d*)")


def parse_comp(comp: str) -> dict:
    """comp → {元素: 计量数}；格式异常时打印并抛错（人工处理，不静默丢弃）。"""
    tokens = COMP_TOKEN.findall(comp)
    rebuilt = "".join(el + num for el, num in tokens)
    if rebuilt != comp or not tokens:
        raise ValueError(f"[comp 解析异常] {comp!r} → {tokens!r}（重建 {rebuilt!r} 不一致），请人工检查")
    return {el: int(num) if num else 1 for el, num in tokens}


def d_electron_count(econf: str) -> int:
    """从电子组态（如 '[Ar] 3d7 4s2'）统计稀有气体核心以外的 d 电子数。
    无角标时（如 '5d 6s2'）按 1 计。"""
    tail = str(econf).split("]")[-1]
    return sum(int(n) if n else 1 for _, n in D_ELEC_RE.findall(tail))


def build_element_table(elements):
    """mendeleev 优先；缺失属性用 HAND_LOOKUP 兜底并记录。"""
    from mendeleev import element as md_element
    table, fallback_used = {}, []
    for sym in elements:
        e = md_element(sym)
        vals = {
            "en": e.en_pauling,
            "radius": e.atomic_radius,
            "group": e.group_id,
            "period": e.period,
            "ie1": (e.ionenergies or {}).get(1),
            "d_el": d_electron_count(e.econf) if str(e.econf) not in ("None", "") else None,
        }
        for k in PROPS:
            if vals[k] is None:
                vals[k] = HAND_LOOKUP[sym][k]
                fallback_used.append(f"{sym}.{k}")
        table[sym] = {k: float(vals[k]) for k in PROPS}
    return table, fallback_used


def wmean(vals, w):
    return float(np.average(vals, weights=w))


def wstd(vals, w):
    mu = np.average(vals, weights=w)
    return float(np.sqrt(np.average((np.asarray(vals) - mu) ** 2, weights=w)))


def main():
    df = pd.read_csv(IN_CSV)
    print(f"[读入] {IN_CSV}  ({len(df)} 行)")

    # ---- comp 解析（异常立即报错）----
    comp_dicts = {}
    for c in df["comp"].unique():
        comp_dicts[c] = parse_comp(c)
    elements = sorted({el for d in comp_dicts.values() for el in d})
    print(f"[comp] {len(comp_dicts)} 个唯一组成，涉及 {len(elements)} 种元素，解析全部通过")

    etab, fallback_used = build_element_table(elements)
    if fallback_used:
        print(f"[兜底] mendeleev 缺失，使用手工 lookup：{fallback_used}")
    else:
        print("[兜底] mendeleev 属性齐全，未使用手工 lookup")

    # ---- 组合统计特征 ----
    feat_rows = []
    for _, row in df.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w = w / w.sum()
        feats = {"n_elements": len(els)}
        for p in PROPS:
            v = np.array([etab[e][p] for e in els], dtype=float)
            feats[f"{p}_wmean"] = wmean(v, w)
            feats[f"{p}_max"] = float(v.max())
            feats[f"{p}_min"] = float(v.min())
            feats[f"{p}_range"] = float(v.max() - v.min())
            feats[f"{p}_wstd"] = wstd(v, w)
        feat_rows.append(feats)
    feats_df = pd.DataFrame(feat_rows, index=df.index)

    # ---- one-hot：facet / term / adsorbate ----
    oh = pd.get_dummies(df[["facet", "term", "adsorbate"]].astype(str),
                        columns=["facet", "term", "adsorbate"], dtype=int)
    oh.columns = [c.replace("term_none", "term_none").replace("adsorbate_", "ads_") for c in oh.columns]

    out = pd.concat([df.reset_index(drop=True),
                     feats_df.reset_index(drop=True),
                     oh.reset_index(drop=True)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    print(f"[写出] {OUT_CSV}  ({len(out)} 行, {out.shape[1]} 列)")

    # ---- descriptor_table.md ----
    num_feats = [f"{p}_{s}" for p in PROPS for s in STATS]
    onehot_feats = list(oh.columns)
    lines = [
        "# 特征描述符表 — dataset_v2",
        "",
        f"输入：dataset_v1.csv（{len(df)} 行）；输出：dataset_v2.csv（{len(out)} 行 × {out.shape[1]} 列）",
        "",
        "## 标识列（非特征）",
        "",
        "| 列 | 含义 |",
        "|---|---|",
        "| comp | 表面组成化学式（数字为下标计量数，如 Pt3Co） |",
        "| facet | 晶面米勒指数（111/211 等） |",
        "| term | L1_2 合金表面终止层标签（AA/AB/BB；none=无标签） |",
        "| adsorbate | 吸附物种（CO / O） |",
        "| energy_eV | 目标值：CatApp reaction_energy（eV，**数值越大=结合越强**，≈ −E_ads+常数） |",
        "| dataset_src | 数据来源子库 |",
        "| n_raw | 去重前原始条数 |",
        "",
        "## 数值特征（组成统计）",
        "",
        "6 种元素性质 × 5 种组合统计 = 30 个特征；权重 = 化学计量分数。",
        "",
        "### 元素性质来源",
        "",
        "- 主源：mendeleev（en_pauling / atomic_radius / group_id / period / ionenergies[1] / 电子组态）",
        "- d_el：由电子组态解析稀有气体核心以外的 d 电子数（无角标按 1 计，如 Sc/Y/La 的 nd¹）",
        f"- 手工 lookup 兜底启用情况：{'**未启用**（全部属性 mendeleev 齐全）' if not fallback_used else '启用 → ' + ', '.join(fallback_used)}",
        "",
        "### 特征清单",
        "",
        "| 特征 | 化学含义 |",
        "|---|---|",
        "| n_elements | 组成元素个数（1=纯金属，2+=合金） |",
    ]
    for p in PROPS:
        for s in STATS:
            lines.append(f"| {p}_{s} | {PROP_CN[p]}；{STAT_CN[s]} |")
    lines += [
        "",
        "## 结构/类别特征（one-hot）",
        "",
        "| 特征 | 化学含义 |",
        "|---|---|",
    ]
    oh_cn = {
        "facet": "晶面（表面原子排列/配位数，(211) 为台阶面更活泼）",
        "term": "表面终止层（L1_2 合金表层为纯 A、混合 AB 或纯 B，决定吸附位点局域组成）",
        "ads": "吸附物种（CO 或 O，化学亲和性不同）",
    }
    for c in onehot_feats:
        prefix = c.split("_")[0]
        lines.append(f"| {c} | {oh_cn[prefix]}：= {c.split('_', 1)[1]} |")
    lines += [
        "",
        f"数值特征 {len(num_feats)} 个 + one-hot {len(onehot_feats)} 个 + n_elements = "
        f"{len(num_feats) + len(onehot_feats) + 1} 个模型输入特征。",
        "",
        "## 已知共线性局限（facet / term × adsorbate）",
        "",
    ]
    ct = pd.crosstab(df["facet"], df["adsorbate"])
    n_co111 = int(ct.loc[111, "CO"]) if (111 in ct.index and "CO" in ct.columns) else 0
    n_co211 = int(ct.loc[211, "CO"]) if (211 in ct.index and "CO" in ct.columns) else 0
    n_o111 = int(ct.loc[111, "O"]) if (111 in ct.index and "O" in ct.columns) else 0
    n_o211 = int(ct.loc[211, "O"]) if (211 in ct.index and "O" in ct.columns) else 0
    ct_t = pd.crosstab(df["term"], df["adsorbate"])
    n_co_term_none = int(ct_t.loc["none", "CO"]) if ("none" in ct_t.index and "CO" in ct_t.columns) else 0
    n_o_term_none = int(ct_t.loc["none", "O"]) if ("none" in ct_t.index and "O" in ct_t.columns) else 0
    lines += [
        f"结构 one-hot 与吸附种高度共线：",
        f"- facet：CO 共 {n_co111 + n_co211} 行（(111) {n_co111} / (211) {n_co211}，后者全部为纯金属）；"
        f"O 共 {n_o111 + n_o211} 行，全部位于 (211)。",
        f"- term：CO 共 {n_co_term_none} 行全部 term=none；O 的 term=none 行为 {n_o_term_none}"
        "（其余全部带 AA/AB/BB 标签）。",
        "因此 `term_none` 与 `ads_CO` 完全共线、`facet_111` 近乎共线，模型/SHAP 中 term_none / facet_111 "
        "的重要性**不能解读为纯终止层/晶面效应**（混杂吸附种效应）。"
        "若要晶面/终止层结论需补 CO(211)（尤其合金）/O(111) 数据。",
    ]
    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[写出] {TABLE_MD}")
    print(json.dumps({"fallback_used": fallback_used, "n_rows": len(out),
                      "n_cols": int(out.shape[1])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
