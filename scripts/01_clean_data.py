# -*- coding: utf-8 -*-
"""
01_clean_data.py — CMR CatApp 原始数据清洗 → data_processed/dataset_v1.csv

清洗步骤（每步删除行数写入 notes/01_cleaning_report.md）：
  1. 只保留 ab ∈ {CO*, O*}
  2. 反应身份过滤（标签污染修复）：CatApp 原始数据中部分 ab=CO* 的行
     实际 a/b 列为 C*/O*，即 C*+O*→CO* 基元反应能而非 CO 吸附能。
     规则（先对 a/b 做 value_counts 核实取值后确定）：
       ab=CO* 只保留 a=CO 且 b=*；ab=O* 只保留 a=hfO2 且 b=*
  3. 剔除 dataset 含 "BEP" 的行（经验 BEP 关系估算值，非直接 DFT）
  4. 解析 surface 列：^(comp)\\((facet 3 位晶面)\\)( (term AA|AB|BB))?$
     尾部 AA/AB/BB 为 L1_2 合金表面终止层标签，保留为独立列 term；
     无法解析的非标准条目（MoS2 类、ZnO(0001) 等）剔除并逐条记录
  5. 按 (comp, facet, term, adsorbate) 去重，能量取中位数，记录 n_raw；
     组内极差 >0.5 eV 的"冲突组"降级为辅助检查：只在报告中统计与列明，
     不再整组剔除（标签污染修复后纯金属 (211) 恢复为 JPCC 单值，冲突组消失）
  6. IQR 法则（1.5×IQR，分吸附种）剔除能量异常值

输出：data_processed/dataset_v1.csv（comp, facet, term, adsorbate, energy_eV, dataset_src, n_raw）
      notes/01_cleaning_report.md
      figures/eda_energy_hist.png / eda_top_elements.png / eda_facet_counts.png
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "catappdata.csv"
OUT_CSV = ROOT / "data_processed" / "dataset_v1.csv"
REPORT = ROOT / "notes" / "01_cleaning_report.md"
FIGDIR = ROOT / "figures"

RNG_SEED = 42
# 低饱和暖色系（无蓝紫）
C_CO = "#D98E5F"   # 陶土橙
C_O = "#A85838"    # 赭石红
C_BAR = "#C98A5A"  # 暖棕
C_BAR2 = "#DFB27E"  # 浅杏
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SURFACE_RE = re.compile(r"^(?P<comp>.+?)\((?P<facet>-?\d{3})\)(?: (?P<term>AA|AB|BB))?$")


def main():
    FIGDIR.mkdir(exist_ok=True)
    (ROOT / "notes").mkdir(exist_ok=True)
    (ROOT / "data_processed").mkdir(exist_ok=True)

    log = []  # (步骤, 删除/保留行数说明)

    df = pd.read_csv(RAW)
    n0 = len(df)
    log.append(("原始数据", n0, f"读入 {RAW.name}，共 {n0} 行"))

    # ---- 步骤 1：只保留 CO* / O* ----
    mask_ab = df["ab"].isin(["CO*", "O*"])
    df = df[mask_ab].copy()
    log.append(("步骤1 吸附种过滤", n0 - len(df),
                f"只保留 ab ∈ {{CO*, O*}}；剔除其余吸附种 {n0 - len(df)} 行，剩 {len(df)} 行"))

    # ---- 步骤 2：反应身份过滤（标签污染修复）----
    # 先核实 a/b 实际取值，再按反应身份过滤：
    #   ab=CO* 只保留 a=CO 且 b=*（CO 分子吸附）；
    #   ab=O*  只保留 a=hfO2 且 b=*（O 原子吸附，hfO2 为 CatApp 的 O2 气相参考标签）
    print("[a/b 取值核实] ab=CO*:")
    print(df.loc[df["ab"] == "CO*", ["a", "b"]].value_counts().to_string())
    print("[a/b 取值核实] ab=O*:")
    print(df.loc[df["ab"] == "O*", ["a", "b"]].value_counts().to_string())
    mask_id = (((df["ab"] == "CO*") & (df["a"] == "CO") & (df["b"] == "*")) |
               ((df["ab"] == "O*") & (df["a"] == "hfO2") & (df["b"] == "*")))
    id_removed = df[~mask_id].copy()
    n_id = len(id_removed)
    # 被剔除行明细（surface、能量、反应物 a/b、文献），供报告逐条列出
    id_removed_rows = [
        (r["surface"], r["ab"], r["a"], r["b"], float(r["reaction_energy"]), r["reference"])
        for _, r in id_removed.sort_values(["surface"]).iterrows()
    ]
    id_removed_refs = id_removed["reference"].value_counts().to_dict()
    df = df[mask_id].copy()
    log.append(("步骤2 反应身份过滤", n_id,
                f"按反应身份过滤（CO* 只留 a=CO,b=*；O* 只留 a=hfO2,b=*）：剔除 {n_id} 行 ab=CO* 但 "
                f"a/b=C*/O* 的行（实为 C*+O*→CO* 基元反应能而非 CO 吸附能，明细见下表），剩 {len(df)} 行"))

    # ---- 步骤 3：剔除 BEP 经验估算行 ----
    mask_bep = df["dataset"].str.contains("BEP", na=False)
    n_bep = int(mask_bep.sum())
    df = df[~mask_bep].copy()
    log.append(("步骤3 BEP 过滤", n_bep,
                f"dataset 含 'BEP' 的行为 BEP 经验关系估算值而非直接 DFT，剔除 {n_bep} 行，剩 {len(df)} 行"))

    # ---- 步骤 4：解析 surface 列 ----
    m = df["surface"].astype(str).str.extract(SURFACE_RE)
    bad = df[m["comp"].isna()]
    n_bad = len(bad)
    bad_list = sorted(bad["surface"].astype(str).unique().tolist())
    n_bad_mos2 = int(bad["surface"].astype(str).str.contains(r"MoS2|MoSe2").sum())
    n_bad_other = n_bad - n_bad_mos2
    df = df[m["comp"].notna()].copy()
    df["comp"] = m["comp"]
    df["facet"] = m["facet"]
    df["term"] = m["term"].fillna("none")  # 无标签（纯金属/常规合金表面）记为 none
    n_term_rows = int((df["term"] != "none").sum())  # 带终止层标签的原始行数
    log.append(("步骤4 surface 解析", n_bad,
                f"正则 `{SURFACE_RE.pattern}`；无法解析的非标准条目剔除 {n_bad} 行"
                f"（MoS2/MoSe2 氢化处理条目 {n_bad_mos2} 行 + 其他 {n_bad_other} 行，逐条见下），剩 {len(df)} 行"))

    # ---- 统一输出列 ----
    df["adsorbate"] = df["ab"].map({"CO*": "CO", "O*": "O"})
    df["energy_eV"] = df["reaction_energy"].astype(float)
    df["dataset_src"] = df["dataset"]

    # ---- 步骤 5：去重（同 comp/facet/term/adsorbate）----
    # 规则：组内能量取中位数合并，记录 n_raw。
    # 旧规则"组内极差 >0.5 eV 整组剔除"降级为辅助检查：标签污染修复（步骤 2）后，
    # 纯金属 (211) CO* 恢复为 2009 JPCC 单一文献值，冲突组应消失；此处只统计并列明，不再剔除。
    key = ["comp", "facet", "term", "adsorbate"]
    CONFLICT_TOL = 0.5  # eV（辅助检查阈值）
    grp = df.groupby(key)
    grp_range = grp["energy_eV"].agg(lambda s: s.max() - s.min())
    conflict_keys = set(grp_range[grp_range > CONFLICT_TOL].index)
    n_conflict_groups = len(conflict_keys)
    n_conflict_rows = int(grp.size()[grp_range > CONFLICT_TOL].sum()) if conflict_keys else 0
    # 冲突组明细（surface、能量、文献），供报告逐组列出（正常应为空）
    conflict_rows = []
    for k in sorted(conflict_keys):
        sub = grp.get_group(k)
        for _, r in sub.iterrows():
            conflict_rows.append((r["surface"], r["ab"], r["energy_eV"], r["reference"]))
    n_merge_before = len(df)
    df = (df.groupby(key, as_index=False)
            .agg(energy_eV=("energy_eV", "median"),
                 dataset_src=("dataset_src", lambda s: s.iloc[0]),
                 n_raw=("energy_eV", "size")))
    n_merged = n_merge_before - len(df)
    log.append(("步骤5 分组去重", n_merged,
                f"按 (comp, facet, term, adsorbate) 分组取中位数（合并 {n_merged} 行），剩 {len(df)} 行；"
                f"辅助检查：组内极差 > {CONFLICT_TOL} eV 的冲突组 {n_conflict_groups} 组"
                f"（{n_conflict_rows} 行，仅报告不剔除——标签污染修复后预期为 0）"))

    # ---- 步骤 6：IQR 异常值（分吸附种）----
    keep_mask = pd.Series(True, index=df.index)
    iqr_info = {}
    for ads in ["CO", "O"]:
        e = df.loc[df["adsorbate"] == ads, "energy_eV"]
        q1, q3 = e.quantile(0.25), e.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = (df["adsorbate"] == ads) & ((df["energy_eV"] < lo) | (df["energy_eV"] > hi))
        iqr_info[ads] = (lo, hi, int(out.sum()))
        keep_mask &= ~out
    n_out = int((~keep_mask).sum())
    df = df[keep_mask].reset_index(drop=True)
    detail = "；".join(f"{a}: 窗口 [{iqr_info[a][0]:.3f}, {iqr_info[a][1]:.3f}] eV，剔除 {iqr_info[a][2]} 行"
                       for a in ["CO", "O"])
    log.append(("步骤6 IQR 异常值", n_out,
                f"1.5×IQR 法则，分吸附种计算。{detail}，剩 {len(df)} 行"))

    # ---- 写出 dataset_v1 ----
    df = df[["comp", "facet", "term", "adsorbate", "energy_eV", "dataset_src", "n_raw"]]
    df.to_csv(OUT_CSV, index=False)
    print(f"[写出] {OUT_CSV}  ({len(df)} 行)")

    # ---- 汇总统计 ----
    n_co = int((df["adsorbate"] == "CO").sum())
    n_o = int((df["adsorbate"] == "O").sum())
    n_unique_comp = df["comp"].nunique()
    n_pure = int((~df["comp"].str.contains(r"\d", regex=True) &
                  ~df["comp"].str.match(r"^[A-Z][a-z]?[A-Z]")).sum())  # 单元素
    # 更稳妥：用元素个数判断
    n_elem = df["comp"].apply(lambda c: len(re.findall(r"[A-Z][a-z]?", c)))
    n_pure = int((n_elem == 1).sum())
    n_alloy = int((n_elem >= 2).sum())
    erng = df.groupby("adsorbate")["energy_eV"].agg(["min", "max"])

    # ---- EDA 图 ----
    # 1) 能量直方图（分吸附种）
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(df["energy_eV"].min(), df["energy_eV"].max(), 40)
    ax.hist(df.loc[df["adsorbate"] == "CO", "energy_eV"], bins=bins, alpha=0.75,
            color=C_CO, label=f"CO (n={n_co})", edgecolor="white", linewidth=0.4)
    ax.hist(df.loc[df["adsorbate"] == "O", "energy_eV"], bins=bins, alpha=0.75,
            color=C_O, label=f"O (n={n_o})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Adsorption energy (eV, larger = stronger binding)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_energy_hist.png")
    plt.close(fig)

    # 2) 元素出现频次 Top 15
    from collections import Counter
    cnt = Counter()
    for c in df["comp"]:
        for el in set(re.findall(r"[A-Z][a-z]?", c)):
            cnt[el] += 1
    top = cnt.most_common(15)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([t[0] for t in top], [t[1] for t in top], color=C_BAR, edgecolor="white")
    ax.set_xlabel("Element")
    ax.set_ylabel("Occurrences (surfaces)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_top_elements.png")
    plt.close(fig)

    # 3) 晶面分布（分吸附种堆叠）
    fc = df.groupby(["facet", "adsorbate"]).size().unstack(fill_value=0)
    fc = fc.reindex(sorted(fc.index))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bottom = np.zeros(len(fc))
    for ads, col in [("CO", C_CO), ("O", C_O)]:
        vals = fc[ads].values if ads in fc.columns else np.zeros(len(fc))
        ax.bar(fc.index.astype(str), vals, bottom=bottom, color=col, label=ads,
               edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_xlabel("Facet")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_facet_counts.png")
    plt.close(fig)
    print("[写出] 3 张 EDA 图 → figures/")

    # 标签污染修复后恢复的纯金属 (211) CO* 值（动态从数据读取，供报告引用）
    def _pure211(sym):
        sel = df[(df["comp"] == sym) & (df["facet"] == "211") & (df["adsorbate"] == "CO")]
        return float(sel["energy_eV"].iloc[0]) if len(sel) else float("nan")
    pt211, pd211 = _pure211("Pt"), _pure211("Pd")

    # ---- 清洗报告 ----
    lines = [
        "# 01 数据清洗报告 — CMR CatApp → dataset_v1",
        "",
        "## 清洗流水（每步删除行数与原因）",
        "",
        "| 步骤 | 删除行数 | 说明 |",
        "|---|---|---|",
    ]
    for name, ndel, desc in log:
        lines.append(f"| {name} | {ndel} | {desc} |")
    lines += [
        "",
        "## 无法解析的 surface 条目（已剔除，共 %d 行，去重后 %d 种）" % (n_bad, len(bad_list)),
        "",
    ]
    lines += [f"- `{s}`" for s in bad_list]
    lines += [
        "",
        "## 终止层标签 term 的处理",
        "",
        "- 尾部 `AA/AB/BB` 是 L1_2 合金表面终止层（termination）标签，原始数据中共 %d 行，"
        "是合金数据主体，解析为独立列 `term` 保留（无标签记为 `none`），Stage 2 中 one-hot 编码为特征。"
        % n_term_rows,
        "",
        "## dataset_v1.csv 汇总",
        "",
        f"- 总行数：**{len(df)}**（CO: {n_co}，O: {n_o}）",
        f"- 唯一组成 comp 数：{n_unique_comp}",
        f"- 纯金属（单元素）：{n_pure} 行；合金（≥2 元素）：{n_alloy} 行",
        f"- 晶面分布：{df['facet'].value_counts().to_dict()}",
        f"- term 分布：{df['term'].value_counts().to_dict()}",
        f"- 能量范围：CO ∈ [{erng.loc['CO','min']:.3f}, {erng.loc['CO','max']:.3f}] eV；"
        f"O ∈ [{erng.loc['O','min']:.3f}, {erng.loc['O','max']:.3f}] eV",
        "- 能量语义：CatApp `reaction_energy`，**数值越大 = 结合越强**（≈ −E_ads + 常数，"
        "源于气相参考态校正方案；经验检验见 notes/04_screening.md）",
        "",
        "## 反应身份过滤剔除明细（步骤 2，标签污染修复，共 %d 行）" % n_id,
        "",
        "CatApp 原始数据中以下行的 `ab` 列为 CO*，但反应物 `a`/`b` 列为 C*/O*，"
        "即 C*+O*→CO* 基元反应能（数值与 CO 吸附能不可比），属标签污染，整批剔除：",
        "",
        "| surface | ab | a | b | energy_eV | reference |",
        "|---|---|---|---|---|---|",
    ]
    for s, ab, a, b, e, ref in id_removed_rows:
        lines.append(f"| {s} | {ab} | {a} | {b} | {e:.3f} | {ref} |")
    lines += [
        "",
        "剔除行的文献来源统计："
        + "；".join(f"{ref} × {cnt}" for ref, cnt in id_removed_refs.items()),
        "",
        f"修复效果：纯金属 (211) 面 CO* 恢复为 2009 JPCC 单一文献值，"
        f"如 Pt(211) = {pt211:.3f} eV、Pd(211) = {pd211:.3f} eV；"
        "此前版本中这些体系因污染行混入而被误判为『文献冲突』整组剔除。",
        "",
        "## 冲突组辅助检查（组内能量极差 > 0.5 eV，仅报告不剔除，共 %d 组 / %d 行）"
        % (n_conflict_groups, n_conflict_rows),
        "",
    ]
    if conflict_rows:
        lines += [
            "以下分组的组内能量极差 > 0.5 eV（旧规则会整组剔除；现降级为辅助检查，仍取中位数合并）：",
            "",
            "| surface | ab | energy_eV | reference |",
            "|---|---|---|---|",
        ]
        for s, ab, e, ref in conflict_rows:
            lines.append(f"| {s} | {ab} | {e:.3f} | {ref} |")
    else:
        lines += [
            "无。标签污染修复后不存在极差 > 0.5 eV 的重复组："
            "旧版报告中 9 组/18 行『纯金属 (211) CO* 两文献矛盾』实为上述 C*/O* 标签污染所致，"
            "非真实文献冲突。",
        ]
    lines += [
        "",
        "## 与旧版（标签污染未修复）的差异说明",
        "",
        f"- 旧版把步骤 2 剔除的 C*/O* 污染行当作 CO 吸附能参与去重，导致 9 组/18 行被误判为"
        "文献冲突整组剔除；本版按反应身份过滤后，这些纯金属 (211) 体系恢复为 JPCC 单值，"
        f"最终行数 = {len(df)} 行（旧版 937 行）。（IQR 步骤在去重后重新计算。）",
        "",
        "## 已知注意事项",
        "",
        "- `Mo2C(001)`（碳化物表面）唯一的 ab=CO* 行实为 C*/O* 标签污染行（步骤 2 已剔除）；"
        "如需研究碳化物体系应单独处理。",
        "- 能量为 CatApp 原始参考（气相 CO / O2 参考态），CO* 与 O* 的零点不同，"
        "跨吸附种直接比较能量需谨慎；模型中 adsorbate 已作 one-hot 特征。",
        "",
        "## EDA 图",
        "",
        "- `figures/eda_energy_hist.png`：分吸附种能量直方图",
        "- `figures/eda_top_elements.png`：元素出现频次 Top 15",
        "- `figures/eda_facet_counts.png`：晶面分布（分吸附种堆叠）",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[写出] {REPORT}")

    # 终端摘要
    print("\n===== 清洗摘要 =====")
    for name, ndel, desc in log:
        print(f"{name}: -{ndel} 行")
    print(f"最终: {len(df)} 行 | CO {n_co} / O {n_o} | 唯一 comp {n_unique_comp} | "
          f"纯金属 {n_pure} / 合金 {n_alloy}")
    print(f"能量范围: CO [{erng.loc['CO','min']:.3f},{erng.loc['CO','max']:.3f}] | "
          f"O [{erng.loc['O','min']:.3f},{erng.loc['O','max']:.3f}] eV")


if __name__ == "__main__":
    main()
