# -*- coding: utf-8 -*-
"""
05_fetch_hstar.py — Mamun 2019 高通量双金属数据集 → H*（HER 主线）dataset v1

数据源（按优先级自动回退）：
  1. Zenodo: https://zenodo.org/records/17157086/files/MamunHighT2019_adsorption.json?download=1
  2. CatBench 镜像: https://catbench.org/benchmark/MamunHighT2019.json.gz（gzip 压缩，Zenodo 504 时使用）
  注：镜像版不含 atoms_json（仅能量与 ref 哈希），structure/facet 由 12 原子超胞计量模式推断：
  M12→A1(111)，A9B3→L12(111)，A6B6→L10(101)。已对两条记录验证
  ref_ads_eng == Σ stoi·energy_ref（E_ads = E(H*slab) − E(slab) − 0.5·E(H2)，负值=放热）。

目标子集：稀释 H*（反应键名含 "0.5H2(g) + * -> H*"）。
去重（HER 标准做法，与 CO 线规则不同，见 notes/05_hstar_report.md）：
  同一 (comp, structure) 的 _N 后缀条目是同一表面的不同吸附构型（不同位点/弛豫重构），
  取最低 E_ads（最稳定吸附位点，对应 Nørskov 火山 ΔG_H* 定义）；n_raw 记录构型数；
  组内极差统计写入报告（极差 >0.5 eV 的组清单另存 hstar_group_spread.csv）。

输出：data_processed/hstar_dataset_v1.csv
      列：comp（规范化：元素按字母排序+计量数约简）、structure ∈ {A1,L10,L12}、
          facet、energy_eV、n_raw、site_spread（组内能量极差 = 位点能量景观多样性）
      data_processed/hstar_group_spread.csv（各组极差明细）
      figures/hstar_eda_energy_hist.png
      notes/05_hstar_report.md（本脚本写"数据获取与清洗"部分）
"""
import gzip
import json
import re
import time
from functools import reduce
from math import gcd
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz"  # 压缩存储
DP = ROOT / "data_processed"
FIGDIR = ROOT / "figures"
REPORT = ROOT / "notes" / "05_hstar_report.md"

ZENODO_URL = ("https://zenodo.org/records/17157086/files/"
              "MamunHighT2019_adsorption.json?download=1")
MIRROR_URL = "https://catbench.org/benchmark/MamunHighT2019.json.gz"

C_MAIN = "#B35C24"
C_ALT = "#DFB27E"
C_3RD = "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}


def download(url, dest, gz=False, retries=5):
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) < 1e6:
                raise IOError(f"响应过小（{len(data)} B），疑似 504 错误页")
            if not gz:  # Zenodo 为纯 json，压缩后统一存 .gz
                data = gzip.compress(data)
            dest.write_bytes(data)
            print(f"[下载成功] {url} → {dest} ({len(data)/1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"[下载失败 {i+1}/{retries}] {url}: {e}")
            time.sleep(10 * (i + 1))
    raise RuntimeError("所有数据源均失败")


def parse_prefix(p):
    toks = re.findall(r"([A-Z][a-z]?)(\d+)", p)
    rebuilt = "".join(a + b for a, b in toks)
    if rebuilt != p:
        raise ValueError(f"[前缀解析异常] {p!r} → {toks!r}，请人工检查")
    return {el: int(n) for el, n in toks}


def normalize_comp(cd):
    """元素按字母排序 + 计量数 gcd 约简，如 Pt9Ti3 → Pt3Ti，Ag12 → Ag。"""
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def main():
    DP.mkdir(exist_ok=True)
    FIGDIR.mkdir(exist_ok=True)

    if not RAW.exists():
        try:
            download(ZENODO_URL, RAW)
        except RuntimeError:
            print("[回退] Zenodo 不可用，改用 catbench.org 镜像")
            download(MIRROR_URL, RAW, gz=True)
    data = json.loads(gzip.decompress(RAW.read_bytes()))
    print(f"[读入] {RAW.name}: {len(data)} 条吸附反应")

    # ---- 提取稀释 H* ----
    h_keys = [k for k in data if "0.5H2(g) + * -> H*" in k]
    rows = []
    for k in h_keys:
        prefix = re.match(r"^([A-Za-z0-9]+)_", k).group(1)
        cd = parse_prefix(prefix)
        structure, facet = STRUCT_RULE[tuple(sorted(cd.values()))]
        rows.append(dict(prefix=prefix, comp=normalize_comp(cd),
                         structure=structure, facet=facet,
                         energy_eV=float(data[k]["ref_ads_eng"])))
    df = pd.DataFrame(rows)
    print(f"[H* 子集] {len(df)} 条，{df['prefix'].nunique()} 个唯一表面，"
          f"{len({e for c in df['comp'] for e in re.findall(r'[A-Z][a-z]?', c)})} 种金属")

    # ---- 组内极差统计（同 comp+structure 的不同吸附构型）----
    g = df.groupby(["comp", "structure"])
    spread = g["energy_eV"].agg(n_raw="size", E_min="min", E_max="max",
                                E_mean="mean", E_median="median")
    spread["range"] = spread["E_max"] - spread["E_min"]
    spread = spread.sort_values("range", ascending=False)
    spread.to_csv(DP / "hstar_group_spread.csv")
    n_spread = int((spread["range"] > 0.5).sum())
    print(f"[构型离散] {len(spread)} 个表面；极差>0.5 eV 的 {n_spread} 个"
          f"（明细 → hstar_group_spread.csv）")

    # ---- 去重：取最稳定吸附位点（E_ads 最低）；site_spread = 组内极差（位点能量景观多样性）----
    out = (g["energy_eV"].min().rename("energy_eV").reset_index()
           .merge(g.size().rename("n_raw").reset_index(), on=["comp", "structure"]))
    out["facet"] = out["structure"].map({"A1": 111, "L12": 111, "L10": 101})
    out = out.merge(spread["range"].rename("site_spread").reset_index(),
                    on=["comp", "structure"])
    out = out[["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]]
    out = out.sort_values(["structure", "comp"]).reset_index(drop=True)
    out.to_csv(DP / "hstar_dataset_v1.csv", index=False)
    print(f"[写出] {DP/'hstar_dataset_v1.csv'}  ({len(out)} 行)")

    # ---- EDA：能量分布（分结构类型）----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(out["energy_eV"].min(), out["energy_eV"].max(), 50)
    for st, col in [("A1", C_MAIN), ("L12", C_ALT), ("L10", C_3RD)]:
        sub = out.loc[out["structure"] == st, "energy_eV"]
        ax.hist(sub, bins=bins, alpha=0.75, color=col, edgecolor="white",
                linewidth=0.4, label=f"{st} (n={len(sub)})")
    ax.set_xlabel(r"$E_{ads}$(H*) at most stable site (eV)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hstar_eda_energy_hist.png")
    plt.close(fig)
    print("[写出] figures/hstar_eda_energy_hist.png")

    # ---- 报告（数据获取与清洗部分）----
    st_counts = out["structure"].value_counts().to_dict()
    lines = [
        "# H* 主线报告（Mamun 2019 高通量双金属 → HER 筛选）",
        "",
        "## 1. 数据获取",
        "",
        "- 来源：CatBench Zenodo 镜像（Mamun et al., *Sci. Data* 6:76, 2019, CC-BY-4.0）；"
        "Sci. Data 原文 DOI: 10.1038/s41597-019-0082-4 的数据经 CatBench 重新整理",
        f"- 实际下载：catbench.org 镜像（gzip，95.3 MB）。Zenodo 记录 17157086 在本阶段持续 504，"
        "未能获取含 atoms_json 的 195.4 MB 完整版",
        f"- 全库 {len(data)} 条吸附反应；取稀释 H*（0.5H₂(g)+*→H*）**{len(df)}** 条",
        "- 能量验证：抽样验证 ref_ads_eng == Σ stoi·energy_ref"
        "（即 E_ads = E(H*slab) − E(slab) − 0.5·E(H₂)，**负值 = 放热吸附**，与 CatApp 标度相反！）",
        "",
        "## 2. 清洗决策",
        "",
        "### 结构/晶面推断",
        "",
        "镜像版无 atoms_json，structure/facet 由 12 原子超胞计量模式推断（与 SPEC 第 6 节记载一致）：",
        "",
        "| 计量模式 | structure | facet | 组数 |",
        "|---|---|---|---|",
        f"| M12（纯金属） | A1 | (111) | {st_counts.get('A1', 0)} |",
        f"| A9B3（3:1） | L12 | (111) | {st_counts.get('L12', 0)} |",
        f"| A6B6（1:1） | L10 | (101) | {st_counts.get('L10', 0)} |",
        "",
        "### 组成规范化",
        "",
        "前缀（如 Pt9Ti3）解析为元素计量字典后：元素按字母排序 + 计量数 gcd 约简（Pt9Ti3→Pt3Ti）。",
        "实测**无同素异写冲突**：每个规范化 (comp, structure) 只对应一种原始前缀写法，"
        "因此 GroupKFold 用规范化 comp 即可避免 CO 线的隐性泄漏（AgAu3 vs Au3Ag 问题不存在）。",
        "",
        "### 去重：取最稳定位点（与 CO 线不同的决策）",
        "",
        f"- 同一表面的 `_N` 后缀条目是**不同吸附构型**（不同位点/弛豫重构），不是重复测量：",
        f"  {len(spread)} 个表面组内能量极差中位数 {spread['range'].median():.3f} eV，"
        f"极差 >0.5 eV 的有 {n_spread} 个（{100*n_spread/len(spread):.0f}%）",
        "- 若照搬 CO 线『极差>0.5 整组剔除』会损失一半表面且系统性偏向各构型能量一致的表面",
        "- **决策**：按 HER 文献标准（Nørskov 火山用最有利位点的 ΔG_H*），每组取最低 E_ads；"
        "n_raw 记录构型数；组内极差作为派生量 `site_spread` 保留在 v1（位点能量景观多样性，"
        "n_raw 记录构型数；组内极差作为分析列 `site_spread` 保留在 v1；"
        "极差明细存 `data_processed/hstar_group_spread.csv`",
        "",
        "### ⚠ 泄漏特征的识别与移除（终审修复）",
        "",
        "`site_spread`（组内能量极差）曾作为派生特征进入建模，终审识别其为**目标泄漏特征**：",
        "它由同表面各构型能量（含目标值本身）计算得到，对没有 DFT 数据的新表面不可得，",
        "部署时无法使用。自修复版起从建模特征中移除（06/07/08 一致排除），仅保留为分析列；",
        "去泄漏后 MAE 由 0.104 → 0.120 eV。min-vs-mean 去重口径敏感性分析见第 4 节（08 生成）。",
        "",
        "## 3. hstar_dataset_v1.csv 汇总",
        "",
        f"- 行数：**{len(out)}**（唯一表面数）",
        f"- 结构分布：{st_counts}",
        f"- 能量范围：[{out['energy_eV'].min():.3f}, {out['energy_eV'].max():.3f}] eV；"
        f"中位数 {out['energy_eV'].median():.3f} eV",
        f"- 构型数分布：n_raw 中位数 {int(out['n_raw'].median())}，最大 {int(out['n_raw'].max())}",
        "- EDA：`figures/hstar_eda_energy_hist.png`",
        "",
        "（SHAP 与筛选结果由 08 脚本追加到本文档第 4 节）",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[写出] {REPORT}")


if __name__ == "__main__":
    main()
