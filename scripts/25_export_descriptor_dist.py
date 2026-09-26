#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""25_export_descriptor_dist.py

导出 web/assets/descriptor_distributions.json：「描述符分布定位」面板的数据。

内容：
- 对 36 个特征中全局 SHAP 前 12 个（读 assets/shap_global.json 取排名）：
  * 训练分布直方图（32 bin，存 bin 边界 + 计数）
  * 5/25/50/75/95 百分位、min、max、mean
- 中文特征名映射（面板标题用）

数据来源：data_processed/hstar_dataset_v2.csv（1836 行 × 42 列，见 feature_spec.md §3）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed" / "hstar_dataset_v2.csv"
SHAP = ROOT / "web" / "assets" / "shap_global.json"
OUT = ROOT / "web" / "assets" / "descriptor_distributions.json"

N_BINS = 32
TOP_K = 12

# 中文特征名映射（与 index.html 的 FEATURE_LABELS 保持一致）
FEATURE_LABELS = {
    "n_elements": "元素个数",
    "en_wmean": "加权平均电负性",
    "en_max": "最大电负性",
    "en_min": "最小电负性",
    "en_range": "电负性极差",
    "en_wstd": "电负性加权标准差",
    "radius_wmean": "加权平均原子半径 (pm)",
    "radius_max": "最大原子半径 (pm)",
    "radius_min": "最小原子半径 (pm)",
    "radius_range": "原子半径极差 (pm)",
    "radius_wstd": "原子半径加权标准差 (pm)",
    "group_wmean": "加权平均族号",
    "group_max": "最大族号",
    "group_min": "最小族号",
    "group_range": "族号极差",
    "group_wstd": "族号加权标准差",
    "period_wmean": "加权平均周期",
    "period_max": "最大周期",
    "period_min": "最小周期",
    "period_range": "周期极差",
    "period_wstd": "周期加权标准差",
    "ie1_wmean": "加权平均第一电离能 (eV)",
    "ie1_max": "最大第一电离能 (eV)",
    "ie1_min": "最小第一电离能 (eV)",
    "ie1_range": "第一电离能极差 (eV)",
    "ie1_wstd": "第一电离能加权标准差 (eV)",
    "d_el_wmean": "加权平均 d 电子数",
    "d_el_max": "最大 d 电子数",
    "d_el_min": "最小 d 电子数",
    "d_el_range": "d 电子数极差",
    "d_el_wstd": "d 电子数加权标准差",
    "structure_A1": "A1 结构指示",
    "structure_L10": "L1₀ 结构指示",
    "structure_L12": "L1₂ 结构指示",
    "facet_101": "(101) 晶面指示",
    "facet_111": "(111) 晶面指示",
}


def rnd(x, sig=6):
    """保留若干位有效数字，控制文件体积。"""
    return float(f"{x:.{sig}g}")


def main():
    df = pd.read_csv(DATA)
    shap = json.loads(SHAP.read_text(encoding="utf-8"))
    top_feats = [d["feature"] for d in shap["top15"]][:TOP_K]

    dists = {}
    for f in top_feats:
        v = df[f].to_numpy(dtype=float)
        counts, edges = np.histogram(v, bins=N_BINS)
        pcts = np.percentile(v, [5, 25, 50, 75, 95])
        dists[f] = {
            "label": FEATURE_LABELS.get(f, f),
            "bin_edges": [rnd(e) for e in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
            "p5": rnd(pcts[0]),
            "p25": rnd(pcts[1]),
            "p50": rnd(pcts[2]),
            "p75": rnd(pcts[3]),
            "p95": rnd(pcts[4]),
            "min": rnd(v.min()),
            "max": rnd(v.max()),
            "mean": rnd(v.mean()),
        }

    out = {
        "meta": {
            "n_rows": int(len(df)),
            "n_bins": N_BINS,
            "source": "data_processed/hstar_dataset_v2.csv",
            "ranking": "shap_global.json top15 前 %d 个" % TOP_K,
            "usage": "描述符分布定位面板：训练分布直方图 + 输入特征值定位",
        },
        "labels": {f: FEATURE_LABELS.get(f, f) for f in top_feats},
        "features": dists,
    }

    txt = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    # 挂载点原子 rename 可能 Permission denied，直接普通写入
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    size_kb = OUT.stat().st_size / 1024
    print(f"已导出 {OUT}（{size_kb:.1f} KB，{TOP_K} 个特征 × {N_BINS} bin）")
    if size_kb > 100:
        raise SystemExit("警告：文件超过 100KB 限制")


if __name__ == "__main__":
    main()
