# -*- coding: utf-8 -*-
"""F4 位点 one-hot 特征过可移植性闸门 — plan-hb v3.0 Workstream F

候选特征（金标准口径）：每个表面取其 min-energy 位点（= v3 标签对应位点）的
site_full one-hot（28 类，物理身份特征：fcc/hcp 空心、桥、顶位及其元素环境）。
仅在开发集（1,560 表面；lockbox 冻结）上走闸门三态判定。
判定逻辑复用 innovation/portability_gate/gate.py 的 portability_check：
  PASS / LEAK / NO_GAIN（Δ_A>0.005 且保留率 r≥0.8 → PASS）。
输出：f4_gate_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "innovation" / "portability_gate"))
from gate import portability_check  # noqa: E402

FEAT_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
DEV_CSV = OUT / "site_dataset_dev.csv"
ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]


def main():
    v3 = pd.read_csv(FEAT_CSV)
    feat_cols = [c for c in v3.columns if c not in ID_COLS]
    site = pd.read_csv(DEV_CSV)
    # min-energy 位点（金标准：用标签选择位点 → 注意这正是"金标准口径"，
    # 部署时不可见，闸门线 B 检验的就是其可推断性）
    idx = site.groupby("comp")["energy_eV"].idxmin()
    minsite = site.loc[idx, ["comp", "site_full"]].set_index("comp")

    dev = v3[v3["comp"].isin(minsite.index)].copy().reset_index(drop=True)
    dev["minsite"] = dev["comp"].map(minsite["site_full"])
    cats = sorted(site["site_full"].unique())
    F = np.zeros((len(dev), len(cats)))
    cat_pos = {c: i for i, c in enumerate(cats)}
    for i, s in enumerate(dev["minsite"]):
        F[i, cat_pos[s]] = 1.0

    X = dev[feat_cols].values.astype(float)
    y = dev["energy_eV"].values.astype(float)
    groups = dev["comp"].values
    rep = portability_check("minsite_onehot_dev", F, X, y, groups)
    rep["note"] = ("开发集 1560 表面（lockbox 冻结）；特征=min-energy 位点的 "
                   "site_full one-hot（28 类，物理身份）；预期 PASS，"
                   "判 LEAK 则记录并分析")
    (OUT / "f4_gate_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
