# -*- coding: utf-8 -*-
"""
innovation/static_geom/s1_static_features.py — S2 静态几何特征 (Phase 6 stage_S2_static)

预注册 (config/preregistered_phase6.yaml stage_S2_static):
  18 个连续局域特征拆静态组 (CN/GCN/近邻加权, 原型几何确定性生成) 与弛豫依赖组
  (层距/位移) 分别过闸门+公平契约。本脚本只做静态组, 10 个特征, 清单锁死禁扩:
    surf_cn_mean / surf_cn_min / surf_cn_std      表面层 CN 分布
    surf_gcn_mean / surf_gcn_std / surf_gcn_min   GCN (CN_max=12 fcc 参考)
    surf_nn_en_mean / surf_nn_en_std              A/B 近邻加权电负性 (Pauling)
    surf_nn_d_mean  / surf_nn_d_std               A/B 近邻加权 d 电子数
  表面层定义与 r2_features.py 完全一致: z > quantile(z, 0.75)。
  邻接/性质计算函数直接复用 innovation/continuous_local/r2_features.py (同一实现)。

输入: innovation/chgnet_features/slabs/*.cif (未弛豫原型 slab, 1836/1836)
幂等: 全量重算, 确定性输出覆盖写 static_geom_features.csv; 失败逐条如实记录。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLAB_DIR = ROOT / "innovation" / "chgnet_features" / "slabs"
sys.path.insert(0, str(ROOT / "innovation" / "continuous_local"))
from r2_features import neighbor_table, d_electrons, CN_MAX_REF  # noqa: E402
from pymatgen.core import Element, Structure  # noqa: E402

STATIC_COLS = ["surf_cn_mean", "surf_cn_min", "surf_cn_std",
               "surf_gcn_mean", "surf_gcn_std", "surf_gcn_min",
               "surf_nn_en_mean", "surf_nn_en_std",
               "surf_nn_d_mean", "surf_nn_d_std"]


def compute_static(cif: Path):
    s = Structure.from_file(cif)
    z = s.cart_coords[:, 2]
    surf = z > np.quantile(z, 0.75)

    nn, _ = neighbor_table(s)
    cn = nn.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        gcn = np.where(cn > 0, (nn @ cn) / CN_MAX_REF, 0.0)

    els = s.species
    en = np.array([Element(e.symbol).X if Element(e.symbol).X is not None
                   else np.nan for e in els])
    de = np.array([d_electrons(Element(e.symbol)) for e in els])

    def nn_weighted(prop):
        out = np.zeros(len(prop))
        for i in range(len(prop)):
            w = nn[i]
            out[i] = np.nanmean(prop[w]) if w.any() else np.nan
        return out

    nn_en, nn_de = nn_weighted(en), nn_weighted(de)
    return {
        "name": cif.stem,
        "surf_cn_mean": float(cn[surf].mean()),
        "surf_cn_min": float(cn[surf].min()),
        "surf_cn_std": float(cn[surf].std()),
        "surf_gcn_mean": float(gcn[surf].mean()),
        "surf_gcn_std": float(gcn[surf].std()),
        "surf_gcn_min": float(gcn[surf].min()),
        "surf_nn_en_mean": float(np.nanmean(nn_en[surf])),
        "surf_nn_en_std": float(np.nanstd(nn_en[surf])),
        "surf_nn_d_mean": float(np.nanmean(nn_de[surf])),
        "surf_nn_d_std": float(np.nanstd(nn_de[surf])),
    }


def main():
    rows, skipped = [], []
    for cif in sorted(SLAB_DIR.glob("*.cif")):
        try:
            rows.append(compute_static(cif))
        except Exception as e:  # 如实记录失败, 不伪造
            skipped.append({"name": cif.stem, "error": str(e)[:300]})
    df = pd.DataFrame(rows)
    parts = df["name"].str.rsplit("_", n=2, expand=True)
    df["comp"], df["structure"], df["facet"] = parts[0], parts[1], parts[2]
    df.to_csv(OUT / "static_geom_features.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT / "feature_failures.csv", index=False)
    print(f"static features: {len(df)} rows; failures: {len(skipped)}")
    print(df[STATIC_COLS].describe().T[["mean", "std", "min", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
