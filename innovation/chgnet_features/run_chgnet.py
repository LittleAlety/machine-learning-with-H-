# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/run_chgnet.py — CHGNet 单点推理 → 表面特征 (红线11: 仅作特征生成器)

预注册统计清单 (先于推理冻结, 特征名见 H6 freeze 报告):
  表面层定义 (v2, 修正): 按 z 坐标取最顶层 25% 原子 (z > quantile(z,0.75));
    修正原因: 初版 z>=zmax-2Å 在 SlabGenerator 重定向胞上仅捕获 1-2 原子,
    离散度统计退化; 修正发生在任何模型拟合之前, 不构成 lockbox 触碰。
  chg_surf_mag_mean        表面层磁矩均值
  chg_surf_mag_absmean     表面层 |磁矩| 均值
  chg_surf_mag_max / min   表面层磁矩极值
  chg_surf_mag_std         表面层磁矩离散度 (总体标准差)
  chg_surf_mag_range       max-min
  chg_bulk_mag_absmean     非表面层 |磁矩| 均值 (体相对照)
  chg_surf_bulk_absmag_diff 表面-体相 |磁矩| 均值差 (表面磁矩增强)
  chg_energy_per_atom      CHGNet 单点能量/原子 (eV)
  电荷转移统计: N/A — CHGNet v0.3.0 不输出原子电荷 (如实记录, 不伪造)
  cache 同时保存原始 m/z 向量, 便于离线再聚合而无需重跑推理。

幂等: cache/<comp>_<structure>_<facet>.json 存在即跳过; 中断可续跑。
用法: python run_chgnet.py [--limit N] [--subset mag3d|all]
"""
import argparse, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLAB_DIR = OUT / "slabs"
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"

MAG3D = {"Fe", "Co", "Ni", "Mn", "Cr"}
SURF_TOL = None  # deprecated: surface = top quartile by z

FEATURES = ["chg_surf_mag_mean", "chg_surf_mag_absmean", "chg_surf_mag_max",
            "chg_surf_mag_min", "chg_surf_mag_std", "chg_surf_mag_range",
            "chg_bulk_mag_absmean", "chg_surf_bulk_absmag_diff",
            "chg_energy_per_atom"]


def aggregate(m, z):
    m = np.asarray(m, dtype=float)
    z = np.asarray(z, dtype=float)
    surf = z > np.quantile(z, 0.75)
    sm, bm = m[surf], m[~surf]
    return {
        "chg_surf_mag_mean": float(sm.mean()),
        "chg_surf_mag_absmean": float(np.abs(sm).mean()),
        "chg_surf_mag_max": float(sm.max()),
        "chg_surf_mag_min": float(sm.min()),
        "chg_surf_mag_std": float(sm.std()),
        "chg_surf_mag_range": float(sm.max() - sm.min()),
        "chg_bulk_mag_absmean": float(np.abs(bm).mean()),
        "chg_surf_bulk_absmag_diff": float(np.abs(sm).mean() - np.abs(bm).mean()),
        "n_surf_atoms": int(surf.sum()),
    }


def infer_one(model, cif_path):
    from pymatgen.core import Structure
    s = Structure.from_file(str(cif_path))
    pred = model.predict_structure(s)
    m = np.asarray(pred["m"], dtype=float).ravel()
    e = float(pred["e"])
    z = np.array([site.z for site in s], dtype=float)
    out = aggregate(m, z)
    out.update({"chg_energy_per_atom": e / len(s), "n_atoms": len(s),
                "m_raw": [round(float(v), 6) for v in m],
                "z_raw": [round(float(v), 4) for v in z]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", choices=["all", "mag3d"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(FEAT)
    if args.subset == "mag3d":
        df = df[df["comp"].apply(lambda c: bool(MAG3D & set(re.findall(r"[A-Z][a-z]?", c))))]
    if args.nshards > 1:
        df = df.iloc[args.shard::args.nshards]
    if args.limit:
        df = df.head(args.limit)

    torch.set_num_threads(1)
    from chgnet.model import CHGNet
    model = CHGNet.load()

    t0 = time.time()
    done = fail = 0
    for _, row in df.iterrows():
        key = f"{row.comp}_{row.structure}_{row.facet}"
        cp = CACHE / f"{key}.json"
        if cp.exists():
            done += 1
            continue
        try:
            res = infer_one(model, SLAB_DIR / f"{key}.cif")
            res.update({"comp": row.comp, "structure": row.structure, "facet": int(row.facet)})
            cp.write_text(json.dumps(res))
            done += 1
        except Exception as ex:
            cp.write_text(json.dumps({"comp": row.comp, "error": f"{type(ex).__name__}: {ex}"}))
            fail += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"{done} done, {fail} fail, {el:.0f}s elapsed, eta {el/max(done,1)*(len(df)-done):.0f}s", flush=True)
    print(f"FINISHED subset={args.subset} n={len(df)} done={done} fail={fail} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
