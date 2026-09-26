# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/h1_build_slabs.py — 检查点 H1: slab 可建率
原型规则:
  A1  (fcc, Fm-3m): 单元素立方晶胞
  L12 (Cu3Au, Pm-3m): 少数元素@(0,0,0), 多数元素@(0.5,0.5,0)+面心
  L10 (CuAu, P4/mmm): 四方 a=b, c/a=0.92(固定), 沿 c 交替层
晶格常数: a = 2*sqrt(2)*加权平均原子半径(Å), 确定性、不依赖目标值 (红线: 无泄漏)
facet: (1,1,1) 或 (1,0,1); SlabGenerator min_slab=8Å, min_vacuum=12Å, 2x2 表面超胞
幂等: slabs/<comp>_<structure>_<facet>.cif 存在即跳过
"""
import json, re, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.periodic_table import Element

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLAB_DIR = OUT / "slabs"
SLAB_DIR.mkdir(exist_ok=True)

FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"


def parse_comp(comp):
    """'Ag3Au' -> [('Ag',3),('Au',1)]"""
    return [(e, int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", comp)]


def est_lattice(comp):
    pairs = parse_comp(comp)
    r = np.average([Element(e).atomic_radius or 1.35 for e, _ in pairs],
                   weights=[n for _, n in pairs])
    return float(2 * np.sqrt(2) * r)


def build_bulk(comp, structure):
    pairs = sorted(parse_comp(comp), key=lambda x: -x[1])
    a = est_lattice(comp)
    if structure == "A1":
        el = pairs[0][0]
        return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [el], [[0, 0, 0]])
    if structure == "L12":
        maj, mino = pairs[0][0], pairs[1][0]
        return Structure.from_spacegroup("Pm-3m", Lattice.cubic(a),
                                         [mino, maj], [[0, 0, 0], [0.5, 0.5, 0]])
    if structure == "L10":
        e1, e2 = pairs[0][0], pairs[1][0]
        lat = Lattice.tetragonal(a, a * 0.92)
        return Structure(lat, [e1, e1, e2, e2],
                         [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
    raise ValueError(structure)


def build_slab(comp, structure, facet):
    bulk = build_bulk(comp, structure)
    miller = tuple(int(c) for c in str(facet))
    sg = SlabGenerator(bulk, miller, min_slab_size=8.0, min_vacuum_size=12.0,
                       center_slab=True, in_unit_planes=True)
    slabs = sg.get_slabs()
    if not slabs:
        raise RuntimeError("no slab generated")
    slab = slabs[0]
    slab.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    slab = slab.get_sorted_structure()
    return slab


def main():
    df = pd.read_csv(FEAT)
    results, failures = [], []
    for _, row in df.iterrows():
        key = f"{row.comp}_{row.structure}_{row.facet}"
        path = SLAB_DIR / f"{key}.cif"
        try:
            if not path.exists():
                slab = build_slab(row.comp, row.structure, row.facet)
                slab.to(filename=str(path))
            n = len(Structure.from_file(str(path)))
            results.append({"comp": row.comp, "structure": row.structure,
                            "facet": int(row.facet), "n_atoms": n, "ok": True})
        except Exception as e:
            failures.append({"comp": row.comp, "structure": row.structure,
                             "facet": int(row.facet),
                             "reason": f"{type(e).__name__}: {e}"})
            results.append({"comp": row.comp, "structure": row.structure,
                            "facet": int(row.facet), "n_atoms": None, "ok": False})
    ok = sum(r["ok"] for r in results)
    reason_dist = {}
    for f in failures:
        reason_dist[f["reason"].split(":")[0]] = reason_dist.get(f["reason"].split(":")[0], 0) + 1
    natoms = [r["n_atoms"] for r in results if r["ok"]]
    report = {
        "checkpoint": "H1_slab_build",
        "n_total": len(results), "n_ok": ok, "n_fail": len(failures),
        "build_rate": ok / len(results),
        "pass_line": 1.0, "passed": ok == len(results),
        "n_atoms_min": int(min(natoms)), "n_atoms_max": int(max(natoms)),
        "n_atoms_mean": float(np.mean(natoms)),
        "failure_reason_dist": reason_dist,
        "failures": failures[:50],
    }
    (OUT / "h1_slab_build.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
