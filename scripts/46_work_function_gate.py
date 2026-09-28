#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Work function (Phi) dimension through the portability gate.

Work function is a surface/electronic property (closer to adsorption physics
than bulk moduli), but a tabulated elemental Phi combined by composition is
still a composition-derived feature. We build five stoichiometry-weighted Phi
statistics and run the same Portability Gate (line A leakage / line B
portability) used for every candidate dimension.

Elemental polycrystalline work functions (eV) are standard tabulated values
(Michaelson 1977 / CRC Handbook); only elements actually present are used.
"""
import importlib.util
import json
import re
from functools import partial
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = ROOT / "innovation" / "portability_gate"
OUT_JSON = ROOT / "outputs" / "work_function_gate.json"
OUT_MD = ROOT / "notes" / "work_function_gate.md"

spec = importlib.util.spec_from_file_location("gate", GATE_DIR / "gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

# Force single-threaded fits (n_jobs=-1 oversubscribes in this env) and use one
# seed to keep the gate tractable; verdict is labeled exploratory.
gate.XGBRegressor = partial(gate.XGBRegressor, n_jobs=1)

# Polycrystalline work functions (eV), standard tabulated values
PHI = {
    "Ag": 4.26, "Al": 4.28, "Au": 5.10, "B": 4.45, "Ba": 2.70, "Be": 5.00,
    "Bi": 4.22, "C": 5.00, "Ca": 2.87, "Cd": 4.22, "Ce": 2.90, "Co": 5.00,
    "Cr": 4.50, "Cs": 2.14, "Cu": 4.65, "Fe": 4.50, "Ga": 4.20, "Ge": 4.80,
    "Hf": 3.90, "Hg": 4.49, "In": 4.12, "Ir": 5.27, "K": 2.30, "La": 3.50,
    "Li": 2.93, "Mg": 3.66, "Mn": 4.10, "Mo": 4.60, "Na": 2.36, "Nb": 4.30,
    "Ni": 5.15, "Os": 4.83, "Pb": 4.25, "Pd": 5.12, "Pt": 5.65, "Rb": 2.26,
    "Re": 4.96, "Rh": 4.98, "Ru": 4.71, "Sb": 4.55, "Sc": 3.50, "Si": 4.60,
    "Sn": 4.42, "Sr": 2.59, "Ta": 4.25, "Tc": 4.80, "Te": 4.95, "Th": 3.40,
    "Ti": 4.33, "Tl": 3.84, "V": 4.30, "W": 4.55, "Y": 3.10, "Zn": 4.33,
    "Zr": 4.05,
}
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def phi_stats(comp):
    toks = COMP_TOKEN.findall(comp)
    els = [e for e, _ in toks]
    c = np.array([int(n) if n else 1 for _, n in toks], dtype=float)
    w = c / c.sum()
    v = np.array([PHI[e] for e in els], dtype=float)
    mu = float(np.average(v, weights=w))
    return [mu, v.max(), v.min(), v.max() - v.min(),
            float(np.sqrt(np.average((v - mu) ** 2, weights=w)))]


def main():
    df, X, y, groups = gate.load_base()
    needed = sorted({e for c in df["comp"] for e, _ in COMP_TOKEN.findall(c)})
    missing = [e for e in needed if e not in PHI]
    assert not missing, f"missing Phi values for: {missing}"
    Fall = np.array([phi_stats(c) for c in df["comp"]], dtype=float)

    # Subsample rows (fixed seed) so the gate finishes inside the background
    # runtime cap; the verdict is an exploratory signal, not a full estimate.
    rng0 = np.random.default_rng(0)
    sub = np.sort(rng0.choice(len(df), size=300, replace=False))
    Xs, ys, gs, Fs = X[sub], y[sub], groups[sub], Fall[sub]

    quick_seeds = [42]
    report = gate.portability_check("work_function_phi", Fs, Xs, ys, gs,
                                    seeds=quick_seeds)
    report["seeds_note"] = "exploratory single-seed (42), n_jobs=1"
    report["rows_note"] = "300-comp fixed-seed subsample (background runtime cap)"
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    md = [
        "# 功函数 Φ 可移植性闸门报告",
        f"- 特征：5 个计量加权 Φ 统计（wmean/max/min/range/wstd），元素 Φ 为标准多晶值（eV）",
        f"- Δ_A（金标准）：{report['delta_A']:+.5f} eV（std {report['delta_A_std']:.5f}），"
        f"种子胜 {report['seeds_win']}",
        f"- Δ_B（Ridge 推断）：{report['delta_B_inferred']:+.5f} eV，保留率 {report['retention']:.3f}",
        f"- 裁决：**{report['verdict']}**（采纳阈值 0.005 eV，保留率阈值 0.8）",
    ]
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
