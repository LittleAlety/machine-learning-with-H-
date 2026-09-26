# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/collect_features.py — 汇总 cache/ 为特征 csv
排除 lockbox 之外的用途不做任何目标值对齐; 输出 1836 行特征表 (若推理全量完成)
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
FEATURES = ["chg_surf_mag_mean", "chg_surf_mag_absmean", "chg_surf_mag_max",
            "chg_surf_mag_min", "chg_surf_mag_std", "chg_surf_mag_range",
            "chg_bulk_mag_absmean", "chg_surf_bulk_absmag_diff",
            "chg_energy_per_atom"]


def main():
    df = pd.read_csv(FEAT)
    rows, errors = [], []
    for _, r in df.iterrows():
        cp = OUT / "cache" / f"{r.comp}_{r.structure}_{r.facet}.json"
        if not cp.exists():
            errors.append({"comp": r.comp, "reason": "missing_cache"})
            continue
        d = json.loads(cp.read_text())
        if "error" in d:
            errors.append({"comp": r.comp, "reason": d["error"]})
            continue
        rows.append({"comp": r.comp, "structure": r.structure, "facet": r.facet,
                     **{f: d[f] for f in FEATURES},
                     "n_atoms": d["n_atoms"], "n_surf_atoms": d["n_surf_atoms"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "chgnet_features.csv", index=False)
    status = {
        "n_rows": len(out), "n_total": len(df), "n_errors": len(errors),
        "errors": errors[:20],
        "coverage": len(out) / len(df),
    }
    (OUT / "collect_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in status.items() if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
