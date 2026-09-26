# -*- coding: utf-8 -*-
"""
innovation/chgnet_features/h6_freeze_lockbox.py — 检查点 H6 (红线14)
在含 CHGNet 特征的模型开工前冻结 lockbox:
  - 沿用 splits/lockbox_v1.json 的组成划分 (不重新划分)
  - 物化 freeze 文件: comp,structure,facet,y 的规范 CSV, SHA256 留证
  - 单次裁决: lockbox 仅在最终评估用一次
幂等: 若 freeze 文件与源一致则复用, 不一致则报错 (禁止悄悄改划分)
"""
import hashlib, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEAT = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
LOCK = ROOT / "splits" / "lockbox_v1.json"
FREEZE_CSV = OUT / "lockbox_chgnet_frozen.csv"
FREEZE_JSON = OUT / "h6_lockbox_freeze.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    lock = json.loads(LOCK.read_text())
    comps = set(lock["lockbox_comps"])
    df = pd.read_csv(FEAT)
    lb = df[df["comp"].isin(comps)][["comp", "structure", "facet", "energy_eV"]]
    lb = lb.sort_values(["comp", "structure", "facet"]).reset_index(drop=True)
    body = lb.to_csv(index=False)
    if FREEZE_CSV.exists():
        if FREEZE_CSV.read_text() != body:
            print("ERROR: freeze file mismatch — lockbox 已冻结, 禁止改动", file=sys.stderr)
            sys.exit(2)
    else:
        FREEZE_CSV.write_text(body)
    digest = sha256(FREEZE_CSV)
    report = {
        "checkpoint": "H6_lockbox_freeze",
        "source_lockbox": "splits/lockbox_v1.json",
        "source_sha256": sha256(LOCK),
        "freeze_csv": str(FREEZE_CSV.name),
        "freeze_sha256": digest,
        "n_lockbox_rows": int(len(lb)),
        "n_lockbox_comps": int(lb["comp"].nunique()),
        "n_total_rows": int(len(df)),
        "structure_counts": lb["structure"].value_counts().to_dict(),
        "rule": "含 CHGNet 特征的模型开工前冻结; lockbox 对开发不可见, 单次裁决 (红线14)",
        "feature_block_planned": [
            "chg_surf_mag_mean", "chg_surf_mag_absmean", "chg_surf_mag_max",
            "chg_surf_mag_min", "chg_surf_mag_std", "chg_surf_mag_range",
            "chg_bulk_mag_absmean", "chg_surf_bulk_absmag_diff", "chg_energy_per_atom"],
    }
    FREEZE_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
