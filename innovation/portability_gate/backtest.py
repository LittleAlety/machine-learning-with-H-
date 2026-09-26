# -*- coding: utf-8 -*-
"""
innovation/portability_gate/backtest.py — 闸门回溯验证：三个已知案例
| 案例 | 预期判定 | 依据 |
| site_spread 位点能量散布 | LEAK | 3.3 节已证（目标泄漏，虚增约 15%） |
| n_raw 构型计数 | LEAK | 3.9 节已证（T3 元数据泄漏） |
| melting_point 族 | PASS 或 NO_GAIN | 3.15 节：新组成可算，单族未过阈值 |
另附 CatApp CO*/O* 线的 T1 构造案例（分类学佐证）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gate import (OUT, ROOT, SEEDS, V2_COLS, load_base, oof_mae,
                  portability_check, ridge_infer)


def main():
    df, X, y, groups = load_base()
    reports = []

    # 案例 1: site_spread（T1 目标泄漏）
    v1 = pd.read_csv(ROOT / "data_processed" / "hstar_dataset_v1.csv")
    key = ["comp", "structure", "facet"]
    m = df.merge(v1[key + ["site_spread"]], on=key, how="left", validate="1:1")
    assert m["site_spread"].notna().mean() > 0.99
    reports.append(portability_check(
        "site_spread (T1 目标泄漏)", m[["site_spread"]].values, X, y, groups))

    # 案例 2: n_raw（T3 元数据泄漏）
    reports.append(portability_check(
        "n_raw (T3 元数据泄漏)", df[["n_raw"]].values.astype(float), X, y, groups))

    # 案例 3: melting_point 族（5 维，v3 采纳族）
    mp_cols = [f"melting_point_{s}" for s in ["wmean", "max", "min", "range", "wstd"]]
    reports.append(portability_check(
        "melting_point 族（v3 采纳）", df[mp_cols].values, X, y, groups))

    # 案例 4（分类学佐证）: CatApp CO* 线构造 T1 特征
    cat = pd.read_csv(ROOT / "data_raw" / "catappdata.csv")
    co = cat[cat["ab"] == "CO*"]
    # 同一 (surface, ab) 的反应能极差 = 同类目标泄漏构造
    co = co.dropna(subset=["reaction_energy"])
    spread = (co.groupby(["surface", "ab"])["reaction_energy"]
                .agg(lambda s: s.max() - s.min()).rename("catapp_spread"))
    n_multi = int((spread > 0).sum())
    reports.append({
        "feature": "CatApp CO* 线同表面能量极差（构造性 T1 案例）",
        "note": f"CatApp 共 {len(co)} 条 CO* 记录，{len(spread)} 个表面键，"
                f"{n_multi} 个键极差>0；该特征与标签同分布机械耦合，"
                "若入模必过线 A（与标签同源的统计量），但新表面不可计算 → 闸门判 LEAK。"
                "构造即证，不实际入模（节省算力，机制与案例 1 同构）。",
        "verdict": "LEAK (by construction)"})

    for r in reports:
        print(json.dumps(r, ensure_ascii=False))
    (OUT / "backtest_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT/'backtest_report.json'}")


if __name__ == "__main__":
    main()
