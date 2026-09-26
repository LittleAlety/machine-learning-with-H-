# -*- coding: utf-8 -*-
"""P4 信息天花板 oracle — 预注册判读规则

构造 3 个作弊（oracle）特征，逐个过 portability_gate 闸门：
  1. site_onehot：真位点 one-hot——同 (comp,structure) 组内取得最小能量（即 target）
     的原始记录在组内能量升序中的位次（0..9），one-hot 编码（k=10）。
     该特征由 target 决定 → 金标准口径必然强泄漏。
  2. config_mean：同组成同结构全部原始构型能量均值（含 target 本身）→ 泄漏。
  3. config_count：组内构型计数 n_raw（元数据，新表面不可预知）。

判读规则（预注册）：
  - 三个特征预期全部判 LEAK（Δ_A > 0.005 且保留率 < 0.8）→ 反向验证闸门灵敏度；
    若出现 PASS/NO_GAIN，如实记录为闸门灵敏度边界证据。
  - 信息侧 Δ_A（取三者最大）vs 容量侧最大改善（P1 GBDT 段 min(OOF) 相对基线的改善）
    画对比柱图 p4_oracle_bar.png。
诊断专用，禁止用于选择交付配置。
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portability_gate"))
from _common import PROBES, ROOT, RAW_GZ, C_MAIN, C_ALT, C_3RD, load_data, plt
import gate

STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}


def parse_prefix(p):
    return {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", p)}


def normalize_comp(cd):
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def build_oracle_features(df):
    """从原始记录构造 3 个作弊特征，与特征表 (comp,structure) 行对齐。"""
    data = json.loads(gzip.decompress(RAW_GZ.read_bytes()))
    recs = defaultdict(list)
    for k, v in data.items():
        if "0.5H2(g) + * -> H*" not in k:
            continue
        cd = parse_prefix(re.match(r"^([A-Za-z0-9]+)_", k).group(1))
        recs[(normalize_comp(cd), STRUCT_RULE[tuple(sorted(cd.values()))][0])].append(
            float(v["ref_ads_eng"]))
    n = len(df)
    cfg_mean = np.zeros(n)
    cfg_cnt = np.zeros(n)
    for i, (c, s) in enumerate(zip(df["comp"], df["structure"])):
        es = np.sort(recs[(c, s)])
        cfg_mean[i] = es.mean()
        cfg_cnt[i] = len(es)
    # site_onehot：target = 组内 min，故"真位点"定义为取得 min 的原始记录在
    # 该组 raw key 字典序中的下标（0..K-1）的 one-hot。该位次由 target 参与
    # 定位，属作弊标签；新表面不可计算。
    site_idx = np.zeros(n, dtype=int)
    raw_keys = defaultdict(list)
    for k in data:
        if "0.5H2(g) + * -> H*" not in k:
            continue
        cd = parse_prefix(re.match(r"^([A-Za-z0-9]+)_", k).group(1))
        key = (normalize_comp(cd), STRUCT_RULE[tuple(sorted(cd.values()))][0])
        raw_keys[key].append(k)
    for i, (c, s) in enumerate(zip(df["comp"], df["structure"])):
        keys = sorted(raw_keys[(c, s)])
        es = [float(data[k]["ref_ads_eng"]) for k in keys]
        site_idx[i] = int(np.argmin(es))
    K = max(site_idx.max() + 1, 2)
    onehot = np.zeros((n, K))
    onehot[np.arange(n), site_idx] = 1.0
    return {"site_onehot": onehot,
            "config_mean": cfg_mean.reshape(-1, 1),
            "config_count": cfg_cnt.reshape(-1, 1)}


def main():
    PROBES.mkdir(exist_ok=True)
    df, X84, y, groups, feats = load_data()
    # 闸门内部基线为 v2 36 维（gate.load_base）；为一致性，直接用 gate 的装载
    gdf, X_base, gy, ggroups = gate.load_base()
    assert np.allclose(gy, y) and (ggroups == groups).all()
    oracles = build_oracle_features(df)
    reports = []
    for name, F in oracles.items():
        rep = gate.portability_check(name, F, X_base, gy, ggroups)
        reports.append(rep)
        print(f"[P4] {name}: Δ_A={rep['delta_A']:.4f} r={rep['retention']:.3f} "
              f"→ {rep['verdict']}", flush=True)
    (PROBES / "p4_gate_reports.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False))

    # 容量侧最大改善（P1 GBDT 段）
    p1v = json.loads((PROBES / "p1_verdict.json").read_text())
    base_mae = reports[0]["mae_base_10seeds"]
    cap_best = float(p1v["gbdt_best"]["oof_mae"])
    cap_gain = base_mae - cap_best
    info_best = max(r["delta_A"] for r in reports)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(["Capacity side\n(P1 GBDT best gain)", "Information side\n(P4 oracle max Δ_A)"],
                  [cap_gain, info_best], color=[C_MAIN, C_3RD], width=0.55)
    for b, v in zip(bars, [cap_gain, info_best]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=11)
    ax.set_ylabel("OOF MAE improvement (eV)")
    ax.set_title("P4 information ceiling vs capacity headroom")
    fig.tight_layout()
    fig.savefig(PROBES / "p4_oracle_bar.png")
    plt.close(fig)

    summary = {"reports": reports,
               "capacity_side_gain": cap_gain, "info_side_max_deltaA": info_best,
               "verdicts": {r["feature"]: r["verdict"] for r in reports}}
    (PROBES / "p4_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["verdicts"], indent=2))


if __name__ == "__main__":
    main()
