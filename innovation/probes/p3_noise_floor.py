# -*- coding: utf-8 -*-
"""P3 噪声地板（noise floor）— 全部论证的锚点 — 预注册判读规则

路径 a（数据内禀散布，写死方法）：
  从 data_raw/MamunHighT2019_adsorption.json.gz 解析稀释 H* 原始记录
  （解析代码复用 scripts/05_fetch_hstar.py 的规则：计量模式 → structure/facet），
  按 (comp, structure) 分组（= 同组成同结构的不同吸附构型，与建模样本一一对应）。
  - floor_a = 各组 MAD/2 的中位数，MAD = median|E_i − median(E)|（仅 n_raw≥2 的组）。
    MAE 的最优常数预测是 median，故 MAD 是"组内不可约绝对偏差"的 MAE 尺度度量，
    取半（MAD/2）作为保守下界：模型即使完美预测组趋势，组内构型级散布的一半
    尺度仍不可避免地漏进任何仅含组成级信息的误差中。
  - 同时报告 min-vs-rest 中位偏差（target 为组内 min 的视角）作敏感性对照。
路径 b（模型侧方差分解，写死方法）：
  基线 XGBoost 10 种子 OOF 预测 p_{s,i}：
  - 可约估计方差 = mean_i Var_s(p_{s,i})（种子间方差）
  - floor_b = MAE(mean_s p_{s,i}, y)（种子均值预测残差 ≈ 不可约近似）

判读规则（预注册）：
  架构理论收益上限 ceiling = OOF MAE（10 种子均值）− noise floor；
  floor 取路径 a 为头条数字，路径 b 为对照。ceiling ≤ 0.01 eV → "架构收益空间已尽"。
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
from _common import PROBES, RAW_GZ, SEEDS10, load_data, oof_pred, mae

STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}


def parse_prefix(p):
    toks = re.findall(r"([A-Z][a-z]?)(\d+)", p)
    return {el: int(n) for el, n in toks}


def normalize_comp(cd):
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def path_a():
    """原始记录组内散布 → 不可约误差上界（MAE 尺度）。"""
    data = json.loads(gzip.decompress(RAW_GZ.read_bytes()))
    recs = defaultdict(list)
    for k, v in data.items():
        if "0.5H2(g) + * -> H*" not in k:
            continue
        prefix = re.match(r"^([A-Za-z0-9]+)_", k).group(1)
        cd = parse_prefix(prefix)
        comp = normalize_comp(cd)
        structure, _ = STRUCT_RULE[tuple(sorted(cd.values()))]
        recs[(comp, structure)].append(float(v["ref_ads_eng"]))
    mad2, min_rest, n_groups_multi = [], [], 0
    for key, es in recs.items():
        if len(es) < 2:
            continue
        n_groups_multi += 1
        e = np.asarray(es)
        med = np.median(e)
        mad2.append(np.median(np.abs(e - med)) / 2.0)
        rest = np.sort(e)[1:]
        min_rest.append(np.median(rest - e.min()))
    return {
        "n_groups_total": len(recs), "n_groups_multi": n_groups_multi,
        "n_raw_records": int(sum(len(v) for v in recs.values())),
        "floor_a_mad2_median": float(np.median(mad2)),
        "floor_a_mad2_mean": float(np.mean(mad2)),
        "floor_a_min_vs_rest_median": float(np.median(min_rest)),
    }


def path_b(X, y, groups):
    ckpt = PROBES / "p3b_seed_ckpt.json"
    preds = []
    done = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    for seed in SEEDS10:
        if str(seed) in done:
            preds.append(np.array(done[str(seed)]))
            continue
        p, _ = oof_pred(X, y, groups, seed)
        done[str(seed)] = p.tolist()
        ckpt.write_text(json.dumps(done))
        Path("/tmp/p3b_seed_ckpt.json").write_text(json.dumps(done))
        preds.append(p)
        print(f"[P3b] seed={seed} OOF MAE={mae(p, y):.4f}", flush=True)
    P = np.vstack(preds)                       # (10, n)
    per_seed = [mae(p, y) for p in preds]
    mean_pred = P.mean(axis=0)
    return {
        "oof_mae_per_seed": per_seed,
        "oof_mae_10seed_mean": float(np.mean(per_seed)),
        "estim_var_seed_mean": float(P.var(axis=0, ddof=1).mean()),
        "estim_std_seed": float(np.sqrt(P.var(axis=0, ddof=1).mean())),
        "floor_b_meanpred_mae": mae(mean_pred, y),
    }


def main():
    PROBES.mkdir(exist_ok=True)
    df, X, y, groups, feats = load_data()
    a = path_a()
    print("[P3a]", json.dumps(a, indent=2))
    b = path_b(X, y, groups)
    floor = a["floor_a_mad2_median"]
    ceiling_a = b["oof_mae_10seed_mean"] - floor
    ceiling_b = b["oof_mae_10seed_mean"] - b["floor_b_meanpred_mae"]
    result = {"path_a": a, "path_b": b, "floor_headline": floor,
              "ceiling_path_a": ceiling_a, "ceiling_path_b": ceiling_b,
              "ceiling_verdict": ("architecture headroom exhausted"
                                  if ceiling_a <= 0.01 else "headroom remains")}
    (PROBES / "p3_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    md = f"""# P3 噪声地板（Noise Floor）— 误差解剖锚点

> 预注册判读：架构理论收益上限 = OOF MAE − 噪声地板；只诊断，不用于选交付配置。

## 路径 a：原始记录组内散布（不可约误差上界，MAE 尺度）

- 数据源：`data_raw/MamunHighT2019_adsorption.json.gz`，稀释 H*（0.5H₂(g)+\\*→H\\*）原始记录 **{a['n_raw_records']}** 条
- 分组：(规范化 comp × structure)，共 {a['n_groups_total']} 组，其中 n_raw≥2 的多构型组 {a['n_groups_multi']} 个
- 方法（写死）：组内 MAD/2 的中位数。MAD = median|E_i − median(E)|；MAE 的最优常数预测为中位数，
  故 MAD 是组内不可约绝对偏差的 MAE 尺度度量，取半作保守下界。

| 统计量 | 数值 (eV) |
|---|---|
| **floor_a = median(MAD/2)** | **{a['floor_a_mad2_median']:.4f}** |
| mean(MAD/2) | {a['floor_a_mad2_mean']:.4f} |
| min-vs-rest 中位偏差（对照） | {a['floor_a_min_vs_rest_median']:.4f} |

## 路径 b：v3 十种子 OOF 残差方差分解

- 10 种子 OOF MAE：{", ".join(f"{v:.4f}" for v in b['oof_mae_per_seed'])}
- OOF MAE 均值：**{b['oof_mae_10seed_mean']:.4f} eV**
- 种子间方差（可约估计方差）：{b['estim_var_seed_mean']:.6f} eV²（std = {b['estim_std_seed']:.4f} eV）
- **floor_b = 种子均值预测残差 MAE：{b['floor_b_meanpred_mae']:.4f} eV**（不可约近似）

## 两路径对比与架构收益上限

| 路径 | 噪声地板 (eV) | OOF MAE (eV) | 架构理论收益上限 (eV) |
|---|---|---|---|
| a（数据内禀 MAD/2） | {floor:.4f} | {b['oof_mae_10seed_mean']:.4f} | **{ceiling_a:.4f}** |
| b（种子均值残差） | {b['floor_b_meanpred_mae']:.4f} | {b['oof_mae_10seed_mean']:.4f} | {ceiling_b:.4f} |

**结论（预注册口径）**：以路径 a 为头条，噪声地板 = **{floor:.4f} eV**，
当前 v3 XGBoost OOF MAE = {b['oof_mae_10seed_mean']:.4f} eV，
**架构理论收益上限 = {ceiling_a:.4f} eV**（判据 ≤0.01 eV →
"{result['ceiling_verdict']}"）。

注意：floor_a 度量的是"同组成不同吸附构型"的能量散布——它是组成级特征体系下
构型选择不确定性的代理，并非 DFT 数值噪声；作为不可约误差上界属保守估计。
"""
    (PROBES / "p3_noise_floor.md").write_text(md)
    print(json.dumps({k: result[k] for k in
                      ["floor_headline", "ceiling_path_a", "ceiling_path_b",
                       "ceiling_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
