# -*- coding: utf-8 -*-
"""汇总 Workstream P：probes_summary.json + README.md（人读版）。
读取 P1–P4 已存盘 verdict/result 文件（幂等，不重新训练）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROBES


def main():
    p1 = json.loads((PROBES / "p1_verdict.json").read_text())
    p2 = json.loads((PROBES / "p2_verdict.json").read_text())
    p3 = json.loads((PROBES / "p3_results.json").read_text())
    p4 = json.loads((PROBES / "p4_results.json").read_text())

    summary = {
        "p1": {
            "saturation_verdict": p1["saturation_verdict"],
            "gbdt_highcap_oof_span": p1["gbdt_highcap_oof_span"],
            "gbdt_best": p1["gbdt_best"],
            "mlp_best_oof_by_width": p1["mlp_best_oof_by_width"],
            "mlp_a4_deepdive_triggered": p1["mlp_a4_deepdive_triggered"],
        },
        "p2": {
            "slope_tail_eV_per_doubling": p2["slope_tail_eV_per doubling"],
            "verdict": p2["verdict"],
            "mean_oof_by_fraction": p2["mean_oof_by_fraction"],
        },
        "p3": {
            "floor_a": p3["floor_headline"],
            "floor_b": p3["path_b"]["floor_b_meanpred_mae"],
            "oof_mae_10seed_mean": p3["path_b"]["oof_mae_10seed_mean"],
            "ceiling": p3["ceiling_path_a"],
            "ceiling_path_b": p3["ceiling_path_b"],
            "verdict": p3["ceiling_verdict"],
        },
        "p4": {
            "deltas": {r["feature"]: {"delta_A": r["delta_A"],
                                      "retention": r["retention"],
                                      "verdict": r["verdict"]}
                       for r in p4["reports"]},
            "capacity_side_gain": p4["capacity_side_gain"],
            "info_side_max_deltaA": p4["info_side_max_deltaA"],
            "verdicts": p4["verdicts"],
        },
        "discipline": "probes are diagnostic only; NOT used to select delivery config",
    }
    (PROBES / "probes_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    md = f"""# Workstream P：误差解剖探针（P1–P4）人读版结论

> 纪律声明：所有探针结果仅用于**诊断**，未用于选择交付配置；判读规则均预注册于各脚本头部。
> 折协议 GroupKFold(5, shuffle=True, random_state=42) 按规范化组成分组；基线 v3 XGBoost
> (lr=0.03, depth=7, n_est=400, subsample=0.8)。

## P1 容量扫描（`p1_capacity_sweep.py` → `p1_capacity_curves.png`）

- GBDT 高容量区（depth≥7 且 n_est≥400）OOF MAE 极差 = **{p1['gbdt_highcap_oof_span']:.4f} eV**
  （判据 ≤ ±0.005）→ 判定：**{p1['saturation_verdict']}**
- GBDT 最优点：depth={p1['gbdt_best']['max_depth']:.0f}, n_est={p1['gbdt_best']['n_estimators']:.0f},
  OOF MAE = {p1['gbdt_best']['oof_mae']:.4f} eV
- MLP 各宽度最优 OOF MAE：{ {k: round(v, 4) for k, v in p1['mlp_best_oof_by_width'].items()} }；
  触发 A-4 深挖：**{p1['mlp_a4_deepdive_triggered']}**（只记录，不行动）

## P2 学习曲线（`p2_learning_curve.py` → `p2_learning_curve.png`）

- 25/50/75/100% 组级子采样 × 10 种子；均值 OOF MAE =
  { {k: round(v, 4) for k, v in p2['mean_oof_by_fraction'].items()} } eV
- 末端斜率 = **{p2['slope_tail_eV_per doubling']:.4f} eV/倍数据**（判据 <0.01）
  → 判定：**{p2['verdict']}**

## P3 噪声地板（`p3_noise_floor.py` → `p3_noise_floor.md`）——锚点

- 路径 a（原始记录 (comp×structure) 组内 MAD/2 中位数，{p3['path_a']['n_raw_records']} 条记录）：
  **floor_a = {p3['floor_headline']:.4f} eV**
  （对照：min-vs-rest 中位偏差 {p3['path_a']['floor_a_min_vs_rest_median']:.4f} eV）
- 路径 b（10 种子 OOF 方差分解）：种子均值预测残差 **floor_b = {p3['path_b']['floor_b_meanpred_mae']:.4f} eV**；
  种子间 std = {p3['path_b']['estim_std_seed']:.4f} eV（可约估计方差）
- 10 种子 OOF MAE = **{p3['path_b']['oof_mae_10seed_mean']:.4f} eV**
- **架构理论收益上限 = OOF MAE − floor_a = {p3['ceiling_path_a']:.4f} eV**
  （路径 b 口径 {p3['ceiling_path_b']:.4f} eV）→ **{p3['ceiling_verdict']}**

## P4 信息天花板 oracle（`p4_info_ceiling.py` → `p4_oracle_bar.png`）

| 作弊特征 | Δ_A (eV) | 保留率 r | 闸门判定 |
|---|---|---|---|
"""
    for r in p4["reports"]:
        md += (f"| {r['feature']} | {r['delta_A']:.4f} | {r['retention']:.3f} "
               f"| **{r['verdict']}** |\n")
    md += f"""
- 信息侧最大 Δ_A = **{p4['info_side_max_deltaA']:.4f} eV** vs 容量侧最大改善（P1 GBDT 段）
  = **{p4['capacity_side_gain']:.4f} eV** —— 差距即"缺信息不缺容量"的直接量化
- 预期全判 LEAK（反向验证闸门灵敏度），实际判定见上表，边界结果如实记录

## 组合图

`p_error_anatomy.png`：四面板（A 容量曲线 / B 学习曲线 / C 噪声地板分解 / D 信息 vs 容量），
300 dpi 暖色系，入正文用。

## 复现

```
cd innovation/probes
/usr/local/bin/python3 p1_capacity_sweep.py   # GBDT+MLP 网格（约数十分钟）
/usr/local/bin/python3 p2_learning_curve.py
/usr/local/bin/python3 p3_noise_floor.py
/usr/local/bin/python3 p4_info_ceiling.py     # 依赖 P1 verdict
/usr/local/bin/python3 p5_combo_figure.py
/usr/local/bin/python3 build_summary.py
```
"""
    (PROBES / "README.md").write_text(md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
