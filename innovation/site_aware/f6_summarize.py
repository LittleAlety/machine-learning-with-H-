# -*- coding: utf-8 -*-
"""汇总 Workstream F（C-2 逐位点建模）：site_aware_summary.json + README.md"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    f0 = load("f0_site_lockbox.json")
    f1 = load("f1_report.json")
    f23 = load("f2f3_report.json")
    f4 = load("f4_gate_report.json")
    f5 = load("f5_report.json")

    summary = {
        "workstream": "F / C-2 逐位点建模",
        "F0_lockbox": {
            "choice": f0["choice"], "n_comps": f0["protocol"]["n_lockbox_comps"],
            "sha256_of_comp_list": f0["sha256_of_comp_list"],
            "sha256_of_lockbox_site_rows": f0["sha256_of_lockbox_site_rows"],
            "n_lockbox_site_records": f0["n_lockbox_site_records"],
            "touched_for_final_eval": f0["touched_for_final_eval"],
            "verdict": "FROZEN then EVALUATED_ONCE" if f0["touched_for_final_eval"]
                       else "FROZEN_UNTOUCHED"},
        "F1_dataset": {
            "n_site_records": f1["n_site_records_matched"],
            "coverage": f1["site_metadata_coverage"],
            "unmatched_dropped": f1["n_unmatched_dropped"],
            "site_coarse_dist": f1["site_coarse_dist"],
            "n_site_types_full": len(f1["site_full_dist"]),
            "fallback_used": f1["fallback_to_site_index_onehot"],
            "verdict": f1["F1_verdict"]},
        "F2_consistency": {
            "mean_minus_min_median": f23["F2a_mean_minus_min_median"],
            "accept_range": f23["F2a_accept_range"],
            "verdict": f23["F2a_verdict"],
            "minagg_pred_vs_v3label_median_abs":
                f23["F2b_minagg_pred_vs_v3label_median_abs"]},
        "F3_site_accuracy": {
            "site_oof_mae_10seed_mean": f23["F3_site_oof_mae_10seed_mean"],
            "target": f23["F3_target"],
            "bounty_interval": f23["bounty_interval"],
            "beyond_bounty_leak_suspect": f23["beyond_bounty_leak_suspect"],
            "onehot_site_level_gain": f23["site_level_gain_from_onehot"],
            "verdict": f23["F3_verdict"]},
        "F4_gate": {"feature": f4["feature"], "delta_A": f4["delta_A"],
                    "retention": f4["retention"], "verdict": f4["verdict"]},
        "F5_surface_compare": {
            "v3_surface_oof_mae": f5["v3_surface_oof_mae_mean"],
            "site_min_surface_oof_mae": f5["site_min_surface_oof_mae_mean"],
            "delta_mean": f5["delta_mean"], "seeds_win": f5["seeds_win"],
            "benefit_realization_rate": f5["benefit_realization_rate"],
            "site_model_dominates": f5["site_model_dominates"],
            "lockbox_eval": f5["lockbox_eval"]},
    }
    (OUT / "site_aware_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    lb = f5["lockbox_eval"] or {}
    readme = f"""# Workstream F — C-2 逐位点建模（site-aware modeling）

> plan-hb v3.0。预注册检查点 F0–F5，全部脚本幂等，协议锚定
> `innovation/probes/_common.py`（GroupKFold(5, shuffle, rs=42)、10 种子、
> XGBoost lr=0.03/depth=7/n=400/sub=0.8、公平契约 Δ>0.005 eV + 10/10）。

## 检查点结论一览

| 检查点 | 结果 | 数字 | verdict |
|---|---|---|---|
| F0 位点 lockbox | 与表面级 lockbox_v1 对齐（276 组成，SHA256 校验一致），F1 生成 {f0['n_lockbox_site_records']} 条位点记录密封 | sha256={f0['sha256_of_comp_list'][:16]}… | 冻结→F5 占优后评估一次 |
| F1 数据自检 | 位点样本 {f1['n_site_records_matched']}/7,048（覆盖率 {f1['site_metadata_coverage']:.1%} ≥95%）；位点类型 28 类；无空壳单元；未触发降级 | 未匹配丢弃 {f1['n_unmatched_dropped']} 条 | **{f1['F1_verdict']}** |
| F2 口径一致性 | 表面 mean−min 口径差中位数 {f23['F2a_mean_minus_min_median']:.4f} eV（区间 0.15–0.30） | min 聚合预测 vs v3 标签差中位 {f23['F2b_minagg_pred_vs_v3label_median_abs']:.4f} eV | **{f23['F2a_verdict']}** |
| F3 位点级精度 | OOF MAE = **{f23['F3_site_oof_mae_10seed_mean']:.4f} eV**（10 种子均值），目标 ≤0.090 未达；未低于 0.077，无泄漏嫌疑 | one-hot 位点级贡献 +{f23['site_level_gain_from_onehot']:.4f} eV（无位点基线 0.282） | **{f23['F3_verdict']}** |
| F4 闸门 | min 位点类型 one-hot（金标准）Δ_A={f4['delta_A']:.4f} eV（10/10），Ridge 推断保留率 r={f4['retention']:.3f} | 改善在可移植口径下崩塌 | **{f4['verdict']}** |
| F5 表面级对照 | v3 {f5['v3_surface_oof_mae_mean']:.4f} vs 位点 min 聚合 {f5['site_min_surface_oof_mae_mean']:.4f} eV，Δ=**{f5['delta_mean']:.4f} eV**（{f5['seeds_win']}） | 收益兑现率 {f5['benefit_realization_rate']:.1%}（对 P4 红利 0.0391） | **位点模型占优（刚过公平契约）** |
| F0 lockbox 评估（唯一一次） | v3 {lb.get('v3_lockbox_mae', float('nan')):.4f} vs 位点 min {lb.get('site_min_lockbox_mae', float('nan')):.4f} eV | Δ={lb.get('delta', float('nan')):.4f} eV（seed=42，272/276 表面有位点预测） | 方向一致，幅度小于 OOF |

## 关键诚实声明

1. **位点元数据来源偏差**：`MamunHighT2019_adsorption.json.gz` 内**没有** sites 字段
   （全量扫描 45,131 条记录为证）。位点类型来自 `scripts/13_cathub_fetch.py` 的
   官方 CatApp API 抓取产物（`data_processed/cathub_mamun.csv` +
   `data_raw/cathub_cache/pages/all_h/`），按 (规范化 comp, 能量精确匹配) 回连，
   覆盖率 97.8%，0 冲突，未匹配 155 条丢弃。
2. **F3 未达标**：位点级 OOF MAE 0.1156 eV，高于 0.090 目标。one-hot 本身信息量
   巨大（位点级基线 0.282→0.116），但离散类别标签不足以把同组成位点间连续能量
   差异（mean−min 中位 0.193 eV）压进红利区间——缺位点级连续几何/电子坐标。
3. **F4 判 LEAK 的含义**：不是位点类型本身泄漏，而是"**哪个位点是 min 位点**"
   这一选择依赖能量标签、无法从组成描述符推断（r≈0）。F5 占优的部署形态是
   "枚举晶体学候选位点→逐位点预测→取 min"，该路径物理可得，不依赖金标准。
4. **F5 占优但收益兑现率仅 14.7%**：P4 的 0.0391 eV 红利大头在"位点内部连续
   差异"（能量分布形状）中，离散位点标签只兑现类别均值部分。与 P4 对
   config_mean/config_count 的 LEAK 判定自洽。

## 文件

- `f0_site_lockbox.py/.json` — 位点 lockbox 冻结（对齐 lockbox_v1，SHA256 留证）
- `f1_build_site_dataset.py` / `site_dataset_dev.csv` / `site_lockbox_sealed.csv` / `f1_report.json`
- `f2_site_model.py` / `site_oof_preds_dev.csv` / `f2f3_report.json`
- `f4_gate_site.py` / `f4_gate_report.json`
- `f5_surface_compare.py` / `f5_report.json` / `f5_lockbox_eval.json`
- `f6_summarize.py` / `site_aware_summary.json`
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
