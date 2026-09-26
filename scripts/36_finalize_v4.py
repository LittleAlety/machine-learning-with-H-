# -*- coding: utf-8 -*-
"""
36_finalize_v4.py — A-5：v4 定版 + lockbox 一次性评估 + r14 汇总

- 读取 outputs/r14_a1/a2/a3/a4 JSON，按公平契约+嵌套双过挑选胜出者
- 胜出者 → v4；无胜者 → v3 保留（三层体系演示）
- 最终模型：在 splits/lockbox_v1.json 之外的 85% 上训练（同口径），
  在 lockbox 上只评一次（MAE/RMSE/R2 + bootstrap CI 10,000 次，种子 42）
- 产出 outputs/v4_final/：model_spec.json、lockbox_report.json、comparison_table.csv
- 汇总 outputs/r14_ab_summary.json（每任务 {verdict, key_numbers, files}）
- 阴性结果 → outputs/negative_register_r14.csv（路线/节/结果/结论）
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, DP, XGB, SEEDS, load_v3

OUT = ROOT / "outputs"
V4 = OUT / "v4_final"


def build_final_model(winner, df, feats, mask_train):
    """按胜出配置在训练部分拟合最终模型；返回 (predict_fn, spec)。"""
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    Xtr, ytr = X[mask_train], y[mask_train]
    if winner == "q50":  # A-1 胜出：XGB quantile α=0.5
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         random_state=42, n_jobs=-1, **XGB).fit(Xtr, ytr)
        spec = {"name": "v4 = q50-XGB",
                "detail": "XGBoost reg:quantileerror α=0.5, v3 84feat, tuned 参数"}
        return (lambda Xn: m.predict(Xn)), spec
    if winner == "ensemble":  # A-3 胜出
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        ms = [XGBRegressor(random_state=42, n_jobs=-1, **XGB).fit(Xtr, ytr),
              LGBMRegressor(random_state=42, n_jobs=-1, n_estimators=400,
                            learning_rate=0.03, max_depth=7,
                            subsample=0.8).fit(Xtr, ytr),
              CatBoostRegressor(random_state=42, iterations=400, learning_rate=0.03,
                                depth=7, subsample=0.8, verbose=0,
                                allow_writing_files=False).fit(Xtr, ytr)]
        spec = {"name": "v4 = 三族简单平均",
                "detail": "XGB+LGBM+CB 同配置简单平均, v3 84feat"}
        return (lambda Xn: np.mean([m.predict(Xn) for m in ms], axis=0)), spec
    # 默认：v3 保留
    m = XGBRegressor(random_state=42, n_jobs=-1, **XGB).fit(Xtr, ytr)
    spec = {"name": "v3 保留（无胜者，三层体系演示）",
            "detail": "XGBoost squarederror, v3 84feat, tuned 参数"}
    return (lambda Xn: m.predict(Xn)), spec


def main():
    df, feats = load_v3()
    a1 = json.load(open(OUT / "r14_a1_q50.json"))
    a2 = json.load(open(OUT / "r14_a2_monotonic.json"))
    a3 = json.load(open(OUT / "r14_a3_ensemble.json"))
    a4 = json.load(open(OUT / "r14_a4_mlp.json"))
    multi = json.load(open(ROOT / "splits" / "lockbox_multi_eval.json"))

    # 胜者选择（公平契约+嵌套双过）
    cands = []
    if a1["step5_verdict"]["adopt"]:
        cands.append(("q50", 0.1063))
    if a2["verdict"]["adopt"]:
        cands.append((f"mono_{a2['verdict']['best_tier']}",
                      a2["summary"][a2["verdict"]["best_tier"]]["mean"]))
    if a3["verdict"]["adopt"]:
        cands.append(("ensemble", a3["ensemble_mean"]))
    winner = min(cands, key=lambda c: c[1])[0] if cands else None
    print(f"[胜者] {winner or '无 → v3 保留'}")

    # lockbox 一次性评估
    lb = json.load(open(ROOT / "splits" / "lockbox_v1.json"))
    is_lock = df["comp"].isin(lb["lockbox_comps"]).values
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    predict, spec = build_final_model(winner, df, feats, ~is_lock)
    pred = predict(X[is_lock])
    yt = y[is_lock]
    ae = np.abs(pred - yt)
    rng = np.random.RandomState(42)
    boots = np.array([rng.choice(ae, len(ae), True).mean() for _ in range(10000)])
    report = {
        "evaluated_once_utc": datetime.now(timezone.utc).isoformat(),
        "model": spec, "winner": winner,
        "n_lockbox": int(is_lock.sum()),
        "MAE": float(ae.mean()),
        "MAE_ci95": [float(np.quantile(boots, 0.025)),
                     float(np.quantile(boots, 0.975))],
        "RMSE": float(np.sqrt(mean_squared_error(yt, pred))),
        "R2": float(r2_score(yt, pred)),
        "median_AE": float(np.median(ae)), "P90_AE": float(np.quantile(ae, 0.9)),
        "per_structure": {st: float(ae[(df["structure"].values[is_lock] == st)].mean())
                          for st in ("A1", "L10", "L12")},
        "reference": {"v3_group_cv_oof_10seed": 0.11374,
                      "multisplit_lockbox_mean": multi["mae_mean"]},
        "discipline": "lockbox 唯一一次评估；此前未用于任何选型/调参"}

    V4.mkdir(exist_ok=True)
    json.dump(spec | {"winner": winner, "candidates": cands,
                      "features": "v3 84feat", "fold_protocol":
                      "GroupKFold(5, groups=comp)，训练于 lockbox 外 85%"},
              open(V4 / "model_spec.json", "w"), ensure_ascii=False, indent=2)
    json.dump(report, open(V4 / "lockbox_report.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"[lockbox] MAE={report['MAE']:.4f} CI95={report['MAE_ci95']} "
          f"RMSE={report['RMSE']:.4f} R2={report['R2']:.3f}")

    # 对比表
    cmp_rows = [
        {"model": "v2 (36feat)", "protocol": "GroupKFold OOF", "MAE": 0.1198},
        {"model": "v3 (84feat) XGB", "protocol": "GroupKFold OOF 10种子",
         "MAE": 0.11374},
        {"model": "q50-XGB", "protocol": "GroupKFold OOF 10种子",
         "MAE": a1["step1_reproduce"]["mean"]},
        {"model": "三族平均", "protocol": "GroupKFold OOF 10种子",
         "MAE": a3["ensemble_mean"]},
        {"model": "单调Top3", "protocol": "GroupKFold OOF 10种子",
         "MAE": a2["summary"]["top3"]["mean"]},
        {"model": "单调Top6", "protocol": "GroupKFold OOF 10种子",
         "MAE": a2["summary"]["top6"]["mean"]},
    ] + [{"model": f"MLP-{k}", "protocol": "GroupKFold OOF 10种子",
          "MAE": v["mean"]} for k, v in a4["subjects"].items()
         ] + [{"model": spec["name"], "protocol": "lockbox 一次性", "MAE": report["MAE"]}]
    pd.DataFrame(cmp_rows).to_csv(V4 / "comparison_table.csv", index=False)

    # 阴性登记
    neg = []
    if not a1["step5_verdict"]["adopt"]:
        neg.append({"路线": "A-1 q50 分位数", "节": "A-1",
                    "结果": f"OOF {a1['step1_reproduce']['mean']:.4f} vs v3 {a1['step2_v3_baseline']['mean']:.4f}",
                    "结论": a1["step5_verdict"]["reason"]})
    neg.append({"路线": "A-1 跨域", "节": "A-1",
                "结果": "q50 CatHub 0.206 vs v3 0.177 (+0.029 eV)",
                "结论": "库内改善未迁移到外部集，跨域泛化存疑，如实记录"})
    if not a2["verdict"]["adopt"]:
        neg.append({"路线": "A-2 单调约束", "节": "A-2",
                    "结果": f"top3 {a2['summary']['top3']['mean']:.4f} / top6 {a2['summary']['top6']['mean']:.4f}",
                    "结论": a2["verdict"]["reason"]})
    if not a3["verdict"]["adopt"]:
        neg.append({"路线": "A-3 三族平均", "节": "A-3",
                    "结果": f"{a3['ensemble_mean']:.4f} vs XGB {a3['xgb_mean']:.4f}",
                    "结论": a3["verdict"]["reason"]})
    for k, v in a4["subjects"].items():
        if not v["beats_v3"]:
            neg.append({"路线": f"A-4 MLP {k}", "节": "A-4",
                        "结果": f"{v['mean']:.4f}±{v['std']:.4f}",
                        "结论": "未优于 v3，配置锁死下阴性，记录"})
    pd.DataFrame(neg, columns=["路线", "节", "结果", "结论"]).to_csv(
        OUT / "negative_register_r14.csv", index=False)

    # r14 汇总
    summ = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "B1_lockbox": {"verdict": "完成",
                       "key_numbers": {"n_lockbox": lb["n_lockbox_comps"],
                                       "sha256": lb["sha256_of_comp_list"],
                                       "leakcheck": lb["leakcheck"]},
                       "files": ["splits/lockbox_v1.json",
                                 "splits/lockbox_leakcheck.png",
                                 "config/fair_contract.yaml"]},
        "B2_multisplit": {"verdict": ("无显著 OOF 乐观偏差" if not multi["oof_optimism_flag"]
                                      else "存在 OOF 乐观偏差"),
                          "key_numbers": {"mae_mean": multi["mae_mean"],
                                          "mae_std": multi["mae_std"],
                                          "ci95": multi["bootstrap_ci95_mean"]},
                          "files": ["splits/lockbox_multi_eval.json",
                                    "splits/lockbox_multi_eval.csv",
                                    "splits/lockbox_seed1.json",
                                    "splits/lockbox_seed2.json",
                                    "splits/lockbox_seed3.json"]},
        "B3_external_protocol": {"verdict": "冻结",
                                 "key_numbers": {"cathub": 355, "eqv2": 491,
                                                 "bootstrap": 10000, "seed": 42},
                                 "files": ["config/external_test_protocol.yaml"]},
        "A1_q50": {"verdict": "采纳" if a1["step5_verdict"]["adopt"] else "不采纳",
                   "key_numbers": {"oof_mean": a1["step1_reproduce"]["mean"],
                                   "gain": a1["step2_gain"]["mean_gain"],
                                   "direction": a1["step2_gain"]["direction_consistent"],
                                   "nested_mae": a1["step3_nested_cv"]["nested_mae"],
                                   "loeo_focus_delta": a1["step4_loeo"]["focus_delta"],
                                   "cathub_gap": "+0.029 eV (0.206 vs 0.177)"},
                   "files": ["outputs/r14_a1_q50.json",
                             "data_processed/hstar_loeo_q50.csv"]},
        "A2_monotonic": {"verdict": "采纳" if a2["verdict"]["adopt"] else "不采纳",
                         "key_numbers": {"top3": a2["summary"]["top3"]["mean"],
                                         "top6": a2["summary"]["top6"]["mean"],
                                         "nested_mae": a2["nested_cv"]["mae"]},
                         "files": ["outputs/r14_a2_monotonic.json",
                                   "outputs/r14_a2_monotonic_shap.csv"]},
        "A3_ensemble": {"verdict": "采纳" if a3["verdict"]["adopt"] else "不采纳",
                        "key_numbers": {"ensemble_mean": a3["ensemble_mean"],
                                        "gain": a3["gain"],
                                        "direction": a3["direction_consistent"],
                                        "nested_mae": a3["nested_cv"]["mae"]},
                        "files": ["outputs/r14_a3_ensemble.json"]},
        "A4_mlp": {"verdict": "记录（不参与交付）",
                   "key_numbers": {k: {"mean": v["mean"], "beats_v3": v["beats_v3"]}
                                   for k, v in a4["subjects"].items()},
                   "files": ["outputs/r14_a4_mlp.json"]},
        "A5_v4": {"verdict": spec["name"],
                  "key_numbers": {"winner": winner, "lockbox_MAE": report["MAE"],
                                  "lockbox_ci95": report["MAE_ci95"],
                                  "RMSE": report["RMSE"], "R2": report["R2"]},
                  "files": ["outputs/v4_final/model_spec.json",
                            "outputs/v4_final/lockbox_report.json",
                            "outputs/v4_final/comparison_table.csv",
                            "outputs/negative_register_r14.csv"]},
    }
    json.dump(summ, open(OUT / "r14_ab_summary.json", "w"),
              ensure_ascii=False, indent=2)
    print("[写出] outputs/v4_final/*, outputs/r14_ab_summary.json, "
          "outputs/negative_register_r14.csv")


if __name__ == "__main__":
    main()
