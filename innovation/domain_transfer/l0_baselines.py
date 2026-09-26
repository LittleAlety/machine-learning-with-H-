# -*- coding: utf-8 -*-
"""
Phase 4 Workstream L — L0 论证卫生包（预注册: config/preregistered_phase4.yaml
workstream_L；红线 16/17/20）。

L0-a 四方案同台（同测试面集合、同锚点预算、同折同种子，seeds 同 Phase 3 I）：
  B1 每域均值偏移校正（朴素下限）：基座 XGB 仅在训练折的 非EqV2 行
     （Mamun+CatHub）上拟合；shift = mean(y_anchor − pred_anchor)，
     锚点 = 训练折内 EqV2 行（红线 16：锚点只进训练折）。
  B2 线性偏移校正（朴素上限）：同基座，锚点上最小二乘拟合
     y = a·pred + b（slope+intercept）。
  B3 域指示 + 线性模型（机制对照，无交互）：[X(中位数填补), 域 one-hot]
     → LinearRegression，全体训练折行。
  B4 域指示 GBDT（主方案）：[X, 域 one-hot] → XGB（锁死
     hstar_best_params.json），全体训练折行 —— 即 Phase 3 I 工作流 i3
     domain_indicator 原口径复现（0.1226 eV 基线）。

锚点预算：与 I 工作流既有口径一致 —— pooled KFold(5, shuffle, seed) 下
训练折内的全部 EqV2 行（≈80%×491≈393 行/折），四方案完全同预算同折。

L0-b 协议口径：
  1. 锚点只进训练折（构造保证 + 显式断言索引不相交，红线 16）；
  2. EqV2 测试面集合与 0.347 基线同一（491 行 (src,comp,facet) 键集合
     与 data_processed/extval_metrics.csv 的 ALL=491 面逐键比对，0 失配）；
  3. 三域组成重叠核查（沿用 I 工作流 i1 口径复算）；
  4. 含锚点新口径 lockbox 在任何分析前冻结：splits/lockbox_l0_anchors.json
     （沿用 splits/lockbox_v1.json 组成 + EqV2 测试面键清单），双文件
     SHA256 留证；本脚本先写 lockbox 再跑任何模型。

裁决（预注册）：B4−B2 改善 >0.02 eV 且 bootstrap 10,000 次 95% CI 不含 0
→ 结局 A（"红利在联合映射"）；否则 → 结局 B（"锚点本体论"：跨库上限由
锚点数量决定）。

产物（幂等，重跑覆盖同名文件）：
  innovation/domain_transfer/l0_four_arms.csv   逐 (arm, seed) 折指标
  innovation/domain_transfer/l0_verdict.json    四方案 MAE[CI]+ρ、B4−B2 差与 CI、结局判定
  innovation/domain_transfer/README_L0.md        结果片段（README 归属 L-B 代理）
  splits/lockbox_l0_anchors.json                含锚点口径 lockbox（先于分析冻结）
"""
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DP = ROOT / "data_processed"
SPLITS = ROOT / "splits"
OUT = ROOT / "innovation" / "domain_transfer"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "innovation" / "functional_invariance"))
from run_invariance import load_domains  # noqa: E402  复用 I 工作流三域口径+自检

SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
XGB_PARAMS = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]
EQV2_BASELINE_MAE = 0.347  # data_processed/extval_metrics.csv: ALL 0.3469996, n=491
B4_REFERENCE_MAE = 0.1226  # Phase 3 I i3 domain_indicator pooled OOF
N_BOOT = 10_000            # 预注册锁死
BOOT_SEED = 20260918       # 预注册冻结日派生，锁死
MAIN_DOMAIN = "Mamun_BEEFvdW"
TARGET_DOMAIN = "EqV2_RPBE"


# ------------------------------------------------------------ EqV2 键重建（行级对齐自检）
def load_eq_keys(domains):
    """复刻 I 工作流 EqV2 聚合，返回逐行 (src, comp)；断言与 domains 行级一致。"""
    from run_invariance import normalize_comp
    d361 = pd.read_csv(ROOT / "data_raw" / "eqv2_361.csv").rename(
        columns={"Full_optimized_DFT_H_ads_energy": "E_true",
                 "Surface": "facet", "symbol": "formula"})
    d361["src"] = "eqv2_361_relaxed"
    d500 = pd.read_csv(ROOT / "data_raw" / "eqv2_500.csv").rename(
        columns={"single_point_DFT_H_ads_E": "E_true",
                 "Surface": "facet", "symbol": "formula"})
    d500["src"] = "eqv2_500_sp"
    cols = ["src", "formula", "facet", "E_true"]
    eq = pd.concat([d361[cols], d500[cols]], ignore_index=True)
    eq = eq.dropna(subset=["formula", "E_true"]).reset_index(drop=True)
    eq["comp"] = eq["formula"].map(normalize_comp)
    eq["facet"] = eq["facet"].astype(int)
    eq_agg = (eq.groupby(["src", "comp", "facet"], as_index=False)
              .agg(E_true=("E_true", "min")))
    y_dom = domains[TARGET_DOMAIN]["y"]
    assert len(eq_agg) == len(y_dom) and np.allclose(eq_agg["E_true"], y_dom), \
        "EqV2 键重建与 I 工作流域装配行级不一致，终止"
    return eq_agg


# ------------------------------------------------------------ L0-b(4) lockbox 先冻结
def freeze_lockbox(domains, eq_agg):
    """在任何模型训练前写出含锚点口径 lockbox，并留 SHA256 证据。"""
    keys = sorted(f"{s}|{c}|{f}" for s, c, f in
                  zip(eq_agg["src"], eq_agg["comp"], eq_agg["facet"]))
    key_blob = "\n".join(keys).encode()
    v1_path = SPLITS / "lockbox_v1.json"
    v1_bytes = v1_path.read_bytes()
    doc = {
        "version": "lockbox_l0_anchors_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_any_analysis": True,
        "protocol": {
            "anchors": "训练折内 EqV2 行（pooled KFold(5) 每折 ≈393 行），"
                       "锚点只进训练折，禁止进入任何测试评估面（红线16）",
            "test_face": "EqV2 OOF 491 行，与 v3 零锚点基线 0.347 eV 同一键集合",
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "ci": 0.95},
            "seeds": SEEDS,
        },
        "inherits_lockbox_v1": {
            "path": "splits/lockbox_v1.json",
            "n_lockbox_comps": json.loads(v1_bytes)["n_lockbox_comps"],
            "sha256": hashlib.sha256(v1_bytes).hexdigest(),
        },
        "eqv2_test_face": {
            "n": len(keys),
            "keys_src_pipe_comp_pipe_facet": keys,
            "sha256": hashlib.sha256(key_blob).hexdigest(),
        },
    }
    out_path = SPLITS / "lockbox_l0_anchors.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[lockbox] 已先于分析冻结 → {out_path}  "
          f"eqv2_face_sha256={doc['eqv2_test_face']['sha256'][:16]}…")
    return doc


# ------------------------------------------------------------ L0-b(2/3) 口径核查
def protocol_checks(domains, eq_agg):
    eq = domains[TARGET_DOMAIN]
    # 与 0.347 基线同一测试面：重建 scripts/11 的 491 行键集合逐键比对
    d361 = pd.read_csv(ROOT / "data_raw" / "eqv2_361.csv").rename(
        columns={"Full_optimized_DFT_H_ads_energy": "E_true",
                 "Surface": "facet", "symbol": "formula"})
    d361["src"] = "eqv2_361_relaxed"
    d500 = pd.read_csv(ROOT / "data_raw" / "eqv2_500.csv").rename(
        columns={"single_point_DFT_H_ads_E": "E_true",
                 "Surface": "facet", "symbol": "formula"})
    d500["src"] = "eqv2_500_sp"
    from run_invariance import normalize_comp
    raw = pd.concat([d361[["src", "formula", "facet", "E_true"]],
                     d500[["src", "formula", "facet", "E_true"]]],
                    ignore_index=True).dropna(subset=["formula", "E_true"])
    raw_keys = set(zip(raw["src"], raw["formula"].map(normalize_comp),
                       raw["facet"].astype(int)))
    face_keys = set(zip(eq_agg["src"], eq_agg["comp"], eq_agg["facet"]))
    same_face = raw_keys == face_keys and len(eq["y"]) == 491
    print(f"[L0-b] EqV2 测试面与 0.347 基线同一集合: {same_face} "
          f"(n={len(eq['y'])}, 键交集={len(raw_keys & face_keys)})")
    assert same_face, "EqV2 测试面与 0.347 基线集合不一致，终止"

    # 三域组成重叠核查（I 工作流 i1 口径）
    comp_sets = {k: set(v["comp"]) for k, v in domains.items()}
    names = list(domains)
    overlap = {"triple": len(set.intersection(*comp_sets.values()))}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap[f"{names[i]}∩{names[j]}"] = len(comp_sets[names[i]] &
                                                    comp_sets[names[j]])
    print(f"[L0-b] 三域组成重叠: {overlap}")
    return {"eqv2_face_matches_0347_baseline": bool(same_face),
            "n_eqv2_face": int(len(eq["y"])), "comp_overlap": overlap}


# ------------------------------------------------------------ L0-a 四方案
def run_four_arms(domains, feat_cols):
    names = list(domains)
    X = pd.concat([domains[k]["X"] for k in names], ignore_index=True)
    y = np.concatenate([domains[k]["y"] for k in names])
    dom = np.concatenate([[k] * len(domains[k]["y"]) for k in names])
    is_eq = dom == TARGET_DOMAIN
    eq_idx = np.where(is_eq)[0]
    X_np = X[feat_cols].values.astype(float)
    dummies = pd.get_dummies(dom, drop_first=True).values.astype(float)
    X_ind = np.hstack([X_np, dummies])

    arms = ["B1_mean_shift", "B2_linear_shift", "B3_domain_linear", "B4_domain_gbdt"]
    rows, oof_store = [], {a: np.zeros((len(SEEDS), is_eq.sum())) for a in arms}
    for si, seed in enumerate(SEEDS):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in kf.split(X_np):
            te_mask = is_eq[te]
            te_eq = te[te_mask]              # 保持 te 原顺序
            tr_eq = tr[is_eq[tr]]
            tr_noneq = tr[~is_eq[tr]]
            # 红线 16 显式断言：锚点（训练折 EqV2 行）与任何测试面不相交
            assert len(np.intersect1d(tr_eq, te)) == 0
            assert len(te_eq) > 0

            # B1/B2 共享基座：仅训练折非 EqV2 行（Mamun+CatHub）
            base = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
            base.fit(X_np[tr_noneq], y[tr_noneq])
            pred_tr_eq = base.predict(X_np[tr_eq])   # 锚点预测（仅训练折）
            pred_te = base.predict(X_np[te])

            # B1 每域均值偏移
            shift = float(np.mean(y[tr_eq] - pred_tr_eq))
            # B2 线性偏移 slope+intercept（锚点最小二乘）
            a, b = np.polyfit(pred_tr_eq, y[tr_eq], 1)

            # B3 域指示+线性（无交互，中位数填补仅训练折拟合）
            imp = SimpleImputer(strategy="median")
            Xtr3 = imp.fit_transform(X_ind[tr])
            lin = LinearRegression().fit(Xtr3, y[tr])
            pred3 = lin.predict(imp.transform(X_ind[te]))

            # B4 域指示 GBDT（= I 工作流 i3 domain_indicator 原口径）
            gbdt = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
            gbdt.fit(X_ind[tr], y[tr])
            pred4 = gbdt.predict(X_ind[te])

            preds = {"B1_mean_shift": pred_te + shift,
                     "B2_linear_shift": a * pred_te + b,
                     "B3_domain_linear": pred3,
                     "B4_domain_gbdt": pred4}
            pos = np.searchsorted(eq_idx, te_eq)
            for arm in arms:
                oof_store[arm][si, pos] = preds[arm][te_mask]
        for arm in arms:
            oof = oof_store[arm][si]
            mae = mean_absolute_error(y[eq_idx], oof)
            rho = float(spearmanr(y[eq_idx], oof)[0])
            rows.append({"arm": arm, "seed": seed, "n_eqv2_oof": int(is_eq.sum()),
                         "EqV2_MAE": mae, "EqV2_spearman_rho": rho})
            print(f"[L0-a] {arm:16s} seed={seed:5d}  MAE={mae:.4f}  ρ={rho:.3f}")
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "l0_four_arms.csv", index=False)
    return met, oof_store, y[eq_idx]


# ------------------------------------------------------------ bootstrap + 裁决
def verdict(met, oof_store, y_eq, lockbox_doc, checks):
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y_eq)
    arms = ["B1_mean_shift", "B2_linear_shift", "B3_domain_linear", "B4_domain_gbdt"]
    # 10 种子逐行平均 OOF → 单一人均预测面，bootstrap 逐行残差
    mean_pred = {a: oof_store[a].mean(axis=0) for a in arms}
    err = {a: np.abs(y_eq - mean_pred[a]) for a in arms}
    boot = {a: np.empty(N_BOOT) for a in arms}
    boot_diff = np.empty(N_BOOT)  # MAE_B2 − MAE_B4（B4 改善为正）
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for a in arms:
            boot[a][i] = err[a][idx].mean()
        boot_diff[i] = err["B2_linear_shift"][idx].mean() - \
            err["B4_domain_gbdt"][idx].mean()

    summ = []
    for a in arms:
        lo, hi = np.percentile(boot[a], [2.5, 97.5])
        summ.append({"arm": a,
                     "EqV2_MAE": float(err[a].mean()),
                     "MAE_CI95": [float(lo), float(hi)],
                     "spearman_rho": float(spearmanr(y_eq, mean_pred[a])[0]),
                     "seed_MAE_mean": float(met[met["arm"] == a]["EqV2_MAE"].mean()),
                     "seed_MAE_sd": float(met[met["arm"] == a]["EqV2_MAE"].std())})
    d = float(err["B2_linear_shift"].mean() - err["B4_domain_gbdt"].mean())
    dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
    outcome = "A" if (d > 0.02 and dlo > 0) else "B"
    doc = {
        "preregistered_rule": ("B4−B2 改善 >0.02 eV 且 bootstrap 95% CI 不含 0 "
                               "→ 结局A（红利在联合映射）；否则结局B（锚点本体论）"),
        "outcome": outcome,
        "outcome_interpretation": {
            "A": "红利在联合映射：域指示×描述符的非线性交互超越线性偏移",
            "B": "锚点本体论：跨库上限由锚点数量决定，GBDT 相对线性偏移无显著红利"}[outcome],
        "B4_minus_B2": {"improvement_eV": d, "CI95": [float(dlo), float(dhi)],
                        "passes_gt_0.02": bool(d > 0.02),
                        "ci_excludes_0": bool(dlo > 0)},
        "arms_summary": summ,
        "baselines": {"v3_zero_anchor_EqV2_MAE_eV": EQV2_BASELINE_MAE,
                      "B4_reference_phase3I_MAE_eV": B4_REFERENCE_MAE},
        "protocol_L0b": {
            "anchors_train_fold_only": True,
            "anchor_budget": "pooled KFold(5) 训练折内全部 EqV2 行（≈393/折），四方案同预算",
            **checks,
            "lockbox": {"path": "splits/lockbox_l0_anchors.json",
                        "frozen_before_any_analysis": True,
                        "eqv2_test_face_sha256":
                            lockbox_doc["eqv2_test_face"]["sha256"],
                        "inherits_lockbox_v1_sha256":
                            lockbox_doc["inherits_lockbox_v1"]["sha256"]},
        },
        "config_locked": {"seeds": SEEDS, "n_bootstrap": N_BOOT,
                          "bootstrap_seed": BOOT_SEED, "xgb_params": XGB_PARAMS,
                          "cv": "KFold(5, shuffle, seed) 同折同种子"},
    }
    (OUT / "l0_verdict.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[裁决] B4−B2 = {d:.4f} eV, CI95=[{dlo:.4f}, {dhi:.4f}] → 结局 {outcome}")
    return doc


def write_readme(doc):
    s = doc["arms_summary"]
    lines = ["# Phase 4 Workstream L — L0 论证卫生包（结果片段）", "",
             "预注册: `config/preregistered_phase4.yaml` workstream_L；红线 16/17/20。",
             "测试面：EqV2 491 行 OOF（与 0.347 eV 零锚点基线同一键集合，已逐键核查）；",
             "锚点只进训练折（红线 16）；lockbox `splits/lockbox_l0_anchors.json` "
             "先于任何分析冻结（SHA256 留证）。", "",
             "| 方案 | EqV2 MAE (eV) | 95% CI (bootstrap 10,000) | Spearman ρ |",
             "|---|---|---|---|"]
    for r in s:
        lines.append(f"| {r['arm']} | {r['EqV2_MAE']:.4f} | "
                     f"[{r['MAE_CI95'][0]:.4f}, {r['MAE_CI95'][1]:.4f}] | "
                     f"{r['spearman_rho']:.3f} |")
    d = doc["B4_minus_B2"]
    lines += ["",
              f"**B4−B2 改善 = {d['improvement_eV']:.4f} eV，"
              f"95% CI [{d['CI95'][0]:.4f}, {d['CI95'][1]:.4f}]** → "
              f"**结局 {doc['outcome']}**：{doc['outcome_interpretation']}",
              "",
              f"参照：v3 零锚点跨库基线 0.347 eV；B4 对 Phase 3 I 既有 0.1226 eV "
              f"口径复现值 {s[3]['EqV2_MAE']:.4f} eV。"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    domains, feat_cols = load_domains()
    eq_agg = load_eq_keys(domains)
    lockbox_doc = freeze_lockbox(domains, eq_agg)   # 先于任何分析
    checks = protocol_checks(domains, eq_agg)
    met, oof_store, y_eq = run_four_arms(domains, feat_cols)
    doc = verdict(met, oof_store, y_eq, lockbox_doc, checks)
    write_readme(doc)
    print("[写出] innovation/domain_transfer/ 全部产物完成")


if __name__ == "__main__":
    main()
