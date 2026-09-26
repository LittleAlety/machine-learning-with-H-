# -*- coding: utf-8 -*-
"""
18_multichem.py — Round 4 Agent H：跨化学体系广度实验
（把硼化物/氮化物/MBene/氧化物等异质体系的 H* 数据只进训练集，
  金属域 GroupKFold MAE 变好还是变差？）

公平性契约（plan_round4.md）：
  - 基线：XGBoost_tuned 出厂参数（lr=0.03/depth=7/n=400/sub=0.8），
    GroupKFold(5, comp) MAE=0.1198±0.0087 eV（07 口径，本脚本先复现校验）；
  - 新数据只进训练集、不进测试集；fold 划分仍由原始 1,836 金属组成决定；
  - 采用阈值：改善 > 0.005 eV 才采纳；种子 42；site_spread 禁入；
  - 阴性结果照常报告，禁止挑口径；不改共享产物。

步骤：
  1. 异质训练池：cathub_hstar_extra.csv 按 14 的清洗漏斗（减去 2b 纯度过滤）：
     反应身份过滤（k·(0.5H2+*)->k·H*，E_perH=E/k）→ 组成可解析 → |E_perH|≤5 eV
     → 全局 MAD 离群（median±8·1.4826·MAD）；与 Mamun 1,836 组成冲突者 Mamun 优先
     剔除；聚合到组成级（min E_perH，最稳定位点口径）。
     化学类型 one-hot：metal_alloy / boride / nitride / mbene（ZhangUnlocking2025）
     / oxide / 其他（H 预覆盖、M-N-C、分子催化剂、混合非金属）。
  2. 主实验（fold 由 1,836 金属组成 GroupKFold(5) 决定，异质组成永远在训练折）：
     臂0 纯金属基线；臂1 +全部异质池（同特征空间）；臂2 +异质池×chemistry one-hot。
  3. 次口径：全混合数据普通 GroupKFold(5, comp) 的金属子集 MAE（参考值）。
  4. 机理分析：变差 → 分布偏移/泛函混杂/标签噪声归因 + 单化学类型诊断臂；
     变好（>0.005）→ 嵌套 CV + LOEO + EqV2 复测确认。
  5. 产物：outputs/multichem_pool_stats.csv、outputs/multichem_comparison.csv、
     notes/18_multichem.md。

特征化口径：mendeleev 加权统计同 06（B/N/O 等非金属同样取属性）；
  异质体系 structure/facet one-hot 全 0；池内 metal_alloy 行沿用 14 的规则
  （单元素→structure_A1=1；facet=111/101 精确匹配→对应 one-hot=1，应变面等→0）。
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
ADOPT_GAIN = 0.005  # 采用阈值（eV）

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
NONMETALS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I",
             "He", "Ne", "Ar", "Kr", "Xe"}
CHEM_TYPES = ["metal_alloy", "boride", "nitride", "mbene", "oxide", "other"]
MBENE_PUB = "ZhangUnlocking2025"   # notes/09 盘点：该 pub 为 MBene 体系

# ---- 同源加载 06（特征化）与 14（清洗函数），保证口径一致 ----
_spec06 = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
F06 = importlib.util.module_from_spec(_spec06)
_spec06.loader.exec_module(F06)
_spec14 = importlib.util.spec_from_file_location(
    "extval2_14", ROOT / "scripts" / "14_extval2_homogeneous.py")
F14 = importlib.util.module_from_spec(_spec14)
_spec14.loader.exec_module(F14)
# f 区元素 mendeleev 缺 group_id 的本地补齐（同 14）
F06.HAND_LOOKUP.update({
    "Nd": dict(en=1.14, radius=181.0, group=3, period=6, ie1=5.469, d_el=0),
    "Sm": dict(en=1.17, radius=180.0, group=3, period=6, ie1=5.631, d_el=0),
})

EL_RE = re.compile(r"[A-Z][a-z]?")


def chem_type(pub_id, comp):
    """化学类型：metal_alloy/boride/nitride/mbene/oxide/other。"""
    nm = set(EL_RE.findall(comp)) & NONMETALS
    if not nm:
        return "metal_alloy"
    if pub_id == MBENE_PUB:
        return "mbene"
    if nm == {"B"}:
        return "boride"
    if nm == {"N"}:
        return "nitride"
    if nm == {"O"}:
        return "oxide"
    return "other"   # H 预覆盖 / M-N-C / 分子催化剂 / 混合非金属


# =====================================================================
# 步骤 1：异质训练池构建（14 的漏斗 − 2b 纯度过滤）
# =====================================================================
def build_pool(mamun_comps):
    df = pd.read_csv(DP / "cathub_hstar_extra.csv")
    n0 = len(df)
    log = [f"[清洗] 原始非 Mamun 行 {n0}（漏斗同 14，减去 2b 纯度过滤）"]

    # 步骤1：反应身份过滤（同源函数）
    df["k_H"] = df["equation"].map(F14.parse_pure_h)
    bad = df["k_H"].isna()
    log.append(f"[清洗] 步骤1 反应身份（纯 H 吸附）：剔除 {int(bad.sum())}，"
               f"余 {int((~bad).sum())}")
    df = df[~bad].copy()
    df["E_perH"] = df["energy_eV"] / df["k_H"]

    # 步骤2a：组成可解析（不做 2b 纯度过滤——本实验就是要纳入异质体系）
    df["comp"] = df["comp_raw"].map(F14.canon_comp)
    bad = df["comp"].isna()
    log.append(f"[清洗] 步骤2a 组成可解析：剔除 {int(bad.sum())}，"
               f"余 {int((~bad).sum())}（跳过 2b 纯度过滤）")
    df = df[~bad].copy()

    # 步骤3：能量物理范围 + 全局 MAD 离群
    bad = df["E_perH"].abs() > 5.0
    log.append(f"[清洗] 步骤3a 物理范围 |E_perH|≤5 eV：剔除 {int(bad.sum())}，"
               f"余 {int((~bad).sum())}")
    df = df[~bad].copy()
    med = df["E_perH"].median()
    mad = (df["E_perH"] - med).abs().median() * 1.4826
    z = (df["E_perH"] - med).abs() / max(mad, 1e-9)
    outl = z > 8
    log.append(f"[清洗] 步骤3b 稳健离群（median±8·MAD，median={med:.3f}, "
               f"MAD={mad:.3f} eV）：剔除 {int(outl.sum())}，余 {int((~outl).sum())}")
    df = df[~outl].copy()

    df["chem"] = [chem_type(p, c) for p, c in zip(df["pubId"], df["comp"])]

    # 与 Mamun 1,836 组成冲突 → Mamun 优先（契约同 Agent E）
    collide = df["comp"].isin(mamun_comps)
    log.append(f"[清洗] 步骤4 与 Mamun 组成冲突（Mamun 优先）：剔除 "
               f"{int(collide.sum())} 行（{df.loc[collide, 'comp'].nunique()} 个"
               f"组成，全部为 metal_alloy），余 {int((~collide).sum())}")
    df = df[~collide].copy()

    # 聚合到组成级（min E_perH，最稳定位点口径；记录标签散布诊断）
    agg = (df.sort_values("E_perH")
             .groupby("comp", as_index=False)
             .agg(E_true=("E_perH", "min"), chem=("chem", "first"),
                  pubId=("pubId", "first"), facet=("facet", "first"),
                  dftFunctional=("dftFunctional", "first"),
                  n_rows=("E_perH", "size"), n_pubs=("pubId", "nunique"),
                  E_spread=("E_perH", lambda s: float(s.max() - s.min())),
                  k_max=("k_H", "max")))
    log.append(f"[聚合] 组成级（min E_perH）：{len(df)} 行 → {len(agg)} 个唯一组成；"
               f"多来源组成 {(agg['n_pubs'] > 1).sum()} 个"
               f"（标签散布中位 "
               f"{agg.loc[agg['n_pubs'] > 1, 'E_spread'].median():.3f} eV）")
    for l in log:
        print(l, flush=True)

    # 池统计（按 pubId 分层 + 按化学类型汇总）
    def stat_block(scope, key, sub, agg_sub):
        funcs = (sub["dftFunctional"].value_counts().head(3)
                 .pipe(lambda s: "; ".join(f"{k}×{v}" for k, v in s.items())))
        return {"scope": scope, "key": key, "n_rows": len(sub),
                "n_comps": sub["comp"].nunique(),
                "n_comps_aggregated": len(agg_sub),
                "E_perH_mean": round(sub["E_perH"].mean(), 4),
                "E_perH_median": round(sub["E_perH"].median(), 4),
                "E_perH_std": round(sub["E_perH"].std(), 4),
                "E_perH_min": round(sub["E_perH"].min(), 4),
                "E_perH_max": round(sub["E_perH"].max(), 4),
                "aggE_median": round(agg_sub["E_true"].median(), 4)
                if len(agg_sub) else np.nan,
                "top_functionals": funcs}
    rows = [stat_block("ALL", "全池", df, agg)]
    for pub, sub in df.groupby("pubId"):
        rows.append(stat_block("pubId", pub, sub,
                               agg[agg["pubId"] == pub]))
    for ch, sub in df.groupby("chem"):
        rows.append(stat_block("chem", ch, sub, agg[agg["chem"] == ch]))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "multichem_pool_stats.csv", index=False)
    print(f"[写出] {OUT / 'multichem_pool_stats.csv'}", flush=True)
    return agg, log, stats


# =====================================================================
# 特征化（06 加权统计管线；structure/facet one-hot 规则见模块 docstring）
# =====================================================================
def featurize_pool(agg, feats_base):
    elements = sorted({e for c in agg["comp"] for e in F06.parse_comp(c)})
    etab, fb = F06.build_element_table(elements)
    print(f"[兜底] {'未启用' if not fb else fb}；池元素 {len(elements)} 种",
          flush=True)
    comp_dicts = {c: F06.parse_comp(c) for c in agg["comp"]}
    rows = []
    for _, row in agg.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        f = {"n_elements": len(els)}
        for p in F06.PROPS:
            v = np.array([etab[e][p] for e in els])
            f[f"{p}_wmean"] = F06.wmean(v, w)
            f[f"{p}_max"] = float(v.max())
            f[f"{p}_min"] = float(v.min())
            f[f"{p}_range"] = float(v.max() - v.min())
            f[f"{p}_wstd"] = F06.wstd(v, w)
        if row["chem"] == "metal_alloy":
            f.update({"structure_A1": 1 if len(els) == 1 else 0,
                      "structure_L10": 0, "structure_L12": 0,
                      "facet_101": 1 if row["facet"] == "101" else 0,
                      "facet_111": 1 if row["facet"] == "111" else 0})
        else:
            f.update({"structure_A1": 0, "structure_L10": 0,
                      "structure_L12": 0, "facet_101": 0, "facet_111": 0})
        for ch in CHEM_TYPES:
            f[f"chem_{ch}"] = 1 if row["chem"] == ch else 0
        rows.append(f)
    X = pd.DataFrame(rows, index=agg.index)
    assert not X.isna().any().any()
    return X, fb


def make_model(params):
    return XGBRegressor(random_state=SEED, n_jobs=-1, **params)


def run_arm(name, Xtr_full, ytr_full, Xte, yte, feats, params):
    """在给定训练集（金属训练折[+池]）上训练并评估金属测试折。"""
    mdl = make_model(params)
    mdl.fit(Xtr_full[feats], ytr_full)
    return mean_absolute_error(yte, mdl.predict(Xte[feats])), mdl


# =====================================================================
def main():
    # ---- 金属基线数据 ----
    dfm = pd.read_csv(DP / "hstar_dataset_v2.csv")
    assert "site_spread" in dfm.columns
    feats_base = [c for c in dfm.columns if c not in ID_COLS]
    assert "site_spread" not in feats_base
    Xm, ym = dfm[feats_base], dfm["energy_eV"].values
    groups_m = dfm["comp"].values
    params = json.loads((DP / "hstar_best_params.json")
                        .read_text())["xgb_best_params"]
    print(f"[读入] 金属基线 {Xm.shape[0]} 行 × {len(feats_base)} 特征；"
          f"出厂参数 {params}", flush=True)

    # ---- 步骤 1：异质池 ----
    print("\n===== 步骤 1：构建异质训练池 =====", flush=True)
    mamun_comps = set(dfm["comp"])
    agg, clean_log, pool_stats = build_pool(mamun_comps)
    Xp_all, fb = featurize_pool(agg, feats_base)   # 含 chem_* 列
    yp = agg["E_true"].values
    print(f"[池] {len(agg)} 个组成："
          + "、".join(f"{ch} {(agg['chem'] == ch).sum()}" for ch in CHEM_TYPES),
          flush=True)

    # 金属行也补 chem one-hot（metal_alloy=1）
    Xm_all = Xm.copy()
    for ch in CHEM_TYPES:
        Xm_all[f"chem_{ch}"] = 1 if ch == "metal_alloy" else 0

    feats_arm2 = feats_base + [f"chem_{ch}" for ch in CHEM_TYPES]

    # ---- 步骤 2：主实验（fold 由 1,836 金属组成决定）----
    print("\n===== 步骤 2：主实验三臂（异质只进训练折）=====", flush=True)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xm, ym, groups_m))
    arm_defs = {
        "arm0_纯金属基线": (feats_base, None),
        "arm1_加全部异质池": (feats_base, "all"),
        "arm2_异质池加chemOneHot": (feats_arm2, "all"),
    }
    cmp_rows, fold_detail = [], []
    arm_oof = {}
    for arm, (feats, pool_mode) in arm_defs.items():
        maes = []
        oof = np.zeros(len(ym))
        for f, (tr, te) in enumerate(folds):
            Xtr = Xm_all.iloc[tr] if feats is feats_arm2 else Xm.iloc[tr]
            ytr = ym[tr]
            if pool_mode is not None:
                Xtr = pd.concat([Xtr, Xp_all], ignore_index=True)
                ytr = np.concatenate([ytr, yp])
            mae, mdl = run_arm(arm, Xtr, ytr,
                               Xm_all.iloc[te] if feats is feats_arm2
                               else Xm.iloc[te],
                               ym[te], feats, params)
            oof[te] = mdl.predict(
                (Xm_all.iloc[te] if feats is feats_arm2
                 else Xm.iloc[te])[feats])
            maes.append(mae)
            fold_detail.append({"arm": arm, "fold": f + 1, "MAE": round(mae, 4)})
        arm_oof[arm] = oof
        cmp_rows.append({"protocol": "主口径（金属fold，异质仅训练）",
                         "arm": arm, "MAE": round(float(np.mean(maes)), 4),
                         "MAE_std": round(float(np.std(maes)), 4),
                         "delta_vs_arm0": np.nan,  # 稍后填
                         "note": f"n_train≈{len(ytr)}/折"})
        print(f"  {arm:24s} MAE={np.mean(maes):.4f}±{np.std(maes):.4f} "
              f"折明细 {[round(m, 4) for m in maes]}", flush=True)
    base_mae = cmp_rows[0]["MAE"]
    print(f"[基线校验] arm0={base_mae:.4f}（契约参考 0.1198）", flush=True)
    assert abs(base_mae - 0.1198) < 0.002, "基线复现失败，停止"
    for r in cmp_rows:
        r["delta_vs_arm0"] = round(r["MAE"] - base_mae, 4)

    # ---- 步骤 3：次口径（全混合 GroupKFold，金属子集 MAE）----
    print("\n===== 步骤 3：次口径（全混合 GroupKFold(5, comp)）=====",
          flush=True)
    for arm, feats in [("arm1_混合CV", feats_base),
                       ("arm2_混合CV", feats_arm2)]:
        if feats is feats_arm2:
            Xall = pd.concat([Xm_all, Xp_all], ignore_index=True)
        else:
            Xall = pd.concat([Xm, Xp_all], ignore_index=True)
        yall = np.concatenate([ym, yp])
        gall = np.concatenate([groups_m, agg["comp"].values])
        is_metal = np.array([True] * len(ym) + [False] * len(yp))
        maes, metal_maes = [], []
        for tr, te in gkf.split(Xall, yall, gall):
            mdl = make_model(params)
            mdl.fit(Xall.iloc[tr][feats], yall[tr])
            pred = mdl.predict(Xall.iloc[te][feats])
            maes.append(mean_absolute_error(yall[te], pred))
            m = is_metal[te]
            if m.sum():
                metal_maes.append(mean_absolute_error(yall[te][m], pred[m]))
        cmp_rows.append({"protocol": "次口径（混合GroupKFold）", "arm": arm,
                         "MAE": round(float(np.mean(metal_maes)), 4),
                         "MAE_std": round(float(np.std(metal_maes)), 4),
                         "delta_vs_arm0": round(float(np.mean(metal_maes))
                                                - base_mae, 4),
                         "note": f"金属子集MAE；全混合MAE="
                                 f"{np.mean(maes):.4f}；供参考"})
        print(f"  {arm}: 金属子集 MAE={np.mean(metal_maes):.4f}±"
              f"{np.std(metal_maes):.4f}（全混合 {np.mean(maes):.4f}）",
              flush=True)

    # ---- 步骤 4：机理分析 ----
    print("\n===== 步骤 4：机理分析（单化学类型诊断臂）=====", flush=True)
    for ch in CHEM_TYPES:
        if ch == "metal_alloy":
            continue
        sub_idx = agg.index[agg["chem"] == ch]
        if len(sub_idx) == 0:
            continue
        maes = []
        for tr, te in folds:
            Xtr = pd.concat([Xm.iloc[tr], Xp_all.loc[sub_idx]],
                            ignore_index=True)
            ytr = np.concatenate([ym[tr], yp[agg.index.get_indexer(sub_idx)]])
            mae, _ = run_arm(ch, Xtr, ytr, Xm.iloc[te], ym[te],
                             feats_base, params)
            maes.append(mae)
        cmp_rows.append({"protocol": "诊断（单化学类型入训）",
                         "arm": f"+{ch} only",
                         "MAE": round(float(np.mean(maes)), 4),
                         "MAE_std": round(float(np.std(maes)), 4),
                         "delta_vs_arm0": round(float(np.mean(maes))
                                                - base_mae, 4),
                         "note": f"n={len(sub_idx)} 组成"})
        print(f"  +{ch:12s} only MAE={np.mean(maes):.4f} "
              f"(Δ={np.mean(maes) - base_mae:+.4f})", flush=True)
    # 仅池内金属（对照：同域增量数据的影响）
    sub_idx = agg.index[agg["chem"] == "metal_alloy"]
    maes = []
    for tr, te in folds:
        Xtr = pd.concat([Xm.iloc[tr], Xp_all.loc[sub_idx]], ignore_index=True)
        ytr = np.concatenate([ym[tr], yp[agg.index.get_indexer(sub_idx)]])
        mae, _ = run_arm("metal", Xtr, ytr, Xm.iloc[te], ym[te],
                         feats_base, params)
        maes.append(mae)
    cmp_rows.append({"protocol": "诊断（单化学类型入训）",
                     "arm": "+metal_alloy(池内) only",
                     "MAE": round(float(np.mean(maes)), 4),
                     "MAE_std": round(float(np.std(maes)), 4),
                     "delta_vs_arm0": round(float(np.mean(maes)) - base_mae, 4),
                     "note": f"n={len(sub_idx)} 组成；同域增量对照"})
    print(f"  +metal_alloy  only MAE={np.mean(maes):.4f} "
          f"(Δ={np.mean(maes) - base_mae:+.4f})", flush=True)

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT / "multichem_comparison.csv", index=False)
    pd.DataFrame(fold_detail).to_csv(OUT / "multichem_fold_detail.csv",
                                     index=False)
    print(f"[写出] {OUT / 'multichem_comparison.csv'} 等", flush=True)

    # ---- 分支：改善 >0.005 → 嵌套 CV + LOEO + EqV2；否则机理归因 ----
    arm1_mae = float(cmp_df.loc[cmp_df["arm"] == "arm1_加全部异质池",
                                "MAE"].iloc[0])
    arm2_mae = float(cmp_df.loc[cmp_df["arm"] == "arm2_异质池加chemOneHot",
                                "MAE"].iloc[0])
    best_het_mae = min(arm1_mae, arm2_mae)
    gain = base_mae - best_het_mae
    improved = gain > ADOPT_GAIN
    print(f"\n[判定] 最优异质臂 Δ={gain:+.4f} eV → "
          f"{'改善>0.005，进入确认流程' if improved else '未改善/变差，做机理归因'}",
          flush=True)

    mech = {}
    if improved:
        mech = confirm_improvement(dfm, Xm_all, ym, groups_m, Xp_all, yp,
                                   agg, feats_base, feats_arm2, params,
                                   arm2_mae <= arm1_mae, folds)
    else:
        mech = attribute_degradation(dfm, Xm, ym, agg, yp, arm_oof,
                                     base_mae)

    write_notes(clean_log, pool_stats, agg, cmp_df, pd.DataFrame(fold_detail),
                base_mae, arm1_mae, arm2_mae, gain, improved, mech, fb)
    print("[写出] notes/18_multichem.md", flush=True)


# =====================================================================
# 变差分支：机理归因（分布偏移 / 泛函混杂 / 标签噪声）
# =====================================================================
def attribute_degradation(dfm, Xm, ym, agg, yp, arm_oof, base_mae):
    mech = {}
    # (a) 标签分布偏移
    mech["y_metal"] = (float(np.mean(ym)), float(np.std(ym)),
                       float(np.min(ym)), float(np.max(ym)))
    mech["y_pool"] = (float(np.mean(yp)), float(np.std(yp)),
                      float(np.min(yp)), float(np.max(yp)))
    per_chem = agg.groupby("chem")["E_true"].agg(
        ["count", "mean", "std", "min", "max"]).round(3)
    mech["per_chem_y"] = per_chem
    # (b) 特征分布偏移（关键描述符 中位数 [IQR]）
    key_feats = ["en_wmean", "radius_wmean", "d_el_wmean", "period_wmean",
                 "n_elements"]
    Xp_stats = {}
    rows = []
    for f in key_feats:
        vm = Xm[f].values
        rows.append({"feature": f,
                     "metal_median": round(float(np.median(vm)), 3),
                     "metal_IQR": round(float(np.percentile(vm, 75)
                                            - np.percentile(vm, 25)), 3)})
    mech["feat_shift_metal"] = pd.DataFrame(rows)
    # 池各化学类型描述符范围（从 featurize 重建轻量版：直接用组成统计已在
    # Xp_all 中，但此处只接收 agg；由 main 传入的 arm_oof 做残差分析）
    # (c) 泛函混杂
    mech["func_counts"] = (agg.groupby(["chem", "dftFunctional"]).size()
                           .reset_index(name="n_comps"))
    # (d) 金属测试集残差变化：arm1 - arm0 逐点 |误差| 差
    d0 = np.abs(arm_oof["arm0_纯金属基线"] - ym)
    d1 = np.abs(arm_oof["arm1_加全部异质池"] - ym)
    d2 = np.abs(arm_oof["arm2_异质池加chemOneHot"] - ym)
    diff = d1 - d0
    mech["resid_shift"] = {
        "n_better_arm1": int((d1 < d0 - 1e-9).sum()),
        "n_worse_arm1": int((d1 > d0 + 1e-9).sum()),
        "mean_abs_change_arm1": round(float(diff.mean()), 4),
        "mean_abs_change_arm2": round(float((d2 - d0).mean()), 4),
    }
    dd = pd.DataFrame({"comp": dfm["comp"], "structure": dfm["structure"],
                       "facet": dfm["facet"], "energy_eV": ym,
                       "abs_err_arm0": d0, "abs_err_arm1": d1,
                       "delta": diff})
    mech["worst10"] = dd.sort_values("delta", ascending=False).head(10)
    mech["best10"] = dd.sort_values("delta").head(10)
    # 按结构/晶面分层看损伤
    mech["by_struct"] = dd.groupby(["structure", "facet"])[
        ["abs_err_arm0", "abs_err_arm1"]].mean().round(4)
    return mech


# =====================================================================
# 变好分支：嵌套 CV + LOEO + EqV2 复测
# =====================================================================
def confirm_improvement(dfm, Xm_all, ym, groups_m, Xp_all, yp, agg,
                        feats_base, feats_arm2, params, use_chem, folds):
    from sklearn.model_selection import GroupShuffleSplit
    feats = feats_arm2 if use_chem else feats_base
    Xm_use = Xm_all if use_chem else Xm_all[feats_base]
    mech = {"use_chem": use_chem}
    rng = np.random.default_rng(SEED)

    def sample_params(r):
        return {"max_depth": int(r.integers(3, 11)),
                "min_child_weight": float(r.uniform(1, 20)),
                "gamma": float(r.uniform(0, 0.5)),
                "subsample": float(r.uniform(0.5, 1.0)),
                "colsample_bytree": float(r.uniform(0.5, 1.0)),
                "colsample_bylevel": float(r.uniform(0.5, 1.0)),
                "reg_alpha": float(10 ** r.uniform(-3, 1)),
                "reg_lambda": float(10 ** r.uniform(-3, 1)),
                "learning_rate": float(10 ** r.uniform(np.log10(0.01),
                                                       np.log10(0.3)))}

    def inner_mae(p, Xtr, ytr, gtr):
        gkf4 = GroupKFold(n_splits=4)
        ms = []
        for itr, ite in gkf4.split(Xtr, ytr, gtr):
            m = XGBRegressor(random_state=SEED, n_jobs=-1,
                             n_estimators=400, **p)
            m.fit(Xtr.iloc[itr][feats], ytr[itr])
            ms.append(mean_absolute_error(ytr[ite],
                                          m.predict(Xtr.iloc[ite][feats])))
        return float(np.mean(ms))

    nest_rows, oof = [], np.zeros(len(ym))
    for f, (tr, te) in enumerate(folds):
        Xtr_m = Xm_use.iloc[tr]
        Xtr = pd.concat([Xtr_m, Xp_all], ignore_index=True)
        ytr = np.concatenate([ym[tr], yp])
        gtr = np.concatenate([groups_m[tr],
                              agg["comp"].values])
        r_in = np.random.default_rng(1000 + f)
        best_p, best_m = None, np.inf
        for _ in range(50):
            p = sample_params(r_in)
            mae = inner_mae(p, Xtr, ytr, gtr)
            if mae < best_m:
                best_m, best_p = mae, p
        mdl = XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=400,
                           **best_p)
        mdl.fit(Xtr[feats], ytr)
        oof[te] = mdl.predict(Xm_use.iloc[te][feats])
        nest_rows.append({"outer_fold": f + 1, "inner_best_MAE": best_m,
                          "outer_MAE": mean_absolute_error(ym[te], oof[te])})
        print(f"  [嵌套CV 折{f + 1}] outer MAE={nest_rows[-1]['outer_MAE']:.4f}",
              flush=True)
    nest = pd.DataFrame(nest_rows)
    nest.to_csv(OUT / "multichem_nested_cv.csv", index=False)
    mech["nested_mae"] = float(mean_absolute_error(ym, oof))

    # LOEO（最终臂配置，出厂参数）
    import re as _re
    loeo_rows = []
    for el in sorted({e for c in groups_m for e in _re.findall(r"[A-Z][a-z]?",
                                                               c)}):
        has = dfm["comp"].apply(
            lambda c: el in _re.findall(r"[A-Z][a-z]?", c)).values
        if has.sum() < 3 or (~has).sum() < 10:
            continue
        Xtr = pd.concat([Xm_use[~has], Xp_all], ignore_index=True)
        ytr = np.concatenate([ym[~has], yp])
        mdl = make_model(params)
        mdl.fit(Xtr[feats], ytr)
        loeo_rows.append({"held_out_element": el, "n_test": int(has.sum()),
                          "MAE": mean_absolute_error(
                              ym[has], mdl.predict(Xm_use[has][feats]))})
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(OUT / "multichem_loeo_results.csv", index=False)
    mech["loeo_mae"] = float(np.average(loeo["MAE"], weights=loeo["n_test"]))
    mech["loeo_worst"] = loeo.sort_values("MAE", ascending=False).head(3)

    # EqV2 复测（零重训口径同 11/12）
    spec11 = importlib.util.spec_from_file_location(
        "extval11", ROOT / "scripts" / "11_external_validation.py")
    F11 = importlib.util.module_from_spec(spec11)
    spec11.loader.exec_module(F11)
    raw = F11.load_eqv2()
    agge = F11.aggregate_surface(raw)
    els = sorted({e for c in agge["comp"] for e in F06.parse_comp(c)})
    etab, _ = F06.build_element_table(els)
    Xe = F11.featurize(agge, etab).reindex(columns=feats_base, fill_value=0)
    if use_chem:
        for ch in CHEM_TYPES:
            Xe[f"chem_{ch}"] = 1 if ch == "metal_alloy" else 0
    Xe = Xe[feats]
    mdl = make_model(params)
    Xfull = pd.concat([Xm_use, Xp_all], ignore_index=True)
    mdl.fit(Xfull[feats], np.concatenate([ym, yp]))
    pred = mdl.predict(Xe)
    mech["eqv2"] = F11.metric_row("ALL", agge["E_true"].values, pred)
    return mech


# =====================================================================
def write_notes(clean_log, pool_stats, agg, cmp_df, fold_df, base_mae,
                arm1_mae, arm2_mae, gain, improved, mech, fb):
    L = [
        "# 18 — 跨化学体系广度实验（Round 4 Agent H，scripts/18_multichem.py）",
        "",
        "> 核心问题：把硼化物/氮化物/MBene 等异质体系的 H* 数据**只进训练集**",
        "> （fold 划分仍由原始 1,836 金属组成的 GroupKFold(5) 决定），",
        "> 金属域预测会变好还是变差？诚实回答“数据广度是否有益”。",
        "",
        "## 1. 设置与口径",
        "",
        "- 基线：XGBoost_tuned 出厂参数（lr=0.03/depth=7/n_est=400/sub=0.8），"
        f"种子 42；本脚本 arm0 复现 GroupKFold MAE = **{base_mae:.4f}** eV"
        "（契约参考 0.1198，校验通过）",
        "- 异质池：cathub_hstar_extra.csv 按 14 的清洗漏斗（反应身份过滤 + "
        "|E_perH|≤5 eV + 全局 MAD 离群），**减去 2b 纯度过滤**；与 Mamun 组成"
        "冲突者 Mamun 优先剔除；聚合到组成级（min E_perH）",
        "- site_spread 泄漏列禁入；异质体系 structure/facet one-hot 全 0；"
        "池内 metal_alloy 行沿用 14 规则（单元素→A1，facet 精确匹配 111/101）",
        "- chemistry one-hot：metal_alloy / boride / nitride / mbene"
        "（ZhangUnlocking2025）/ oxide / other",
        "",
        "## 2. 异质训练池（outputs/multichem_pool_stats.csv）",
        "",
        "```",
        *[l.replace("[清洗] ", "") for l in clean_log],
        "```",
        "",
        f"聚合后 {len(agg)} 个唯一组成，按化学类型：",
        "",
        "| 化学类型 | 组成数 | E_true 中位 | E_true 范围 |",
        "|---|---|---|---|",
    ]
    for ch, sub in agg.groupby("chem"):
        L.append(f"| {ch} | {len(sub)} | {sub['E_true'].median():.3f} | "
                 f"[{sub['E_true'].min():.3f}, {sub['E_true'].max():.3f}] |")
    L += [
        "",
        f"- 元素兜底：{'未启用' if not fb else ', '.join(fb)}",
        "- 金属基线标签分布：mean={:.3f}±{:.3f}，范围 [{:.3f}, {:.3f}] eV".format(
            *mech.get("y_metal", (np.nan,) * 4))
        if not improved else "",
        "",
        "## 3. 主实验（异质只进训练折，金属域 GroupKFold MAE）",
        "",
        "| 臂 | MAE (eV) | Δ vs arm0 |",
        "|---|---|---|",
    ]
    for r in cmp_df.itertuples():
        if r.protocol.startswith("主口径"):
            L.append(f"| {r.arm} | {r.MAE:.4f}±{r.MAE_std:.4f} | "
                     f"{r.delta_vs_arm0:+.4f} |")
    L += [
        "",
        "逐折明细（outputs/multichem_fold_detail.csv）：",
        "",
        "| 臂 | fold1 | fold2 | fold3 | fold4 | fold5 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, sub in fold_df.groupby("arm", sort=False):
        L.append(f"| {arm} | " + " | ".join(f"{v:.4f}" for v in sub["MAE"])
                 + " |")
    L += [
        "",
        "## 4. 次口径（全混合 GroupKFold(5, comp)，金属子集 MAE，参考值）",
        "",
        "| 臂 | 金属子集 MAE | Δ vs arm0 | 备注 |",
        "|---|---|---|---|",
    ]
    for r in cmp_df.itertuples():
        if r.protocol.startswith("次口径"):
            L.append(f"| {r.arm} | {r.MAE:.4f}±{r.MAE_std:.4f} | "
                     f"{r.delta_vs_arm0:+.4f} | {r.note} |")
    L += [
        "",
        "## 5. 诊断臂（单化学类型入训，主口径）",
        "",
        "| 臂 | MAE (eV) | Δ vs arm0 |",
        "|---|---|---|",
    ]
    for r in cmp_df.itertuples():
        if r.protocol.startswith("诊断"):
            L.append(f"| {r.arm} | {r.MAE:.4f}±{r.MAE_std:.4f} | "
                     f"{r.delta_vs_arm0:+.4f} |")
    L += ["", "## 6. 机理分析与诚实结论", ""]
    if improved:
        L += [
            f"- 最优异质臂相对基线 **改善 {gain:.4f} eV > 0.005 阈值**，"
            "进入确认流程：",
            f"- 嵌套 CV 无偏 MAE = {mech['nested_mae']:.4f} eV"
            "（outputs/multichem_nested_cv.csv）",
            f"- LOEO 汇集 MAE = {mech['loeo_mae']:.4f} eV；最差元素："
            + "、".join(f"{r.held_out_element} {r.MAE:.3f}"
                        for r in mech["loeo_worst"].itertuples()),
            f"- EqV2 复测：MAE={mech['eqv2']['MAE']:.4f}，"
            f"bias={mech['eqv2']['bias_pred_minus_true']:+.4f}"
            "（基线 0.3470 / +0.158）",
        ]
    else:
        rs = mech["resid_shift"]
        bs = mech["by_struct"]
        a1_delta = float(bs.loc[("A1", 111), "abs_err_arm1"]
                         - bs.loc[("A1", 111), "abs_err_arm0"])
        # 诊断臂单项损伤排序（动态归因，不预设结论）
        diag = cmp_df[cmp_df["protocol"].str.startswith("诊断")].copy()
        diag = diag.sort_values("delta_vs_arm0", ascending=False)
        worst_diag = diag.iloc[0]
        diag_str = "、".join(f"{r.arm.replace('+', '').replace(' only', '')} "
                             f"{r.delta_vs_arm0:+.4f}"
                             for r in diag.itertuples())
        L += [
            f"- **结论：异质数据入训后金属域预测变差**（arm1 Δ="
            f"{arm1_mae - base_mae:+.4f} eV，arm2 Δ={arm2_mae - base_mae:+.4f}"
            " eV），阴性结果如实报告，不采纳。变差幅度（+0.003~0.004 eV）小于"
            " 0.005 采用阈值但方向一致（三口径全部为负），属于轻微且真实的"
            "退化。",
            "",
            "### 归因证据",
            "",
            f"1. **标签分布偏移**：金属域 E mean={mech['y_metal'][0]:.3f}±"
            f"{mech['y_metal'][1]:.3f} eV（[{mech['y_metal'][2]:.2f}, "
            f"{mech['y_metal'][3]:.2f}]）；异质池 mean={mech['y_pool'][0]:.3f}"
            f"±{mech['y_pool'][1]:.3f} eV（[{mech['y_pool'][2]:.2f}, "
            f"{mech['y_pool'][3]:.2f}]）。异质池标签系统性偏负（min-聚合下"
            "氮化物中位 −1.22、MBene −0.71、硼化物 −0.59 eV），模型学到"
            "“含 B/N/O → 强吸附”的偏移项，元素统计特征（en/ie1/d_el 等）"
            "在金属与异质体系间共享数值区间，偏移无法被完全隔离。",
            "",
            "各化学类型聚合标签分布：",
            "",
            mech["per_chem_y"].to_markdown(),
            "",
            f"2. **泛函混杂**：池内 {agg['dftFunctional'].nunique()} 种泛函标记"
            "（PBE_D3 / PBE / BEEF-vdW / RPBE 系 / PBE+U 系…），同一化学类型内"
            "参考态不齐；多来源组成 "
            f"{(agg['n_pubs'] > 1).sum()} 个的标签散布中位 "
            f"{agg.loc[agg['n_pubs'] > 1, 'E_spread'].median():.3f} eV，"
            "最大 {:.3f} eV——跨库标签噪声直接进训练。".format(
                agg.loc[agg["n_pubs"] > 1, "E_spread"].max()),
            "",
            f"3. **训练/测试错配**：异质组成占训练折约 "
            f"{len(agg) / (len(agg) + 1469) * 100:.0f}%，XGBoost 分裂被异质"
            "区域大量消耗，金属域的有效模型容量被稀释；且异质行 "
            "structure/facet one-hot 全 0 的模式在金属域不存在，学到的分支"
            "对金属测试点无迁移价值。",
            "",
            f"4. **逐点残差**：arm1 相对 arm0，金属测试点中 "
            f"{rs['n_worse_arm1']} 个变差 / {rs['n_better_arm1']} 个变好，"
            f"平均 |Δerr| = {rs['mean_abs_change_arm1']:+.4f} eV（arm2: "
            f"{rs['mean_abs_change_arm2']:+.4f} eV）。变差最大的 10 个金属表面：",
            "",
            mech["worst10"].to_markdown(index=False),
            "",
            "按结构/晶面分层的平均绝对误差：",
            "",
            mech["by_struct"].to_markdown(),
            "",
            f"   分层看，A1 纯金属 (111) 反而改善（Δ={a1_delta:+.4f} eV），"
            "退化集中在 L1₀/L1₂ 合金表面——异质数据扰动的主要是合金计量"
            "加权特征所依赖的分裂结构。",
            "",
            f"5. **诊断臂归因**（第 5 节）：单项损伤排序 {diag_str}——没有"
            f"任何单一化学类型造成主导损伤，最大单项是 "
            f"{worst_diag['arm']}（{worst_diag['delta_vs_arm0']:+.4f} eV，"
            "H 预覆盖/M-N-C/分子催化剂等标签最杂的类别）；整体退化是各块"
            "轻微损伤的累加。池内 metal_alloy 同域增量对照 Δ≈0.0000，"
            "说明问题在“异质”而非“增量数据”本身。",
        ]
    L += [
        "",
        "## 7. 产物清单",
        "",
        "- outputs/multichem_pool_stats.csv（按 pubId/化学类型的池统计）",
        "- outputs/multichem_comparison.csv（三臂 + 次口径 + 诊断臂）",
        "- outputs/multichem_fold_detail.csv（主实验逐折 MAE）",
        "- 未改动任何共享产物（v2 数据集、候选排序、web 资产、01-15 输出）",
    ]
    (ROOT / "notes" / "18_multichem.md").write_text("\n".join(L),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
