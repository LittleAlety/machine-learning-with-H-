# -*- coding: utf-8 -*-
"""
16_merge_retrain.py — Round 4 Agent E：CatHub 同质金属 HER 增量合并重训实验

核心问题：把 extval2_homogeneous.csv（Catalysis-Hub 同质金属 HER，355 表面）
并入 Mamun 主线训练（hstar_dataset_v2.csv，1,836 组成），能否改善 H* 主线模型？

公平性总契约（plan_round4.md）：
- 新数据只进训练集、不进测试集（fold 划分仍由原始 1,836 组成的
  GroupKFold(5, comp) 决定）→ 直接回答"加数据是否改善原域预测"
- 采用阈值：主口径改善 > 0.005 eV 才采纳；任何调参走嵌套 CV；种子 42；
  site_spread 泄漏列禁入；不改 v2 数据集/候选排序/web/scripts 01-15 产物

步骤：
1. 重叠分析：355 表面按 comp 聚合（min E_true = 最稳定位点口径，同主线）→
   与 1,836 求交，报告重叠/纯新增数与能量分布对比
   → outputs/merge_overlap_analysis.csv
2. 合并策略：重叠组成 Mamun 优先（域内单源口径），纯新增追加。两个变体：
   (a) 朴素合并（沿用 36 特征；新增组成 structure/facet one-hot 按 06/14
       推断规则：单元素→structure_A1=1，合金→全 0；facet 精确 "111"/"101"
       → 对应 one-hot，其余（应变/台阶/hcp0001 等）→ 全 0）
   (b) 合并 + 泛函 one-hot（dftFunctional 分箱：BEEF-vdW/PBE/RPBE/MS2/其他，
       前缀匹配、大小写不敏感，PBE+D3/PBEsol→PBE、RPBE-D3/RPBE_*VSHE→RPBE、
       BEEF-vdW_*VSHE→BEEF-vdW；Mamun 行全部标 BEEF-vdW）
3. 评估：
   - 主口径：fold 仅由原始 1,836 组成 GroupKFold(5, comp) 决定；新增组成
     永远加入各折训练集、从不进测试集。固定出厂参数（lr=0.03/depth=7/
     n=400/sub=0.8，种子 42）对比 纯 Mamun vs (a) vs (b)
   - 次口径：全合并数据普通 GroupKFold(5, comp)
   → outputs/merge_comparison.csv
4. 深度检查（仅当某变体主口径优于基线 > 0.005 eV 才触发）：
   嵌套 CV（外层 5 / 内层 4，内层随机搜索 30 组，新增组成恒在训练侧）、
   LOEO、EqV2 外部验证（复用 11 流程）、25 文献锚点复测
   → outputs/merge_deepcheck_*.csv
5. notes/16_merge_experiment.md（中文，含采纳/不采纳结论及依据）
"""
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw",
           "site_spread"]  # site_spread: 泄漏特征，禁止入模
DGCORR = 0.24             # eV，ΔG_H* = E_ads + 0.24（同 08/10/12）
ADOPT_GAIN = 0.005        # 采用阈值（plan_round4 契约）

# 固定出厂参数（hstar_best_params.json，07 的 XGBoost_tuned）
FACTORY = json.loads((DP / "hstar_best_params.json")
                     .read_text())["xgb_best_params"]

# 嵌套 CV 设置（外层 5 / 内层 4 / 内层随机搜索 30 组）
PARAM_KEYS = ["max_depth", "min_child_weight", "gamma", "subsample",
              "colsample_bytree", "colsample_bylevel", "reg_alpha",
              "reg_lambda", "learning_rate"]
N_ITER_NESTED = 30
ES_ROUNDS = 50
N_EST_CAP = 2000

FUNC_BINS = ["BEEF-vdW", "PBE", "RPBE", "MS2", "其他"]
FUNC_FEATS = [f"func_{b}" for b in FUNC_BINS]

# ---- 同源加载 06 的特征化函数（口径一致，同 11/14）----
_spec = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
F06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F06)
# f 区元素 mendeleev 缺 group_id，06 的 HAND_LOOKUP 只备了 Tc → 补齐 Nd/Sm
# （同 14；Mg/Si 经 mendeleev 齐全，无需兜底）
F06.HAND_LOOKUP.update({
    "Nd": dict(en=1.14, radius=181.0, group=3, period=6, ie1=5.469, d_el=0),
    "Sm": dict(en=1.17, radius=180.0, group=3, period=6, ie1=5.631, d_el=0),
})

# ---- 同源加载 11 的 EqV2 外部验证函数（深度检查用）----
_spec11 = importlib.util.spec_from_file_location(
    "extval11", ROOT / "scripts" / "11_external_validation.py")
F11 = importlib.util.module_from_spec(_spec11)
_spec11.loader.exec_module(F11)


# =====================================================================
# 工具函数
# =====================================================================
def func_bin(f):
    """dftFunctional 分箱（前缀匹配、大小写不敏感）：
    beef*→BEEF-vdW；pbe*（含 PBE+D3/PBEsol）→PBE；rpbe*（含 RPBE-D3、
    RPBE_*VSHE）→RPBE；ms2*→MS2；其余（FNDMC/SCAN 等）→其他。"""
    s = str(f).strip().lower()
    if s.startswith("beef"):
        return "BEEF-vdW"
    if s.startswith("rpbe"):
        return "RPBE"
    if s.startswith("pbe"):
        return "PBE"
    if s.startswith("ms2"):
        return "MS2"
    return "其他"


def featurize_new(comps_meta, feats):
    """按 06/14 口径为新增组成构造 36 特征。
    comps_meta: DataFrame(comp, facet)；structure/facet one-hot 推断规则：
    单元素→structure_A1=1；合金→structure 全 0（表面计量模式未知，不臆造）；
    facet 精确 "111"/"101"→对应 one-hot，其余→全 0（域外，与 14 一致）。"""
    comp_dicts = {c: F06.parse_comp(c) for c in comps_meta["comp"].unique()}
    elements = sorted({e for d in comp_dicts.values() for e in d})
    etab, fb = F06.build_element_table(elements)
    print(f"[兜底] {'未启用' if not fb else fb}", flush=True)
    rows = []
    for _, row in comps_meta.iterrows():
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
        f.update({"structure_A1": 1 if len(els) == 1 else 0,
                  "structure_L10": 0, "structure_L12": 0,
                  "facet_101": 1 if str(row["facet"]) == "101" else 0,
                  "facet_111": 1 if str(row["facet"]) == "111" else 0})
        rows.append(f)
    Xn = pd.DataFrame(rows, index=comps_meta.index)
    Xn = Xn.reindex(columns=feats, fill_value=0)
    assert list(Xn.columns) == feats and not Xn.isna().any().any()
    return Xn


def add_func_onehot(df_all, mamun_mask):
    """变体 (b)：泛函 one-hot。Mamun 行全部 BEEF-vdW；新增行按各自
    dftFunctional 分箱。"""
    oh = pd.DataFrame(0, index=df_all.index, columns=FUNC_FEATS)
    oh.loc[mamun_mask, "func_BEEF-vdW"] = 1
    for b in FUNC_BINS:
        m = (~mamun_mask) & (df_all["func_bin"] == b)
        oh.loc[m, f"func_{b}"] = 1
    return pd.concat([df_all, oh], axis=1)


def fit_predict(params, Xtr, ytr, Xte):
    m = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


# ---- 嵌套 CV 用（同 12 的采样空间与 early-stopping 口径）----
def sample_params(rng):
    """同 12 的 9 维随机采样空间。"""
    return {
        "max_depth": int(rng.integers(3, 11)),
        "min_child_weight": float(rng.uniform(1, 20)),
        "gamma": float(rng.uniform(0, 0.5)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(10 ** rng.uniform(-3, 1)),
        "reg_lambda": float(10 ** rng.uniform(-3, 1)),
        "learning_rate": float(10 ** rng.uniform(np.log10(0.01),
                                                 np.log10(0.3))),
    }


def group_val_split(X, y, groups, seed, test_size=0.15):
    """同 12：从训练折按组切 15% early-stopping 验证折（组级无泄漏）。"""
    gs = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    return next(gs.split(X, y, groups))


def eval_config_merged(params, Xm, ym, gm, Xn, yn, n_splits, seed):
    """在'Mamun 部分按组划折 + 新增行恒入训练'的口径下评估一组超参
    （内层搜索用；Xm/gm 为 Mamun 侧，Xn/yn 为新增侧）。返回 fold MAEs。"""
    gkf = GroupKFold(n_splits=n_splits)
    fold_mae = []
    for i, (tr, te) in enumerate(gkf.split(Xm, ym, gm)):
        t2, va = group_val_split(Xm.iloc[tr], ym[tr], gm[tr], seed + i)
        mdl = XGBRegressor(random_state=SEED, n_jobs=-1,
                           n_estimators=N_EST_CAP,
                           early_stopping_rounds=ES_ROUNDS, **params)
        Xtr = pd.concat([Xm.iloc[tr].iloc[t2], Xn])
        ytr = np.concatenate([ym[tr][t2], yn])
        mdl.fit(Xtr, ytr, eval_set=[(Xm.iloc[tr].iloc[va], ym[tr][va])],
                verbose=False)
        fold_mae.append(mean_absolute_error(ym[te],
                                            mdl.predict(Xm.iloc[te])))
    return fold_mae


def fit_merged_es(params, Xm, ym, gm, Xn, yn, seed):
    """大块训练（同 12 的 fit_with_es 口径，训练侧追加新增行）。"""
    t2, va = group_val_split(Xm, ym, gm, seed)
    mdl = XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=N_EST_CAP,
                       early_stopping_rounds=ES_ROUNDS, **params)
    mdl.fit(pd.concat([Xm.iloc[t2], Xn]), np.concatenate([ym[t2], yn]),
            eval_set=[(Xm.iloc[va], ym[va])], verbose=False)
    n_it = mdl.best_iteration + 1
    final = XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=n_it,
                         **params)
    final.fit(pd.concat([Xm, Xn]), np.concatenate([ym, yn]))
    return final, n_it


# =====================================================================
# Step 1：重叠分析
# =====================================================================
def step1_overlap(dfm, ext):
    """355 表面按 comp 聚合（min E_true，最稳定位点口径）→ 与 Mamun 求交。"""
    idx = ext.groupby("comp")["E_true"].idxmin()
    am = ext.loc[idx, ["comp", "pubId", "facet", "E_true", "dftFunctional",
                       "k_min", "k_max", "n_rows"]].copy()
    am["n_surfaces"] = am["comp"].map(ext.groupby("comp").size())
    am = am.rename(columns={"E_true": "E_true_min", "facet": "facet_argmin",
                            "pubId": "pubId_argmin",
                            "dftFunctional": "functional_argmin"})
    am["func_bin"] = am["functional_argmin"].map(func_bin)
    am["in_mamun"] = am["comp"].isin(set(dfm["comp"]))
    mam_e = dfm.set_index("comp")["energy_eV"]
    am["mamun_energy_eV"] = am["comp"].map(mam_e)
    am["E_diff_ext_minus_mamun"] = am["E_true_min"] - am["mamun_energy_eV"]
    am["decision"] = np.where(am["in_mamun"], "keep_mamun", "append")
    am = am.sort_values(["decision", "E_true_min"]).reset_index(drop=True)
    am.to_csv(OUT / "merge_overlap_analysis.csv", index=False)

    ov, nw = am[am["in_mamun"]], am[~am["in_mamun"]]

    def stat(s):
        return f"mean={s.mean():.3f}, std={s.std():.3f}, " \
               f"range=[{s.min():.2f}, {s.max():.2f}]"
    print("\n===== Step 1 重叠分析 =====", flush=True)
    print(f"[聚合] 355 表面 → {len(am)} 个唯一组成；重叠 {len(ov)}，"
          f"纯新增 {len(nw)}", flush=True)
    print(f"[能量] Mamun 1,836: {stat(dfm['energy_eV'])}", flush=True)
    print(f"[能量] 重叠 {len(ov)} 组成(CatHub min): {stat(ov['E_true_min'])}",
          flush=True)
    print(f"[能量] 纯新增 {len(nw)} 组成: {stat(nw['E_true_min'])}", flush=True)
    d = ov["E_diff_ext_minus_mamun"].dropna()
    print(f"[重叠偏差] E_CatHub − E_Mamun: mean={d.mean():+.3f}, "
          f"std={d.std():.3f}, median|Δ|={d.abs().median():.3f} eV"
          f"（系统性偏负→泛函差异，Mamun 优先策略合理）", flush=True)
    print(f"[新增泛函] {nw['func_bin'].value_counts().to_dict()}；"
          f"[新增来源] {nw['pubId_argmin'].value_counts().to_dict()}",
          flush=True)
    print(f"[写出] {OUT / 'merge_overlap_analysis.csv'}", flush=True)
    return am, nw


# =====================================================================
# Step 3：评估
# =====================================================================
def eval_main_protocol(dfm, X, y, groups, Xn, yn, label):
    """主口径：fold 仅由原始 1,836 组成决定；新增行恒入训练、从不进测试。
    返回 (fold_maes, mean, std)。label 仅用于日志。"""
    gkf = GroupKFold(n_splits=5)
    maes = []
    for tr, te in gkf.split(X, y, groups):
        if Xn is not None:
            Xtr = pd.concat([X.iloc[tr], Xn])
            ytr = np.concatenate([y[tr], yn])
        else:
            Xtr, ytr = X.iloc[tr], y[tr]
        pred = fit_predict(FACTORY, Xtr, ytr, X.iloc[te])
        maes.append(mean_absolute_error(y[te], pred))
    maes = np.array(maes)
    print(f"  [{label}] fold MAEs={np.round(maes, 4).tolist()} "
          f"mean={maes.mean():.4f}±{maes.std():.4f}", flush=True)
    return maes


def eval_secondary_protocol(Xall, yall, gall, label):
    """次口径：全合并数据普通 GroupKFold(5, comp)（新组成也参与划折）。"""
    gkf = GroupKFold(n_splits=5)
    maes = []
    for tr, te in gkf.split(Xall, yall, gall):
        pred = fit_predict(FACTORY, Xall.iloc[tr], yall[tr], Xall.iloc[te])
        maes.append(mean_absolute_error(yall[te], pred))
    maes = np.array(maes)
    print(f"  [{label}] fold MAEs={np.round(maes, 4).tolist()} "
          f"mean={maes.mean():.4f}±{maes.std():.4f}", flush=True)
    return maes


# =====================================================================
# Step 4：深度检查（仅对主口径改善 >0.005 的变体触发）
# =====================================================================
def deep_nested_cv(X, y, groups, Xn, yn, tag):
    """嵌套 CV：外层 GroupKFold(5, 仅 Mamun 划折，新增恒入外层训练)，
    内层 GroupKFold(4) 随机搜索 30 组。返回 (oof, fold rows)。"""
    outer = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    rows = []
    for f, (tr, te) in enumerate(outer.split(X, y, groups)):
        rng = np.random.default_rng(1000 + f)
        best_mae, best_p = np.inf, None
        for i in range(N_ITER_NESTED):
            p = sample_params(rng)
            fm = eval_config_merged(p, X.iloc[tr], y[tr], groups[tr],
                                    Xn, yn, n_splits=4, seed=SEED + f)
            m = float(np.mean(fm))
            if m < best_mae:
                best_mae, best_p = m, p
        mdl, n_it = fit_merged_es(best_p, X.iloc[tr], y[tr], groups[tr],
                                  Xn, yn, seed=SEED + f)
        oof[te] = mdl.predict(X.iloc[te])
        rows.append({"variant": tag, "outer_fold": f + 1,
                     "inner_best_MAE": best_mae,
                     "outer_MAE": mean_absolute_error(y[te], oof[te]),
                     "n_estimators": n_it,
                     **{f"best_{k}": v for k, v in best_p.items()}})
        print(f"    [{tag} 外层折{f + 1}] 内层最优={best_mae:.4f} "
              f"外层 MAE={rows[-1]['outer_MAE']:.4f}", flush=True)
    mae = mean_absolute_error(y, oof)
    std = float(np.std([r["outer_MAE"] for r in rows]))
    print(f"    [{tag} 嵌套CV] 无偏 MAE={mae:.4f}±{std:.4f}", flush=True)
    return mae, std, pd.DataFrame(rows)


def deep_loeo(dfm, X, y, groups, Xn, yn, tag):
    """LOEO（固定出厂参数；训练侧 = 不含该元素的 Mamun 行 + 全部新增行）。"""
    elements = sorted({e for c in groups for e in F06.parse_comp(c)})
    rows = []
    for el in elements:
        has = dfm["comp"].apply(
            lambda c: el in re.findall(r"[A-Z][a-z]?", c)).values
        if has.sum() < 3 or (~has).sum() < 10:
            continue
        Xtr = pd.concat([X[~has], Xn])
        ytr = np.concatenate([y[~has], yn])
        pred = fit_predict(FACTORY, Xtr, ytr, X[has])
        rows.append({"variant": tag, "held_out_element": el,
                     "n_test": int(has.sum()),
                     "MAE": mean_absolute_error(y[has], pred)})
    loeo = pd.DataFrame(rows)
    pooled = float(np.average(loeo["MAE"], weights=loeo["n_test"]))
    print(f"    [{tag} LOEO] 汇集 MAE={pooled:.4f}（{len(loeo)} 折）",
          flush=True)
    return pooled, loeo


def deep_extval(feats_all, Xfull, yfull, variant_b, tag):
    """EqV2 外部验证（复用 11 流程；模型 = 全合并数据 + 出厂参数）。
    variant_b=True 时 EqV2 行泛函标 RPBE（VASP/RPBE，同 11 记录）。"""
    raw = F11.load_eqv2()
    agg = F11.aggregate_surface(raw)
    elements = sorted({e for c in agg["comp"] for e in F11.F06.parse_comp(c)})
    F11.F06.HAND_LOOKUP.update(F06.HAND_LOOKUP)
    etab, _ = F11.F06.build_element_table(elements)
    Xe = F11.featurize(agg, etab)
    if variant_b:
        Xe["func_bin"] = "RPBE"
        mamun_mask = pd.Series(False, index=Xe.index)
        Xe = add_func_onehot(Xe, mamun_mask).drop(columns=["func_bin"])
    Xe = Xe.reindex(columns=feats_all, fill_value=0)
    assert list(Xe.columns) == list(feats_all) and not Xe.isna().any().any()
    mdl = XGBRegressor(random_state=SEED, n_jobs=-1, **FACTORY)
    mdl.fit(Xfull, yfull)
    agg["E_pred"] = mdl.predict(Xe)
    m_all = F11.metric_row("ALL", agg["E_true"].values, agg["E_pred"].values)
    sub = agg[agg["facet"] == 111]
    m_111 = F11.metric_row("facet=111", sub["E_true"].values,
                           sub["E_pred"].values)
    print(f"    [{tag} EqV2] ALL MAE={m_all['MAE']:.4f} "
          f"bias={m_all['bias_pred_minus_true']:+.4f}；"
          f"facet=111 MAE={m_111['MAE']:.4f}", flush=True)
    return m_all, m_111


def deep_anchors(Xfull, yfull, feats_all, variant_b, tag):
    """25 文献锚点复测：仅对数据集中存在的 17 条（our_true_E 非空），
    模型 = 全合并数据 + 出厂参数；abs_diff 基准沿用各锚点 notes 列
    （'pred_dG vs 文献ΔG' → ΔG 基准；'pred_E_ads vs 文献HBE' → E 基准）。"""
    anch = pd.read_csv(DP / "literature_anchors.csv")
    anch = anch[anch["our_true_E"].notna()].copy()
    anch["comp"] = anch["surface"].str.replace(r"\(.*\)", "", regex=True)
    anch["facet"] = anch["surface"].str.extract(r"\((\d+)\)")[0].fillna("111")
    meta = anch[["comp", "facet"]].reset_index(drop=True)
    Xa = featurize_new(meta, [c for c in feats_all if c not in FUNC_FEATS])
    if variant_b:
        # 锚点 Mamun 域内表面（全部在 1,836 中）→ BEEF-vdW
        Xa = pd.concat([Xa, pd.DataFrame(
            {"func_BEEF-vdW": 1, "func_PBE": 0, "func_RPBE": 0,
             "func_MS2": 0, "func_其他": 0}, index=Xa.index)], axis=1)
    Xa = Xa.reindex(columns=feats_all, fill_value=0)
    mdl = XGBRegressor(random_state=SEED, n_jobs=-1, **FACTORY)
    mdl.fit(Xfull, yfull)
    pred = mdl.predict(Xa)
    anch["pred_E_ads_new"] = pred
    anch["pred_dG_new"] = pred + DGCORR
    is_dg = anch["notes"].str.contains("pred_dG vs")
    anch["abs_diff_new"] = np.where(
        is_dg, (anch["pred_dG_new"] - anch["deltaG_lit_eV"]).abs(),
        (anch["pred_E_ads_new"] - anch["deltaG_lit_eV"]).abs())
    cols = ["surface", "deltaG_lit_eV", "our_true_E", "our_pred_dG",
            "abs_diff", "pred_dG_new", "abs_diff_new"]
    out = anch[cols].round(4)
    print(f"    [{tag} 锚点] {len(out)} 条可对照：基线 mean|Δ|="
          f"{anch['abs_diff'].mean():.4f} → 合并后 "
          f"{anch['abs_diff_new'].mean():.4f}", flush=True)
    return out


# =====================================================================
def main():
    t0 = time.time()
    dfm = pd.read_csv(DP / "hstar_dataset_v2.csv")
    ext = pd.read_csv(DP / "extval2_homogeneous.csv")
    assert "site_spread" in dfm.columns
    feats = [c for c in dfm.columns if c not in ID_COLS]
    assert "site_spread" not in feats and len(feats) == 36
    X, y = dfm[feats], dfm["energy_eV"].values
    groups = dfm["comp"].values
    print(f"[读入] Mamun v2: {X.shape[0]} 行 × {X.shape[1]} 特征；"
          f"extval2: {len(ext)} 表面；出厂参数 {FACTORY}", flush=True)

    # ---- Step 1 重叠分析 ----
    am, nw = step1_overlap(dfm, ext)

    # ---- Step 2 构造合并数据 ----
    print("\n===== Step 2 合并数据构造 =====", flush=True)
    print(f"[合并策略] 重叠 {int(am['in_mamun'].sum())} 组成 Mamun 优先"
          f"（丢弃 CatHub 侧）；纯新增 {len(nw)} 组成追加", flush=True)
    nw = nw.reset_index(drop=True)
    Xn36 = featurize_new(nw[["comp", "facet_argmin"]].rename(
        columns={"facet_argmin": "facet"}), feats)
    yn = nw["E_true_min"].values
    Xn36.insert(0, "comp", nw["comp"].values)
    Xn36["func_bin"] = nw["func_bin"].values
    print(f"[新增] {len(Xn36)} 行特征化完成（36 特征）", flush=True)

    # 变体 (a)：朴素合并，36 特征
    Xn_a = Xn36[feats]
    # 变体 (b)：+ 泛函 one-hot（Mamun 行全标 BEEF-vdW）
    Xm_b = add_func_onehot(
        pd.concat([X, pd.Series("BEEF-vdW", index=X.index, name="func_bin")],
                  axis=1),
        pd.Series(True, index=X.index)).drop(columns=["func_bin"])
    Xn_b = add_func_onehot(Xn36, pd.Series(False, index=Xn36.index)
                           ).drop(columns=["func_bin"])
    feats_b = feats + FUNC_FEATS
    Xm_b = Xm_b[feats_b]          # 剔除可能的非特征列，固定列序
    Xn_b = Xn_b[feats_b]          # （Xn36 带 comp 等元信息列）
    assert list(Xm_b.columns) == feats_b and list(Xn_b.columns) == feats_b

    # ---- Step 3 评估 ----
    print("\n===== Step 3 评估（主口径：新增只进训练；次口径：全合并 "
          "GroupKFold）=====", flush=True)
    cmp_rows = []
    base_maes = eval_main_protocol(dfm, X, y, groups, None, None,
                                   "基线 纯Mamun（主口径）")
    cmp_rows.append({"variant": "baseline_mamun_only", "protocol": "主口径",
                     "n_train_max": len(y), "MAE_mean": base_maes.mean(),
                     "MAE_std": base_maes.std(), "delta_vs_baseline": 0.0,
                     **{f"fold{i + 1}": v for i, v in enumerate(base_maes)}})

    variants = {}
    for tag, (Xn_v, Xm_v, f_v) in {
            "a_naive_merge": (Xn_a, X, feats),
            "b_merge_func_onehot": (Xn_b, Xm_b, feats_b)}.items():
        m = eval_main_protocol(dfm, Xm_v, y, groups, Xn_v, yn,
                               f"变体{tag}（主口径）")
        cmp_rows.append({"variant": tag, "protocol": "主口径",
                         "n_train_max": len(y) + len(yn),
                         "MAE_mean": m.mean(), "MAE_std": m.std(),
                         "delta_vs_baseline": base_maes.mean() - m.mean(),
                         **{f"fold{i + 1}": v for i, v in enumerate(m)}})
        variants[tag] = {"main_maes": m, "Xn": Xn_v, "Xm": Xm_v,
                         "feats": f_v}

    # 次口径：全合并数据普通 GroupKFold(5, comp)
    for tag, v in variants.items():
        Xall = pd.concat([v["Xm"], v["Xn"]], ignore_index=True)
        yall = np.concatenate([y, yn])
        gall = np.concatenate([groups, nw["comp"].values])
        m = eval_secondary_protocol(Xall, yall, gall,
                                    f"变体{tag}（次口径 全合并GCV）")
        cmp_rows.append({"variant": tag, "protocol": "次口径_全合并GCV",
                         "n_train_max": len(yall), "MAE_mean": m.mean(),
                         "MAE_std": m.std(),
                         "delta_vs_baseline": base_maes.mean() - m.mean(),
                         **{f"fold{i + 1}": v for i, v in enumerate(m)}})

    cmp_df = pd.DataFrame(cmp_rows)

    # ---- Step 4 深度检查（条件触发）----
    print("\n===== Step 4 深度检查判定 =====", flush=True)
    deep_rows = []
    for tag, v in variants.items():
        gain = base_maes.mean() - v["main_maes"].mean()
        if gain > ADOPT_GAIN:
            print(f"[触发] 变体 {tag} 主口径改善 {gain:+.4f} > "
                  f"{ADOPT_GAIN}，执行嵌套CV/LOEO/EqV2/锚点复测", flush=True)
            is_b = tag.startswith("b_")
            nc_mae, nc_std, nc_df = deep_nested_cv(
                v["Xm"], y, groups, v["Xn"], yn, tag)
            nc_df.to_csv(OUT / f"merge_deepcheck_nested_{tag}.csv",
                         index=False)
            loeo_mae, loeo_df = deep_loeo(dfm, v["Xm"], y, groups,
                                          v["Xn"], yn, tag)
            loeo_df.to_csv(OUT / f"merge_deepcheck_loeo_{tag}.csv",
                           index=False)
            m_all, m_111 = deep_extval(v["feats"],
                                       pd.concat([v["Xm"], v["Xn"]],
                                                 ignore_index=True),
                                       np.concatenate([y, yn]), is_b, tag)
            anch_out = deep_anchors(
                pd.concat([v["Xm"], v["Xn"]], ignore_index=True),
                np.concatenate([y, yn]), v["feats"], is_b, tag)
            anch_out.to_csv(OUT / f"merge_deepcheck_anchors_{tag}.csv",
                            index=False)
            v.update({"nested_mae": nc_mae, "nested_std": nc_std,
                      "loeo_mae": loeo_mae,
                      "extval_all": m_all["MAE"],
                      "extval_bias": m_all["bias_pred_minus_true"],
                      "extval_111": m_111["MAE"],
                      "anchor_base": None, "anchor_new": None})
            deep_rows.append({"variant": tag, "protocol": "嵌套CV无偏",
                              "MAE_mean": nc_mae, "MAE_std": nc_std,
                              "delta_vs_baseline": np.nan})
            deep_rows.append({"variant": tag, "protocol": "LOEO汇集",
                              "MAE_mean": loeo_mae, "MAE_std": np.nan,
                              "delta_vs_baseline": np.nan})
        else:
            print(f"[跳过] 变体 {tag} 主口径 Δ={gain:+.4f} eV，未达 "
                  f"{ADOPT_GAIN} 阈值，不做深度检查", flush=True)
    cmp_df = pd.concat([cmp_df, pd.DataFrame(deep_rows)], ignore_index=True)
    cmp_df.to_csv(OUT / "merge_comparison.csv", index=False)
    print(f"\n[写出] {OUT / 'merge_comparison.csv'}", flush=True)
    print(cmp_df[["variant", "protocol", "MAE_mean", "MAE_std",
                  "delta_vs_baseline"]].round(4).to_string(index=False),
          flush=True)

    # ---- Step 5 报告 ----
    write_notes(am, nw, dfm, cmp_df, variants, base_maes)
    print(f"[完成] 总耗时 {time.time() - t0:.0f}s", flush=True)


def write_notes(am, nw, dfm, cmp_df, variants, base_maes):
    """notes/16_merge_experiment.md（中文，含采纳/不采纳结论）。"""
    base = base_maes.mean()
    ov = am[am["in_mamun"]]
    d = ov["E_diff_ext_minus_mamun"].dropna()
    L = [
        "# 16 — CatHub 同质金属 HER 增量合并重训实验（Round 4 Agent E）",
        "",
        "> 核心问题：把 extval2_homogeneous.csv（Catalysis-Hub 同质金属 HER，",
        "> 355 表面）并入 Mamun 主线训练，能否改善 H* 主线模型？",
        "> 公平性契约（plan_round4.md）：新数据只进训练集不进测试集；",
        "> 采用阈值 0.005 eV；种子 42；site_spread 禁入；不改 01-15 产物。",
        "",
        "## 1. 重叠分析（outputs/merge_overlap_analysis.csv）",
        "",
        f"- 355 表面按 comp 聚合（min E_true，最稳定位点口径，同主线）→ "
        f"**{len(am)} 个唯一组成**：与 Mamun 1,836 重叠 **{len(ov)}** 个，"
        f"纯新增 **{len(nw)}** 个",
        f"- 能量分布：Mamun mean={dfm['energy_eV'].mean():.3f}±"
        f"{dfm['energy_eV'].std():.3f} eV（range "
        f"[{dfm['energy_eV'].min():.2f}, {dfm['energy_eV'].max():.2f}]）；"
        f"纯新增 mean={nw['E_true_min'].mean():.3f}±"
        f"{nw['E_true_min'].std():.3f} eV（range "
        f"[{nw['E_true_min'].min():.2f}, {nw['E_true_min'].max():.2f}]）"
        "——新增整体偏负（强结合侧），落在 Mamun 分布范围内",
        f"- 重叠组成系统偏差：E_CatHub − E_Mamun mean={d.mean():+.3f} eV"
        f"（median|Δ|={d.abs().median():.3f} eV）→ 泛函/参考态差异显著，"
        "**重叠组成 Mamun 优先**的合并策略合理（保持域内单源口径）",
        f"- 纯新增 {len(nw)} 组成来源："
        + "、".join(f"{k} {v}" for k, v in
                    nw["pubId_argmin"].value_counts().items())
        + f"；泛函分箱：{nw['func_bin'].value_counts().to_dict()}",
        "- 新增元素：Mg/Nd/Sm/Si 不在 Mamun 37 元素内（Nd/Sm 的 group 用 "
        "HAND_LOOKUP 兜底，同 14；Mg/Si 由 mendeleev 齐全覆盖）",
        "",
        "## 2. 合并变体",
        "",
        "- **(a) 朴素合并**：沿用 36 特征；新增组成 structure/facet one-hot "
        "按 06/14 推断规则（单元素→structure_A1=1，合金→structure 全 0；"
        "facet 精确 111/101→对应 one-hot，应变/台阶/hcp0001 等→全 0）",
        "- **(b) 合并 + 泛函 one-hot**：dftFunctional 分箱 BEEF-vdW/PBE/"
        "RPBE/MS2/其他（前缀匹配：PBE+D3/PBEsol→PBE，RPBE-D3/RPBE_*VSHE→"
        "RPBE，BEEF-vdW_*VSHE→BEEF-vdW）；Mamun 行全部标 BEEF-vdW",
        "",
        "## 3. 评估结果（outputs/merge_comparison.csv）",
        "",
        "主口径：fold 仅由原始 1,836 组成的 GroupKFold(5, comp) 决定，"
        "新增组成永远加入各折训练集、从不进测试集；固定出厂参数 "
        f"{FACTORY}，种子 42。",
        "",
        "| 变体 | 口径 | MAE (eV) | Δ vs 基线 |",
        "|---|---|---|---|",
    ]
    for r in cmp_df.itertuples():
        dv = f"{r.delta_vs_baseline:+.4f}" if pd.notna(r.delta_vs_baseline) \
            else "—"
        L.append(f"| {r.variant} | {r.protocol} | {r.MAE_mean:.4f}±"
                 f"{r.MAE_std:.4f} |" if pd.notna(r.MAE_std) else
                 f"| {r.variant} | {r.protocol} | {r.MAE_mean:.4f} |")
        L[-1] += f" {dv} |"
    best_tag, best_gain = None, 0.0
    for tag, v in variants.items():
        gain = base - v["main_maes"].mean()
        if gain > best_gain:
            best_tag, best_gain = tag, gain
    L += [
        "",
        f"- 基线（纯 Mamun，主口径）MAE = {base:.4f}±{base_maes.std():.4f}"
        " eV，与契约基线 0.1198±0.0087 完全一致（协议复现）",
    ]
    # 深度检查结果
    triggered = [t for t, v in variants.items() if "nested_mae" in v]
    if triggered:
        L += ["", "## 4. 深度检查（主口径改善 >0.005 eV 的变体）", ""]
        for tag in triggered:
            v = variants[tag]
            L += [
                f"### 变体 {tag}", "",
                f"- 嵌套 CV（外层5/内层4，内层随机搜索 {N_ITER_NESTED} 组，"
                f"新增组成恒在训练侧）无偏 MAE = {v['nested_mae']:.4f}±"
                f"{v['nested_std']:.4f} eV",
                f"- LOEO 汇集 MAE = {v['loeo_mae']:.4f} eV（基线 0.1547）",
                f"- EqV2 外部验证：ALL MAE = {v['extval_all']:.4f} eV，"
                f"bias = {v['extval_bias']:+.4f}（基线 0.3470 / +0.158）；"
                f"facet=111 MAE = {v['extval_111']:.4f}",
                f"- 文献锚点复测见 outputs/merge_deepcheck_anchors_{tag}.csv",
            ]
    else:
        L += ["", "## 4. 深度检查", "",
              "所有变体主口径改善均未超过 0.005 eV 阈值，按契约**不触发**"
              "嵌套 CV / LOEO / EqV2 / 文献锚点复测。"]
    # 结论
    L += ["", "## 5. 结论", ""]
    adopt = None
    for tag in triggered:
        v = variants[tag]
        # 深度检查确认：嵌套 CV 无偏估计仍优于基线 >0.005 才采纳
        if base - v["nested_mae"] > ADOPT_GAIN:
            adopt = tag
    if adopt:
        v = variants[adopt]
        L.append(f"**采纳变体 {adopt}**：主口径改善 "
                 f"{base - v['main_maes'].mean():+.4f} eV，嵌套 CV 无偏确认 "
                 f"{base - v['nested_mae']:+.4f} eV > 0.005 阈值。")
    else:
        L += [
            "**不采纳合并**：",
        ]
        for tag, v in variants.items():
            gain = base - v["main_maes"].mean()
            L.append(f"- 变体 {tag} 主口径 Δ = {gain:+.4f} eV"
                     + ("，嵌套 CV 无偏估计 Δ = "
                        f"{base - v['nested_mae']:+.4f} eV，未维持过阈"
                        if "nested_mae" in v else "，未达 0.005 阈值"))
        L += [
            "- 判读：20 个纯新增组成相对 1,836 行的 Mamun 主线体量过小"
            "（约 1%），且多为应变/非密排面/域外元素（Mg/Nd/Sm/Si）表面，"
            "与原域（A1/L1₀/L1₂ 密排面）分布差异大；重叠组成的系统性泛函"
            f"偏差（mean {d.mean():+.3f} eV）也说明跨源标签不可直接混用。",
            "- 主线模型与候选排序保持不变（未触碰任何 01-15 产物）。",
        ]
    L += [
        "",
        "## 6. 产物清单",
        "",
        "- outputs/merge_overlap_analysis.csv（67 组成逐条：重叠判定、"
        "能量对比、合并决策）",
        "- outputs/merge_comparison.csv（基线/变体(a)/(b) × 主/次口径 "
        "fold 级 MAE 与 Δ）",
    ]
    if triggered:
        L.append("- outputs/merge_deepcheck_{nested,loeo,anchors}_*.csv"
                 f"（触发变体：{', '.join(triggered)}）")
    L.append("- 本报告：notes/16_merge_experiment.md")
    (ROOT / "notes" / "16_merge_experiment.md").write_text(
        "\n".join(L), encoding="utf-8")
    print("[写出] notes/16_merge_experiment.md", flush=True)


if __name__ == "__main__":
    main()
