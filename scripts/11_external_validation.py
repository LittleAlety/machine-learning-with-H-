# -*- coding: utf-8 -*-
"""
11_external_validation.py — H* 出厂模型的跨库外部验证（SPEC 第 7 节）

流程：
1. 出厂模型：data_processed/hstar_dataset_v2.csv + hstar_best_params.json 的
   XGBoost 最优参数，全量训练（种子 42），与 07 的特征/口径完全一致。
2. 外部数据：EqV2-HER-Discovery（Oguz et al., ACS Catal. 2025, 15, 19461;
   GitHub ergroup/EqV2-HER-Discovery, VASP/RPBE）：
   - data_raw/eqv2_361.csv：361 条 DFT 全弛豫 H 吸附能（最终候选，能量区间窄）
   - data_raw/eqv2_500.csv：500 条 AdsorbML/EqV2 单点 DFT H 吸附能（区间宽）
   能量口径：E_ads(H*) 参考 1/2 H2(g)（OC20/AdsorbML 约定，与 Mamun
   ref_ads_eng 同口径；361 集能量聚集在 -0.40~-0.10 eV、区间中点 ≈ -0.25 eV，
   与 ΔG_H* = E_ads + 0.24 ≈ 0 的 HER 最优筛选逻辑自洽，佐证参考态一致）。
3. 特征化（复用 06 的函数，保证口径一致）：
   - comp 规范化同 05（元素字母序 + gcd 约简，如 Sn14Ir6 → Ir3Sn7）
   - 元素描述符：06 的 build_element_table / wmean / wstd（mendeleev 同源）
   - 结构 one-hot 处理规则：EqV2 体相为任意晶系金属间化合物，无 A1/L1₀/L1₂
     结构类型标注 → structure_A1/L10/L12 全部置 0（=缺失，非"属于某类"）；
     facet one-hot：Surface=111 → facet_111=1；其余晶面（100/110/210/211/221）
     两个 facet one-hot 均置 0（=训练域外，模型外推）。该规则造成的外部验证
     失真在 notes/07 中分层讨论（仅 facet=111 子集是严格的域内比较）。
   - 同一 (comp, facet) 多条记录（不同吸附位点/终端）取最低 E_ads —— 与
     H* 线"每表面最稳定位点"口径一致；361（全弛豫）与 500（单点）分源报告。
4. 零重训预测，报 MAE/RMSE/R²、系统性偏差 mean(pred-true)、去偏差平移后
   MAE，按来源/晶面/元素/含 Sb 与否分层（Sb 不在 Mamun 37 元素内）。
产物：data_processed/extval_results.csv、data_processed/extval_metrics.csv、
      figures/extval_parity.png
"""
import importlib.util
import json
import re
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
RAW = ROOT / "data_raw"
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]  # 同 07
C_MAIN = "#B35C24"   # 暖色主色
C_ALT = "#DFB27E"    # 低饱和暖色
C_BG = "#E8D9C6"     # 浅暖灰（Mamun OOF 背景）
C_DEEP = "#7A4A2B"
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- 复用 06 的特征化函数（同源保证口径一致）----
_spec = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
F06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F06)

COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
ELEMS_EQV2 = ["Ir", "Pt", "Ru", "Ni", "Co", "Fe", "Mn", "Ta", "Ti", "Nb",
              "Mo", "Cu", "Sn", "Sb"]  # EqV2 论文化学空间


def normalize_comp(formula):
    """同 05：解析化学式 → 元素字母序 + gcd 计量约简（Sn14Ir6 → Ir3Sn7）。"""
    cd = {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(formula)}
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def load_eqv2():
    """读取两个 EqV2 文件 → 统一长表（src, comp, facet, E_true），去 NaN。"""
    d361 = pd.read_csv(RAW / "eqv2_361.csv")
    d361 = d361.rename(columns={"Full_optimized_DFT_H_ads_energy": "E_true",
                                "Surface": "facet", "symbol": "formula"})
    d361["src"] = "eqv2_361_relaxed"
    d500 = pd.read_csv(RAW / "eqv2_500.csv")
    d500 = d500.rename(columns={"single_point_DFT_H_ads_E": "E_true",
                                "Surface": "facet", "symbol": "formula"})
    d500["src"] = "eqv2_500_sp"
    cols = ["src", "formula", "facet", "E_true"]
    both = pd.concat([d361[cols], d500[cols]], ignore_index=True)
    n0 = len(both)
    both = both.dropna(subset=["formula", "E_true"]).reset_index(drop=True)
    print(f"[读入] EqV2 原始 {n0} 行，剔除 formula/E_true 缺失后 {len(both)} 行")
    both["comp"] = both["formula"].map(normalize_comp)
    both["facet"] = both["facet"].astype(int)
    return both


def aggregate_surface(df):
    """同 (src, comp, facet) 多条（不同位点/终端）取最低 E_ads（最稳定位点口径）。"""
    agg = (df.groupby(["src", "comp", "facet"], as_index=False)
             .agg(E_true=("E_true", "min"), n_sites=("E_true", "size"),
                  E_spread=("E_true", lambda s: float(s.max() - s.min()))))
    return agg


def featurize(agg, etab):
    """按 06 口径构造特征；structure/facet one-hot 规则见模块 docstring。"""
    comp_dicts = {c: F06.parse_comp(c) for c in agg["comp"].unique()}
    rows = []
    for _, row in agg.iterrows():
        d = comp_dicts[row["comp"]]
        els = list(d)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        feats = {"n_elements": len(els)}
        for p in F06.PROPS:
            v = np.array([etab[e][p] for e in els])
            feats[f"{p}_wmean"] = F06.wmean(v, w)
            feats[f"{p}_max"] = float(v.max())
            feats[f"{p}_min"] = float(v.min())
            feats[f"{p}_range"] = float(v.max() - v.min())
            feats[f"{p}_wstd"] = F06.wstd(v, w)
        # 结构类型缺失 → 全 0；晶面仅 111 在训练域内
        feats.update({"structure_A1": 0, "structure_L10": 0, "structure_L12": 0,
                      "facet_101": 0,
                      "facet_111": 1 if row["facet"] == 111 else 0})
        rows.append(feats)
    return pd.DataFrame(rows, index=agg.index)


def metric_row(name, y_true, y_pred):
    res = y_pred - y_true
    bias = float(res.mean())
    return {"stratum": name, "n": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            "bias_pred_minus_true": bias,
            "MAE_after_bias_shift": float(np.abs(res - bias).mean())}


def main():
    # ---- 1. 出厂模型 ----
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    best = json.loads((DP / "hstar_best_params.json").read_text())
    params = best["xgb_best_params"]
    print(f"[出厂模型] XGBoost_tuned {params}，全量 {X.shape[0]} 行 × "
          f"{X.shape[1]} 特征重训（种子 {SEED}）")
    model = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    model.fit(X, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    oof_mae = mean_absolute_error(y, oof)
    print(f"[内部对照] Mamun OOF MAE={oof_mae:.3f} eV（07 记录 "
          f"{best['random_cv_MAE']:.3f}）")

    # ---- 2. EqV2 外部集特征化 ----
    raw = load_eqv2()
    agg = aggregate_surface(raw)
    print(f"[聚合] 唯一 (src,comp,facet) 表面 {len(agg)} 个"
          f"（多位点组 {(agg['n_sites'] > 1).sum()} 个，最大位点极差 "
          f"{agg['E_spread'].max():.2f} eV）")
    elements = sorted({e for c in agg["comp"] for e in F06.parse_comp(c)})
    etab, fb = F06.build_element_table(elements)
    print(f"[兜底] {'未启用' if not fb else fb}；外部集元素 {elements}")
    Xe = featurize(agg, etab)
    Xe = Xe.reindex(columns=feats, fill_value=0)
    assert list(Xe.columns) == feats, "外部特征列与训练特征不一致"
    assert not Xe.isna().any().any(), "外部特征存在 NaN"

    agg["E_pred"] = model.predict(Xe)
    agg["residual"] = agg["E_pred"] - agg["E_true"]
    agg["in_domain_facet"] = agg["facet"] == 111
    agg["has_Sb"] = agg["comp"].str.contains("Sb")
    agg.to_csv(DP / "extval_results.csv", index=False)
    print(f"[写出] {DP / 'extval_results.csv'}  ({len(agg)} 行)")

    # ---- 3. 跨库指标（分层）----
    yt, yp = agg["E_true"].values, agg["E_pred"].values
    mrows = [metric_row("ALL", yt, yp)]
    for src, sub in agg.groupby("src"):
        mrows.append(metric_row(f"src={src}", sub["E_true"].values,
                                sub["E_pred"].values))
    sub = agg[agg["in_domain_facet"]]
    mrows.append(metric_row("facet=111（域内晶面）", sub["E_true"].values,
                            sub["E_pred"].values))
    sub = agg[~agg["has_Sb"]]
    mrows.append(metric_row("不含Sb（元素域内）", sub["E_true"].values,
                            sub["E_pred"].values))
    sub = agg[agg["in_domain_facet"] & ~agg["has_Sb"]]
    mrows.append(metric_row("facet=111 且不含Sb", sub["E_true"].values,
                            sub["E_pred"].values))
    sub = agg[agg["has_Sb"]]
    mrows.append(metric_row("含Sb（元素域外）", sub["E_true"].values,
                            sub["E_pred"].values))
    for el in ELEMS_EQV2:
        sub = agg[agg["comp"].map(lambda c: el in F06.parse_comp(c))]
        if len(sub) >= 5:
            mrows.append(metric_row(f"含{el}", sub["E_true"].values,
                                    sub["E_pred"].values))
    metrics = pd.DataFrame(mrows)
    metrics.to_csv(DP / "extval_metrics.csv", index=False)
    print(f"[写出] {DP / 'extval_metrics.csv'}")
    show = metrics.round(3)
    print("\n===== 跨库外部验证指标 =====")
    print(show.to_string(index=False))

    # ---- 4. 图：parity + 残差分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    ax = axes[0]
    ax.scatter(y, oof, s=7, alpha=0.25, color=C_BG, edgecolors="none",
               label=f"Mamun in-house OOF (MAE={oof_mae:.3f} eV)")
    for (flag, mk, col, lab) in [(True, "o", C_MAIN, "EqV2 facet=111 (in-domain)"),
                                 (False, "s", C_ALT, "EqV2 other facets (OOD)")]:
        m = agg["in_domain_facet"] == flag
        ax.scatter(agg.loc[m, "E_true"], agg.loc[m, "E_pred"], s=22, alpha=0.75,
                   marker=mk, color=col, edgecolors="none", label=lab)
    lim = [min(y.min(), yt.min(), yp.min()) - 0.3,
           max(y.max(), yt.max(), yp.max()) + 0.3]
    ax.plot(lim, lim, "--", color="#8c8c8c", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    allm = mrows[0]
    ax.set_xlabel(r"True $E_{ads}$(H*) (eV, EqV2 / Mamun)")
    ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"EqV2 external: MAE={allm['MAE']:.3f} eV, "
                 f"$R^2$={allm['R2']:.3f}, bias={allm['bias_pred_minus_true']:+.3f} eV",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    ax.hist(oof - y, bins=60, alpha=0.6, color=C_BG, density=True,
            label="Mamun OOF residuals")
    ax.hist(agg.loc[agg["in_domain_facet"], "residual"], bins=30, alpha=0.7,
            color=C_MAIN, density=True, label="EqV2 facet=111 residuals")
    ax.hist(agg.loc[~agg["in_domain_facet"], "residual"], bins=30, alpha=0.6,
            color=C_ALT, density=True, label="EqV2 other-facet residuals")
    ax.axvline(0, color="#8c8c8c", ls="--", lw=1)
    ax.set_xlabel(r"Residual = pred $-$ true (eV)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (systematic bias)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "extval_parity.png")
    plt.close(fig)
    print(f"[写出] {FIGDIR / 'extval_parity.png'}")


if __name__ == "__main__":
    main()
