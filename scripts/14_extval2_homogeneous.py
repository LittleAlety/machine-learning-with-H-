# -*- coding: utf-8 -*-
"""
14_extval2_homogeneous.py — 第二外部验证集：Catalysis-Hub 同质金属 HER 子集（零重训）

决策背景（用户授权）：主线保持 Mamun 单源纯净，不合并重训；把 Cathub 非 Mamun 池中的
同质金属 HER 条目做成第二个外部验证集（notes/09 的增量池盘点为筛选依据）。

数据源：data_processed/cathub_hstar_extra.csv（13_cathub_fetch.py 产物，多 pass 合并）。
纳入（逐项判断）：
  - BoesAdsorption2018（fcc(111) 纯金属 Ag/Au/Cu/Ir/Pd/Pt/Rh，RPBE）
  - TangModeling2020（显式水 HER，纯金属 comp 子集，BEEF-vdW）
  - HansenFirst2018（金属/表面合金密排面，BEEF-vdW，方程 k=2 需 E/2 归一）
  - 其他纯金属小集：MontoyaThe2015、TangFrom2020、Gauthierrole2021、
    SchumannSelectivity2018、ClarkInfluence2018、PengTrends2022、YangIntrinsic2016、
    DStructure2024、WangAchieving2021、SharadaAdsorption2019 等逐项过纯度/反应过滤
排除（异质体系，不混入）：Zhang 系硼化物/MBene、Yohannes 氮化物、Dickens/Kim/JungRuO2
  氧化物、Jung/Hossain M-N-C、Dheer 分子酶簇、E.Molecular 分子催化剂、Bukas C8O3 等。

清洗（每步记录剔除数）：
  1. 反应身份过滤（05 同规则 "0.5H2(g) + * -> H*" 的化学计量推广）：方程须为
     k·(0.5H2(g)+*) -> k·H* 的纯 H 吸附（两侧物种仅 H2(g)/*/H* 且 H 原子配平），
     记录覆盖倍数 k，E_perH = E_reaction / k（已实测 Cathub reactionEnergy 为
     按方程式书写的总反应能：Mamun 内 2H 方程 E≈2×1H 方程，median|Δ|≈0.21 eV）；
     排除共吸附方程（含 N2/CH4/H2O/H2S/CO* 等）与不平衡方程（'H2(g) -> H*'）。
  2. 组成规范化（同 05：元素字母序 + gcd 约简）；仅保留纯金属/金属间组成
     （含 H,B,C,N,O,F,P,S,Cl,Se,Br,I 等非金属者一律排除——同时滤掉 H 预覆盖表面）。
  3. 能量物理范围：|E_perH| > 5 eV 剔除（Boes 共吸附 +22 eV 类异常已被步骤1排除，
     此步兜底）；再做全局稳健离群检查（median ± 8×1.4826·MAD），剔除并记录。
  4. 按 (pubId, comp, facet) 聚合取最低 E_perH（最稳定位点口径，同 11/v1）。

特征化（复用 06，口径与 11 对齐）：
  - structure one-hot：单元素组成 → structure_A1=1（训练集中 M12 超胞≡A1(111)，
    纯金属 fcc(111) 与之同模式）；合金 → 全 0（表面计量模式未知，不臆造）；
  - facet one-hot：111→facet_111=1、101→facet_101=1，其余晶面（211/100/001/0001/…）
    均置 0（训练域外，模型外推），与 11 的缺失处理一致。

零重训验证：hstar_best_params.json 的 XGBoost_tuned（lr=0.03/depth=7/n=400/sub=0.8，
种子 42，主线 1,836 行全量训练）直接预测；报 MAE/RMSE/R²/bias/去偏差 MAE，
分层：全体 / 按 pub / 仅 k=1 严格稀释 / 域内（单元素且 facet=111）/ 按泛函；
与 EqV2 跨库验证（MAE 0.347, bias +0.158，读自 extval_metrics.csv）对照。

产物：data_processed/extval2_homogeneous.csv（清洗后行级子集 + 预测）、
     outputs/extval2_metrics.csv、figures/extval2_parity.png（300dpi 低饱和暖色）
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
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "figures"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]  # 同 07/11
C_MAIN = "#B35C24"   # 暖色主色
C_ALT = "#DFB27E"    # 低饱和暖色
C_BG = "#E8D9C6"     # 浅暖灰（Mamun OOF 背景）
plt.rcParams.update({"savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# 非金属块名单（组成纯度过滤；Si/Ge/Sb/Te 等类金属不在其中，但本池未出现）
NONMETALS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I",
             "He", "Ne", "Ar", "Kr", "Xe"}

# ---- 复用 06 的特征化函数（同源保证口径一致，同 11）----
_spec = importlib.util.spec_from_file_location(
    "hstar_features", ROOT / "scripts" / "06_hstar_features.py")
F06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F06)

# f 区元素 mendeleev 缺 group_id，06 的 HAND_LOOKUP 只备了 Tc → 本地补齐（镧系按惯例记 3 族，
# 仅在 mendeleev 缺失时启用；这些元素本就在 Mamun 37 元素域外，仅作粗粒度位置编码）
F06.HAND_LOOKUP.update({
    "Nd": dict(en=1.14, radius=181.0, group=3, period=6, ie1=5.469, d_el=0),
    "Sm": dict(en=1.17, radius=180.0, group=3, period=6, ie1=5.631, d_el=0),
})

COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def canon_comp(formula):
    """同 05：元素字母序 + gcd 计量约简（Sn14Ir6 → Ir3Sn7）；无法解析返回 None。"""
    toks = COMP_TOKEN.findall(formula or "")
    if not toks or "".join(a + b for a, b in toks) != formula:
        return None
    d = {}
    for el, n in toks:
        d[el] = d.get(el, 0) + int(n or 1)
    g = reduce(gcd, d.values())
    return "".join(f"{el}{d[el]//g if d[el]//g > 1 else ''}" for el in sorted(d))


def parse_pure_h(eq):
    """判定方程是否为 k·(0.5H2(g)+*) -> k·H* 纯 H 吸附（05 规则的计量推广）。
    返回 H* 计量数 k（float）；非纯 H / 不配平 / 含共吸附物种 → None。"""
    if "->" not in str(eq):
        return None
    lhs, rhs = str(eq).split("->")

    def parse_side(s):
        out = {}
        for term in s.split("+"):
            m = re.match(r"^\s*(-?[\d.]*)\s*([A-Za-z0-9*()]+)\s*$", term)
            if not m:
                return None
            c = m.group(1)
            coef = float(c) if c not in ("", "+", "-") else (-1.0 if c == "-" else 1.0)
            out[m.group(2)] = out.get(m.group(2), 0.0) + coef
        return out

    lsp, rsp = parse_side(lhs), parse_side(rhs)
    if lsp is None or rsp is None:
        return None
    if set(lsp) - {"H2(g)", "*"} or set(rsp) != {"H*"}:
        return None                       # 共吸附 / 非 H 产物
    if any(c <= 0 for c in list(lsp.values()) + list(rsp.values())):
        return None                       # 负系数（差式方程）
    if "H2(g)" not in lsp or abs(rsp["H*"] - 2 * lsp["H2(g)"]) > 1e-6:
        return None                       # H 原子不配平（如 'H2(g) -> H*'）
    return rsp["H*"]


def metric_row(name, y_true, y_pred):
    res = y_pred - y_true
    bias = float(res.mean())
    return {"stratum": name, "n": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            "bias_pred_minus_true": bias,
            "MAE_after_bias_shift": float(np.abs(res - bias).mean())}


def clean_subset():
    """cathub_hstar_extra.csv → 清洗后同质金属子集（含逐步剔除日志）。"""
    df = pd.read_csv(DP / "cathub_hstar_extra.csv")
    n0 = len(df)
    log = [f"[清洗] 原始非 Mamun 行 {n0}"]

    # 步骤1：反应身份过滤
    df["k_H"] = df["equation"].map(parse_pure_h)
    bad_eq = df["k_H"].isna()
    log.append(f"[清洗] 步骤1 反应身份（纯 H 吸附 k·(0.5H2+*)->k·H*）："
               f"剔除 {int(bad_eq.sum())}（共吸附/不配平方程），余 {int((~bad_eq).sum())}")
    df = df[~bad_eq].copy()
    df["E_perH"] = df["energy_eV"] / df["k_H"]

    # 步骤2：组成规范化 + 纯度
    df["comp"] = df["comp_raw"].map(canon_comp)
    bad_comp = df["comp"].isna()
    log.append(f"[清洗] 步骤2a 组成可解析：剔除 {int(bad_comp.sum())}，余 {int((~bad_comp).sum())}")
    df = df[~bad_comp].copy()
    elsets = df["comp"].map(lambda c: set(F06.parse_comp(c)))
    hetero = elsets.map(lambda s: bool(s & NONMETALS))
    log.append(f"[清洗] 步骤2b 同质金属纯度（排除含非金属组成：硼/氮/氧化物、M-N-C、"
               f"分子催化剂、H 预覆盖表面）：剔除 {int(hetero.sum())}，余 {int((~hetero).sum())}")
    df = df[~hetero].copy()

    # 步骤3：能量物理范围
    bad_e = df["E_perH"].abs() > 5.0
    log.append(f"[清洗] 步骤3a 物理范围 |E_perH|≤5 eV：剔除 {int(bad_e.sum())}"
               f"（{df.loc[bad_e, 'E_perH'].round(2).tolist()[:8]}），余 {int((~bad_e).sum())}")
    df = df[~bad_e].copy()
    med = df["E_perH"].median()
    mad = (df["E_perH"] - med).abs().median() * 1.4826
    z = (df["E_perH"] - med).abs() / max(mad, 1e-9)
    outl = z > 8
    log.append(f"[清洗] 步骤3b 稳健离群（median±8·MAD，median={med:.3f}, "
               f"MAD={mad:.3f} eV）：剔除 {int(outl.sum())}，余 {int((~outl).sum())}")
    df = df[~outl].copy()

    df["facet"] = df["facet"].astype(str)
    df["n_elements"] = df["comp"].map(lambda c: len(F06.parse_comp(c)))
    for l in log:
        print(l, flush=True)
    print("[清洗] 保留来源分布:", df.groupby("pubId").size().to_dict(), flush=True)
    return df


def featurize(agg, etab, feats):
    """按 06/11 口径构造特征；structure/facet one-hot 规则见模块 docstring。"""
    comp_dicts = {c: F06.parse_comp(c) for c in agg["comp"].unique()}
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
        f.update({"structure_A1": 1 if len(els) == 1 else 0,
                  "structure_L10": 0, "structure_L12": 0,
                  "facet_101": 1 if row["facet"] == "101" else 0,
                  "facet_111": 1 if row["facet"] == "111" else 0})
        rows.append(f)
    Xe = pd.DataFrame(rows, index=agg.index)
    return Xe.reindex(columns=feats, fill_value=0)


def main():
    # ---- 1. 出厂模型（与 11 完全一致）----
    dfm = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in dfm.columns if c not in ID_COLS]
    X, y = dfm[feats], dfm["energy_eV"].values
    params = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]
    print(f"[出厂模型] XGBoost_tuned {params}，全量 {X.shape[0]} 行 × {X.shape[1]} 特征"
          f"（种子 {SEED}）", flush=True)
    model = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    model.fit(X, y)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    oof_mae = mean_absolute_error(y, oof)
    print(f"[内部对照] Mamun OOF MAE={oof_mae:.3f} eV", flush=True)

    # ---- 2. 清洗 + 聚合 ----
    sub = clean_subset()
    agg = (sub.groupby(["pubId", "comp", "facet"], as_index=False)
              .agg(E_true=("E_perH", "min"), n_rows=("E_perH", "size"),
                   k_min=("k_H", "min"), k_max=("k_H", "max"),
                   E_spread=("E_perH", lambda s: float(s.max() - s.min())),
                   dftFunctional=("dftFunctional", "first")))
    print(f"[聚合] 唯一 (pub,comp,facet) 表面 {len(agg)} 个"
          f"（k>1 覆盖倍数涉及 {(agg['k_max'] > 1).sum()} 个表面）", flush=True)

    elements = sorted({e for c in agg["comp"] for e in F06.parse_comp(c)})
    etab, fb = F06.build_element_table(elements)
    print(f"[兜底] {'未启用' if not fb else fb}；外部集元素 {elements}", flush=True)
    mamun_elems = {e for c in dfm["comp"] for e in F06.parse_comp(c)}
    Xe = featurize(agg, etab, feats)
    assert list(Xe.columns) == feats and not Xe.isna().any().any()

    agg["E_pred"] = model.predict(Xe)
    agg["residual"] = agg["E_pred"] - agg["E_true"]
    agg["n_elements"] = agg["comp"].map(lambda c: len(F06.parse_comp(c)))
    agg["in_domain"] = (agg["n_elements"] == 1) & (agg["facet"] == "111")
    agg["elems_in_mamun"] = agg["comp"].map(
        lambda c: set(F06.parse_comp(c)) <= mamun_elems)
    agg.to_csv(DP / "extval2_homogeneous.csv", index=False)
    print(f"[写出] {DP / 'extval2_homogeneous.csv'}（{len(agg)} 行）", flush=True)

    # ---- 3. 指标（分层）----
    yt, yp = agg["E_true"].values, agg["E_pred"].values
    mrows = [metric_row("ALL", yt, yp)]
    k1 = agg[agg["k_max"] == 1]
    mrows.append(metric_row("仅k=1（严格稀释）", k1["E_true"].values, k1["E_pred"].values))
    ind = agg[agg["in_domain"]]
    mrows.append(metric_row("域内（单元素+facet=111）", ind["E_true"].values,
                            ind["E_pred"].values))
    ine = agg[agg["elems_in_mamun"]]
    mrows.append(metric_row("元素全在 Mamun 37 内", ine["E_true"].values,
                            ine["E_pred"].values))
    for pid, s in agg.groupby("pubId"):
        if len(s) >= 3:
            mrows.append(metric_row(f"pub={pid}", s["E_true"].values,
                                    s["E_pred"].values))
    for fn, s in agg.groupby("dftFunctional"):
        if len(s) >= 3:
            mrows.append(metric_row(f"func={fn}", s["E_true"].values,
                                    s["E_pred"].values))
    metrics = pd.DataFrame(mrows)
    metrics.to_csv(OUT / "extval2_metrics.csv", index=False)
    print(f"[写出] {OUT / 'extval2_metrics.csv'}", flush=True)
    print("\n===== 第二外部验证（同质金属 Cathub）指标 =====")
    print(metrics.round(3).to_string(index=False), flush=True)

    # EqV2 对照
    ev = pd.read_csv(DP / "extval_metrics.csv")
    ev_all = ev[ev["stratum"] == "ALL"].iloc[0]
    print(f"\n[对照] EqV2 跨库验证：MAE={ev_all['MAE']:.3f}, "
          f"bias={ev_all['bias_pred_minus_true']:+.3f}（n={int(ev_all['n'])}）", flush=True)

    # ---- 4. 图：parity + 残差分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    ax = axes[0]
    ax.scatter(y, oof, s=7, alpha=0.25, color=C_BG, edgecolors="none",
               label=f"Mamun in-house OOF (MAE={oof_mae:.3f} eV)")
    ax.scatter(ind["E_true"], ind["E_pred"], s=26, alpha=0.85, marker="o",
               color=C_MAIN, edgecolors="none",
               label=f"Cathub pure-metal (111), in-domain (n={len(ind)})")
    ax.scatter(agg.loc[~agg["in_domain"], "E_true"], agg.loc[~agg["in_domain"], "E_pred"],
               s=22, alpha=0.7, marker="s", color=C_ALT, edgecolors="none",
               label=f"Cathub alloys / other facets (n={(~agg['in_domain']).sum()})")
    lim = [min(y.min(), yt.min(), yp.min()) - 0.3,
           max(y.max(), yt.max(), yp.max()) + 0.3]
    ax.plot(lim, lim, "--", color="#8c8c8c", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    a0 = mrows[0]
    ax.set_xlabel(r"True $E_{ads}$(H*) per H (eV, Catalysis-Hub)")
    ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"Extval-2 homogeneous: MAE={a0['MAE']:.3f} eV, "
                 f"$R^2$={a0['R2']:.3f}, bias={a0['bias_pred_minus_true']:+.3f} eV",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    ax.hist(oof - y, bins=60, alpha=0.6, color=C_BG, density=True,
            label="Mamun OOF residuals")
    ax.hist(ind["residual"], bins=20, alpha=0.75, color=C_MAIN, density=True,
            label="Extval-2 in-domain residuals")
    ax.hist(agg.loc[~agg["in_domain"], "residual"], bins=20, alpha=0.6,
            color=C_ALT, density=True, label="Extval-2 OOD residuals")
    ax.axvline(0, color="#8c8c8c", ls="--", lw=1)
    ax.set_xlabel(r"Residual = pred $-$ true (eV)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (systematic bias)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "extval2_parity.png")
    plt.close(fig)
    print(f"[写出] {FIGDIR / 'extval2_parity.png'}", flush=True)


if __name__ == "__main__":
    main()
