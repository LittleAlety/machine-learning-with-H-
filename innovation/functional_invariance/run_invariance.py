# -*- coding: utf-8 -*-
"""
Phase 3 Workstream I — 跨泛函不变性分析（预注册: config/preregistered_phase3.yaml
workstream_I_invariance；红线 15：先判读规则后实验；公平契约 0.005 eV / 10-10；
seeds [0,1,2,7,13,42,99,123,2024,31337]）。

三域：
  D1 主域 Mamun/BEEF-vdW：data_processed/hstar_features_v3_84feat.csv（1836 行 × 84 特征）
  D2 EqV2/RPBE：data_raw/eqv2_361.csv + eqv2_500.csv，复用 scripts/11 的口径
     （normalize_comp + 同表面取最低 E_ads + structure one-hot=0 + facet_111）
  D3 CatHub/PBE：data_processed/extval2_homogeneous.csv 中 dftFunctional==PBE（203 行，
     已按 14 号脚本口径聚合），facet 取前导数字 → facet_101/111

特征口径验证：重算 1836 主域行的 84 特征与快照逐值比对（0 失配方继续）。
步骤：
  i1 三域重叠组成盘点 → i1_overlap.csv
  i2 单描述符跨域稳定性（逐域一元斜率 + Pearson r，跨域 std 排序）
     → i2_stability.csv + i2_stability_heatmap.png（300dpi 暖色系，英文标注）
  i3 三方案同台（同折同种子，XGBoost 锁死 hstar_best_params.json 出厂 tuned 参数）：
     (a) naive_pooled  (b) domain_indicator  (c) IRM_lite（线性 IRMv1，λ=100 锁死）
     目标：EqV2 跨库 OOF MAE 相对 v3 基线 0.347 eV 的改善
     → i3_fold_metrics.csv + i3_verdict.json
结局判读（预注册）：A = EqV2 MAE≤0.25 且改善>0.05 eV 且 Spearman ρ≥0.35；否则 B。
产物幂等：重跑覆盖同名文件。
"""
import json
import re
import warnings
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DP = ROOT / "data_processed"
RAW = ROOT / "data_raw"
OUT = ROOT / "innovation" / "functional_invariance"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
XGB_PARAMS = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]
EQV2_BASELINE_MAE = 0.347  # v3 出厂模型零重训 EqV2 跨库 MAE（data_processed/extval_metrics.csv: ALL 0.3469996）
IRM_LAMBDA = 100.0         # 锁死
IRM_EPOCHS = 2000          # 锁死（300 epoch 内收敛，2000 留足裕量）
IRM_LR = 0.01              # 锁死

ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
SUBSHELL_RE = re.compile(r"(\d)([spdf])(\d*)")
SUBSHELL_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}
PROPS13 = ["en", "radius", "group", "period", "ie1", "d_el", "melting_point",
           "mendeleev_number", "covalent_radius", "n_valence", "nd_valence",
           "n_unfilled", "nd_unfilled"]
STATS = ["wmean", "max", "min", "range", "wstd"]
# mendeleev 兜底：6 基线性质沿用 06 脚本 HAND_LOOKUP；Sn 熔点在 mendeleev 1.3.0
# 中因同素异形体返回 None，按主域快照实测值兜底（505.08 K，与 1836 行快照一致）
HAND_LOOKUP_06 = {"Tc": {"en": 2.1, "radius": 135.0, "group": 7, "period": 5,
                         "ie1": 7.11938, "d_el": 5},
                  "Sn": {"melting_point": 505.08}}

CMAP = "YlOrBr"  # 暖色系


# ------------------------------------------------------------ 特征化（同 06/23 口径）
def parse_comp(comp):
    return {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(comp)}


def normalize_comp(formula):
    cd = {el: int(n) if n else 1 for el, n in COMP_TOKEN.findall(formula)}
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def parse_subshells(econf):
    out = {}
    for m in SUBSHELL_RE.finditer(str(econf)):
        n, l = int(m.group(1)), m.group(2)
        occ = int(m.group(3)) if m.group(3) else 1
        out[(n, l)] = out.get((n, l), 0) + occ
    return out


def d_electron_count(econf):
    tail = str(econf).split("]")[-1]
    return sum(int(n) if n else 1 for _, n in re.findall(r"(\d)d(\d*)", tail))


def build_element_table(elements):
    """13 性质元素表（6 基线 + 7 扩展），同 scripts/23 口径。"""
    from mendeleev import element as mde
    table = {}
    for sym in elements:
        e = mde(sym)
        conf = parse_subshells(e.econf)
        nd_val = sum(o for (n, l), o in conf.items()
                     if l == "d" and n == max(nn for (nn, ll) in conf if ll == "d")) \
            if any(l == "d" for _, l in conf) else 0
        n_unf = sum(SUBSHELL_CAP[l] - o for (n, l), o in conf.items()
                    if SUBSHELL_CAP[l] > o)
        nd_unf = sum(SUBSHELL_CAP["d"] - o for (n, l), o in conf.items()
                     if l == "d" and SUBSHELL_CAP["d"] > o)
        try:
            n_val = e.nvalence()
        except Exception:
            n_val = None
        vals = {"en": e.en_pauling, "radius": e.atomic_radius, "group": e.group_id,
                "period": e.period, "ie1": (e.ionenergies or {}).get(1),
                "d_el": d_electron_count(e.econf),
                "melting_point": e.melting_point,
                "mendeleev_number": e.mendeleev_number,
                "covalent_radius": e.covalent_radius, "n_valence": n_val,
                "nd_valence": float(nd_val), "n_unfilled": float(n_unf),
                "nd_unfilled": float(nd_unf)}
        for k, v in list(vals.items()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                fb = HAND_LOOKUP_06.get(sym, {}).get(k)
                vals[k] = float(fb) if fb is not None else np.nan
        table[sym] = {k: float(v) for k, v in vals.items()}
    return table


def comp_stats(d, etab):
    """84 特征中的 78 个组成统计（13 性质 × 5 统计 + 13 mode）。"""
    els = list(d)
    w = np.array([d[e] for e in els], dtype=float)
    w /= w.sum()
    feats = {}
    for p in PROPS13:
        vals = [etab[e][p] for e in els]
        if any(np.isnan(v) for v in vals):
            feats.update({f"{p}_{s}": np.nan for s in STATS})
            feats[f"{p}_mode"] = np.nan
            continue
        v = np.array(vals)
        mu = np.average(v, weights=w)
        feats[f"{p}_wmean"] = float(mu)
        feats[f"{p}_max"] = float(v.max())
        feats[f"{p}_min"] = float(v.min())
        feats[f"{p}_range"] = float(v.max() - v.min())
        feats[f"{p}_wstd"] = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
        i_mode = int(np.argmax([d[e] for e in els]))  # 并列取字母序首元素（同 23）
        feats[f"{p}_mode"] = float(v[i_mode])
    return feats


def featurize_external(df, facet_col, feat_cols, etab):
    """structure one-hot 全 0（外部集无 A1/L10/L12 标注）；facet 前导数字→one-hot。"""
    rows = []
    for _, r in df.iterrows():
        d = parse_comp(r["comp"])
        f = comp_stats(d, etab)
        f["n_elements"] = len(d)
        f.update({"structure_A1": 0, "structure_L10": 0, "structure_L12": 0,
                  "facet_101": 1 if str(r[facet_col]).startswith("101") else 0,
                  "facet_111": 1 if str(r[facet_col]).startswith("111") else 0})
        rows.append(f)
    return pd.DataFrame(rows, index=df.index)[feat_cols]


# ------------------------------------------------------------ 数据装配
def load_domains():
    main = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
    feat_cols = [c for c in main.columns if c not in ID_COLS]
    assert len(feat_cols) == 84

    # EqV2（scripts/11 口径）
    d361 = pd.read_csv(RAW / "eqv2_361.csv").rename(
        columns={"Full_optimized_DFT_H_ads_energy": "E_true",
                 "Surface": "facet", "symbol": "formula"})
    d361["src"] = "eqv2_361_relaxed"
    d500 = pd.read_csv(RAW / "eqv2_500.csv").rename(
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

    # CatHub PBE 系（scripts/14 聚合产物）
    ct = pd.read_csv(DP / "extval2_homogeneous.csv")
    ct = ct[ct["dftFunctional"] == "PBE"].copy().reset_index(drop=True)
    ct["comp"] = ct["comp"].map(normalize_comp)
    ct["facet_str"] = ct["facet"].astype(str)

    # 元素表 + 特征口径自检（重算主域 1836 行 × 78 统计特征，须 0 失配）
    all_comps = pd.concat([main["comp"], eq_agg["comp"], ct["comp"]]).unique()
    elements = sorted({el for c in all_comps for el in parse_comp(c)})
    etab = build_element_table(elements)
    stat_feats = [f"{p}_{s}" for p in PROPS13 for s in STATS] + \
                 [f"{p}_mode" for p in PROPS13]
    n_bad = 0
    for c in main["comp"].unique():
        mine = comp_stats(parse_comp(c), etab)
        r = main[main["comp"] == c].iloc[0]
        for k in stat_feats:
            a, b = mine[k], r[k]
            if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, atol=1e-6):
                n_bad += 1
    print(f"[口径自检] 主域 84 特征重算失配数 = {n_bad}（须为 0）")
    assert n_bad == 0, "特征口径与 v3_84 快照不一致，终止"

    X_eq = featurize_external(eq_agg, "facet", feat_cols, etab)
    X_ct = featurize_external(ct, "facet_str", feat_cols, etab)

    domains = {
        "Mamun_BEEFvdW": {"X": main[feat_cols].reset_index(drop=True),
                          "y": main["energy_eV"].values,
                          "comp": main["comp"].values},
        "EqV2_RPBE": {"X": X_eq.reset_index(drop=True),
                      "y": eq_agg["E_true"].values,
                      "comp": eq_agg["comp"].values},
        "CatHub_PBE": {"X": X_ct.reset_index(drop=True),
                       "y": ct["E_true"].values,
                       "comp": ct["comp"].values},
    }
    for k, v in domains.items():
        print(f"[域] {k}: n={len(v['y'])}, 唯一组成={len(set(v['comp']))}, "
              f"E_ads ∈ [{v['y'].min():.2f}, {v['y'].max():.2f}] eV")
    return domains, feat_cols


# ------------------------------------------------------------ i1 重叠组成
def step_i1(domains):
    names = list(domains)
    comp_sets = {k: set(v["comp"]) for k, v in domains.items()}
    all_overlap = set.intersection(*comp_sets.values())
    print(f"[i1] 三域共同组成数 = {len(all_overlap)}")
    rows = []
    for c in sorted(all_overlap):
        row = {"comp": c}
        for k in names:
            y = domains[k]["y"][np.array(domains[k]["comp"]) == c]
            row[f"{k}_n"] = len(y)
            row[f"{k}_E_ads_mean_eV"] = round(float(y.mean()), 4)
            row[f"{k}_E_ads_min_eV"] = round(float(y.min()), 4)
        rows.append(row)
    ov = pd.DataFrame(rows)
    # 两两配对统计
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = comp_sets[names[i]] & comp_sets[names[j]]
            pair_rows.append({"pair": f"{names[i]} ∩ {names[j]}",
                              "n_overlap_comps": len(inter)})
    pair_df = pd.DataFrame(pair_rows)
    ov.to_csv(OUT / "i1_overlap.csv", index=False)
    pair_df.to_csv(OUT / "i1_pair_counts.csv", index=False)
    print(pair_df.to_string(index=False))
    return ov, pair_df


# ------------------------------------------------------------ i2 单描述符稳定性
def step_i2(domains, feat_cols):
    names = list(domains)
    recs = []
    for f in feat_cols:
        rec = {"feature": f}
        slopes = []
        for k in names:
            x = domains[k]["X"][f].values.astype(float)
            y = domains[k]["y"]
            mask = ~np.isnan(x)
            if mask.sum() < 10 or np.nanstd(x[mask]) < 1e-12:
                rec[f"slope_{k}"] = np.nan
                rec[f"r_{k}"] = np.nan
                slopes.append(np.nan)
                continue
            xm, ym = x[mask], y[mask]
            slope = float(np.polyfit(xm, ym, 1)[0])
            r = float(pearsonr(xm, ym)[0])
            rec[f"slope_{k}"] = slope
            rec[f"r_{k}"] = r
            slopes.append(slope)
        s = np.array(slopes, dtype=float)
        n_valid = int((~np.isnan(s)).sum())
        rec["n_domains_valid"] = n_valid
        if n_valid == 3:
            rec["slope_mean"] = float(np.mean(s))
            rec["slope_std_crossdomain"] = float(np.std(s))
            rec["sign_flips"] = int(np.max(np.sign(s)) - np.min(np.sign(s)) > 0)
        else:  # 外部域常量特征（structure/facet one-hot）不参与稳定性排序
            rec["slope_mean"] = float(np.nanmean(s))
            rec["slope_std_crossdomain"] = np.nan
            rec["sign_flips"] = np.nan
        rec["r_mean"] = float(np.nanmean([rec[f"r_{k}"] for k in names]))
        rec["r_std_crossdomain"] = float(np.nanstd([rec[f"r_{k}"] for k in names]))
        recs.append(rec)
    st = pd.DataFrame(recs)
    st["stability_rank"] = st["slope_std_crossdomain"].rank()
    st = st.sort_values("slope_std_crossdomain").reset_index(drop=True)
    st.to_csv(OUT / "i2_stability.csv", index=False)

    # 热图：按平均 |斜率| 取 Top-30 展示（全 84 行过密），值=逐域斜率
    top = st.reindex(st["slope_mean"].abs().sort_values(ascending=False)
                     .index).head(30)
    M = top[[f"slope_{k}" for k in names]].values
    fig, ax = plt.subplots(figsize=(7.5, 9))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Mamun\n(BEEF-vdW)", "EqV2\n(RPBE)", "CatHub\n(PBE)"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Univariate slope of E_ads (eV per descriptor unit)")
    ax.set_title("Cross-functional descriptor–E_ads slope stability (Top-30 by |mean slope|)\n"
                 "Rows sorted stable→unstable available in i2_stability.csv")
    fig.tight_layout()
    fig.savefig(OUT / "i2_stability_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"[i2] 最稳定 Top5: {st['feature'].head(5).tolist()}")
    print(f"[i2] 最不稳 Bottom5: {st['feature'].tail(5).tolist()}")
    return st


# ------------------------------------------------------------ i3 三方案同台
def irm_lite_fit_predict(Xtr, ytr, envtr, Xte):
    """线性 IRMv1（dummy scale=1）：loss = Σ_e MSE_e + λ·||∇_w MSE_e||²。配置锁死。"""
    import torch
    torch.manual_seed(0)
    mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0)
    sd[sd < 1e-12] = 1.0
    Xtr_s = np.where(np.isnan(Xtr), 0.0, (Xtr - mu) / sd)
    Xte_s = np.where(np.isnan(Xte), 0.0, (Xte - mu) / sd)
    Xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte_s, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1] + 1, dtype=torch.float32, requires_grad=True)

    def mse_env(wv, mask):
        Xb = torch.cat([Xt[mask], torch.ones(mask.sum(), 1)], dim=1)
        return ((Xb @ wv - yt[mask]) ** 2).mean()

    opt = torch.optim.Adam([w], lr=IRM_LR)
    envs = sorted(set(envtr))
    masks = [torch.tensor(np.array(envtr) == e) for e in envs]
    for _ in range(IRM_EPOCHS):
        opt.zero_grad()
        loss = 0.0
        for mk in masks:
            mse = mse_env(w, mk)
            g = torch.autograd.grad(mse, w, create_graph=True)[0]
            loss = loss + mse + IRM_LAMBDA * (g ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xb = torch.cat([Xe, torch.ones(Xe.shape[0], 1)], dim=1)
        return (Xb @ w).numpy()


def step_i3(domains, feat_cols):
    names = list(domains)
    X = pd.concat([domains[k]["X"] for k in names], ignore_index=True)
    y = np.concatenate([domains[k]["y"] for k in names])
    dom = np.concatenate([[k] * len(domains[k]["y"]) for k in names])
    is_eq = dom == "EqV2_RPBE"
    n_eq = int(is_eq.sum())
    X_np = X[feat_cols].values.astype(float)

    dummies = pd.get_dummies(dom, drop_first=True).values.astype(float)
    X_ind = np.hstack([X_np, dummies])

    arms = {"naive_pooled": "xgb", "domain_indicator": "xgb", "IRM_lite": "irm"}
    rows, oof_store = [], {}
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(X_np))
        for arm, kind in arms.items():
            oof = np.full(len(y), np.nan)
            for tr, te in folds:
                if kind == "xgb":
                    Xa = X_ind if arm == "domain_indicator" else X_np
                    m = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
                    m.fit(Xa[tr], y[tr])
                    oof[te] = m.predict(Xa[te])
                else:
                    oof[te] = irm_lite_fit_predict(X_np[tr], y[tr], dom[tr], X_np[te])
            oof_store[(arm, seed)] = oof
            mae = mean_absolute_error(y[is_eq], oof[is_eq])
            rho = float(spearmanr(y[is_eq], oof[is_eq])[0])
            rows.append({"arm": arm, "seed": seed, "n_eqv2_oof": n_eq,
                         "EqV2_MAE": mae, "EqV2_spearman_rho": rho,
                         "gain_vs_0.347": EQV2_BASELINE_MAE - mae})
            print(f"[i3] {arm:18s} seed={seed:5d}  EqV2 MAE={mae:.4f}  ρ={rho:.3f}")
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "i3_fold_metrics.csv", index=False)

    summ = []
    for arm in arms:
        s = met[met["arm"] == arm]
        summ.append({"arm": arm,
                     "EqV2_MAE_mean": float(s["EqV2_MAE"].mean()),
                     "EqV2_MAE_sd": float(s["EqV2_MAE"].std()),
                     "gain_mean": float(s["gain_vs_0.347"].mean()),
                     "gain_min_10seeds": float(s["gain_vs_0.347"].min()),
                     "contract_10of10_gain_ge_0.005":
                         bool((s["gain_vs_0.347"] >= 0.005).all()),
                     "rho_mean": float(s["EqV2_spearman_rho"].mean())})
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(OUT / "i3_summary.csv", index=False)
    print(summ_df.to_string(index=False))

    # ---- 结局判读（预注册 dual_outcome）----
    best = summ_df.sort_values("EqV2_MAE_mean").iloc[0]
    crit = {"MAE_le_0.25": bool(best["EqV2_MAE_mean"] <= 0.25),
            "gain_gt_0.05": bool(best["gain_mean"] > 0.05),
            "rho_ge_0.35": bool(best["rho_mean"] >= 0.35)}
    verdict = "A" if all(crit.values()) else "B"
    verdict_doc = {
        "preregistered_rule": ("A: EqV2 MAE≤0.25 且改善>0.05 eV 且排序 ρ≥0.35；"
                               "否则 B（机制解释：跨库误差=描述符-泛函耦合伪相关）"),
        "baseline": {"v3_factory_zero_retrain_EqV2_MAE_eV": EQV2_BASELINE_MAE},
        "best_arm": str(best["arm"]),
        "criteria_check": crit,
        "verdict": verdict,
        "arms_summary": summ,
        "config_locked": {"seeds": SEEDS, "xgb_params": XGB_PARAMS,
                          "cv": "KFold(5, shuffle, seed) 同折同种子",
                          "irm_lite": {"lambda": IRM_LAMBDA, "epochs": IRM_EPOCHS,
                                       "lr": IRM_LR, "model": "linear, dummy-scale IRMv1"}},
    }
    (OUT / "i3_verdict.json").write_text(
        json.dumps(verdict_doc, indent=2, ensure_ascii=False))
    print(f"[i3] 结局判定 = {verdict}（best arm={best['arm']}）")
    return summ_df, verdict, verdict_doc


def main():
    domains, feat_cols = load_domains()
    ov, pair_df = step_i1(domains)
    st = step_i2(domains, feat_cols)
    summ_df, verdict, verdict_doc = step_i3(domains, feat_cols)
    st_valid = st.dropna(subset=["slope_std_crossdomain"])
    summary = {
        "workstream": "I_functional_invariance",
        "domains": {k: {"n": len(v["y"]),
                        "n_unique_comps": len(set(v["comp"]))}
                    for k, v in domains.items()},
        "i1_overlap": {"triple_overlap_comps": int(len(ov)),
                       "pair_counts": pair_df.to_dict("records")},
        "i2_stability": {
            "top5_stable": st_valid["feature"].head(5).tolist(),
            "bottom5_unstable": st_valid["feature"].tail(5).tolist(),
            "n_sign_flip_features": int(st_valid["sign_flips"].sum())},
        "i3": {"verdict": verdict,
               "best_arm": verdict_doc["best_arm"],
               "arms_summary": verdict_doc["arms_summary"],
               "criteria_check": verdict_doc["criteria_check"]},
    }
    (OUT / "invariance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("[写出] innovation/functional_invariance/ 全部产物完成")


if __name__ == "__main__":
    main()
