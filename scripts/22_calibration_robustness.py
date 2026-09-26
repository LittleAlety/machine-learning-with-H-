# -*- coding: utf-8 -*-
"""
22_calibration_robustness.py — 偏差校准的诚实检验 + 稳健性检验（Round 5）

输入：data_processed/hstar_dataset_v2.csv（1,836 行 = 组成×结构×晶面，
      目标 energy_eV = 该构型下 min E_ads；36 特征，site_spread 为泄漏特征禁入）
基线：XGBoost(lr=0.03, max_depth=7, n_estimators=400, subsample=0.8, rs=42)
      GroupKFold(5, groups=comp) OOF MAE ≈ 0.1198 eV（07/12 口径，本脚本复现核对）

公平契约（Round 4）：校准参数仅在外层训练折内拟合（用内层 GroupKFold 生成
训练折的 OOF 预测来拟合校准器，再应用到外层模型对验证折的预测——嵌套防泄漏）；
改善 > 0.005 eV 才采纳；阴性结果同样交付。

内容：
1. 偏差校准
   a) 全局线性重校准：y_true = a + b·y_pred（内层 OOF 上拟合）
   b) 分层偏移校准：对"含3d/镧系"层与其余层各学一个残差偏移量（内层 OOF 上拟合）
   报告两方案 OOF MAE vs 基线 + 分层（含3d/镧系层）MAE，判定采纳与否（阈值 0.005）
2. ΔG 修正值敏感性：ΔG = E_ads + c，c ∈ {0.20,0.22,0.24,0.26,0.28}，
   用 OOF 预测重排候选（按 |ΔG| 升序），报各 c 下 Top-50 与 c=0.24 基线的
   Spearman 秩相关（对两 Top-50 并集计算）与 Top-20 留存率
3. 去重口径稳健性：三种聚合口径的组成级候选排名对比
   (i) 现行：每 组成×结构×晶面 取 min E_ads（即原行级值），组成得分 = min|ΔG|
   (ii) 每组成全局 min E_ads → |ΔG|
   (iii) 组态级均值 E_ads → |ΔG|
   报告口径间 Top-20 重叠率与 Spearman（全体组成）

产物：
  outputs/calib_comparison.csv          校准对比（含分层明细、采纳判定）
  outputs/calib_oof_predictions.csv     逐样本 OOF 真值/基线/方案a/方案b 预测
  outputs/calib_dgcorr_sensitivity.csv  ΔG 修正值敏感性矩阵
  outputs/calib_dedup_robustness.csv    去重口径稳健性矩阵
  notes/23_calibration_robustness.md    报告

红线：种子 42；site_spread 禁入特征；不改 01-21 既有产物。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
OUT = ROOT / "outputs"
NOTES = ROOT / "notes"
SEED = 42

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
XGB_KW = dict(learning_rate=0.03, max_depth=7, n_estimators=400,
              subsample=0.8, random_state=SEED, n_jobs=-1)
BASELINE_MAE_REF = 0.1198          # 07/12 报告的基线 OOF MAE (eV)
ADOPT_THRESHOLD = 0.005            # 采纳阈值 (eV)
DG_BASE = 0.24                     # 现行 ΔG 修正值
DG_GRID = [0.20, 0.22, 0.24, 0.26, 0.28]

# 周期类别口径与 19_error_analysis.py 完全一致
D3 = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
LANTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
         "Ho", "Er", "Tm", "Yb", "Lu"}


def parse_elements(comp):
    """解析规范化化学式 → 元素集合。"""
    return set(re.findall(r"[A-Z][a-z]?", comp))


def fit_xgb(X, y):
    m = XGBRegressor(**XGB_KW)
    m.fit(X, y)
    return m


# =====================================================================
# 数据加载
# =====================================================================
df = pd.read_csv(DP / "hstar_dataset_v2.csv")
feats = [c for c in df.columns if c not in ID_COLS]
X = df[feats]
y = df["energy_eV"].values
groups = df["comp"].values
els = df["comp"].map(parse_elements)
mask_3d_ln = els.map(lambda e: bool(e & (D3 | LANTH))).values
print(f"[数据] {df.shape[0]} 行 × {len(feats)} 特征；组成数={df['comp'].nunique()}")
print(f"[分层] 含3d/镧系层 n={mask_3d_ln.sum()}；其余层 n={(~mask_3d_ln).sum()}")

# =====================================================================
# 1. 基线 OOF + 两种校准方案（校准器均在外层训练折内、经内层 OOF 拟合）
# =====================================================================
gkf = GroupKFold(n_splits=5)
oof_base = np.full(len(df), np.nan)
oof_lin = np.full(len(df), np.nan)
oof_strat = np.full(len(df), np.nan)
calib_log = []

for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
    # 外层模型（训练折全量拟合），预测验证折
    mdl = fit_xgb(X.iloc[tr], y[tr])
    pred_te = mdl.predict(X.iloc[te])
    oof_base[te] = pred_te

    # 内层 GroupKFold(4)：生成训练折的 OOF 预测（拟合校准器用，防泄漏）
    inner = GroupKFold(n_splits=4)
    pred_tr_oof = np.full(len(tr), np.nan)
    for itr, iva in inner.split(X.iloc[tr], y[tr], groups[tr]):
        m_in = fit_xgb(X.iloc[tr].iloc[itr], y[tr][itr])
        pred_tr_oof[iva] = m_in.predict(X.iloc[tr].iloc[iva])

    # 方案 a：全局线性重校准 y_true = a + b·y_pred（内层 OOF 上最小二乘）
    b, a = np.polyfit(pred_tr_oof, y[tr], 1)
    oof_lin[te] = a + b * pred_te

    # 方案 b：分层偏移校准（含3d/镧系层 vs 其余层，各学残差均值偏移）
    res_tr = y[tr] - pred_tr_oof
    m_tr = mask_3d_ln[tr]
    off_in = res_tr[m_tr].mean()
    off_out = res_tr[~m_tr].mean()
    shift_te = np.where(mask_3d_ln[te], off_in, off_out)
    oof_strat[te] = pred_te + shift_te

    calib_log.append({"fold": k, "n_test": len(te),
                      "lin_a": a, "lin_b": b,
                      "offset_3dLn": off_in, "offset_rest": off_out})
    print(f"[fold {k}] 线性 a={a:+.4f}, b={b:.4f} | "
          f"偏移 3d/Ln={off_in:+.4f}, rest={off_out:+.4f}")

calib_log = pd.DataFrame(calib_log)

# ---- 指标汇总 ----
def metrics(pred, name):
    mae_all = mean_absolute_error(y, pred)
    mae_in = mean_absolute_error(y[mask_3d_ln], pred[mask_3d_ln])
    mae_out = mean_absolute_error(y[~mask_3d_ln], pred[~mask_3d_ln])
    bias_in = np.mean(pred[mask_3d_ln] - y[mask_3d_ln])
    return {"scheme": name, "MAE_all": mae_all, "MAE_3dLn": mae_in,
            "MAE_rest": mae_out, "bias_3dLn": bias_in,
            "dMAE_vs_base": mae_all - metrics_base["MAE_all"]}


# 方差压缩诊断（3d/镧系层）：预测/真值标准差与回归斜率
def shrink_diag(pred):
    pi, ti = pred[mask_3d_ln], y[mask_3d_ln]
    slope = np.polyfit(pi, ti, 1)[0]   # 真值~预测 斜率，<1 提示过冲，>1 提示压缩
    return np.std(pi), np.std(ti), slope


metrics_base = {"MAE_all": mean_absolute_error(y, oof_base)}  # 先算基线供对比
rows = []
for pred, name in [(oof_base, "baseline"), (oof_lin, "a_linear"),
                   (oof_strat, "b_stratified")]:
    r = metrics(pred, name)
    sd_p, sd_t, slope = shrink_diag(pred)
    r.update({"std_pred_3dLn": sd_p, "std_true_3dLn": sd_t,
              "shrink_slope_3dLn": slope})
    rows.append(r)
cmp_df = pd.DataFrame(rows)
base_mae = cmp_df.loc[cmp_df.scheme == "baseline", "MAE_all"].iloc[0]
cmp_df["adopt"] = cmp_df.apply(
    lambda r: ("基线" if r.scheme == "baseline"
               else ("采纳" if base_mae - r.MAE_all > ADOPT_THRESHOLD
                     else "不采纳")), axis=1)
print("\n===== 校准对比 =====")
print(cmp_df.round(4).to_string(index=False))
print(f"[核对] 基线 OOF MAE={base_mae:.4f}（参考值 {BASELINE_MAE_REF}）")

cmp_df.to_csv(OUT / "calib_comparison.csv", index=False)
calib_log.to_csv(OUT / "calib_fold_params.csv", index=False)
pd.DataFrame({"comp": df["comp"], "structure": df["structure"],
              "facet": df["facet"], "y_true": y, "pred_base": oof_base,
              "pred_lin": oof_lin, "pred_strat": oof_strat,
              "has_3d_or_Ln": mask_3d_ln}).to_csv(
    OUT / "calib_oof_predictions.csv", index=False)

# =====================================================================
# 2. ΔG 修正值敏感性（用基线 OOF 预测重排候选，按 |ΔG| 升序）
# =====================================================================
def rank_candidates(pred, c):
    d = df[["comp", "structure", "facet"]].copy()
    d["key"] = (d["comp"].astype(str) + "|" + d["structure"].astype(str)
                + "|" + d["facet"].astype(str))
    d["abs_dG"] = np.abs(pred + c)
    return d.sort_values("abs_dG").reset_index(drop=True)


base_rank = rank_candidates(oof_base, DG_BASE)
base_top50 = set(base_rank["key"].head(50))
base_top20 = set(base_rank["key"].head(20))

sens_rows = []
for c in DG_GRID:
    rk = rank_candidates(oof_base, c)
    top50, top20 = set(rk["key"].head(50)), set(rk["key"].head(20))
    inter = list(base_top50 & top50)
    # Spearman 仅在两 Top-50 交集成员上计算（并集口径会因成员不相交产生伪负相关）
    if len(inter) >= 5:
        rb = base_rank.set_index("key")["abs_dG"].loc[inter].rank().values
        rc = rk.set_index("key")["abs_dG"].loc[inter].rank().values
        rho_inter = spearmanr(rb, rc).statistic
    else:
        rho_inter = np.nan
    # 全量 1836 行排名的 Spearman（参考）
    rho_full = spearmanr(base_rank["abs_dG"].rank(),
                         rk.set_index("key")["abs_dG"]
                           .loc[base_rank["key"]].rank()).statistic
    row = {"c": c, "spearman_top50_intersect": rho_inter,
           "n_top50_intersect": len(inter),
           "spearman_full1836": rho_full,
           "top50_overlap": len(inter),
           "top20_retention": len(base_top20 & top20) / 20.0,
           "best_candidate": rk["key"].iloc[0]}
    sens_rows.append(row)
    print(f"[ΔG c={c:.2f}] Spearman(Top50交集,n={len(inter)})="
          f"{rho_inter if np.isfinite(rho_inter) else float('nan'):.3f}  "
          f"Spearman(全量)={rho_full:.4f}  Top50重叠={row['top50_overlap']}/50  "
          f"Top20留存={row['top20_retention']:.2f}")
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUT / "calib_dgcorr_sensitivity.csv", index=False)

# =====================================================================
# 3. 去重口径稳健性
# 关键事实：v2 数据集每组成恰有 1 行（1836 行 = 1836 个唯一
# comp×structure×facet）；且模型特征均为组成级描述符，对同一
# comp×structure 内的不同吸附位点不可区分。因此预测侧的三种口径
# ((i) 行级min / (ii) 组成全局min / (iii) 组态均值) 在数学上完全等价，
# 排名 Spearman ≡ 1、Top-20 重叠 ≡ 20/20。以下为实证核对（Panel A），
# 并追加 Panel B：真值侧建库口径 min vs mean（用 hstar_group_spread.csv
# 的原始位点能量统计），量化"若建库时取均值而非 min"对排名的影响。
# =====================================================================
tmp = df[["comp"]].copy()
tmp["E"] = oof_base
# (i) 现行：行级(组成×结构×晶面) min E_ads（数据集行即该口径），组成得分 = min|ΔG|
s_i = tmp.assign(dg=np.abs(oof_base + DG_BASE)).groupby("comp")["dg"].min()
# (ii) 每组成全局 min E_ads → |ΔG|
s_ii = np.abs(tmp.groupby("comp")["E"].min() + DG_BASE)
# (iii) 组态级均值 E_ads → |ΔG»
s_iii = np.abs(tmp.groupby("comp")["E"].mean() + DG_BASE)

agg = pd.DataFrame({"现行_行级min": s_i, "组成全局min": s_ii,
                    "组态均值": s_iii})
ranks = agg.rank()  # 分数越小越优 → 升序秩
tops = {c: set(agg[c].nsmallest(20).index) for c in agg.columns}

rob_rows = []
cols = list(agg.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        ci, cj = cols[i], cols[j]
        rho = spearmanr(ranks[ci], ranks[cj]).statistic
        rob_rows.append({"panel": "A_预测侧", "口径A": ci, "口径B": cj,
                         "spearman": rho,
                         "top20_overlap": len(tops[ci] & tops[cj]) / 20.0,
                         "n_common": len(tops[ci] & tops[cj])})
        print(f"[口径A面] {ci} vs {cj}: Spearman={rho:.4f}  "
              f"Top20重叠={rob_rows[-1]['n_common']}/20")

# ---- Panel B：真值侧建库口径（min vs mean vs median 原始位点能量）----
sp = pd.read_csv(DP / "hstar_group_spread.csv")  # 每行 = 一个 comp×structure
sp["dg_min"] = np.abs(sp["E_min"] + DG_BASE)
sp["dg_mean"] = np.abs(sp["E_mean"] + DG_BASE)
sp["dg_median"] = np.abs(sp["E_median"] + DG_BASE)
tpairs = [("E_min(现行)", "dg_min"), ("E_mean", "dg_mean"),
          ("E_median", "dg_median")]
ttops = {name: set(sp.nsmallest(20, col)["comp"])
         for name, col in tpairs}
for (na, ca), (nb, cb) in [((tpairs[0]), (tpairs[1])),
                           ((tpairs[0]), (tpairs[2]))]:
    rho = spearmanr(sp[ca], sp[cb]).statistic
    ncom = len(ttops[na] & ttops[nb])
    rob_rows.append({"panel": "B_真值侧建库", "口径A": na, "口径B": nb,
                     "spearman": rho, "top20_overlap": ncom / 20.0,
                     "n_common": ncom})
    print(f"[口径B面] {na} vs {nb}: Spearman(全量)={rho:.4f}  "
          f"Top20重叠={ncom}/20")
rob_df = pd.DataFrame(rob_rows)
rob_df.to_csv(OUT / "calib_dedup_robustness.csv", index=False)

# 各口径 Top-20 明细（便于报告引用）
top_detail = pd.DataFrame({c: agg[c].nsmallest(20).index for c in cols})
top_detail.to_csv(OUT / "calib_dedup_top20.csv", index=False)

print("\n[完成] 产物：outputs/calib_comparison.csv, calib_fold_params.csv, "
      "calib_oof_predictions.csv, calib_dgcorr_sensitivity.csv, "
      "calib_dedup_robustness.csv, calib_dedup_top20.csv")
