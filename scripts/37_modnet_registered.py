# -*- coding: utf-8 -*-
"""
37_modnet_registered.py — Phase 3 Workstream G：MODNet 注册受试者（封口 NN 函数类证伪）

预注册判读规则：config/preregistered_phase3.yaml :: workstream_G_modnet
  config_locked: feature_select "greedy_physics <=20 of 84", net "64-32",
                 early_stop "inner_val_fold"
  fair_contract: min_gain 0.005 eV, direction 10/10, nested_cv required
  adopt:  fair_contract 双过 + LOEO 复测 + 跨域复测
  reject: 入表14；3.7 扩展节升级为"含领域最强小数据 NN 模板在内函数类无红利"

锁定决策（本脚本头即注册记录，禁止扩配置）：
  [锁1] 特征筛选打分 = |Pearson(feature, residual)|（二选一锁死，弃用互信息）。
        贪心前向：第 1 步取与 y 的 |Pearson| 最大者；之后每步取与当前
        最小二乘残差 |Pearson| 最大者；上限 20 维。
        全部统计仅用外层训练折，禁全数据泄漏。
  [锁2] 网络 = torch 64-32-1 ReLU MLP（同 35_mlp.py M1 族），Adam(lr=1e-3)，
        MSE，max 500 epoch，early stopping(patience=30) 仅看内层验证折。
  [锁3] 标准化 StandardScaler 仅在外层训练折的内层训练子折内拟合。
  [锁4] 协议与 v3 完全同折同种子：GroupKFold(5, groups=comp)，
        seeds = [0,1,2,7,13,42,99,123,2024,31337]。
  [锁5] 嵌套 CV 复核：内层折仅用于 early stopping，不调任何超参；
        外层 OOF 即为嵌套评估（nested_cv: by-design，无内层信息外泄）。

环境：CPU 即可（torch CPU）。
输出：outputs/modnet_registered/modnet_registered.json（幂等可复跑，确定性结果）
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, SEEDS, load_v3, mae

OUT = ROOT / "outputs" / "modnet_registered"
BASELINE_V3 = 0.11374
MAX_FEATS = 20
EPOCHS, PATIENCE = 500, 30


def greedy_select(Xtr, ytr, feat_names, k=MAX_FEATS):
    """MODNet 精神贪心前向选择：|Pearson| 对残差，仅用训练折。"""
    selected = []
    resid = ytr - ytr.mean()
    for _ in range(min(k, Xtr.shape[1])):
        best_j, best_s = None, -1.0
        for j in range(Xtr.shape[1]):
            if j in selected:
                continue
            x = Xtr[:, j]
            sx = x.std()
            if sx < 1e-12:
                continue
            s = abs(np.corrcoef(x, resid)[0, 1])
            if np.isnan(s):
                continue
            if s > best_s:
                best_s, best_j = s, j
        if best_j is None:
            break
        selected.append(best_j)
        # OLS 残差更新（对所选特征做最小二乘投影）
        A = np.column_stack([np.ones(len(ytr)), Xtr[:, selected]])
        beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        resid = ytr - A @ beta
    return [feat_names[j] for j in selected]


def build(nin):
    return nn.Sequential(nn.Linear(nin, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 1))


def train_one(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    net = build(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad()
        loss = lossf(net(xt), yt)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv), yv).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def oof_modnet(X, y, groups, seed, feat_names):
    """外层 GroupKFold(5) OOF；特征筛选仅用外层训练折；内层 15% 组感知验证折仅用于 early stopping。"""
    p = np.full(len(y), np.nan)
    sel_per_fold = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        sel = greedy_select(X[tr], y[tr], feat_names)
        sel_per_fold.append(sel)
        idx = [feat_names.index(f) for f in sel]
        gss = GroupShuffleSplit(1, test_size=0.15, random_state=seed)
        itr, ivl = next(gss.split(X[tr], y[tr], groups[tr]))
        sc = StandardScaler().fit(X[tr][itr][:, idx])
        net = train_one(sc.transform(X[tr][itr][:, idx]), y[tr][itr],
                        sc.transform(X[tr][ivl][:, idx]), y[tr][ivl], seed)
        with torch.no_grad():
            p[te] = net(torch.tensor(sc.transform(X[te][:, idx]),
                                     dtype=torch.float32)).numpy().ravel()
    return mae(p, y), sel_per_fold


def main():
    df, feats = load_v3()
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values
    OUT.mkdir(parents=True, exist_ok=True)

    maes, sels = [], None
    for s in SEEDS:
        m, sel = oof_modnet(X, y, groups, s, feats)
        maes.append(m)
        if sels is None:
            sels = sel
        print(f"  seed {s}: OOF MAE {m:.4f}", flush=True)
    mean, std = float(np.mean(maes)), float(np.std(maes))
    gains = [BASELINE_V3 - m for m in maes]
    dir_ok = int(sum(g > 0 for g in gains))
    gain_ok = bool(mean < BASELINE_V3 - 0.005)
    nested_ok = True  # 内层仅 early stopping，不调参；外层 OOF 即嵌套评估
    adopt = bool(gain_ok and dir_ok == 10 and nested_ok)

    res = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workstream": "phase3_G_modnet_registered",
        "preregistration": "config/preregistered_phase3.yaml::workstream_G_modnet",
        "locked_config": {
            "feature_select": "greedy forward, |Pearson| on OLS residual, <=20 of 84, outer-train-fold only",
            "net": "torch 64-32-1 ReLU MLP, Adam(lr=1e-3), MSE, max 500 epoch",
            "early_stop": "patience=30 on inner group-aware 15% val fold only",
            "scaler": "StandardScaler fit inside inner-train split only",
            "nested_cv": "inner fold used ONLY for early stopping, no hyperparameter tuning",
            "data": "data_processed/hstar_features_v3_84feat.csv (84 维组成描述符, 主线 energy_eV)",
            "folds": "GroupKFold(5, groups=comp) 与 v3 完全同折（确定性）",
            "seeds": SEEDS,
        },
        "baseline_v3_oof_mae": BASELINE_V3,
        "per_seed_mae": maes,
        "mean": mean,
        "std": std,
        "per_seed_gain_vs_v3": gains,
        "selected_features_fold0_seed0": sels[0],
        "fair_contract_check": {
            "gain_eV": float(BASELINE_V3 - mean),
            "gain_gt_0.005": gain_ok,
            "direction_10of10": f"{dir_ok}/10",
            "direction_ok": dir_ok == 10,
            "nested_cv_by_design": nested_ok,
        },
        "verdict": {
            "adopt": adopt,
            "action": ("触发完整采纳流程（加做 LOEO + CatHub 跨域复测）" if adopt
                       else "如实阴性：入表14；3.7 扩展节升级为'含领域最强小数据 NN 模板在内函数类无红利'"),
        },
    }
    json.dump(res, open(OUT / "modnet_registered.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"[MODNet] mean {mean:.4f}±{std:.4f} vs v3 {BASELINE_V3} "
          f"(gain {BASELINE_V3 - mean:+.4f} eV, 方向 {dir_ok}/10)")
    print(f"[判定] {'触发采纳流程' if adopt else '阴性，不触发采纳流程'}")
    print("[写出] outputs/modnet_registered/modnet_registered.json")


if __name__ == "__main__":
    main()
