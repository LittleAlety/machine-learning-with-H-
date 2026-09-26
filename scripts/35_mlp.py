# -*- coding: utf-8 -*-
"""
35_mlp.py — A-4：MLP 受试者（配置锁死，输赢都记录）

架构三选一受试（锁死）：
  M1: 84→64→32→1 ReLU
  M2: 84→64→32→1 ReLU + BatchNorm
  M3: 84→32→16→1 tanh
训练：Adam(lr=1e-3)，MSE，500 epoch + early stopping(patience=30, 内部验证折)
标准化：StandardScaler 仅在各训练折内拟合（内层折），禁止泄漏
协议：同 GroupKFold(5) 同 10 种子 OOF；与 v3 基线对照记录，不做采纳承诺。
输出：outputs/r14_a4_mlp.json
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

OUT = ROOT / "outputs"
ARCHS = {"M1_64_32_relu": (64, 32, "relu", False),
         "M2_64_32_relu_bn": (64, 32, "relu", True),
         "M3_32_16_tanh": (32, 16, "tanh", False)}
EPOCHS, PATIENCE = 500, 30


def build(nin, h1, h2, act, bn):
    A = nn.ReLU if act == "relu" else nn.Tanh
    layers = [nn.Linear(nin, h1)]
    if bn:
        layers.append(nn.BatchNorm1d(h1))
    layers += [A(), nn.Linear(h1, h2)]
    if bn:
        layers.append(nn.BatchNorm1d(h2))
    layers += [A(), nn.Linear(h2, 1)]
    return nn.Sequential(*layers)


def train_one(Xtr, ytr, Xva, yva, arch, seed):
    torch.manual_seed(seed)
    net = build(Xtr.shape[1], *arch)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
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


def oof_mlp(X, y, groups, seed, arch):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        # 内层验证折（组感知 15%），标准化仅训练折内拟合
        gss = GroupShuffleSplit(1, test_size=0.15, random_state=seed)
        itr, ivl = next(gss.split(X[tr], y[tr], groups[tr]))
        sc = StandardScaler().fit(X[tr][itr])
        net = train_one(sc.transform(X[tr][itr]), y[tr][itr],
                        sc.transform(X[tr][ivl]), y[tr][ivl], arch, seed)
        with torch.no_grad():
            p[te] = net(torch.tensor(sc.transform(X[te]),
                                     dtype=torch.float32)).numpy().ravel()
    return mae(p, y)


def main():
    df, feats = load_v3()
    X = df[feats].values.astype(float)
    y = df["energy_eV"].values
    groups = df["comp"].values

    res = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "config": {"archs": ARCHS, "epochs": EPOCHS, "patience": PATIENCE,
                      "opt": "Adam(lr=1e-3)", "loss": "MSE",
                      "scaler": "StandardScaler 仅训练折内拟合"},
           "baseline_v3_oof_mae": 0.11374, "subjects": {}}
    for name, arch in ARCHS.items():
        maes = []
        for s in SEEDS:
            maes.append(oof_mlp(X, y, groups, s, arch))
            print(f"  [{name}] seed {s}: {maes[-1]:.4f}", flush=True)
        res["subjects"][name] = {"per_seed_mae": maes,
                                 "mean": float(np.mean(maes)),
                                 "std": float(np.std(maes)),
                                 "delta_vs_v3": float(np.mean(maes) - 0.11374),
                                 "beats_v3": bool(np.mean(maes) < 0.11374)}
        print(f"[{name}] {np.mean(maes):.4f}±{np.std(maes):.4f}", flush=True)
    best = min(res["subjects"], key=lambda k: res["subjects"][k]["mean"])
    res["verdict"] = {"best": best, "adopt": False,
                      "reason": (f"MLP 受试者仅记录输赢，不参与 v4 交付候选；"
                                 f"最佳 {best} {res['subjects'][best]['mean']:.4f} "
                                 f"vs v3 0.11374")}
    json.dump(res, open(OUT / "r14_a4_mlp.json", "w"), ensure_ascii=False, indent=2)
    print(f"[判定] {res['verdict']['reason']}")
    print("[写出] outputs/r14_a4_mlp.json")


if __name__ == "__main__":
    main()
