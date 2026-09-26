# -*- coding: utf-8 -*-
"""P1 容量扫描（capacity sweep）— 预注册判读规则

网格（预注册，仅此一表，禁止事后追加）：
- GBDT：max_depth ∈ {3,5,7,10,16} × n_estimators ∈ {100,400,1000,2000}（lr=0.03,
  subsample=0.8 固定），3 种子 (0,1,2)，每点记录 OOF MAE 与 train MAE
- GBDT 附属行：depth=7, n_est=400, min_child_weight ∈ {1,5,20} × 3 种子
- MLP：width ∈ {16,64,256} × depth ∈ {1,2,3,4}，3 种子；StandardScaler 仅在
  训练折拟合；early stopping（训练折内再切 15% 验证，patience=30，上限 500 epoch）

判读规则（预注册）：
- 容量饱和成立 ⇔ 最高容量区（GBDT: depth≥7 且 n_est≥400 的所有点）的 OOF MAE
  两两差异全部 ≤ ±0.005 eV
- 若 MLP 段随容量 OOF 仍显著下降（width=256 最优 OOF 比 width=64 最优 OOF 低 >0.005 eV）
  → 记录 "触发 A-4 深挖"（只记录，不行动）
诊断专用，禁止用于选择交付配置。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PROBES, C_MAIN, C_ALT, C_3RD, SEEDS3, load_data,
                     gkf_splits, oof_pred, mae, plt)

DEPTH_GRID = [3, 5, 7, 10, 16]
NEST_GRID = [100, 400, 1000, 2000]
MCW_GRID = [1, 5, 20]
MLP_WIDTHS = [16, 64, 256]
MLP_DEPTHS = [1, 2, 3, 4]


def _checkpoint(df, path):
    df.to_csv(path, index=False)
    df.to_csv("/tmp/" + path.name, index=False)


def run_gbdt(X, y, groups, csv_path):
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    have = set(zip(done.get("model", []), done.get("max_depth", []),
                   done.get("n_estimators", []), done.get("min_child_weight", []),
                   done.get("seed", [])))
    for depth in DEPTH_GRID:
        for nest in NEST_GRID:
            for seed in SEEDS3:
                if ("GBDT", depth, nest, 1, seed) in have:
                    continue
                pred, tr_pred = oof_pred(X, y, groups, seed,
                                         dict(max_depth=depth, n_estimators=nest))
                rows.append(dict(model="GBDT", max_depth=depth, n_estimators=nest,
                                 min_child_weight=1, seed=seed,
                                 oof_mae=mae(pred, y), train_mae=mae(tr_pred, y)))
            _checkpoint(pd.DataFrame(rows), csv_path)
            print(f"[GBDT] depth={depth} n_est={nest} done", flush=True)
    for mcw in MCW_GRID:
        for seed in SEEDS3:
            if ("GBDT_mcw", 7, 400, mcw, seed) in have:
                continue
            pred, tr_pred = oof_pred(X, y, groups, seed, dict(min_child_weight=mcw))
            rows.append(dict(model="GBDT_mcw", max_depth=7, n_estimators=400,
                             min_child_weight=mcw, seed=seed,
                             oof_mae=mae(pred, y), train_mae=mae(tr_pred, y)))
        _checkpoint(pd.DataFrame(rows), csv_path)
    return pd.DataFrame(rows)


def mlp_fit_predict(Xtr, ytr, Xte, seed, width, depth):
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)          # 仅训练折拟合
    Xs = sc.transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    idx = np.arange(len(Xs))
    tr_i, va_i = train_test_split(idx, test_size=0.15, random_state=seed)
    xt = torch.tensor(Xs[tr_i]); yt = torch.tensor(ytr[tr_i], dtype=torch.float32)
    xv = torch.tensor(Xs[va_i]); yv = torch.tensor(ytr[va_i], dtype=torch.float32)
    xte = torch.tensor(Xte_s)
    layers, d_in = [], Xs.shape[1]
    for _ in range(depth):
        layers += [torch.nn.Linear(d_in, width), torch.nn.ReLU()]
        d_in = width
    layers.append(torch.nn.Linear(d_in, 1))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.L1Loss()
    best, bad, best_state = np.inf, 0, None
    for epoch in range(500):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(net(xt[b]).squeeze(-1), yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv).squeeze(-1), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_te = net(xte).squeeze(-1).numpy()
        pred_tr = net(torch.tensor(Xs)).squeeze(-1).numpy()
    return pred_tr, pred_te


def run_mlp(X, y, groups, csv_path):
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    have = set(zip(done.get("width", []), done.get("depth", []),
                   done.get("seed", [])))
    splits = gkf_splits(X, y, groups)
    for width in MLP_WIDTHS:
        for depth in MLP_DEPTHS:
            for seed in SEEDS3:
                if (width, depth, seed) in have:
                    continue
                pred = np.full(len(y), np.nan)
                tr_pred = np.full(len(y), np.nan)
                for tr, te in splits:
                    ptr, pte = mlp_fit_predict(X[tr], y[tr], X[te], seed, width, depth)
                    pred[te] = pte
                    tr_pred[tr] = ptr
                rows.append(dict(model="MLP", width=width, depth=depth, seed=seed,
                                 oof_mae=mae(pred, y), train_mae=mae(tr_pred, y)))
                _checkpoint(pd.DataFrame(rows), csv_path)
            print(f"[MLP] width={width} depth={depth} done", flush=True)
    return pd.DataFrame(rows)


def make_figure(gbdt, mlp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    g = (gbdt[gbdt.model == "GBDT"]
         .groupby(["max_depth", "n_estimators"])[["oof_mae", "train_mae"]]
         .mean().reset_index())
    colors = {3: C_ALT, 5: C_3RD, 7: C_MAIN, 10: "#8C3B12", 16: "#4A2A12"}
    for depth in DEPTH_GRID:
        sub = g[g.max_depth == depth].sort_values("n_estimators")
        ax.plot(sub.n_estimators, sub.oof_mae, "o-", color=colors[depth],
                label=f"OOF depth={depth}")
        ax.plot(sub.n_estimators, sub.train_mae, "s--", color=colors[depth],
                alpha=0.55, label=f"train depth={depth}")
    ax.set_xscale("log")
    ax.set_xlabel("n_estimators (log scale)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("P1a GBDT capacity sweep")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax = axes[1]
    m = mlp.groupby(["width", "depth"])[["oof_mae", "train_mae"]].mean().reset_index()
    mcolors = {16: C_ALT, 64: C_3RD, 256: C_MAIN}
    for width in MLP_WIDTHS:
        sub = m[m.width == width].sort_values("depth")
        ax.plot(sub.depth, sub.oof_mae, "o-", color=mcolors[width],
                label=f"OOF width={width}")
        ax.plot(sub.depth, sub.train_mae, "s--", color=mcolors[width],
                alpha=0.55, label=f"train width={width}")
    ax.set_xticks(MLP_DEPTHS)
    ax.set_xlabel("hidden layers (depth)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("P1b MLP capacity sweep")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(PROBES / "p1_capacity_curves.png")
    plt.close(fig)


def main():
    PROBES.mkdir(exist_ok=True)
    df, X, y, groups, feats = load_data()
    gbdt_csv = PROBES / "p1_gbdt_results.csv"
    mlp_csv = PROBES / "p1_mlp_results.csv"
    gbdt = run_gbdt(X, y, groups, gbdt_csv)
    mlp = run_mlp(X, y, groups, mlp_csv)
    make_figure(gbdt, mlp)

    # ---- 预注册判读 ----
    g = (gbdt[gbdt.model == "GBDT"]
         .groupby(["max_depth", "n_estimators"]).oof_mae.mean().reset_index())
    hc = g[(g.max_depth >= 7) & (g.n_estimators >= 400)].oof_mae
    sat_span = float(hc.max() - hc.min())
    saturation = bool(sat_span <= 0.005)
    m = mlp.groupby(["width", "depth"]).oof_mae.mean().reset_index()
    best64 = m[m.width == 64].oof_mae.min()
    best256 = m[m.width == 256].oof_mae.min()
    a4_trigger = bool((best64 - best256) > 0.005)
    verdict = {
        "gbdt_highcap_oof_span": sat_span,
        "gbdt_highcap_min": float(hc.min()), "gbdt_highcap_max": float(hc.max()),
        "gbdt_best": g.loc[g.oof_mae.idxmin()].to_dict(),
        "saturation_verdict": "capacity saturated" if saturation else "NOT saturated",
        "mlp_best_oof_by_width": {str(w): float(m[m.width == w].oof_mae.min())
                                  for w in MLP_WIDTHS},
        "mlp_a4_deepdive_triggered": a4_trigger,
    }
    (PROBES / "p1_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
