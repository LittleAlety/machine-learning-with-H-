# -*- coding: utf-8 -*-
"""
30_lockbox_split.py — B-1：划定 15% lockbox（最终盲测集）

- 按结构类型（A1/L10/L12）分层，按规范化组成（comp）分组抽 15%（≈275 个 comp）
- 禁用目标值（energy_eV）参与任何 stratify 决策
- 输出 splits/lockbox_v1.json（时间戳 + SHA256 + 锁定纪律）
- 泄漏自检：lockbox vs 训练集在标准化描述符空间的最近邻距离分布
  → splits/lockbox_leakcheck.png + 数字写入 JSON
- config/fair_contract.yaml：写入 lockbox 不可见纪律

幂等：若 splits/lockbox_v1.json 已存在且 --force 未给，直接复用并仅补图。
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r14_common import ROOT, DP, load_v3

SPLIT_SEED = 42          # 划分种子（主划分）
FRAC = 0.15


def make_split(df, split_seed, frac=FRAC):
    """结构分层 × comp 分组抽取；返回 lockbox comp 列表（排序）。"""
    rng = np.random.RandomState(split_seed)
    lock = []
    for st, sub in df.groupby("structure"):
        comps = np.sort(sub["comp"].unique())
        n = max(1, int(round(len(comps) * frac)))
        lock.extend(rng.choice(comps, size=n, replace=False).tolist())
    return sorted(lock)


def leakcheck(df, feats, lock_comps, out_png):
    """标准化描述符空间最近邻距离分布（仅用训练集统计量标准化）。"""
    is_lock = df["comp"].isin(lock_comps).values
    Xtr = df.loc[~is_lock, feats].values.astype(float)
    Xlb = df.loc[is_lock, feats].values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Ztr, Zlb = (Xtr - mu) / sd, (Xlb - mu) / sd
    # 最近邻距离（分块计算）
    d_lb = np.empty(len(Zlb))
    for i in range(0, len(Zlb), 256):
        d2 = ((Zlb[i:i+256, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        d_lb[i:i+256] = np.sqrt(d2.min(1))
    d_tr = np.empty(len(Ztr))
    for i in range(0, len(Ztr), 256):
        d2 = ((Ztr[i:i+256, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2 if d2.shape[0] == d2.shape[1] else d2, np.inf)
        d_tr[i:i+256] = np.sqrt(np.where(np.eye(len(Ztr))[i:i+256], np.inf, d2).min(1))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(d_tr, bins=60, density=True, alpha=0.6, color="#DFB27E",
            label=f"train→train NN (n={len(d_tr)})")
    ax.hist(d_lb, bins=60, density=True, alpha=0.6, color="#B35C24",
            label=f"lockbox→train NN (n={len(d_lb)})")
    ax.set_xlabel("Nearest-neighbor distance (standardized descriptor space)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    # KS 统计量：若 lockbox 距离分布显著小于 train 自分布则提示泄漏风险
    from scipy.stats import ks_2samp
    ks = ks_2samp(d_lb, d_tr)
    return {"lockbox_nn_mean": float(d_lb.mean()),
            "train_nn_mean": float(d_tr.mean()),
            "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def main():
    df, feats = load_v3()
    sdir = ROOT / "splits"
    sdir.mkdir(exist_ok=True)
    cdir = ROOT / "config"
    cdir.mkdir(exist_ok=True)
    out = sdir / "lockbox_v1.json"

    if out.exists() and "--force" not in sys.argv:
        rec = json.load(open(out))
        lock_comps = rec["lockbox_comps"]
        print(f"[复用] {out}（{len(lock_comps)} comps）")
    else:
        lock_comps = make_split(df, SPLIT_SEED)
        payload = {
            "version": "lockbox_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "stratify": "structure (A1/L10/L12)，禁用目标值 stratify",
                "group": "规范化 comp（GroupKFold 同口径）",
                "fraction": FRAC, "split_seed": SPLIT_SEED,
                "rule": "lockbox 自划定起对开发不可见；最终模型只评一次（A-5）",
            },
            "n_lockbox_comps": len(lock_comps),
            "n_total_comps": int(df["comp"].nunique()),
            "structure_counts_lockbox":
                df[df["comp"].isin(lock_comps)]["structure"].value_counts().to_dict(),
            "lockbox_comps": lock_comps,
        }
        sha = hashlib.sha256(
            json.dumps(lock_comps, sort_keys=True).encode()).hexdigest()
        payload["sha256_of_comp_list"] = sha
        json.dump(payload, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"[写出] {out}  n={len(lock_comps)}  sha256={sha[:16]}…")

    stats = leakcheck(df, feats, lock_comps, sdir / "lockbox_leakcheck.png")
    print(f"[泄漏自检] {stats}")
    rec = json.load(open(out))
    rec["leakcheck"] = stats
    rec["leakcheck_fig"] = "splits/lockbox_leakcheck.png"
    json.dump(rec, open(out, "w"), ensure_ascii=False, indent=2)

    # fair_contract.yaml
    fc = cdir / "fair_contract.yaml"
    text = """# fair_contract.yaml — plan-hb v2.0 r14 公平契约（30_lockbox_split.py 生成）
lockbox:
  file: splits/lockbox_v1.json
  rule: "lockbox 自划定起对开发不可见，最终模型只评一次（A-5）；其余任何步骤不得触碰"
  fraction: 0.15
  stratify: "structure only（禁用目标值）"
  group: 规范化 comp
adoption:  # 精度优化采纳门槛
  multi_seed_oof_mae_gain_eV: 0.005      # 相对基线 10 种子 OOF MAE 均值改善需 > 0.005
  direction_consistency: "10/10 种子方向一致"
  nested_cv: "外层 GroupKFold(5) 复核通过方判定采纳"
seeds: [0, 1, 2, 7, 13, 42, 99, 123, 2024, 31337]
fold_protocol: "GroupKFold(n_splits=5, groups=规范化comp)，禁止重切"
baseline_v3_oof_mae_10seed: 0.11374
created_utc: "%s"
""" % datetime.now(timezone.utc).isoformat()
    fc.write_text(text, encoding="utf-8")
    print(f"[写出] {fc}")


if __name__ == "__main__":
    main()
