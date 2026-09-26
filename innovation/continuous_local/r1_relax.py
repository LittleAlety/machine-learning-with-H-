# -*- coding: utf-8 -*-
"""
innovation/continuous_local/r1_relax.py — Phase5 route2: CHGNet 结构弛豫 (幂等分片)

预注册 (config/preregistered_phase5.yaml route2_foundation_relax):
  原型 slab (1836) → CHGNet 弛豫 (FIRE, fmax=0.05 eV/Å, max 100 步)
  → 连续局域描述符 → 闸门 → 公平契约
  预算 4h CPU; 超时降级: 3d 磁性 + 逃逸角相关子集 (含 Fe/Co/Ni/Mn/Cr/La/Y/Sc)

幂等: relaxed_subset/<name>.json 存在且 converged 字段存在即跳过。
单进程分片 (--shard k --nshards K), 每分片独立加载模型, 防 4GB cgroup OOM。
用法: python r1_relax.py --subset mag3d_escape --shard 0 --nshards 3
"""
import argparse, json, re, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLAB_DIR = ROOT / "innovation" / "chgnet_features" / "slabs"
RELAX_ALL = OUT / "relaxed"
RELAX_SUB = OUT / "relaxed_subset"

SUBSET_ELS = {"Fe", "Co", "Ni", "Mn", "Cr", "La", "Y", "Sc"}


def comp_of(name: str) -> str:
    return name.rsplit("_", 2)[0]


def in_subset(name: str) -> bool:
    return bool(set(re.findall(r"[A-Z][a-z]?", comp_of(name))) & SUBSET_ELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="mag3d_escape", choices=["mag3d_escape", "all"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    torch.set_num_threads(int(os.environ.get("RELAX_THREADS", "1")))

    files = sorted(p.name for p in SLAB_DIR.glob("*.cif"))
    if args.subset == "mag3d_escape":
        files = [f for f in files if in_subset(f)]
        outdir = RELAX_SUB
    else:
        outdir = RELAX_ALL
    # 伪随机顺序 (md5): 限时预算下覆盖的组成是无偏抽样, 避免字母序偏置
    import hashlib
    files.sort(key=lambda f: hashlib.md5(f.encode()).hexdigest())
    files = files[args.shard :: args.nshards]
    if args.limit:
        files = files[: args.limit]
    outdir.mkdir(exist_ok=True)

    todo = [f for f in files if not (outdir / f.replace(".cif", ".json")).exists()]
    print(f"[shard {args.shard}] total={len(files)} todo={len(todo)}", flush=True)
    if not todo:
        return

    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.core import Structure

    model = CHGNet.load()
    opt = StructOptimizer(model, optimizer_class="FIRE")

    n_done = n_fail = 0
    t0 = time.time()
    for i, fn in enumerate(todo):
        stem = fn[:-4]
        try:
            s = Structure.from_file(SLAB_DIR / fn)
            e0 = float(model.predict_structure(s)["e"])
            res = opt.relax(s, fmax=args.fmax, steps=args.steps, verbose=False)
            traj = res["trajectory"]
            fs = res["final_structure"]
            rec = {
                "name": stem,
                "n_atoms": len(fs),
                "e_before": e0,
                "e_after": float(traj.energies[-1]),
                "n_steps": len(traj.energies),
                "fmax_final": float(np.max(np.linalg.norm(np.asarray(traj.forces[-1]), axis=1))),
                "converged": bool(np.max(np.linalg.norm(np.asarray(traj.forces[-1]), axis=1)) < args.fmax),
                "final_structure": fs.as_dict(),
            }
            (outdir / f"{stem}.json").write_text(json.dumps(rec))
            n_done += 1
        except Exception as e:
            n_fail += 1
            (outdir / f"{stem}.fail.json").write_text(
                json.dumps({"name": stem, "error": traceback.format_exc()[-2000:]})
            )
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(todo) - i - 1)
            print(f"[shard {args.shard}] {i+1}/{len(todo)} done={n_done} fail={n_fail} "
                  f"elapsed={dt/60:.1f}min eta={eta/60:.1f}min", flush=True)

    (outdir / f"_shard{args.shard}_done.json").write_text(json.dumps(
        {"shard": args.shard, "n_done": n_done, "n_fail": n_fail,
         "minutes": (time.time() - t0) / 60}))
    print(f"[shard {args.shard}] FINISHED done={n_done} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
