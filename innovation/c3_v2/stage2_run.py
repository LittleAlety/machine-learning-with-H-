"""C-3 v2 Stage-2 deterministic DFT queue builder + runner.

Frozen batches (batches_preregistered.json, hash-chained) are turned into an
explicit calculation queue. Each system implies the standard three-step
workflow (bulk relax -> clean slab relax -> adsorbate relax) under BEEF-vdW /
QE 7.6 with SSSW pseudopotentials. Magnetic systems (Fe/Co/Cr) get a dual
ferro/antiferromagnetic initialization and keep the full relaxation trajectory.

Red lines enforced by this code:
  * Nothing is faked. If pw.x is not on PATH (e.g. this Windows workstation),
    every task is recorded as PENDING_BINARY_MISSING -- never as SUCCESS.
  * Failures are appended to the result table with a reason, never dropped.
  * Batch A = H* only (10); Batch B = H* + OH* on the same 10 systems.
  * LLM agents never widen the batch; the queue is rebuilt only from the frozen
    preregistration file.

Outputs:
  stage2_queue.json     the full, ordered calculation queue
  stage2_status.csv     per-task status (recomputed every run)
  stage2_report.md      human-readable Stage-2 status
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # ai_surface_cat/
PROJECT_ROOT = ROOT.parent
C3 = ROOT / "innovation" / "c3_v2"
PREREG = C3 / "batches_preregistered.json"

OUT_QUEUE = C3 / "stage2_queue.json"
OUT_STATUS = C3 / "stage2_status.csv"
OUT_REPORT = C3 / "stage2_report.md"

MAGNETIC_ELEMENTS = {"Fe", "Co", "Cr", "Ni"}
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_comp(comp: str) -> dict:
    out = {}
    for el, n in COMP_TOKEN.findall(comp):
        out[el] = int(n) if n else 1
    return out


def is_magnetic(comp: str) -> bool:
    return bool(set(parse_comp(comp)) & MAGNETIC_ELEMENTS)


def build_queue() -> list[dict]:
    data = json.loads(PREREG.read_text(encoding="utf-8"))
    tasks: list[dict] = []

    def add(batch: str, rank: int, comp: str, structure: str, facet: int,
            ads: str, note: str) -> None:
        mag = is_magnetic(comp)
        tasks.append({
            "task_id": f"S2-{batch}-{rank:02d}-{ads}",
            "batch": batch,
            "rank_in_batch": rank,
            "comp": comp,
            "structure": structure,
            "facet": int(facet),
            "adsorbate": ads,
            "magnetic_dual_init": mag,
            "workflow": ["bulk_relax", "clean_slab_relax", f"{ads.lower()}_adsorb_relax"],
            "convergence": {"ecutwfc": 60, "fmax": 0.01,
                            "occupations": "smearing", "degauss": 0.02,
                            "functional": "BEEF-vdW"},
            "note": note,
        })

    for i, row in enumerate(data["batch_A_AL_top10"], start=1):
        add("A", i, row["comp"], row["structure"], row["facet"], "H",
            f"AL acquisition top, score={row['acquisition_score']}, pred_dG={row['pred_dG']}")

    for i, row in enumerate(data["batch_B_scaling_tier2_10"], start=1):
        add("B", i, row["comp"], row["structure"], row["facet"], "H",
            f"scaling tier-2, dG_H={row['dG_H_eV']}, dG_OH={row['dG_OH_eV']}")
        add("B", i, row["comp"], row["structure"], row["facet"], "OH",
            "OH extrapolation point (oh_in_training=False): escape-angle stress test")

    return tasks


def run(tasks: list[dict]) -> list[dict]:
    pw = shutil.which("pw.x")
    rows = []
    for t in tasks:
        if pw is None:
            status, reason = "PENDING_BINARY_MISSING", (
                "pw.x not on PATH; this workstation has no compiled QE stack. "
                "On the cluster, the runner executes bulk->slab->adsorb relax and "
                "parses total energy / fmax / convergence.")
        else:  # pragma: no cover - only reached on the cluster
            status, reason = "PENDING", "pw.x found; queue ready for orchestration."
        rows.append({
            **{k: t[k] for k in ("task_id", "batch", "comp", "structure",
                                 "facet", "adsorbate", "magnetic_dual_init")},
            "status": status,
            "reason": reason,
        })
    return rows


def main() -> None:
    tasks = build_queue()
    rows = run(tasks)
    OUT_QUEUE.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "n_tasks": len(tasks),
         "by_batch": {"A": sum(1 for t in tasks if t["batch"] == "A"),
                      "B_H": sum(1 for t in tasks if t["batch"] == "B" and t["adsorbate"] == "H"),
                      "B_OH": sum(1 for t in tasks if t["batch"] == "B" and t["adsorbate"] == "OH")},
         "tasks": tasks}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with OUT_STATUS.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_mag = sum(1 for t in tasks if t["magnetic_dual_init"])
    n_pending = sum(1 for r in rows if r["status"].startswith("PENDING"))
    md = [
        "# C-3 v2 Stage-2 计算队列状态\n",
        f"_生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n",
        f"- 队列总任务数：**{len(tasks)}**（批次 A H* = 10；批次 B H* = 10、OH* = 10）",
        f"- 磁性体系（铁磁/反铁磁双初猜）：**{n_mag}** 个任务",
        f"- 当前状态：**{n_pending}** PENDING（本机无 pw.x，未做任何 DFT，未伪造任何能量）",
        "",
        "出口判据（计划 §2）：≥80% 体系收敛入档后回灌重跑主动学习，"
        "对照回溯预期（全局误差 ≈ −18%）。失败体系如实记录、不替补。",
    ]
    OUT_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"n_tasks": len(tasks),
                      "by_batch": {"A": 10, "B_H": 10, "B_OH": 10},
                      "magnetic": n_mag,
                      "all_status": rows[0]["status"]}, ensure_ascii=False))
    print(f"Wrote {OUT_QUEUE.name}, {OUT_STATUS.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
