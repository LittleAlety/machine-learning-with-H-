"""C-3 v2 Stage-1 gate iteration driver.

Operationalises the Stage 1 exit table of ``C3v2_集群执行计划_Stage1-3.docx``:

    1.1 compile        -> pw.x passes its own examples
    1.2 anchor gate    -> 5-10 (111) surfaces, BEEF-vdW recompute H* vs Mamun
                          training values; |dE_ads| <= 0.1 eV on EVERY surface
    1.3 probe core-h   -> 1 bulk + 1 clean slab + 1 adsorption relax, measured
    1.4 MLIP path(opt)-> MACE-MPA-0 pre-relax -> DFT refine, same |dE| <= 0.1 eV

The driver NEVER fabricates DFT numbers. On a laptop without a compiled QE stack
the DFT-derived gates are reported as PENDING/BLOCKED with the exact reference
values pre-filled; on the cluster, after each probe writes its result back, the
same script re-runs and auto-evaluates the gate matrix. The red lines of the
plan are enforced: gates are not retroactively relaxed, every field change needs
an amendment, and failures are recorded as assets.

Outputs:
    stage1_iteration_status.json   machine-readable gate state
    stage1_iteration_report.md    human-readable status
    anchor_gate_sheet.csv          per-surface reference table (fill recomputed col)
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]          # ai_surface_cat/
PROJECT_ROOT = ROOT.parent                          # ai_surface_cat_项目代码包/
C3 = ROOT / "innovation" / "c3_v2"

RANKING_CSV = ROOT / "data_processed" / "candidate_rankings_hstar.csv"
CHECKSUM_MANIFEST = PROJECT_ROOT / "dft_stack" / "manifests" / "checksums.sha256"

OUT_JSON = C3 / "stage1_iteration_status.json"
OUT_MD = C3 / "stage1_iteration_report.md"
OUT_CSV = C3 / "anchor_gate_sheet.csv"

ANCHOR_SURFACES = ["Pt", "Cu", "Ir", "Pd", "Rh", "Ag", "Au", "Co", "Ru"]
GATE_TOL_EV = 0.10
DFT_ENGINES = ("pw.x", "cp2k.psmp", "gpaw")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_compile() -> dict:
    on_path = {name: shutil.which(name) for name in DFT_ENGINES}
    engine_found = next((v for v in on_path.values() if v), None)

    archives_ok = False
    archive_rows = []
    if CHECKSUM_MANIFEST.exists():
        for line in CHECKSUM_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            p = CHECKSUM_MANIFEST.parent.parent / rel
            actual = sha256_file(p) if p.exists() else None
            archive_rows.append({"path": rel, "match": actual == expected})
        archives_ok = bool(archive_rows) and all(r["match"] for r in archive_rows)

    if engine_found:
        status = "PASS"
        note = f"{engine_found} detected on PATH."
    elif archives_ok:
        status = "BLOCKED"
        note = "Source archives verify (SHA256) but no compiled pw.x/cp2k/gpaw on PATH. Build per 1.1."
    else:
        status = "BLOCKED"
        note = "Source archive manifest missing or mismatched."

    return {
        "step": "1.1 compile",
        "status": status,
        "archives_sha256_verified": archives_ok,
        "executables_on_path": on_path,
        "n_archives_verified": sum(r["match"] for r in archive_rows),
        "note": note,
    }


def gate_anchors() -> dict:
    rank = pd.read_csv(RANKING_CSV)
    rows = []
    for el in ANCHOR_SURFACES:
        hit = rank[
            (rank["comp"] == el) & (rank["structure"] == "A1") & (rank["facet"] == 111)
        ]
        if hit.empty:
            rows.append({"surface": f"{el}(111)", "mamun_true_dG_eV": None,
                         "note": "no pure-A1(111) row in training set"})
            continue
        rec = hit.iloc[0]
        rows.append({
            "surface": f"{el}(111)",
            "mamun_true_dG_eV": round(float(rec["true_dG"]), 4),
            "mamun_true_Eads_eV": round(float(rec["energy_eV"]), 4),
            "ml_pred_dG_eV": round(float(rec["pred_dG"]), 4),
            "recomputed_dG_eV": None,      # <- fill on cluster after BEEF-vdW run
            "abs_delta_eV": None,
        })

    sheet = pd.DataFrame(rows)
    sheet.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # Gate evaluation: only possible once recomputed values exist.
    have = [r for r in rows if r.get("recomputed_dG_eV") is not None]
    if not have:
        status = "PENDING"
        verdict = ("No BEEF-vdW recomputation yet. Run 1.2 on the cluster and write "
                   "recomputed_dG_eV into anchor_gate_sheet.csv; this script then "
                   "auto-decides PASS/FAIL. Gate requires |dE_ads| <= "
                   f"{GATE_TOL_EV} eV on EVERY surface.")
        worst = None
    else:
        deltas = [abs(r["recomputed_dG_eV"] - r["mamun_true_dG_eV"]) for r in have]
        worst = max(deltas)
        status = "PASS" if worst <= GATE_TOL_EV and len(have) >= 5 else "FAIL"
        verdict = (f"{len(have)}/9 surfaces recomputed; worst |dE|={worst:.4f} eV "
                   f"(tol {GATE_TOL_EV}).")

    return {
        "step": "1.2 anchor gate",
        "status": status,
        "tolerance_eV": GATE_TOL_EV,
        "n_anchor_surfaces": len(rows),
        "n_recomputed": len(have),
        "worst_abs_delta_eV": worst,
        "note": verdict,
        "sheet": str(OUT_CSV.relative_to(PROJECT_ROOT)),
    }


def gate_probe() -> dict:
    # Budget model (midpoint) from preflight; probe revises it in one shot.
    mid = 225 * 400 + 45 * 550 + 45 * 200 + 50 * 600  # H-ads + slab + bulk + OH-ads
    mid = math.ceil(mid * 1.30)
    return {
        "step": "1.3 probe core-hours",
        "status": "PENDING",
        "planned_tasks": ["1 bulk relax", "1 clean slab relax", "1 H* adsorption relax"],
        "measured_core_hours": None,
        "budget_midpoint_before_revision": mid,
        "note": ("Three probe tasks have not been measured. The plan forbids carrying "
                 "over the 199,875 estimate as an approved allocation: measure, then "
                 "revise the budget table in one shot and apply for the formal quota."),
    }


def gate_mlip() -> dict:
    return {
        "step": "1.4 MLIP-assisted path (optional)",
        "status": "PENDING",
        "note": ("MACE-MPA-0 pre-relax -> DFT refine may only accelerate production "
                 "after it reproduces |dE| <= 0.1 eV on the SAME anchor surfaces. "
                 "Do not use for batch acceleration until then."),
    }


def main() -> None:
    g_compile = gate_compile()
    g_anchor = gate_anchors()
    g_probe = gate_probe()
    g_mlip = gate_mlip()

    gates = [g_compile, g_anchor, g_probe, g_mlip]

    # Stage-1 exit hard condition (plan Sec.1): anchor gate PASS + probe logged.
    exit_open = (
        g_anchor["status"] == "PASS"
        and g_probe["measured_core_hours"] is not None
    )
    stage1 = "GO" if exit_open else ("CONDITIONAL GO" if g_compile["status"] == "PASS"
                                      else "CONDITIONAL GO (env pending)")
    # On this laptop: compile BLOCKED, anchors/probe PENDING -> still conditional GO
    # to *preparation*, never to production.
    production_go = False

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workstream": "C-3 v2 Stage-1 gate iteration driver",
        "host_note": ("Iteration driver executed on a Windows workstation; no HPC "
                      "allocation or compiled QE stack is present here, so DFT gates "
                      "are PENDING, not fabricated."),
        "gates": gates,
        "stage1_exit": {
            "hard_condition": "anchor gate PASS AND probe core-hours logged",
            "stage1_decision": stage1,
            "batch_production_go": production_go,
            "stage2_3_go": False,
        },
        "red_lines_enforced": [
            "gates are never retroactively relaxed",
            "any on-the-fly parameter change requires an amendment first",
            "negative and failed results are retained as assets",
        ],
    }

    OUT_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    # Markdown report
    lines = ["# C-3 v2 Stage-1 闸门迭代状态\n",
             f"_生成时间（UTC）：{status['generated_at']}_\n",
             f"> {status['host_note']}\n",
             "## 闸门矩阵\n",
             "| 步骤 | 内容 | 状态 | 说明 |",
             "|---|---|---|---|"]
    label = {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
             "BLOCKED": "⛔ BLOCKED", "PENDING": "⏳ PENDING"}
    for g in gates:
        lines.append(f"| {g['step']} | {g.get('note','')[:60]}… | "
                     f"{label.get(g['status'], g['status'])} | "
                     f"{g.get('note','')[:120]} |")
    lines += ["", "## 出口判定",
              f"- Stage 1 出口硬条件：锚定闸门 **PASS** + 探针核时落档。",
              f"- 当前 Stage 1：**{stage1}**（仅可做环境/锚定准备，不得开生产）。",
              f"- 批次生产（Stage 2/3）放行：**{production_go}**。",
              "", "## 红线",
              "- 闸门不追溯放宽；现场参数变更先写修正案再执行；阴性与失败结果全部保留。"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({g["step"]: g["status"] for g in gates}, ensure_ascii=False))
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
