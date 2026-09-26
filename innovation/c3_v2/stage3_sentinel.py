"""C-3 v2 Stage-3 sentinel stop-rule evaluator.

Batch C was contracted down from 25 to 8 sentinels that probe whether the
large CHGNet surface reconstructions are real or a learned-potential artifact.
The stop rule (plan Sec. 3) is deterministic:

  * >= 6/8 sentinels reproduce top-layer displacement < 0.5 A
        -> CHGNet reconstruction CONFIRMED ARTIFACT; cancel the other 17 systems.
  * >= 3/8 sentinels show top-layer displacement >= 0.5 A
        -> ESCALATE to a full-batch DFT audit (requires a new preregistration
           amendment first).
  * 4/8 or 5/8 (middle interval)
        -> UNDEFINED; the plan explicitly leaves this for a Stage-0 amendment.

The DFT-measured displacements do not exist yet (no cluster run), so the
evaluator reports PENDING and pre-fills the CHGNet predictions. On the
cluster, after each sentinel writes its measured displacement, re-running this
script auto-decides.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C3 = ROOT / "innovation" / "c3_v2"
PREREG = C3 / "batches_preregistered.json"

OUT_JSON = C3 / "stage3_sentinel_status.json"
OUT_CSV = C3 / "stage3_sentinel_sheet.csv"
OUT_MD = C3 / "stage3_sentinel_report.md"

# The 8 contracted sentinels (plan Sec. 3; Amendment 2).
SENTINELS = ["Cr3Pb", "LaPb", "CrTl", "Co", "CuFe3", "Fe3Mo", "FeTi", "Ag3Sc"]
THRESHOLD_A = 0.5


def main() -> None:
    data = json.loads(PREREG.read_text(encoding="utf-8"))
    by_comp = {r["comp"]: r for r in data["batch_C_relax_audit_30"]}

    rows = []
    for comp in SENTINELS:
        r = by_comp.get(comp)
        rows.append({
            "comp": comp,
            "structure": r["structure"] if r else "",
            "facet": r["facet"] if r else "",
            "chgnet_surf_disp_A": r["surf_disp_mean_A"] if r else None,
            "dft_measured_disp_A": None,   # <- fill on cluster
            "over_threshold": None,
        })

    measured = [r for r in rows if r["dft_measured_disp_A"] is not None]
    for r in measured:
        r["over_threshold"] = bool(r["dft_measured_disp_A"] >= THRESHOLD_A)

    n_over = sum(1 for r in measured if r["over_threshold"])
    n_under = len(measured) - n_over

    if not measured:
        decision = "PENDING"
        verdict = ("No DFT-measured displacements yet. Fill dft_measured_disp_A in "
                   "stage3_sentinel_sheet.csv after relaxing each sentinel; re-run to decide.")
    elif n_over >= 3:
        decision = "ESCALATE_FULL_AUDIT"
        verdict = f"{n_over}/8 sentinels show disp >= {THRESHOLD_A} A: escalate (needs amendment)."
    elif n_under >= 6:
        decision = "CONFIRMED_ARTIFACT"
        verdict = (f"{n_under}/8 sentinels reproduce disp < {THRESHOLD_A} A: CHGNet "
                   "reconstruction confirmed artifact; cancel the remaining 17 systems.")
    else:
        decision = "UNDEFINED_MIDDLE_INTERVAL"
        verdict = (f"{n_under}/8 under, {n_over}/8 over (4-5 interval): plan leaves this "
                   "undefined -- a Stage-0 amendment is required before proceeding.")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold_A": THRESHOLD_A,
        "n_sentinels": len(rows),
        "n_measured": len(measured),
        "n_over_threshold": n_over,
        "decision": decision,
        "verdict": verdict,
        "sentinels": rows,
    }
    OUT_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    md = ["# C-3 v2 Stage-3 哨兵停止规则\n",
          f"_生成时间（UTC）：{status['generated_at']}_\n",
          f"| 哨兵 | CHGNet 位移(Å) | DFT 实测(Å) | 是否 ≥{THRESHOLD_A}Å |",
          "|---|---:|---:|---|"]
    for r in rows:
        dft = "—" if r["dft_measured_disp_A"] is None else r["dft_measured_disp_A"]
        over = "—" if r["over_threshold"] is None else ("是" if r["over_threshold"] else "否")
        md.append(f"| {r['comp']} | {r['chgnet_surf_disp_A']} | {dft} | {over} |")
    md += ["", f"**当前裁决：{decision}**", "", verdict, ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"decision": decision, "n_measured": len(measured),
                      "n_sentinels": len(rows)}, ensure_ascii=False))
    print(f"Wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
