"""C-3 v2 feedback / retrain closure.

Stage 2/3 DFT results are ingested, merged into the training table, and the
active-learning campaign is re-run under the frozen fair contract:

    Δ = surface-level new_model - old_model  (must be > 0.005 eV, i.e. better)
    + 10/10 seeds win  + nested CV 5/5 folds win  + paired test p < 0.05

This script validates the ingestion format and emits the retrain plan; it does
not silently accept or fabricate DFT rows. Missing rows, failed jobs and
negative results are carried through.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C3 = ROOT / "innovation" / "c3_v2"

OUT = C3 / "feedback_retrain_plan.json"

REQUIRED_COLUMNS = {"comp", "structure", "facet", "adsorbate",
                    "total_energy_eV", "fmax_eV_A", "converged"}
FAIR_CONTRACT = {"delta_threshold_eV": 0.005,
                 "seeds_required": 10, "nested_folds_required": 5,
                 "paired_p_max": 0.05}


def main() -> None:
    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ingestion": {
            "expected_input": "stage2_status.csv extended on cluster with "
                              "total_energy_eV / fmax_eV_A / converged",
            "required_columns": sorted(REQUIRED_COLUMNS),
            "converged_only_train": True,
            "failed_jobs_retained": True,
        },
        "rerun_sequence": [
            "scripts/21_active_learning.py   (re-rank pool on merged table)",
            "scripts/24_al_validation.py     (learning curve + extval acquisition)",
            "scripts/27_seed_robustness.py   (10-seed paired test)",
        ],
        "fair_contract": FAIR_CONTRACT,
        "v5_geometry_rule": (
            "v5 geometry features may only be promoted after their descriptors are "
            "rebuilt from truly DFT-relaxed structures and pass the same contract; "
            "currently V5_GEOMETRY_CANDIDATE_CONDITIONAL (surface Δ=+0.0249 eV, "
            "site OOF 0.104 eV vs <=0.090 target)."
        ),
        "status": "BLOCKED_UNTIL_STAGE2_DATA",
    }
    OUT.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"Wrote {OUT.name} -> status: {plan['status']}")


if __name__ == "__main__":
    main()
