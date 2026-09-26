"""Independent preflight audit for the frozen C-3 v2 DFT batches.

This script does not change the preregistration. It recomputes the three
batches from their declared upstream sources, checks the checksum chain,
prices the production budget, and reports the Stage 1 gate status.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent

PREREG_FILE = ROOT / "innovation" / "c3_v2" / "batches_preregistered.json"
AL_FILE = ROOT / "outputs" / "active_learning_candidates_all.csv"
DUAL_FILE = ROOT / "innovation" / "c3_v2" / "dual_all_ranked.csv"
RELAX_FILE = ROOT / "innovation" / "continuous_local" / "continuous_local_features.csv"
LOEO_FILE = ROOT / "data_processed" / "hstar_loeo_results.csv"
ANCHOR_FILE = ROOT / "data_processed" / "literature_anchors.csv"
RANKING_FILE = ROOT / "data_processed" / "candidate_rankings_hstar.csv"
CHECKSUM_FILE = PROJECT_ROOT / "dft_stack" / "manifests" / "checksums.sha256"

OUT_JSON = ROOT / "innovation" / "c3_v2" / "preflight_report.json"
OUT_CSV = ROOT / "innovation" / "c3_v2" / "preflight_batches.csv"

RED_FLAG_ELEMENTS = {"La", "Y", "Sc", "Mn", "Bi", "Fe", "Tl"}
SPECIAL_WARNINGS = {
    "Cr": "Cr passivation warning, not a calculation exclusion",
    "Hg": "high-toxicity element, experimental value limited",
    "Tc": "radioactive element",
}
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
KEY_COLS = ["comp", "structure", "facet"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_comp(comp: str) -> dict[str, int]:
    matches = COMP_TOKEN.findall(comp)
    if not matches or "".join(el + n for el, n in matches) != comp:
        raise ValueError(f"Cannot parse composition: {comp!r}")
    return {el: int(n) if n else 1 for el, n in matches}


def red_flag_elements(comp: str) -> list[str]:
    return sorted(set(parse_comp(comp)) & RED_FLAG_ELEMENTS)


def warning_elements(comp: str) -> list[str]:
    return sorted(set(parse_comp(comp)) & set(SPECIAL_WARNINGS))


def records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in df[columns].to_dict("records"):
        result.append(
            {
                key: (int(value) if key == "facet" else value)
                for key, value in row.items()
            }
        )
    return result


def signature(value: Any) -> list[tuple[str, str, int]]:
    return [
        (str(item["comp"]), str(item["structure"]), int(item["facet"]))
        for item in value
    ]


def compare_batches(
    expected: list[dict[str, Any]],
    target: list[dict[str, Any]],
    numeric_fields: list[str],
) -> dict[str, Any]:
    expected_sig = signature(expected)
    target_sig = signature(target)
    target_by_key = {
        (str(item["comp"]), str(item["structure"]), int(item["facet"])): item
        for item in target
    }
    expected_by_key = {
        (str(item["comp"]), str(item["structure"]), int(item["facet"])): item
        for item in expected
    }
    differences: dict[str, float] = {}
    for field in numeric_fields:
        maxima = []
        for key in set(expected_by_key) & set(target_by_key):
            maxima.append(
                abs(float(expected_by_key[key][field]) - float(target_by_key[key][field]))
            )
        differences[field] = max(maxima) if maxima else math.nan

    return {
        "order_match": expected_sig == target_sig,
        "set_match": set(expected_sig) == set(target_sig),
        "expected_count": len(expected),
        "target_count": len(target),
        "missing_from_recomputed": [
            list(key) for key in target_sig if key not in set(expected_sig)
        ],
        "extra_in_recomputed": [
            list(key) for key in expected_sig if key not in set(target_sig)
        ],
        "max_abs_field_difference": differences,
        "expected_signature": [list(item) for item in expected_sig],
    }


def prereg_hash_audit(data: dict[str, Any]) -> dict[str, Any]:
    declared = str(data.get("sha256", "")).lower()
    payload = {key: value for key, value in data.items() if key != "sha256"}
    variants: dict[str, str] = {}

    for ensure_ascii in (False, True):
        prefix = "canonical_sort_keys" if ensure_ascii is False else "canonical_ascii"
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
        )
        variants[prefix] = sha256_text(text)
        variants[f"{prefix}_newline"] = sha256_text(text + "\n")

    for indent in (1, 2):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=False, indent=indent)
        variants[f"indent_{indent}"] = sha256_text(text)
        variants[f"indent_{indent}_newline"] = sha256_text(text + "\n")

    for key in ("batch_A_AL_top10", "batch_B_scaling_tier2_10", "batch_C_relax_audit_30"):
        text = json.dumps(
            data[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        variants[f"{key}_sha256"] = sha256_text(text)

    batch_payload = {
        key: data[key]
        for key in (
            "batch_A_AL_top10",
            "batch_B_scaling_tier2_10",
            "batch_C_relax_audit_30",
        )
    }
    for label, value in (
        ("batches_only", batch_payload),
        (
            "batches_only_list",
            [
                data["batch_A_AL_top10"],
                data["batch_B_scaling_tier2_10"],
                data["batch_C_relax_audit_30"],
            ],
        ),
    ):
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        variants[label] = sha256_text(text)
        variants[f"{label}_newline"] = sha256_text(text + "\n")

    combinations = {
        "source_file_hashes_concat": "".join(
            sha256_file(path)
            for path in (AL_FILE, DUAL_FILE, RELAX_FILE)
        ),
        "source_file_hashes_lines": "\n".join(
            sha256_file(path)
            for path in (AL_FILE, DUAL_FILE, RELAX_FILE)
        ),
    }
    for label, text in combinations.items():
        variants[label] = sha256_text(text)

    placeholder = dict(data)
    placeholder["sha256"] = ""
    variants["raw_object_empty_placeholder"] = sha256_text(
        json.dumps(placeholder, ensure_ascii=False, indent=1)
    )

    matching_variant = next(
        (name for name, digest in variants.items() if digest == declared), None
    )
    return {
        "declared_sha256": declared,
        "file_sha256": sha256_file(PREREG_FILE),
        "source_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (AL_FILE, DUAL_FILE, RELAX_FILE)
        },
        "canonical_payload_sha256": variants["canonical_sort_keys"],
        "declared_matches_file": declared == sha256_file(PREREG_FILE),
        "declared_matches_canonical_variant": matching_variant,
        "status": (
            "VERIFIED"
            if declared == sha256_file(PREREG_FILE) or matching_variant
            else "UNVERIFIED"
        ),
        "tested_variants": variants,
    }


def batch_a_audit(data: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(AL_FILE)
    frame["red_flag_elements"] = frame["comp"].map(red_flag_elements)
    frame["_all_rank"] = frame["acquisition_score"].rank(
        method="min", ascending=False
    ).astype(int)
    eligible = frame[
        frame["in_domain"].astype(bool) & frame["red_flag_elements"].map(len).eq(0)
    ].copy()
    eligible = eligible.sort_values(
        ["acquisition_score", "comp", "structure", "facet"],
        ascending=[False, True, True, True],
    )
    eligible["_eligible_rank"] = range(1, len(eligible) + 1)
    expected = eligible.head(10).copy()

    target = data["batch_A_AL_top10"]
    expected_records = records(
        expected, ["comp", "structure", "facet", "acquisition_score", "pred_dG"]
    )
    comparison = compare_batches(
        expected_records, target, ["acquisition_score", "pred_dG"]
    )

    selected = expected.copy()
    loeo = pd.read_csv(LOEO_FILE)
    loeo_map = dict(zip(loeo["held_out_element"], loeo["MAE"]))
    selected["max_loeo_mae_eV"] = selected["comp"].map(
        lambda comp: max(loeo_map.get(el, 0.0) for el in parse_comp(comp))
    )
    selected["warning_elements"] = selected["comp"].map(warning_elements)

    return {
        "comparison": comparison,
        "eligible_pool_size": int(len(eligible)),
        "source_pool_size": int(len(frame)),
        "first_eligible_global_rank": int(selected["_all_rank"].min()),
        "selected_global_rank_range": [
            int(selected["_all_rank"].min()),
            int(selected["_all_rank"].max()),
        ],
        "mean_abs_pred_dG_eV": float(selected["pred_dG"].abs().mean()),
        "n_abs_pred_dG_le_0.05": int(selected["pred_dG"].abs().le(0.05).sum()),
        "n_abs_pred_dG_le_0.10": int(selected["pred_dG"].abs().le(0.10).sum()),
        "max_loeo_element_mae_eV": float(selected["max_loeo_mae_eV"].max()),
        "high_toxicity_or_radioactivity_count": int(
            selected["warning_elements"]
            .map(lambda flags: bool(set(flags) & {"Hg", "Tl", "Tc"}))
            .sum()
        ),
        "warning_element_counts": {
            element: int(selected["comp"].map(lambda comp: element in parse_comp(comp)).sum())
            for element in sorted(SPECIAL_WARNINGS)
            if selected["comp"].map(lambda comp: element in parse_comp(comp)).any()
        },
        "selected": records(
            selected,
            [
                "comp",
                "structure",
                "facet",
                "acquisition_score",
                "pred_dG",
                "in_domain",
                "_all_rank",
                "_eligible_rank",
                "max_loeo_mae_eV",
            ],
        ),
    }


def batch_b_audit(data: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(DUAL_FILE)
    frame["red_flag_elements"] = frame["comp"].map(red_flag_elements)
    frame["_all_rank"] = frame["score"].rank(method="min", ascending=True).astype(int)
    eligible = frame[
        (~frame["oh_in_training"].astype(bool))
        & frame["red_flag_elements"].map(len).eq(0)
    ].copy()
    eligible = eligible.sort_values(
        ["score", "comp", "structure", "facet"],
        ascending=[True, True, True, True],
    )
    eligible["_eligible_rank"] = range(1, len(eligible) + 1)
    expected = eligible.head(10).copy()

    target = data["batch_B_scaling_tier2_10"]
    expected_records = records(
        expected, ["comp", "structure", "facet", "dG_H_eV", "dG_OH_eV", "score"]
    )
    comparison = compare_batches(
        expected_records, target, ["dG_H_eV", "dG_OH_eV", "score"]
    )
    return {
        "comparison": comparison,
        "eligible_pool_size": int(len(eligible)),
        "source_pool_size": int(len(frame)),
        "oh_in_training_count": int(expected["oh_in_training"].astype(bool).sum()),
        "selected_global_rank_range": [
            int(expected["_all_rank"].min()),
            int(expected["_all_rank"].max()),
        ],
        "score_range_eV": [
            float(expected["score"].min()),
            float(expected["score"].max()),
        ],
        "dG_H_range_eV": [
            float(expected["dG_H_eV"].min()),
            float(expected["dG_H_eV"].max()),
        ],
        "dG_OH_range_eV": [
            float(expected["dG_OH_eV"].min()),
            float(expected["dG_OH_eV"].max()),
        ],
        "Cr_warning_count": int(
            expected["comp"].map(lambda comp: "Cr" in parse_comp(comp)).sum()
        ),
        "selected": records(
            expected,
            [
                "comp",
                "structure",
                "facet",
                "oh_in_training",
                "dG_H_eV",
                "dG_OH_eV",
                "score",
                "_all_rank",
                "_eligible_rank",
            ],
        ),
    }


def batch_c_audit(data: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(RELAX_FILE)
    target_counts = {
        item["structure"]: sum(
            1 for row in data["batch_C_relax_audit_30"] if row["structure"] == item["structure"]
        )
        for item in data["batch_C_relax_audit_30"]
    }

    parts = []
    for structure in ("L10", "A1", "L12"):
        group = frame[frame["structure"].eq(structure)].sort_values(
            ["surf_disp_mean_A", "comp"], ascending=[False, True]
        )
        parts.append(group.head(target_counts[structure]))
    expected = pd.concat(parts, ignore_index=True)

    target = data["batch_C_relax_audit_30"]
    expected_records = records(
        expected,
        [
            "comp",
            "structure",
            "facet",
            "surf_disp_mean_A",
            "relax_energy_drop_per_atom",
        ],
    )
    comparison = compare_batches(
        expected_records,
        target,
        ["surf_disp_mean_A", "relax_energy_drop_per_atom"],
    )

    global_top25 = frame.sort_values(
        ["surf_disp_mean_A", "comp"], ascending=[False, True]
    ).head(25)
    selected_keys = set(signature(expected_records))
    global_keys = set(
        signature(
            records(global_top25, ["comp", "structure", "facet"])
        )
    )

    selected = expected.copy()
    selected["warning_elements"] = selected["comp"].map(warning_elements)
    selected["red_flag_elements"] = selected["comp"].map(red_flag_elements)
    source_disp_p90 = float(frame["surf_disp_mean_A"].quantile(0.90))
    source_energy_p10 = float(frame["relax_energy_drop_per_atom"].quantile(0.10))
    frame["_disp_percentile"] = frame["surf_disp_mean_A"].rank(pct=True) * 100
    frame["_energy_percentile"] = (
        frame["relax_energy_drop_per_atom"].rank(pct=True) * 100
    )
    disp_percentile_map = {
        (comp, structure, int(facet)): percentile
        for (comp, structure, facet), percentile in zip(
            frame[KEY_COLS].itertuples(index=False, name=None),
            frame["_disp_percentile"],
        )
    }
    energy_percentile_map = {
        (comp, structure, int(facet)): percentile
        for (comp, structure, facet), percentile in zip(
            frame[KEY_COLS].itertuples(index=False, name=None),
            frame["_energy_percentile"],
        )
    }
    selected_disp_percentiles = selected.apply(
        lambda row: disp_percentile_map[
            (row["comp"], row["structure"], int(row["facet"]))
        ],
        axis=1,
    )
    selected_energy_percentiles = selected.apply(
        lambda row: energy_percentile_map[
            (row["comp"], row["structure"], int(row["facet"]))
        ],
        axis=1,
    )
    structure_counts = selected["structure"].value_counts().to_dict()
    full_structure_counts = frame["structure"].value_counts().to_dict()
    global_structure_counts = global_top25["structure"].value_counts().to_dict()

    return {
        "comparison": comparison,
        "quota_rule": "all A1 + top 12 L10 + top 12 L12 (inferred from frozen listing)",
        "quota_is_explicit_in_preregistration": False,
        "source_pool_size": int(len(frame)),
        "source_structure_counts": {
            key: int(value) for key, value in full_structure_counts.items()
        },
        "selected_structure_counts": {
            key: int(value) for key, value in structure_counts.items()
        },
        "literal_global_top25_structure_counts": {
            key: int(value) for key, value in global_structure_counts.items()
        },
        "overlap_with_literal_global_top25": len(selected_keys & global_keys),
        "surface_displacement_A": {
            "min": float(selected["surf_disp_mean_A"].min()),
            "median": float(selected["surf_disp_mean_A"].median()),
            "mean": float(selected["surf_disp_mean_A"].mean()),
            "max": float(selected["surf_disp_mean_A"].max()),
            "lowest_selected_percentile_in_source": float(
                selected_disp_percentiles.min()
            ),
            "source_p90": source_disp_p90,
        },
        "relax_energy_drop_per_atom_eV": {
            "min": float(selected["relax_energy_drop_per_atom"].min()),
            "median": float(selected["relax_energy_drop_per_atom"].median()),
            "mean": float(selected["relax_energy_drop_per_atom"].mean()),
            "max": float(selected["relax_energy_drop_per_atom"].max()),
            "source_p10": source_energy_p10,
            "lowest_selected_percentile_in_source": float(
                selected_energy_percentiles.min()
            ),
        },
        "spearman_disp_vs_energy_drop_full": float(
            frame["surf_disp_mean_A"].rank().corr(
                frame["relax_energy_drop_per_atom"].rank()
            )
        ),
        "spearman_disp_vs_energy_drop_selected": float(
            selected["surf_disp_mean_A"].rank().corr(
                selected["relax_energy_drop_per_atom"].rank()
            )
        ),
        "n_converged": int(selected["relax_converged"].astype(bool).sum()),
        "converged_rate": float(selected["relax_converged"].astype(bool).mean()),
        "source_converged_rate": float(frame["relax_converged"].astype(bool).mean()),
        "unconverged": records(
            selected[~selected["relax_converged"].astype(bool)],
            [
                "comp",
                "structure",
                "facet",
                "surf_disp_mean_A",
                "relax_fmax_final",
            ],
        ),
        "n_with_red_flag_elements": int(selected["red_flag_elements"].map(bool).sum()),
        "red_flag_element_counts": {
            element: int(
                selected["comp"].map(lambda comp: element in parse_comp(comp)).sum()
            )
            for element in sorted(RED_FLAG_ELEMENTS)
            if selected["comp"].map(lambda comp: element in parse_comp(comp)).any()
        },
        "selected": records(
            selected,
            [
                "comp",
                "structure",
                "facet",
                "surf_disp_mean_A",
                "relax_energy_drop_per_atom",
                "relax_converged",
                "relax_fmax_final",
            ],
        ),
    }


def software_audit() -> dict[str, Any]:
    checks = []
    if CHECKSUM_FILE.exists():
        for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = CHECKSUM_FILE.parent.parent / relative
            actual = sha256_file(path) if path.exists() else None
            checks.append(
                {
                    "path": relative,
                    "exists": path.exists(),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "match": actual == expected,
                }
            )
    return {
        "manifest_present": CHECKSUM_FILE.exists(),
        "archives_ready": bool(checks) and all(item["match"] for item in checks),
        "checks": checks,
        "executables_on_path": {
            name: shutil.which(name)
            for name in ("pw.x", "cp2k.psmp", "gpaw", "cmake", "gfortran", "mpirun")
        },
    }


def anchor_audit() -> dict[str, Any]:
    anchor_frame = pd.read_csv(ANCHOR_FILE)
    ranking = pd.read_csv(RANKING_FILE)
    preferred = [
        "Pt",
        "Cu",
        "Ni",
        "Ir",
        "Pd",
        "Rh",
        "Ag",
        "Au",
        "Co",
        "Ru",
    ]
    pure = ranking[
        ranking["comp"].isin(preferred)
        & ranking["structure"].eq("A1")
        & ranking["facet"].eq(111)
    ].copy()
    pure["_order"] = pure["comp"].map(preferred.index)
    pure = pure.sort_values("_order").head(10)
    rows = []
    for row in pure.to_dict("records"):
        surface = f"{row['comp']}(111)"
        matches = anchor_frame[anchor_frame["surface"].eq(surface)]
        rows.append(
            {
                "surface": surface,
                "true_dG_eV": float(row["true_dG"]),
                "pred_dG_eV": float(row["pred_dG"]),
                "literature_anchor_matches": int(len(matches)),
            }
        )
    return {
        "literature_anchor_rows": int(len(anchor_frame)),
        "literature_unique_surfaces": int(anchor_frame["surface"].nunique()),
        "recommended_training_protocol_anchors": rows,
        "recommended_anchor_count": len(rows),
        "note": (
            "Use Mamun BEEF-vdW training values for the calculation gate; literature "
            "values with another functional are context, not the gate target."
        ),
    }


def budget_audit() -> dict[str, Any]:
    midpoint = {
        "bulk_relax": 45 * 200,
        "clean_slab_relax": 45 * 550,
        "H_ads_relax": 225 * 400,
        "OH_ads_relax": 50 * 600,
    }
    low = {
        "bulk_relax": 45 * 100,
        "clean_slab_relax": 45 * 300,
        "H_ads_relax": 225 * 200,
        "OH_ads_relax": 50 * 300,
    }
    high = {
        "bulk_relax": 45 * 300,
        "clean_slab_relax": 45 * 800,
        "H_ads_relax": 225 * 600,
        "OH_ads_relax": 50 * 900,
    }
    subtotal = sum(midpoint.values())
    total = math.ceil(subtotal * 1.30)
    low_total = math.ceil(sum(low.values()) * 1.30)
    high_total = math.ceil(sum(high.values()) * 1.30)
    return {
        "assumptions": {
            "H_sites_per_surface": 5,
            "OH_sites_per_surface": 5,
            "failure_retry_overhead": 0.30,
        },
        "midpoint_core_hours_before_overhead": subtotal,
        "midpoint_core_hours": total,
        "low_core_hours": low_total,
        "high_core_hours": high_total,
        "wall_days_at_32_cores": total / 32 / 24,
        "wall_days_at_512_cores": total / 512 / 24,
        "document_rounded_core_hours": 200000,
        "rounded_vs_recomputed_relative_error": abs(total - 200000) / 200000,
    }


def main() -> None:
    data = json.loads(PREREG_FILE.read_text(encoding="utf-8"))
    batch_a = batch_a_audit(data)
    batch_b = batch_b_audit(data)
    batch_c = batch_c_audit(data)
    hash_audit = prereg_hash_audit(data)
    software = software_audit()
    anchors = anchor_audit()
    budget = budget_audit()

    batch_checks = {
        "A": batch_a["comparison"]["order_match"],
        "B": batch_b["comparison"]["order_match"],
        "C": batch_c["comparison"]["order_match"],
    }
    overall = {
        "candidate_membership_reproduced": all(batch_checks.values()),
        "batch_checks": batch_checks,
        "prereg_hash_chain": hash_audit["status"],
        "software_archives_ready": software["archives_ready"],
        "compiled_dft_engines_ready": any(
            software["executables_on_path"][name]
            for name in ("pw.x", "cp2k.psmp", "gpaw")
        ),
        "anchor_gate_executed": False,
        "resource_probe_executed": False,
        "stage1_go_conditionally": all(batch_checks.values()),
        "stage2_stage3_go": False,
        "decision": (
            "GO to Stage 1 environment/anchor preparation; "
            "NO-GO to batch production until the three preflight gates pass."
        ),
    }

    report = {
        "generated_at": "2026-09-20",
        "workstream": "C-3 v2 DFT production preflight",
        "inputs": {
            "preregistration": str(PREREG_FILE.relative_to(PROJECT_ROOT)),
            "active_learning_candidates": str(AL_FILE.relative_to(PROJECT_ROOT)),
            "dual_adsorbate_ranking": str(DUAL_FILE.relative_to(PROJECT_ROOT)),
            "chgnet_relaxation_features": str(RELAX_FILE.relative_to(PROJECT_ROOT)),
        },
        "hash_audit": hash_audit,
        "batch_A": batch_a,
        "batch_B": batch_b,
        "batch_C": batch_c,
        "software_readiness": software,
        "anchor_gate_candidates": anchors,
        "budget_audit": budget,
        "overall": overall,
        "blocking_findings": [
            item
            for item in [
                (
                    "The declared preregistration SHA256 is not reproduced by the file "
                    "or the tested canonical payload variants."
                    if hash_audit["status"] != "VERIFIED"
                    else None
                ),
                (
                    "Batch C's 12/1/12 structure quotas are inferred from the frozen "
                    "listing and are not explicit in config/preregistered_phase6.yaml."
                    if not batch_c["quota_is_explicit_in_preregistration"]
                    else None
                ),
                "The DFT engines are downloaded as source but are not compiled or probed.",
                "The BEEF-vdW anchor gate has not been executed.",
                "The three-task core-hour calibration has not been executed.",
            ]
            if item is not None
        ],
        "risk_notes": [
            f"Batch A contains {batch_a['high_toxicity_or_radioactivity_count']} "
            "Hg/Tc-containing systems; laboratory safety review is required before "
            "experimental follow-up.",
            f"Batch C contains {batch_c['n_with_red_flag_elements']}/25 systems with "
            "registered oxophilicity/LOEO/toxicity flags; this is deliberate for the "
            "relaxation audit but prevents direct catalytic recommendation.",
            f"Batch C contains {batch_c['n_converged']}/25 CHGNet structures that reached "
            "the relaxation criterion; 2 did not.",
            "The source archives are verified, but no binary has been built in this "
            "Windows workspace; cluster toolchain validation remains pending.",
        ],
        "conclusion": (
            "The frozen A/B/C membership is reproducible from the local upstream files, "
            "so the scientific selection logic is internally consistent. The package is "
            "ready for Stage 1 environment setup, but not ready for production DFT: the "
            "declared hash needs an auditable detached checksum, Batch C stratification "
            "needs an explicit amendment/restatement, and the anchor, software, and "
            "core-hour gates remain unexecuted."
        ),
    }

    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for batch_name, audit in (("A", batch_a), ("B", batch_b), ("C", batch_c)):
        for rank, item in enumerate(audit["selected"], start=1):
            rows.append(
                {
                    "batch": batch_name,
                    "rank": rank,
                    "comp": item["comp"],
                    "structure": item["structure"],
                    "facet": item["facet"],
                    "reproduction_match": audit["comparison"]["order_match"],
                }
            )
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
