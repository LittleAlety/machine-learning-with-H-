# Explainable Machine Learning for Surface Hydrogen Adsorption Energies: A Preregistered, Honest-Evaluation Pipeline and an Active-Learning DFT Loop

**Author:** [LittleAlety]
**Code repository:** https://github.com/LittleAlety/machine-learning-with-H-
**Project status:** ML pipeline finalized (v4); C-3 v2 DFT active-learning loop preregistered and in the Stage-1 gate-holding state.

---

## Abstract

We present an open, reproducible machine-learning (ML) pipeline that predicts *surface* hydrogen adsorption energies and screens hydrogen-evolution-reaction (HER) catalyst surfaces directly from composition and structure descriptors, without first-principles relaxation at prediction time. Trained on the Mamun *et al.* high-throughput dual-metal dataset (1,836 unique surfaces across 37 metals, derived from 7,048 diluted H\* entries out of 45,131 adsorption reactions), a tuned XGBoost model reaches a random 5-fold mean absolute error (MAE) of **0.120 ± 0.011 eV** and R² = 0.817, matching the GroupKFold MAE (0.120 ± 0.009 eV) and leaving-one-element-out (LOEO) pooled MAE of 0.155 eV. The same architecture on an independent CO\*/O\* validation line (937 rows, CatApp) reaches 0.227 ± 0.017 eV (R² = 0.890), and a 15-adsorbate unified model reaches 0.316 eV random / 0.363 eV grouped MAE. Zero-retrain prediction on an external, independently curated Catalysis-Hub homogeneous-HER set (355 surfaces) gives MAE 0.177 eV (R² = 0.742), collapsing to a near-zero systematic bias (+0.001 eV) on the BEEF-vdW-matched subset. Every claim is audited against leakage: target-derived features (`site_spread`) were removed, the grouping key was normalized to canonical composition, and nested cross-validation is used for model selection. To turn the surrogate into evidence rather than a black box, we preregister a three-batch active-learning DFT loop (C-3 v2) with a hash-chain, anchor gate (|ΔE_ads| ≤ 0.1 eV), core-hour budget probe, and an eight-sentinel stop rule designed to falsify a CHGNet relaxation artifact (25/25 CHGNet reconstructions were pre-adjudicated as artifacts: median top-layer displacement 3.503 Å versus a DFT-anchor expectation of ≈0.117 Å). A geometry-aware v5 feature set improves surface-level agreement by +0.0249 eV (10/10 seeds, nested CV 5/5) but has not yet been validated on truly relaxed DFT structures. The pipeline, preregistration chain, gate driver, and interactive predictor are released as accompanying material.

**Keywords:** surface adsorption energy, hydrogen evolution reaction, XGBoost, SHAP, transfer learning, active learning, density functional theory, preregistration

---

## 1. Introduction

Adsorption energy descriptors — most notably the hydrogen adsorption free energy ΔG_H* — are the central quantity of computational electrocatalysis. Under the Sabatier principle, an ideal HER catalyst binds H neither too strongly nor too weakly (ΔG_H* ≈ 0), and high-throughput density-functional-theory (DFT) screening has been the workhorse for finding such surfaces. DFT, however, remains expensive and its accuracy is bounded by the choice of exchange–correlation functional, pseudopotentials, slab convergence, and relaxation protocol.

Machine-learning surrogate models promise to predict adsorption energies at orders of magnitude lower cost, but published surrogate studies frequently overstate accuracy through (i) data leakage from target-derived features, (ii) random splits that leak chemically equivalent compositions across train and test, and (iii) failure to test extrapolation to elements or adsorbates absent from training. We address these failure modes explicitly and build a pipeline whose *evaluation discipline* is as much the contribution as its accuracy.

Our contributions are:

1. **An adsorption-species-agnostic, leakage-audited pipeline** (composition/structure descriptors → gradient-boosted trees → SHAP interpretation → Sabatier screening), evaluated with random 5-fold, GroupKFold on canonical composition, and LOEO.
2. **Two independent external validations** — a cross-database zero-retrain test on EqV2-HER and a homogeneous-metal test against the official Catalysis-Hub GraphQL library — plus a 99.2%/98.4% record-level cross-check of the training data itself.
3. **A preregistered active-learning DFT loop (C-3 v2)** with a frozen batch list, hash chain, amendments, anchor gate, core-hour probe, and sentinel stop rule that can falsify a learned-interatomic-potential (CHGNet) relaxation artifact.
4. **An interactive, browser-based predictor** and a fully open code release.
5. **A systematic, fair-contract descriptor-saturation ablation** across three independent families — tabulated elemental stats (Mendeleev), DFT bulk elastic constants (Materials Project), and DFT bulk reference energies (OMat24) — establishing that static composition is saturated and that further gains require relaxed surface geometry.

## 2. Data

### 2.1 Main line — H\* (HER)

The main training set is the Mamun *et al.* high-temperature dual-metal dataset (Mamun *et al.*, *Sci. Data* **6**:76, 2019; CC-BY-4.0), distributed via CatBench. Of 45,131 adsorption reactions we retain the diluted H\* channel (0.5 H₂(g) + \* → H\*) yielding 7,048 records that collapse to **1,836 unique surfaces** spanning **37 metals** and three structure types: A1 (pure-metal (111)), L1₂ (111), and L1₀ (101). The target `ref_ads_eng = E(H\*slab) − E(slab) − 0.5·E(H₂)`; negative values denote exothermic adsorption. Because the mirror distribution omits `atoms_json`, structure/facet are inferred from the 12-atom supercell stoichiometric pattern.

Data integrity was independently verified against the official Catalysis-Hub GraphQL API: composition match **99.2%** (1,821/1,836) and per-record energy match **98.4%** (6,893/7,005), with a maximum deviation of 5.2 × 10⁻⁴ eV.

### 2.2 Validation line — CO\*/O\*

The method line uses the CMR CatApp database (Hummelshøj *et al.*, *Angew. Chem. Int. Ed.* 2012, RPBE functional). After retaining only CO\*/O\*, dropping BEP empirical estimates, parsing surface-termination labels (AA/AB/BB), median-aggregating duplicates, and removing 1.5×IQR outliers, **937 rows** remain. A subtle sign convention was detected and corrected: in this database larger numbers denote *stronger* binding.

### 2.3 Extension — multi-adsorbate

From the same Mamun archive we built a unified 15-adsorbate dataset (C/N/O/S and CH_x/NH/OH/SH fragments), taking the lowest adsorption energy per (composition, structure, adsorbate) configuration.

### 2.4 External test sets

- **EqV2-HER-Discovery** (ergroup, *ACS Catal.* 2025): ~861 DFT H-adsorption records used for zero-retrain prediction.
- **Catalysis-Hub homogeneous HER** (`scripts/14`): a curated 355-surface homogeneous-metal subset used as a second, official-library external test, stratified by publication and by functional.

## 3. Methods

### 3.1 Descriptors

Element descriptors come from Mendeleev: Pauling electronegativity, atomic radius, group, period, first ionization energy, and d-electron count (with special handling for Sc/Y/La nd¹). Composition-weighted statistics (weighted mean/max/min/range/std) plus element count are combined with one-hot structure/facet/adsorbate indicators. For the multi-adsorbate model we add adsorbating-atom identity and its elemental properties.

### 3.2 Models

We compare Linear/Ridge/Lasso (StandardScaler pipeline), Random Forest, and XGBoost (small grid search with early stopping; n_estimators up to 2000). The tuned XGBoost is the production model.

### 3.3 Honest evaluation (the central methodological choice)

- **Three CV gauges are always reported together:** random 5-fold, GroupKFold on *canonical* composition (elements sorted + gcd-reduced, e.g. Pt9Ti3 → Pt3Ti, eliminating the AgAu3/Au3Ag leakage class), and LOEO.
- **No label-derived features.** The within-group energy spread `site_spread` contains the target and was removed; doing so moved the H\* MAE from a leaky 0.104 eV to the honest 0.120 eV.
- **Nested CV** governs hyperparameter selection (inner search, outer 5-fold unbiased estimate); 10-seed robustness and paired tests are reported separately.
- **Red flags** (radioactive Tc, strongly oxophilic La/Y/Sc, high-LOEO-error Mn/Bi/Fe, toxic Hg/Tl) are carried into the candidate table rather than hidden.

### 3.4 Interpretation and screening

SHAP TreeExplainer attributes predictions; HER candidates are ranked by |ΔG_H\*| with ΔG_H\* ≈ E_ads + 0.24 eV (zero-point + entropy correction, Nørskov approximation). Candidates within one model MAE of the optimum are flagged "statistically indistinguishable."

## 4. Results

### 4.1 H\* main line (1,836 surfaces)

| Model | Random 5-fold MAE (eV) | R² |
|---|---:|---:|
| Linear | 0.181 ± 0.010 | 0.715 |
| Ridge | 0.180 ± 0.010 | 0.715 |
| Lasso | 0.429 ± 0.018 | ~0.00 |
| Random Forest | 0.122 ± 0.008 | 0.817 |
| XGBoost (default) | 0.128 ± 0.009 | 0.814 |
| **XGBoost (tuned)** | **0.120 ± 0.011** | **0.817** |

GroupKFold MAE is 0.120 ± 0.009 eV — essentially identical to random CV, indicating the grouping correction removed the leak rather than merely inflating the gap. LOEO pooled MAE is 0.155 eV; the worst held-out elements are Mn (0.482 eV), Fe (0.283 eV), and Bi (0.279 eV), which we explicitly carry as red flags rather than smoothing over them. The Sabatier screening top candidates (e.g. CuPt₃, CrPt₃, AgNi, Zn₃Zr) place Pt-family surfaces near ΔG_H\* ≈ 0 as expected, serving as a literature sanity anchor.

### 4.2 CO\*/O\* validation line (937 rows)

XGBoost-tuned reaches 0.227 ± 0.017 eV (random) and 0.219 ± 0.017 eV (GroupKFold, normalized composition), R² = 0.890 — confirming the pipeline is adsorption-species-agnostic and not overfit to H\*.

### 4.3 Unified multi-adsorbate model

Across 15 adsorbates, XGBoost-tuned reaches 0.316 eV (random, R² = 0.964) and 0.363 eV (grouped). SHAP and the H\*-anchored scaling matrix reproduce the well-known scaling relations between C/N/O/S adsorbates, providing a built-in physical plausibility check.

### 4.4 External validation

Zero-retrain prediction on the 355-surface Catalysis-Hub homogeneous-HER set yields:

| Stratum | n | MAE (eV) | R² | Bias (pred−true) |
|---|---:|---:|---:|---:|
| All | 355 | 0.177 | 0.742 | +0.088 eV |
| In-domain (element+facet 111) | 75 | 0.141 | 0.629 | +0.022 eV |
| Elements ⊂ Mamun-37 | 326 | 0.181 | 0.742 | +0.088 eV |
| **BEEF-vdW-matched** | 114 | **0.142** | 0.741 | **+0.001 eV** |
| PBE-only | 203 | 0.184 | 0.702 | +0.146 eV |

The model carries a small functional-offset bias (Mamun is BEEF-vdW; external PBE/RPBE references differ), and on the functional-matched stratum that bias collapses to 0.001 eV — i.e. the surrogate generalizes, and the residual error is largely the known functional reference offset, not a structural failure.

### 4.5 Composition-only descriptors are saturated

A natural question is whether adding more descriptors — the literature proposes many elemental and bulk properties — can close the remaining error without new calculations. We therefore screened three independent families of additional descriptors under an identical fair contract (GroupKFold on canonical composition, fixed tuned hyperparameters, three seeds; the within-screen baseline MAE is 0.1166 eV):

| Family (source) | Best single descriptor Δ (eV) | All-joint Δ (eV) |
|---|---:|---:|
| Elemental stats — electron affinity, atomic weight, density, atomic volume, polarizability, thermal conductivity, specific heat, Ghosh EN (Mendeleev) | −0.0004 (density / atomic volume) | +0.0014 |
| DFT bulk elastic constants — bulk modulus K_VRH, shear modulus G_VRH, anisotropy (Materials Project) | +0.0002 | +0.0002 |
| DFT elemental bulk reference energy per atom (OMat24 references, PBE) | +0.0002 | — |

Every single delta lies within ±0.0016 eV — an order of magnitude below the fold-to-fold variability (≈0.01 eV) — and pooling descriptors never helps (the all-joint models are neutral-to-worse). We therefore do **not** promote any of these to the production model. The result is informative rather than null: the composition-weighted elemental representation already saturates the information available from *unrelaxed* composition, and bulk-derived properties (moduli, bulk energies) are merely proxies for the same compositional signal. The missing information is specifically the **surface geometry and local electronic structure that only a relaxation returns** — consistent with the v5 geometry model, which gains a substantial +0.025 eV (OOF) once relaxed geometric features are available (Section 5.4). This directly motivates the DFT campaign below and explains why further descriptor engineering on static composition cannot substitute for it.

## 5. The preregistered active-learning DFT loop (C-3 v2)

A surrogate model is only as trustworthy as the experiments used to correct it. Because the training structures were inferred from stoichiometry rather than relaxed, and because a learned potential (CHGNet) flagged large surface reconstructions, we designed a **preregistered, gated DFT campaign** (plan `C3v2_集群执行计划_Stage1-3`) with three frozen batches and a hash chain, so that no exploratory choice can be made after seeing results.

### 5.1 Frozen batches

- **Batch A (10 systems, H\*):** top of the active-learning acquisition function, in-domain, red-flag filtered — chosen to compress global error (expected ≈ −18% MAE).
- **Batch B (10 systems × H\*/OH\*, ≈15 system-equivalents):** the nearest HER×ORR-scaling neighbors with *no* OH in training — an independent stress test of the "empty HER×ORR intersection" conclusion and of the escape-angle prediction.
- **Batch C:** originally 25 CHGNet-flagged systems, reduced to **8 sentinels** (Cr₃Pb, LaPb, CrTl, Co, CuFe₃, Fe₃Mo, FeTi, Ag₃Sc) spanning three structure types.

An independent preflight audit recomputes all three batches from upstream files and confirms membership reproduction (A/B/C order match = true), verifies the detached SHA-256 checksum chain, and prices the budget.

### 5.2 Anchor gate and stop rules

Stage 1 may open only when (i) the DFT stack compiles, (ii) a 5–10 surface BEEF-vdW anchor gate reproduces Mamun H\* values to |ΔE_ads| ≤ 0.1 eV on *every* surface, and (iii) three probe tasks (bulk, clean slab, adsorbate relaxation) have measured core hours. The Stage-1 gate driver shipped in this repo (`innovation/c3_v2/stage1_driver.py`) encodes these as a state machine and pre-fills the Mamun reference values; on the cluster, filling the recomputed column auto-decides PASS/FAIL.

For Batch C, a **stop rule** governs the 8 sentinels: if ≥6/8 reproduce top-layer displacement < 0.5 Å, CHGNet's large reconstruction is confirmed as a surrogate artifact and the remaining 17 systems are cancelled; if ≥3/8 show ≥0.5 Å, a full-batch DFT audit (requiring a new preregistration amendment) is triggered.

### 5.3 Why this loop is needed: the CHGNet artifact

The 25 Batch-C systems were pre-adjudicated at the atomic level: **25/25 labeled ARTIFACT, 0/25 real reconstruction, 0/25 mixed**. The DFT-anchored median top-layer displacement is **0.117 Å** (range 0.022–0.403 Å), whereas CHGNet reports **3.503 Å** — roughly a 40× overstatement, and ≈22× the adsorption-induced displacement (0.21 Å) even though the sign agrees. In other words, the headline "reconstruction" in the surrogate-relaxed structures is a learned-potential artifact, not physics. The 8 sentinels exist to confirm this with independent DFT before the surrogate geometries are either trusted or discarded.

### 5.4 v5 geometry-aware model — conditional, not declared

Adding 84 + 36 real-geometry descriptors improves the surface-level fair contract by **Δ = +0.0249 eV (10/10 seeds win)** and nested CV by Δ = +0.0237 eV (5/5 folds win), with a +0.0428 eV gain on the 3d magnetic sublayer. However, the site-level out-of-fold MAE (0.104 eV) misses the ≤0.090 eV target and, crucially, the features have not yet been recomputed on truly DFT-relaxed structures. We therefore declare v5 `V5_GEOMETRY_CANDIDATE_CONDITIONAL` rather than shipping it as production — a deliberate negative result.

### 5.5 Budget and honesty

The production budget midpoint is 199,875 core-hours (low 101,400 / high 298,350); after Batch C contraction it is expected to fall to ≈150k core-hours and will be revised in one shot after the probes. On the present workstation no compiler/MPI/`pw.x` is available, so the DFT gates are correctly reported as PENDING/BLOCKED rather than fabricated — the plan's red line ("gates are never retroactively relaxed; failures are assets") is enforced in code.

## 6. Limitations

- Structures are inferred from stoichiometry, not from relaxed geometries, for the main H\* line; this is exactly what the C-3 v2 loop will correct.
- The mirror dataset lacks `atoms_json`, limiting true geometric features until DFT re-relaxation returns.
- LOEO error concentrates on Mn/Fe/Bi — extrapolation to 3d magnetic and heavy elements is the weakest regime.
- The external-test bias tracks the functional offset (BEEF-vdW vs PBE/RPBE); absolute energies across functional families should be bias-corrected.
- The middle decision interval (4/8 or 5/8 sentinels) of the stop rule is not yet defined and is flagged as a required Stage-0 amendment.

## 7. Conclusion

A deliberately conservative, leakage-audited tree-model pipeline predicts H\* adsorption energies to ~0.12 eV on 1,836 surfaces and generalizes to a 355-surface external set at ~0.18 eV, with the residual dominated by a known functional reference offset. Rather than stopping at a leaderboard number, we pair the surrogate with a preregistered, hash-chained, gate-protected active-learning DFT campaign designed to (a) compress error by batch, (b) stress-test the HER×ORR extrapolation, and (c) falsify a 40× CHGNet relaxation artifact via eight sentinels. The gate driver, preflight audit, interactive predictor, and all metrics are released open to support reproducible, falsifiable computational catalysis.

---

## Data and code availability

- Code, descriptors, metrics, SHAP values, the preregistration chain, and the Stage-1 gate driver: https://github.com/LittleAlety/machine-learning-with-H-
- Main training data: Mamun *et al.*, *Sci. Data* **6**:76 (2019), CC-BY-4.0, via CatBench (Zenodo concept DOI 10.5281/zenodo.17157085).
- Validation line: CMR CatApp (Hummelshøj *et al.*, *Angew. Chem. Int. Ed.* 2012).
- External cross-check: Catalysis-Hub GraphQL API.

## A selected reference list

1. Mamun et al., *Sci. Data* **6**:76 (2019). DOI 10.1038/s41597-019-0082-4.
2. Nørskov et al., *J. Electrochem. Soc.* **152**, J23 (2005).
3. Hummelshøj et al., *Angew. Chem. Int. Ed.* (2012). DOI 10.1002/anie.201107947.
4. Duan et al. / CatBench, *Cell Rep. Phys. Sci.* (2025).
5. Chen et al., XGBoost, *KDD* (2016).
6. Lundberg & Lee, SHAP, *NeurIPS* (2017).
7. Batatia et al., MACE Foundation Model (MACE-MPA-0).
8. Deng et al., CHGNet, *Nat. Mach. Intell.* (2023).
9. Barroso-Luque et al., Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models, arXiv:2410.12771 (2024).
