#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate a deterministic full-data v3 model.json AND the parity inputs in
ONE process, guaranteeing they come from the same in-memory model.

Multithreaded histogram reductions can introduce tiny float-order differences
between separately trained models, so we train with n_jobs=1 (deterministic)
and reuse scripts/26_export_web_assets_v2.export_model to emit model.json.

Outputs:
  web/assets/model.json              deterministic full-data v3 (single-thread)
  web/assets/parity_check_inputs.json  schema for scripts/parity_check.js
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
ASSETS = ROOT / "web" / "assets"
SEED = 42

# import script 26 for export_model (guarded, no side effects)
spec = importlib.util.spec_from_file_location(
    "m26", ROOT / "scripts" / "26_export_web_assets_v2.py")
m26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m26)

cases = json.loads((ASSETS / "parity_cases.json").read_text(encoding="utf-8"))
FN = cases["feature_names"]
assert len(FN) == 84

df = pd.read_csv(DP / "hstar_features_v3_84feat.csv")
assert len(df) == 1836 and df["comp"].nunique() == 1836
comps = df["comp"].tolist()
X = df[FN].astype(float)
y = df["energy_eV"].astype(float).values

params = json.loads((DP / "hstar_best_params.json").read_text(encoding="utf-8"))["xgb_best_params"]
model = XGBRegressor(random_state=SEED, n_jobs=1, **params)
model.fit(X, y)
py_preds = model.predict(X).astype(float)
print(f"Trained deterministic full-data v3 (n_jobs=1); params={params}")

# emit model.json from THIS model (dump to a path we control; avoid the
# NamedTemporaryFile reopen that fails on Windows)
import tempfile
booster = model.get_booster()
cfg = json.loads(booster.save_config())
base_raw = cfg["learner"]["learner_model_param"]["base_score"]
base_score = float(np.float32(float(str(base_raw).strip("[]"))))
assert cfg["learner"]["objective"]["name"] == "reg:squarederror"
dump_dir = Path(tempfile.mkdtemp(prefix="xgb_dump_"))
dump_path = dump_dir / "trees.json"
booster.dump_model(str(dump_path), dump_format="json")
trees = json.loads(dump_path.read_text(encoding="utf-8"))
assert len(trees) == params["n_estimators"]
payload = {
    "meta": {"target": "E_ads (H* adsorption energy)", "unit": "eV",
             "dG_H_formula": "dG_H = E_ads + 0.24", "objective": "reg:squarederror",
             "n_trees": len(trees), "n_features": len(FN), "trained_rows": int(X.shape[0]),
             "model_version": "v3-84feat deterministic n_jobs=1", "params": params,
             "seed": SEED},
    "base_score": base_score, "feature_names": FN, "trees": trees,
}
(ASSETS / "model.json").write_text(json.dumps(payload, separators=(",", ":")))
print(f"Emitted model.json: {len(trees)} trees, base_score={base_score!r}")

rng = np.random.default_rng(42)
ridx = rng.choice(len(comps), size=50, replace=False)
parity = {
    "comps": comps,
    "py_preds": [float(v) for v in py_preds],
    "csv_features": X.values.tolist(),
    "feature_names": FN,
    "random_comps": [comps[i] for i in ridx],
    "random_py_preds": [float(py_preds[i]) for i in ridx],
    "random_note": "50 comps reproducibly sampled (seed 42) from the 1836 rows",
    "reference_model": "full-data v3, deterministic n_jobs=1 (matches model.json)",
}
(ASSETS / "parity_check_inputs.json").write_text(json.dumps(parity), encoding="utf-8")
print("Wrote parity_check_inputs.json rows=1836 random=50")
