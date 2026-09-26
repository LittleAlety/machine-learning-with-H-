# -*- coding: utf-8 -*-
"""39_cathub_batch_lookup.py - check if frozen C-3 v2 systems already exist in CatHub.

Before spending ~200k core-hours, query Catalysis-Hub for each pre-registered
Batch A / B / sentinel composition. If H* or OH* records already exist (any
publication/functional), they give free validation and may shrink the DFT ask.

Key is read ONLY from the CATHUB_API_KEY environment variable (never hard-coded).
Outputs:
    outputs/cathub_batch_lookup.csv
    notes/cathub_batch_lookup.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://api.catalysis-hub.org/graphql"
BATCH_FILE = ROOT / "innovation" / "c3_v2" / "batches_preregistered.json"
OUT_CSV = ROOT / "outputs" / "cathub_batch_lookup.csv"
OUT_MD = ROOT / "notes" / "cathub_batch_lookup.md"
NODE_FIELDS = "id Equation chemicalComposition reactionEnergy facet sites pubId dftCode dftFunctional"

# Composition forms to try (CatHub reduces/orders; we try canonical and a couple variants).
SYSTEMS = {
    "A": ["HgRu", "Co3Ga", "CoPt3", "AuRu", "HgTc", "AuOs", "CrHg", "CrIn", "HgNi3", "CdCr"],
    "B": ["Cr3Hf", "CrNb3", "Al3Ta", "InZr", "AlTa", "Al3Zr", "Al3Hf", "AlNb", "Al3Nb", "HfPb"],
    "S": ["Cr3Pb", "LaPb", "CrTl", "Co", "CuFe3", "Fe3Mo", "FeTi", "Ag3Sc"],
}


def gql(query, variables, key, retries=4):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for i in range(retries):
        try:
            req = Request(API_URL, data=body,
                          headers={"Content-Type": "application/json", "X-API-Key": key})
            with urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
            if "errors" in out:
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception as e:  # noqa
            wait = 10 * (i + 1)
            print(f"  [retry {i+1}] {e}; wait {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("query failed repeatedly")


def lookup(comp, key):
    q = ("query($c: String!) { reactions(first: 50, chemicalComposition: $c) {"
         f" totalCount edges {{ node {{ {NODE_FIELDS} }} }} }} }}")
    d = gql(q, {"c": comp}, key)
    return d["reactions"]["totalCount"], [e["node"] for e in d["reactions"]["edges"]]


def main():
    key = os.environ.get("CATHUB_API_KEY")
    if not key:
        sys.exit("[fatal] set CATHUB_API_KEY env var first")
    rows = []
    for grp, comps in SYSTEMS.items():
        for comp in comps:
            try:
                total, nodes = lookup(comp, key)
            except Exception as e:
                print(f"[{grp}] {comp}: ERROR {e}", flush=True)
                total, nodes = -1, []
            hstar = [n for n in nodes if "H*" in (n.get("Equation") or "")]
            ohstar = [n for n in nodes if "OH*" in (n.get("Equation") or "")]
            print(f"[{grp}] {comp:7s} total={total:3d} H*={len(hstar):2d} OH*={len(ohstar):2d}", flush=True)
            for n in nodes:
                rows.append({"group": grp, "comp": comp, "equation": n.get("Equation"),
                             "energy_eV": n.get("reactionEnergy"), "facet": n.get("facet"),
                             "sites": json.dumps(n.get("sites"), sort_keys=True),
                             "pubId": n.get("pubId"), "dftCode": n.get("dftCode"),
                             "dftFunctional": n.get("dftFunctional"), "totalCount": total})
            time.sleep(2.0)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    md = ["# CatHub 预注册体系存量查询（决定 DFT 是否可缩减）", "",
          "| 组 | 组成 | 总记录 | H* | OH* |", "|---|---|---:|---:|---:|"]
    for grp, comps in SYSTEMS.items():
        for comp in comps:
            sub = df[df["comp"] == comp]
            tot = int(sub["totalCount"].iloc[0]) if len(sub) else 0
            nh = sum("H*" in str(e) for e in sub["equation"])
            no = sum("OH*" in str(e) for e in sub["equation"])
            md.append(f"| {grp} | {comp} | {tot} | {nh} | {no} |")
    md += ["", "若某体系已存在 BEEF-vdW 的 H*/OH* 记录，可直接用作锚定/验证，相应缩减 Stage-2 任务。",
           f"明细见 `{OUT_CSV.name}`。"]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT_CSV.name} ({len(df)} rows), {OUT_MD.name}")


if __name__ == "__main__":
    main()
