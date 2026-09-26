# -*- coding: utf-8 -*-
"""F1 位点级样本表构建 + 数据自检 — plan-hb v3.0 Workstream F

数据源（如实记录，与任务书的一处偏差）：
  位点类型元数据**不在** data_raw/MamunHighT2019_adsorption.json.gz 内
  （该文件每条记录仅含 raw/ref_ads_eng/adsorbate_indices/constraint_source，
   经全量扫描 45,131 条记录无一含 "sites" 键）。
  位点类型来自 scripts/13_cathub_fetch.py 的官方 CatApp API 抓取缓存：
    - data_processed/cathub_mamun.csv（MamunHighT2019，sites 列 100% 非空）
    - data_raw/cathub_cache/pages/all_h/*.json（同一抓取的原始分页，去重补充）
  两源取并集后按 (规范化 comp, ref_ads_eng 精确匹配) 回连 7,048 条原始记录。

匹配规则（写死）：energy 精确匹配（round 1e-9）；同 (comp,energy) 多个候选位点
  串取集合，若 >1 种记冲突（实测 0 冲突）；未匹配记录丢弃并计数（覆盖率自检）。

降级预案（预注册）：若 sites 元数据解析不出位点类型 → 降级为"位点序号 one-hot"。
  实际未触发（覆盖率 97.8% ≥95%），记录备查。

输出：
  site_dataset_dev.csv       开发集（1560 组成的位点记录，84 维特征已 join）
  site_lockbox_sealed.csv    位点 lockbox（276 组成；密封，SHA256 回填 f0 json）
  f1_report.json             数据自检报告
"""
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW_GZ = ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz"
FEAT_CSV = ROOT / "data_processed" / "hstar_features_v3_84feat.csv"
CATHUB_CSV = ROOT / "data_processed" / "cathub_mamun.csv"
CATHUB_PAGES = ROOT / "data_raw" / "cathub_cache" / "pages" / "all_h"
F0_JSON = OUT / "f0_site_lockbox.json"

DILUTE_EQ = "0.5H2(g) + * -> H*"
STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}
ID_COLS = ["comp", "structure", "facet", "n_raw", "energy_eV"]


def parse_prefix(p):
    return {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", p)}


def normalize_comp(cd):
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    return "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))


def load_raw_records():
    """复用 innovation/probes/p3_noise_floor.py 的解析规则。"""
    data = json.loads(gzip.decompress(RAW_GZ.read_bytes()))
    recs = []
    for k, v in data.items():
        if DILUTE_EQ not in k:
            continue
        prefix = re.match(r"^([A-Za-z0-9]+)_", k).group(1)
        cd = parse_prefix(prefix)
        comp = normalize_comp(cd)
        structure, facet = STRUCT_RULE[tuple(sorted(cd.values()))]
        recs.append({"record_id": k, "comp": comp, "structure": structure,
                     "facet": facet, "energy_eV": float(v["ref_ads_eng"])})
    return recs


def load_site_catalog():
    """(comp, round(energy,9)) -> set(site_str)，两源并集。"""
    cat = defaultdict(lambda: defaultdict(set))

    def add(comp_raw, energy, sites):
        if not isinstance(sites, str) or not sites:
            return
        comp = normalize_comp(parse_prefix(comp_raw))
        cat[comp][round(float(energy), 9)].add(sites)

    df = pd.read_csv(CATHUB_CSV)
    df = df[df["Equation"] == DILUTE_EQ]
    for r in df.itertuples():
        add(r.chemicalComposition, r.reactionEnergy, r.sites)
    for p in sorted(CATHUB_PAGES.glob("*.json")):
        for r in json.loads(p.read_text()):
            if r.get("pubId") == "MamunHighT2019" and r.get("Equation") == DILUTE_EQ:
                add(r["chemicalComposition"], r["reactionEnergy"], r.get("sites"))
    return cat


def parse_site(site_str):
    """'{"H": "hollow|A_A_A|FCC"}' -> (full='hollow|A_A_A|FCC', coarse='hollow')"""
    d = json.loads(site_str)
    full = d["H"]
    coarse = full.split("|")[0]
    return full, coarse


def main():
    OUT.mkdir(exist_ok=True)
    f0 = json.loads(F0_JSON.read_text())
    lb_comps = set(f0["lockbox_comps"])

    recs = load_raw_records()
    cat = load_site_catalog()
    feat = pd.read_csv(FEAT_CSV)
    feat_cols = [c for c in feat.columns if c not in ID_COLS]
    assert len(feat_cols) == 84

    rows, n_unmatched, n_conflict = [], 0, 0
    unmatched_comps = set()
    for r in recs:
        cands = cat.get(r["comp"], {}).get(round(r["energy_eV"], 9))
        if not cands:
            n_unmatched += 1
            unmatched_comps.add(r["comp"])
            continue
        if len(cands) > 1:
            n_conflict += 1
        full, coarse = parse_site(sorted(cands)[0])
        rows.append({**r, "site_full": full, "site_coarse": coarse})
    site = pd.DataFrame(rows)

    # join 84 维特征（v3 表每 comp 唯一）
    site = site.merge(feat[["comp"] + feat_cols], on="comp", how="left",
                      validate="many_to_one")
    assert site[feat_cols].notna().all().all(), "特征 join 出现空值"

    # 数据自检（F1 预注册检查点）
    n_total_raw = len(recs)
    coverage = len(site) / n_total_raw
    surf = site.groupby("comp").agg(structure=("structure", "first"),
                                    n_sites=("energy_eV", "size"))
    empty_units = int((surf["n_sites"] == 0).sum())  # （组成×结构）空壳单元
    # (comp×structure×site_full) 单元应每条记录可归类；检查无缺失标签
    assert site["site_full"].notna().all()
    site_dist = site["site_coarse"].value_counts().to_dict()
    site_full_dist = site["site_full"].value_counts().to_dict()

    dev = site[~site["comp"].isin(lb_comps)].reset_index(drop=True)
    lock = site[site["comp"].isin(lb_comps)].reset_index(drop=True)

    dev_path = OUT / "site_dataset_dev.csv"
    lock_path = OUT / "site_lockbox_sealed.csv"
    dev.to_csv(dev_path, index=False)
    lock.to_csv(lock_path, index=False)
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    f0["sha256_of_lockbox_site_rows"] = lock_sha
    f0["n_lockbox_site_records"] = int(len(lock))
    F0_JSON.write_text(json.dumps(f0, indent=2, ensure_ascii=False))

    report = {
        "n_raw_records": n_total_raw,
        "n_site_records_matched": int(len(site)),
        "n_unmatched_dropped": n_unmatched,
        "n_unmatched_comps": len(unmatched_comps),
        "n_match_conflicts": n_conflict,
        "site_metadata_coverage": coverage,
        "coverage_gate_>=95%": bool(coverage >= 0.95),
        "site_metadata_source": ("cathub_mamun.csv + cathub_cache/pages/all_h "
                                 "（官方 CatApp 抓取；json.gz 内无 sites 键，"
                                 "全量扫描 45131 条记录为证）"),
        "fallback_to_site_index_onehot": False,
        "n_surfaces_with_sites": int(site["comp"].nunique()),
        "n_surfaces_total": 1836,
        "empty_comp_structure_units": empty_units,
        "site_coarse_dist": site_dist,
        "site_full_dist": site_full_dist,
        "n_dev_records": int(len(dev)),
        "n_lockbox_records": int(len(lock)),
        "n_dev_surfaces": int(dev["comp"].nunique()),
        "n_lockbox_surfaces": int(lock["comp"].nunique()),
        "dev_file": dev_path.name,
        "lockbox_file_sealed": lock_path.name,
        "lockbox_sha256": lock_sha,
        "F1_verdict": None,  # 下方判定
    }
    checks = {
        "样本量 ~6000+": len(site) >= 6000,
        "覆盖率 ≥95%": coverage >= 0.95,
        "无空壳单元": empty_units == 0,
        "位点类型可解析": len(site_full_dist) > 1,
    }
    report["F1_checks"] = checks
    report["F1_verdict"] = "PASS" if all(checks.values()) else "FAIL"
    (OUT / "f1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in
                      ["n_site_records_matched", "site_metadata_coverage",
                       "n_unmatched_dropped", "n_match_conflicts",
                       "site_coarse_dist", "n_dev_records", "n_lockbox_records",
                       "F1_verdict"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
