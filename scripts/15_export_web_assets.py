# -*- coding: utf-8 -*-
"""
15_export_web_assets.py — 把 H* 吸附能 XGBoost 模型导出为纯前端 JS 可推理资产

流程：
1. 重训出厂模型：hstar_dataset_v2.csv 全量 1836 行 × 36 特征，
   参数取 data_processed/hstar_best_params.json（lr=0.03/depth=7/n=400/sub=0.8），种子 42；
2. 导出 web/assets/model.json（dump_model JSON 树 + base_score + 特征名 + 元信息）；
3. 导出 web/assets/elements.json（mendeleev 六属性 × 数据集 37 元素 + 常见补充金属）；
4. 导出页面数据：predictions_all.json / candidates_top.json / shap_global.json /
   descriptor_stats.json；
5. 生成 parity 输入并调用 node scripts/parity_check.js 做 JS↔Python 逐行奇偶校验
   （阈值 max|diff| < 1e-9），写 web/assets/parity_report.json。

特征规范唯一事实来源：web/assets/feature_spec.md。
"""
import json
import math
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
ASSETS = ROOT / "web" / "assets"
SEED = 42
DG_SHIFT = 0.24  # ΔG_H* = E_ads + 0.24 eV

ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw",
           "site_spread"]  # site_spread: 泄漏特征，禁止入模

PROPS = ["en", "radius", "group", "period", "ie1", "d_el"]
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
D_ELEC_RE = re.compile(r"(\d)d(\d*)")

# v2 数据集元素之外、补充收录的常见金属（mendeleev 六属性须齐全）
EXTRA_ELEMENTS = ["Ge", "Sb"]


# ---------- 特征管线镜像（与 06_hstar_features.py 逐行一致，供 parity 使用） ----------

def parse_comp(comp):
    tokens = COMP_TOKEN.findall(comp)
    rebuilt = "".join(el + num for el, num in tokens)
    if rebuilt != comp or not tokens:
        raise ValueError(f"[comp 解析异常] {comp!r}")
    d = {}
    for el, num in tokens:
        if el in d:
            raise ValueError(f"[comp 重复元素] {comp!r}")
        d[el] = int(num) if num else 1
    return d


def canonicalize(comp):
    """排序 + gcd 约简（与 predict.js parseComp 一致）。"""
    d = parse_comp(comp)
    els = sorted(d)
    g = 0
    for e in els:
        g = math.gcd(g, d[e])
    counts = {e: d[e] // g for e in els}
    canon = "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in els)
    return counts, els, canon


def d_electron_count(econf):
    tail = str(econf).split("]")[-1]
    return sum(int(n) if n else 1 for _, n in D_ELEC_RE.findall(tail))


def build_element_table(symbols):
    from mendeleev import element as md_element
    table = {}
    for sym in symbols:
        e = md_element(sym)
        vals = {
            "en": e.en_pauling, "radius": e.atomic_radius, "group": e.group_id,
            "period": e.period, "ie1": (e.ionenergies or {}).get(1),
            "d_el": d_electron_count(e.econf) if str(e.econf) not in ("None", "") else None,
        }
        miss = [k for k, v in vals.items() if v is None]
        if miss:
            raise ValueError(f"mendeleev 缺失 {sym}.{miss}，请补 HAND_LOOKUP 或移除该元素")
        table[sym] = {k: float(v) for k, v in vals.items()}
    return table


def infer_structure(counts, els):
    """structure/facet 推断（feature_spec §4；源自 05 的计量模式规则）。"""
    if len(els) == 1:
        return "A1", "111", True
    if len(els) == 2:
        c = sorted(counts[e] for e in els)
        if c[0] == c[1]:
            return "L10", "101", True
        if c[1] == 3 * c[0]:
            return "L12", "111", True
    return None, None, False


def featurize(comp, etab):
    """Python 端全管线镜像：comp 字符串 → 36 特征 dict（供随机组成 parity）。"""
    counts, els, canon = canonicalize(comp)
    structure, facet, known = infer_structure(counts, els)
    w = np.array([counts[e] for e in els], dtype=float)
    w /= w.sum()
    feats = {"n_elements": len(els)}
    for p in PROPS:
        v = np.array([etab[e][p] for e in els], dtype=float)
        mu = float(np.average(v, weights=w))
        feats[f"{p}_wmean"] = mu
        feats[f"{p}_max"] = float(v.max())
        feats[f"{p}_min"] = float(v.min())
        feats[f"{p}_range"] = float(v.max() - v.min())
        feats[f"{p}_wstd"] = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
    feats["structure_A1"] = 1 if structure == "A1" else 0
    feats["structure_L10"] = 1 if structure == "L10" else 0
    feats["structure_L12"] = 1 if structure == "L12" else 0
    feats["facet_101"] = 1 if facet == "101" else 0
    feats["facet_111"] = 1 if facet == "111" else 0
    return feats, canon, structure, facet, known


# ---------- 导出 ----------

def _json_clean(o):
    """NaN/inf → None（Python json 默认输出 NaN，非法 JSON，node 无法解析）。"""
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def export_model(model, feats, n_rows, out_path):
    booster = model.get_booster()
    cfg = json.loads(booster.save_config())
    base_raw = cfg["learner"]["learner_model_param"]["base_score"]
    base_score = float(np.float32(float(str(base_raw).strip("[]"))))
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "reg:squarederror", objective
    with tempfile.NamedTemporaryFile("r", suffix=".json") as tf:
        booster.dump_model(tf.name, dump_format="json")
        trees = json.load(open(tf.name))
    assert len(trees) == model.n_estimators
    payload = {
        "meta": {
            "target": "E_ads (H* adsorption energy)",
            "unit": "eV",
            "dG_H_formula": "dG_H = E_ads + 0.24",
            "objective": objective,
            "n_trees": len(trees),
            "n_features": len(feats),
            "trained_rows": int(n_rows),
            "params": _json_clean(model.get_params()),
            "seed": SEED,
            "note": "推理: acc=f32(base_score); 每棵树 acc=f32(acc+f32(leaf)); "
                    "分裂比较 f32(feature)<f32(split_condition)，yes/no/missing 见树节点",
        },
        "base_score": base_score,
        "feature_names": feats,
        "trees": trees,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def f32_tree_predict(trees, base_score, fdict):
    """Python 端 f32 累加镜像（自洽性断言用，与 predict.js 同语义）。"""
    acc = np.float32(base_score)
    for t in trees:
        n = t
        while "leaf" not in n:
            fv = np.float32(fdict[n["split"]])
            cond = np.float32(n["split_condition"])
            nxt = n["yes"] if fv < cond else n["no"]
            n = next(c for c in n["children"] if c["nodeid"] == nxt)
        acc = np.float32(acc + np.float32(n["leaf"]))
    return float(acc)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    feats = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feats], df["energy_eV"].values
    assert X.shape[0] == 1836 and len(feats) == 36, X.shape
    print(f"[读入] hstar_dataset_v2.csv  {X.shape[0]} 行 × {X.shape[1]} 特征")

    # ---- 结构推断规则与数据一致性断言（JS 端按同一规则推断）----
    for _, r in df.iterrows():
        counts = parse_comp(r["comp"])
        st, fa, known = infer_structure(counts, sorted(counts))
        assert known and st == r["structure"] and str(fa) == str(r["facet"]), (
            r["comp"], r["structure"], r["facet"], st, fa)
    print("[断言] 1836 行的 structure/facet 均可由计量模式规则唯一推断 ✓")

    # ---- 1. 重训出厂模型 ----
    best = json.loads((DP / "hstar_best_params.json").read_text())
    params = best["xgb_best_params"]
    print(f"[训练] XGBRegressor(random_state={SEED}, {params}) 全量 1836 行")
    model = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    model.fit(X, y)
    py_preds = model.predict(X)  # float32

    # ---- 2. model.json ----
    payload = export_model(model, feats, X.shape[0], ASSETS / "model.json")
    size_mb = (ASSETS / "model.json").stat().st_size / 1e6
    print(f"[写出] model.json  {len(payload['trees'])} 棵树, "
          f"base_score={payload['base_score']!r}, {size_mb:.2f} MB")

    # 自洽性断言：f32 树遍历复现 booster.predict（全量逐行）
    trees = payload["trees"]
    bs = payload["base_score"]
    n_bad = 0
    for i in range(X.shape[0]):
        manual = f32_tree_predict(trees, bs, X.iloc[i].to_dict())
        if manual != float(py_preds[i]):
            n_bad += 1
    assert n_bad == 0, f"f32 镜像与 booster.predict 不一致: {n_bad} 行"
    print("[断言] Python f32 树遍历 == booster.predict（1836/1836 位级一致）✓")

    # ---- 3. elements.json ----
    train_elements = sorted({e for c in df["comp"] for e in COMP_TOKEN.findall(c)
                             for e in [e[0]]})
    all_symbols = sorted(set(train_elements) | set(EXTRA_ELEMENTS))
    etab = build_element_table(all_symbols)
    elements_payload = {
        "meta": {
            "source": "mendeleev 1.2.0 (en_pauling/atomic_radius/group_id/period/"
                      "ionenergies[1]/econf→d_el)",
            "props": PROPS,
            "training_elements": train_elements,
        },
        "elements": etab,
    }
    (ASSETS / "elements.json").write_text(
        json.dumps(elements_payload, ensure_ascii=False, indent=1))
    print(f"[写出] elements.json  {len(etab)} 种元素（训练集 {len(train_elements)} 种）")

    # ---- 4a. predictions_all.json ----
    pred_rows = []
    for i, r in df.iterrows():
        p = float(py_preds[i])
        pred_rows.append({
            "comp": r["comp"], "structure": r["structure"], "facet": str(r["facet"]),
            "n_raw": int(r["n_raw"]),
            "E_ads": float(r["energy_eV"]), "pred_E_ads": p,
            "dG_true": float(r["energy_eV"]) + DG_SHIFT, "dG_pred": p + DG_SHIFT,
            "abs_err": abs(p - float(r["energy_eV"])),
            "in_domain": True,
        })
    (ASSETS / "predictions_all.json").write_text(
        json.dumps(pred_rows, separators=(",", ":")))
    mae = float(np.mean([r["abs_err"] for r in pred_rows]))
    print(f"[写出] predictions_all.json  {len(pred_rows)} 行, 训练集内 MAE={mae:.4f} eV")

    # ---- 4b. candidates_top.json ----
    cand = pd.read_csv(DP / "candidate_rankings_hstar.csv").head(50)
    cand["red_flags"] = cand["red_flags"].fillna("")
    (ASSETS / "candidates_top.json").write_text(cand.to_json(
        orient="records", force_ascii=False))
    print(f"[写出] candidates_top.json  top {len(cand)}")

    # ---- 4c. shap_global.json ----
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if hasattr(sv, "values"):
        sv = sv.values
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
    order = np.argsort(-mean_abs)
    shap_rows = [{"feature": feats[i], "mean_abs_shap": float(mean_abs[i]),
                  "rank": int(r + 1)}
                 for r, i in enumerate(order[:15])]
    (ASSETS / "shap_global.json").write_text(json.dumps({
        "meta": {"method": "shap.TreeExplainer, mean(|SHAP|) over 1836 training rows",
                 "unit": "eV"},
        "top15": shap_rows}, ensure_ascii=False, indent=1))
    print(f"[写出] shap_global.json  top1={shap_rows[0]['feature']} "
          f"({shap_rows[0]['mean_abs_shap']:.4f} eV)")

    # ---- 4d. descriptor_stats.json ----
    stats = {}
    for f in feats:
        col = X[f].astype(float)
        stats[f] = {"min": float(col.min()), "max": float(col.max()),
                    "median": float(col.median()), "mean": float(col.mean())}
    (ASSETS / "descriptor_stats.json").write_text(json.dumps({
        "meta": {"n_rows": int(X.shape[0]), "usage": "输入特征在训练分布中的位置"},
        "stats": stats}, indent=1))
    print(f"[写出] descriptor_stats.json  {len(stats)} 个特征")

    # ---- 5. parity：写输入，调用 node ----
    rng = random.Random(SEED)
    random_comps, seen = [], set()
    while len(random_comps) < 50:
        k = rng.choice([2, 2, 3])
        els = sorted(rng.sample(train_elements, k))
        stoich = [rng.randint(1, 3) for _ in els]
        g = 0
        for s in stoich:
            g = math.gcd(g, s)
        stoich = [s // g for s in stoich]
        canon = "".join(e + (str(s) if s > 1 else "") for e, s in zip(els, stoich))
        if canon in seen:
            continue
        seen.add(canon)
        random_comps.append(canon)
    rand_feats = [featurize(c, etab)[0] for c in random_comps]
    rand_X = pd.DataFrame(rand_feats)[feats]
    rand_preds = model.predict(rand_X)

    parity_inputs = {
        "comps": df["comp"].tolist(),
        "py_preds": [float(p) for p in py_preds],
        "csv_features": X.astype(float).values.tolist(),
        "feature_names": feats,
        "random_comps": random_comps,
        "random_py_preds": [float(p) for p in rand_preds],
    }
    tmp = Path(tempfile.mkdtemp(prefix="hstar_parity_"))
    (tmp / "parity_inputs.json").write_text(json.dumps(parity_inputs))
    node_script = ROOT / "scripts" / "parity_check.js"
    print(f"[parity] node {node_script} ...")
    r = subprocess.run(["node", str(node_script), str(tmp / "parity_inputs.json"),
                        str(ASSETS)],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"parity_check.js 失败（exit {r.returncode}）")
    report = json.loads((ASSETS / "parity_report.json").read_text())
    print(f"[parity] dataset max|diff|={report['dataset']['max_diff']:.3e} "
          f"random max|diff|={report['random']['max_diff']:.3e} "
          f"pass={report['pass']}")
    if not report["pass"]:
        raise SystemExit("parity 未通过（阈值 1e-9）")
    print("[完成] 全部资产已导出到 web/assets/")


if __name__ == "__main__":
    main()
