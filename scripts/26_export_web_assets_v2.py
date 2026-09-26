# -*- coding: utf-8 -*-
"""
26_export_web_assets_v2.py — 全量合并 84 特征正式模型落地 + web 资产导出（v2）

背景：notes/26_descriptor_refinement.md —— 84 特征配置（基线 36 + 文献 4 MagpieData
新描述符 48）通过公平契约（嵌套 CV OOF MAE 0.1139，Δ=+0.0059 eV vs 0.1198，
5/5 外折配对改善），现落成正式模型。

特征同源保证：元素表/统计展开直接 import scripts/23_descriptor_refine.py 的
build_element_table / comp_stats（含 Sn 熔点 505.08 K 手工值、Tc 兜底），
并对 1836 行重建特征与 outputs/desrefine_new_features.csv 逐位断言一致。

产物（全部先写 /tmp，main 末尾统一 cp 覆盖；rename 在挂载盘上会 Permission denied）：
  web/assets/model.json                400 棵树 + base_score(f32) + 84 特征名
  web/assets/elements.json             39 元素 × 13 属性 + 单位说明
  web/assets/feature_spec.md           84 特征完整规范（mode 平局规则逐位定义）
  web/assets/predictions_all.json      新模型 OOF（GroupKFold5，同 fold）预测
  web/assets/candidates_top.json       按 OOF |ΔG_H*| 重排的 top-50 候选
  web/assets/shap_global.json          新模型全量 SHAP top-15（面板取 top-12）
  web/assets/descriptor_distributions.json  新 SHAP top-12 的 32-bin 分布
  web/assets/descriptor_stats.json     84 特征训练分布统计（同步更新）
  web/assets/parity_cases.json         20 个奇偶校验用例（域内 15 + 域外 5）
  data_processed/hstar_features_v3_84feat.csv  1836 行 × 84 特征 + 目标快照

自洽性断言：f32 树遍历 == booster.predict（1836/1836 位级一致）。
"""
import importlib.util
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DP = ROOT / "data_processed"
ASSETS = ROOT / "web" / "assets"
OUT23 = ROOT / "outputs"
SEED = 42
DG_SHIFT = 0.24
ID_COLS = ["comp", "structure", "facet", "energy_eV", "n_raw", "site_spread"]
EXTRA_ELEMENTS = ["Ge", "Sb"]  # 与 15 号脚本一致的补充元素
COMP_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# 红旗规则（沿用 08_hstar_interpret.py）
RADIOACTIVE = {"Tc"}
OXOPHILIC = {"La", "Y", "Sc"}
LOEO_RISKY = {"Mn", "Bi", "Fe"}

PROP_CN = {
    "en": "电负性", "radius": "原子半径 (pm)", "group": "族号", "period": "周期",
    "ie1": "第一电离能 (eV)", "d_el": "d 电子数",
    "melting_point": "熔点 (K)", "mendeleev_number": "门捷列夫数",
    "covalent_radius": "共价半径 (pm)", "n_valence": "价电子数",
    "nd_valence": "d 价电子数", "n_unfilled": "未填满电子数",
    "nd_unfilled": "未填满 d 电子数",
}
STAT_CN = {"wmean": "加权平均{p}", "max": "最大{p}", "min": "最小{p}",
           "range": "{p}极差", "wstd": "{p}加权标准差", "mode": "{p}众数（主组元）"}
UNITS = {
    "en": "Pauling 标度，无量纲", "radius": "pm", "group": "族序数，无量纲",
    "period": "周期序数，无量纲", "ie1": "eV", "d_el": "电子数",
    "melting_point": "K", "mendeleev_number": "序数，无量纲",
    "covalent_radius": "pm（Pyykkö 双键半径）", "n_valence": "电子数",
    "nd_valence": "电子数", "n_unfilled": "电子数", "nd_unfilled": "电子数",
}


def load_m23():
    """import scripts/23_descriptor_refine.py 作为特征同源模块。"""
    spec = importlib.util.spec_from_file_location(
        "m23", ROOT / "scripts" / "23_descriptor_refine.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- 组成规范化与结构推断（与 15 号脚本 / predict.js 逐位一致） ----------
def canonicalize(comp):
    tokens = COMP_TOKEN.findall(comp)
    if "".join(el + num for el, num in tokens) != comp or not tokens:
        raise ValueError(f"[comp 解析异常] {comp!r}")
    d = {}
    for el, num in tokens:
        if el in d:
            raise ValueError(f"[comp 重复元素] {comp!r}")
        d[el] = int(num) if num else 1
    els = sorted(d)
    g = 0
    for e in els:
        g = math.gcd(g, d[e])
    counts = {e: d[e] // g for e in els}
    canon = "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in els)
    return counts, els, canon


def infer_structure(counts, els):
    if len(els) == 1:
        return "A1", "111", True
    if len(els) == 2:
        c = sorted(counts[e] for e in els)
        if c[0] == c[1]:
            return "L10", "101", True
        if c[1] == 3 * c[0]:
            return "L12", "111", True
    return None, None, False


def featurize84(comp, etab, m23, base_feats, new_cols):
    """comp 字符串 → 84 特征 dict（与训练侧同源：comp_stats 复用 m23）。"""
    counts, els, canon = canonicalize(comp)
    structure, facet, known = infer_structure(counts, els)
    d = {e: counts[e] for e in els}  # 字母序插入 → mode 平局取字母序首元素
    feats = {"n_elements": len(els)}
    for p in m23.BASE_PROPS:
        v = np.array([etab[e][p] for e in els], dtype=float)
        w = np.array([d[e] for e in els], dtype=float)
        w /= w.sum()
        mu = float(np.average(v, weights=w))
        feats[f"{p}_wmean"] = mu
        feats[f"{p}_max"] = float(v.max())
        feats[f"{p}_min"] = float(v.min())
        feats[f"{p}_range"] = float(v.max() - v.min())
        feats[f"{p}_wstd"] = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
    feats["structure_A1"] = int(structure == "A1")
    feats["structure_L10"] = int(structure == "L10")
    feats["structure_L12"] = int(structure == "L12")
    feats["facet_101"] = int(facet == "101")
    feats["facet_111"] = int(facet == "111")
    # 新 48 特征：与 23 号脚本同函数同顺序（35 统计 + 13 mode）
    nf = m23.comp_stats(d, etab, m23.NEW_PROPS)
    nf = {k: v for k, v in nf.items() if not k.endswith("_mode")}
    for p in m23.BASE_PROPS + m23.NEW_PROPS:
        nf[f"{p}_mode"] = m23.comp_stats(d, etab, [p])[f"{p}_mode"]
    feats.update(nf)
    out = {k: feats[k] for k in base_feats + new_cols}
    return out, canon, structure, facet, known


# ---------- 导出辅助（与 15 号脚本一致） ----------
def _json_clean(o):
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
    base_score = float(np.float32(float(str(base_raw).strip("[]"))))  # 括号陷阱处理
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
            "model_version": "v3-84feat（baseline36 + MagpieData 衍生 48）",
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


def red_flags(comp):
    els = set(re.findall(r"[A-Z][a-z]?", comp))
    fl = []
    if els & RADIOACTIVE:
        fl.append("含放射性" + "/".join(sorted(els & RADIOACTIVE)))
    if els & OXOPHILIC:
        fl.append("强亲氧" + "/".join(sorted(els & OXOPHILIC)) + "(表面稳定性存疑)")
    if els & LOEO_RISKY:
        fl.append("含LOEO高误差元素" + "/".join(sorted(els & LOEO_RISKY)))
    return ";".join(fl)


def feat_label(f):
    if f == "n_elements":
        return "元素个数"
    if f.startswith("structure_"):
        return {"structure_A1": "结构=A1 (FCC 单质)", "structure_L10": "结构=L1₀ (AB)",
                "structure_L12": "结构=L1₂ (A₃B)"}[f]
    if f.startswith("facet_"):
        return f"晶面=({f.split('_')[1]})"
    p, s = f.rsplit("_", 1)
    return STAT_CN[s].format(p=PROP_CN[p])


# ---------- feature_spec.md 生成 ----------
def build_feature_spec(feats):
    base_order = ["en", "radius", "group", "period", "ie1", "d_el"]
    new_props = ["melting_point", "mendeleev_number", "covalent_radius",
                 "n_valence", "nd_valence", "n_unfilled", "nd_unfilled"]
    lines = []
    lines.append("# H* 吸附能模型 v3（84 特征）— Web 端特征规范（feature_spec）\n")
    lines.append("本文件是 `web/assets/predict.js` 与 `scripts/26_export_web_assets_v2.py` "
                 "共同遵守的**唯一事实来源**。任何一端修改特征逻辑，两端必须同步。\n"
                 "来源代码：`scripts/06_hstar_features.py`（基线 36 特征化）、\n"
                 "`scripts/23_descriptor_refine.py`（新 48 特征，文献 4 MagpieData 复刻）、\n"
                 "`scripts/05_fetch_hstar.py`（structure/facet 推断）。\n"
                 "v2（36 特征）版规范见 git 历史。\n")
    lines.append("## 1. 输入与组成规范化\n")
    lines.append("输入：化学式字符串，如 `Pt3Ti`、`CuPt3`、`Ag`。\n")
    lines.append("- 解析正则：`([A-Z][a-z]?)(\\d*)`，数字缺省 = 1。\n"
                 "- 规范化：① 元素符号按 ASCII 字典序排序；② 计量数除以最大公约数；"
                 "③ 计量数 1 省略。\n"
                 "- 例：`TiPt3 → Pt3Ti`；`Cu2Pt2 → CuPt`。\n")
    lines.append("## 2. 元素属性（13 项，来源 mendeleev 1.3.0；完整数值见 elements.json）\n")
    lines.append("| 键 | 定义 | 单位 |\n|---|---|---|")
    prop_src = {
        "en": "`en_pauling`，Pauling 电负性",
        "radius": "`atomic_radius`，原子半径",
        "group": "`group_id`，族号",
        "period": "`period`，周期",
        "ie1": "`ionenergies[1]`，第一电离能",
        "d_el": "电子组态尾部（最后一个 `]` 之后）正则 `(\\d)d(\\d*)` 全部 d 占据求和（缺省=1）",
        "melting_point": "`melting_point`，熔点。**Sn 因同素异形体 mendeleev 缺值，"
                         "按 Magpie/文献手工值 505.08 K 补齐**",
        "mendeleev_number": "`mendeleev_number`，门捷列夫数",
        "covalent_radius": "`covalent_radius`（= Pyykkö 双键半径），共价半径",
        "n_valence": "`nvalence()`，价电子总数",
        "nd_valence": "最外层 d 亚层电子数（econf 解析；单占据写作 `5d` 缺省=1）",
        "n_unfilled": "Σ(亚层容量−占据)，对 econf 中部分占据亚层求和（容量 s=2/p=6/d=10/f=14）",
        "nd_unfilled": "仅 d 亚层的 (10−占据) 求和",
    }
    for p in base_order + new_props:
        lines.append(f"| `{p}` | {prop_src[p]} | {UNITS[p]} |")
    lines.append("")
    lines.append("## 3. 84 个建模特征（顺序 = model.json feature_names 顺序）\n")
    lines.append("组成解析得 `{元素: 计量数}`，元素列表 `els`（规范化后字母序），"
                 "权重 `w_i = c_i / Σc`。\n")
    lines.append("| # | 特征 | 定义 |\n|---|---|---|")
    lines.append("| 1 | `n_elements` | 元素个数 |")
    lines.append(f"| 2–31 | `{{prop}}_{{stat}}` | 6 基线属性（{'→'.join(base_order)}）× "
                 "5 统计（wmean→max→min→range→wstd） |")
    lines.append("| 32–34 | `structure_A1/L10/L12` | 结构 one-hot（推断规则见 §4） |")
    lines.append("| 35–36 | `facet_101/111` | 晶面 one-hot |")
    lines.append(f"| 37–71 | `{{prop}}_{{stat}}` | 7 新属性（{'→'.join(new_props)}）× "
                 "5 统计（同 wmean→max→min→range→wstd 顺序） |")
    lines.append(f"| 72–84 | `{{prop}}_mode` | 13 属性（6 基线 + 7 新，按 "
                 f"{'→'.join(base_order + new_props)} 顺序）的 mode 统计 |")
    lines.append("")
    lines.append("5 种加权统计（对属性值向量 v，权重 w）：\n")
    lines.append("- `wmean = Σ(v_i·w_i)/Σw_i`（numpy `np.average(v, weights=w)`）\n"
                 "- `max / min = 组元最大/最小值`；`range = max − min`\n"
                 "- `wstd = sqrt(Σ(w_i·(v_i−μ)²)/Σw_i)`，μ=wmean（加权，非无偏）\n")
    lines.append("**mode 统计（众数，逐位定义供 JS 复现）**：取**计量数最大的组元**的"
                 "该属性原始值；\n"
                 "若计量数并列（如 L1₀ 型 `CuPt` 1:1），取 `els` 中**字母序（ASCII）"
                 "最靠前**的元素\n（即规范化化学式中的首元素；例：`CuPt` 的 "
                 "`en_mode = en(Cu) = 1.90`）。\n单元素组成的 mode = 该元素属性值。\n")
    lines.append("**泄漏列**：`site_spread` 禁止入模。ID 列：`comp, structure, facet, "
                 "energy_eV, n_raw, site_spread`。\n")
    lines.append("## 4. structure / facet 推断规则（与 v2 一致）\n")
    lines.append("| 计量模式 | structure | facet |\n|---|---|---|\n"
                 "| 单元素 | A1 | 111 |\n| 双元素 3:1 | L12 | 111 |\n"
                 "| 双元素 1:1 | L10 | 101 |\n| 其他 | 未知（one-hot 全 0，in_domain=false） | 未知 |\n")
    lines.append("## 5. 模型与推理语义\n")
    lines.append("- XGBoost 3.4.1 `XGBRegressor`，`learning_rate=0.03, max_depth=7, "
                 "n_estimators=400, subsample=0.8, random_state=42`，"
                 "`objective=reg:squarederror`。\n"
                 "- 训练数据：`hstar_dataset_v2.csv` 全量 1836 行 × 84 特征"
                 "（快照 `data_processed/hstar_features_v3_84feat.csv`）。\n"
                 "- 目标：`energy_eV`；派生 **ΔG_H\\* = E_ads + 0.24 eV**。\n"
                 "- 推理（f32 语义，与 C++ 预测器位级一致）：\n"
                 "  1. `base_score` 取 `save_config()` 的 `learner_model_param.base_score`"
                 "（带方括号字符串，去括号取 float32）；\n"
                 "  2. 内部节点 `float32(feature) < float32(split_condition)` 走 yes 否则 no，"
                 "缺失走 missing；\n"
                 "  3. `acc = float32(acc + float32(leaf))`，自 `float32(base_score)` 逐树累加；\n"
                 "  4. 无 link 变换，acc 即 E_ads。\n")
    lines.append("## 6. in_domain 判定\n")
    lines.append("`in_domain = true` 当且仅当：① 全部元素在训练集 37 元素内"
                 "（elements.json `training_elements`）；② 计量模式命中 §4 前三种。\n"
                 "否则仍输出预测，`in_domain=false` 并给 warnings（外推，仅供参考）。\n")
    return "\n".join(lines)


# ---------- 主流程 ----------
def main():
    tmp = Path(tempfile.mkdtemp(prefix="hstar_v3_assets_"))
    print(f"[暂存] 全部产物先写 {tmp}")

    m23 = load_m23()
    df = pd.read_csv(DP / "hstar_dataset_v2.csv")
    base_feats = [c for c in df.columns if c not in ID_COLS]
    new_ref = pd.read_csv(OUT23 / "desrefine_new_features.csv")
    new_cols = list(new_ref.columns)
    feats = base_feats + new_cols
    assert len(feats) == 84, len(feats)
    print(f"[读入] v2 {df.shape[0]} 行 × {len(base_feats)} 基线 + {len(new_cols)} 新 = 84")

    # ---- 元素表（与 23 号脚本同源）与 1836 行特征重建断言
    train_elements = sorted({t[0] for c in df["comp"] for t in COMP_TOKEN.findall(c)})
    etab, notes = m23.build_element_table(train_elements)
    print(f"[元素表] {len(train_elements)} 训练元素；兜底记录: {notes}")
    recompute = pd.DataFrame(
        [featurize84(c, etab, m23, base_feats, new_cols)[0] for c in df["comp"]])
    X84 = pd.concat([df[base_feats].reset_index(drop=True), new_ref], axis=1)[feats]
    diff = (recompute[feats] - X84).abs().max().max()
    assert diff < 1e-9, f"重建特征与训练侧不一致，max|diff|={diff}"
    print(f"[断言] featurize84 重建 1836 行 × 84 特征与训练侧一致 ✓"
          f"（max|diff|={diff:.2e}，浮点求和顺序级）")
    # 结构/晶面推断一致性
    for _, r in df.iterrows():
        counts, els, _ = canonicalize(r["comp"])
        st, fa, known = infer_structure(counts, els)
        assert known and st == r["structure"] and str(fa) == str(r["facet"])
    print("[断言] structure/facet 计量模式推断规则与 1836 行一致 ✓")

    X, y = X84, df["energy_eV"].values
    groups = df["comp"].values

    # ---- 快照
    snap = pd.concat([df[["comp", "structure", "facet", "n_raw", "energy_eV"]]
                      .reset_index(drop=True), X84.reset_index(drop=True)], axis=1)
    snap.to_csv(tmp / "hstar_features_v3_84feat.csv", index=False)
    print(f"[写出] hstar_features_v3_84feat.csv {snap.shape}")

    # ---- 1. 最终模型（全量训练）
    params = json.loads((DP / "hstar_best_params.json").read_text())["xgb_best_params"]
    model = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    model.fit(X, y)
    py_preds = model.predict(X)
    print(f"[训练] 全量 1836 行 × 84 特征，参数 {params}")

    # ---- 2. model.json + f32 位级自洽断言
    payload = export_model(model, feats, X.shape[0], tmp / "model.json")
    n_bad = sum(f32_tree_predict(payload["trees"], payload["base_score"],
                                 X.iloc[i].to_dict()) != float(py_preds[i])
                for i in range(X.shape[0]))
    assert n_bad == 0, f"f32 镜像不一致 {n_bad} 行"
    print(f"[写出] model.json 400 棵树, base_score={payload['base_score']!r}; "
          f"f32 位级断言 1836/1836 ✓")

    # ---- 3. elements.json（39 元素 × 13 属性 + 单位）
    all_symbols = sorted(set(train_elements) | set(EXTRA_ELEMENTS))
    etab_all, notes_all = m23.build_element_table(all_symbols)
    print(f"[元素表/全集] {len(all_symbols)} 元素；兜底记录: {notes_all}")
    elements_payload = {
        "meta": {
            "source": "mendeleev 1.3.0；与 scripts/23_descriptor_refine.py "
                      "build_element_table 完全同源同值",
            "props": m23.BASE_PROPS + m23.NEW_PROPS,
            "units": UNITS,
            "manual_overrides": {
                "Sn.melting_point": "505.08 K（mendeleev 因同素异形体缺值，"
                                    "按 Magpie/文献值补齐）"},
            "prop_definitions": {
                "d_el": "econf 最后一个 ] 之后尾部的全部 d 占据求和（缺省=1）",
                "nd_valence": "最外层 d 亚层电子数",
                "n_unfilled": "Σ(亚层容量−占据)，容量 s=2/p=6/d=10/f=14",
                "nd_unfilled": "仅 d 亚层 (10−占据)",
                "n_valence": "mendeleev nvalence()",
            },
            "training_elements": train_elements,
            "extra_elements": EXTRA_ELEMENTS,
        },
        "elements": etab_all,
    }
    (tmp / "elements.json").write_text(
        json.dumps(elements_payload, ensure_ascii=False, indent=1))
    print(f"[写出] elements.json {len(etab_all)} 元素 × "
          f"{len(m23.BASE_PROPS + m23.NEW_PROPS)} 属性")

    # ---- 4. OOF（GroupKFold(5, comp)，与评估同一批 fold）
    oof = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    oof_mae = mean_absolute_error(y, oof)
    print(f"[OOF] GroupKFold(5) MAE = {oof_mae:.4f} eV（预期复现 0.1134）")

    # ---- 5. predictions_all.json（同 15 号 schema，pred 换为 OOF）
    pred_rows = [{
        "comp": r["comp"], "structure": r["structure"], "facet": str(r["facet"]),
        "n_raw": int(r["n_raw"]), "E_ads": float(r["energy_eV"]),
        "pred_E_ads": float(oof[i]),
        "dG_true": float(r["energy_eV"]) + DG_SHIFT,
        "dG_pred": float(oof[i]) + DG_SHIFT,
        "abs_err": abs(float(oof[i]) - float(r["energy_eV"])),
        "in_domain": True,
    } for i, r in df.iterrows()]
    (tmp / "predictions_all.json").write_text(
        json.dumps(pred_rows, separators=(",", ":")))
    print(f"[写出] predictions_all.json 1836 行（OOF；注：旧版为全量拟合值）")

    # ---- 6. candidates_top.json（按 OOF |ΔG| 重排，规则同 08 号脚本）
    rk = df[["comp", "structure", "facet", "n_raw", "energy_eV"]].copy()
    rk["pred_E_ads"] = oof
    rk["pred_dG"] = rk["pred_E_ads"] + DG_SHIFT
    rk["true_dG"] = rk["energy_eV"] + DG_SHIFT
    rk["abs_pred_dG"] = rk["pred_dG"].abs()
    rk = rk.sort_values("abs_pred_dG").reset_index(drop=True)
    rk.insert(0, "rank", rk.index + 1)
    best_abs = rk["abs_pred_dG"].min()
    rk["statistically_indistinguishable"] = (rk["abs_pred_dG"] - best_abs) < oof_mae
    rk["red_flags"] = rk["comp"].apply(red_flags)
    out_cols = ["rank", "comp", "structure", "facet", "n_raw", "energy_eV",
                "pred_E_ads", "true_dG", "pred_dG", "abs_pred_dG",
                "statistically_indistinguishable", "red_flags"]
    rk50 = rk[out_cols].head(50).round(4)
    (tmp / "candidates_top.json").write_text(rk50.to_json(
        orient="records", force_ascii=False))
    print(f"[写出] candidates_top.json top50；榜首 {rk50.iloc[0]['comp']} "
          f"pred_dG={rk50.iloc[0]['pred_dG']:+.4f}")
    print("[候选 top5]\n" + rk50.head(5)[["rank", "comp", "pred_dG", "true_dG",
                                          "red_flags"]].to_string(index=False))

    # ---- 7. shap_global.json
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if hasattr(sv, "values"):
        sv = sv.values
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
    order = np.argsort(-mean_abs)
    shap_rows = [{"feature": feats[i], "mean_abs_shap": float(mean_abs[i]),
                  "rank": int(r + 1)} for r, i in enumerate(order[:15])]
    (tmp / "shap_global.json").write_text(json.dumps({
        "meta": {"method": "shap.TreeExplainer, mean(|SHAP|) over 1836 training rows",
                 "unit": "eV", "model_version": "v3-84feat"},
        "top15": shap_rows}, ensure_ascii=False, indent=1))
    print(f"[写出] shap_global.json top1={shap_rows[0]['feature']} "
          f"({shap_rows[0]['mean_abs_shap']:.4f})")
    top12 = [r["feature"] for r in shap_rows[:12]]
    print(f"[SHAP top-12] {top12}")

    # ---- 8. descriptor_distributions.json（新 top-12，格式同旧版：32 bin）
    labels = {f: feat_label(f) for f in feats}
    dist = {}
    for f in top12:
        col = X[f].astype(float).values
        counts, edges = np.histogram(col, bins=32)
        dist[f] = {
            "label": labels[f],
            "bin_edges": [round(float(e), 6) for e in edges],
            "counts": [int(c) for c in counts],
            "p5": round(float(np.percentile(col, 5)), 5),
            "p25": round(float(np.percentile(col, 25)), 5),
            "p50": round(float(np.percentile(col, 50)), 5),
            "p75": round(float(np.percentile(col, 75)), 5),
            "p95": round(float(np.percentile(col, 95)), 5),
            "min": round(float(col.min()), 5), "max": round(float(col.max()), 5),
            "mean": round(float(col.mean()), 5),
        }
    (tmp / "descriptor_distributions.json").write_text(json.dumps({
        "meta": {"n_rows": 1836, "n_bins": 32,
                 "source": "data_processed/hstar_features_v3_84feat.csv",
                 "ranking": "shap_global.json top15 前 12 个（v3-84feat）",
                 "usage": "描述符分布定位面板：训练分布直方图 + 输入特征值定位"},
        "labels": {f: labels[f] for f in top12},
        "features": dist}, ensure_ascii=False))
    print(f"[写出] descriptor_distributions.json（top-12: {', '.join(top12)}）")

    # ---- 9. descriptor_stats.json（同步 84 特征）
    stats = {f: {"min": float(X[f].min()), "max": float(X[f].max()),
                 "median": float(X[f].median()), "mean": float(X[f].mean())}
             for f in feats}
    (tmp / "descriptor_stats.json").write_text(json.dumps({
        "meta": {"n_rows": 1836, "usage": "输入特征在训练分布中的位置",
                 "model_version": "v3-84feat"},
        "stats": stats}, indent=1))
    print(f"[写出] descriptor_stats.json 84 特征")

    # ---- 10. feature_spec.md
    (tmp / "feature_spec.md").write_text(build_feature_spec(feats))
    print("[写出] feature_spec.md（84 特征规范，含 mode 平局规则）")

    # ---- 11. parity_cases.json（域内 15 + 域外 5）
    rng = random.Random(SEED)
    cases, seen = [], set()

    def add_case(comp):
        fdict, canon, st, fa, known = featurize84(comp, etab_all, m23,
                                                  base_feats, new_cols)
        if canon in seen:
            return False
        seen.add(canon)
        in_domain = known and all(e in train_elements for e in
                                  re.findall(r"[A-Z][a-z]?", canon))
        xrow = pd.DataFrame([fdict])[feats]
        cases.append({
            "comp": comp, "canon": canon,
            "structure": st, "facet": fa, "in_domain": bool(in_domain),
            "features": {k: float(v) for k, v in fdict.items()},
            "py_pred": float(model.predict(xrow)[0]),
        })
        return True

    # 域内 15：训练元素、可推断计量模式（单质 / 1:1 / 1:3）
    patterns = ["single", "11", "13"]
    while sum(c["in_domain"] for c in cases) < 15:
        pat = patterns[len(cases) % 3]
        if pat == "single":
            comp = rng.choice(train_elements)
        else:
            a, b = sorted(rng.sample(train_elements, 2))
            comp = f"{a}{b}" if pat == "11" else (f"{a}{b}3" if rng.random() < 0.5
                                                  else f"{a}3{b}")
        add_case(comp)
    # 域外 5：3 个含 Ge/Sb（元素外推）+ 2 个不可推断计量（2:1 / 三元素）
    ood = []
    while len(ood) < 3:
        a = rng.choice(EXTRA_ELEMENTS)
        b = rng.choice(train_elements)
        pair = sorted([a, b])
        c = f"{pair[0]}{pair[1]}" if rng.random() < 0.5 else f"{pair[0]}{pair[1]}3"
        if c not in seen:
            ood.append(c)
    ood += ["Cu2Pt", "CoNiPt"]  # 2:1 与三元素：结构不可推断
    for c in ood:
        add_case(c)
    assert len(cases) == 20
    (tmp / "parity_cases.json").write_text(json.dumps({
        "meta": {"usage": "JS↔Python 逐位奇偶校验：按 feature_names 顺序把 "
                          "features 组装成向量，经 model.json 推理后与 py_pred 比对",
                 "model_version": "v3-84feat",
                 "n_in_domain": sum(c["in_domain"] for c in cases),
                 "n_out_domain": sum(not c["in_domain"] for c in cases),
                 "mode_tiebreak": "计量数并列时取规范化化学式（ASCII 字母序）首元素"},
        "feature_names": feats,
        "cases": cases}, ensure_ascii=False, indent=1))
    print(f"[写出] parity_cases.json 20 例（域内 "
          f"{sum(c['in_domain'] for c in cases)} + 域外 "
          f"{sum(not c['in_domain'] for c in cases)}）")
    print("[parity 用例] " + ", ".join(f"{c['canon']}({'域内' if c['in_domain'] else '域外'})"
                                      for c in cases))

    # ---- 统一 cp 覆盖（挂载盘 rename 会 Permission denied）
    targets = {
        "model.json": ASSETS, "elements.json": ASSETS, "feature_spec.md": ASSETS,
        "predictions_all.json": ASSETS, "candidates_top.json": ASSETS,
        "shap_global.json": ASSETS, "descriptor_distributions.json": ASSETS,
        "descriptor_stats.json": ASSETS, "parity_cases.json": ASSETS,
        "hstar_features_v3_84feat.csv": DP,
    }
    for name, dstdir in targets.items():
        shutil.copyfile(tmp / name, dstdir / name)
        print(f"[部署] {name} → {dstdir}/ ({(dstdir / name).stat().st_size:,} B)")

    mj = (ASSETS / "model.json").stat().st_size / 1e6
    ej = (ASSETS / "elements.json").stat().st_size / 1e3
    print(f"\n[摘要] OOF MAE={oof_mae:.4f} eV | model.json {mj:.2f} MB | "
          f"elements.json {ej:.1f} KB | parity_cases.json → web/assets/")
    return {"oof_mae": oof_mae, "top5": rk50.head(5)[
        ["rank", "comp", "pred_dG", "true_dG"]].to_dict("records")}


if __name__ == "__main__":
    main()
