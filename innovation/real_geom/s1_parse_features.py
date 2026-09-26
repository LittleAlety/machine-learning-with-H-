# -*- coding: utf-8 -*-
"""
innovation/real_geom/s1_parse_features.py — S4 真实 DFT 弛豫几何位点级特征

数据源（分支 α）：data_raw/MamunHighT2019_adsorption.json.gz 顶层 _structures 块，
ASE-db 风格弛豫终态结构（cell/numbers/positions/constraints），逐 H* 构型取
Hstar（吸附态）与 star（洁净表面，弛豫参照）。

特征清单（18 维，与 Phase5 route2 预注册清单同构，禁新增族）：
  位点锚定（H 吸附位点）：
    site_cn            H 位点配位数（金属近邻数，共价/原子半径和×1.2 截断，同 r2 口径）
    site_gcn           H 位点广义配位数 Σ CN_j/12
    h_height_A         H 高出顶层金属层高度（z_H − max z_metal，Å）
  近邻集合统计（H 的金属近邻 N(H)，同构映射 surf_* → nn-of-site）：
    nn_cn_mean/min/std    近邻金属 CN 分布
    nn_gcn_mean/min/std   近邻金属 GCN 分布
    nn_en_mean/std        近邻加权电负性（Pauling）
    nn_d_mean/std         近邻加权 d 电子数（基态组态）
  层距/弛豫位移（真实弛豫几何；参照为同 slab 洁净态 star，非理想原型——如实记口径差）：
    d12_relaxed_A      H* 态顶-次金属层间距（Å）
    d12_relax_pct      吸附诱导层距变化 = (d12_Hstar − d12_star)/d12_star ×100
    surf_disp_mean_A / surf_disp_max_A   洁净→吸附态表面金属位移统计
    surf_rumpling_A    H* 态表面层 z 标准差
  与 Phase5 的 16 维相比：relax_energy_drop_per_atom 不可用（db 无弛豫前后能量），
  且其属能量族（与标签同源风险），替换为预注册族内的 nn_gcn_min/nn_gcn_std 补齐 18 维。

幂等：输出 real_geom_site_features.csv（record 级）+ parse_failures.csv（如有）。
"""
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW_GZ = ROOT / "data_raw" / "MamunHighT2019_adsorption.json.gz"
F0_JSON = ROOT / "innovation" / "site_aware" / "f0_site_lockbox.json"

from pymatgen.core import Element  # noqa: E402

COV_SCALE = 1.2
CN_MAX_REF = 12.0
DILUTE_EQ = "0.5H2(g) + * -> H*"
STRUCT_RULE = {(12,): ("A1", 111), (3, 9): ("L12", 111), (6, 6): ("L10", 101)}

FEAT_COLS = ["site_cn", "site_gcn", "h_height_A",
             "nn_cn_mean", "nn_cn_min", "nn_cn_std",
             "nn_gcn_mean", "nn_gcn_min", "nn_gcn_std",
             "nn_en_mean", "nn_en_std", "nn_d_mean", "nn_d_std",
             "d12_relaxed_A", "d12_relax_pct",
             "surf_disp_mean_A", "surf_disp_max_A", "surf_rumpling_A"]

_d_cache, _en_cache, _rad_cache = {}, {}, {}


def d_electrons(sym):
    if sym not in _d_cache:
        el = Element(sym)
        _d_cache[sym] = float(sum(o[2] for o in el.full_electronic_structure
                                  if o[1] == "d"))
    return _d_cache[sym]


def en_of(sym):
    if sym not in _en_cache:
        x = Element(sym).X
        _en_cache[sym] = float(x) if x is not None else np.nan
    return _en_cache[sym]


def rad_of(sym):
    """共价半径（预注册口径"共价半径和×1.2"；r2 代码误用 atomic_radius，
    对金属两者近似但对 H(0.25 vs 0.31) 会系统性漏掉 bridge/hollow 近邻，
    此处按 docstring 本义取 covalent_radius，如实记录口径修正）。"""
    if sym not in _rad_cache:
        # 金属: pymatgen atomic_radius（与 Phase5 r2 代码同口径，数值≈共价半径）;
        # H: atomic_radius=0.25 会系统性漏掉 bridge/hollow 近邻（实测 312 例 H-M
        # 距离 1.8-2.1 Å 被截断误排），取共价半径 0.31（Cordero）——如实记录修正。
        if sym == "H":
            _rad_cache[sym] = 0.31
        else:
            r = Element(sym).atomic_radius
            _rad_cache[sym] = float(r) if r is not None else 1.3
    return _rad_cache[sym]


def _arr(field):
    nd = field["__ndarray__"]
    return np.array(nd[2], dtype=float).reshape(nd[0])


def load_structure(s_structs, uid):
    row = s_structs[uid]
    if isinstance(row, str):
        row = json.loads(row)
    r = row[str(row["ids"][0])]
    cell = _arr(r["cell"]["array"] if "array" in r["cell"] else r["cell"])
    pos = _arr(r["positions"])
    nums = np.array(r["numbers"]["__ndarray__"][2], dtype=int)
    return cell, pos, nums


def neighbor_table(cell, pos, nums):
    syms = [Element.from_Z(int(z)).symbol for z in nums]
    radii = np.array([rad_of(s) for s in syms])
    dxyz = pos[:, None, :] - pos[None, :, :]
    dfrac = np.linalg.solve(cell.T, dxyz.reshape(-1, 3).T).T.reshape(dxyz.shape)
    dfrac -= np.round(dfrac)          # 最小像（pbc=True,True,True；真空层 >> 截断）
    dist = np.linalg.norm(dfrac @ cell, axis=-1)
    cut = COV_SCALE * (radii[:, None] + radii[None, :])
    np.fill_diagonal(dist, np.inf)
    return dist < cut, dist, syms


def layers_z(z, tol=0.5):
    order = np.argsort(z)[::-1]
    layers, cur = [], [z[order[0]]]
    for idx in order[1:]:
        if cur[-1] - z[idx] < tol:
            cur.append(z[idx])
        else:
            layers.append(float(np.mean(cur)))
            cur = [z[idx]]
    layers.append(float(np.mean(cur)))
    return layers


def align_star(cell, pos_h, nums_h, pos0, nums0):
    """star → Hstar 金属原子配对准（同 species 内 Hungarian，最小像距离）。
    db 中 star 与 Hstar 原子序不保证一致（实测 3218/7048 乱序），
    位移类特征必须先配对。返回 star 金属坐标按 Hstar 金属序重排后的数组。"""
    from scipy.optimize import linear_sum_assignment
    out = np.empty_like(pos0)
    for z in np.unique(nums_h):
        ih = np.where(nums_h == z)[0]
        i0 = np.where(nums0 == z)[0]
        if len(ih) != len(i0):
            raise ValueError(f"species Z={z} count mismatch")
        d = pos_h[ih][:, None, :] - pos0[i0][None, :, :]
        dfrac = np.linalg.solve(cell.T, d.reshape(-1, 3).T).T.reshape(d.shape)
        dfrac -= np.round(dfrac)
        cost = np.linalg.norm(dfrac @ cell, axis=-1)
        r, c = linear_sum_assignment(cost)
        out[ih[r]] = pos0[i0[c]]
    return out


def compute_record(s_structs, refs):
    cell, pos, nums = load_structure(s_structs, refs["Hstar"])
    cell0, pos0, nums0 = load_structure(s_structs, refs["star"])
    hidx = np.where(nums == 1)[0]
    if len(hidx) != 1:
        raise ValueError(f"n_H={len(hidx)}")
    ih = int(hidx[0])
    met = nums > 1
    if sorted(nums[met].tolist()) != sorted(nums0.tolist()):
        raise ValueError("metal composition mismatch star/Hstar")
    pos0 = align_star(cell, pos[met], nums[met], pos0, nums0)

    nn, dist, syms = neighbor_table(cell, pos, nums)
    cn = nn.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        gcn = np.where(cn > 0, (nn @ cn) / CN_MAX_REF, 0.0)
    en = np.array([en_of(s) for s in syms])
    de = np.array([d_electrons(s) for s in syms])

    nbr = nn[ih] & met
    if nbr.sum() == 0:                       # 截断内无金属近邻 → 如实记失败
        raise ValueError("H has 0 metal neighbors within cutoff")
    zm = pos[met, 2]
    surf_m = zm > np.quantile(zm, 0.75)

    lz_h = layers_z(zm)
    lz_s = layers_z(pos0[:, 2])
    d12_h = lz_h[0] - lz_h[1] if len(lz_h) > 1 else np.nan
    d12_s = lz_s[0] - lz_s[1] if len(lz_s) > 1 else np.nan
    disp = np.linalg.norm(pos[met] - pos0, axis=1)   # 同 cell、原子序一致

    return {
        "site_cn": float(nbr.sum()),
        "site_gcn": float(gcn[ih]),
        "h_height_A": float(pos[ih, 2] - zm.max()),
        "nn_cn_mean": float(cn[nbr].mean()),
        "nn_cn_min": float(cn[nbr].min()),
        "nn_cn_std": float(cn[nbr].std()),
        "nn_gcn_mean": float(gcn[nbr].mean()),
        "nn_gcn_min": float(gcn[nbr].min()),
        "nn_gcn_std": float(gcn[nbr].std()),
        "nn_en_mean": float(np.nanmean(en[nbr])),
        "nn_en_std": float(np.nanstd(en[nbr])),
        "nn_d_mean": float(np.nanmean(de[nbr])),
        "nn_d_std": float(np.nanstd(de[nbr])),
        "d12_relaxed_A": float(d12_h),
        "d12_relax_pct": float((d12_h - d12_s) / d12_s * 100) if d12_s else np.nan,
        "surf_disp_mean_A": float(disp[surf_m].mean()),
        "surf_disp_max_A": float(disp[surf_m].max()),
        "surf_rumpling_A": float(zm[surf_m].std()),
    }


def parse_prefix(p):
    from functools import reduce
    from math import gcd
    cd = {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", p)}
    g = reduce(gcd, cd.values())
    red = {el: n // g for el, n in cd.items()}
    comp = "".join(f"{el}{red[el] if red[el] > 1 else ''}" for el in sorted(red))
    return comp, cd


def main():
    t0 = time.time()
    data = json.loads(gzip.decompress(RAW_GZ.read_bytes()))
    s = data["_structures"]
    lb = set(json.loads(F0_JSON.read_text())["lockbox_comps"])

    keys = [k for k in data if k != "_structures" and DILUTE_EQ in k]
    rows, fails = [], []
    for k in sorted(keys):
        v = data[k]
        prefix = re.match(r"^([A-Za-z0-9]+)_", k).group(1)
        comp, cd = parse_prefix(prefix)
        structure, facet = STRUCT_RULE[tuple(sorted(cd.values()))]
        try:
            feats = compute_record(s, {sp: r["ref"] for sp, r in v["raw"].items()})
            rows.append({"record_id": k, "comp": comp, "structure": structure,
                         "facet": facet, "energy_eV": float(v["ref_ads_eng"]),
                         "in_site_lockbox": comp in lb, **feats})
        except Exception as e:
            fails.append({"record_id": k, "comp": comp, "error": str(e)[:300]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_geom_site_features.csv", index=False)
    if fails:
        pd.DataFrame(fails).to_csv(OUT / "parse_failures.csv", index=False)

    meta = {"n_hstar_records": len(keys), "n_parsed": len(df),
            "n_failed": len(fails),
            "parse_success_rate": len(df) / max(len(keys), 1),
            "n_comps_parsed": int(df["comp"].nunique()),
            "feature_cols": FEAT_COLS, "n_features": len(FEAT_COLS),
            "site_cn_dist": df["site_cn"].value_counts().sort_index().to_dict(),
            "runtime_min": (time.time() - t0) / 60,
            "sha256_features_csv": hashlib.sha256(
                (OUT / "real_geom_site_features.csv").read_bytes()).hexdigest()}
    (OUT / "s1_parse_report.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(df[FEAT_COLS].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    sys.exit(main())
