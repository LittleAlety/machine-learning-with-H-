# -*- coding: utf-8 -*-
"""
innovation/continuous_local/r2_features.py — 弛豫后连续局域描述符 (预注册清单, 禁事后挑)

预注册统计清单 (config/preregistered_phase5.yaml route2_foundation_relax + 任务书):
  表面层定义沿用 chgnet_features 工作流 v2: z > quantile(z, 0.75)
  配位数 CN (共价半径和 ×1.2 截断):
    surf_cn_mean / surf_cn_min / surf_cn_std        表面层 CN 分布
  GCN (广义配位数: GCN_i = sum_{j in nn(i)} CN_j / CN_max, CN_max=12 体相 fcc 参考):
    surf_gcn_mean / surf_gcn_std / surf_gcn_min
  A/B 近邻加权性质 (对表面位点 i, 近邻组成加权的均值; 再对表面取统计):
    surf_nn_en_mean / surf_nn_en_std                近邻加权电负性 (Pauling)
    surf_nn_d_mean  / surf_nn_d_std                 近邻加权 d 电子数 (基态电子组态)
  层距/弛豫位移:
    d12_relaxed_A      弛豫后顶-次层间距 (Å)
    d12_relax_pct      相对未弛豫原型的层间距变化 (%)
    surf_disp_mean_A / surf_disp_max_A              表面原子位移统计
    surf_rumpling_A    表面层 z 标准差 (起伏)
    relax_energy_drop_per_atom  弛豫能降/原子 (eV)
幂等: 输出 features csv; 只处理 relaxed_subset 中已成功弛豫且无 fail 的条目。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLAB_DIR = ROOT / "innovation" / "chgnet_features" / "slabs"
RELAX_SUB = OUT / "relaxed_subset"

from pymatgen.core import Element, Structure

COV_SCALE = 1.2
CN_MAX_REF = 12.0  # fcc 体相配位参考 (GCN 标准定义)

_d_cache = {}


def d_electrons(el: Element) -> float:
    """基态电子组态中 d 轨道占据数 (如 Fe [Ar]3d6 4s2 -> 6)。"""
    if el.symbol not in _d_cache:
        n = 0.0
        for orb in el.full_electronic_structure:
            if orb[1] == "d":
                n += orb[2]
        _d_cache[el.symbol] = n
    return _d_cache[el.symbol]


def neighbor_table(struct: Structure):
    """共价半径和 × COV_SCALE 截断的邻居表 (周期性)。"""
    from pymatgen.core.periodic_table import Element as E
    species = struct.species
    radii = np.array([E(sp.symbol).atomic_radius or 1.3 for sp in species])
    # pymatgen get_all_neighbors: 用最大可能截断再逐对过滤代价高;
    # 直接距离矩阵 (128 原子, 周期性最小像)
    frac = struct.frac_coords
    lat = struct.lattice.matrix
    dfrac = frac[:, None, :] - frac[None, :, :]
    dfrac -= np.round(dfrac)
    dist = np.linalg.norm(dfrac @ lat, axis=-1)
    cut = COV_SCALE * (radii[:, None] + radii[None, :])
    np.fill_diagonal(dist, np.inf)
    nn = dist < cut
    return nn, dist


def layers_z(z: np.ndarray, tol: float = 0.5):
    """按 z 聚类成层 (tol Å), 返回从高到低的层均值列表。"""
    order = np.argsort(z)[::-1]
    layers, cur = [], [z[order[0]]]
    for idx in order[1:]:
        if cur[-1] - z[idx] < tol:
            cur.append(z[idx])
        else:
            layers.append(np.mean(cur))
            cur = [z[idx]]
    layers.append(np.mean(cur))
    return layers


def compute_features(relax_json: Path):
    rec = json.loads(relax_json.read_text())
    name = rec["name"]
    fs = Structure.from_dict(rec["final_structure"])
    init = Structure.from_file(SLAB_DIR / f"{name}.cif")

    z = fs.cart_coords[:, 2]
    surf = z > np.quantile(z, 0.75)

    nn, dist = neighbor_table(fs)
    cn = nn.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        gcn = np.where(cn > 0, (nn @ cn) / CN_MAX_REF, 0.0)

    els = fs.species
    en = np.array([Element(e.symbol).X if Element(e.symbol).X is not None else np.nan for e in els])
    de = np.array([d_electrons(Element(e.symbol)) for e in els])

    def nn_weighted(prop):
        out = np.zeros(len(prop))
        for i in range(len(prop)):
            w = nn[i]
            out[i] = np.nanmean(prop[w]) if w.any() else np.nan
        return out

    nn_en, nn_de = nn_weighted(en), nn_weighted(de)

    lz_r = layers_z(z)
    lz_i = layers_z(init.cart_coords[:, 2])
    d12_r = lz_r[0] - lz_r[1] if len(lz_r) > 1 else np.nan
    d12_i = lz_i[0] - lz_i[1] if len(lz_i) > 1 else np.nan

    disp = np.linalg.norm(fs.cart_coords - init.cart_coords, axis=1)  # 原子序保持

    return {
        "name": name,
        "n_atoms": rec["n_atoms"],
        "relax_n_steps": rec["n_steps"],
        "relax_converged": rec["converged"],
        "relax_fmax_final": rec["fmax_final"],
        "relax_energy_drop_per_atom": (rec["e_after"] - rec["e_before"]) / rec["n_atoms"],
        "surf_cn_mean": float(cn[surf].mean()),
        "surf_cn_min": float(cn[surf].min()),
        "surf_cn_std": float(cn[surf].std()),
        "surf_gcn_mean": float(gcn[surf].mean()),
        "surf_gcn_std": float(gcn[surf].std()),
        "surf_gcn_min": float(gcn[surf].min()),
        "surf_nn_en_mean": float(np.nanmean(nn_en[surf])),
        "surf_nn_en_std": float(np.nanstd(nn_en[surf])),
        "surf_nn_d_mean": float(np.nanmean(nn_de[surf])),
        "surf_nn_d_std": float(np.nanstd(nn_de[surf])),
        "d12_relaxed_A": float(d12_r),
        "d12_relax_pct": float((d12_r - d12_i) / d12_i * 100) if d12_i else np.nan,
        "surf_disp_mean_A": float(disp[surf].mean()),
        "surf_disp_max_A": float(disp[surf].max()),
        "surf_rumpling_A": float(z[surf].std()),
    }


def main():
    rows, skipped = [], []
    for fj in sorted(RELAX_SUB.glob("*.json")):
        if fj.name.startswith("_") or fj.name.endswith(".fail.json"):
            continue
        try:
            rows.append(compute_features(fj))
        except Exception as e:  # 如实记录失败, 不伪造
            skipped.append({"name": fj.stem, "error": str(e)[:300]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no relaxed structures yet; nothing to do")
        return
    # 与 v3 表对齐的 key: name = comp_structure_facet
    parts = df["name"].str.rsplit("_", n=2, expand=True)
    df["comp"], df["structure"], df["facet"] = parts[0], parts[1], parts[2]
    df.to_csv(OUT / "continuous_local_features.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT / "feature_failures.csv", index=False)
    print(f"features: {len(df)} rows, {len(df.columns)} cols; failures: {len(skipped)}")
    print(df.describe().T[["mean", "std", "min", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
