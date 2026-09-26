# -*- coding: utf-8 -*-
"""
13_cathub_fetch.py — Catalysis-Hub GraphQL API 数据增援（权威交叉验证 + 元数据增强 + 增量发现）

背景：H* 主线基于 CatBench 镜像（data_raw/MamunHighT2019_adsorption.json.gz），
镜像不含 atoms_json 与 facet/sites 元数据，structure/facet 系从 12 原子超胞计量数推断
（M12→A1(111)、A9B3→L12(111)、A6B6→L10(101)），这是当前主要局限。
本脚本直接查询 Catalysis-Hub 官方库（pubId=MamunHighT2019 即原始出处）：

  任务A 权威交叉验证：Mamun H 子集（products~H* 且 reactants~H2，探测 8,856 条）
        → data_processed/cathub_mamun.csv，与镜像 H*（0.5H2(g) + * -> H*）
        按规范化组成逐能量核对（容差 1e-3 eV），输出匹配率/偏差分布。
  任务B 元数据增强：官方 facet/sites 与 hstar_dataset_v1/v2 推断标签逐组成对比
        → data_processed/hstar_facet_check.csv（复用任务A已拉数据，零额外请求）。
  任务C 增量发现：同一查询的非 Mamun 条目（探测 17,621 条）
        → data_processed/cathub_hstar_extra.csv + 来源盘点（pub/泛函/能量范围）。

请求规划：all_h 一次性拉取（8,856+17,621=26,477 条 @200 行/页 ≈ 133 页）+ publications
≈ 134 请求，A/B/C 共享同一份快照，无重复拉取。

API 纪律（超限封号）：
  - Key 从环境变量 CATHUB_API_KEY 读取（缺失时回退项目根 .env），绝不硬编码；
  - 限速 ≤9 请求/分钟（间隔 6.7 s），日预算硬上限 500 次（state.json 持久计数，跨日自动重置）；
  - 断点续传：每页原始 JSON 存 data_raw/cathub_cache/pages/{name}/，游标存 {name}_state.json，
    重跑时自动跳过已拉页；合并输出前自动去重。
用法：python scripts/13_cathub_fetch.py            # 全部任务（可反复重跑）
"""
import gzip
import json
import os
import re
import sys
import time
from datetime import date
from functools import reduce
from math import gcd
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
DP = ROOT / "data_processed"
CACHE = RAW / "cathub_cache"
CACHE.mkdir(exist_ok=True)
(CACHE / "pages").mkdir(exist_ok=True)

API_URL = "https://api.catalysis-hub.org/graphql"
MIN_INTERVAL = 6.7          # 秒/请求，≤9 req/min（限额 10/min，要求 sleep ≥6.5s）
MAX_DAILY = 500             # 日请求硬上限（全天预算 500 次，超限即收尾）
PAGE_SIZE = 200             # 已探测：单页上限 200 行（first:1000 也只返回 200）
REQ_STATE = CACHE / "req_state.json"

DILUTE_EQ = "0.5H2(g) + * -> H*"   # 稀释 H* 目标反应（与镜像键名一致）
NODE_FIELDS = "id Equation chemicalComposition reactionEnergy facet sites pubId dftCode dftFunctional"
MAMUN = "MamunHighT2019"
TOL = 1e-3                  # 能量核对容差 (eV)

# 已实测：该 API 游标分页顺序不稳定（OFFSET 无 ORDER BY，跨请求行序漂移），
# 单次全量分页会重复 ~23% 行并漏掉相应数量的行（已用 id 字段验证库内内容基本唯一）。
# 对策：同一查询跑 3 个独立 pass（pages/all_h{,_p2,_p3}/），合并后按内容去重取并集，
# 经验覆盖率：1 pass ~77% → 2 pass ~94% → 3 pass ~98-99%（实测见 notes/09）。
PASS_DIRS = ["all_h", "all_h_p2", "all_h_p3"]


# ----------------------------------------------------------------------------- API 基础设施
def load_api_key():
    """Key 一律从环境变量读取；本地运行时回退 .env（.env 不入库、分发前删除）。"""
    key = os.environ.get("CATHUB_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("CATHUB_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("[致命] 未找到 CATHUB_API_KEY（环境变量或 .env），无法访问 Catalysis-Hub API")


class RateLimiter:
    """限速 + 日预算持久计数（跨进程/跨重跑安全）。"""

    def __init__(self):
        self.last_ts = 0.0
        st = json.loads(REQ_STATE.read_text()) if REQ_STATE.exists() else {}
        today = str(date.today())
        self.count = st.get("count", 0) if st.get("date") == today else 0
        print(f"[预算] 今日已用请求 {self.count}/{MAX_DAILY}", flush=True)

    def wait(self):
        if self.count >= MAX_DAILY:
            sys.exit(f"[收尾] 已达日请求上限 {MAX_DAILY}，已拉页均已存盘，可改日重跑续传")
        dt = time.time() - self.last_ts
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)

    def mark(self):
        self.count += 1
        self.last_ts = time.time()
        REQ_STATE.write_text(json.dumps({"date": str(date.today()), "count": self.count}))


LIM = RateLimiter()
API_KEY = None   # 惰性加载：analyze 模式命中本地缓存时无需 Key，仅在真正发请求前读取


def gql(query, variables=None, retries=5):
    """发一次 GraphQL 请求；失败指数退避重试。返回 data dict。"""
    global API_KEY
    if API_KEY is None:
        API_KEY = load_api_key()
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    body = json.dumps(payload).encode()
    for i in range(retries):
        LIM.wait()
        try:
            req = Request(API_URL, data=body, headers={
                "Content-Type": "application/json", "X-API-Key": API_KEY})
            with urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            LIM.mark()
            if "errors" in out:
                raise RuntimeError(f"GraphQL errors: {out['errors']}")
            return out["data"]
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            LIM.mark()
            wait = 20 * (i + 1)
            print(f"[重试 {i+1}/{retries}] {e}；{wait}s 后重试", flush=True)
            time.sleep(wait)
    raise RuntimeError("请求多次失败，已拉页均存盘，可重跑续传")


# ----------------------------------------------------------------------------- 分页拉取（断点续传）
def fetch_all(name, where):
    """按游标分页拉取 reactions，页存 cache/pages/{name}/，游标存 cache/{name}_state.json。
    where: reactions() 的过滤子句，如 'products: "Hstar", reactants: "H2"'。
    返回去重后的 DataFrame。"""
    pdir = CACHE / "pages" / name
    pdir.mkdir(exist_ok=True)
    sfile = CACHE / f"{name}_state.json"
    state = json.loads(sfile.read_text()) if sfile.exists() else {"after": None, "idx": 0}
    total = None
    while True:
        page_file = pdir / f"{state['idx']:04d}.json"
        if page_file.exists():  # 已存盘页跳过（游标推进与写页同事务）
            state["idx"] += 1
            continue
        data = gql(
            "query($after: String) { reactions(first: %d, after: $after, %s) {"
            " totalCount pageInfo { endCursor hasNextPage }"
            " edges { node { %s } } } }" % (PAGE_SIZE, where, NODE_FIELDS),
            variables={"after": state["after"]})
        r = data["reactions"]
        total = r["totalCount"]
        page_file.write_text(json.dumps([e["node"] for e in r["edges"]]))
        state["idx"] += 1
        state["after"] = r["pageInfo"]["endCursor"]
        sfile.write_text(json.dumps(state))
        print(f"[{name}] 页 {state['idx']} 存盘（累计约 {min(state['idx']*PAGE_SIZE, total)}/{total}，"
              f"今日请求 {LIM.count}/{MAX_DAILY}）", flush=True)
        if not r["pageInfo"]["hasNextPage"]:
            break
    nodes = []
    for f in sorted(pdir.glob("*.json")):
        nodes.extend(json.loads(f.read_text()))
    df = pd.DataFrame(nodes)
    # sites 可能为 dict，先序列化再去重，再还原
    if "sites" in df.columns:
        df["sites"] = df["sites"].map(lambda s: json.dumps(s, sort_keys=True)
                                      if isinstance(s, (dict, list)) else s)
    df = df.drop_duplicates()
    print(f"[{name}] 合并 {len(nodes)} 行 → 去重后 {len(df)} 行（totalCount={total}）", flush=True)
    return df


def load_pass(name):
    """读取单个 pass 的全部缓存页 → DataFrame（未去重）。"""
    nodes = []
    for f in sorted((CACHE / "pages" / name).glob("*.json")):
        nodes.extend(json.loads(f.read_text()))
    return pd.DataFrame(nodes)


def merge_passes():
    """合并 PASS_DIRS 中所有已存在的 pass，按内容去重（库内内容唯一，id 仅作参考）。
    打印各 pass 增量贡献以评估覆盖率。"""
    df = None
    for name in PASS_DIRS:
        if not (CACHE / "pages" / name).exists():
            continue
        d = load_pass(name)
        if d.empty:
            continue
        d["sites"] = d["sites"].map(lambda s: json.dumps(s, sort_keys=True)
                                    if isinstance(s, (dict, list)) else s)
        content_cols = [c for c in d.columns if c != "id"]
        before = 0 if df is None else len(df)
        df = d.drop_duplicates(subset=content_cols) if df is None \
            else pd.concat([df, d]).drop_duplicates(subset=content_cols)
        print(f"[merge] +{name}: 原始 {len(d)} 行 → 并集 {before}→{len(df)} "
              f"（新增 {len(df)-before}）", flush=True)
    return df


def fetch_passes(names, where):
    for n in names:
        if (CACHE / f"{n}_state.json").exists() or (CACHE / "pages" / n).exists():
            print(f"[{n}] 缓存已存在，跳过拉取", flush=True)
            continue
        fetch_all(n, where)


# ----------------------------------------------------------------------------- 组成规范化（与 05 口径一致）
def canon_comp(formula):
    """'Co9Sn3' → 元素按字母排序 + 计量数约简 → 'Co3Sn'；'Ag' → 'Ag'；无法解析返回 None。"""
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", formula or "")
    if not toks or "".join(a + b for a, b in toks) != formula:
        return None
    d = {}
    for el, n in toks:
        d[el] = d.get(el, 0) + int(n or 1)
    g = reduce(gcd, d.values())
    return "".join(f"{el}{d[el]//g if d[el]//g > 1 else ''}" for el in sorted(d))


def norm_facet(f):
    """'111'/'(111)'/111 → '111'；空 → None。"""
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return None
    s = re.sub(r"[()\s]", "", str(f).strip())
    return s or None


def parse_site_H(s):
    """sites 字段（JSON 字符串或 dict）→ H 的吸附位点；失败返回 None。"""
    try:
        d = json.loads(s) if isinstance(s, str) else s
        if isinstance(d, dict):
            return d.get("H")
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------- 任务A：Mamun 官方子集 + 交叉核对
def task_a(allh):
    print("\n===== 任务A：MamunHighT2019 官方 H 子集与镜像交叉核对 =====", flush=True)
    hub = allh[allh["pubId"] == MAMUN].copy()
    hub.to_csv(DP / "cathub_mamun.csv", index=False)
    print(f"[A] data_processed/cathub_mamun.csv ← {len(hub)} 行（products~H* 且 reactants~H2）", flush=True)
    print("[A] 方程分布 Top10:", flush=True)
    print(hub["Equation"].value_counts().head(10).to_string(), flush=True)

    dil = hub[hub["Equation"] == DILUTE_EQ].copy()
    dil["comp"] = dil["chemicalComposition"].map(canon_comp)
    print(f"[A] 其中稀释 H*（{DILUTE_EQ}）{len(dil)} 行，唯一组成 {dil['comp'].nunique()} 个", flush=True)

    # 镜像 H* 子集
    with gzip.open(RAW / "MamunHighT2019_adsorption.json.gz", "rt") as f:
        mirror = json.load(f)
    rows = []
    for k, v in mirror.items():
        if "-> H*" not in k or "0.5H2" not in k:
            continue
        prefix = k.split("_", 1)[0]
        rows.append({"key": k, "comp": canon_comp(prefix), "E_mirror": v["ref_ads_eng"]})
    mir = pd.DataFrame(rows)
    print(f"[A] 镜像 H* 构型 {len(mir)} 条，唯一组成 {mir['comp'].nunique()} 个", flush=True)

    # 按规范化组成匹配，组内能量多重集核对（消耗式一对一）
    hub_g = dil.dropna(subset=["comp"]).groupby("comp")["reactionEnergy"].apply(list)
    mir_g = mir.dropna(subset=["comp"]).groupby("comp")["E_mirror"].apply(list)
    common = sorted(set(hub_g.index) & set(mir_g.index))
    only_mir = sorted(set(mir_g.index) - set(hub_g.index))
    only_hub = sorted(set(hub_g.index) - set(mir_g.index))
    n_match = n_total = 0
    diffs = []
    for c in common:
        rem = sorted(hub_g[c])          # 官方能量池（消耗式匹配）
        for e in mir_g[c]:
            n_total += 1
            if rem:
                j = int(np.argmin([abs(e - x) for x in rem]))
                d = abs(e - rem[j])
                if d < TOL:
                    n_match += 1
                    diffs.append(d)
                    rem.pop(j)
    diffs = np.array(diffs)
    n_comp_mir = mir["comp"].nunique()
    print(f"[A] 组成匹配率：{len(common)}/{n_comp_mir} "
          f"（{len(common)/n_comp_mir:.1%}）；仅镜像 {len(only_mir)}、仅官方 {len(only_hub)}", flush=True)
    if only_mir:
        print(f"[A] 仅镜像组成示例: {only_mir[:10]}", flush=True)
    if only_hub:
        print(f"[A] 仅官方组成示例: {only_hub[:10]}", flush=True)
    print(f"[A] 能量逐条匹配（容差 {TOL} eV）：{n_match}/{n_total} "
          f"（{n_match/max(n_total,1):.2%}）", flush=True)
    if len(diffs):
        qs = np.percentile(diffs, [50, 90, 99])
        print(f"[A] 匹配能量偏差分布：max={diffs.max():.2e} eV, mean={diffs.mean():.2e} eV, "
              f"p50={qs[0]:.2e}, p90={qs[1]:.2e}, p99={qs[2]:.2e}", flush=True)

    # 与 hstar_dataset_v1（去重后每表面最低能）核对
    v1 = pd.read_csv(DP / "hstar_dataset_v1.csv")
    hub_min = dil.dropna(subset=["comp"]).groupby("comp")["reactionEnergy"].min()
    v1m = v1.set_index("comp")["energy_eV"]
    both = v1m.index.intersection(hub_min.index)
    dv = (v1m[both] - hub_min[both]).abs()
    print(f"[A] 去重后表面级核对（v1 最低能 vs 官方最低能）：{len(both)}/{len(v1)} 表面，"
          f"|Δ| max={dv.max():.2e} eV, mean={dv.mean():.2e} eV, {(dv < TOL).mean():.2%} 在容差内", flush=True)
    return dil   # 稀释 H* 官方子集，供任务B


# ----------------------------------------------------------------------------- 任务B：facet 校验（零额外请求）
def task_b(dil):
    print("\n===== 任务B：官方 facet/sites vs v1/v2 推断标签 =====", flush=True)
    v1 = pd.read_csv(DP / "hstar_dataset_v1.csv")
    v2 = pd.read_csv(DP / "hstar_dataset_v2.csv")
    same_comp = set(v1["comp"]) == set(v2["comp"])
    print(f"[B] v1 {len(v1)} 表面 / v2 {len(v2)} 表面，组成集合一致：{same_comp}", flush=True)

    d = dil.dropna(subset=["comp"]).copy()
    d["facet_official"] = d["facet"].map(norm_facet)
    d["site_H"] = d["sites"].map(parse_site_H)

    recs = []
    for _, r in v1.iterrows():
        g = d[d["comp"] == r["comp"]]
        facets = sorted(g["facet_official"].dropna().unique())
        sites = sorted(g["site_H"].dropna().unique())
        f_inf = norm_facet(r["facet"])
        recs.append({
            "comp": r["comp"], "structure_inferred": r["structure"],
            "facet_inferred": f_inf, "n_configs_official": len(g),
            "facet_official": "|".join(facets) if facets else None,
            "site_H_official": "|".join(sites) if sites else None,
            "facet_match": (len(facets) == 1 and facets[0] == f_inf) if facets else None,
        })
    chk = pd.DataFrame(recs)
    chk.to_csv(DP / "hstar_facet_check.csv", index=False)
    ok = chk["facet_match"] == True
    bad = chk["facet_match"] == False
    nodata = chk["facet_match"].isna()
    print(f"[B] hstar_facet_check.csv ← {len(chk)} 表面：facet 一致 {ok.sum()} "
          f"（{ok.mean():.1%}），不一致 {bad.sum()}，官方缺 facet {nodata.sum()}", flush=True)
    if bad.sum():
        print("[B] 不一致组成（推断 facet vs 官方 facet）:", flush=True)
        print(chk[bad][["comp", "structure_inferred", "facet_inferred",
                        "facet_official"]].to_string(index=False), flush=True)
    cov_f = chk["facet_official"].notna().mean()
    cov_s = chk["site_H_official"].notna().mean()
    print(f"[B] 官方 facet 覆盖率 {cov_f:.1%}；官方 H 吸附位点（sites）覆盖率 {cov_s:.1%}", flush=True)
    print("[B] 官方 facet 分布:", chk["facet_official"].value_counts().head(10).to_dict(), flush=True)
    print("[B] H 位点类型分布:", d["site_H"].value_counts().head(10).to_dict(), flush=True)
    return chk


# ----------------------------------------------------------------------------- 任务C：非 Mamun 增量 H* 数据 + 来源盘点
def task_c(allh):
    print("\n===== 任务C：非 Mamun H 吸附增量盘点 =====", flush=True)
    # publications 列表（题名/年份，用于来源盘点；缓存到本地避免重复花费请求）
    pcache = CACHE / "publications.json"
    if pcache.exists():
        pubmap = json.loads(pcache.read_text())
        ptot = len(pubmap)
    else:
        pubmap = {}
        after = None
        while True:
            pubs = gql("query($after: String) { publications(first: 200, after: $after) {"
                       " totalCount pageInfo { endCursor hasNextPage }"
                       " edges { node { pubId title year authors journal } } } }",
                       variables={"after": after})
            p = pubs["publications"]
            ptot = p["totalCount"]
            for e in p["edges"]:
                pubmap[e["node"]["pubId"]] = e["node"]
            if not p["pageInfo"]["hasNextPage"]:
                break
            after = p["pageInfo"]["endCursor"]
        pcache.write_text(json.dumps(pubmap))
    print(f"[C] 库内 publication 总数：{ptot}（已取 {len(pubmap)}）", flush=True)

    extra = allh[allh["pubId"] != MAMUN].copy()
    extra["comp"] = extra["chemicalComposition"].map(canon_comp)
    print(f"[C] 全库 H 相关子集 {len(allh)} 行；非 Mamun {len(extra)} 行，"
          f"唯一组成 {extra['comp'].nunique()} 个", flush=True)
    print("[C] 方程分布 Top10:", flush=True)
    print(extra["Equation"].value_counts().head(10).to_string(), flush=True)

    out = extra.rename(columns={"chemicalComposition": "comp_raw",
                                "reactionEnergy": "energy_eV", "Equation": "equation"})
    out = out[["comp", "comp_raw", "facet", "sites", "equation", "energy_eV",
               "pubId", "dftCode", "dftFunctional"]]
    out.to_csv(DP / "cathub_hstar_extra.csv", index=False)
    print(f"[C] data_processed/cathub_hstar_extra.csv ← {len(out)} 行", flush=True)

    # 来源盘点（后续是否合并重训的决策依据）
    print("[C] 来源 pub 分布（全部）:", flush=True)
    by_pub = extra.groupby("pubId").agg(
        n=("reactionEnergy", "size"),
        n_comp=("comp", "nunique"),
        E_min=("reactionEnergy", "min"),
        E_max=("reactionEnergy", "max"),
        E_mean=("reactionEnergy", "mean"),
    ).sort_values("n", ascending=False)
    for pid, r in by_pub.iterrows():
        p = pubmap.get(pid, {})
        print(f"    {pid:32s} {int(r['n']):6d} 行 {int(r['n_comp']):5d} 组成 "
              f"E∈[{r['E_min']:7.3f},{r['E_max']:7.3f}]  ({p.get('year','?')}) "
              f"{str(p.get('title'))[:60]}", flush=True)
    by_pub_out = by_pub.reset_index()
    by_pub_out["title"] = by_pub_out["pubId"].map(lambda x: pubmap.get(x, {}).get("title"))
    by_pub_out["year"] = by_pub_out["pubId"].map(lambda x: pubmap.get(x, {}).get("year"))
    by_pub_out.to_csv(DP / "cathub_hstar_extra_by_pub.csv", index=False)
    print(f"[C] data_processed/cathub_hstar_extra_by_pub.csv ← {len(by_pub_out)} 个来源", flush=True)

    e = extra["reactionEnergy"]
    print(f"[C] 能量范围：[{e.min():.3f}, {e.max():.3f}] eV，mean={e.mean():.3f}，"
          f"std={e.std():.3f}；|E|>3 eV 离群 {int((e.abs()>3).sum())} 行", flush=True)
    print(f"[C] 泛函分布: {extra['dftFunctional'].value_counts().to_dict()}", flush=True)
    print(f"[C] DFT 代码分布: {extra['dftCode'].value_counts().to_dict()}", flush=True)
    print(f"[C] 晶面分布 Top10: {extra['facet'].astype(str).value_counts().head(10).to_dict()}",
          flush=True)
    return extra


WHERE_ALL_H = 'products: "Hstar", reactants: "H2"'


def main():
    t0 = time.time()
    args = sys.argv[1:]
    if args and args[0] in ("p2", "p3"):          # 仅拉取指定 pass
        fetch_passes([f"all_h_{args[0]}"], WHERE_ALL_H)
        return
    if args and args[0] == "fetch":               # 拉齐全部 3 个 pass
        fetch_passes(PASS_DIRS, WHERE_ALL_H)
        return
    # analyze（默认）：合并所有已存在 pass → 任务A/B/C
    allh = merge_passes(WHERE_ALL_H)
    print(f"[merge] 合并后总量 {len(allh)} 行（totalCount 应为 26,477；"
          f"覆盖率 {len(allh)/26477:.1%}）", flush=True)
    dil = task_a(allh)
    task_b(dil)
    task_c(allh)
    print(f"\n[完成] 总耗时 {(time.time()-t0)/60:.1f} min，今日累计请求 {LIM.count}/{MAX_DAILY}",
          flush=True)


if __name__ == "__main__":
    main()
