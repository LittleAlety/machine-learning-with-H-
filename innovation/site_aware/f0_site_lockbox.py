# -*- coding: utf-8 -*-
"""F0 位点级 lockbox 冻结 — plan-hb v3.0 Workstream F（红线 8：开发前冻结）

选择（预注册，写死）：**与表面级 lockbox_v1.json 对齐**。
理由：
  1. 分组协议（GroupKFold groups=规范化 comp）下，信息泄漏的单位是"组成"，
     同一组成的所有位点记录必须同进同出；直接复用表面级 276 个 lockbox 组成
     即天然得到位点级 lockbox（这些组成下的全部位点记录冻结）。
  2. 全项目只维护一份 holdout 定义，避免双 holdout 交叉污染，
     F5 最终评估与 A-5"最终模型只评一次"承诺同口径。
  3. lockbox_v1 已按 structure 分层（禁用目标值 stratify），位点级继承该性质。

留证：lockbox 组成列表的 SHA256（与 splits/lockbox_v1.json 内嵌值比对）+
      位点级 lockbox 记录表（由 F1 生成）落盘后的 SHA256 回填到本文件。
纪律：本脚本运行后，开发流程（F1 自检之后的一切建模）不得读取位点 lockbox 能量值；
      仅当 F5 表面级对照位点模型占优时，才允许在 lockbox 上评估一次。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LB_V1 = ROOT / "splits" / "lockbox_v1.json"


def sha256_comps(comps) -> str:
    """与 scripts/30_lockbox_split.py 同口径。"""
    return hashlib.sha256(json.dumps(comps, sort_keys=True).encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    lb = json.loads(LB_V1.read_text())
    comps = sorted(lb["lockbox_comps"])
    sha = sha256_comps(lb["lockbox_comps"])
    assert sha == lb["sha256_of_comp_list"], (
        f"lockbox_v1 组成列表 SHA256 不一致: {sha} != {lb['sha256_of_comp_list']}")
    doc = {
        "version": "site_lockbox_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "choice": "aligned_with_surface_lockbox_v1",
        "choice_reason": __doc__.split("选择（预注册，写死）：")[1].split("留证：")[0].strip(),
        "protocol": {
            "group_unit": "规范化 comp（与 GroupKFold 折协议同口径）",
            "fraction": lb["protocol"]["fraction"],
            "n_lockbox_comps": len(comps),
            "n_total_comps": lb["n_total_comps"],
            "structure_counts_lockbox": lb["structure_counts_lockbox"],
            "rule": ("位点级 lockbox = lockbox_v1 组成下的全部位点记录；"
                     "自划定起对开发不可见；最终只评一次（仅当 F5 占优）"),
        },
        "lockbox_comps": comps,
        "sha256_of_comp_list": sha,
        "sha256_of_lockbox_site_rows": None,  # F1 生成记录表后回填
        "touched_for_final_eval": False,
    }
    (OUT / "f0_site_lockbox.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[F0] lockbox comps={len(comps)} sha256={sha[:16]}... "
          f"(与 lockbox_v1 一致) → {OUT/'f0_site_lockbox.json'}")


if __name__ == "__main__":
    main()
