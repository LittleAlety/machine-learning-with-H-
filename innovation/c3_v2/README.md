# C-3 v2 预注册批次清单（Phase 6，红线 16）

`batches_preregistered.json`：计算前公开的三批次清单（时间戳 + SHA256）。

- **批次 A（10）**：主动学习采集函数顶部，红旗过滤 + in_domain；目标：压全局误差（回溯预期 −18% 量级）。
- **批次 B（10）**：HER×ORR score 最近邻且无红旗的第二梯队（由 dual_all_ranked.csv 全量 1,836 排序重算）；全部 oh_in_training=False（模型外推），DFT 结果即"逃逸角"裁决的独立证据。Cr 体系附带 3.14 节钝化警示。
- **批次 C（25，目标 30）**：CHGNet 弛豫重构最剧烈者（surf_disp_mean Top，三结构分层），裁决 Phase 5 Route 2 弛豫悬案；仅 25 个因弛豫子集（256/663）覆盖所限，如实记录。

执行前置（未满足前不得开跑）：协议一致性（BEEF-vdW/QE）、锚定闸门 5–10 体系 |ΔE|≤0.1 eV、3 探针任务核时校准。DFT 生产需外部集群，本包为执行就绪态。

## 预检迭代（2026-09-20）后补文件

- `preflight_report_2026-09-20.md`：DFT 生产前独立预检报告。A/B 名单完全复现闭合，C 在隐含分层配额下复现闭合；判决 Stage 1 CONDITIONAL GO、Stage 2/3 NO-GO。
- `PREREGISTRATION_AMENDMENT_1.md`：修正案 1——①哈希链澄清（内嵌声明值不可复现，权威校验移交独立校验文件）；②批次 C 的 12 L1₀+1 A1+12 L1₂ 分层配额显式化；③9 个锚定闸门候选预注册；④核时复核 199,875 核时确认；⑤批次 C 定位为"基座势伪影检测集"。
- `preregistration_checksums.sha256`：独立校验文件（detached checksums），含文件字节摘要、可复现的规范内容摘要（payload 去 sha256 字段后 `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)`）与三个上游源文件摘要。**权威校验入口以此文件为准**。
