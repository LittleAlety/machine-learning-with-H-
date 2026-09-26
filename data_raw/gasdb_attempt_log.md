# GASdb 获取尝试记录（外部验证支线，闸门触发）

- 下载：**成功**。`gasdb_docs.pkl` 1,254,607,956 字节（≈1.25 GB），
  来源 https://media.githubusercontent.com/media/ulissigroup/uncertainty_benchmarking/master/preprocessing/pull_data/gaspy/docs.pkl
  pickle 协议 3，顶层为 list of dict（gaspy mongo 文档）。
- 环境补装：ase 3.29.0、pymongo（bson 依赖）。
- 解析尝试 #1：直接 `pickle.load` → **MemoryError**（2.6 GB 地址空间限制内；
  本机物理内存仅 3 GB，1.25 GB pickle 反序列化展开后远超内存）。
- 解析尝试 #2：自定义 `pickle.Unpickler` 流式拦截顶层 list 的 APPENDS、
  仅保留 H 吸附物轻量字段 → 仍 **MemoryError**（pickle memo 表对全流
  数百万对象持续引用，无法随条目释放）。
- 尝试加 swap（fallocate+mkswap 成功，`swapon` 被容器拒绝：Invalid argument）。
- 结论：按预设闸门（2 次解析失败即放弃），GASdb 支线终止，不阻塞主任务；
  放弃原因为内存不足（不变）。**交付状态**：该 1.25 GB pkl 已由主控删除
  以控制交付体积，不在 data_raw/ 中；上方 URL 保留，可随时重新下载
  供有大内存环境者复用。
