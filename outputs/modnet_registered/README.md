# MODNet 注册受试者（Phase 3 Workstream G）— 结果阴性

预注册：`config/preregistered_phase3.yaml :: workstream_G_modnet`（判读规则先于实验冻结）。

## 锁定配置
- 特征筛选：84 维内贪心前向 ≤20 维，打分 = **|Pearson(feature, OLS 残差)|**（二选一锁死，弃互信息），仅用外层训练折统计，无泄漏。
- 网络：torch 64-32-1 ReLU MLP，Adam(lr=1e-3)，MSE，max 500 epoch；early stopping(patience=30) 仅看内层 15% 组感知验证折。
- 标准化：StandardScaler 仅在内层训练子折拟合。
- 协议：与 v3 完全同折（GroupKFold(5, groups=comp)）同 10 种子。
- 嵌套 CV：内层折仅用于 early stopping，不调任何超参 → 外层 OOF 即嵌套评估（by-design 双过）。

## 结果（`modnet_registered.json`）
- MODNet OOF MAE = **0.1523 ± 0.0042 eV**（10 种子），v3 基线 0.11374 eV
- 改善 = **−0.0386 eV**（劣化），方向一致性 **0/10**

## 判定
fair_contract（改善 >0.005 eV 且 10/10 且嵌套双过）**不过 → 如实阴性**，
**不触发采纳流程**（不做 LOEO / CatHub 跨域复测）。
按预注册 reject 条款：入表14；3.7 扩展节升级为"含领域最强小数据 NN 模板在内函数类无红利"。

## 复跑
```bash
python scripts/37_modnet_registered.py   # CPU 即可，幂等，覆盖写 modnet_registered.json
```
