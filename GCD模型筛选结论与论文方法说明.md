# 温度依赖 GCD 模型筛选结论与论文方法说明

## 1. 最终结论

本研究不再将某个复杂的自命名组合模型作为创新主体。对相同的 25 个温度×电流密度 GCD 条件统一筛选后，最终采用：

**完整 GCD 的 PCA 低维表示 + Gaussian Process Regression（PCA-GPR）**

并将 **PCA-Ridge** 作为强小样本基准，将 **Peukert discharge law** 作为物理诊断。

核心任务仍沿用已有温度 GCD 研究的科学问题：利用低温条件数据预测真正未见的高温 GCD；但不再强行预测不稳定的指数函数参数，而是直接预测完整曲线的低维 latent coefficients 与 total period。

## 2. 为什么放弃 Ghosh a/b/c 参数化

对每条实验曲线直接拟合（已知目标曲线，不涉及 ML）：

- Ghosh-3P charge 平均 RMSE ≈ 11.66 mV，平均 R² ≈ 0.935；
- Ghosh-3P discharge 平均 RMSE ≈ 38.68 mV，平均 R² ≈ 0.715；
- discharge 的 a/branch-duration 中位数约 2809，说明指数时间常数进入严重退化区；
- 加 voltage offset 后 discharge 仍没有根本改善。

因此，继续让 GPR 去预测 a/b/c 会把“参数化误差/不可辨识性”带入温度外推，故不采用。

## 3. 为什么放弃 Fernández 2R(C+kU)

按照其已发表的 potential-dependent capacitance 等效电路进行了质量归一化实现，并在 25 条已知实验 GCD 上直接拟合。

结果：
- 2R-C control 平均整周期 RMSE ≈ 120.17 mV；
- 2R(C+kU) 平均整周期 RMSE ≈ 87.70 mV；
- 2R(C+kU) 平均 charge RMSE ≈ 105.63 mV；
- 平均 discharge RMSE ≈ 61.43 mV；
- 拟合参数大量触及约束边界。

因此，该 EDLC-oriented 参数化与本材料 GCD 的 plateau/redox-like 形貌不匹配，不进入后续参数预测。

## 4. 为什么 PCA 合适

在前期 branch-wise 诊断中，charge 和 discharge 各自只需约 5 个 PCA components 即可解释约 99.75% 的曲线方差，已知曲线重构误差约为亚 mV～1 mV 量级。

最终方法为了避免额外的 branch decomposition，直接对整个 normalized GCD cycle 建立 PCA。按 99.5% cumulative variance 目标，在当前全部 25 条曲线上自动选出：

- **9 PCA components**
- **99.6284% cumulative explained variance**

说明当前 GCD family 具有很强的低维结构。

## 5. 为什么 GPR 而不是继续堆神经网络

当前只有 25 个实验工况。GPR 的优势是：

- 适合小样本连续回归；
- 可对 `(Temperature, log Current density)` 建立连续映射；
- 可同时输出 posterior mean 与 uncertainty；
- uncertainty 能自然延伸到实验工况筛选。

正式模型分别训练：
1. `(T, log J) → PCA latent coefficients`
2. `(T, log J) → log(total period)`

再通过 inverse PCA 和预测 total period 恢复完整 `V(t)`。

## 6. 正式 rolling blocked-temperature 结果

验证严格按温度整块留出：

- 30, 40 → 50 °C
- 30, 40, 50 → 60 °C
- 30, 40, 50, 60 → 70 °C

### PCA-GPR

| Target | Shape RMSE | Shape R² | Physical-time RMSE | Total-period error |
|---|---:|---:|---:|---:|
| 50 °C | 16.281 mV | 0.9513 | 30.626 mV | 6.721% |
| 60 °C | 7.428 mV | 0.9902 | 112.365 mV | 16.561% |
| 70 °C | 8.324 mV | 0.9871 | 47.290 mV | 4.086% |
| Overall | **10.678 mV** | **0.9762** | **63.427 mV** | **9.123%** |

### PCA-Ridge

Overall:
- Shape RMSE = 13.705 mV
- Shape R² = 0.9644
- Physical-time RMSE = 84.659 mV
- Total-period error = 15.354%

因此 PCA-GPR 相对 PCA-Ridge：
- normalized-shape RMSE 下降约 22.1%；
- physical-time RMSE 下降约 25.1%；
- total-period relative error 下降约 40.6%。

## 7. 重要限制：时间尺度比曲线形貌更难

60 °C 是最明显的例子：

- shape RMSE 只有 7.43 mV；
- 但 total-period error 为 16.56%；
- physical-time RMSE 上升到 112.37 mV。

这说明仅使用 normalized time 的完整曲线指标会掩盖真实时间尺度误差。

因此论文中必须同时报告：
- normalized-shape metrics；
- total-period error；
- physical-time metrics。

## 8. Peukert 物理诊断

固定温度下，discharge duration 与 current density 的 Peukert-like power law 拟合非常好：

- discharge exponent k 约 1.07–1.12；
- 各温度 duration R² 均约 0.998–1.000。

但把该规律直接作为跨温度的 physics prior，并没有在 rolling temperature extrapolation 中稳定优于 conventional GPR。

因此本文不为了“physics-informed”标签而强行采用 PhysGPR：
- Peukert 用于物理合理性/速率效应诊断；
- Conventional GPR 用于最终 period extrapolation；
- 后续 temperature-dependent EIS 提供独立电化学机制解释。

## 9. 论文创新点应这样写

不是“发明一种全新的机器学习算法”，而是：

1. **任务层面**：针对超级电容未知温度完整 GCD，采用严格 rolling blocked-temperature extrapolation，而非随机拆点。
2. **表示层面**：针对直接参数化在本材料上的不可辨识问题，使用低维完整曲线表示，将高维 `V(t)` 转化为少量 latent coefficients。
3. **概率预测层面**：使用 GPR 同时预测完整曲线 latent 与 total-period，并输出 uncertainty。
4. **评价层面**：同时评价 normalized shape 与 physical-time response，揭示 time-scale extrapolation 才是当前主要瓶颈。
5. **应用层面**：构建 uncertainty-aware virtual GCD testing / experimental planning，根据不确定度优先选择值得补测的 T×J 工况。
6. **机理层面（下一阶段）**：temperature-dependent EIS 用于解释温度对阻性、动力学与扩散过程的影响，并与 GCD 时间尺度/IR-drop/charge-storage trends 关联。

## 10. 下一步

GCD 方法冻结。下一阶段转向 EIS，不再继续制造新的 GCD 模型版本，除非发现真实程序错误。
