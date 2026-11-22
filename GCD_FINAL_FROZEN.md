# GCD 最终模型冻结与数据质量检查

## 1. 最终冻结模型

最终保留：

**Whole-cycle PCA-GPR + native total-period GPR**

输入：
- Temperature
- log(Current density)

输出：
- PCA latent coefficients -> inverse PCA -> normalized full GCD
- log(total period) -> physical time axis
- GP posterior uncertainty

不再继续开发：
- Ghosh 3P/4P parameter regression
- Fernández 2R(C+kU)
- branch-separated PCA-GPR
- polynomial/Ridge time heads
- Peukert-only time heads

Peukert 只保留为物理诊断，不作为最终预测器。

## 2. 为什么冻结这个模型

### 正式 rolling blocked-temperature

Whole-cycle PCA-GPR:
- 50 °C：shape RMSE ≈ 16.28 mV；physical-time RMSE ≈ 30.63 mV；total-period error ≈ 6.72%
- 60 °C：shape RMSE ≈ 7.43 mV；physical-time RMSE ≈ 112.37 mV；total-period error ≈ 16.56%
- 70 °C：shape RMSE ≈ 8.32 mV；physical-time RMSE ≈ 47.29 mV；total-period error ≈ 4.09%
- overall：shape RMSE ≈ 10.68 mV；physical-time RMSE ≈ 63.43 mV；total-period error ≈ 9.12%

PCA-Ridge:
- overall shape RMSE ≈ 13.70 mV
- physical-time RMSE ≈ 84.66 mV
- total-period error ≈ 15.35%

Branch-separated PCA-GPR:
- charge shape RMSE ≈ 11.06 mV
- discharge shape RMSE ≈ 17.44 mV
- physical-time RMSE ≈ 67.99 mV
- 未优于 whole-cycle PCA-GPR

Last-temperature template:
- normalized shape RMSE ≈ 9.87 mV（略优于 PCA-GPR）
- 但结合相同 time head 后 physical-time RMSE ≈ 65.17 mV
- 不提供连续 T/J 概率预测与 GP uncertainty，因此作为诊断 baseline，不作为最终部署模型。

## 3. Time-head 筛选结论

测试过：
- native total-period GPR
- separate log-time GPR
- specific-charge GPR
- polynomial Ridge
- Peukert linear surface
- Peukert parameters vs T
- Peukert trend + GP residual
- per-current temperature trend

没有任何 time model 在 50/60/70 °C 三个外推折上稳定占优。

典型现象：
- 50 °C：native total-period GPR 最稳定之一
- 60 °C：Peukert-linear 类时间模型更好
- 70 °C：separate log-time GPR / native GPR 明显更好

说明 temperature dependence 存在非线性/阶段性变化，继续针对 15 个 blocked 条件调 time head 很容易变成 validation overfitting。

因此冻结更简单、概率化、整体表现最平衡的 native total-period GPR。

## 4. 数据有没有问题？

### 4.1 没发现“数据坏掉”或明显录入错误

25 个 GCD 条件：
- Temperature = 30, 40, 50, 60, 70 °C
- Current density = 0.5, 1, 1.5, 2, 2.5 A/g

所有文件实验设置一致：
- CHI660E
- voltage window = 0–0.5 V
- data storage interval = 0.1 s
- Segment = 4
- anodic/cathodic current magnitude一致
- 文件中的绝对电流与电流密度严格对应 3 mg 活性材料质量

### 4.2 文件内部重复 discharge 的重复性很好

每个 Segment=4 文件含两个完整 discharge 段，可以做内部重复性检查。

25 个条件统计：
- 两次 discharge duration 平均差异 ≈ 0.50%
- 中位 duration 差异接近 0
- 两次 normalized discharge shape RMSE 平均 ≈ 4.75 mV
- 中位 ≈ 2.85 mV
- 最大 ≈ 14.18 mV（主要出现在较高 current density）

所以 GCD 本身不是明显“乱数据”。

### 4.3 真正困难来自 temperature trend，而不是随机噪声

Discharge specific charge 在所有 current density 上都随温度系统性增加。

相邻 10 °C 的增幅大致：
- 30→40 °C：约 21–24%
- 40→50 °C：约 9–15%
- 50→60 °C：约 22–25%
- 60→70 °C：约 28–37%

这种增长不是简单线性或单一指数趋势。
因此仅凭 2–4 个较低温度去推断下一个温度，本身就是困难的 extrapolation 问题。

### 4.4 normalized shape 反而在高温区越来越接近

相邻温度完整 normalized GCD shape 的平均 RMSE：
- 30→40 °C：≈ 21.95 mV
- 40→50 °C：≈ 15.52 mV
- 50→60 °C：≈ 6.95 mV
- 60→70 °C：≈ 7.12 mV

说明高温区曲线“长什么样”已经相对稳定，真正变化更大的是“持续多长时间”。

这也是为什么模型经常出现：
- normalized shape 很好
- physical-time curve 错位

## 5. 目前 GCD 的主要实验设计限制

不是原始数据错误，而是：

1. 只有 5 个温度点，每 10 °C 一个点；
2. 每个 T×J 没有独立重复样品/独立重复实验；
3. 只有文件内部 cycle 重复，不能完全替代独立重复；
4. 高 current 条件下一个 branch 只有几十个 0.1 s 采样点，transition timing 的相对量化误差天然更大；
5. 因此不能合理期待未知温度 full V(t) 误差被压到几 mV，同时仍保持真正的 temperature extrapolation。

## 6. 下一阶段：EIS

GCD 主模型冻结后，下一步进入 EIS。

初步检查 5 组 EIS：
- 30/40/50/60/70 °C 各 1 条
- 84 frequency points
- nominal range 100 kHz → 0.01 Hz
- AC amplitude = 5 mV
- quiet time = 2 s

高频 Z''=0 近似交点对应的 series-resistance-like 值：
- 30 °C：≈ 0.687 Ω
- 40 °C：≈ 0.638 Ω
- 50 °C：≈ 0.640 Ω
- 60 °C：≈ 0.612 Ω
- 70 °C：≈ 0.572 Ω

总体随温度升高下降，和温度升高后 GCD duration / charge storage 增大的趋势方向上是相容的，但正式机制结论必须等 equivalent-circuit / residual / uncertainty 分析后再写。

一个重要限制：
EIS 的 Init E 并不完全相同：
- 30 °C：0.224 V
- 40 °C：0.273 V
- 50 °C：0.319 V
- 60 °C：0.319 V
- 70 °C：0.314 V

因此后续温度 EIS 比较必须明确说明：temperature effect 与 DC bias / SOC effect 可能存在一定耦合，尤其对 pseudocapacitive / redox-active electrode 更要谨慎。
