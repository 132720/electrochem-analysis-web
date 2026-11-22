# PG-BDL V2 文献依据

本模型不是对单篇论文的逐项复现，而是把近年来几条已经被验证的高质量方法路线，
针对“温度 + 电流密度 → 超级电容完整 GCD”重新组合为一个物理引导的小样本曲线预测框架。

## 1. 超级电容温度依赖 GCD 的直接文献基准

S. Ghosh et al.  
**Temperature-dependent performance prediction for cerium oxynitride solid-state symmetric supercapacitor using machine learning**  
Journal of Energy Storage, 113 (2025) 115562.  
DOI: 10.1016/j.est.2025.115562

作用：
- 直接证明 Temperature + Current 可以用于超级电容温度依赖充放电预测；
- 提供 DT / RF + 参数化 GCD 的直接 literature baseline；
- PG-BDL 不复制其指数-线性 a/b/c 方程，而把该模型作为对照。

## 2. 降维 + GP 的完整 charge–discharge curve prediction

A. A. Shah, F. Yu, W. Xing, P. K. Leung.  
**Machine learning for predicting fuel cell and battery polarisation and charge–discharge curves**  
Energy Reports 8 (2022) 4811–4821.  
DOI: 10.1016/j.egyr.2022.03.191

作用：
- 明确研究“完整充放电曲线”而不是单一 scalar；
- 使用 PCA / reduced-dimensional representation；
- 结合 Gaussian Process / SVR 预测高维曲线；
- 文中指出在有限训练数据条件下，深网络表现不理想，而降维 GP / SVR 更适合。
这与当前实验“小样本 + 高维 GCD 曲线”的数据结构高度相关。

## 3. Physics-informed 完整电化学曲线预测

A. Tang, Y. Xu, J. Tian, X. Shu, Q. Yu.  
**Physics-informed battery degradation prediction: Forecasting charging curves using one-cycle data**  
Journal of Energy Chemistry 101 (2025) 825–836.  
DOI: 10.1016/j.jechem.2024.10.018

作用：
- 证明 domain knowledge / physics constraints 可以进入完整电化学曲线预测；
- 在有限历史数据下进行完整 V–Q curve prediction；
- 支撑 PG-BDL 的 endpoint anchoring、monotonicity、physically interpretable decomposition 思路。

注意：
Tang 原文是锂离子电池 V–Q degradation prediction，不是超级电容温度 GCD。
PG-BDL 只迁移其 physics-informed 方法思想，不声称复现其模型。

## 4. 稀疏电化学数据 + Gaussian Process hybrid modeling

J. Fogelquist, X. Lin.  
**Combining electrochemistry and data-sparse Gaussian process regression for lithium-ion battery hybrid modeling**  
Applied Energy 399 (2025) 126458.  
DOI: 10.1016/j.apenergy.2025.126458

作用：
- 直接讨论 data-sparse electrochemical modeling；
- GPR 用于补偿模型误差并利用稀疏数据；
- 实验验证平均 voltage RMSE 约 7.3 mV；
- 支撑 PG-BDL 在小数据条件下使用 GPR，并保留不确定度信息。

注意：
本文是锂离子电池 hybrid electrochemical model + GPR；
PG-BDL 中的 Ridge/GPR latent regression 是方法迁移，不是原文模型复现。

## 5. Deep operator learning 对照路线

**Inferring electrochemical performance and parameters of Li-ion batteries based on deep operator networks**  
Journal of Energy Storage 65 (2023) 107176.  
DOI: 10.1016/j.est.2023.107176

作用：
- 支撑软件中 PI-BD-DeepONet 对照模式；
- 证明电化学 current → voltage 可以表述为 operator mapping；
- 不直接构成 PG-BDL 的核心算法。

---

# PG-BDL V2 的拟创新点

文献已经支持：
- complete curve prediction；
- reduced-dimensional / PCA representation；
- GPR under sparse electrochemical data；
- physics-informed curve learning。

本研究拟提出的组合是：

1. Temperature + current-density conditioned full supercapacitor GCD prediction；
2. Charge / transition / discharge decomposition；
3. total-period + charge-fraction separation；
4. explicit initial / peak / end voltage prediction；
5. explicit short transition / IR-drop prediction；
6. endpoint hard anchoring；
7. branch-wise normalized latent shape learning；
8. data-adaptive monotonic projection；
9. extrapolation-aware Ridge/GPR hybrid；
10. highest-temperature-blocked validation gate；
11. validation-selected PCA-Ridge ↔ PG-BDL convex ensemble。

论文中不应写“首次使用 PCA / GPR / Physics-Informed”。
更严谨的表述是：
“a physics-guided, branch-decoupled, validation-gated latent-space framework
was developed for temperature- and current-density-dependent full GCD prediction.”
