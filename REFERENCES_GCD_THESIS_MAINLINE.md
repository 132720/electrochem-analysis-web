# GCD 温度预测主线与文献依据

## 核心定位

本项目不宣称已有文献完整提出“PG-BDL”这一模型。论文中应将方法描述为：
**在直接的超级电容温度-GCD机器学习研究基础上，结合完整电化学曲线的低维表示、
物理引导全曲线预测、物理模型+稀疏数据残差学习、以及快慢动力学解耦等高水平文献思想，
构建适用于本研究小样本温度外推任务的 physics-guided branch-decoupled latent model。**

## 1. 直接领域基准：超级电容温度 GCD + 机器学习

S. Ghosh et al., “Temperature-dependent performance prediction for cerium oxynitride
solid-state symmetric supercapacitor using machine learning,” Journal of Energy Storage,
113 (2025) 115562. DOI: 10.1016/j.est.2025.115562.

直接支撑：
- 使用温度和电流研究超级电容 GCD；
- 采用 Linear Regression、SVR、Decision Tree、Random Forest；
- 先预测充放电 time period；
- 进一步用“指数项 + 线性项”的 V(t) 参数化函数重构充放电曲线。
因此本项目把该路线实现为 Ghosh-inspired literature baseline 是合理的。

## 2. PCA + 回归用于完整电化学曲线：高水平方法学支撑

L. Su et al., “Battery Charge Curve Prediction via Feature Extraction and Supervised
Machine Learning,” Advanced Science 10 (2023) 2301737.
DOI: 10.1002/advs.202301737.

关键支撑：
- 对 10,066 条恒流充电曲线进行 PCA / NMF / Autoencoder 特征提取；
- 使用低维特征和线性回归预测完整充电曲线；
- 说明完整电化学曲线可由少量低维基函数表示，再预测对应系数。
本项目的 PCA + Ridge 并非照搬该论文，而是将“低维曲线表示 + 正则化线性回归”
迁移到超级电容 GCD 温度外推问题。

## 3. Physics-informed 完整曲线预测

A. Tang et al., “Physics-informed battery degradation prediction: Forecasting charging
curves using one-cycle data,” Journal of Energy Chemistry 101 (2025) 825–836.
DOI: 10.1016/j.jechem.2024.10.018.

关键支撑：
- 目标不是只预测一个容量标量，而是预测完整 V-Q 曲线；
- 使用 lightweight interpretable physical model + physics-informed neural network；
- 将领域知识加入模型结构/训练过程；
- 强调有限历史数据、小样本和可解释性。
本项目应表述为“迁移其 physics-informed full-curve prediction 思想”，
不能宣称逐项复现 Tang 模型。

## 4. 小样本下“物理基线 + 数据残差”思想

J. Fogelquist, X. Lin, “Combining electrochemistry and data-sparse Gaussian process
regression for lithium-ion battery hybrid modeling,” Applied Energy 399 (2025) 126458.
DOI: 10.1016/j.apenergy.2025.126458.

关键支撑：
- 将 physics-based electrochemical model 与 GPR 结合；
- GPR 专门补偿物理模型未描述的 voltage residual；
- 强调 sparse-data 场景下 GPR 的优势和较低参数量。
这直接支撑本项目“尽量把实验边界、时间尺度、IR-drop等已知结构显式建模，
只让数据模型学习剩余 shape residual”的设计原则。

## 5. 动力学解耦和 latent representation

T. Desai, A. J. Gallo, R. M. G. Ferrari,
“Multi timescale battery modeling: Integrating physics insights to data-driven model,”
Applied Energy 393 (2025) 126040.
DOI: 10.1016/j.apenergy.2025.126040.

关键支撑：
- 通过架构区分快动态响应与慢变化状态；
- 把物理知识同时加入 loss 和 architecture；
- latent space 被设计为具有可解释的状态表示；
- 在稀疏循环数据上验证。
本项目的“充电—换向—放电分支解耦 + latent residual”属于同类方法思想迁移，
不是对该文模型的复现。

## 6. 为什么必须做真正的 unseen-condition / blocked extrapolation

T. Li et al., “Coupling a capacity fade model with machine learning for early prediction
of the battery capacity trajectory,” Applied Energy 389 (2025) 125703.
DOI: 10.1016/j.apenergy.2025.125703.

关键支撑：
- 纯数据模型可能在训练分布外泛化不足；
- 显式比较 in-distribution 与 out-of-distribution (OOD)；
- 通过经验/物理轨迹模型与 ML 结合改善 OOD trajectory prediction。
因此本项目用整温度留出、滚动高温外推，而不是随机拆分曲线点，是方法学上必要的。

## 7. GCD 的 IR-drop / 阻性结构与温度 EIS 的连接

L. Köps, F. A. Kreth, M. Klein, A. Balducci,
“An in-depth investigation into the influence of temperature on the electrochemical
behavior of electric double-layer capacitors containing ethyl isopropyl sulfone-based
electrolytes,” Journal of Power Sources 581 (2023) 233480.
DOI: 10.1016/j.jpowsour.2023.233480.

关键支撑：
- 对 EDLC 同时进行 temperature-dependent EIS 和 GCD；
- 温度升高可显著改变器件内阻与 Warburg/离子传输表现；
- 温度 EIS 所揭示的阻抗变化与温度 GCD 性能变化相互对应。
因此用 EIS 的 Rs / Rct / 扩散相关参数解释 GCD 中温度依赖的 IR-drop/阻性行为，
在研究逻辑上是成立的。

进一步机理文献：
T. Zhao et al., “How the Thermal Effect Regulates Cyclic Voltammetry of Supercapacitors,”
ACS Energy Letters 8 (2023) 3365–3372.
DOI: 10.1021/acsenergylett.3c01045.

该工作提醒：温度效应可能是非单调的，不能简单硬约束为“温度越高电容越大”。

## 8. 论文写法建议

建议方法章节按以下逻辑写：

1. Literature baseline：Ghosh-inspired parametric DT/RF。
2. Strong small-data baseline：PCA + Ridge。
3. Proposed model：
   - protocol/experimental boundaries；
   - charge / transition / discharge decomposition；
   - duration / resistance-like scalar heads；
   - low-dimensional latent branch shape；
   - residual-only data learning。
4. Evaluation：
   - rolling blocked-temperature extrapolation；
   - common-condition fair comparison；
   - normalized-shape metrics；
   - physical-time V(t) metrics；
   - branch-region metrics；
   - scalar-head diagnostics；
   - ablation study。
5. Mechanism validation：
   - temperature EIS；
   - Rs / Rct / CPE(or Cdl) / diffusion-related parameters；
   - Arrhenius analysis where physically applicable；
   - compare resistance-related temperature trends with GCD IR-drop behavior。

## 重要边界

- Ghosh 2025 是“直接超级电容温度 GCD 预测”支撑。
- PCA、physics-informed full-curve、latent/residual hybrid 的高水平依据主要来自锂离子电池。
  论文必须写成“methodological transfer / inspired by”，不能写成“已被证明直接适用于本超级电容体系”。
- PG-BDL 是本项目自己组合形成的模型，不应伪装成某篇论文已有模型。
- 最终是否成立必须依靠本研究的 blocked-temperature 外推、消融和 EIS 机理验证。


## Benchmark V2 后的 V4 消融设计

最新 blocked benchmark 显示：PG-BDL V3 在归一化时间上的 shape 可取得较低误差，但 physical-time V(t) 指标明显恶化，且 charge/main-discharge duration 在 60–70 °C 外推时出现系统性低估。因此 V4 首先只修改 scalar heads，不改变 latent branch，以保证消融可归因。

### V4-Q：specific-charge timing

恒流条件下，单位质量电荷量满足：

    q_sp = J * Δt

若电压窗口固定，则通常 GCD 比电容满足：

    C_sp = J * Δt / ΔV

因此 V4-Q 不再自由预测 duration，而是预测 q_sp 后用 Δt = q_sp / J 恢复时间。
直接超级电容依据：Ghosh et al. (2025) 在其温度 GCD 研究中使用 ACG = I_s * t_dis / (ESA * ΔV)；常规 gravimetric GCD 同样由 I*Δt/(m*ΔV) 得到比电容。
高水平补充依据：Journal of Power Sources 2026 的 GCD 电化学建模工作强调 charge balance / dynamic-capacitance circuits 对完整 GCD profile 的约束作用（DOI: 10.1016/j.jpowsour.2025.239107）。

### V4-QIR：current-normalized voltage-drop heads

V4-QIR 进一步把起始边界、末端边界、换向电压降写成“电流密度 × drop coefficient”。因为当前输入为 A/g，而不是绝对电流 A，本程序输出单位为 V/(A/g)，仅称 resistance-like coefficient；只有已知 active mass 并转为绝对电流时才可进一步解释为 Ω 量纲的等效阻性参数。

这一设计与超级电容 ESR/阻性建模文献一致，但不宣称换向瞬态完全等于单一 ESR：
- Journal of Power Sources 2014 review: Review of characterization methods for supercapacitor modelling, DOI 10.1016/j.jpowsour.2013.07.116.
- Journal of Power Sources 2024 review: Unleashing capabilities of supercapacitors: Strategies to reduce internal resistances, DOI 10.1016/j.jpowsour.2024.234068.
- Journal of Power Sources 2010: Characterization of equivalent series resistance of electric double-layer capacitor electrodes using transient analysis, DOI 10.1016/j.jpowsour.2010.01.086.

### 科学边界

V4-Q / V4-QIR 现在只是 ablation candidates。若它们不能在至少两个 blocked test temperatures、多个 current densities 上稳定降低 physical-time error，则不应升级为 dissertation proposed model。V3 仍需作为对照保留。
