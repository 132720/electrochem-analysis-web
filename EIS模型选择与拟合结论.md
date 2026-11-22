# 温度 EIS 模型选择、统一拟合与 GCD 机理关联结论

## 1. 数据与实验条件

当前原始 ZIP 中共有 5 条温度 EIS：

- 30、40、50、60、70 °C；
- 每个温度 1 条 EIS；
- 每条 84 个有效频率点；
- 实际频率约 82.5 kHz–0.01 Hz；
- AC amplitude = 5 mV；
- quiet time = 2 s。

重要实验限制：

| T / °C | Init E / V |
|---:|---:|
| 30 | 0.224 |
| 40 | 0.273 |
| 50 | 0.319 |
| 60 | 0.319 |
| 70 | 0.314 |

因此温度变化与 DC bias / electrochemical state 并没有完全解耦。特别是对于具有 redox / pseudocapacitive contribution 的体系，后续不能把所有阻抗变化全部归因于温度。

此外，每个温度只有一条 EIS，没有独立重复谱。因此拟合得到的局部 95% CI 只反映 CNLS/Jacobian 局部参数不确定度，不能代替独立实验重复性。

---

## 2. 原始谱特征

Nyquist 图表现出：

1. 最高频存在小的感抗响应，因此正式电路中保留 series inductance `L`；
2. 高频 Z''=0 交点对应的 series-resistance-like intercept 随温度总体下降；
3. 中低频没有一个非常清晰、稳定、可唯一归属的单一半圆；
4. 低频阻抗急剧增大，呈明显 non-ideal distributed capacitive / porous response；
5. 因此不能仅凭 Nyquist 外形就把某一段武断地命名为 `Rct` 或 Warburg diffusion。

通过 Z''=0 线性/对数频率插值获得的高频截距：

| T / °C | Z''=0 时 Z' / Ω |
|---:|---:|
| 30 | 0.6870 |
| 40 | 0.6376 |
| 50 | 0.6403 |
| 60 | 0.6117 |
| 70 | 0.5716 |

---

## 3. KK compatibility screening

使用项目中的 linear measurement-model KK-compatible screening，软件 QA 阈值设置为 5%。

| T / °C | relative RMSE | max normalized error | software screening |
|---:|---:|---:|---|
| 30 | 1.574% | 4.182% | pass |
| 40 | 1.315% | 3.098% | pass |
| 50 | 1.183% | 2.663% | pass |
| 60 | 1.263% | 2.789% | pass |
| 70 | 1.252% | 2.456% | pass |

因此 5 条谱均可继续进入模型比较。

注意：这个 5% 是软件 QA screening threshold，不是 Kramers–Kronig 理论规定的普适“合格线”。

---

## 4. 候选电路筛选

统一使用 modulus weighting / relative-complex residual，并显式包含高频 series inductance。

测试候选：

1. `L + Rs + CPE`
2. `L + Rs + (Rpar || CPE)`
3. `L + Rs + TLM(CPE)`
4. `L + Rs + RQ1 + RQ2`

平均结果：

| Circuit | Mean weighted RMSE | Boundary fraction | Median Jacobian condition number |
|---|---:|---:|---:|
| L + Rs + RQ1 + RQ2 | 1.742% | 40% | 1.12×10⁴ |
| L + Rs + (Rpar || CPE) | 4.599% | 20% | 3.65×10⁴ |
| L + Rs + TLM(CPE) | 4.653% | 40% | 2.11×10⁴ |
| **L + Rs + CPE** | **4.681%** | **0%** | **8.94×10¹** |

双 RQ 模型虽然数值残差最低，但不能据此作为最终机理模型。其原因是：

- 部分温度参数触及约束边界；
- Jacobian conditioning 明显变差；
- 两个 RQ 的 characteristic frequency 在不同温度之间不能稳定对应同一物理过程；
- 部分参数局部 CI 很宽；
- 因此将其中 `R1` / `R2` 命名为 `Rct`、`Rads` 或其他具体机理参数会过度解释。

`TLM(CPE)` 在 30/40 °C 也出现退化趋势；`Rpar || CPE` 在 30 °C 的并联电阻趋向极大，相当于退化回纯 CPE。

因此采用“最简、全温度可辨识、参数不触边”的原则，最终正式定量电路冻结为：

\[
\boxed{
Z(\omega)
=
R_s+j\omega L+
\frac{1}{Q(j\omega)^n}
}
\]

复杂模型只作为“数据可能包含多个分布式时间尺度”的描述性证据，不用于强行分配具体 Rct。

---

## 5. 最终 L + Rs + CPE 拟合参数

| T / °C | Rs / Ω | Q / S·sⁿ | n | Weighted RMSE |
|---:|---:|---:|---:|---:|
| 30 | 0.64185 | 0.008919 | 0.66255 | 6.728% |
| 40 | 0.61573 | 0.007841 | 0.70744 | 2.677% |
| 50 | 0.63450 | 0.008983 | 0.72500 | 4.682% |
| 60 | 0.60345 | 0.008699 | 0.72200 | 4.778% |
| 70 | 0.55709 | 0.009022 | 0.71138 | 4.540% |

Rs 的局部 95% CI：

| T / °C | Rs / Ω | approx. 95% CI / Ω |
|---:|---:|---:|
| 30 | 0.64185 | 0.62283–0.66146 |
| 40 | 0.61573 | 0.60883–0.62270 |
| 50 | 0.63450 | 0.62280–0.64642 |
| 60 | 0.60345 | 0.59188–0.61524 |
| 70 | 0.55709 | 0.54663–0.56775 |

总体趋势是 `Rs` 随温度升高下降，50 °C 存在一个轻微局部抬升。

`Q` 不应直接称为 capacitance，因为 CPE 的 Q 单位依赖 n。软件同时给出 frequency-referenced phenomenological effective magnitude，但只用于趋势描述，不把它当独立真实电容常数。

---

## 6. 低频 / relaxation 分析

将阻抗转换为 complex capacitance 后，所有温度的 `C''` 最大值均出现在最低测量频率 0.01 Hz。

这意味着真正的 C'' peak 很可能位于当前测量窗以下，因此：

\[
\boxed{\text{当前数据不能可靠提取正式 relaxation time }\tau_0}
\]

后续论文不要为了得到一个 relaxation time 而直接把 0.01 Hz 当成峰频率。

如果未来要做可靠 relaxation-time / characteristic-frequency 机理，建议 EIS 扩展到更低频。

---

## 7. Arrhenius 诊断

仅对当前最稳健的 `Rs` 做探索性 Arrhenius-like 检查：

\[
\ln(1/R_s)\ \text{vs.}\ 1/T
\]

得到：

- apparent Ea ≈ **2.58 kJ mol⁻¹**
- R² ≈ **0.708**
- p ≈ **0.074**

由于：

1. 只有 5 个温度点；
2. Arrhenius 线性并不充分；
3. 50 °C 有局部偏离；
4. 每温度只有一条 EIS；
5. Init E 随温度不同；

因此不建议把 2.58 kJ mol⁻¹ 写成确定的 charge-transfer / diffusion activation energy。

更合适的论文表述是：

> `Rs` exhibits a temperature-dependent decreasing tendency, while the linear Arrhenius representation is only moderate under the present measurement conditions; therefore the derived apparent activation energy is reported only as an exploratory descriptor.

---

## 8. EIS 与 GCD 的跨模态关联

对每个温度，将 5 个电流密度的 GCD discharge specific charge 做平均：

| T / °C | mean discharge q / C g⁻¹ | EIS Rs / Ω |
|---:|---:|---:|
| 30 | 9.68 | 0.64185 |
| 40 | 11.91 | 0.61573 |
| 50 | 13.45 | 0.63450 |
| 60 | 16.63 | 0.60345 |
| 70 | 21.91 | 0.55709 |

温度级别相关性：

- Pearson r ≈ **−0.944**, p ≈ **0.016**
- Spearman ρ ≈ **−0.900**, p ≈ **0.037**

这表明在当前数据中：

\[
T\uparrow
\quad\Rightarrow\quad
R_s\ \text{总体下降}
\]

同时：

\[
T\uparrow
\quad\Rightarrow\quad
q_{\mathrm{dis}}\uparrow
\]

二者具有明显伴随关系。

但这不能证明：

\[
R_s\downarrow \Longrightarrow q_{\mathrm{dis}}\uparrow
\]

是单一因果机制，因为：

- 两者都共同受 temperature 驱动；
- 只有 5 个温度点；
- EIS Init E 不完全相同；
- 当前简化电路没有把界面动力学、孔道分布、redox process 唯一解耦。

所以论文中建议使用：

> “The decrease in the robust series-resistance descriptor is accompanied by an increase in accessible GCD charge with increasing temperature, consistent with improved overall ionic/electrical transport; however, this temperature-level association is not interpreted as proof of a direct causal relationship.”

---

## 9. 本章现在形成的最终科学链

GCD 部分已经得到：

\[
(T,J)
\rightarrow
\text{probabilistic full GCD}
\]

并发现：

> normalized GCD shape 比 absolute time scale 更容易跨温度外推。

EIS 现在补充：

\[
T
\rightarrow
R_s,\ Q,\ n,\ \text{distributed impedance response}
\]

并观察到：

\[
R_s(T)\downarrow
\]

与：

\[
q_{\mathrm{dis}}(T)\uparrow
\]

方向一致。

因此论文可以形成：

\[
\boxed{
\text{ML predicts what happens}
\quad+\quad
\text{EIS provides independent mechanism-related evidence for why temperature changes the response}
}
\]

但必须保持措辞：EIS 是独立支持性机理证据，不是对 ML 的“证明”。

---

## 10. 主要文献链

1. Civan et al., **Temperature-Dependent Electrochemical Impedance Spectroscopy of Batteries**, ACS Electrochemistry, 2026. DOI: `10.1021/acselectrochem.5c00512`.
2. Kötz, Hahn, Gallay, **Temperature behavior and impedance fundamentals of supercapacitors**, Journal of Power Sources, 2006. DOI: `10.1016/j.jpowsour.2005.10.048`.
3. Köps et al., **An in-depth investigation into the influence of temperature on the electrochemical behavior of electric double-layer capacitors...**, Journal of Power Sources, 2023. DOI: `10.1016/j.jpowsour.2023.233480`.
4. Barua & Paul, **Unravelling the role of temperature in a redox supercapacitor...**, RSC Advances, 2020. DOI: `10.1039/C9RA09768F`.
5. Scisco et al., **On the rate capability of supercapacitors characterized by a constant-phase element**, Journal of Power Sources, 2021. DOI: `10.1016/j.jpowsour.2021.230700`.
6. Abouelamaiem et al., **Correlating electrochemical impedance with hierarchical structure for porous carbon-based supercapacitors using a truncated transmission line model**, Electrochimica Acta, 2018. DOI: `10.1016/j.electacta.2018.07.190`.

## 11. 下一步

EIS 的第一阶段可以冻结为：

**raw spectrum QC → KK screening → candidate-model screening → L + Rs + CPE robust parameterization → cautious Arrhenius diagnostic → EIS–GCD cross-modal association.**

后续重点不应继续为了降低拟合残差而增加更多 R/CPE 元件。

真正值得做的是：
- 把最终 EIS 图、参数趋势、GCD–EIS 关联组织成论文图表；
- 明确实验限制；
- 如果导师允许补实验，优先补相同 DC bias / electrochemical state 下的 EIS replicate，或扩展更低频，以增强机理可信度。
