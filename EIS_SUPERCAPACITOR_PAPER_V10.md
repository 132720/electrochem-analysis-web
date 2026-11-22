# v10 超级电容器 EIS 论文绘图工作流

## 文献依据（本次核对了原始全文/图表）
1. You et al., *Electrochimica Acta* 538 (2025) 146968, DOI 10.1016/j.electacta.2025.146968。Fig.6f：三种电极 Nyquist、局部高频、等效电路；讨论 Rs、Rct 和低频斜率。
2. Hu et al., *Electrochimica Acta* 548 (2026) 147917, DOI 10.1016/j.electacta.2025.147917。Fig.8h：四种电极 Nyquist 与电路拟合；Table 1 Rs/Rct（Ω）：
   - ZMS: 0.762, 5.589
   - ZMS/MXene-2.5: 0.761, 3.159
   - ZMS/MXene-5: 0.736, 1.887
   - ZMS/MXene-7.5: 0.758, 2.033
3. You et al., *Journal of Colloid and Interface Science* 698 (2025) 138011, DOI 10.1016/j.jcis.2025.138011。Fig.7f：四个电极 Nyquist；Table 1 及正文显示 Rs/Rct：
   - NiCo2O4 1.98/14.31 Ω
   - NiCo2O4@LDH 0.96/3.97 Ω
   - NiCo2O4@MXene 0.93/1.91 Ω
   - NiCo2O4@MXene@LDH 0.87/0.76 Ω
4. Perdana et al., *Chemical Record* 24 (2024), e202400007, DOI 10.1002/tcr.202400007；超级电容器 EIS 参数：bulk/charge-transfer/total resistance, capacitance, response frequency/time。
5. *Journal of Energy Storage* 99 (2024) 113267：Nyquist（实验/拟合）、等效电路、C'/C''、Bode 例子。
6. RSC *Materials Advances* 2025, DOI/地址 https://pubs.rsc.org/en/content/articlehtml/2025/ma/d5ma00352k；高频 Nyquist 局部放大与电路插图。
7. *Batteries* 11 (2025) 116, https://www.mdpi.com/2313-0105/11/4/116；Nyquist、Rs/Rct/W、阻抗拟合。

## 在 EIS 页面怎样用
1. 上传多个 EIS TXT/CSV/XLSX/ZIP，使用样品名称而非温度作为图例。
2. 选择下拉电路：
   - Rs+CPE，适合无清晰可辨阻抗弧的非理想容抗；
   - Rs+(Rct||CPE)，适合有可辨阻抗弧且可能有界面过程的情况；
   - Rs+(Rct||CPE)+W，需要有半无限扩散机制解释的依据。
3. 可选是否排除 Z''>0 的感抗频点；所有数据保持原样不覆盖。
4. 点击“拟合并生成论文图”。可以修改英文图例、图宽、图高及高频局部放大开关。
5. 下载 ZIP：Nyquist、Bode 幅值与相位、残差、Rs/Rct 对比、等效电路、复电容曲线各自的 **SVG/PDF 矢量文件和 600dpi PNG**；附原始阻抗、逐频拟合、参数 CSV 和元数据 JSON。
6. 旧版固定电路/KK 筛查在“高级：旧版 KK 与固定电路诊断”折叠栏中；Z-HIT/DRT 亦为可选专业模块。

## 科研解释风险（必须阅读）
- Rs 是所采用电路中的串联电阻。它的物理组成与测试系统有关，不能机械等同于某一个单独的欧姆贡献。
- Rct 只有在所选择模型及阻抗弧具有合理物理意义且**参数可辨识**时才能解释为电荷转移电阻。遇到近垂直电容尾部而 Rct 极大，软件会给出警告，不能将数值作为确定的界面 Rct。
- Warburg σ 的半无限模型在有限频率/孔结构复杂材料中可能不适用；不能自动从低频斜率推导绝对扩散系数。
- 图中 C* = 1/(jωZ)，报告的是表观**未归一化复电容**。未输入质量、面积、几何与电极配置，不报告 F/g 或 F/cm²。
- 当 C'' 峰落在测试频率边界时，不报告 f0 或 τ0；否则仅报告以 1/f0 定义的表观响应时间。
- 质量/格式检测、参数误差、残差统计用于辅助分析；所有真实论文结论须对照原始实验、测量条件、所选电路并独立验证。

## 当前验证状态
- 内置 84 点 CHI EIS 示例上完成三种电路拟合，所有图型能导出有效 PNG/SVG/PDF。
- 对已知生成的 Rs=0.9 Ω、Rct=3.4 Ω、Q=0.0025、n=0.86 理想阻抗谱实现参数回收。
- v10 回归测试以数值与文件结构为重点，未在 Windows Streamlit 界面做实际浏览器测试；仍需用你自己的实验数据做标注、字体、拟合物理合理性验收。
