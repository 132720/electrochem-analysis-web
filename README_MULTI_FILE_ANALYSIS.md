# CV / GCD / EIS 多文件批量分析升级

本版本基于用户当前源码直接修改。

## 重要说明

原源码中的 CV / GCD / EIS `file_uploader` 已经设置了：

`accept_multiple_files=True`

但是原来的多文件模式主要用于曲线叠加：
- CV 多文件没有完整批量参数计算；
- GCD 多文件没有完整批量参数计算；
- EIS 多文件主要用于 Nyquist 叠图，没有批量等效电路拟合。

本版本把“多文件上传”升级成真正的“多文件批量分析”。

---

# 1. CV 多文件

支持一次上传多个 CV TXT。

参数输入方式：

1. 所有文件使用相同参数
2. 每个文件单独设置
3. 混合模式（逐项决定）

可控制：

- 活性物质质量
- 计算基准 k=1 / k=4
- 扫描速率
- 温度
- 分析循环
- 图例名称

扫描速率支持：

- 每个文件自动读取；
- 所有文件统一手动输入；
- 每个文件单独手动覆盖。

循环支持：

- 每个文件最后一个完整循环；
- 每个文件分别选择循环。

输出：

- 每个文件独立 CV 参数计算；
- 比电容；
- peak / integral 等已有 CV 参数；
- 批量结果表；
- UTF-8-SIG CSV；
- Plotly 多文件叠图；
- Matplotlib publication overlay。

---

# 2. GCD 多文件

支持一次上传多个 GCD TXT。

参数输入方式：

1. 所有文件使用相同参数
2. 每个文件单独设置
3. 混合模式（逐项决定）

可控制：

- 活性物质质量
- 放电电流
- 计算基准 k=1 / k=4
- 温度
- 分析循环
- 图例名称

放电电流支持：

- 每个文件自动从 CHI metadata 读取；
- 所有文件统一手动输入；
- 每个文件独立覆盖。

每个文件独立计算：

- discharge time
- voltage window
- IR drop / discharge analysis 中已有参数
- current density
- device/mass-basis specific capacitance
- displayed k=1 / k=4 specific capacitance
- energy density
- power density

能量/功率仍沿用原 core 逻辑：
即使显示 k=4 单电极等效 Csp，Energy / Power 仍用 k=1 device/mass-basis capacitance。

输出：

- 批量结果表；
- CSV；
- 交互式叠图；
- publication overlay。

---

# 3. EIS 多文件

支持一次上传多个 EIS TXT，并真正批量拟合。

参数输入方式：

1. 所有文件使用相同参数
2. 每个文件单独设置
3. 混合模式（逐项决定）

可控制：

- Equivalent circuit
- residual weighting
- inductive-point exclusion
- fitting frequency range
- temperature
- legend
- color
- line style

支持：

- 同一组温度数据统一使用同一 ECM；
- 不同文件独立 circuit / weighting / range；
- common frequency range；
- 自动 Rs / Rct initial estimate；
- 一键批量 least-squares complex fit；
- 每个文件独立拟合失败不会阻断其他文件；
- fitted parameter summary；
- RMSE；
- relative RMSE；
- CSV export；
- interactive Nyquist raw/fitted overlay。

科学约定不变：

- raw `z_imag_ohm` 不改符号；
- Nyquist 显示 `-Z''`；
- publication multi-Nyquist 默认仍只显示实验数据；
- 不用 semicircle diameter 冒充 fitted Rct。

---

# 4. 容错

旧版列表推导读取多个文件时，只要其中一个文件读取异常，
可能整个批次直接进入 error。

本版本改成：

- 每个文件独立读取；
- 错误文件单独报告；
- 其余合法文件继续分析。

---

# 5. 修改文件

只需要覆盖：

- `screens/cv_screen.py`
- `screens/gcd_screen.py`
- `screens/eis_screen.py`

其他 core / plotting / temperature-prediction 文件没有改动。

完整 ZIP 中保留了用户原项目的其他全部文件，方便整包覆盖。

---

# 6. 建议启动方式

Windows 下如果之前遇到 PyTorch `shm.dll / WinError 127`：

```bat
conda activate electrochem
cd /d C:\Users\12761\electrochem_app
python run_app_torch_first.py
```

如果不涉及 PyTorch 页面，普通：

```bat
python -m streamlit run app.py
```

也可以。
