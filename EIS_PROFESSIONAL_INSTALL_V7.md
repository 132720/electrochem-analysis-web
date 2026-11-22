# v7 专业 EIS 功能安装与启用

本次保持原 EIS 通用分析页面，仅改进 pyimpspec 安装、启动和依赖检查。

## 启用方法（Windows）
1. 首先激活你平时启动 electrochem_app 所使用的 Python/conda 环境。
2. 双击 `start_electrochem_app.bat`。如果缺少 pyimpspec，启动脚本会询问是否安装专业 EIS 依赖。选择 Y 后运行 pip 安装。
3. 也可单独双击 `install_eis_professional.bat`，结束后重新启动界面。
4. 检查：`python check_eis_professional.py`。成功时显示 `PASS`，并打印实际 Python 路径和包版本。

命令行方式：`python -m pip install -r requirements_eis_professional_optional.txt`。

安装时需要互联网及兼容的 Python 环境。pip 会在当前激活的 Python 环境中安装 pyimpspec；压缩包内不预装第三方库，也不能代替在你的电脑上安装。

## 专业功能
- Z-HIT：相位辅助一致性分析。
- DRT (TR-NNLS)：弛豫时间分布；正则化 λ=-1 自动选择，正数为显式指定。
- CDC 自定义电路：等效电路复非线性最小二乘拟合，支持选择方法和权重。

仅在安装检查通过后显示可运行控件。原有 Nyquist/Bode、KK 类筛查、固定电路拟合及数据导出可独立运行。

## 保持不变
- CV 与 GCD 功能、主导航不变。
- 不新增温度诊断、论文 v3.3 对照界面或 EIS–GCD 联合分析。
- 论文算法原文件不动。

## 检验状态
- 检查脚本已在本环境执行：发现 pyimpspec 未安装，因此正确显示失败状态与安装说明。
- 专业分析的三个 Python 接口进行了模拟依赖调用测试。
- 当前环境未安装真实 pyimpspec 和 Streamlit，未进行真实后端计算与浏览器验收。
- 实际能否安装取决于你的系统 Python 版本、网络与依赖冲突；成功安装后建议分别运行 Z-HIT、DRT 和 CDC 拟合验证。
