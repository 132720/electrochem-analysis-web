# 专业 EIS 模块：NumPy / Pandas 依赖冲突修复（v8）

## 用户出现的冲突
- Python: `E:\\Anaconda\\python.exe`（Anaconda base）, Python 3.13.5
- `pyimpspec 5.3.0` 需要 `numpy>=2.5` 且 `pandas>=3.0`
- `numba 0.61.0` 要求 `numpy<2.2`
- `streamlit 1.45.1` 要求 `pandas<3.0`
这四个已安装版本无法在**同一个环境**同时满足依赖要求。虽然 EIS API 导入成功、Streamlit 页面可打开，但运行其它模块时仍可能失败。`pip check` 比单纯 import 检查更全面。

## 推荐修复方案：隔离 conda 环境
在 **Anaconda Prompt** 中：
1. 进入 electrochem_app 项目文件夹。
2. 执行 `setup_electrochem_pro_env.bat`。
3. 安装结束会运行 `pip check` 和 `check_eis_professional.py`。
4. 运行 `start_electrochem_app.bat`。

或者手动执行：
```
conda create -n electrochem_pro python=3.13 -y
conda run --no-capture-output -n electrochem_pro python -m pip install -r requirements_eis_pro_compatible.txt
conda run --no-capture-output -n electrochem_pro python -m pip check
conda run --no-capture-output -n electrochem_pro python check_eis_professional.py
conda run --no-capture-output -n electrochem_pro python -m streamlit run app.py
```

新版启动脚本**不再直接在 Anaconda base 中运行 pip**，也不会默默修改原环境。以前错误安装造成的 base 环境冲突并未自动修复；如需要修复 base，应先备份其包列表并独立处理。

## 在新环境中采用的约束
- Python 3.13
- pyimpspec 5.3.0
- NumPy >=2.5,<2.6
- Pandas >=3,<4
- Streamlit >=1.66,<2
- Numba >=0.68.1,<0.69
- numexpr >=2.11
- bottleneck >=1.5

这些是根据包元数据与兼容范围选择的版本约束，**本运行环境未联网完成 Windows/conda 的真实安装测试**。如果 pip 求解出现错误，不要使用 `--no-deps` 强行安装，应保存完整日志处理。

## 安装前的另一条建议
最稳妥是把论文项目和旧 GCD 训练环境与专业 EIS 后端隔离，避免为安装 EIS 插件改变它们的已验证模型运行环境。

## 不变的项目功能
- 通用 EIS：灵活导入、Nyquist/Bode、KK 类检查、原有电路拟合、Z-HIT、DRT、CDC。
- 原有 CV、GCD 页面与导航未改。
- 不恢复论文 v3.3 专用页面，也不恢复 EIS-GCD 联合分析。
