# EIS 智能导入 / 专业分析（本次升级）

## 如何导入
在原 EIS 页面中直接上传多个 TXT / CSV / TSV / XLSX 文件，或一个/多个 ZIP。
不需要先组织成固定的 `25/EIS/01.txt` 目录。
- 优先检测传统 CHI 原始 TXT 表头（保留已有解析算法）。
- 对带有标准列名的 CSV/TXT/XLSX，自动识别频率、实部、虚部，并支持显式 `-Zimag` / `-Z''` 符号转换。
- 不能识别列名时，页面允许指定频率/实部/虚部列及虚部符号；如果没有明确温度标签，必须由用户输入温度，不自动假设 25°C。
- 从名称 `EIS_25C.txt`、`temp35.xlsx`、`25/EIS/1.txt` 或明显的头部温度元数据读取温度。
- 明确属于 GCD、CV 的文件会被忽略；无法确定的文件在“未导入文件”里逐条提示。
- 所有原始数据保留在内存，不修改原始上传文件。输入限制：每个压缩包最多500个文件，单文件30 MB，总解压EIS数据160 MB。

## 需要安装的依赖
已有基础分析依赖请参照项目的 requirements_v3.txt 和 requirements_eis_temperature.txt。
- XLSX：`python -m pip install openpyxl`
- 额外仪器解析（BioLogic .mpt、Gamry .dta、Ivium .ids/.idf）及 Z-HIT/DRT/自定义电路：
  `python -m pip install -r requirements_eis_professional_optional.txt`

## 专业功能
在**同一个 EIS 页面**的“专业 EIS 分析”折叠栏中：
- Z-HIT （pyimpspec），显示拟合频率、阻抗与统计结果。
- DRT (TR-NNLS) （pyimpspec），显示/导出弛豫时间分布。需检验数据范围与正则化稳定性。
- CDC 表达式等效电路 CNLS 拟合（pyimpspec），统计、参数和预测谱线分开保存。

这些分析是对论文 v3.3 原有分析的**可选拓展**，不会覆盖论文 v3.3 的拟合参数，也不能未经外部验证宣称与 DearEIS 功能及数值完全一致。
专业后端没安装时，页面会说明安装方法，原 EIS 模型、KK、论文复核不受影响。

## 验证情况
- 使用项目自身 EIS CHI 示例，直接 TXT 读取通过。
- 使用生成的含正确列头的 CSV 和 XLSX，自动转换 `-Zimag` 符号通过。
- 用混有 CV/GCD/EIS 的 3 个温度目录 ZIP 测试，只解析 3 条 EIS；原有 EIS 主流程、候选模型、最终拟合、KK 和温度 Arrhenius 计算通过。
- 全部 Python 源文件语法检查通过。
- **尚未在本环境中测试**：真实 BioLogic/Gamry 文件、实际 pyimpspec Z-HIT/DRT/自定义电路执行、Streamlit 浏览器交互（当前执行环境未安装 pyimpspec 或 Streamlit）。

## 不变部分
`app.py` 主导航、CV/GCD 页面和论文 `core/paper_stage33_eis.py` / `core/paper_stage33_gcd.py` 源码不修改，不恢复 EIS-GCD 联合分析。
