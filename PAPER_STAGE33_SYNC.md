# electrochem_app × paper-stage3.3 同源同步说明

## 改动范围
本包仍是单独运行的 `electrochem_app`，不需要论文项目目录。未修改 `app.py` 导航、CV 页面或 CV 核心分析代码。

### GCD 页面（原“GCD 温度预测”）
- 原有 PCA-GPR 完整曲线、周期、95% 区间及严格前向温度验证保持可用。论文 v3.3 采用的 PCA-GPR 核心版本与本项目原始 `core/gcd_pca_gpr_final.py` 相同。
- 将论文 `GCD/01_周期预测与模型评价/01_完整周期预测与误差分解.py` 的分段 Ridge 算法函数原样复制到 `core/paper_stage33_gcd.py`，保留相同的固定节点、单调 PCHIP、标准化 Ridge 和周期/曲线重建公式。
- 在原 GCD 预测结果区域的折叠栏中对照显示分段 Ridge 预测、归一化曲线误差、导出 CSV；沿用相同训练/验证温度边界，避免真实目标泄漏。
- 论文的大规模离线批量评估、独立再训练不确定度模型及整套论文图表生成脚本没有在界面中整体重跑。

### EIS 页面（原“温度依赖 EIS”）
- 从论文 `EIS_GCD/02_15工况窗口电容探索/.../final_evidence_audit.py` 直接复用独立 LinKK `linkk_fit` 的原始算法函数到 `core/paper_stage33_eis.py`，包括 mu、RC 分布、频段和权重口径。
- 从论文 `新增分析_v3.3/additional_checks.py` 对应公式、边界、筛选和最小二乘设置移植了 CPE / 阻塞型有限长度传输线对照拟合；补充按实际频率最近点取 10kHz、1Hz、0.1Hz 特征的计算。
- 在现有 EIS 页面折叠栏中运行这些计算，输出表格与 CSV；保留原 EIS 主分析和原联合分析入口。
- LinKK 数值筛查不是物理 KK 通过认证；候选拟合也不能作为唯一机制的证明。
- 论文其他离线跨模态审计及固定窗口电容批量脚本未整体嵌入界面。

## 同源校验
- `paper_stage33_gcd.py` 的 8 个计算函数与论文相应函数逐字相同。
- `paper_stage33_eis.py` 的 `linkk_fit` 与论文相应函数逐字相同。
- EIS CPE/TL 核心公式和拟合边界沿用论文，外部的数据结构适配为本项目 `TemperatureEISRecord`。
- CV、主导航保持原压缩包版本。

## 检查
- Python 语法检查：通过。
- EIS 内置示例谱（84 点）：频率特征、LinKK 与两个候选模型拟合调用均成功。
- GCD 示例：5 份示例记录组成的训练集通过分段 Ridge 训练并输出 401 点预测曲线。
- **未完成的检查**：尚未使用论文的全部原始多浓度数据运行全流程，也未在浏览器中人工逐项验收 Streamlit 页面。具体论文结论请继续以论文结果表为准。

## 运行
按照原项目 `start_electrochem_app.bat` 或 `streamlit run app.py` 运行；需要 numpy、pandas、scipy、scikit-learn、streamlit、plotly 等原项目依赖。

## 2026-11-21 EIS 单温度/双温度兼容修复
- `screens/temperature_eis_screen.py`：温度数小于 3 时跳过探索性 Rs Arrhenius 诊断，不再使正式 EIS 计算整体失败。
- 当 EIS 与 GCD 共同温度少于 3 时，跳过跨模态相关性，保留其他结果。
- 导出汇总中的 Arrhenius 指标改为 NaN、诊断 CSV 在条件不满足时省略；页面显示明确提示。
- 对三个及以上温度，沿用原来的论文 Arrhenius 数学实现和统计口径，不改拟合公式。
- 完成 Python 源码语法检查与温度数量分支检查；当前执行环境未安装 Streamlit，无法完成网页交互验收。

## EIS NoneType 电流密度修复
- 根因定位：`core/eis_temperature_final.py` 的 `gcd_temperature_summary` 直接对可能为 None 的 GCD `current_value` 执行 `float`，从而中断 EIS 主分析。通过项目自带的 `gcd_test.txt` 可复现。
- 仅将含有效正数电流密度、有效充放电时间的 GCD 记录纳入跨模态电荷汇总；不推测缺失电流，EIS 主分析不再因此中断。
- 如共同温度不足三组，EIS–GCD 关联功能提示跳过。
- 与论文同源的 GCD Ridge、EIS LinKK、CPE/TL 原始算法均未修改。
- 已通过单温度 EIS 内置数据的 KK、模型筛选、统一拟合与低频描述符测试，以及 GCD 缺失电流回归测试；浏览器 Streamlit 未在本环境实际执行。

## EIS `float(None)` 深度修复（2026-11-21）
- `gcd_temperature_summary` 对电流、充/放电时间、温度逐一验证：任何缺失或非数值字段均只跳过该 GCD 记录，不对缺失实验数值做填充。
- EIS 正式分析中的 Arrhenius、EIS–GCD 汇总及相关性均为**可选诊断**：这些分支输入不足或数据异常时，仅给出提示，不再中止主 EIS 模型计算。
- 如果 KK、候选电路拟合、最终参数计算等**核心**步骤仍出错，界面会显示具体失败阶段及可展开的完整 Python traceback，便于继续定位；没有吞掉核心错误。
- 使用项目自带 EIS/GCD 样例构造 1 温度、3 温度的完整实验 ZIP；测试通过上传 ZIP 的读取、KK 筛查、候选模型、最终拟合、低频描述和 Arrhenius 条件执行；GCD 测试文件缺少可解析电流密度，关联会被跳过。
- 未修改 paper-stage3.3 移植的 GCD/EIS 核心论文函数；未修改 CV 和导航。
- 当前环境无 Streamlit 依赖，未完成实际浏览器端测试。
