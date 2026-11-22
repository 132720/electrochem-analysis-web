# EIS-only 页面更新说明

- EIS 页面仅解析 `<温度>/EIS/*.txt` 路径中的电化学阻抗谱，CV、GCD 文件无条件忽略。
- ZIP 可以是只含 EIS 的数据包，也可以包含其他文件夹；只有 EIS 数据被解析。每个 EIS 文件的上一级目录须为 EIS，其上一级目录须为数值温度（例如 `实验/25/EIS/01.txt`）。
- 完全删除 EIS 页面的 GCD 文件解析、EIS–GCD 温度关联/相关性计算与图表、GCD 相关导出文件及字段。
- 保留 EIS 原始 Nyquist/Bode、KK 筛查、候选模型、最终 L+Rs+CPE 参数拟合、频率特征、论文 v3.3 独立 LinKK 与 CPE/TL 对照。
- Arrhenius 只依赖 EIS 温度数据，温度不足三组时跳过，不影响 EIS 主计算。
- CV/GCD 独立功能及导航文件未作修改；论文同源 EIS 核心算法未修改。
- 使用项目自带 EIS 数据构造单温度/三温度 ZIP（同时混入 GCD/CV 干扰文件）：读取、KK 筛查、候选模型、最终拟合、EIS-only 数据包导出均通过；仅 GCD 的 ZIP 被正确拒绝。
- 本执行环境缺少 Streamlit，未进行浏览器手工验收。
