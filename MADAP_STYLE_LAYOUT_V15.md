# v15 MADAP 风格界面（Figure 2 布局参考）

## 设计依据
参考 Rahmanian 等发表于 *Scientific Data* 10, 43 (2023) 的 MADAP **Figure 2**：
https://www.nature.com/articles/s41597-023-01936-3

以及 MADAP 官方使用文档：
https://fuzhanrahmanian.github.io/MADAP/

**注意：这是布局结构的参考实现，不是原始 GUI 的逐像素复刻。** MADAP 使用 PySimpleGUI；我们的项目继续保留已在 Windows 能运行的 Streamlit 和原有科研计算内核。没有引入 MADAP 的计算代码，也没有复制其品牌/官方图像。

## 新界面结构
- **顶部横向导航**：首页 / 电化学分析界面 / GCD 温度预测 / EIS 阻抗分析。
- **左侧 Data & Analysis Settings**：
  1. Data Path：上传 CHI TXT / CSV / TSV / XLSX / ZIP。
  2. Result Name：填写浏览器下载的 ZIP 文件名（不写入本地硬盘指定目录）。
  3. Data Selection：文件识别、数据点、列名和单位映射；非 CHI 文件必须确认。
  4. Methods：CV / GCD / EIS 各自参数、电路与选取方式，保留不同文件参数设置。
  5. RUN ANALYSIS：批量分析，左侧显示成功数与错误状态。
- **右侧 Analysis Workspace**：像原 MADAP EIS 界面一样采用 2×2 科研诊断区域，按 EIS/CV/GCD 数据自动切换。
  - EIS：原始 Nyquist、Bode 模值/相位、拟合 Nyquist、拟合相对残差。
  - CV：CV 曲线、CV 细节、峰候选位置、比电容对扫描速率。
  - GCD：充放电曲线、有效放电段、比电容与电流密度、IR drop。
  - 右侧附有只读参数表、**可编辑论文图例和备注**。
  - 修改论文图例只修改图上显示文本，不覆盖分析记录或数值。
- **工作台下方 Full analysis, provenance & export**：原有 CV/GCD/EIS 结果表、质量检查、Origin ZIP、b/Dunn、GCD 倍率功能保留。

## 论文图与 SCI 可追溯性
- 页面图型以诊断预览为主，不声称绘图输出直接满足某个期刊标准。
- 原始数值和分析参数不可通过“编辑论文图例”篡改。
- EIS 无拟合时不假造拟合曲线或残差，明确提示必须先 RUN。
- 所有 scientific algorithms 和数据导出计算仍由 v14 内核控制。

## 启动
在解压后包含 `app.py` 的 `electrochem_app` 目录，执行：
```
conda activate electrochem_pro
python -m streamlit run app.py
```
或双击现有 `start_electrochem_app.bat`。不要求安装 PySimpleGUI 或 MADAP。

## 试用建议
1. 进入“电化学分析界面”。
2. 上传 v14 提供的示例 ZIP 或你的真实 CV/GCD/EIS。
3. 在左侧确认列映射、扫描速率、质量、电流、电路等条件。
4. 点击 RUN ANALYSIS。
5. 在右侧选 EIS：检查 Nyquist、Bode、拟合 Nyquist 和残差四图同屏。
6. 展开图例备注进行编辑，查看曲线实时更新；展开下方完整分析与 Origin 数据导出。

## 限制
- Windows Streamlit 浏览器交互需要你本机继续验收。
- v15 重构的是**主要的“电化学分析界面”**；GCD 温度预测与 EIS 专业工具仍沿用原有布局，避免改变已有复杂模块的运行逻辑。
- 不支持从网页直接指定 Windows 本地结果保存目录；`Result Name` 只控制导出 ZIP 文件名，下载目录由浏览器决定。
- 单纯拷贝 MADAP 的视觉布局不会改变 CV/GCD/EIS 数值拟合方法，真实论文结论仍须结合实验和模型核验。
