# EIS v5：专业功能与可复现数值对照

## 与 v4 的区别
- EIS 现有入口内部增加“专业 EIS 分析与数值基准验证”，**不新增主导航**。
- 即使没有 pyimpspec，也可以选择一条谱线运行论文 v3.3 `L + Rs + CPE` 数值基准，并显示 RMSE。
- 安装 pyimpspec 后支持 Z-HIT、TR-NNLS DRT、自定义 CDC 电路；新增最优化方法（auto/leastsq/least_squares）、权重（auto/modulus/proportional/unity/boukamp）和 TR-NNLS 的 λ。
- 自定义等效电路拟合完成后，可显式运行同一实测谱线的论文 v3.3 模型比较：
  - 同频率交集（**不插值**，至少6个频点）
  - 复数阻抗 RMSE（Ω）
  - 相对复数阻抗 RMSE、MAE、最大误差（%）
  - 完整逐频残差表
  - ZIP 打包：原始实测谱线、模型汇总、逐点误差、参数表、统计量、JSON 计算设置及 Python/后端版本
- 对照信息明确声明：**不同电路结构或拟合权重时，参数不具备直接可比性**。拟合残差更小也不意味着机理更正确。

## 与 DearEIS 的关系
本软件仅在安装 pyimpspec 时调用其 API。DearEIS 使用了 pyimpspec 的部分核心算法，但**本项目并未直接运行 DearEIS GUI，也尚未完成跨软件同条件的量化验收**。不能写成“已完全达到 DearEIS 的功能和数值精度”。

## 安装
```bash
python -m pip install -r requirements_eis_professional_optional.txt
```
支持 `pyimpspec>=5.0,<6`，建议在现有 electrochem conda 环境中安装。

## 验证
- `python -m unittest discover -s tests -p "test_eis_v5_benchmark.py" -v`
- 测试使用项目自带的 `data/eis_test.txt`（84点）及合成同结构输入。
- 论文模型的复阻抗 RMSE 和统一计算结果校验成功；完全相同的预测值应返回相同误差。
- 单独使用模拟 pyimpspec 接口完成三项可选分析的调用契约测试，**并非真实 pyimpspec 数值验证**。
- pyimpspec、Streamlit 在此环境未安装，专业功能与页面交互仍需在实际电脑上验收。

## 原则
- 保留论文 v3.3 算法原文件，不使用新后端覆盖论文拟合。
- 保留 CV 与 GCD 原页面，不恢复 EIS-GCD 联合分析。
- 保留 v4 路径无关的 EIS 直接上传和列映射能力。
