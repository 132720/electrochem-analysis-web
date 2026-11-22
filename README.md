# 超级电容器电化学分析平台 — v19

本项目提供三个独立的电化学分析工作区：**CV、GCD、EIS**，以及一个单独的 **GCD 温度预测**页面。

## GCD 温度预测模型
**仅使用 PCA-Ridge。** 已从用户提供的“大论文第二部分项目代码.zip”中核对并分离出阶段一的 PCA-Ridge 实现。曲线 PCA 及岭回归超参数与原始代码一致。

- 输入：电解液浓度（1、2、4 mol/L；分别训练）、目标温度 (°C)、电流密度 (A/g)。
- 训练数据：内置用户项目 `testwendu3` 中的 GCD TXT（69 条），或通过页面上传同结构 ZIP 替换。
- 训练条件：严格使用低于目标温度的实验记录，至少两个不同的较低温度；对目标条件有实测数据时，将其作为未参与训练的验证数据。
- 输出：PCA-Ridge 预测的完整 GCD 曲线及总周期、曲线 CSV、训练文件清单、JSON 配置、可选的实测对照误差。
- 不再提供 PCA-GPR、PG-BDL、神经网络、Random Forest 或其他温度预测模式。
- 本模型是对目标温度的预测；若超出训练数据范围，不能将其理解为已经实验验证。

## 运行
在项目所在文件夹的 Anaconda Prompt 中执行：
```bat
conda activate electrochem_pro
python -m streamlit run app.py
```
或者双击 `start_electrochem_app.bat`。

原有 CV b 值、Dunn、GCD 倍率和 EIS 拟合与 Origin CSV 导出保留。

## 详细说明
参见 `V19_PCA_RIDGE_ONLY_GUIDE.md`。
