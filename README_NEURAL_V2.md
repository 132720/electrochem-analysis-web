# GCD 温度预测 — Neural V2 优化完整版

## 这次为什么要改

旧版 ANN / Physics-Informed NN / PI-BD-DeepONet 的主要问题不是简单的 Epoch 太少，而是任务定义不公平：

- 三个神经模型分别自己预测 duration；
- 直接预测 absolute voltage；
- 起点/终点没有严格边界；
- charge→discharge transition / IR drop 没有单独处理；
- sigmoid / tanh 条件通道在高温外推时容易饱和；
- 小样本条件下单随机种子结果波动大。

因此旧版约 R²≈0.8 不能直接解释成“ANN/PINN/DeepONet 本身只能做到这么差”。

---

# 最终六个 UI 模式

0. PCA + Ridge 小样本强基准
1. **ANN V2（Pozo 路线优化版）**
2. RF 参数化（Ghosh 2025 文献基准）
3. **Physics-Informed NN V2（Tang 路线优化版）**
4. **PI-BD-DeepONet V2（Zheng / Lu 路线优化版）**
5. PG-BDL V2（拟创新主模型）

Ghosh 模式保持文献基准，不为了提高精度而改变其核心参数化路线。

旧版 Pozo / Tang / DeepONet 函数仍完整保留在：
`core/gcd_temperature_literature_models.py`
中，便于之后需要时做 legacy / literature-route 对照。

---

# V2 三个神经模型统一的核心逻辑

## 1. Shared physical heads

三种 V2 神经模型不再各自猜：

- total period
- charge fraction
- initial voltage
- peak voltage
- end voltage
- short transition / IR drop
- transition duration fraction

它们统一使用同一套 PG-BDL scalar subsystem：

`Polynomial RidgeCV + sparse-data GPR`

因此模型比较主要反映“shape learner”的差异，而不是 duration head 谁碰巧外推得更好。

## 2. Branch-decoupled normalized shape

模型只学习：

Charging:
`S_c(τ)`

Discharging:
`S_d(τ)`

而不是直接学习绝对 `V(t)`。

## 3. Hard endpoint constraint

Charging：

`S_c(τ) = τ + τ(1-τ) g(τ)`

所以严格满足：

`S_c(0)=0`
`S_c(1)=1`

Discharging：

`S_d(τ) = 1-τ + τ(1-τ) g(τ)`

所以严格满足：

`S_d(0)=1`
`S_d(1)=0`

这从结构上消除了旧版“实验终点接近 0 V，但网络仍停在约 0.2 V”的问题。

## 4. Transition / IR drop 独立处理

charge→discharge 的短时急剧转折不再要求平滑神经网络直接学习。

网络只学习转折前后的内部 branch shape。

最终完整曲线：

`Charging → Transition / IR drop → Discharging`

## 5. Temperature extrapolation

条件变量：

- Temperature
- log(Current density)

标准化后保留线性项并加入低阶 polynomial interactions。

ANN 使用 ReLU。
PINN / DeepONet 使用 SiLU。

避免旧版 sigmoid/tanh condition bottleneck 在超训练温度范围时快速饱和。

---

# ANN V2

- hidden layers: `(64, 64, 32)`
- ReLU
- Adam
- L2 regularization
- adaptive learning rate
- early stopping
- 默认 3 random seeds ensemble
- hard endpoint
- explicit transition
- optional monotonic projection when training data itself is highly monotonic

Pozo 2018 是 ANN supercapacitor 路线依据；
本实现是针对当前 temperature/current-density full-GCD extrapolation 的优化版，不是逐项复现。

---

# Physics-Informed NN V2

- SiLU
- AdamW
- ReduceLROnPlateau
- gradient clipping
- best-state restoration
- 默认 2 random seeds ensemble
- hard endpoint
- data loss
- measured local-slope supervision
- monotonicity loss
- very light curvature regularization

旧版最大问题之一是 smoothness loss 会同时惩罚真实 charge/discharge transition。
V2 已把 transition 从 branch learning 中分离，所以 smoothness 只作用于内部平滑分支。

Tang 2025 是 physics-informed complete electrochemical curve prediction 的方法思想依据；
原文是锂离子电池 V-Q，不是超级电容温度 GCD。

---

# PI-BD-DeepONet V2

Branch Net：
`Temperature + log(J) polynomial condition encoding`

Trunk Net：
`τ, τ², sin(πτ), cos(πτ)`

输出：
`shape correction g(τ)`

而不是 absolute voltage。

同时具有：

- hard endpoints
- explicit transition
- SiLU
- AdamW
- LR scheduler
- gradient clipping
- default 2-seed ensemble

DeepONet 理论：Lu et al., Nature Machine Intelligence 2021。
电化学 operator-learning 路线：Zheng et al., Journal of Energy Storage 2023。

---

# 覆盖文件

## 新增

`core/gcd_temperature_neural_v2.py`

## 完整覆盖

`screens/temperature_gcd_screen.py`

为了避免版本混乱，压缩包同时附上当前完整：

- `core/gcd_temperature_prediction.py`
- `core/gcd_temperature_literature_models.py`
- `core/gcd_temperature_pg_bdl.py`

`app.py` 不需要修改。

---

# PyTorch 诊断改进

旧代码 `torch_available()` 会吞掉所有 import exception，
所以 DLL 错误也可能显示成“没有安装”。

V2 会显示实际错误，例如：

`OSError: [WinError ...] DLL ...`

请在当前环境确认：

```bat
conda activate electrochem
python -c "import sys; print(sys.executable); import torch; print(torch.__version__)"
python -m streamlit run app.py
```

---

# 推荐使用

先使用完全相同的训练数据分别训练：

1. ANN V2
2. Physics-Informed NN V2
3. PI-BD-DeepONet V2
4. PCA + Ridge
5. PG-BDL V2

再用同一组完全未参与训练的目标温度实验文件验证。

不要根据训练 loss 选论文主模型。

最终必须根据 unseen-temperature：

- RMSE
- MAE
- R²
- NRMSE
- time-period error
- charge-duration error
- discharge-duration error
- transition / endpoint error

决定。

---

# 一个重要的论文写作边界

因为三个 V2 神经模型共享相同的 physical scalar heads，
论文中应把它们描述成：

`controlled shape-learner comparison / ablation`

而不是声称它们是完全独立的端到端 ANN/PINN/DeepONet。

旧版独立 literature-route 实现仍保留在
`gcd_temperature_literature_models.py`
中，需要时可作为额外 literature baseline。
