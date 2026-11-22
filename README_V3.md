# GCD 温度预测 V3 — 完整优化包

## 这次 V3 是针对真实验证结果做的结构优化

用户 70 °C / 0.5 A/g 实验验证中：

- PCA + Ridge: R² ≈ 0.980925, RMSE ≈ 9.94 mV
- ANN V2: R² ≈ 0.291008, RMSE ≈ 60.57 mV
- Physics-Informed NN V2: R² ≈ -0.100313, RMSE ≈ 75.46 mV
- PI-BD-DeepONet V2: R² ≈ -0.028048, RMSE ≈ 72.94 mV
- Ghosh RF: R² ≈ -0.506683, RMSE ≈ 88.30 mV

V2 neural models的核心问题不是单纯 epoch 不够，而是 sharp transient 的表示方式不合适。

旧表示：

S = base + τ(1-τ) * 3*tanh(raw/3)

存在固定 correction 上限。对快速充电起始段和放电末端陡降，这个参数化会直接限制模型可表达的曲线形状。

---

# V3 六个模式

0. PCA + Ridge small-data baseline
1. ANN V3 — Warped residual ANN
2. Ghosh 2025 RF literature baseline
3. Physics-Guided NN V3
4. Latent PI-BD-DeepONet V3
5. PG-BDL V3 — 推荐拟创新模型

---

# V3 核心改动

## 1. Conditioned monotonic time warp

每个 branch 使用：

xi(tau) = 1 - (1 - tau^a)^b

其中 a,b 由 Temperature + log(Current density) 预测。

这可以表示：

- 快速早期充电 transient
- 主体平缓区
- 慢前期 + 快速末端放电

而不需要让神经网络在 tau≈0 或 tau≈1 附近学习巨大 correction。

warp 参数训练采用 gradient-aware weighted fitting，使陡峭区域虽然占点数少，也不会在拟合中被忽略。

---

## 2. 完全去除旧版 ±3 correction cap

V3 神经 residual 不再使用：

3*tanh(raw/3)

PGNN V3 的 hard endpoint 写成：

residual = xi(1-xi) * N_theta

因此：

residual(0)=0
residual(1)=0

但 residual 内部幅值不再被人工限制在 ±3。

---

## 3. Timing 从两段改成三段独立预测

V2 主要依赖：

total duration + charge fraction

V3 改为独立预测：

- charge_duration_s
- transition_duration_s
- main_discharge_duration_s

最后：

total = charge + transition + main discharge

这直接针对 PCA 验证中最明显的 charge→discharge transition timing error。

---

## 4. PG-BDL V3

完整结构：

Temperature + log(J)
        ↓
scalar physical heads
        ↓
charge/discharge time warp
        ↓
warped-coordinate residual
        ↓
PCA latent representation
        ↓
Polynomial RidgeCV + extrapolation-aware GPR
        ↓
physical reconstruction

预测头包括：

- charge duration
- transition duration
- main discharge duration
- V0
- Vpeak
- Vend
- transition drop
- charge warp a,b
- discharge warp a,b

这是 V3 推荐的主模型。

---

## 5. ANN V3

ANN 不再直接预测 absolute voltage。

ANN 学习：

Temperature + log(J) + multiscale warped-coordinate features
            ↓
warped residual

coordinate features：

- xi
- xi²
- sin(pi xi), cos(pi xi)
- sin(2pi xi), cos(2pi xi)
- sin(4pi xi), cos(4pi xi)
- sin(8pi xi), cos(8pi xi)

使用：

- ReLU
- Adam
- L2 regularization
- early stopping
- multi-seed ensemble

并通过最高温度 blocked validation 自动决定 ANN prediction 与 PG-BDL V3 的 blend weight。

如果 ANN 在 blocked temperature 上不能真正改善 PG-BDL V3，权重会自动收缩到 0。

---

## 6. Physics-Guided NN V3

该模式当前更准确地称为 Physics-Guided Neural Network，而不是严格 governing-equation PINN。

它使用：

- conditioned monotonic time warp
- hard residual endpoints
- gradient-aware sample weighting
- adaptive monotonicity loss
- adaptive smoothness loss
- multiscale Fourier coordinate features
- SiLU
- AdamW
- ReduceLROnPlateau
- gradient clipping
- best-state restoration
- multi-seed ensemble
- blocked-temperature validation gate

V3 不再使用固定 physics loss 权重。

---

## 7. Latent PI-BD-DeepONet V3

考虑到当前输入只有 Temperature + constant current density，V3 不把它描述为原始 DeepONet 的逐项复现。

结构：

warped residual field
        ↓
PCA encoder
        ↓
low-dimensional latent coefficients
        ↓
PyTorch condition network
        ↓
predicted latent coefficients
        ↓
fixed PCA decoder
        ↓
residual field
        ↓
physical reconstruction

这是 latent neural-operator-inspired small-data adaptation。

同样使用 blocked-temperature validation gate。

---

# 文献设计依据

1. Ghosh et al., Journal of Energy Storage 113 (2025) 115562.
   Temperature-dependent supercapacitor ML baseline.
   DOI: 10.1016/j.est.2025.115562

2. Zhang, Su, Yuan, Journal of Computational Physics 541 (2025) 114338.
   Coordinate-transformed PINN for sharp boundary-layer gradients.
   DOI: 10.1016/j.jcp.2025.114338

3. Kontolati et al., Nature Communications 15 (2024) 5101.
   Learning nonlinear operators in latent spaces / L-DeepONet.
   DOI: 10.1038/s41467-024-49411-w

4. Fogelquist & Lin, Applied Energy 399 (2025) 126458.
   Sparse-data GPR residual correction combined with electrochemical modeling.
   DOI: 10.1016/j.apenergy.2025.126458

V3 是上述思想针对 temperature/current-density supercapacitor GCD 的组合与适配，
不是任何一篇论文的直接复制。

---

# 需要覆盖 / 新增的文件

## 新增核心

core/gcd_temperature_v3_models.py

## 完整覆盖

screens/temperature_gcd_screen.py

## 同时提供兼容依赖完整版

core/gcd_temperature_prediction.py
core/gcd_temperature_literature_models.py
core/gcd_temperature_pg_bdl.py
core/gcd_temperature_neural_v2.py

其中 gcd_temperature_neural_v2.py 仅保留旧模型兼容/追溯用途；
V3 界面不再调用它。

---

# PyTorch Windows 启动

如果普通 Streamlit 启动再次出现 shm.dll / WinError 127，
使用：

python run_app_torch_first.py

它会：

import torch
↓
再 import / 启动 Streamlit

避免 Windows DLL load-order 冲突。

---

# 推荐论文验证方式

使用完全相同的低温训练集，最高温度完全不参与训练：

- PCA + Ridge
- ANN V3
- Ghosh RF
- Physics-Guided NN V3
- Latent PI-BD-DeepONet V3
- PG-BDL V3

统一报告：

- RMSE
- MAE
- R²
- NRMSE
- maximum absolute error
- total period error
- charge duration error
- transition timing error
- discharge duration error
- endpoint error

不要只根据训练 loss 选择模型。

---

# 重要科学边界

V3 synthetic smoke test 只用于确认程序结构能够表达 fast-start / fast-tail curve，并确认代码可收敛。
它不是用户真实 70 °C 数据的性能结果。

真实论文结论必须以新的 unseen-temperature CSV validation 为准。
