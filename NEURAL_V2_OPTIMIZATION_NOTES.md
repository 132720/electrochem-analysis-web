# Neural V2 代码逻辑优化说明

## 旧版问题 → V2 修复

| 旧版问题 | V2 |
|---|---|
| ANN/PINN/DeepONet 各自预测 duration | 三者共享相同 physical scalar heads |
| 直接预测绝对 V | 只学习 normalized branch shape |
| 起点终点靠 loss 学 | hard boundary function 数学保证 |
| IR drop 混在 discharge shape 中 | transition / IR drop 单独预测 |
| Logistic/Tanh 容易外推饱和 | ReLU / SiLU 条件通道 |
| 全局 voltage min/max sigmoid | V0 / Vpeak / Vend 独立预测后重构 |
| smoothness 会磨平 transition | transition 先分离，再轻量 curvature |
| 单 seed | 2–3 seed ensemble |
| torch import exception 被吞掉 | 返回真实 import / DLL 错误 |

## Hard-boundary representation

Charge:

S_c(τ)=τ+τ(1-τ)g_c(T,J,τ)

Discharge:

S_d(τ)=1-τ+τ(1-τ)g_d(T,J,τ)

因此无论神经网络输出如何：

S_c(0)=0
S_c(1)=1
S_d(0)=1
S_d(1)=0

## 为什么这是更公平的模型比较

如果 ANN、PINN、DeepONet 使用不同 duration model，
最终 full-curve R² 同时混入：

- shape error
- duration error
- endpoint error
- transition error

V2 统一 physical heads 后，三者主要比较其 branch shape learning 能力。

这使论文中“PCA vs ANN vs PINN vs DeepONet”的比较更可解释。

## 仍然需要真实实验验证

代码优化不能保证某一个模型必然超过 PCA。

如果优化后 DeepONet 仍低于 PCA，
这反而可以支持：

“在当前实验规模下，低维正则化模型比高容量神经 operator model
具有更好的 bias–variance trade-off。”

但只有在 V2 优化完成后，这个结论才更有说服力。
