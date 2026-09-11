# 路线 B：在线 MPC 动态 PWL 实施方案

> 版本：v1.0  
> 日期：2026-08-04  
> 状态：架构研究方案，尚未实施  
> 上游准入要求：[加热配方代理模型必备条件](../../../加热配方代理模型必备条件.md)  
> 对照路线：[路线 A：离线加热配方代理模型实施方案](路线A_离线加热配方代理模型实施方案.md)

---

## 0. 执行结论

路线 B 的目标是让代理模型直接替代 MPC 中反复调用的动态状态方程：

\[
\hat x_{k+1}=f(x_k,u_k,d_k),
\qquad
\hat x_{k+1:k+H}=\operatorname{rollout}(x_k,u_{k:k+H-1},d_{k:k+H-1}).
\]

**可行性判断：研究上可行，但当前 PWL 不能直接使用，必须进行架构级升级。**

当前热迁移是静态标量映射，缺少当前状态、时间推进、多输出、滚动训练、UQ、状态估计、
梯度和在线更新。路线 B 应被正式命名为“动态 PWL”或“PWL 状态转移扩展”，不能将其
表述为原始 PWL 论文已经包含的功能。

建议的总体路线是：

```text
高维温度场
   ↓ POD/物理降阶
低维状态 z_k ← 传感器 + 状态估计器
   ↓
动态多保真 PWL：z_k,u_k,d_k → z_(k+1)
   ↓ rollout + Jacobian + UQ + 域诊断
MPC 求解器
   ↓
约束收紧、回退控制与在线校正
```

其中，是否能够构造可观测/可估计的状态 \(z_k\)，是路线 B 的第一个硬门槛。没有状态，
后续动态模型和 MPC 接口都无从成立。

---

## 1. 当前实现与目标模型的差距

| 能力 | 当前热 PWL | 在线 MPC 所需 | 结论 |
|---|---|---|---|
| 输入 | \(Q,h,T_\infty,P\) | \(z_k,u_k,d_k\) | 必须重定义 |
| 输出 | 单一界面平均温度 | 下一步低维状态/约束输出 | 必须多输出 |
| 时间 | 独立静态样本 | 单步递推和多步 rollout | 当前缺失 |
| 训练目标 | 单点监督损失 | 单步 + 多步 + 约束损失 | 当前缺失 |
| 速度 | 静态矩阵计算 | MPC 时域内批量轨迹 | 尚未基准测试 |
| 梯度 | 无公开接口 | \(\partial z_{k+1}/\partial(z_k,u_k)\) | 当前缺失 |
| UQ | 确定性输出 | 时域传播后的置信区间 | 当前缺失 |
| 在线更新 | 离线 BCD/`theta` 刷新 | 受保护的残差/参数更新 | 当前缺失 |
| 域诊断 | 无 | 状态—控制联合信赖域 | 当前缺失 |

当前代码证据见[`heat.py`](../../src/pwl_migration/heat.py)和
[`model.py`](../../../reproduction/src/pwl_repro/core/model.py)。当前 `refresh_weak_labels`
只是在离线训练中随 `theta` 重算弱标签，不等同于运行期 `partial_fit` 或在线系统辨识。

---

## 2. 状态定义与可观测性

### 2.1 不建议直接预测全温度场

如果高保真温度场有上千个节点，直接把所有节点作为 PWL 输出会造成：

- 输出维度和参数矩阵过大；
- 传感器无法提供同维度在线状态；
- 多步误差和物理一致性难以约束；
- MPC 求解时间不可控。

首选做法是把温度场 \(T_k\in\mathbb R^n\) 降为 \(z_k\in\mathbb R^r\)：

\[
T_k\approx \bar T+V_r z_k,
\qquad
z_k=V_r^\top(T_k-\bar T),
\qquad r\ll n.
\]

其中 \(V_r\) 可由 POD/SVD 得到，也可采用少量物理状态，例如表面、中心、关键截面
温度与若干温度梯度。

### 2.2 状态选择门槛

状态必须同时满足：

1. 能重构表面温度、中心温度和断面温差；
2. 在 MPC 时域内具有足够的 Markov 性；
3. 能由在线传感器直接测量或由观测器稳定估计；
4. 在不同初始状态、钢种和控制序列下保持有效；
5. 降阶误差小于代理模型允许误差预算。

仅按 POD 能量累计率选 \(r\) 不够；必须同时检查约束量重构误差和闭环控制性能。

### 2.3 状态估计器是独立模块

如果在线只能测到少量表面温度，必须增加 Kalman、扩展 Kalman、无迹 Kalman或移动
时域估计器，将传感器数据映射为 \(\hat z_k\)。状态估计误差必须进入 UQ 与约束收紧，
不能把仿真中的真实 POD 系数当成现场可直接获得的输入。

---

## 3. 动态 PWL 数学结构

### 3.1 推荐预测增量而不是绝对状态

定义联合输入：

\[
s_k=[z_k,u_k,d_k].
\]

推荐模型为：

\[
\widehat{\Delta z}_k
=H(s_k,\theta)G+B(s_k)D,
\]

\[
\hat z_{k+1}=z_k+\widehat{\Delta z}_k.
\]

- \(H\in\mathbb R^{N\times m}\)：物理主基；
- \(G\in\mathbb R^{m\times r}\)：低维状态各通道的主系数；
- \(B\in\mathbb R^{N\times p}\)：差异基；
- \(D\in\mathbb R^{p\times r}\)：多输出差异系数。

预测增量有三个优点：零时间步时自然接近恒等映射、容易表达能量变化、通常比直接拟合
绝对温度更适合递推。但这只是优先候选，必须用消融比较 `next_state` 与 `delta_state`。

### 3.2 低保真弱标签

低保真动态模型对同一 \((z_k,u_k,d_k)\) 给出：

\[
\Delta z_k^L=F_L(z_k,u_k,d_k,\theta)-z_k.
\]

大量低保真轨迹提供弱标签，少量高保真/实测轨迹提供真实标签。PWL 仍保留原有思想：
主基蒸馏低保真规律，差异基补偿模型遗漏。

### 3.3 与原始 PWL 的边界

原始实现是单输出静态质量预测。将 \(g,d\) 扩成矩阵 \(G,D\) 是自然推广，但下列内容
均属于新增方法：

- 状态递推；
- 多输出参数矩阵；
- 多步 rollout 损失；
- 状态估计；
- 不确定性随时域传播；
- 在线更新和 MPC 安全回退。

论文原有算法不能自动为这些扩展提供收敛、可识别性或闭环稳定性保证。

---

## 4. 动态 `H/B` 基函数设计

### 4.1 `H`：物理动力学主基

候选物理基可包括：

- 当前 POD 状态 \(z_k\) 和低阶交互；
- 控制输入、驻留时间/步长和控制变化率；
- 低保真一步增量 \(\Delta z_k^L\)；
- Fourier 数、Biot 数、辐射或对流换热组合；
- 当前表面—中心温差与驱动力；
- 能量平衡残差；
- 材料规格与时间尺度交互。

每一列都要通过量纲、可计算时点、连续性和仿射 `theta` 检查。若物性随温度强非线性，
可以：

1. 在工作点局部线性化并由 `B` 吸收残差；
2. 固定 `theta`，只在线更新差异系数；
3. 使用外循环估计 `theta`；
4. 重写优化器以支持非仿射参数。

不能在保留当前 ADMM 公式的同时，默认非仿射 `theta` 更新仍然正确。

### 4.2 `B`：动态差异基

`B` 应针对低保真动态模型遗漏的机制，例如：

- 炉况与装料方式；
- 氧化层、接触热阻、积灰和衬里老化；
- 控制输入与状态梯度的交互；
- 传感器偏差和慢变化残差；
- 低保真模型在约束边界附近的系统性误差。

必须检查 `H/B` 列空间混淆；当前仓库已有正交化研究基础，但动态场景要在轨迹组和不同
状态区域重新验证，不能直接复制静态热场景的列。

### 4.3 物理一致性

模型至少应实施以下可测试约束或诊断：

- 温度取值范围；
- 零热输入下的冷却方向；
- 增大炉温/功率时的局部响应方向；
- 能量变化与输入热量的数量级一致性；
- 重构表面温度、中心温度和断面温差的代数一致性。

这些诊断不等同于严格稳定性证明，但能阻止明显非物理解进入 MPC。

---

## 5. 轨迹数据合同

### 5.1 基本记录

建议采用长表或等价的数组制品：

```text
episode_id, batch_id, fidelity, k, timestamp, dt,
state_*, control_*, disturbance_*, next_state_*,
surface_temperature, center_temperature, section_delta,
source_model, source_version, split_group
```

同时保存：

- 原始高维温度快照或可追溯路径；
- POD 基、均值场及版本哈希；
- 控制与扰动时间对齐规则；
- 传感器缺失、滤波与插值标记；
- 每条轨迹的初始状态和终止原因。

### 5.2 轨迹覆盖

训练集必须覆盖：

- 多种初始温度分布，而不仅是统一冷态；
- 正常配方、扰动配方和约束附近配方；
- 控制上升、下降、保持与组合激励；
- 规格、钢种、节奏和环境扰动；
- 低保真模型已知失准的区域。

如果控制输入没有足够激励，状态动力学和控制效应将不可辨识。

### 5.3 数据划分

训练/验证/测试必须按完整 `episode_id`、批次和工况组划分，禁止把同一轨迹相邻时间步
随机分到训练集与测试集，否则单步指标会严重乐观。

应额外保留：

- 长时域 rollout 测试集；
- 约束临界测试集；
- 包络外压力测试集；
- 新批次/时间漂移测试集。

---

## 6. 训练策略与优化器影响

### 6.1 第一阶段：单步矩阵回归

先训练：

\[
\mathcal L_{1}=\sum_k\|\Delta z_k-\widehat{\Delta z}_k\|_W^2
+\lambda_{ph}\mathcal L_{weak}
+\lambda_1\|G\|_1
+\lambda_g\sum_q\|G_q\|_2
+\lambda_D\|D\|_F^2.
\]

权重矩阵 \(W\) 应提高与表面温度、断面温差相关模态的权重。首个 MVP 可以暂时为每个
状态维度训练一个独立 `PWLRegressor`，从而最大化复用现有代码；但它不能共享稀疏结构，
也不能自然表达跨状态协方差，只适合作为架构验证基线。

### 6.2 第二阶段：多步与约束损失

单步通过后再加入：

\[
\mathcal L=\mathcal L_1
+\alpha\sum_{h=2}^{H}w_h\|z_{k+h}-\hat z_{k+h}\|_{W_h}^2
+\beta\mathcal L_{constraint}.
\]

`constraint` 项直接惩罚表面温度、断面温差和出炉温度误差。

**重要影响**：递归 rollout 使损失对 \(G,D,\theta\) 形成跨步非线性耦合，当前 PWL
的分块凸子问题和闭式近端更新不再自动成立。实施时有三种选择：

1. 保持现有单步训练，只把 rollout 作为验收；
2. 交替进行单步 PWL 拟合和多步局部校正；
3. 为动态版本引入自动微分/非线性优化器，并重新研究收敛性。

路线 B 文档不能预先承诺仍然完全沿用原始 BCD–ADMM。

### 6.3 模型选择指标

选参主指标不能再是随机样本 RMSE。建议采用：

\[
\text{score}=a_1E_{1\text{-step}}+a_HE_{H\text{-rollout}}
+a_cE_{constraint}+a_sP_{divergence}.
\]

必须预注册 MPC 时域 \(H\)、约束权重和发散惩罚。当前热场景的 `minimum` 规则可以作为
代码机制参考，但不能不经验证直接成为动态路线默认规则。

---

## 7. 运行时 API

建议建立独立的动态接口，而不是破坏当前静态 `PWLRegressor`：

```python
class DynamicPWLModel:
    def reset(self, state, covariance=None): ...
    def predict_step(self, state, control, disturbance=None): ...
    def rollout(self, state, controls, disturbances=None): ...
    def jacobian(self, state, control, disturbance=None): ...
    def predict_distribution(self, state, controls, disturbances=None): ...
    def in_domain(self, state, control, disturbance=None): ...
    def update(self, transition, *, mode="shadow"): ...
```

接口要求：

- `predict_step` 不修改隐式状态，便于优化器并行调用；
- `rollout` 可批量处理候选控制序列；
- `jacobian` 返回状态和控制 Jacobian，并与有限差分对拍；
- `in_domain` 返回布尔值、距离和原因；
- `update` 默认影子更新，验证通过后才切换生产参数；
- 所有输入输出单位、顺序和缩放器版本写入模型制品。

---

## 8. Jacobian、平滑性与求解器

当前 `H/B` 多为解析连续函数，域内预测值具有建立可微接口的基础，但 NumPy 实现没有
自动微分。可选路径：

1. 为固定基函数手写解析 Jacobian；
2. 用中心有限差分做首期 MPC 原型；
3. 将动态特征实现迁移到 JAX/CasADi 等可微后端；
4. 使用无梯度优化器作为性能基线。

生产路线更推荐解析或自动微分。无论采用哪条路径，都要验证：

- 梯度与有限差分误差；
- 控制边界附近的连续性；
- 标准化/反标准化链式导数；
- 稀疏激活集变化是否只发生在训练阶段，而非运行输入映射中。

训练优化不可微不代表训练后预测映射不可微，两者应分开说明。

---

## 9. UQ、信赖域与安全约束

### 9.1 不确定性来源

至少区分：

- 状态估计不确定性；
- PWL 参数/结构不确定性；
- 低保真模型差异；
- 过程噪声和未测扰动；
- 多步递推累积误差。

首期可用按 `episode_id` Bootstrap 的动态 PWL 集成，逐成员 rollout 后估计每个时域的
状态和约束分布。不能把单步残差标准差原样复制到所有未来时刻。

### 9.2 约束收紧

MPC 中可采用：

\[
\mu_{T,h}+\beta_h\sigma_{T,h}\le T_{\max},
\qquad h=1,\ldots,H.
\]

\(\beta_h\) 和协方差传播方法必须通过闭环仿真校准。高斯假设、独立误差或线性传播若
只是近似，应在安全论证中明确。

### 9.3 状态—控制信赖域

包络诊断应作用于 \((z_k,u_k,d_k)\) 联合空间，并对 rollout 每一步执行。任一步越界时：

- 拒绝该候选控制序列；或
- 增加越界惩罚并要求回到域内；或
- 切换到经过验证的低保真/降阶物理模型；或
- 启用保守回退控制器。

仅在第一步检查输入域不能阻止后续预测状态逐步漂出训练包络。

---

## 10. 在线校正与模型治理

### 10.1 建议先更新差异项

首期不要在线重跑完整 BCD–ADMM。更稳妥的方式是冻结 `H/G` 和 POD 基，只对差异项
或输出偏置做 RLS/Kalman 更新：

\[
D_{k+1}=\operatorname{RLSUpdate}(D_k,B(s_k),e_{k+1}).
\]

原因是 `D` 代表慢变化的模型差异，更新范围更容易限制；而 `theta` 在当前热迁移中不具备
可靠物理解，在线快速更新可能吸收传感器故障或短期扰动。

### 10.2 更新保护

每次更新必须具备：

- 遗忘因子、参数盒约束和最大步长；
- 异常值/传感器故障过滤；
- 激励不足时冻结更新；
- 影子模型与生产模型并行评估；
- 回滚到上一签名版本；
- 更新前后 rollout、约束和闭环回归测试。

“能够在线拟合”不等于“允许直接在线切换”。

### 10.3 漂移监控

运行期监控至少包括：

- 单步创新残差；
- 不同预测时域的误差；
- 约束量偏差和覆盖率；
- 域外查询比例；
- 参数漂移和边界命中；
- 回退控制触发率。

---

## 11. MPC 接入与回退结构

### 11.1 在线循环

```text
读取传感器
   ↓
状态估计 z_hat_k
   ↓
动态 PWL rollout + UQ + 域检查
   ↓
MPC 求解候选控制序列
   ↓
安全过滤/高层约束检查
   ↓
只执行第一个控制量
   ↓
新测量、残差记录、受保护更新
```

### 11.2 回退控制

至少提供一条独立于学习模型的回退路径：

- 已验证的规则配方；
- 保守 PID/现有控制器；
- 低阶物理模型 MPC；
- 安全停车或保持策略。

以下情况触发回退：求解器超时、模型域外、UQ 超阈值、传感器异常、预测约束无可行解或
模型版本自检失败。

### 11.3 时间预算

设控制周期为 \(T_c\)，求解器评估次数为 \(N_{eval}\)，则模型全时域 rollout 的预算应由
\(T_c/N_{eval}\) 反推。必须分别测量：

- 单步延迟；
- 单条长度 \(H\) 的 rollout；
- 批量候选 rollout；
- Jacobian；
- UQ 集成；
- 完整 MPC 求解周期 P50/P95/P99。

当前文档中的“毫秒级”是方向性准入要求，最终阈值必须由真实控制周期和硬件确定。

---

## 12. 建议代码与制品结构

```text
reproduction/src/pwl_repro/core/
├── model.py                         # 保留当前静态 PWL
├── dynamic_model.py                 # 通用动态/矩阵输出 PWL
└── dynamic_features.py              # 动态 FeatureSpec 与 Jacobian 契约

migration/src/pwl_migration/
├── heat_dynamic.py                  # 热状态、低保真转移、动态 H/B
├── heat_state_reduction.py          # POD 编码/重构与状态版本
├── heat_observer.py                 # 传感器到状态估计
├── heat_mpc_adapter.py              # rollout、约束和求解器接口
└── heat_online_update.py            # 受保护 RLS/Kalman 更新

migration/configs/
├── heat_dynamic_smoke.yaml
├── heat_dynamic_one_step_v1.yaml
└── heat_dynamic_mpc_v1.yaml

migration/tests/
├── test_heat_dynamic.py
├── test_heat_rollout.py
├── test_heat_jacobian.py
├── test_heat_observer.py
└── test_heat_mpc_closed_loop.py

migration/results/heat_dynamic_v1/
├── BASELINE.md
├── one_step_metrics.csv
├── rollout_metrics.csv
├── constraint_metrics.csv
├── timing_metrics.csv
├── uq_coverage.csv
└── closed_loop_metrics.csv
```

动态核心只有在至少两个场景共享后才应完全下沉到 `pwl_repro.core`；首个原型可以先放在
`pwl_migration`，待接口稳定再抽象，避免过早把热场景假设写成通用规范。

---

## 13. 分阶段实施计划

| 阶段 | 工作 | 主要交付物 | 硬门槛 |
|---|---|---|---|
| B0 | 冻结 MPC 问题 | 控制周期、时域、状态/输入/约束表 | 用途和时间预算明确 |
| B1 | 建立轨迹数据合同 | 时间对齐轨迹、split、manifest | 无相邻时间步泄漏 |
| B2 | 状态降阶与观测 | POD/物理状态、观测器基线 | 约束量可重构、状态可估计 |
| B3 | 单步动态 PWL MVP | 独立输出或矩阵输出模型 | 单步优于低保真基线 |
| B4 | rollout 验证/训练 | 多时域误差曲线、发散率 | MPC 时域内误差不发散 |
| B5 | Jacobian 与性能 | 梯度对拍、时延基准 | 满足求解器精度和时间预算 |
| B6 | UQ 与信赖域 | 时域覆盖、约束收紧、OOD | 假安全率达到预注册门槛 |
| B7 | 闭环与回退 | 闭环仿真、扰动测试、回退控制 | 约束违反不劣于安全基线 |
| B8 | 在线校正与影子运行 | 更新策略、回滚、漂移报告 | 更新不破坏已验证安全域 |

B2 不通过则停止动态模型开发；B4 不通过则不能进入 MPC；B7 不通过则不能现场影子运行；
B8 即使通过，也仍需要现场工程安全审查才能控制真实设备。

---

## 14. 验收矩阵

### 14.1 模型层

- 单步状态 RMSE/MAE 和约束输出误差；
- 不同 \(H\) 下 rollout error 曲线；
- 最坏轨迹误差、发散率和物理一致性违规数；
- POD 重构误差与观测器估计误差单独核算；
- 表面温度、断面温差、出炉温度误差小于 \(\delta/2\)；
- 域内/边界/域外分层指标；
- 时域预测区间的经验覆盖率。

### 14.2 接口与性能层

- `reset/predict_step/rollout` 确定性和无隐式状态污染；
- 解析/自动梯度与有限差分一致；
- 单步、全时域、批量和 UQ 时延；
- 求解器超时率和无解率；
- 模型制品版本、缩放器、POD 基和配置可追溯。

### 14.3 闭环层

- 跟踪误差、能耗和加热时间；
- 表面过热、断面温差和出炉温度违规率；
- 初始状态偏差、外扰、参数漂移和传感器噪声压力测试；
- 相对原控制器、低阶物理 MPC 和高保真数字孪生的配对比较；
- 域外、超时和异常时回退成功率；
- 在线更新前后闭环性能和安全指标。

闭环验收优先级高于离线 \(R^2\) 或单步 RMSE。

---

## 15. 主要研究风险

| 风险 | 判断 | 应对 |
|---|---|---|
| 状态不可观测 | 致命结构风险 | 先完成观测器和可观测性研究 |
| POD 能量高但约束量失真 | 高风险 | 按约束重构误差选模态 |
| 单步好、rollout 发散 | 高风险 | 多时域验证，必要时加入 rollout 损失 |
| 多步损失破坏原优化结构 | 方法风险 | 明确采用新优化器并重新验证 |
| 多输出独立训练失去耦合 | 中高风险 | MVP 后升级矩阵输出和共享结构 |
| UQ 随时域严重低估 | 高风险 | 逐成员 rollout、按时域校准 |
| 在线更新吸收故障 | 安全风险 | 异常过滤、影子更新、冻结和回滚 |
| MPC 利用代理漏洞 | 安全风险 | 联合信赖域、约束收紧、回退控制 |
| 实时预算不足 | 工程风险 | 早期建立批量/Jacobian时延基线 |

---

## 16. Go / No-Go 判据

### 可以进入动态 PWL 原型

- 已确定 MPC 控制周期、预测时域、控制变量和安全约束；
- 有时间对齐的低/高保真轨迹；
- 当前温度状态可直接测量或稳定估计；
- 低维状态能够重构关键约束量；
- 有独立数字孪生或高保真模型用于闭环验证。

### 应暂停路线 B

- 只有“配方参数—最终质量”的静态表格；
- 缺少当前状态或传感器不足以估计状态；
- 相邻时间步被随机拆分，无法形成可信测试；
- 安全阈值和回退控制尚未定义；
- 控制周期与硬件时间预算未知；
- 期望只修改当前 `H/B` 列就直接上线 MPC。

---

## 17. 参考文献与用途说明

以下文献截至 2026-08-04 已通过论文页面或 DOI 核对。它们分别支持动态降阶、受控
动力学辨识、多保真时序代理和学习型 MPC 的设计选择，但没有任何一篇直接证明本项目的
动态 PWL 已经可用于真实加热设备。

1. Alenezi, D. F., Biehler, M., Shi, J., & Li, J. (2025).
   *Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes*.
   IEEE Transactions on Automation Science and Engineering, 22, 2006–2018.
   [DOI: 10.1109/TASE.2024.3374098](https://doi.org/10.1109/TASE.2024.3374098)。  
   **用途**：动态扩展所继承的物理弱标签、差异项与校准思想；原文不是动态控制模型。

2. Benner, P., Gugercin, S., & Willcox, K. (2015).
   *A Survey of Projection-Based Model Reduction Methods for Parametric Dynamical Systems*.
   SIAM Review, 57(4), 483–531.
   [DOI: 10.1137/130932715](https://doi.org/10.1137/130932715)。  
   **用途**：参数化动力系统降阶、离线—在线分解及 POD/投影方法总览。

3. Shen, J., Singler, J. R., & Zhang, Y. (2019).
   *HDG–POD Reduced Order Model of the Heat Equation*.
   Journal of Computational and Applied Mathematics, 362, 663–679.
   [DOI: 10.1016/j.cam.2018.09.031](https://doi.org/10.1016/j.cam.2018.09.031)。  
   **用途**：热方程 POD 降阶及误差分析的直接领域参考。

4. Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016).
   *Dynamic Mode Decomposition with Control*.
   SIAM Journal on Applied Dynamical Systems, 15(1), 142–161.
   [DOI: 10.1137/15M1013857](https://doi.org/10.1137/15M1013857)。  
   **用途**：从受控高维数据中区分系统动力学与控制作用，可作为动态 PWL 基线。

5. Korda, M., & Mezić, I. (2018).
   *Linear Predictors for Nonlinear Dynamical Systems: Koopman Operator Meets Model Predictive Control*.
   Automatica, 93, 149–160.
   [DOI: 10.1016/j.automatica.2018.03.046](https://doi.org/10.1016/j.automatica.2018.03.046)。  
   **用途**：学习到的低维/升维线性预测器与 MPC 约束结构衔接的参考，也是路线 B 应比较
   的重要基线。

6. Conti, P., Guo, M., Manzoni, A., Frangi, A., Brunton, S. L., & Kutz, J. N. (2024).
   *Multi-fidelity Reduced-order Surrogate Modelling*.
   Proceedings of the Royal Society A, 480(2283), 20230655.
   [DOI: 10.1098/rspa.2023.0655](https://doi.org/10.1098/rspa.2023.0655)。  
   **用途**：POD 低维状态、多保真时序学习和全场重构的近期直接参考；其模型为 LSTM，
   与 PWL 的稀疏基函数结构不同。

7. Venkatraman, A., Hebert, M., & Bagnell, J. A. (2015).
   *Improving Multi-Step Prediction of Learned Time Series Models*.
   AAAI 2015.
   [论文页](https://ojs.aaai.org/index.php/AAAI/article/view/9590)。  
   **用途**：说明仅优化单步损失与递归多步部署之间的分布偏移问题，为 rollout 验收和
   多步训练提供依据。

8. Hewing, L., Kabzan, J., & Zeilinger, M. N. (2020).
   *Cautious Model Predictive Control Using Gaussian Process Regression*.
   IEEE Transactions on Control Systems Technology, 28(6), 2736–2743.
   [DOI: 10.1109/TCST.2019.2949757](https://doi.org/10.1109/TCST.2019.2949757)。  
   **用途**：学习残差不确定性传播、机会约束和谨慎 MPC 的设计参考；PWL 需要额外 UQ
   包装才能采用类似思想。

9. Hewing, L., Wabersich, K. P., Menner, M., & Zeilinger, M. N. (2020).
   *Learning-Based Model Predictive Control: Toward Safe Learning in Control*.
   Annual Review of Control, Robotics, and Autonomous Systems, 3, 269–296.
   [DOI: 10.1146/annurev-control-090419-075625](https://doi.org/10.1146/annurev-control-090419-075625)。  
   **用途**：学习模型、约束安全、在线适应和安全过滤的整体框架参考。

### 仓库内关联材料

- [PWL 论文解析](../../../references/docs/legacy/02_PWL论文解析.md)
- [PWL 基函数设计逻辑](../../../references/docs/08_基函数设计逻辑.md)
- [当前热迁移说明](../../README.md)
- [当前项目架构](../../../docs/ARCHITECTURE.md)

---

## 18. 最终定位

> 路线 B 不是把现有静态 PWL 放进循环调用，而是把 PWL 的物理弱标签与差异校正思想
> 扩展为一个可观测、可递推、可滚动验证、可量化不确定性并带安全回退的动态状态模型。
> 其技术路线成立的前提是先解决状态定义与轨迹数据，再讨论多输出 PWL 和 MPC；单步
> 离线精度只能作为中间指标，闭环约束与回退能力才是最终验收标准。

