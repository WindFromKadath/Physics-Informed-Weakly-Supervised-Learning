# COMSOL 场景 B 库正交化实验报告（W1）

> 日期：2026-08-01
> 状态：W1 完成。v1（残差化）拒绝；v2（去 Q/Q²）与 base 打平、可作后续基线；
> theta 不可识别被判定为场景结构性问题，W2（theta 冻结）升级为主配置建议
> 结果目录：`results/heat_v1_residualize/`、`results/heat_v2_no_qq/`（对照 `results/heat_default/`）
> 运行日志：`runs/heat_v1_residualize/`、`runs/heat_v2_no_qq/`
> 上游文档：`COMSOL场景迁移后续工作计划.md`（W1 定义）、`../COMSOL场景PWL迁移实验报告.md`（首版）

---

## 1. 实验目的与变体

首版 theta 估计 90% 撞边界，归因为 H 的 theta 组（theta·[1,Q,Q²,Q/h]）与
B 的 Q/Q² 列空间重叠（d 吸收 k 标定偏差）。本实验用两个变体检验该归因
并尝试恢复 theta 可识别性：

| 变体 | 做法 | 配置 |
|---|---|---|
| v1 | B 对 H 列空间做 Plumlee 残差化（fit 时冻结投影，任意 x 可求值） | `configs/heat_v1_residualize.yaml`（`feature_spec: heat_residualized`） |
| v2 | B 直接移除 Q、Q² 列，保留 6 列 [1, P, 1/P, P^-1/2, Q/P, Q·P^-1/2] | `configs/heat_v2_no_qq.yaml`（`feature_spec: heat_no_qq`） |
| base | 首版完整 8 列 B，不动 | `results/heat_default/`（已有） |

三次运行规模相同：10 批 × 12 档 × 36 候选，exit code 均为 0。

## 2. 实现要点（代码）

- `features.py`：`FeatureSpec.orthogonalize_b` 字段；`PWLFeatureLibrary.fit`
  冻结投影矩阵 M 与 theta_reference，`transform_b` 对任意输入应用投影。
- `heat.py`：`get_heat_feature_spec` 注册 `heat` / `heat_residualized` /
  `heat_no_qq` 三个变体；`heat_experiments.py` 锚点按 spec 读取 B 列名。
- 两处数值修复（实现中发现）：
  1. **lstsq 列归一化**：原始 Q² 列量级 1e10，lstsq 的 rcond 截断按最大
     奇异值计算会丢弃小尺度方向，投影失真；列归一化后投影不变式成立；
  2. **近零残差列置零**：完全被 H 张成的 B 列（截距、Q、Q²）残差化后只剩
     ~1e-10 数值噪声，标准化会把噪声放大成单位方差垃圾列（实测交叉项
     115.8）；fit 时检测残差范数 ≤1e-8×原列范数的列并显式置零，
     transform 同步置零。
- 正交性验证（`tests/test_heat.py::test_residualized_b_is_orthogonal_to_h`）：
  标准化后 B⊥ 对任意 theta 与 H 的交叉项 ≤ 7.5e-13（H 形状空间 theta
  不变 + H 含截距 ⟹ 对角标准化保持正交，证明见代码注释）。
- 全套 37 项测试通过（含仿真回归，默认路径零行为变化）。

## 3. 结果对比

### 3.1 PWL 测试 RMSE（10 批均值，K）

| n_labeled | base | v1（残差化） | v2（no_qq） |
|---:|---:|---:|---:|
| 10 | **9.04** | 11.20 | 9.27 |
| 20 | 7.69 | 13.07（非单调异常） | **7.62** |
| 30 | **6.93** | 7.84 | 7.04 |
| 60 | **6.36** | 6.76 | 6.42 |
| 100 | 6.12 | **6.05** | **6.02** |
| 120 | 5.91 | **5.81** | 5.90 |

### 3.2 锚点判定

| 锚点 | base | v1 | v2 |
|---|---|---|---|
| 收敛率 / 无效候选 | 1.0 / 0.25% | 1.0 / 0.56% | 1.0 / 0.44% |
| 小样本优势 | ✅ +1.01 | ❌ −1.14 | ✅ +0.78（对 GP p=0.036） |
| theta 学习判定（≥0.5） | 0.2 ❌ | 0.4 ❌ | 0.2 ❌ |
| theta 撞界率 | 0.90 | **1.00** | 0.80 |
| MSE/σ² @120 | 1.74 | 1.68 | 1.73 |
| 优于 PhysicsDirect | ✅ | ✅ | ✅ |
| 优于 Physics-GP | ❌ 差 0.38 | ❌ 差 0.28 | ❌ 差 0.37 |

### 3.3 theta 行为

- v1：10/10 批全部钉在**上界** 1/15（k=15，最大热阻）；
- v2：8/10 批钉在**下界** 1/35（与 base 相同方向）；
- base：8/10 批下界、1 批上界。

### 3.4 决定性对照实验（式(12) 纯标定）

绕过 PWL，直接用式 (12) 对每批 84 个训练标签做 L-BFGS-B 标定
（无 B 项、无联合优化）：**10/10 批 theta 全部钉在上界 k=15**，
包括 k_true=34.5（真值贴下界）的批次。

## 4. 归因：theta 不可识别是场景结构性问题

H/B 重叠确实存在（首版 §5.2），但本实验证明它只是次要因素：

1. **物理不可识别**：k 与 R_c 串联进入输出，温度测量只能识别总热阻；
   R_c ∈ [0.005, 0.049] m²K/W 是半杆导热热阻 L/(2k) ∈ [0.0014, 0.0033]
   的 2–30 倍——主导误差项与 k 正交且量级大一个数量级；
2. **盒约束过窄**：纯 k 补偿 R_c 需要等效 k ≈ 2–3 W/(m·K)，远在
   [15,35] 盒外，任何优化器都只能把 theta 钉在盒边（式(12) 标定证实）；
3. **重叠只决定退化方向**：d 能吸收 k 误差时（base/v2），theta 沿惩罚
   梯度滑向下界；d 不能吸收时（v1），theta 被迫补偿 R_c 而撞上界。
   两个方向的"学习判定通过"批次均为真值贴边的巧合。

v1 的额外代价：B⊥ 失去 k 误差吸收通道后，7–14 个训练点时模型失稳
（@20 RMSE 13.07 非单调），小样本优势（论文核心趋势）被摧毁。

## 5. 结论

1. **v1（残差化）拒绝**：theta 依然撞界且小样本优势消失；
2. **v2（no_qq）与 base 全面打平**：小样本优势保持（+0.78）、120 档
   RMSE 持平、特征库无重叠（更干净的归因叙事）——**作为后续实验的
   配置基线**（等价性能 + 无列空间重叠）；
3. **theta 的校准角色在本场景不成立**：W2（theta 冻结 1/k_nominal）
   从消融项升级为**主配置建议**；锚点 4 的判据（theta_hat 逼近
   1/k_true）在本场景物理上不可达，需修订为"theta 冻结 + 等效热阻
   归因"（见后续工作计划增补）。

## 6. 对后续工作的影响（已同步至工作计划）

- W2 升级为主配置决策（冻结 theta）；
- W3/W4/W5 的基线配置改为 v2 + 冻结 theta（W2 运行确认后定案）；
- 工作计划增补 W9：锚点 4 判据修订与等效热阻归因分析
  （theta_hat 与"1/k_true + R_c 均值效应"复合值的一致性检验）。

## 7. 运行日志摘要

```text
v1: [PWL tune] seed=2035 valid=35/36 selected={λ_physics: 0.01, ...}
    Wrote 1080 metric rows and 216000 predictions (exit 0)
v2: [PWL tune] seed=2035 valid=36/36 selected={λ_physics: 10.0, ...}
    Wrote 1080 metric rows and 216000 predictions (exit 0)
```

完整日志见 `runs/heat_v1_residualize/stdout.log`（240 条件）、
`runs/heat_v2_no_qq/stdout.log`（240 条件）；stderr 为 joblib 进度，
另有 sklearn GP 核参数收敛警告（既有现象，与基线行为一致）。
