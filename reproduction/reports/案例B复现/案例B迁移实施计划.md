# 案例 B（点焊熔核直径）迁移实施计划

> 日期：2026-08-01
> 状态：**已执行完毕**（2026-08-03 注：复现完成，结论见同目录《案例B复现报告》——
> 协议全链路实现，Table V 优势未复现，根因已分层归因；该线当前暂停展开）
> 上游文档：`references/docs/09_案例B物理代理模型解析.md`（三层结构与缺口）、
> `datasets/case_b_spotweld/README.md`（数据契约）、
> `references/docs/07_PWL复现指南.md`（全局缺口）、`references/docs/08_基函数设计逻辑.md`（约束 A/C）
> 定位：论文唯一"仿真 + 真实"融合的真实案例，此前因缺数据列为不可复现；
> 数据已由 SAVE 1.0 官方包补足（35 组仿真 + 120 过程样本，合并方差
> 0.2006 ≈ 论文 0.2），现立项复现。

---

## 1. 论文案例 B 的真实结构（复现目标）

三层结构（09 文档 §1）：

```text
第 1 层：35 组仿真 (x,θ)→y ──拟合──→ 克里金代理 η(x,θ)     [真实数据不参与]
第 2 层：120 过程样本 ──式(12)──→ θ̂（仅初始化，代理不重训）
第 3 层：n 真实标记（10/100）+ N 代理弱标签 η(x,当前θ) ──→ BCD 联合优化 g,d,θ
```

关键事实：

- θ = tuning（接触电阻校准参数），盒约束 **[0.8, 8.0]**，初值参考 4.0
  （SAVE 官方示例）；θ 作为**第 4 维输入**与 (load, current, thickness)
  一起进克里金（SAVE 数据结构证实）；
- **x^pr = ∅**：三个过程变量全部进入仿真，B 退化为 B(x^ph)，正交化必需；
- thickness 二值（1/2），高阶项无独立信息；
- 协议：10 / 100 标记两场景 × 5 次随机划分，报均值±标准差；
- 锚点：MSE@100 ≈ **0.206**（噪声方差 0.2 为理论下限）、RMSE@10 ≈ 0.5566、
  Physics 基线全场最差 ≈ 0.90/0.83；
- 论文未给：克里金超参（MLE/DiceKriging 对齐）、弱标签 N 与输入设计
  （取 N=120 LHS + 敏感性）、校准诊断（按案例 A 口径补报）。

## 2. 当前实现与真实结构的差距（改造项总览）

| # | 差距 | 现状（位置） | 严重度 |
|---|---|---|---|
| G1 | 弱标签不支持随当前 θ 重估 | `model.py:174-177` 弱标签在 BCD 前计算一次，全程固定；physics_model 仅用于初始化与 phi | **核心**——这是论文相对两步法的核心机制 |
| G2 | 无克里金代理适配器 | `physics_model` 只有显式函数（simulation/heat 闭式） | 高 |
| G3 | x^pr = ∅ 未验证 | `FeatureSpec` 支持任意维度但 (n,0) 路径无测试 | 中 |
| G4 | 无案例 B 特征库 | 现有 simulation/heat 两个 spec | 中 |
| G5 | 协议差异：随机划分 × 两场景 | 现有驱动只有嵌套扩展（IV-A） | 中 |
| G6 | 校准诊断缺失（论文缺口 B5） | 无 LOO/校准质量报告机制 | 低（新增审计项） |

正交化机制（W1，`orthogonalize_b`）已具备，直接复用，不计入改造。

## 3. 改造清单（文件级）

### M1：`model.py` — 弱标签滞后重估（G1）

- `PWLRegressor.__init__` 新增 `refresh_weak_labels: bool = False`
  （默认固定，仿真/热传导行为不变）；
- 开启时 `fit` 要求提供 `physics_model`；**每轮 BCD 迭代进入块更新前**：
  `weak_y^(k) = physics_model(weak_x_ph, theta^(k−1))`，经既有 phi 映射与
  y 标准化后送入本轮 g/θ 更新与 `_objective`；
- 滞后方案保证命题 1/2 的闭式 prox 不变（块内目标固定）；θ 收敛后
  weak_y 自然稳定，沿用现有组合收敛判据（目标 + 参数 + 双 ADMM）；
- `PhysicsGPRegressor` 基线**不动**（两步法本来就是对照组）；
- 审计：history 记录 weak 标签漂移量供检查（加字段为增量兼容）。

### M2：`case_b.py`（新模块）— 代理、特征库、数据适配（G2/G3/G4）

- `CaseBSurrogate`：普通克里金（常数趋势 + 高斯协方差、各向异性
  length-scale、MLE）。用 sklearn `GaussianProcessRegressor`
  （ConstantKernel × 各向异性 RBF + 固定微 nugget 实现插值，
  `normalize_y=True`，输入 z-score）复刻 DiceKriging 设定；
  方法 `fit(spotweldmodel.csv)`、`predict(x_ph, theta)`（θ 拼为第 4 维）、
  `loo_diagnostics()`（35 点 LOO RMSE/最大误差/各维 length-scale 记录，
  补足缺口 B5）；
- H 库（仿射 θ、4 组，[INFERRED] 草案）：G_load {1, x1, x1²}、
  G_current {x2, x2², x1·x2}、G_thickness {x3}（二值仅线性）、
  G_theta {θ, θ·x1, θ·x2, θ·x3}，共 11 列；
- B 库（B(x^ph)，7 列草案，正交化必需）：{1, x1, x2, x3, x1x2, x1x3,
  x2x3}——与 H 重叠的列由 W1 的自动置零机制处理；
- `case_b_feature_spec()`：`x_ph_dim=3, x_pr_dim=0, theta_dim=1,
  orthogonalize_b=True`，注册名 `case_b`；
- `CaseBScenarioData`：实现 `ScenarioData` 协议；`x_pr` 为 (n,0)；
  `theta_true` 记 θ̂（真实案例无真值，显式标注）；`noise_sigma` 由
  12 设置重复样本合并方差估计（≈√0.2006）；
  `physics_model(x_ph, theta)` → 代理求值；`labeled_physics_correlation`
  → (η̂, y) 相关性（校准诊断，缺口 B5）；
- 弱标签：LHS N=120（敏感性 {60, 120, 300} 后置），采样域限**真实实验
  范围**（load [4.0,5.3]、current [21,29]、thickness ∈ {1,2}），初始
  weak_y 取 θ̂ 处代理值。

### M3：`case_b_experiments.py`（新驱动）— 协议与锚点（G5/G6）

- `run_case_b_experiment(config)`：场景 {10, 100} × 5 次随机划分；
  代理全划分共享（拟合一次）；标记集内 70/30 嵌套训练/验证
  （[INFERRED]，论文未给案例 B 调参协议）；测试集 = 其余过程样本；
- 基线：Ridge/SVR/DT/RF/GBDT/GP/Physics（同现有机器）；
  半监督基线（COREG/ICT/Π/Mean Teacher）属 Table VI，**不在本期范围**；
- 锚点检查（并入 quality_checks）：
  1. MSE@100 与噪声方差 0.2 的比值（目标 ≈1，论文 0.206）；
  2. RMSE@10 与论文 0.5566 的偏差（趋势对照，不作硬门槛）；
  3. PWL ≥ 全部基线（两场景），Physics 为最差或次差；
  4. 收敛率 / 无效候选率 / 弱标签漂移收敛（G1 新增诊断）；
  5. 代理 LOO 质量门槛（插值可信）。

### M4：x_pr_dim=0 管道验证（G3）

- 输入校验、`baselines.column_stack`、B raw 对 (n,0) 的处理、predict
  路径——以测试锁定。

### M5：配置（新建 2 个）

- `configs/case_b_smoke.yaml`：1 划分、两场景、小网格（2×1×1）、n_jobs 1；
- `configs/case_b_default.yaml`：5 划分、两场景、网格 4×3×3、n_jobs 4；
- 公共段：`feature_spec: case_b`、`theta_lower: [0.8]`、
  `theta_upper: [8.0]`、`refresh_weak_labels: true`、
  `mapping: identity`（ϕ 恒等，09 文档 §4.4）、`d_ridge` 沿用 10（敏感性后置）。

### M6：测试（新建 `tests/test_case_b.py`）

1. 数据契约：35/120 形状、列名、12 设置 × 10 重复、合并方差 ≈ 0.2006；
2. 代理：LOO RMSE 门槛（插值质量）、确定性（同点重复预测一致）、
   θ 第 4 维参与（θ 变化输出变化）；
3. 特征库：H 11 列 4 组、θ 仿射 ≤1e-8、thickness 无高阶项、
   spec 维度 (3,0,1)、正交化开启；
4. x_pr_dim=0：fit/predict/baseline 全路径；
5. 弱标签重估：构造合成代理，验证 weak_y 随 θ 移动且收敛后稳定、
   关闭时行为与现状一致（回归）；
6. 协议：划分无泄漏（train/test 不相交、尺寸正确）、种子可复现；
7. 端到端：单划分两场景拟合，预测有限、θ ∈ [0.8, 8.0]。

### M7：运行与报告

- smoke → default（规模极小：35 点代理 + 120 样本，单次秒级）；
- 报告《案例B复现报告.md》（本子文件夹）：锚点对照（MSE@100 vs 0.206、
  RMSE@10 vs 0.5566、Physics 垫底）、校准诊断、缺口 B1–B6 的处置记录、
  与论文 Table V 的逐项对照。

## 4. 实施顺序

```text
M1（refresh_weak_labels）→ M2（case_b.py）→ M4/M6（管道与测试）
→ M3（驱动）→ M5（配置）→ smoke → default → M7（报告）
```

M1 先行是因为它是唯一动共享代码的改造；M2 之后全部独立。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| sklearn GPR 与 DiceKriging 的 MLE 结果有差异 | 记录 length-scale 与 LOO 诊断；差异仅影响代理质量，锚点 5 把关 |
| 弱标签重估导致 BCD 目标移动、收敛变慢 | 滞后方案的θ稳定后目标自稳；监控弱标签漂移诊断，必要时放宽 bcd_tolerance |
| x^pr=∅ 暴露 (n,0) 边界 bug | M4 测试先行 |
| 10 标签 + 70/30 后训练仅 7 点，调参方差大 | 沿用 physics_one_standard_error；必要时并入论文口径报告 |
| 设置级泄漏（同设置的重复样本分处 train/test） | 论文未说明划分层级；按样本级随机（论文口径），另做设置级敏感性对照 |
| 噪声方差 0.2 下 MSE@100=0.206 逼近下限，实现微小差异都会显形 | 先复现协议再调数值；差距按缺口清单归因，不硬凑 |

## 6. 交付物清单

| 类型 | 内容 |
|---|---|
| 代码 | `model.py`（refresh 模式）、`case_b.py`、`case_b_experiments.py`、cli `--scenario case_b` |
| 配置 | `case_b_smoke.yaml`、`case_b_default.yaml` |
| 测试 | `tests/test_case_b.py`（现有 37 项保持全绿） |
| 结果 | `results/case_b_smoke/`、`results/case_b_default/` |
| 报告 | `reports/案例B复现/案例B复现报告.md`（含 Table V 对照） |
| 文档 | 本计划；`REPRODUCTION.md` 真实案例边界一节更新（案例 B 从"不可复现"改为"已复现，口径见报告"） |

## 7. 与既有工作的衔接

- 复用：FeatureSpec 注册表（阶段 0）、`orthogonalize_b`（W1）、
  ScenarioData 接口与调参/统计/落盘机器、Physics-GP 基线；
- 不动：仿真与热传导两条线的任何行为（refresh 默认关闭、新 spec 独立注册）；
- 热传导线 W2–W9 与案例 B 线**并行**，互不阻塞；案例 B 的
  "弱标签随 θ 重估"机制若验证有益，可作为热传导线的候选对照（热传导
  弱标签语义是固定观测，不改默认）。

## 8. 案例 B → COMSOL 迁移映射（对照实验 W10）

案例 B 的三层结构映射到 COMSOL 热传导场景时，第 1 层退化、第 2 层小改、
第 3 层为核心改造：

| 案例 B（论文真实结构） | COMSOL 热传导场景 | 改动 |
|---|---|---|
| 第 1 层：35 组仿真 → 克里金代理 η(x,θ) | **退化**：低保真闭式解 `low_fidelity_temperature(x_ph, theta)` 本身就是免费、精确的 (x,θ)→y "代理" | 零改动，接口签名一致 |
| 第 2 层：120 field 样本 → 式(12) 标定 θ̂ | 已有 `calibrate_theta` | 小改：只在训练标签上标定（防测试泄漏） |
| 第 3 层：弱标签 = 代理在**当前 θ** 的求值，随 BCD 逐轮刷新 | 当前为 `engineering_predictions.csv`（k=25 固定输出） | **核心改造**：`weak_y^(k) = low_fidelity(weak_x_ph, θ^(k))`，复用 M1 的 `refresh_weak_labels` |

迁移后的 COMSOL 数据流：

```text
高保真 BVP（k_true 逐批）──→ 参考测量 120（标签）+ 验收集 200（测试）   [不动]
低保真闭式解 low_fidelity(x,θ) ── "代理"角色
        ├─ 训练标签 ──式(12)──→ θ̂（初始化；训练域标定）
        └─ 弱标签：200 个 engineering 输入点（复用，覆盖性已验收）
           weak_y^(k) = low_fidelity(weak_x_ph, θ^(k))   ← 逐轮刷新（M1）
```

**预期与科学价值**：刷新模式下弱项从"锚定 nominal-k"退化为"物理分量与
物理模型族的一致性约束"（H 对任意 θ 都能拟合低保真模型，h_weak_r²=0.997），
θ 识别压力全部落到标签拟合项；W1 已证明该路径在本场景结构性撞界。
因此本对照的预期产出**不是性能改善**，而是机制对齐证据：撞界成为论文机制
在该场景下的真实行为（场景属性），而非实现选择（实现偏差）。与案例 B
"θ 校准有效"的对照（ANSYS 中 θ 有真实作用通道）构成 PWL 适用边界的实证。

**θ 处理三臂对照**（COMSOL 线）：固定弱标签（现状 base/v2）｜
冻结 θ（W2）｜刷新弱标签（W10，本项）。W2 与 W10 联合写入
《消融实验报告》的 theta 对照章节。

**配置**：`migration/configs/heat_refresh.yaml`（v2 基线 +
`refresh_weak_labels: true` + θ̂ 训练域标定）；M1 实现后边际成本为零。
