# COMSOL 场景迁移后续工作计划

> 日期：2026-08-01
> 状态：工作计划 v1.3（W1 已完成；W2/W10 已完成——theta 三臂对照见
> 《COMSOL场景消融实验报告》第一章，**主配置基线修订为 v2+刷新弱标签**；
> W9 待办；案例 B 线首轮完成，见《案例B复现报告》）
> 上游文档：`../COMSOL场景PWL迁移实验报告.md`（首版结果与锚点判定）、
> `../COMSOL迁移检查清单.md`（§5 最小实验集、§6 Go/No-Go）、
> `../COMSOL三维热传导场景迁移分析.md`（§8 验证锚点、§11 开放问题）
> 范围：迁移首版（阶段 0–5）之后的全部后续工作，按四个波次排序；
> 每项工作给出目标、依据、实施细节、验收标准与产物清单。

---

## 1. 首版结论回顾（后续工作的出发点）

首版（`results/heat_default/`，10 批 × 12 档 × 36 候选）锚点判定：

- ✅ 小样本优势复现：10 标签 PWL 全场最优（9.04 K），对最佳监督 +1.0 K；
- ✅ 噪声下限：120 标签 MSE/σ² = 1.74；d 系数物理一致性 1.0；
- ❌ theta 不可识别：90% 边界命中，根因是 **H 的 theta 组（theta·[1,Q,Q²,Q/h]）
  与 B 的 Q/Q² 列空间重叠**，d 吸收 k 标定偏差（§5.2 已归因）；
- ⚠️ 大样本端 Physics-GP（5.53）反超 PWL（5.91）：8 维线性 B 表达力不足。

后续工作围绕两个失败锚点展开修复（W1、W4），并补全检查清单 §5 的
最小实验集（W2、W3、W6）与协议扩展（W5、W7、W8）。

---

## 2. 波次一：theta 可识别性修复（最高优先级）

### W1：B 库正交化实验 ✅ 已完成（2026-08-01）

**结论**（详见《COMSOL场景B库正交化实验报告》）：

- v1（残差化）**拒绝**：theta 全部撞上界、小样本优势被摧毁（@10 RMSE
  11.20 vs base 9.04）；
- v2（去 Q/Q² 列）与 base **全面打平**且特征库无重叠，**作为后续实验的
  配置基线**；
- 式(12) 纯标定对照实验证明 **theta 不可识别是场景结构性问题**（k 与
  R_c 串联、R_c 量级大导热热阻一个数量级、等效 k 在盒外），重叠仅决定
  退化方向。

**对计划的修订**：W2 升级为主配置建议；增补 W9（锚点 4 判据修订）。

~~**目标**：解除 H/B 列空间重叠，恢复 theta 可识别性（锚点 4 与边界命中率）。~~

**依据**：首版 §5.2 归因；迁移分析 §6.3 预警（Plumlee 2017 原则）；
`references/docs/08_基函数设计逻辑.md` 的正交化约束。

**实施**（两个变体 + base 对照）：

| 变体 | 做法 | 代码改动 |
|---|---|---|
| v1 残差化 | B⊥ = B − H_ref·M，M = (H_refᵀH_ref)⁻¹H_refᵀB_ref 在 fit 时冻结（H_ref 取 theta_reference 处的 H 原始列，与 `_Scaler` 冻结统计量同一哲学；transform 时按同一 M 变换，可对任意 x 求值） | `FeatureSpec` 增加 `orthogonalize_b: bool = False`；`PWLFeatureLibrary.fit` 计算并保存 M，`transform_b` 应用；`heat_feature_spec(orthogonalize_b=True)` 变体 |
| v2 去重叠列 | B 移除 Q、Q² 列，保留 6 列 `[1, P, 1/P, P^-1/2, Q/P, Q·P^-1/2]`（Q/P 与 Q·P^-1/2 含 P，不与 H 的 theta·Q 重叠） | `heat.py` 增加 `heat_b_raw_no_qq` 与 spec 变体 `heat_no_qq` 注册 |

**注意**：B 截距列与 H 截距列的共线是框架固有（仿真库同样存在），
y 标准化后影响可忽略，不在本变体处理。

**运行**：`configs/heat_v1_residualize.yaml`、`configs/heat_v2_no_qq.yaml`
（其余同 heat_default.yaml），各一次 default 规模运行（约 15 分钟/次）。

**验收**：

- theta 边界命中率从 0.90 显著下降（目标 ≤ 0.3）；
- theta_learning_fraction 从 0.2 上升（目标 ≥ 0.5）；
- 120 档 PWL RMSE 不劣于 base 的 5.91（允许持平，识别性改善不应牺牲精度）；
- 新增测试：残差化后 B⊥ 与 H_ref 列空间最大典型相关 < 1e-8；
  theta 仿射性测试不受影响（B 不参与 theta 更新的仿射分解）。

**产物**：`results/heat_v1_residualize/`、`results/heat_v2_no_qq/`、
对比分析写入《B 库正交化实验报告》（见 §6 文件清单）。

### W2：theta 冻结消融 ✅ 已完成（2026-08-01）

**结论**（详见《COMSOL场景消融实验报告》第一章）：frozen(v2) 与 v2
全尺寸打平（Δ ≤ 0.17 RMSE），撞界率 0.8 → 0.0。原拟据此定案主配置；
**W10 刷新臂全面更优，主配置改定为 v2+刷新**，冻结臂保留为审计对照。

### W9（新增）：锚点 4 判据修订与等效热阻归因

**目标**：把"theta_hat 逼近 1/k_true"修订为符合本场景物理的判据。

**依据**：k 与 R_c 串联，y 只能识别总热阻；theta_hat 的物理含义是
**等效热阻率**（k_true 与 R_c 均值效应的复合），而非 1/k_true。

**实施**：推导等效热阻率参考值 theta_eff = 1/k_true + α·E[R_c(P)]·(分流系数)
的解析形式（基于一维参考 §3.5 热阻网络）；在冻结配置下改为报告
"theta_cal（式(12)）与 theta_eff 的一致性"以及"等效热阻中 R_c 贡献占比"。
同步修订 `heat_experiments._heat_anchor_checks` 的 anchor 4 与
迁移分析文档 §8 锚点 4 的表述（标注版本与理由）。

**验收**：锚点 4 新判据在 v2/冻结配置上通过或给出可解释的偏差；
文档修订与代码判据一致。

**产物**：并入《消融实验报告》锚点章节；迁移分析 §8 加注。

### W10（新增）：案例 B 式弱标签刷新对照 ✅ 已完成（2026-08-01）

**结论**（详见《COMSOL场景消融实验报告》第一章）：

- 刷新臂**全尺寸最优**（@10 RMSE 8.37 vs 固定 9.27 / 冻结 9.23；@120
  5.78 vs 5.90/5.89），小样本优势 +1.69 为三臂最强；
- theta 不再退化钉界（撞界率 0.30，盒内散布），但仍不追踪真值
  （学习判定 0.3）——与 W1 的结构性不可识别结论一致；
- 代价：无效候选率 26%（移动目标致部分候选不收敛），为已知工程负债；
- **主配置基线据此修订为 v2 + 刷新弱标签**（替代 W2 的冻结定案）；
  冻结臂保留为审计对照。

---

## 3. 波次二：消融与标签节省（检查清单 §5 补全）

### W3：三项消融

| 消融 | 做法 | 代码改动 | 回答的问题 |
|---|---|---|---|
| 无 B | spec 变体 `heat_no_b`：B 仅截距列（d 只剩常数项） | `heat.py` 注册变体；配置 `heat_ablate_no_b.yaml` | 差异项是否有效（P 通道学习的唯一载体） |
| 无弱标签 | `lambda_grid.physics: [0.0]`（prox_physics 权重 0 退化为恒等，现有代码路径已支持） | 仅配置 `heat_ablate_no_weak.yaml` | 物理蒸馏是否有效 |
| 弱标签数量曲线 | `load_heat_batch(..., n_weak=N)` 嵌套截断 N ∈ {50, 100, 200}；如需 500 档再用外部管线零成本补生成 | `heat.py` loader 加参数；驱动支持 `heat.n_weak` 列表循环 | 弱标签规模收益是否饱和 |

**运行**：无 B / 无弱标签各一次 default 规模；弱标签数量为 3 档合并一次
驱动运行（复用嵌套划分，每档独立调参）。

**验收**：与 base 对照——无 B 应在 120 档显著退化（P 通道丢失）；无弱标签
应在 10–30 档退化（蒸馏是小样本优势来源）；弱标签曲线给出收益拐点。

**产物**：`results/heat_ablate_*/`，《消融实验报告》。

### W6：IV-C 标签节省分析

**目标**：回答"100 物理弱标签等效多少监督标签"（论文 IV-C 同构）。

**实施**：直接复用 `experiments.derive_label_savings_experiment` 于波次一
重跑后的 IV-A 结果（PWL 固定 30 标签 vs 监督 30–110 标签逐档最优）；
输出配对单侧检验 + TOST（等效界限默认 ±10%）。

**验收**：给出监督交点（或如实报告未交点，参照 REPAIR_REPORT 的口径）。

**产物**：《标签节省分析》（章节并入《消融实验报告》或独立成篇，见 §6）。

---

## 4. 波次三：大样本端表达力（针对 Physics-GP 反超）

### W4：B 库扩展

**目标**：缩小 120 档与 Physics-GP 的 0.38 K 差距（锚点 6 完全通过）。

**依据**：首版 §5.3——标签充足时非参数 GP 的残差学习能力强于 8 维线性 B。

**实施**：在 W1 正交化框架下扩充 B（候选：`P^-0.7` 精确幂次项、`Q·P^-0.7`、
`Q²·P^-α`、`T_inf` 通道交互项）；新增列一律经 H 列空间残差化；
同步做 `d_ridge ∈ {1, 10, 100}` 敏感性，防止扩展引入过拟合。

**验收**：120 档 PWL vs Physics-GP 差距 ≤ 0（或不劣于 GP 的 5.67）；
小样本端不得退化（10 档优势保持）。

**产物**：`results/heat_v3_expanded/`，《B 库扩展报告》。

---

## 5. 波次四：协议扩展与外部依赖

### W5：IV-B 物理精度三档

**目标**：验证 lambda1 自适应趋势（锚点 3）：弱标签变差时验证选择应调低
lambda_physics。

**场景内档位构造**（一维下无网格档可用，改用参数偏差档）：

- 弱标签 = 闭式解在**偏移导热系数** k_used 下的输出；k_used = k_nominal 即
  现状（基线相关约 0.8）；
- 每批每档在独立标定池上搜索 k_used，使标签-弱标签相关达到
  0.85 / 0.70 / 0.50 目标（复用 `_calibrate_accuracy_scales` 的对数网格
  搜索思想，作用域改为 k_used；标定不用正式标签与验收集）；
- 学习者的 `physics_model` 保持闭式函数形式——退化来自弱标签生成参数与
  真值的不匹配，与论文 IV-B 的 `eta + c·delta` 协议同构。

**实施**：`HeatScenarioData.with_weak_conductivity(k_used)`（frozen
dataclass 用 `dataclasses.replace` 重建 weak_y）；驱动
`run_heat_physics_accuracy_experiment`（固定 40 标签档，PWL-H/M/L +
监督基线训练一次）；配置 `heat_iv_b.yaml`。

**验收**：PWL-H < PWL-M < PWL-L 排序成立；三档选中 lambda_physics 均值
随相关性降低而下降（锚点 3 的核心证据）。

**产物**：`results/heat_iv_b/`，《物理精度实验报告》。

### W7：sigma_low = 0.5 K 敏感性档位（外部依赖）

**目标**：检验结论对噪声水平的稳健性（主规格 §4.3 预留档位）。

**阻塞点**：仓库内只有注入 σ ≈ 4.49 K 噪声后的 y_obs，无法去噪重构
y_det；须用仓库外管线（`COMSOL_Link/step6b_generate_1d.py`）以
sigma_low 重新生成 10 批（秒级），再跑 default 规模对照。

### W8：批次扩展与预注册敏感性矩阵

- 批次 10 → 20（外部管线，秒级）：收紧配对检验的标准误，特别是
  PWL vs Physics-GP 的 0.38 K 差距的显著性判定；
- 按复现收口建议，对协方差结构、R0、d_ridge、B 设计做**预注册**敏感性
  矩阵（先冻结假设与判定标准，再运行），避免事后解释。

---

## 6. 交付物与 md 文件清单

### 6.1 本计划已生成

| 文件 | 说明 |
|---|---|
| `reproduction/reports/迁移后续工作/COMSOL场景迁移后续工作计划.md` | 本文档；后续各波次的实验报告统一放入 `reproduction/reports/迁移后续工作/` 子文件夹 |

### 6.2 各工作项完成后应产出

| # | 文件 | 对应工作 | 波次 |
|---|---|---|---|
| 1 | `reproduction/reports/迁移后续工作/COMSOL场景B库正交化实验报告.md` ✅ 已产出 | W1（含 W2 对照证据） | 一 |
| 2 | `reproduction/reports/迁移后续工作/COMSOL场景消融实验报告.md` | W3 三项消融 + W6 标签节省 | 二 |
| 3 | `reproduction/reports/迁移后续工作/COMSOL场景B库扩展报告.md` | W4 | 三 |
| 4 | `reproduction/reports/迁移后续工作/COMSOL场景物理精度实验报告.md` | W5（IV-B） | 四 |
| 5 | `reproduction/reports/COMSOL场景PWL迁移实验报告.md`（更新） | 每波完成后回写锚点状态与结论 | 持续 |
| 6 | `reproduction/reports/COMSOL热传导场景与数据集设计.md`（升 v1.4） | 仅当 W7/W8 触发生产数据再生成 | 四 |

### 6.3 代码与配置产物（非 md，供核对）

| 工作 | 代码改动 | 新增配置 |
|---|---|---|
| W1 ✅ | features.py（残差化）、heat.py（变体注册）、test_heat.py | heat_v1_residualize.yaml、heat_v2_no_qq.yaml |
| W2 | model.py（freeze_theta）、experiments.py（透传）、test_model.py | heat_theta_frozen.yaml |
| W3 | heat.py（no_b 变体、n_weak 截断）、heat_experiments.py（n_weak 循环） | heat_ablate_no_b.yaml、heat_ablate_no_weak.yaml、heat_ablate_weak_counts.yaml |
| W4 | heat.py（扩展 B，v2 基线上加列，不再残差化） | heat_v3_expanded.yaml |
| W5 | heat.py（with_weak_conductivity）、heat_experiments.py（IV-B 驱动） | heat_iv_b.yaml |
| W6 | heat_experiments.py（复用 derive_label_savings_experiment） | 并入波次一配置 |
| W10 | 复用案例 B 线 M1（model.py refresh 模式）；θ̂ 训练域标定 | heat_refresh.yaml |
| 案例 B 线 | model.py（M1）、case_b.py、case_b_experiments.py、cli `--scenario case_b`（详见《案例B迁移实施计划》） | case_b_smoke.yaml、case_b_default.yaml |

---

## 7. 执行顺序与依赖

```text
W1（B 正交化）✅ 完成：v1 拒绝，v2 ≈ base
M1（refresh 机制，案例 B 线共享）✅ 完成
W2（theta 冻结）✅ 完成：与 v2 打平，保留为审计对照
W10（refresh 对照）✅ 完成：全尺寸最优——主配置修订为 v2+刷新
案例 B 线 ✅ 首轮完成（Table V 优势未复现，归因见《案例B复现报告》）；
    后续最有价值对照：设置级划分、gage=2 子集
W9（锚点 4 修订）── 下一项（文档与判据工作，吸收三臂对照结论）
W3（三项消融）── 用主配置（v2+刷新）──> W6（标签节省，复用 IV-A 结果）
W4（B 扩展）── v2 基线上加列，监控刷新臂无效候选率
W5（IV-B）── 用主配置运行
W7/W8 依赖仓库外管线，时间另议
```

---

## 8. 一句话总结

> 首版迁移试点成立；后续按"先修 theta 可识别性（W1/W2）、再补消融证据
> （W3/W6）、然后攻大样本表达力（W4）、最后扩协议（W5/W7/W8）"的次序推进，
> 每项工作都有明确的锚点判定标准与对应报告文件。
