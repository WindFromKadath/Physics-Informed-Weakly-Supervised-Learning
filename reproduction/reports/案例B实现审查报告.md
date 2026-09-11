# 案例 B 实现审查报告

> **审查日期**：2026-08-02
> **审查范围**：`reproduction/src/pwl_repro/` 全部核心模块（optimization / model / features / simulation / baselines / case_b / case_b_experiments / experiments 关键路径）+ `configs/case_b_*.yaml` + `results/case_b_default`、`results/case_b_ext_default` 两次实际运行产物
> **审查基准**：论文式 (4)(5)(9)–(12)、补充材料命题 1–3、[07_PWL复现指南](../../references/docs/07_PWL复现指南.md)、[08_基函数设计逻辑](../../references/docs/08_基函数设计逻辑.md)、[09_案例B物理代理模型解析](../../references/docs/09_案例B物理代理模型解析.md)
> **总结论**：核心算法（ADMM / BCD / 近端算子 / θ 仿射设计）实现正确；案例 B 存在 **2 个实质性错误**（1 个实现 bug + 1 个设计错误），均已数值证实并影响实验结果；4 项论文锚点当前未达标。

---

## 1. 确认无误的部分

| 模块 | 审查点 | 结论 |
|---|---|---|
| `optimization.py` | consensus ADMM 的 x/z/u 更新次序 | ✅ 正确（z = mean(local + dual)，u += local − z） |
| `optimization.py` | 对偶残差 √M·ρ·‖z−z_prev‖、停止阈值、自适应 ρ 残差平衡 | ✅ 符合 Boyd 一致性 ADMM 惯例（ρ 调整属论文未公开的工程补全，已标注） |
| `model.py _update_g` | 命题 1 的 4 个近端算子（两个最小二乘型 + 软阈值 + 组收缩） | ✅ 公式正确；内部统一 ½‖·‖² 缩放，与 `_objective` 自洽 |
| `model.py _update_theta` | 命题 2 的 3 个近端算子（含盒约束投影） | ✅ 正确 |
| `features.py affine_prediction_parts` | 用数值差分构造 c + G·θ，化解补充材料"I 为 p 阶单位阵"的维度歧义 | ✅ 正确且比原文记号更稳健 |
| `features.py _Scaler` | 标准化统计量 fit 时冻结 | ✅ 保持 H 对 θ 的仿射性，不破坏闭式 θ 更新 |
| `model.py` BCD 主循环 | g(ADMM) → d(闭式) → θ(ADMM) 次序、初始化（g=0, d=0, θ=校准值） | ✅ 与 Algorithm 1 一致 |
| `simulation.py` | 式 (9)–(11) 公式、SNR=Var(signal)/Var(noise)、θ~U(0,1]、极点拒采策略 | ✅ 与 07 文档 §9 确认公式一致，未公开细节均有审计标注 |
| `case_b.py` 数据契约 | θ∈[0.8, 8.0]（SAVE 先验）、LHS 弱标签采样、thickness 二值处理、数据校验（120/35 行） | ✅ 与 09 文档 §4 一致 |
| `case_b.py CaseBSurrogate` | 普通克里金（常数趋势 + 各向异性 RBF + 固定 1e-10 WhiteKernel 插值）、θ 作第 4 维输入、MLE 超参 | ✅ 对齐 SAVE/DiceKriging 参照实现 |
| `case_b.py` 三层结构 | 代理仅用 35 组仿真训练；θ̂ 仅初始化；弱标签随当前 θ 刷新（lagged refresh） | ✅ 忠实于 09 文档 §1 三层结构 |
| `experiments.py` 参数传递链 | config → `_make_pwl` → `PWLRegressor.fit`（theta 盒约束、refresh_weak_labels、feature_spec） | ✅ 无断点 |

## 2. 发现的错误

### Bug 1（实现错误，严重）：B 正交化在小样本下整体湮灭

**位置**：`features.py` `PWLFeatureLibrary.fit()`（orthogonalize_b 分支，约 326 行）

**问题**：Plumlee 投影的设计矩阵 `h_ref = self._h_raw(b_x_ph, theta_reference)` 只用了 **n_train 个标记点**。当 n_train < H 列数时（10 标记场景：7 行 vs 11/21 列），欠定 lstsq 可以**精确插值任意 B 列**，残差 ≈ 1e-10 → 触发 `b_dropped_` 判定 → B 的 7 列全部被丢弃 → **补偿项 B·d 完全失效（d ≡ 0）**。

**数值证据**（`results/case_b_default/results.csv` 的 details 字段）：

| 场景 | repeat | d_l2_norm |
|---|---|---|
| n=10 | 0–4 全部 | **0.0000**（B 湮灭） |
| n=100 | 0–4 | 0.012–0.082（仅 x1x3、x2x3 两列幸存） |

独立验证脚本（7 行 vs 70 行投影对比）确认：7 行时两种 spec 的 B 全部 dropped；70 行时默认 spec 有 2 列幸存。

**影响**：论文的核心机制之一"差异补偿"在最需要它的小样本场景被静默关闭，是 PWL@10 锚点（RMSE 0.5566）未达标（实测 0.7385）的主要原因之一。

**修复方向**：案例 B 中 B 不依赖 x^pr（x^pr = ∅），投影设计可在**标记 + 弱标签堆叠的 127 行**上计算（`h_x_ph` 已在 fit 签名中）；通用情形下应在投影行数 < H 列数时跳过正交化并显式告警，而不是静默丢弃。

### Bug 2（设计错误）：case_b_ext 的 B 被 H 完全张成

**位置**：`case_b.py` `CASE_B_EXT_H_NAMES` vs `CASE_B_B_NAMES`

**问题**：ext 的 21 列 H 已包含 B 的全部 7 列（1、x1、x2、x3、x1x2、x1x3、x2x3 均可在 H 的列空间中表示）→ 即使投影行数足够，B 在任何样本量下都被正交化湮灭。实测 `case_b_ext_default` 在 n=100 时 5 个 repeat 的 `d_l2_norm` 也全为 0。

**影响**：`case_b_ext` 配置实质上是"无补偿项消融实验"，与其 docstring 宣称的"扩展蒸馏库"目的不符；`case_b_ext_min` 同理。若保留正交化，B 必须包含 H 列空间之外的方向（如 x1²x2、x1x2² 等 H 未携带的交互）；否则应对 ext 场景关闭正交化、靠 λ₂/λ₃ 调节混淆。

## 3. 结果层面：论文锚点核对

| 锚点（期刊 Table V） | 论文值 | `case_b_default` | `case_b_ext_default` | 状态 |
|---|---|---|---|---|
| PWL MSE@100 | **0.206** | 0.3082 | 0.2923 | ⚠️ 偏高约 45% |
| PWL RMSE@10 | **0.5566** | 0.7385 | 0.7484 | ⚠️ 明显偏高 |
| PWL 全场景最优 | 是 | **否**（Ridge 0.5929 @10、DT 0.4666 @100 均优于 PWL） | 同左 | ❌ |
| Physics 基线最差 | 0.9021 / 0.8279 | 0.8205 / **0.4916**（100 标记时优于 PWL） | 同左 | ❌ 协议分歧 |
| 数据一致性（噪声方差） | ≈ 0.2 | 0.2006 | — | ✅ |
| 代理 LOO 诊断 | 论文未报告 | LOO RMSE 0.4769（LOO R²≈0.63） | 同左 | ✅ 补充诊断 |

**偏差归因分析**：

1. **Bug 1 / Bug 2**：补偿项失效或极弱（见 §2）；
2. **θ 弱可辨识**：θ 估计跨 repeat 在 [1.82, 7.48] 剧烈摆动，rep4 两次触上界（7.42–7.48，上界 8.0）；蒸馏 R² 在 −1.01 ~ 0.93 间波动；
3. **验证集过小**：10 标记场景仅 3 个验证点选 36 组 λ，选择方差极大（λ_physics 在 0.01~10 间跳动）；
4. **Physics 基线偏强（协议分歧，非 bug）**：我们的 GP 差异模型用各向异性 RBF + WhiteKernel + 充足校准点，明显强于论文报告的实现（RMSE 0.49 vs 0.83），导致"Physics 最差"锚点结构性不可达。

## 4. 次要观察（非错误）

| # | 观察 | 位置 | 说明 |
|---|---|---|---|
| O1 | θ̂ 用含验证点的全标记池计算并生成初始弱标签 | `case_b.py build_case_b_scenario` | 轻微验证泄漏；refresh 模式下首次 BCD 扫描即被覆盖，影响短暂 |
| O2 | 弱标签输入逐 split 变化（weak_seed + split） | `case_b_experiments.py` | 工程选择；论文未披露，建议固定一组做敏感性对照 |
| O3 | θ 多次触及盒约束上界 | results details（`theta_boundary_hit`） | 弱可辨识症状；可考虑 theta_starts > 1 或 θ 固定消融 |
| O4 | 代理质量中等（LOO R²≈0.63） | `surrogate_loo.json` | 35 点 4 维插值的固有上限；论文未报告该诊断，本实现已补充，属优点 |
| O5 | `pyreadr` 安装于 .venv 但未写入 pyproject.toml | 环境 | 数据提取的一次性依赖，建议写入 dev extras 或在数据集 README 注明 |

## 5. 修复建议（按优先级）

1. **P0 — 修复 Bug 1**：B 正交化投影改用标记+弱标签堆叠设计（案例 B 可行）；通用路径加"行数 < 列数则跳过正交化并告警"守卫；重跑 `case_b_smoke` + `case_b_default` 验证 d_l2 > 0 且 PWL@10 改善；
2. **P1 — 修复 Bug 2**：为 B 增加 H 之外的方向（如 x1²x2、x1x2²、x1²x3），或对 ext spec 关闭正交化；
3. **P2 — θ 稳定化**：`theta_starts: 3–4` 多点初始化，或增加"θ 固定为 θ̂"消融配置；在报告中披露 θ 触界频率；
4. **P3 — 选择方差**：10 标记场景考虑 5 折交叉验证替代单次 70/30 划分选 λ（改动协议，需在报告中标注与论文的差异）；
5. **P4 — 记录协议分歧**：Physics 基线偏强导致"Physics 最差"锚点不可达，写入复现报告的已知差异清单。

## 6. 复核清单（修复后重跑时应验证）

- [ ] n=10 全部 repeat 的 `d_l2_norm > 0`（默认 spec）
- [ ] n=100 ext spec 的 `d_l2_norm > 0`（若扩充 B）或正交化已关闭并注明
- [ ] `weak_distillation_r2` 稳定为正（> 0.5）
- [ ] θ 估计不再系统性触界
- [ ] PWL RMSE@10 向 0.5566 靠拢、MSE@100 向 0.206 靠拢
- [ ] `case_b_pwl_best@*` 锚点状态变化

---

**关联文档**：[09_案例B物理代理模型解析](../../references/docs/09_案例B物理代理模型解析.md)（缺口与配置草案）、[案例B复现报告.md](案例B复现/案例B复现报告.md)、[案例B迁移实施计划.md](案例B复现/案例B迁移实施计划.md)

---

## 7. 修复执行记录（2026-08-02，同日审查后执行）

### 7.1 已实施修改

| 项 | 修改 | 位置 |
|---|---|---|
| Bug 1（P0） | x^pr 为空时正交化投影设计改用标记+弱标签堆叠行；新增 `rows <= rank(H)` 精确插值守卫（触发时 `warnings.warn` 并跳过正交化）；B scaler 统计量改在投影行上估计（中心化不再破坏正交性） | `features.py` `PWLFeatureLibrary.fit` |
| Bug 2（P1） | ext B 库扩充 3 个 H 外方向 `x1^2*x2`、`x1*x2^2`、`x1^2*x3`（均经秩验证；`x3^2` 因 thickness 二值被 H 张成，不采用） | `case_b.py` `CASE_B_EXT_B_NAMES` |
| O1 | θ̂ 校准与初始弱标签改用 train 行（新增 `theta_calibration_indices` 参数），消除验证泄漏 | `case_b.py` / `case_b_experiments.py` |
| P2 | `case_b_default` / `case_b_ext_default` 的 `theta_starts: 3`；新增 `case_b_theta_frozen.yaml` 冻结消融（freeze_theta + 固定校准初值） | `configs/` |
| 审计 | `results.csv` details 新增 `dropped_b_features` 字段，丢弃列不再静默 | `experiments.py` `_fit_pwl_condition` |
| P4 | "Physics 基线偏强属协议分歧"写入复现报告 §6 已知差异 | `案例B复现报告.md` |
| 测试 | 正交性断言改为投影行语义；新增小样本 B 存活、ext B 存活、守卫告警 3 个测试；**pytest 51 项全部通过** | `tests/test_case_b.py` |

### 7.2 验证结果（修复前 → 修复后，5 划分均值）

机制层（修复目标）：

| 指标 | 修复前 | 修复后 default | 修复后 ext |
|---|---|---|---|
| d_l2 @10（min/med） | **0 / 0**（湮灭） | **0.032 / 0.075** | 0.156 / 0.248 |
| d_l2 @100（min/med） | 0.012 / 0.068 | 0.014 / 0.060 | **0 / 0 → 0.171 / 0.322** |
| B 存活列 | 默认 2 列（仅 n=100） | 默认稳定 2 列（x1x3、x2x3） | ext 3 列（新增交互） |
| θ 触界率 @10 | rep4 两次触上界 | **0%**（θ 范围 [1.82,7.48]→[1.24,4.60]） | 20%（1/5 触 8.0） |
| PWL 收敛率 / 无效候选率 | 100% / — | 100% / 18% | 100% / 14% |

锚点层（论文 Table V）：

| 锚点 | 论文 | 修复前 default | 修复后 default | 修复后 ext |
|---|---|---|---|---|
| PWL RMSE@10 | 0.5566 | 0.7385 | 0.7376（基本未动） | 0.8109（变差） |
| PWL MSE@100 | 0.206 | 0.3082（warn） | 0.2976（**pass**，ratio 1.49） | **0.2564（pass，ratio 1.28）** |
| PWL 全场景最优 | 是 | ❌ | ❌（未变） | ❌（未变） |
| Physics 最差 | 是 | ❌ @100 | ❌ @100（协议分歧，已记录 P4） | 同左 |

### 7.3 §6 复核清单逐条核对

- [x] n=10 全部 repeat `d_l2_norm > 0`（默认 spec，min 0.032）
- [x] n=100 ext spec `d_l2_norm > 0`（扩充 B 后 0.171–0.322）
- [ ] `weak_distillation_r2` 稳定为正（> 0.5）——**未达成**：default@10 中位 0.04、default@100 中位 0.29；仅 ext@100 中位 0.64。蒸馏质量仍是主要短板，与代理保真（LOO R²≈0.63）同源
- [x] θ 不再系统性触界（default 多起点后 0%；ext@10 仍有 1/5 触界）
- [~] PWL RMSE@10 未向 0.5566 靠拢（0.7376）；MSE@100 明显向 0.206 靠拢（0.3082→0.2564，锚点 warn→**pass**）
- [ ] `case_b_pwl_best@*` 仍 fail——与修复无关的结构性问题（复现报告 R4：样本级划分下监督侧直达噪声下限）

### 7.4 新证据与遗留

1. **θ 联合更新 + 弱标签刷新（M1）有正贡献**：冻结消融（θ 固定在校准值）
   在 n=10（0.7939 vs 0.7376）与 n=100（0.5503 vs 0.5384）均差于联合更新，
   排除了"θ 块是误差来源"的假设；该消融同时关闭 refresh（两模式互斥），
   贡献无法在二者间进一步分解。此证据与复现报告 R1 的表述存在张力，
   建议复现报告下一轮修订时纳入。
2. **存活的 B 在 7 训练点下增加方差**：ext@10 修复后变差（0.7484→0.8109），
   补偿项只在标记数据充足时（n=100）转化为收益（0.5338→0.5019，历史最优）。
   这与审查预期一致——Bug 修复恢复的是机制而非小样本精度。
3. **锚点未达项的归因维持不变**：θ 弱可辨识（蒸馏 R² 低）、监督侧结构性
   强劲（R4）、Physics 协议分歧（P4 已记录）。剩余候选见复现报告 §6
   （设置级划分、gage=2 子集、代理保真升级）。
4. 修复后产物：`results/case_b_{smoke,default,ext_default,theta_frozen}_postfix/`；
   修复前产物保留在原目录供对照。

---

## 8. 复审意见（2026-08-02，修复后独立复核）

**复审结论：修复实施正确，报告声明与代码、产物一致，予以通过。**

### 8.1 代码复核

| 修复项 | 复核结果 |
|---|---|
| Bug 1：堆叠行投影（x^pr=∅ 时 h_ref 用标记+弱标签 127 行） | ✅ `features.py` 实现正确；`b_ref` 同步在堆叠行求值；x^pr>0 时显式回退标记行路径并注释原因 |
| Bug 1：秩守卫（rows ≤ rank(H) 时 `warnings.warn` 并跳过正交化） | ✅ 独立构造 5 行 × 秩 5 用例实测：告警触发、`b_projection_` 不生成、B 原样保留（不再静默丢弃） |
| B scaler 改在投影行上估计 | ✅ 推理正确：B_perp 在投影行上零均值（H 含截距），中心化不再破坏正交性 |
| Bug 2：ext B 扩充 x1²x2 / x1x2² / x1²x3 | ✅ 三列均在 ext H 列空间外；x3² 排除理由正确（thickness 二值 → x3²=3x3−2 在支撑集上被 {1,x3} 张成） |
| O1：`theta_calibration_indices=train` | ✅ θ̂ 与初始弱标签不再接触验证行 |
| P2：`theta_starts: 3` + `case_b_theta_frozen.yaml` | ✅ 配置到位；冻结消融与 refresh 互斥处理正确 |

### 8.2 产物复核（报告 §7.2 声明 vs 实测重算）

| 声明 | 复核 |
|---|---|
| default d_l2@10 min/med = 0.032/0.075 | ✅ 实测一致 |
| ext d_l2@100 min/med = 0.171/0.322 | ✅ 实测一致 |
| default RMSE@10 = 0.7376、MSE@100 = 0.2976（pass, ratio 1.488） | ✅ 实测一致 |
| ext RMSE@10 = 0.8109、MSE@100 = 0.2564（pass, ratio 1.282）、RMSE@100 = 0.5019 历史最优 | ✅ 实测一致 |
| 冻结消融 0.7939 / 0.5503（均差于联合更新） | ✅ 实测一致 |
| θ 范围 default [1.24, 4.60]（0% 触界）；ext@10 1/5 触 8.0 | ✅ 实测一致 |
| pytest 51 项全部通过 | ✅ 复跑确认（51 passed） |

### 8.3 遗留事项（非本次修复范围，维持 §7.3–7.4 结论）

1. `weak_distillation_r2` 未达标（default@10 中位 0.038），蒸馏质量与代理保真（LOO R²≈0.63）同源，是 RMSE@10 锚点（0.5566）未靠拢的主要原因；
2. `case_b_pwl_best@*` 仍 fail，属样本级划分下监督侧结构性强劲（复现报告 R4），候选方向为设置级划分 / gage=2 子集 / 代理保真升级；
3. x^pr>0 场景（仿真、heat）的正交化仍只走标记行回退路径——有秩守卫兜底告警，但小样本下正交化会被跳过（机制变为"不投影"），相关场景重跑时需留意警告；
4. ext@10 精度变差（0.7484→0.8109）已归因（7 训练点下存活 B 增方差），建议在复现报告正文保留该权衡说明。
