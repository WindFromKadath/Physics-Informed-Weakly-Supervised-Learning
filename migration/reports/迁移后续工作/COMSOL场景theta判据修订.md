# COMSOL 场景 theta 判据修订报告（M2）

> 日期：2026-08-03
> 状态：定稿。锚点 4（theta 学习）判据修订的实验依据、决策与代码同步记录
> 上游文档：《COMSOL三维热传导场景迁移分析.md》§8（锚点 4 原始定义）、
> 《PWL 当前必须项目与实施方案》M2、《COMSOL场景v3唯一可行配置与通过机制.md》§4
> 分析脚本：`migration/verify_theta_effective.py`（可复跑）
> 代码同步：`pwl_migration/heat_experiments._heat_anchor_checks` + 测试
> `migration/tests/test_heat.py::test_anchor_checks_theta_m2_revision`

---

## 1. 问题

原锚点 4 判据为"theta_hat 较 1/k_nominal 更接近 1/k_true，且边界命中率低"。
本场景中 k（材料导热系数）与 R_c(P)（界面接触热阻）**串联**进入输出温度，
温度数据只能约束**总热阻**，不能保证把 theta=1/k 恢复为真实材料参数。
原判据因此在物理上不可达，长期挂 fail/warn 且不提供信息量。

## 2. 实验依据（verify_theta_effective.py，逐批标定）

对每批计算四个量：theta_true = 1/k_true；theta_eff = 无噪声高保真输出上对
低保真闭式解的最优标定（等效热阻口径）；theta_cal = 120 个含噪标签上的
式 (12) 标定（实际可恢复量）；theta_hat = v3 基线 @120 的 PWL 估计。

| 批 | k_true | theta_true | theta_eff | theta_cal | theta_hat (v3) | 撞界 |
|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 18.24 | 0.0548 | 0.1185 | 0.1198 | 0.0667 | 上界 |
| 2 | 23.49 | 0.0426 | 0.1007 | 0.1045 | 0.0483 | 否 |
| 3 | 22.10 | 0.0453 | 0.1088 | 0.1069 | 0.0286 | 下界 |
| 4 | 16.48 | 0.0607 | 0.1309 | 0.1346 | 0.0666 | 上界 |
| 5 | 27.19 | 0.0368 | 0.1030 | 0.1060 | 0.0414 | 否 |
| 6 | 28.22 | 0.0354 | 0.1020 | 0.0979 | 0.0489 | 否 |
| 7 | 34.51 | 0.0290 | 0.0942 | 0.0945 | 0.0667 | 上界 |
| 8 | 16.16 | 0.0619 | 0.1205 | 0.1193 | 0.0310 | 否 |
| 9 | 32.97 | 0.0303 | 0.1024 | 0.1022 | 0.0286 | 下界 |
| 10 | 15.10 | 0.0662 | 0.1313 | 0.1296 | 0.0286 | 下界 |

关键数值：

- **theta_eff 全部出盒**：10/10 批 theta_eff ∈ [0.094, 0.131]，高于盒上界
  1/15 ≈ 0.0667；对应等效导热系数 k_eff = 1/theta_eff ∈ [7.6, 10.6] W/(m·K)，
  不在材料范围 [15, 35] 内。R_c ~ 0.015–0.06 m²K/W 相对材料通路
  L/(2k) ~ 0.002 m²K/W 占主导——"逼近真实 k"在物理上不可达。
- **标定方向一致且可恢复**：theta_cal > theta_true 10/10 批（等效热阻方向
  正确）；|theta_cal − theta_eff| 中位数 0.0018（n=120 噪声底），
  即"数据想标定的值"就是 theta_eff。
- **盒内预测敏感度 ≈ 1.7σ**：theta 在整个盒 [1/35, 1/15] 上移动时，低保真
  预测在验收集上的 RMS 变化为 σ 的 1.71–1.73 倍——theta 在盒内的位置
  对预测有实质影响，不是无关紧要的冗余量。
- **v3 后 theta 不再追逐 theta_eff**：theta_hat 撞界率 0.6，其中 3 批撞
  **下界**（远离 theta_eff 方向）——v3 的两列机理 B 列接管了接触热阻机理，
  theta 的等效热阻角色被 B 部分替代。theta_hat 批间散布 std = 0.016。

## 3. theta 更新通道的净预测贡献（M2.4）

复用既有对照结果（v2/no_qq 特征、1-SE 选参，仅 theta 处理方式不同）：

| 配置 | @10 RMSE | @120 RMSE |
|---|---:|---:|
| heat_theta_frozen（θ 冻结在标定初值） | 9.225 | 5.888 |
| heat_refresh（θ 联合更新 + 弱标签刷新） | 8.365 | 5.784 |
| **θ 更新通道净贡献** | **−0.860 K** | **−0.104 K** |

θ 更新通道的预测收益集中在小样本端（蒸馏锚定作用），大样本端边际收益小。
这与"预测敏感度 1.7σ"一致：theta 位置影响预测，但其值不对应任何可恢复的
材料常数。

## 4. 决策（M2.2 二选一定案）

**theta 解释为 nuisance parameter（弱标签校准与预测辅助参数），等效热阻
theta_eff 作为其背景物理解释，不作为验收目标。** 理由：

1. theta_eff 全部出盒——若把"逼近 theta_eff"定为目标，必须放大盒约束，
   改变 v3 主模型，违背 M2"收口阶段不改变 v3 主模型"的约束；
2. v3 的 B 修复接管接触机理后，theta_hat 不再一致追逐 theta_eff
   （3/10 批反向撞下界），"等效热阻参数"的角色定位与模型实际行为不符；
3. theta 的预测贡献真实存在（refresh 优于 frozen），nuisance 定位
   既不否认其预测价值，也不要求其数值可物理解释。

对外表述口径：**"theta 是弱标签校准与预测辅助参数；其可识别的物理对应物
是总热阻（等效导热系数 k_eff ≈ 7.6–10.6，位于材料盒外），不是材料导热
系数 k 本身。"**

## 5. 新验收判据（代码与文档同步）

锚点 4 的所有 theta 指标改为 **info 级**，不再设置 pass/fail 门禁：

| quality_checks 行 | 内容 | 旧状态 | 新状态 |
|---|---|---|---|
| `heat_theta_learning_fraction` | 比标称更接近真值的比例（溯源保留） | fail | **info**（明确标注"非成功标准"） |
| `heat_theta_boundary_hit_rate` | 撞界率 | warn | **info**（结构性，theta_eff 出盒所致） |
| `heat_theta_batch_spread`（新增） | theta_hat 批间标准差 | — | **info** |
| `heat_theta_prediction_sensitivity`（新增） | 盒内预测 RMS 变化 / σ | — | **info** |

锚点 4 的"判定"改为机制叙述：报告四项指标 + §3 的净贡献量化 +
§2 的 theta_eff 背景；**不再有 fail/warn 挂账**。

## 6. 验收对照（实施方案 M2.3）

- [x] 不再将"接近真实 k"作为唯一成功标准——旧判据降级为 info 级溯源指标；
- [x] 新判据能够解释 v3 的边界命中而不否定其预测结果——撞界是
  theta_eff 出盒的结构性后果，refresh 净贡献 @10 −0.86 K 予以肯定；
- [x] 代码质量检查与文档判据一致——`_heat_anchor_checks` 已同步，
  测试 `test_anchor_checks_theta_m2_revision` 锁定；
- [x] 对外材料明确区分预测参数与可物理解释参数——§4 口径。

## 7. 同步修改清单

| 文件 | 修改 |
|---|---|
| `migration/src/pwl_migration/heat_experiments.py` | 锚点 4 改 info 级 + 两项新检查 |
| `migration/tests/test_heat.py` | 新增 `test_anchor_checks_theta_m2_revision` |
| `migration/verify_theta_effective.py` | 新增分析脚本（本报告 §2 数据来源） |
| `migration/reports/COMSOL三维热传导场景迁移分析.md` | §8 锚点 4 判据更新 |
| `migration/reports/COMSOL场景PWL迁移实验报告.md` | §0 回写锚点 4 终态 |
| `阶段性总结_PWL算法实现框架.md` | 锚点记分卡 θ 行更新 |
