# PWL 盲测 S1 预注册（批 21–40）

> 冻结时间：2026-08-05（**先于批 21–40 数据生成**）
> 状态：冻结。生成数据、运行实验、查看结果之前固定本文件全部内容
> 上游：《PWL算法真实能力与信息边界审计.md》§7（盲测协议）、
> 《PWL输出对齐A0-A4小范围实验报告.md》（R4 触发，待独立确认）
> 本文档角色：审计 §7.4 执行顺序的"冻结问题定义、特征来源和比较模型"环节

---

## 1. 目的与待决问题

S1 只回答一个问题：**在从未参与任何诊断与模型设计的新鲜批次上，
A0–A4 小范围实验的关键结论是否仍然成立？** 具体为四个预注册判定
（§6 D1–D4），核心是 D1——"机理对齐回归 A4 优于机制辅助型 PWL"（b5
口径 A4−A0 @120 = −0.621 K）脱离开发数据后是否成立。

S1 不回答"PWL 能否应对未知机理"（那是 S2/S3 的 G0/G1 与机理变化实验）。
同机理新批次上，A0（v3 qint PWL）即审计 §7.2 的 G2 层，A4 即
oracle 灰盒层。

## 2. 数据（生成前固定）

- **批次 21–40**，与批 1–20 同生成器物理（k(T)=k_true(1+0.002(T−300))、
  R_c(P)=0.03P^(-0.7)、σ=4.493844638867134、v1.3 输入域与 LHS、
  120/200/200 规模、种子公式 SEED=20260731+batch×100+offset——批 21–40
  使用的种子区间从未被批 1–20 使用）。
- 生成器：仓内 `migration/generate_blind_batches.py`，高保真求解复用
  `verify_attribution.py::hf_solve`（已经与外部管线数据三方核验：
  LF 极限 5.7e-14、dT_int 列 1.7e-11、批 01 MSE≈σ²）。
- **物理回归核验（生成前完成，只用批 1–20 既有数据）**：全部 20 批
  参考+验收集池化残差 y_obs−hf_solve = mean −0.033 K / std 4.519 K
  （n=6400，σ=4.494）——仓内求解器与外部生成器物理一致。
- 数据层验收：`migration/accept_blind_batches.py`，硬门槛 = v1.3 §5.3
  同款（有限性、sha256、k_true 范围、种子协议、Newton 残差、噪声统计、
  dT_int 与工程集一致性、集合隔离、覆盖 p95<0.3）；双保真相关性与
  SNR 为数据属性（报告值，软门槛 [0.6, 0.95]，与原管线后续锚点史一致）。
- **补救规则（预声明）**：验收失败若源于代码缺陷——修复代码、记录原因、
  重新生成（此时数据未被任何建模使用）；若源于合法抽样涨落（如相关性
  边际出界）——保留批次并如实记录为数据属性，**不允许换种子重抽**。

## 3. 模型与实验臂（固定）

唯一 PWL 配置 = 冻结 A0 逐项复制（`heat_v3_qint`、identity 映射、
相同 lambda 网格与求解器参数、相同种子 2026 与嵌套划分公式），仅批次
改为 21–40。比较模型（预注册紧凑族，提升检验功效；GP 为 b20 口径最强
监督基线）：

| 模型 | 角色 | 信息等级（审计 §4） |
|---|---|---|
| PWL（A0 配置） | 机制辅助型 PWL / G2 层 | L2 特征，结构已冻结 |
| PWL-noWeak（lambda_physics=0） | 弱标签通道消融 | 同上 |
| MechanismAligned（A4） | 已知机理灰盒 / oracle 参照 | L2 |
| AffineAligned（M1 形式化） | 纯全局输出对齐参照 | L1 |
| GP | 强监督基线 | L0 |
| Physics（标定+GP 差异） | 两步法基线 | L0 |
| PhysicsDirect | 未训练低保真直出 | L0 |

12 标签档（10–120，步长 10）× 20 批；配对单位 = 批（同划分、同验收集）。
配置：`migration/configs/heat_blind_s1.yaml`、
`heat_blind_s1_no_weak.yaml`（后者 lambda_grid.physics=[0]、仅 PWL 行）。

## 4. 比较族与终点（固定）

对照族（负值 = 前者更优；分析脚本 `migration/analyze_blind_s1.py`）：

- C1 MechanismAligned < PWL（D1：R4 独立确认）
- C2 PWL < GP；C3 PWL < Physics（D2：框架对新批强基线）
- C4 PWL < AffineAligned（D4：结构化 PWL 相对纯仿射的增量）
- C5 PWL < PWL-noWeak（D3：弱标签通道价值，审计 §5.2.3）
- C6 MechanismAligned < AffineAligned（描述性：机理项相对纯对齐的增量）

- **主终点**：@120 RMSE，逐对照单侧配对 t，**Holm 族 = 6 对照**；
- **次终点**：@120 MSE（同族同校正）；小样本端（10–60）与大样本端
  （70–120）分段均值配对；MSE/σ²；符号一致率；Cohen dz；
- 全曲线逐档 p 值仅作描述，不进任何校正族、不作判定依据。

## 5. 信息边界与解封纪律

1. 本文件及 §7 所列工件冻结后才生成批 21–40；
2. 数据验收只看聚合 QA 指标，不做任何模型间比较；
3. 实验运行期间只允许查看收敛/候选计数等运行日志，不查看
   results.csv / quality_checks.csv 的任何精度行；
4. 结果齐备后**一次性**运行 `analyze_blind_s1.py`，无论结果好坏均
   成文报告；
5. 解封后若要改模型/特征/比较族，批 21–40 立即降级为开发集，须再
   生成下一套未触碰批次（S2 批次区间届时另定）。

## 6. 判定规则（固定措辞）

- **D1**：C1 的 Holm 校正 p<0.05 且均值差为负 → "A4 优于机制辅助型
  PWL"在新批上成立（R4 确认）；均值差为正时报告反向单侧 p，校正后
  <0.05 → R4 推翻；否则不能定案；
- **D2**：C2、C3 均 Holm 显著 → PWL 在新批上保持锚点 6 声索；
- **D3**：C5 Holm 显著 → 弱标签通道有可测价值；
- **D4**：C4 Holm 显著 → 结构化 PWL 相对纯仿射对齐有正当理由。

判定映射（审计 §7.5 表）：D1 成立且 D2 不成立 → "当前场景更适合直接
灰盒校准"；D1、D2 同成立 → "PWL 有效但非最优，价值需定位在部署约束
（蒸馏 vs 在线调用）"；D1 不成立且 D2 成立 → "机制辅助型 PWL 在新批上
优于全部既有基线，但结论仅限机理已知场景"。

## 7. 冻结工件哈希（sha256 前 16 位，生成数据前计算）

| 文件 | sha256(前16) |
|---|---|
| migration/generate_blind_batches.py | 65ce70e661b288ae |
| migration/accept_blind_batches.py | e0d805780136012d |
| migration/analyze_blind_s1.py | d29fc264a95f04a7 |
| migration/configs/heat_blind_s1.yaml | 0294cb6bb1b36e9b |
| migration/configs/heat_blind_s1_no_weak.yaml | a05f2ebfc800bbc0 |
| migration/src/pwl_migration/heat_experiments.py | 6fbc0999b9f3eefb |
| migration/src/pwl_migration/heat.py | e0efd22da7593e70 |
| reproduction/src/pwl_repro/experiments/tuning.py | 55fae4a672acf24e |
| reproduction/src/pwl_repro/core/model.py | 2768d98f4dbc905d |

仓库基线提交：40eb12463a5f3c523ee861aa6d40df5dd98f3406（2026-07-28；
其后工作树改动即本项目当前工作）。冻结前已在开发批 1–2 上完成管线
冒烟（6 模型 × 2 档 × 2 批，数值合理），冒烟结果目录
`results/heat_blind_s1_smoke/` 为一次性开发产物，不进入任何分析。

## 8. 执行序列（审计 §7.4）

```text
冻结本文件与工件（已完成）
→ 生成批 21–40（generate_blind_batches.py）
→ 数据层验收（accept_blind_batches.py；只看 QA）
→ 运行 heat_blind_s1 与 heat_blind_s1_no_weak（后台，日志仅限运行进度）
→ 一次性 analyze_blind_s1.py
→ 撰写 S1 报告（无论结果）
```
