# 热传导迁移：多保真学习与灰盒对照

本目录研究 PWL 在一维稳态热传导多保真数据上的适用性，复用 `pwl_repro.core` 与公共实验接口。包含 40 批输入数据、实验配置、分析代码、预注册与报告。

**当前证据：在本场景已知机理与低成本物理模型条件下，直接灰盒校准 A4 比机制辅助型 PWL 更合适。** 本目录保留 PWL 的均值优势、失败边界和消融证据，不将单一场景的结果推广为普适结论。

[仓库首页](../README.md) · [论文复现](../reproduction/README.md) · [当前项目状态](../docs/CURRENT_STATUS.md) · [S1 完整报告](reports/迁移后续工作/PWL盲测S1报告.md)

## 场景与数据来源

输入为 `x_ph=(Q, h, T_inf)`、过程变量 `x_pr=P`，校准参数 `theta=1/k`；输出为双层杆界面两侧平均温度，单位 K。高保真模型包括温度相关导热率与接触热阻，低保真模型使用恒定导热率和完美界面闭式解。

这是**数值生成的研究数据**，不是现场测量数据。CSV 中的 `reference_measurements` 是高保真解加噪声。虽然历史报告以“COMSOL 场景”命名，当前一维运行不调用 COMSOL；已有 CSV 与仓内 Python 物理实现足以运行以下实验。

| 批次 | 来源与用途 | 当前状态 |
|---|---|---|
| 1–20 | 历史外部一维生成管线；输入与验收报告已入库 | 开发及冻结 v3 基线 |
| 21–40 | 仓内 `generate_blind_batches.py`；S1 新批确认 | S1 解封后已转为开发数据 |
| `datasets/legacy/` | 早期三维参考与历史材料 | 不作为默认训练输入 |

每批包含参考集 120 条、独立验收集 200 条、工程预测集 200 条，共 520 条；40 批合计 20,800 条。CSV 的原始字节由 `.gitattributes` 保留，以匹配 `metadata.json` 的 SHA-256。

[数据规格](datasets/README.md) 主要记录批 1–20；批 21–40 的生成过程与冻结协议见 [S1 预注册](reports/迁移后续工作/PWL盲测S1预注册.md) 和 [S1 报告](reports/迁移后续工作/PWL盲测S1报告.md)。历史外部管线不在本仓库，日常运行直接读取已提供的数据。

## 快速开始

需要 Git、uv、Python ≥3.11 且 <3.14，本次验证使用 Python 3.11。获取 `HeatTest` 分支的方法见 [仓库首页](../README.md#获取与运行)。**以下单行命令均从仓库根目录执行**，适用于 PowerShell 和常见 Unix shell。

```sh
uv sync --locked --python 3.11 --extra dev
uv run python migration/run_migration.py --help
uv run python migration/run_migration.py --config migration/configs/heat_smoke.yaml --output migration/results/quickstart
uv run pytest migration/tests
```

smoke 使用小规模配置验证数据读取、训练与结果保存。输出目录自动创建；已有目录可能被覆盖，正式实验请使用新目录。通用测试全部通过与科学质量锚点全部通过是不同判定。

## 当前结果与结论边界

以下为既有 S1 报告的批 21–40 均值，不是本次文档更新重跑所得；`@120` 指配置中的标记样本档位，训练/验证划分依预注册执行。

| 模型 | @10 RMSE / K | @60 RMSE / K | @120 RMSE / K |
|---|---:|---:|---:|
| A4（MechanismAligned） | 7.477 | 4.866 | 4.727 |
| PWL | 7.872 | 5.932 | 5.449 |
| PWL 无弱标签 | 11.045 | 6.072 | 5.434 |
| GP | 10.785 | 6.039 | 5.612 |
| Physics（标定 + GP） | 9.327 | 6.559 | 5.554 |

- A4 比 PWL 的 @120 RMSE 低 0.722 K，20/20 批方向一致，Holm 校正 p<1e-4；A4 使用强机理结构，优势仅在本场景得到确认。
- PWL 对 GP/Physics 的 Holm 校正 p 分别为 0.066/0.266，证据不足以声称校正后显著优于两者。
- 弱标签收益主要见于小样本端；@120 未检出弱标签优势。
- G0/G1 的批 21–25 结果仅为开发性证据，见 [分层报告](reports/迁移后续工作/PWL分层G0G1开发测试报告.md)。后续改动须使用新的预注册与未触碰批次确认，不能继续把 21–40 称为未见数据。
- `theta` 在本场景存在结构性不可识别性，不应解释成可靠恢复的真实导热率，见 [判据修订](reports/迁移后续工作/COMSOL场景theta判据修订.md)。

早期批 1–20 的 PWL @120=5.351 K、Physics-GP=5.648 K，以及“12 档均值最优”属于当时的对照集合；不包含后来加入的 A4。详见 [20 批报告](reports/迁移后续工作/COMSOL场景20批统计闭环报告.md) 与 [信息边界审计](reports/迁移后续工作/PWL算法真实能力与信息边界审计.md)。

## 配置选择

| 目的 | 配置 | 证据定位 |
|---|---|---|
| 流程检查 | `configs/heat_smoke.yaml` | 小规模环境检查 |
| 历史 v3 基线 | `configs/heat_v3_qint_min.yaml` | 10 批开发结果 |
| v3 扩展 | `configs/heat_v3_qint_min_b20.yaml` | 20 批冻结比较 |
| S1 主臂 | `configs/heat_blind_s1.yaml` | 批 21–40，含灰盒对照 |
| S1 弱标签消融 | `configs/heat_blind_s1_no_weak.yaml` | 与 S1 配对 |
| G0/G1 开发 | `configs/heat_g0_generic_dev.yaml`、`configs/heat_g1_prior_dev.yaml` | 5 批方向性实验 |

`heat_default.yaml` 及早期变体用于历史研究，不作为最新结果的默认推荐配置。运行时间随硬件、批数和候选网格变化，不能以 smoke 的耗时代替正式实验预算。

历史基线复核使用新目录：

```sh
uv run python migration/run_migration.py --config migration/configs/heat_v3_qint_min_b20.yaml --output migration/results/v3_b20_recheck
```

### S1 完整复算顺序

首次克隆包含输入数据和人工报告，**不包含两条 S1 臂的训练结果**。分析脚本固定读取下面两个目录，因此先完成两条训练，再分析；在已有工作区执行前应保留已有产物。本流程重现历史 S1，不构成新的盲测。

```sh
uv run python migration/run_migration.py --config migration/configs/heat_blind_s1.yaml --output migration/results/heat_blind_s1
uv run python migration/run_migration.py --config migration/configs/heat_blind_s1_no_weak.yaml --output migration/results/heat_blind_s1_no_weak
uv run python migration/analyze_blind_s1.py
```

数据已提供，无需重新生成。`generate_blind_batches.py` 用于生成过程追溯，并拒绝覆盖已有批次；`accept_blind_batches.py --start 21 --end 40` 可重新验收，执行后会重写 `datasets/acceptance_report_blind_21_40.json`。相关命令应通过 `uv run python migration/脚本名.py` 运行。

## 目录与输出

```text
migration/
├── src/pwl_migration/       # 热传导物理、特征与实验驱动
├── configs/                 # 基线、S1、消融与开发配置
├── datasets/                # 40 批 CSV、元数据与验收报告
├── tests/                   # 物理、特征、生成器和管线测试
├── reports/                 # 人工结论、预注册与方案文档
├── run_migration.py         # 实验入口
├── analyze_*.py             # 统计分析；部分固定读取 results 子目录
└── results/                 # 本地运行产物，默认不上传
```

实验输出包括 `results.csv`、逐样本预测、配置和条件记录、统计摘要、图表、`quality_checks.csv`；以实际配置与输出为准。后者包含通用检查和场景锚点，需逐项解释，不应只依据进程退出码判断方法有效。

运行结果、日志和缓存不进入 Git；[人工基线卡](results/heat_v3_qint_min/BASELINE.md) 为例外。历史报告引用的 `results/` 图片在首次克隆时可能不存在，需运行对应训练与绘图脚本生成。部分中文绘图脚本使用 Microsoft YaHei，其他系统需配置可用中文字体。

## 测试范围与反馈

本目录只保留实际热传导测试及其数据、配置和分析证据；不包含其他应用框架或应用方案。

修改本场景时遵守 [架构约定](../docs/ARCHITECTURE.md)：通过 `pwl_repro.core` 和 `experiment_api` 复用核心，不反向引入场景依赖。问题反馈请包含提交号、命令、配置、批号、Python 版本和最小错误输出；引用结果请区分开发、S1 确认，并追溯原论文和数据来源。
