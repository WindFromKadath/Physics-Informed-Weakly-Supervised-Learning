# 论文复现：PWL 核心、仿真与点焊案例 B

本目录实现论文 *Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes* 的 PWL 数学结构与 BCD–ADMM 优化方法，提供可运行、可审计的研究基线。它是独立实现，**不代表原作者官方代码，也不声称严格复现全部表格数值**。

[仓库首页](../README.md) · [技术实现与假设](REPRODUCTION.md) · [热传导迁移](../migration/README.md) · [当前项目状态](../docs/CURRENT_STATUS.md)

## 实现内容

| 模块 | 内容 |
|---|---|
| `src/pwl_repro/core/` | FeatureSpec、PWLRegressor、BCD–ADMM、稀疏近端算子 |
| `src/pwl_repro/scenarios/` | 论文式 (9)–(11) 仿真、点焊数据适配与代理模型 |
| `src/pwl_repro/experiments/` | 调参、嵌套划分、实验协议、统计与结果输出 |
| `configs/` | smoke、日常规模、论文规模和案例 B 配置 |
| `tests/`、`scripts/` | 回归测试与独立数值验证 |
| `reports/` | 复现审计、差异解释；`legacy/` 保存历史分析 |

模型结构为 `y_hat = H(x_ph, theta) g + B(x_pr, x_ph) d`。场景无关算法位于 `core`；场景必须提供物理模型和适用的基函数，不能直接把某个场景的 H/B 库套用到所有任务。

## 环境准备

尚未获取仓库时，执行 [首页获取步骤](../README.md#获取与运行)，使用 `HeatTest` 分支。两个业务包共享根级环境，**以下命令全部从仓库根目录执行**。

```sh
uv sync --locked --python 3.11 --extra dev
uv run python reproduction/run_reproduction.py --help
```

需要预先安装 uv；项目 Python 范围为 ≥3.11、<3.14，本次验证使用 3.11。点焊运行直接使用仓内 CSV 和 Python 代理，无需安装 ANSYS 或 R；重新提取原始 R 数据不属于最小运行流程。

## 最小运行

先用小规模配置检查端到端流程：

```sh
uv run python reproduction/run_reproduction.py --config reproduction/configs/smoke.yaml --mode sample-size --output reproduction/results/quickstart
```

点焊案例 B 使用独立场景参数：

```sh
uv run python reproduction/run_reproduction.py --scenario case_b --config reproduction/configs/case_b_smoke.yaml --mode sample-size --output reproduction/results/case_b_quickstart
```

输出目录由程序创建。重复使用相同目录可能覆盖产物，建议为每次正式运行指定新目录。smoke 验证的是程序流程，不是论文数值一致性。

## 选择实验协议

| 目标 | 配置与参数 |
|---|---|
| 快速检查 | `configs/smoke.yaml` |
| 日常仿真 | `configs/default.yaml` |
| 趋势诊断 | `configs/diagnostic.yaml` |
| 论文规模 | `configs/paper_protocol.yaml`；完整网格计算量显著增大 |
| 案例 B | `configs/case_b_default.yaml`，必须加 `--scenario case_b` |

仿真 `--mode` 支持 `sample-size`、`physics-accuracy`、`label-savings` 和 `all`；案例 B 仅支持 `sample-size`。默认模式为 `sample-size`，不会自动运行全部协议。

```sh
uv run python reproduction/run_reproduction.py --config reproduction/configs/default.yaml --mode all --output reproduction/results/default_recheck
```

划分规则、相关性目标和统计口径见 [Section IV 协议](SECTION_IV_PROTOCOL.md)。选择规则与稳定项均来自具体配置；不要把仿真的 1-SE 规则误当成所有迁移场景的默认值。

## 数据来源与复现边界

- **仿真数据**：由仓内生成器按配置和种子生成。H/B、输入协方差、奇点拒绝采样和噪声口径包含工程假设，详见 [技术说明](REPRODUCTION.md#论文缺失信息与本复现的明确假设)。
- **案例 B**：使用 SAVE 1.0 的 120 条过程样本和 35 条模型样本。来源、字段及原始材料见 [数据说明](../datasets/case_b_spotweld/README.md)。
- **已见限制**：案例 B 的协议已实现，但论文 Table V 的 PWL 优势未复现，见 [案例 B 报告](reports/案例B复现/案例B复现报告.md)。案例 A 未在本仓库提供完整可运行复现。
- **参数解释**：校准参数是否具有物理可识别性取决于具体场景，不能仅凭预测精度证明参数恢复正确。

## 输出与验证

输出包含 `results.csv`、预测、条件与配置记录、摘要、图表和 `quality_checks.csv`；具体文件集合随场景/模式变化。运行产物默认被 Git 忽略，首次克隆没有历史训练结果。报告里的本地图表引用可能需要运行对应配置后才能查看。

```sh
uv run pytest reproduction/tests
uv run python reproduction/scripts/verify_sim_correctness.py
```

2026-09-11 的全项目验证为 70 项测试和 30 项独立仿真检查通过；这是当时的检查记录，不表示所有实验质量指标或论文结论均通过。详见 [整理记录](../docs/CONSOLIDATION.md)。

## 排查与贡献

- 找不到数据或配置：先确认当前目录是仓库根目录。
- 缺少依赖：使用根级 `uv sync --locked --python 3.11 --extra dev`，不要另建子目录环境。
- 完整网格耗时过长：先运行 smoke；并行规模由配置控制。
- GP 边界警告或质量检查失败：结合配置与 `quality_checks.csv` 判断；进程正常结束不等于实验假设成立。

提交问题请附运行命令、配置、Python 版本和日志片段；修改核心请运行相关测试与独立数值检查。引用结果时同时记录原论文、数据来源、仓库提交号和配置，并明确工程假设。
