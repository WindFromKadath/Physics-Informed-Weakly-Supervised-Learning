# PWL Paper 工作区

> **最新入口（2026-09-11 整理）：[当前有用内容与后续主线](docs/CURRENT_STATUS.md)。**
> 后续 S1 盲测中 A4 @120 RMSE 为 4.727 K，PWL 为 5.449 K；当前热传导场景更适合直接灰盒校准。
> 下方 2026-08-04 结果属于历史开发基线，不包含 A4；下一步叶轮配方应用仍处于方案阶段。

当前内容按用途分为四个目录：**论文复现**（reproduction/）与**迁移实验**
（migration/）已分离，共享根级 uv 环境。

```text
PWL_Paper/
├── pyproject.toml       # 根项目依赖与两个包的构建配置
├── uv.lock              # 唯一依赖锁
├── docs/                # 整体架构与开发约定
├── reproduction/        # PWL 核心、论文仿真和案例 B
├── migration/           # 热传导迁移场景（包 pwl_migration）
├── references/          # 论文、解析材料和文献工具
└── datasets/            # 案例 B 点焊数据
```

## 历史开发基线（2026-08-04）

- 论文公开的 PWL 数学结构与 BCD–ADMM 优化骨架已完成工程实现，
  **57 项测试与 30 项独立数值检查全部通过**（非论文全部数值的严格复现）。
- 迁移线 20 批终态：PWL @120 RMSE **5.351** < Physics-GP 5.648，
  **12 个标签档均值全部最优**；配对单侧 p=0.036，Holm 校正后 p=0.071，
  因此结论是“均值优势趋势保持，但尚无充分的校正后显著性证据”。
- 事实基线与追溯规则：[`migration/results/heat_v3_qint_min/BASELINE.md`](migration/results/heat_v3_qint_min/BASELINE.md)；
  配置因果链：[`migration/reports/迁移后续工作/COMSOL场景v3唯一可行配置与通过机制.md`](migration/reports/迁移后续工作/COMSOL场景v3唯一可行配置与通过机制.md)。
- PWL 应定位为**可实例化的物理—数据联合学习框架**：算法内核可复用，
  每个新场景必须重新完成物理建模、基函数设计与可识别性验证
  （迁移门禁见《阶段性总结》§10）。

## 历史阶段成果（最新证据见上方入口）

| 内容 | 当前证据 |
|---|---|
| 通用算法框架 | `pwl_repro.core` 提供 FeatureSpec、PWLRegressor、BCD–ADMM 与近端算子 |
| 多场景能力 | 论文仿真、案例 B 点焊、热传导迁移三条实验线 |
| 热传导结果 | 20 批、12 档均值全优；@120 为 5.351 vs Physics-GP 5.648 |
| 机制证据 | 无弱标签损害小样本、无 B 损害大样本、去 qint 使锚点翻转、n_weak≈100 饱和 |
| 质量基础 | 57 项 pytest、30 项独立数值检查、数据验收 20/20、GitHub Actions smoke |

适合浏览和汇报的完整入口见
[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)。

## 最小复现

```powershell
uv sync --python 3.11 --extra dev
uv run pytest
uv run python reproduction/scripts/verify_sim_correctness.py
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/smoke.yaml `
  --output reproduction/results/smoke
uv run python migration/run_migration.py `
  --config migration/configs/heat_smoke.yaml `
  --output migration/results/heat_smoke_check
```

## 完整实验

```powershell
# 论文仿真线（Section IV 三协议）
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/default.yaml --mode all `
  --output reproduction/results/default

# 迁移线当前主配置（约 15 分钟；复核请写新目录，勿覆盖冻结基线）
uv run python migration/run_migration.py `
  --config migration/configs/heat_v3_qint_min.yaml `
  --output migration/results/heat_v3_qint_min_recheck

# 20 批统计终态复核
uv run python migration/run_migration.py `
  --config migration/configs/heat_v3_qint_min_b20.yaml `
  --output migration/results/heat_v3_qint_min_b20_recheck
```

## 已知边界

- 未严格复现论文全部数值：作者 `H/B`、输入协方差、噪声与奇点处理未公开，
  采用"结构性复现"口径（[`reproduction/REPRODUCTION.md`](reproduction/REPRODUCTION.md)）。
- 案例 B 协议已复现，论文 Table V 优势未复现，根因已分层归因
  （[`reproduction/reports/案例B复现/案例B复现报告.md`](reproduction/reports/案例B复现/案例B复现报告.md)）。
- 热传导 theta 结构性不可识别，定位 nuisance parameter
  （[`migration/reports/迁移后续工作/COMSOL场景theta判据修订.md`](migration/reports/迁移后续工作/COMSOL场景theta判据修订.md)）。
- 热传导 20 批结果保持均值优势，但 Holm 校正后 p=0.071；不得表述为
  “对 Physics-GP 具有校正后统计显著优势”。
- 选参规则场景相关：仿真线 1-SE、热传导 minimum，非全局默认
  （[`reproduction/reports/选参规则跨场景对照报告.md`](reproduction/reports/选参规则跨场景对照报告.md)）。

## 入口

- 论文复现（仿真 + 案例 B）：[`reproduction/REPRODUCTION.md`](reproduction/REPRODUCTION.md)
- GitHub 展示与项目概览：[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
- 当前必须项目与实施方案：[`PWL当前必须项目与实施方案.md`](PWL当前必须项目与实施方案.md)
- 代码分层与依赖规则：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 第 IV 节实验协议：[`reproduction/SECTION_IV_PROTOCOL.md`](reproduction/SECTION_IV_PROTOCOL.md)
- 迁移实验（COMSOL 热传导）：[`migration/README.md`](migration/README.md)
- 案例 B 数据集：[`datasets/case_b_spotweld/README.md`](datasets/case_b_spotweld/README.md)
- 阅读论文解析：[`references/docs/README.md`](references/docs/README.md)
- 查看参考材料结构：[`references/README.md`](references/README.md)

`PWL_Paper/` 根目录是唯一的 uv 项目。场景无关算法位于
`pwl_repro.core`，仿真与案例 B 位于 `pwl_repro.scenarios`；`migration/`
的 `pwl_migration` 只依赖核心包和稳定的 `experiment_api`。旧模块路径保留为
兼容别名，现有脚本无需迁移。两条业务线继续维护独立的 configs / tests /
results / reports / datasets。

## 归档约定

数据和文档目录中的 `legacy/` 只保存历史溯源、旧版分析与已被后续材料替代的
内容。默认开发、测试和实验不得依赖 `legacy/`；如需复核历史结论，应从对应
`legacy/README.md` 进入。当前可执行数据、配置和结论文档继续保留在各业务目录
的主层级。
