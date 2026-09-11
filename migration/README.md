# migration/ — 迁移实验（COMSOL 热传导多保真场景）

> 最新结论见 [S1 报告](reports/迁移后续工作/PWL盲测S1报告.md) 和 [当前状态总览](../docs/CURRENT_STATUS.md)。
> A4 在 S1 @120 优于 PWL；下方 20 批 v3 数据为历史开发基线。批 21–40 解封后已转为开发数据，G0/G1 仍待新批确认。

> 本目录是从 `reproduction/` 分离出的**迁移实验**业务线（2026-08-02 重构）。
> 论文复现（Section IV 仿真 + 案例 B 点焊）在 [`../reproduction/`](../reproduction/REPRODUCTION.md)。

## 内容

```text
migration/
├── src/pwl_migration/     # 迁移场景包（依赖 reproduction 的 pwl_repro 核心包）
│   ├── heat.py            # 热传导场景：闭式低保真解、H/B 库、数据适配器
│   ├── heat_experiments.py# 实验驱动与迁移锚点
│   └── cli.py             # 迁移实验命令行入口
├── configs/               # heat_*.yaml（主配置、20批、消融与弱标签数量变体）
├── tests/test_heat.py     # 场景物理、适配器、特征库、端到端测试
├── results/               # heat_* 运行产物
├── reports/               # COMSOL 场景设计与迁移分析报告
├── datasets/              # 当前一维数据与验收报告；legacy/ 保存历史溯源
├── runs/                  # heat_* 运行日志
└── run_migration.py       # 免安装入口脚本
```

## 场景概要

一维热传导多保真数据集（规格见 [`datasets/README.md`](datasets/README.md)）：
x_ph=(Q, h, T_inf)、x_pr=P、theta=1/k；批次相当于重复（每批一个 k_true）；
弱标签为低保真闭式解（`pwl_migration/heat.py`），训练过程无在线仿真调用。
场景设计与验证锚点见
[`reports/COMSOL三维热传导场景迁移分析.md`](reports/COMSOL三维热传导场景迁移分析.md)。

## 运行

从仓库根目录执行：

```powershell
# 冒烟验证（2 批、小网格）
uv run python migration/run_migration.py `
  --config migration/configs/heat_smoke.yaml `
  --output migration/results/heat_smoke_check

# 日常规模
uv run python migration/run_migration.py `
  --config migration/configs/heat_default.yaml `
  --output migration/results/heat_default

# 当前主配置（v3_qint_min：10 批 × 12 档，约 15 分钟；复核请写入新目录，勿覆盖主结果）
uv run python migration/run_migration.py `
  --config migration/configs/heat_v3_qint_min.yaml `
  --output migration/results/heat_v3_qint_min_recheck

# 20 批统计终态复核
uv run python migration/run_migration.py `
  --config migration/configs/heat_v3_qint_min_b20.yaml `
  --output migration/results/heat_v3_qint_min_b20_recheck
```

> 当前状态（2026-08-04）：20 批下 PWL @120 RMSE 5.351 < Physics-GP 5.648，
> 12/12 标签档均值全优；配对单侧 p=0.036、Holm 校正后 p=0.071，终态为
> “均值优势趋势保持，但尚无充分的校正后显著性证据”。
> 事实基线与结果追溯见 `results/heat_v3_qint_min/BASELINE.md`，
> 配置因果链见 `reports/迁移后续工作/COMSOL场景v3唯一可行配置与通过机制.md`，
> 统计终态见 `reports/迁移后续工作/COMSOL场景20批统计闭环报告.md`。

除通用质量门槛外，`quality_checks.csv` 追加场景锚点（双保真相关性、
小样本优势、theta 学习、噪声下限逼近、PWL 对物理基线优势、d 系数物理
一致性）。

## 依赖关系

`pwl_migration` 通过 `pwl_repro` 的 `FeatureSpec` 注册表接入核心估计器；
热传导场景模块导入时显式注册自己的特征工厂，核心包不会反向导入迁移包。
测试与运行需要两个 src 目录都在 `PYTHONPATH` 上——根 `pyproject.toml`
的 pytest 配置与 `run_migration.py` 已处理，无需手工设置。
