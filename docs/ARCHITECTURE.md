# 项目架构

仓库只有两部分：论文 Section IV 仿真复现和一维稳态热传导测试。二者共享一个根级 uv 环境和 PWL 核心，不以通用接口的存在宣称其他应用已实现。

## 依赖方向

```text
pwl_repro.core                  特征库、PWLRegressor、BCD–ADMM
        ▲
        ├── pwl_repro.scenarios 仿真数据与仿真特征
        ├── pwl_repro.experiments 调参、协议、统计与报告
        └── pwl_migration      热传导测试
                    └── pwl_repro.experiment_api
```

核心不导入具体场景。仿真特征在包初始化时注册；热传导特征在迁移入口导入场景模块时注册。跨包实验复用通过 `experiment_api`。旧的 `model`、`features`、`optimization`、`simulation` 路径保留为兼容别名。

## 目录职责

| 目录 | 内容 |
|---|---|
| `reproduction/` | Section IV 仿真、敏感性配置、测试与复现说明 |
| `migration/` | 一维热传导物理、40 批数据、配置、验收、分析和报告 |
| `docs/` | 两部分的当前状态与架构 |
| `references/` | 论文复现的文献依据与提取工具 |

`results/`、`runs/`、缓存和临时目录不进入 Git，人工冻结的 v3 基线卡除外。`legacy/` 中保留的材料仅用于这两部分的历史追溯，不作为当前实现入口。

## 运行与检查

全部命令从仓库根目录运行：

```sh
uv sync --locked --python 3.11 --extra dev
uv run pytest
uv run python reproduction/scripts/verify_sim_correctness.py
```

实验命令分别见 [论文复现](../reproduction/README.md) 和 [热传导测试](../migration/README.md)。
