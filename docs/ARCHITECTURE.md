# 项目架构

本仓库使用一个根级 `uv` 环境，同时构建两个 Python 包：

- `pwl_repro`：场景无关的 PWL 算法、论文仿真和案例 B；
- `pwl_migration`：热传导迁移场景，单向复用 `pwl_repro` 的公共接口。

## 依赖方向

```text
pwl_repro.core                 # 模型、特征、优化；不依赖具体场景
        ▲
        ├── pwl_repro.scenarios       # 仿真与案例 B
        ├── pwl_repro.experiments     # 合成实验协议
        └── pwl_migration             # 热传导场景
                    │
                    └── pwl_repro.experiment_api
```

依赖约束：

1. `core` 不导入 `scenarios` 或 `pwl_migration`；
2. 场景模块在导入时使用 `register_feature_spec` 注册特征工厂；
3. `experiments` 子包内部按 types → tuning/statistics → protocols/reporting
   的方向组织，不形成循环依赖；
4. 跨包实验复用只通过 `experiment_api`，不依赖 `_` 开头的实现细节；
5. CLI 负责选择场景，模型本身不负责发现或加载场景。

## 代码目录

```text
reproduction/src/pwl_repro/
├── core/
│   ├── model.py              # PWLRegressor、theta 标定、BCD
│   ├── features.py           # FeatureSpec、H/B 特征库与注册表
│   └── optimization.py       # consensus ADMM 与近端算子
├── scenarios/
│   ├── simulation.py         # Section IV 数据生成
│   └── case_b.py             # 点焊数据适配和代理模型
├── experiments/
│   ├── types.py              # 场景契约、制品容器、指标与嵌套划分
│   ├── tuning.py             # 候选评估、超参数选择与条件拟合
│   ├── protocols.py          # IV-A/B/C 协议与配置调度
│   ├── statistics.py         # 配对检验、TOST 与 Holm 校正
│   └── reporting.py          # 汇总、图表、质量检查与持久化
├── experiment_api.py         # 跨场景稳定公共实验接口
├── case_b_experiments.py     # 案例 B 驱动
├── baselines.py              # 监督/物理基线
└── cli.py                    # reproduction 命令行入口

migration/src/pwl_migration/
├── heat.py                   # 热传导物理、数据适配和 FeatureSpec
├── heat_experiments.py       # 迁移实验驱动与质量锚点
└── cli.py                    # migration 命令行入口
```

根级 `pwl_repro.model`、`features`、`optimization`、`simulation` 和
`case_b` 是兼容模块别名。已有代码无需修改；新代码应优先从 `pwl_repro.core`
或 `pwl_repro.scenarios` 导入。

## 数据与产物

- `datasets/`：论文案例 B 的输入数据；
- `migration/datasets/`：热传导迁移输入数据；
- `references/`：论文、提取文本和解析材料；
- `results/`、`runs/`、`.joblib/`：可再生成产物，不进入版本控制；
- `reports/`：人工整理、需要长期保存的分析结论。

数据与文档目录允许设置 `legacy/`，用于保存原始备份、历史日志、早期方案和
已被当前报告替代的材料。运行时代码不得读取 `legacy/` 作为默认输入；每个
归档目录都必须有 README 说明归档原因与当前替代位置。

## 新增场景

新增场景时应：

1. 实现 `ScenarioData` 所需的数据属性和物理模型接口；
2. 构造 `FeatureSpec` 并在场景模块导入时调用 `register_feature_spec`；
3. 从 `pwl_repro.experiment_api` 复用训练、指标和制品逻辑；
4. 将场景配置、测试、数据说明和报告分别放入对应业务目录。

## 验证

从仓库根目录执行：

```powershell
uv sync --python 3.11 --extra dev
uv run pytest
uv run pwl-repro --help
uv run pwl-migration --help
```
