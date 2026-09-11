# PWL 论文复现框架（结构性复现）

> 本页保留数学实现、工程假设与复现边界。安装、仿真/案例 B 运行命令及公开仓库导航请先读 [README](README.md)。热传导最新证据见 [迁移说明](../migration/README.md)。

本工程实现 Alenezi 等人在 *Physics-Informed Weakly-Supervised Learning for
Quality Prediction of Manufacturing Processes* 中提出的 PWL 框架，并将论文未公开
的实现选择显式配置化。目标是得到一个可运行、可审计、可扩展的复现基线，而不是
声称在缺少原作者代码与案例数据时逐点重建论文表格。**已完成范围为论文公开的
数学结构与 BCD–ADMM 优化骨架的工程实现**，非论文全部数值的严格复现。

## 已实现范围

- 论文式 (9)–(11) 的仿真数据生成器，SNR 定义为
  `Var(signal) / Var(noise)`；
- 25 维 `H(x_ph, theta)`；实验默认使用3维紧凑 `B`，并保留12维扩展 `B`；
- `H` 对两个校准参数严格仿射，并带 5 组稀疏组 Lasso 分组；
- BCD 主循环：`g` 一致性 ADMM → `d` 闭式最小二乘 →
  `theta` 一致性 ADMM；
- PWL 超参数候选支持 joblib 多进程并行，工作进程限制为单个 BLAS 线程；
- 命题 1 的 4 个近端算子与命题 2 的 3 个近端算子；
- 盒约束校准、恒等/线性映射 `phi`、训练/验证超参数选择；
- Ridge、SVR、决策树、随机森林、GBDT、GP、两步校准 + GP 基线；
- 样本量、物理模型精度、标签节省三组仿真实验；
- CSV 结果、聚合摘要、RMSE 曲线和运行环境元数据；
- 数据、基函数仿射性、ADMM 数值解和端到端模型测试。

## 快速开始

仓库根目录 `PWL_Paper/` 是统一的 uv 项目，`.venv`、`pyproject.toml` 与
`uv.lock` 都保存在根目录。请从仓库根目录运行：

```powershell
uv sync --python 3.11 --extra dev
uv run pytest
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/smoke.yaml `
  --output reproduction/results/smoke
```

运行日常规模实验：

```powershell
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/default.yaml `
  --mode sample-size `
  --output reproduction/results/sample_size
```

可用模式：

- `sample-size`：标记样本量曲线；
- `physics-accuracy`：目标相关性 0.85、0.70、0.50；
- `label-savings`：30–110 个标记样本的等效性能对比；
- `all`：依次运行以上三组实验。

三个实验的数据共享、统计检验和论文图表协议详见
[`SECTION_IV_PROTOCOL.md`](SECTION_IV_PROTOCOL.md)。

## 热传导迁移场景（已迁至 migration/）

热传导迁移实验（COMSOL 试点）已分离到独立的
[`migration/`](../migration/README.md) 目录树（包 `pwl_migration`），
本目录仅保留论文复现内容。迁移实验入口：

```powershell
uv run python migration/run_migration.py `
  --config migration/configs/heat_default.yaml `
  --output migration/results/heat_default

# 迁移线当前主配置（v3_qint_min，均值通过锚点 6）
uv run python migration/run_migration.py `
  --config migration/configs/heat_v3_qint_min.yaml `
  --output migration/results/heat_v3_qint_min_recheck
```

`reproduction/configs/paper_protocol.yaml` 使用论文的 12 个样本量、20 次重复和
`10^-3` 至 `10^3` 的完整三维网格，计算量非常大。先用 `smoke.yaml` 验证环境，
再用 `diagnostic.yaml` 检查论文趋势，最后才运行论文规模配置。

每次运行保存 `results.csv`、压缩的逐样本预测、实验条件、统计检验、
均值/标准差摘要、`quality_checks.csv`，以及三个实验对应的图表和表格。

## 工程结构

```text
PWL_Paper/
├── .venv/                      # 根项目唯一 uv 环境
├── pyproject.toml
├── uv.lock
├── reproduction/               # 论文复现（本目录）
│   ├── configs/
│   │   ├── smoke.yaml
│   │   ├── default.yaml
│   │   ├── diagnostic.yaml
│   │   ├── paper_protocol.yaml
│   │   └── case_b_*.yaml       # 案例 B（点焊熔核直径）
│   ├── src/pwl_repro/
│   │   ├── core/
│   │   │   ├── features.py     # FeatureSpec、通用 H/B 变换与注册表
│   │   │   ├── optimization.py # 通用一致性 ADMM
│   │   │   └── model.py        # PWL BCD 估计器（任意 theta/输入维度）
│   │   ├── scenarios/
│   │   │   ├── simulation.py   # 式 (9)–(11) 数据生成
│   │   │   ├── simulation_features.py
│   │   │   └── case_b.py       # 克里金代理、H/B 库、数据契约
│   │   ├── baselines.py        # 论文对比模型
│   │   ├── experiments/
│   │   │   ├── types.py        # 场景契约、指标、嵌套划分
│   │   │   ├── tuning.py       # 候选评估与超参数选择
│   │   │   ├── protocols.py    # IV-A/B/C 协议调度
│   │   │   ├── statistics.py   # 配对检验与 Holm 校正
│   │   │   └── reporting.py    # 图表、质量检查和制品
│   │   ├── experiment_api.py   # 跨场景稳定公共实验接口
│   │   ├── case_b_experiments.py
│   │   └── cli.py
│   ├── tests/
│   └── run_reproduction.py
├── migration/                  # 迁移实验（COMSOL 热传导，包 pwl_migration）
├── datasets/case_b_spotweld/   # 案例 B 官方数据（SAVE 1.0）
└── references/                 # PDF、解析 Markdown 与文献工具
```

## 数学实现对应

模型为

```text
y = H(x_ph, theta) g + B(x_pr, x_ph) d + epsilon
```

优化目标在代码中按以下形式实现：

```text
0.5 ||y - H1 g - B1 d||^2
+ 0.5 lambda1 ||weak_y - H2 g||^2
+ lambda2 ||g||_1
+ lambda3 sum_q ||g_q||_2
+ 0.5 lambda_d ||d||_2^2
```

`g` 子问题拆成标记拟合、物理蒸馏、L1、组 Lasso 四个 consensus 节点。
其中最后一项是本复现为小样本 `B` 系数增加的显式稳定项，令
`lambda_d=0` 即恢复论文给出的未正则化 `d` 更新。
`theta` 子问题不照抄补充材料中维度含混的 `g g^T` 写法，而是先从实际特征库
整理出

```text
H(x, theta) g = c(x, g) + G(x, g) theta
```

再对 `G` 使用标准最小二乘近端算子，因此矩阵维度始终是两个校准参数。

## 论文缺失信息与本复现的明确假设

| 缺失项 | 本工程默认值 |
|---|---|
| `H`、`B` 的具体基函数 | 25维 `H`；实验默认3维紧凑 `B`，12维扩展库可配置 |
| `phi` | 恒等映射；可配置为标记样本上的一维线性映射 |
| `lambda` 网格 | 配置文件明确给出 |
| 超参数选择 | 物理优先的一标准误差规则；可改回纯最小验证MSE |
| ADMM 步长/阈值 | 初始 `rho=1`，按原始/对偶残差自适应，阈值见配置 |
| `theta` 边界 | 仿真实验为 `[0,1]^2` |
| `Sigma_x` | `Sigma[i,j] = 0.9^|i-j|`；这是使0.85相关性目标可达的显式复现假设 |
| 物理噪声方差 | 输出噪声标准差的 0.25 倍，可配置 |
| 输出噪声尺度 | 在独立参考总体上按SNR校准，避免随场景样本量漂移 |
| `d` 稳定项 | 紧凑且标准化的 `B` 使用 `d_ridge=10`，防止7个训练点下过程系数放大 |
| 三个有理式的奇点处理 | 对整个参数搜索盒的分母极点邻域做显式拒绝采样 |
| 物理精度缩放系数 | 每个重复在独立校准池搜索，包含物理噪声并记录误差 |

拒绝采样尤其重要：论文同时声明零均值多元正态输入和含 `1/x3`、
`1/(x4*x5)` 的函数，却未说明奇点处理。代码没有静默截断输出，而是把拒绝规则
集中放在 `simulation._stability_masks` 和 `simulation._sample_inputs` 中，并在
`conditions.csv` 保存接受率和输出分位数，便于敏感性分析。

## 真实案例边界

案例 B（点焊熔核直径）数据已获取（SAVE 1.0 官方，`../datasets/case_b_spotweld/`），
复现协议已全链路实现并完成三轮迭代实验；**论文 Table V 的 PWL 优势未复现**，
根因已分层归因——详见 `reports/案例B复现/案例B复现报告.md`（结果归档于
`results/legacy/case_b_*`，该线当前暂停展开）。案例 A 的 35 组实验数据与
原作者实际基函数仍未公开，论文 Table III–VI 不能诚实地逐点重建。核心估计器
`PWLRegressor.fit(...)` 已把数据和物理模型作为公开接口；拿到数据后只需提供：

```python
model.fit(
    x_ph, x_pr, y,
    weak_x_ph, weak_y,
    physics_model=my_physics_model,
)
```

其中 `my_physics_model(x_ph, theta)` 返回给定校准参数下的物理模型输出。

## 验证标准

代码层验证：

1. 所有生成数据有限且固定种子可复现；
2. `H(x, theta)g == c + G@theta` 达到浮点误差精度；
3. consensus ADMM 在已知二次问题上恢复解析解；
4. 端到端训练产生有限预测、校准参数不越界；
5. 每次实验保留配置、种子、真实/估计 `theta`、残差、收敛状态与依赖环境；
6. `quality_checks.csv` 检查指标有限性、PWL收敛率、无效候选率、IV-B校准误差，
   以及IV-A端点下降、IV-B精度排序和IV-C监督交点是否实际出现。

论文层验证应关注趋势而非强行匹配数值：小样本时 PWL 优势、物理精度降低时
验证选择减小 `lambda1`、以及 30 个标签 + 100 个物理弱标签的等效监督样本量。
数值不一致时，优先检查本页表格中的“论文缺失信息”，而不是把差异归因于算法。
