# PWL：论文复现与热传导迁移

本仓库提供 Physics-Informed Weakly-Supervised Learning（PWL）的独立研究实现：将物理弱标签、少量标记数据和过程差异补偿结合，用于质量预测。内容包括论文仿真、点焊案例 B，以及一维稳态热传导迁移实验。

**当前整理分支：`HeatTest`。** 本项目是论文数学结构与优化方法的工程复现，不是原作者官方实现，也未严格复现论文全部数值。

## 两条实验线

| 部分 | 内容 | 入口 |
|---|---|---|
| 论文复现 | 通用 PWL 核心、Section IV 三组仿真、点焊案例 B | [reproduction/README.md](reproduction/README.md) |
| 热传导迁移 | 40 批数据、PWL/灰盒/GP 对照、S1 及机制分层 | [migration/README.md](migration/README.md) |

## 获取与运行

需要 Git、uv 和 Python 3.11（项目声明支持 Python ≥3.11、<3.14；本次验证使用 3.11）。在已安装 uv 的终端中执行；命令使用单行格式，适用于 PowerShell 与常见 Unix shell。

```sh
git clone --branch HeatTest https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning.git
cd Physics-Informed-Weakly-Supervised-Learning
uv sync --locked --python 3.11 --extra dev
uv run pytest
uv run python reproduction/run_reproduction.py --config reproduction/configs/smoke.yaml --output reproduction/results/quickstart
uv run python migration/run_migration.py --config migration/configs/heat_smoke.yaml --output migration/results/quickstart
```

两个包共享根目录的 `pyproject.toml` 和 `uv.lock`，所有命令均从仓库根目录运行。已有 CSV 足以运行以上流程；不需要安装 COMSOL、ANSYS 或 R。环境首次安装需要下载 Python/依赖。

## 当前结论

- **算法资产**：可复用的特征接口、PWLRegressor、BCD–ADMM、实验调参与统计管线。
- **复现边界**：论文存在未公开实现选择；案例 B 的 Table V 优势未复现。
- **热传导 S1**：批 21–40 的 @120 RMSE 为 A4 **4.727 K**、PWL **5.449 K**。当前场景中，带强机理先验的灰盒 A4 更合适；PWL 对 GP/Physics 的 Holm 校正后优势证据不足。见 [S1 报告](migration/reports/迁移后续工作/PWL盲测S1报告.md)。
- **后续研究**：G0/G1 是开发期证据，仍待新批确认；叶轮离线加热配方代理处于方案设计阶段。

完整证据、历史基线与后续优先级见 [当前有用内容与后续主线](docs/CURRENT_STATUS.md)。不要将早期“12 档均值最优”推广到包含 A4 的最新对照集合。

## 数据与可复现性

- [点焊数据来源](datasets/case_b_spotweld/README.md)：SAVE 1.0 发布数据，保留原始来源材料。
- [热传导数据说明](migration/datasets/README.md)：批 1–20 的规格与来源；批 21–40 的仓内生成器、验收及状态见 [迁移说明](migration/README.md)。
- 代码、配置、CSV 输入、验收报告、人工报告及依赖锁进入 Git；运行结果、缓存和日志默认不上传，人工冻结的 [v3 基线卡](migration/results/heat_v3_qint_min/BASELINE.md) 是例外。
- 历史报告中的 `results/` 图表需要在本地生成；公开仓库并不包含每次历史运行的完整产物。

2026-09-11 整理验证：70 项测试、30 项独立数值检查通过，两条 smoke 流程完成。具体环境与限制见 [整理记录](docs/CONSOLIDATION.md)；远端 CI 状态以 [Actions](https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning/actions) 为准。

## 文档与反馈

- [架构与扩展接口](docs/ARCHITECTURE.md)
- [论文复现技术说明](reproduction/REPRODUCTION.md)
- [汇报资料索引](docs/PWL组会汇报资料/README.md)
- [参考材料与论文解析](references/README.md)

复用研究结果时，请区分原论文、第三方数据来源与本仓库的工程假设，并记录分支/提交号、配置、批次和随机种子。报告问题可提交 [Issue](https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning/issues)，附上命令、Python 版本、配置及最小错误信息。修改模型或实验协议时，请同步相关测试和报告中的结论边界。
