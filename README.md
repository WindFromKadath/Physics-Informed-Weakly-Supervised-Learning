# PWL：论文复现与热传导迁移

本仓库提供 Physics-Informed Weakly-Supervised Learning（PWL）的独立研究实现：将物理弱标签、少量标记数据和过程差异补偿结合，用于质量预测。本仓库仅包含论文 Section IV 仿真复现与实际使用的一维稳态热传导测试。

**当前整理分支：`HeatTest`。** 本项目是论文数学结构与优化方法的工程复现，不是原作者官方实现，也未严格复现论文全部数值。

## 相关论文

**Dhari F. Alenezi, Michael Biehler, Jianjun Shi, and Jing Li.**  
*Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes.*  
**IEEE Transactions on Automation Science and Engineering**, vol. 22, pp. 2006–2019, 2025.  
DOI：[10.1109/TASE.2024.3374098](https://doi.org/10.1109/TASE.2024.3374098) · [作者团队提供的论文 PDF](https://sites.gatech.edu/jianjun-shi/files/2025/02/Physics-Informed_Weakly-Supervised_Learning_for_Quality_Prediction_of_Manufacturing_Processes.pdf) · [作者发表列表](https://sites.gatech.edu/jianjun-shi/publications/)

论文于 2024 年 3 月在线发表，收录于 2025 年第 22 卷，因此 DOI 中的年份与卷年不同。本文引用使用正式卷年 2025。

### 研究问题与核心方法

制造过程的最终质量标签通常需要昂贵甚至破坏性的检测，真实标记数据较少；物理模型能提供低成本预测，但会受到参数不确定性与模型偏差影响。论文提出 PWL，将物理模型输出作为弱标签，与少量真实标签共同训练质量预测模型。

本项目沿用的预测结构为：

$$
\hat y = H(x^{\mathrm{ph}},\theta)g + B(x^{\mathrm{pr}},x^{\mathrm{ph}})d.
$$

- `H g`：利用物理相关变量与校准参数构造特征，从物理弱标签中学习预测关系。
- `B d`：利用过程变量等信息补偿物理模型不能充分解释的差异。
- `theta`：参与联合学习的物理模型校准参数，其可识别性需要单独验证。
- 训练结合真实标签拟合、物理弱监督拟合、L1 与组稀疏正则；通过块坐标下降（BCD）交替更新参数，并使用 ADMM 求解相应子问题。

方法细节见原论文；本项目的实际目标函数、稳定项和工程假设见 [复现技术说明](reproduction/REPRODUCTION.md#数学实现对应)。

### 原论文与本仓库的对应关系

| 内容 | 论文关注的问题 | 本仓库对应位置 |
|---|---|---|
| PWL 建模与优化 | 物理弱标签、差异补偿和参数联合估计 | `reproduction/src/pwl_repro/core/` |
| Section IV-A | 标记样本数变化时的预测性能 | `--mode sample-size` |
| Section IV-B | 物理模型精度变化对学习效果的影响 | `--mode physics-accuracy` |
| Section IV-C | 物理弱监督能否减少真实标签需求 | `--mode label-savings` |
| 一维稳态热传导测试 | 本项目对 PWL 的独立迁移与适用性检验 | `migration/`；不是原论文实验 |

论文仿真的数据生成采用式 (9)–(11)，具体划分、相关性目标和运行配置见 [Section IV 复现协议](reproduction/SECTION_IV_PROTOCOL.md)。原论文未公开的基函数、输入协方差与数值处理等细节，由本项目显式补全；论文报告的优势不能直接视为本仓库已经取得的结果。原论文其他实验案例不属于本项目保留范围。

### 论文引用

引用原方法时可使用以下 BibTeX；引用本仓库的复现或热传导结果时，还应注明实际使用的提交号、配置和报告。

```bibtex
@article{alenezi2025pwl,
  author  = {Alenezi, Dhari F. and Biehler, Michael and Shi, Jianjun and Li, Jing},
  title   = {Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2025},
  volume  = {22},
  pages   = {2006--2019},
  doi     = {10.1109/TASE.2024.3374098}
}
```

## 两条实验线

| 部分 | 内容 | 入口 |
|---|---|---|
| 论文复现 | PWL 核心、Section IV 三组仿真 | [reproduction/README.md](reproduction/README.md) |
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

两个包共享根目录的 `pyproject.toml` 和 `uv.lock`，所有命令均从仓库根目录运行。已有 CSV 足以运行以上流程；不需要安装 COMSOL。环境首次安装需要下载 Python/依赖。

## 当前结论

- **算法资产**：可复用的特征接口、PWLRegressor、BCD–ADMM、实验调参与统计管线。
- **复现边界**：仅复现论文仿真与优化结构；论文未公开的实现选择已记录为工程假设。
- **热传导 S1**：批 21–40 的 @120 RMSE 为 A4 **4.727 K**、PWL **5.449 K**。当前场景中，带强机理先验的灰盒 A4 更合适；PWL 对 GP/Physics 的 Holm 校正后优势证据不足。见 [S1 报告](migration/reports/迁移后续工作/PWL盲测S1报告.md)。
- **热传导分层测试**：G0/G1 是开发期证据，不能作为新批确认结果。

完整证据、历史基线与后续优先级见 [当前有用内容与后续主线](docs/CURRENT_STATUS.md)。不要将早期“12 档均值最优”推广到包含 A4 的最新对照集合。

## 数据与可复现性

- [热传导数据说明](migration/datasets/README.md)：批 1–20 的规格与来源；批 21–40 的仓内生成器、验收及状态见 [迁移说明](migration/README.md)。
- 代码、配置、CSV 输入、验收报告、人工报告及依赖锁进入 Git；运行结果、缓存和日志默认不上传，人工冻结的 [v3 基线卡](migration/results/heat_v3_qint_min/BASELINE.md) 是例外。
- 历史报告中的 `results/` 图表需要在本地生成；公开仓库并不包含每次历史运行的完整产物。

验证方式：在本地运行两部分测试与独立数值检查，见 [当前状态](docs/CURRENT_STATUS.md)。本分支不配置 GitHub Actions 自动测试。

## 文档与反馈

- [架构与扩展接口](docs/ARCHITECTURE.md)
- [论文复现技术说明](reproduction/REPRODUCTION.md)
- [参考材料与论文解析](references/README.md)

复用研究结果时，请区分原论文、第三方数据来源与本仓库的工程假设，并记录分支/提交号、配置、批次和随机种子。报告问题可提交 [Issue](https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning/issues)，附上命令、Python 版本、配置及最小错误信息。修改模型或实验协议时，请同步相关测试和报告中的结论边界。
