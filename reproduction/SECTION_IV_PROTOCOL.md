# 第 IV 节三个仿真实验的复现协议

本文档描述代码实际执行的 Section IV 协议。论文没有公开基函数、完整超参数
网格、输入协方差、物理噪声方差和有理函数奇点处理，因此这里区分“论文规定”
与“复现补全”，避免把补全配置误称为作者原始配置。

## 公共数据生成

论文规定：

- \(x=(x^{ph},x^{pr})\) 来自零均值多元正态分布；
- \(\theta\in(0,1]^2\)；
- \(q_1=3,q_2=2,q_3=2\)；
- 使用式 (9)–(11)；
- SNR=5；
- 指标为MSE、RMSE、MAE。

复现补全：

- 默认 \(\Sigma_{ij}=0.9^{|i-j|}\)。论文未公开协方差；这里选择0.9是为了使
  IV-B的0.85目标相关性在当前物理函数与差异函数下可达，配置中显式记录；
- 每次重复抽取一个系统级全局 \(\theta\)；
- 默认使用 `reject_near_pole`：输入不仅在真实参数点，而且在整个
  \([0,1]^2\) 参数搜索盒内都必须远离式 (9)–(11) 的分母极点；
- 输出噪声尺度在独立参考总体上估计，因此不会随当前标签池大小漂移；
- 物理噪声标准差默认为输出噪声标准差的0.25倍；
- `B` 默认使用可精确张成式 (11) 过程项的3维紧凑基；
- 标准化后的 `B` 系数使用 `d_ridge=10` 稳定小样本估计；这是论文未公开
  `B` 维度后必须显式补全的正则选择；
- 超参数采用“物理优先的一标准误差规则”：与最低验证MSE相差不超过其标准误差
  时，选择物理约束和稀疏正则更强的候选，降低3个验证点造成的选择方差；
- 完整网格的候选彼此独立，论文/单次配置默认使用6个 joblib 进程；每个工作
  进程限制为1个BLAS线程，并且只返回指标而不保留全部拟合模型；
- ADMM采用原始/对偶残差联合停止与自适应 `rho`，BCD还同时检查目标值和参数变化；
- 默认不在验证选参后合并训练集和验证集：
  `refit_on_train_validation: false`。

每次重复保存数据指纹 `dataset_id`、随机种子、真实/估计 \(\theta\)、噪声尺度、
拒绝采样诊断、选中超参数、BCD/ADMM残差、收敛状态和逐测试样本预测。

## IV-A：不同标签样本量

论文协议为10至120个标签、步长10、100个物理样本、200个独立测试样本、
70/30训练验证划分、20次重复。

代码每次重复只生成一次：

```text
标签池120 + 物理弱标签100 + 测试集200
```

场景10、20、…、120均从同一标签池逐步扩展。每新增10个标签时，7个固定进入
训练集、3个固定进入验证集，所以不同场景之间的训练和验证样本分别保持嵌套。
全部模型共享相同测试集、物理数据、真实参数和噪声。

统计检验使用同一重复编号之间的配对单侧t检验：

```text
H1: metric(PWL) < metric(competing model)
```

同时输出Holm多重比较校正后的p值。

对应制品：

- `figure_iv_a_rmse.png`
- `figure_iv_a_boxplots.png`
- `table_iv_a.csv`

## IV-B：物理模型精度

论文规定40个标签、100个物理样本，并通过差异项幅度控制相关性为
0.85、0.70、0.50。

代码为每个重复和该重复抽到的系统参数，在独立Monte Carlo校准池上求差异倍数
\(c_{H,r},c_{M,r},c_{L,r}\)。校准相关性包含物理噪声，但不使用正式的40个标签，
从而既匹配该系统，又避免测试/训练泄漏。每次重复的H/M/L三档共享完全相同的
标签、物理噪声和测试数据，只改变：

```text
eta = eta_true + c_level * delta + epsilon_eta
```

监督基线每次重复只训练一次；PWL分别训练为PWL-H、PWL-M和PWL-L。条件文件
同时记录目标相关性、校准池相关性和每次重复实际实现的相关性。

PWL-L与每个监督基线使用配对单侧t检验，并输出Holm校正。

对应制品：

- `figure_iv_b_accuracy.png`
- `table_iv_b.csv`

## IV-C：标签节省

该实验不是独立生成一组不断增加标签的PWL实验。代码直接复用IV-A：

- PWL固定取场景3，即30标签+100物理样本；
- 监督模型使用30、40、…、110标签；
- 每个标签量选择20次重复中平均测试MSE最低的监督模型；
- PWL结果在曲线上复制为固定水平线，但底层预测只来自30标签场景。

论文风格统计量为配对单侧检验，判断监督模型何时不再显著差于PWL。由于
“未显著不同”不等价于“统计等效”，代码还提供TOST检验；默认等效界限为
PWL平均MSE的±10%，可通过 `equivalence_margin_fraction` 修改。

对应制品：

- `figure_iv_c_label_savings.png`
- `table_iv_c.csv`

## 原始输出

每次运行均生成：

| 文件 | 内容 |
|---|---|
| `results.csv` | 每个模型、场景和重复的指标与超参数 |
| `predictions.csv.gz` | 每个测试样本的真实值和预测值 |
| `conditions.csv` | 数据种子、精度系数及共享条件 |
| `statistical_tests.csv` | 配对检验、Holm校正和TOST |
| `summary.csv` | 均值与标准差 |
| `quality_checks.csv` | 数值、收敛、校准及 IV-A/B/C 论文趋势质量门槛 |
| `metadata.json` | 完整YAML配置与运行环境 |

## 运行层级

协议级快速检查：

```powershell
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/smoke.yaml `
  --mode all `
  --output reproduction/results/section_iv_smoke
```

中等规模趋势诊断：

```powershell
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/diagnostic.yaml `
  --mode all `
  --output reproduction/results/section_iv_diagnostic
```

完整网格的单次全流程：

```powershell
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/paper_single_run.yaml `
  --mode all `
  --output reproduction/results/section_iv_single_full
```

论文规模：

```powershell
uv run python reproduction/run_reproduction.py `
  --config reproduction/configs/paper_protocol.yaml `
  --mode all `
  --output reproduction/results/section_iv_full
```

论文规模配置包含完整 \(7^3\) PWL网格，约需102,900次PWL拟合，执行前应先用
`diagnostic.yaml` 完成数值尺度、趋势和收敛性预检。
