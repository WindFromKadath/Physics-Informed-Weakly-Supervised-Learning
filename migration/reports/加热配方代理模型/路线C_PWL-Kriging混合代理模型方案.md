# 路线 C：PWL–Kriging 混合代理模型方案

> 版本：v1.1
> 日期：2026-08-11
> 更新：2026-08-12 补充相关工作检索结论与对标基线（§10；文献 10–16）
> 状态：设计方案，尚未实施
> 上游准入要求：[加热配方代理模型必备条件](../../../加热配方代理模型必备条件.md)
> 对照路线：[路线 A：离线加热配方代理模型实施方案](路线A_离线加热配方代理模型实施方案.md)、
> [路线 B：在线 MPC 动态 PWL 实施方案](路线B_在线MPC动态PWL实施方案.md)

---

## 0. 执行结论

路线 C 是一种**模型形态方案**，而不是第三条用途路线：保持 PWL 的均值通道不动，
为其外挂一个克里金（Kriging）式的统计层，使代理模型同时输出预测均值与预测方差：

\[
\hat y(\mathbf x)=\mu_{\rm PWL}(\mathbf x),
\qquad
s^2(\mathbf x)=\sigma^2_{\rm channel}(\mathbf x).
\]

- **均值通道**：\(\mu_{\rm PWL}(\mathbf x)=H(x^{ph},\theta)g+B(x^{ph},x^{pr})d\)，
  完全沿用当前 `PWLRegressor`，不改变 v3 主配置、求解器和选参规则；
- **方差通道**：在独立校准集上对残差 \(r_i=y_i-\mu_{\rm PWL}(\mathbf x_i)\) 建立
  统计层，首期两个候选臂：残差 GP（V1）与 Bootstrap 集成 + 保形校准（V2）。

**可行性判断：有条件可行，建议作为路线 A 的 UQ 环节首发验证。**

依据有两条：

1. **结构同构**：通用克里金为"趋势项 + GP 残差"，PWL 为"物理蒸馏趋势 + 参数化
   差异项"，两者是同一分解的不同实例化。为 PWL 补方差通道，在数学上就是把它
   补齐为一个"物理趋势 + 参数化残差 + 校准方差"的通用克里金类代理。
2. **实证边界**：热传导 20 批终态 PWL @120 RMSE 5.351 优于 Physics-GP 的 5.648
   （见[阶段性总结](../../../阶段性总结_PWL算法实现框架.md) §5.2、
   [v3 事实基线](../../results/heat_v3_qint_min/BASELINE.md)与
   [20 批统计闭环报告](../迁移后续工作/COMSOL场景20批统计闭环报告.md)）。Physics-GP
   本身就是"物理趋势克里金"形态，其精度劣势说明**不应全盘 GP 化**——混合形态
   保留 PWL 均值、只外包方差，正是为了在不牺牲该精度优势的前提下获得克里金的
   UQ 与外推预警能力。

路线 C 不负责在线状态反馈，也不应被表述为已实现 MPC；其动态化扩展仅在 §9 预留
接口设计，实施属于路线 B 的范畴。

---

## 1. 动机与定位

### 1.1 对应必备条件中的哪些条目

对照[必备条件](../../../加热配方代理模型必备条件.md)：

| 条目 | PWL 现状 | 路线 C 的作用 |
|---|---|---|
| 6 输出平滑（可微更佳） | 已满足：基函数解析连续，拟合后预测为基函数线性组合 | 不改动 |
| 7 外推行为受控且可诊断 | 物理趋势外推优于纯 GP（退回均值），但无内置包络诊断 | 方差通道随距校准数据增大，提供"克里金式"外推预警信号；与距离类包络诊断联用 |
| 8 提供不确定性量化 | 确定性点估计，无方差通道 | **本方案的核心增量**：校准后的预测区间，支撑安全裕度 \(\beta\sigma\) 与采集函数 |

克里金在必备条件 6.1/6.2 中被点名的两项天然优势（预测方差、外推退化可诊断），
正是 PWL 当前的两处缺口；路线 C 以最小结构改动补齐这两项。

### 1.2 与全盘克里金化（Physics-GP 形态）的分界

Physics-GP 基线中 GP 同时承担均值修正与方差估计；热传导场景证据表明该形态精度
劣于参数化 \(Bd\)。路线 C 的纪律是：

- GP/统计层**只允许接触残差，不允许修正均值**。若校准集上残差 GP 的均值显著
  偏离 0，说明均值通道存在结构缺失，应按迁移门禁回报并修复 \(H/B\)，而不是让
  GP 静默吸收；
- 均值通道的任何改动仍走原有消融与统计闭环流程，方差通道不得成为绕过
  模型类检查（Gate 4）的捷径。

### 1.3 与论文算法的边界

混合统计层是工程包装，不属于 PWL 论文（Alenezi 等，2025）的公开算法内容，
对外报告时必须分开表述，口径与路线 A §7.2 一致。

---

## 2. 数学结构

### 2.1 均值通道（不变）

\[
\mu_{\rm PWL}(\mathbf x)=H(x^{ph},\theta)g+B(x^{ph},x^{pr})d,
\]

训练流程、BCD–ADMM 求解、\(\theta\) 刷新模式与场景化选参规则（热传导 minimum、
仿真 1-SE）全部沿用现状。\(\theta\) 维持 nuisance parameter 口径，不进入方差
解释。

### 2.2 方差通道候选臂

设独立校准集 \(\mathcal C=\{(\mathbf x_i,y_i)\}_{i=1}^{n_c}\)，残差
\(r_i=y_i-\mu_{\rm PWL}(\mathbf x_i)\)。

**V1：残差 GP（克里金式解析方差）**

\[
r(\mathbf x)\sim\mathcal{GP}\!\big(0,\;k(\mathbf x,\mathbf x';\psi)\big),
\qquad
s^2(\mathbf x)=k(\mathbf x,\mathbf x)-\mathbf k^\top(K+\sigma_n^2 I)^{-1}\mathbf k
+\sigma_n^2 .
\]

- 核型首期取 Matérn-5/2（光滑性假设可审计，与必备条件第 6 条一致）；
- 输入为标准化后的联合输入 \((x^{ph},x^{pr})\)，长度尺度设上界约束，防止核把
  \(H\) 已表达的长波长趋势再次吸收（防 H/GP 混淆，对应项目既有 H/B 重叠教训）；
- 均值固定为零，超参数 \(\psi\) 只在校准集上拟合。

**V2：Bootstrap 集成 + 保形校准（路线 A §6.1 既定方案的正式化）**

1. 按配方组/批次整组重采样训练集，训练 \(M\) 个 PWL 成员；
2. 成员离散度给出 \(s^2_{\rm ens}(\mathbf x)\) 与分位数；
3. 在独立校准集上做保形修正，报告每组覆盖率。

**V3：常量/分区方差（工程占位）**

校准残差的全局或分区标准差，仅作占位与下界对照，文档中必须标注不是概率安全
保证（口径同路线 A §3.2）。

### 2.3 与通用克里金的对应关系

| 通用克里金 | 路线 C 混合体 |
|---|---|
| 趋势项 \(\mathbf f(\mathbf x)^\top\boldsymbol\beta\) | \(H(x^{ph},\theta)g\)（物理蒸馏趋势，含校准参数） |
| GP 残差的均值修正 | \(B(x^{ph},x^{pr})d\)（稀疏参数化差异项，实证优于 GP 均值修正） |
| GP 残差的方差 | 方差通道 \(s^2(\mathbf x)\)（V1/V2），仅校准、不修正均值 |

---

## 3. 关键设计决策与纪律

1. **校准数据独立**：方差通道只允许在独立校准集上拟合，禁止复用 PWL 的训练
   残差（训练残差被惩罚项压缩，直接复用将系统性低估方差）。嵌套划分设施
   （train/validation/test/校准池）已在 `pwl_repro` 数据层存在，直接复用。
2. **H/GP 混淆监测**：报告校准集残差 GP 的拟合均值与有效自由度；均值显著非零
   或长度尺度撞上界时，按"均值通道结构缺失"回报，走 Gate 4 无噪声类偏置检查，
   禁止静默吸收。
3. **覆盖率校准**：区间给出后必须用独立数据做经验覆盖率检验（90%/95% 名义
   水平）；保形区间的可交换性假设在跨批次、跨钢种或时间漂移下需分组重校准
   （口径同路线 A §6.1）。
4. **多输出**：每个约束指标一个独立混合模型 + 一致性后处理，继承路线 A §5.4
   的安排；方差通道暂不建模跨输出协方差，列入后续升级。
5. **\(\theta\) 口径**：沿用 M2 定案，\(\theta\) 为 nuisance parameter，不参与
   方差通道的物理解释。
6. **对外表述**：混合体的均值、方差、采集函数三层必须分别标注来源（论文算法 /
   工程包装 / 代理优化策略），禁止合并表述为"PWL 已具备概率输出"。

---

## 4. 验证实验设计（接入迁移门禁）

### 4.1 首发场景

先在**现有热传导静态场景**做离线验证：数据、冻结基线、消融臂和统计设施全部
现成，可在不引入配方参数化变量的前提下，单独归因"方差通道"这一新增量。通过
后再嵌入路线 A 的配方场景（对应路线 A 的 A4 阶段）。

### 4.2 实验臂

| 臂 | 构成 | 回答的问题 |
|---|---|---|
| M0 | 纯 PWL 点估计（现状） | 均值精度基准 |
| M1 | PWL + 残差 GP（V1） | 解析方差是否校准、均值是否受损 |
| M2 | PWL + Bootstrap + 保形（V2） | 无核假设路线是否足够 |
| M3 | Physics-GP（现有基线） | GP 均值修正形态对照 |
| M4 | 纯 GP（现有基线） | 无物理趋势对照 |

已发表形态与各臂的对标关系见 §10.2（PGBML↔M2、物理增强 GP↔M1/M3、
轧钢物理信息回归↔路线 A 场景）。

### 4.3 验收指标

- **均值保真**：M1/M2 的 RMSE 相对 M0 不劣化（预注册容差，建议配对检验）；
- **区间质量**：90%/95% 名义水平的经验覆盖率、平均区间宽度，按域内/边界/域外
  分层报告；
- **外推预警**：域外查询点的 \(s^2\) 单调性诊断与 `in_domain` 触发一致性；
- **混淆检查**：校准集残差 GP 均值与长度尺度诊断（§3 第 2 条）；
- **统计口径**：沿用 20 批配对、效应量、Holm 校正与预注册模板（M4 闭环惯例），
  结论表述遵守同一终态表述规则。

---

## 5. 接口与代码结构（建议）

```python
class HybridPWLSurrogate:
    def fit_mean(self, train, valid): ...          # 复用 PWLRegressor
    def fit_variance(self, calibration): ...       # V1 或 V2，只用校准集
    def predict_distribution(self, x): ...         # (mean, var, quantiles)
    def in_domain(self, x): ...                    # 距离分数 + 触发原因
```

- 新模块：`migration/src/pwl_migration/hybrid_surrogate.py`（首期放场景包，
  接口稳定且第二场景复用后再考虑下沉 `pwl_repro.core`，纪律同路线 B §12）；
- 配套：`configs/heat_hybrid_smoke.yaml`、`tests/test_hybrid_surrogate.py`、
  结果目录 `migration/results/heat_hybrid_v1/`（含 BASELINE 登记、覆盖率报告、
  混淆诊断表）；
- 通用估计器与嵌套划分继续复用 `pwl_repro.core` 与 `experiment_api`，方差通道
  不得反向写入通用核心。

---

## 6. 分阶段实施计划

| 阶段 | 工作 | 交付物 | 通过门槛 |
|---|---|---|---|
| C0 | 冻结用途与数据合同 | 校准集定义、划分规则、预注册指标 | 校准数据独立于训练/验证 |
| C1 | 热传导 V1/V2 双臂实现 | 模块、测试、smoke 配置 | 与 M0 均值一致（机器精度） |
| C2 | 覆盖率与混淆诊断 | 覆盖率报告、GP 均值/长度尺度表 | 覆盖率达预注册目标；无静默吸收 |
| C3 | 域外分层与包络联调 | 域内/边界/域外分层报告 | 域外方差预警有效 |
| C4 | 嵌入路线 A 优化闭环 | 约束收紧 \(\mu+\beta\sigma\) 接入、候选复核记录 | 假安全率不劣于固定裕度占位 |
| C5 | 路线 B 接口预留（仅设计） | `predict_distribution` 时域传播接口说明 | 不实施动态化 |

C2 不通过则先回到均值通道做 Gate 4 检查，不得用更灵活的核继续掩盖；C4 未完成
前，不得宣称"已可用于配方寻优"。

---

## 7. 主要风险与应对

| 风险 | 后果 | 应对 |
|---|---|---|
| 校准集不足 | 方差低估、覆盖不达标 | 最小校准样本门禁；不足时只允许 V3 占位并显式标注 |
| GP 吸收均值结构缺陷 | 掩盖 \(H/B\) 缺失方向 | §3 第 2 条混淆监测，强制回报走 Gate 4 |
| 覆盖率随漂移失效 | 安全裕度失真 | 分组校准、定期重校准、漂移监控（沿用 M2/M4 判据纪律） |
| GP 计算成本随样本增长 | 优化闭环内评估变慢 | 校准集子集化/稀疏 GP 选项；均值通道评估成本不受影响 |
| 混合区间被当作概率安全保证 | 安全论证失真 | 文档口径纪律：校准区间只在验证过的包络与群体上有效 |
| 方差通道被用来规避模型类修复 | 技术债 | 均值/方差改动分开评审，方差通道 PR 不得改动均值配置 |

---

## 8. Go / No-Go 判据

### 可以进入路线 C 原型开发

- 热传导冻结基线与 20 批统计设施可用（现状满足）；
- 能划出独立于训练/验证的校准数据池；
- 区间覆盖率目标与使用场景（路线 A 约束收紧）已明确。

### 应暂停

- 试图用方差通道修正均值精度问题；
- 无独立校准数据，或计划用训练残差直接拟合方差；
- 期望路线 C 直接提供动态/MPC 能力（属于路线 B 范畴）。

---

## 9. 与路线 A/B 的关系

- **路线 A**：路线 C 是其 §6.1 UQ 环节的候选实现之一（V2 已在路线 A 规划内，
  V1 为新增的解析方差臂）；C4 阶段即路线 A 的 A4–A5 接入点。
- **路线 B**：动态化所需的时域不确定性传播、状态—控制联合信赖域，按路线 B
  §9 的架构实施；路线 C 只承诺 `predict_distribution` 接口形态，不承诺动态
  能力。
- **v4 研究项**：active-set debias、refit 等均值通道升级与路线 C 正交，可并行
  立项，但必须在同一冻结基线上分别消融。

---

## 10. 相关工作检索结论与对标基线（2026-08-12）

> 检索范围：Kimi Scholar 与 Semantic Scholar API，2023–2026 年；词组覆盖
> "physics-informed weakly-supervised learning"、"physics-guided machine learning
> manufacturing quality"、"multi-fidelity physics-informed learning"、
> "small sample quality prediction prior knowledge" 等。被引数以各数据库口径
> 为准，均非 Scopus/WoS 全量；检索结果 CSV 存于 `tmp/scholar/`。

### 10.1 未发现 PWL 的直接改进实现

文献 1（PWL 原论文）在 Kimi Scholar 被引 11 次，Semantic Scholar 收录施引
文献 8 篇。逐篇核对后，8 篇均为应用或背景引用（数字孪生质量预测、弧焊/焊接
传感综述、再制造预测、超维计算、以及文献 10/11/15 等同族新算法），**不存在
对 PWL 框架本身的算法性扩展**——没有改进版目标函数、求解器或基函数设计的
发表工作。原作者团队的后续工作转向图结构波动传播建模（hybrid multi-stage
manufacturing systems, IEEE T-ASE 2025），亦非 PWL 扩展。

**含义**：路线 C 的混合形态与 v4 研究候选（refit、active-set debias、变量
投影、rank-7 H 等）当前不与已发表工作撞车；但也意味着混合体没有现成实现可
借用，必须按 §6 的 C0–C5 阶段自行完成验证。

### 10.2 同族新算法与路线 C 实验臂的对标关系

| 族 | 代表工作 | 核心机制 | 与路线 C 的关系 |
|---|---|---|---|
| 物理引导（贝叶斯）元学习 | 文献 10（PGML）、文献 11（PGBML） | MAML 元学习 + 物理先验，小样本自带 UQ | **M2/V2 臂的发表对标**：同属"点估计 + 集成/后验不确定度"，但无物理弱标签蒸馏通道；报告覆盖率与样本效率时应对照 |
| 物理增强 GP | 文献 12（GPR 伪样本增强）、文献 13（物理信息 GP + BO） | 物理伪样本/物理核增强 GP | **M1/V1 臂近亲**；其 GP 同时接触均值与方差，属 §1.2 禁止形态，可作 M3 臂的文献例证与讨论对象 |
| 部分物理多保真 | 文献 14（Partial-physics-informed multi-fidelity） | 简化物理 + 多保真迁移 | 与 H+B 结构思想最接近的发表形态；其保真间函数相似性约束对应迁移门禁 Gate 3–4 |
| 钢铁轧制物理信息回归 | 文献 15（Physics-informed generative regression） | 钢带轧制过程物理信息生成式回归 | **路线 A 场景对标**：加热/轧钢方向最贴近的已发表工作，C4 阶段应纳入基线比较 |
| 物理信息少样本迁移 | 文献 16（PI few-shot，跨材料 LPBF） | 少样本跨材料/跨工况迁移 | 路线 A 跨钢种/跨规格泛化时的对照 |

### 10.3 引用与表述纪律

- 对外报告中，路线 C 必须同时引用文献 1（PWL 原型）与文献 8–9（克里金/GP
  方法学），并明确说明混合体本身无直接发表先例；
- 文献 10–16 的题录来自 2026-08-12 学术检索数据源，卷期页码以出版社页面
  为准；精读原文前不得引用其具体数值结论；
- 若后续检索出现 PWL 的直接扩展论文，本节结论必须复核更新。

---

## 11. 参考文献与用途说明

以下文献中 1–7 已在路线 A/B 文档中核对过页面或 DOI；8–9 为克里金与 GP 代理的
经典方法学依据；10–16 为 2026-08-12 学术检索新增的同族新算法（见 §10），
题录与链接来自检索数据源，卷期页码以出版社页面为准。它们支持方法设计，
不构成路线 C 已经实验验证的证据。

1. Alenezi, D. F., Biehler, M., Shi, J., & Li, J. (2025).
   *Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes*.
   IEEE T-ASE, 22, 2006–2018. [DOI: 10.1109/TASE.2024.3374098](https://doi.org/10.1109/TASE.2024.3374098)。
   **用途**：均值通道算法原型；论文不含概率输出，混合层属工程包装。
2. Kennedy, M. C., & O'Hagan, A. (2000). *Predicting the output from a complex
   computer code when fast approximations are available*. Biometrika, 87(1), 1–13.
   [DOI: 10.1093/biomet/87.1.1](https://doi.org/10.1093/biomet/87.1.1)。
   **用途**：多保真"趋势 + 差异"联合建模的经典依据。
3. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). *Efficient Global
   Optimization of Expensive Black-Box Functions*. Journal of Global Optimization,
   13, 455–492. [DOI: 10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147)。
   **用途**：均值+方差驱动采集函数（EGO）的框架来源，路线 C 方差通道的主要
   消费场景。
4. Gardner, J. R., et al. (2014). *Bayesian Optimization with Inequality
   Constraints*. PMLR 32. [论文页](https://proceedings.mlr.press/v32/gardner14.html)。
   **用途**：约束收紧与可行概率设计参考。
5. Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile
   Regression*. NeurIPS 2019. [论文页](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)。
   **用途**：V2 臂保形校准依据；跨群体使用须重查可交换性。
6. Forrester, A. I. J., Sóbester, A., & Keane, A. J. (2007). *Multi-fidelity
   optimization via surrogate modelling*. Proc. R. Soc. A, 463, 3251–3269.
   [DOI: 10.1098/rspa.2007.1900](https://doi.org/10.1098/rspa.2007.1900)。
   **用途**：代理模型与高保真复核闭环管理参考。
7. Hewing, L., Kabzan, J., & Zeilinger, M. N. (2020). *Cautious Model Predictive
   Control Using Gaussian Process Regression*. IEEE TCST, 28(6), 2736–2743.
   [DOI: 10.1109/TCST.2019.2949757](https://doi.org/10.1109/TCST.2019.2949757)。
   **用途**：方差传播与机会约束思想参考；路线 B 阶段才适用。
8. Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989). *Design and
   Analysis of Computer Experiments*. Statistical Science, 4(4), 409–423.
   **用途**：克里金作为计算机试验代理模型的奠基文献，"趋势 + GP 残差"结构
   出处。
9. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
   Machine Learning*. MIT Press.
   **用途**：残差 GP 的核型选择、超参数拟合与预测方差公式依据。
10. Hu, Z., Wang, T., Chen, H., & Zhang, K. (2026). *Physics-Guided Meta-Learning
    for Surface Roughness Prediction Under Various Working Conditions With Limited
    Data*. IEEE/ASME Transactions on Mechatronics.
    [DOI: 10.1109/TMECH.2025.3598351](https://doi.org/10.1109/TMECH.2025.3598351)。
    **用途**：V2 臂的发表对标；MAML 元学习 + 物理先验的小样本质量预测，
    无物理弱标签通道。
11. Hu, Z., Fu, Y., & Liu, Y. (2026). *A physics-guided Bayesian meta-learning
    method for surface quality prediction under data-scarce industrial conditions*.
    Journal of Intelligent Manufacturing.
    [DOI: 10.1007/s10845-026-02908-1](https://link.springer.com/article/10.1007/s10845-026-02908-1)。
    **用途**：V2 臂的贝叶斯化发表对标；自带后验不确定度，覆盖率与样本效率
    应作为路线 C 报告的对照对象。
12. Nguyen, H. P., Nguyen, D. T., & Kim, J. M. (2026). *Gaussian process
    regression with physics-guided pseudo-sample augmentation for wear prediction
    under sparse measurements in milling*. Scientific Reports.
    [DOI: 10.1038/s41598-026-38067-9](https://www.nature.com/articles/s41598-026-38067-9)。
    **用途**：V1 臂近亲；其 GP 经物理伪样本同时影响均值与方差，是 §1.2 所
    禁止形态的文献例证。
13. Wang, Z., Duan, L., Kuang, L., Zhou, H., & Duan, J. (2025). *Physics-informed
    Gaussian process regression with Bayesian optimization for laser welding
    quality control in coaxial laser diodes*. Computers, Materials & Continua.
    [论文页](https://www.sciencedirect.com/org/science/article/pii/S1546221825006472)。
    **用途**：物理信息 GP + 贝叶斯优化闭环，对应 C4 阶段"约束收紧 + 采集
    函数"的已发表简化形态。
14. Cleeman, J., Agrawala, K., & Nastarowicz, E., et al. (2023).
    *Partial-physics-informed multi-fidelity modeling of manufacturing processes*.
    Journal of Materials Processing Technology.
    [论文页](https://www.sciencedirect.com/science/article/pii/S0924013623002704)。
    **用途**：与 H+B 结构思想最接近的发表形态；其保真间函数相似性约束
    对应本项目 H/B 可识别性门禁（Gate 3–4）。
15. Deng, J., Sierla, S. A., Sun, J., & Vyatkin, V. (2025). *Physics-informed
    generative regression for industrial process modeling in steel strip rolling*.
    Expert Systems with Applications.
    [DOI: 10.1016/j.eswa.2025.127713](https://doi.org/10.1016/j.eswa.2025.127713)。
    **用途**：钢带轧制加热过程的物理信息回归，路线 A 场景最贴近的已发表
    工作；路线 C 嵌入路线 A 时应纳入基线比较。
16. Dhakal, P., Kim, J. G., Hou, A., Wu, X., & Wang, S. (2026). *Physics-informed
    few-shot learning for cross-material relative density prediction in laser
    powder bed fusion*. Journal of Intelligent Manufacturing.
    [DOI: 10.1007/s10845-026-02889-1](https://link.springer.com/article/10.1007/s10845-026-02889-1)。
    **用途**：物理信息少样本跨材料迁移；跨工况泛化对照。

### 仓库内关联材料

- [加热配方代理模型必备条件](../../../加热配方代理模型必备条件.md)
- [阶段性总结_PWL算法实现框架](../../../阶段性总结_PWL算法实现框架.md)（§5.2 基线数值、§10 迁移门禁）
- [PWL 基函数设计逻辑](../../../references/docs/08_基函数设计逻辑.md)
- [当前热迁移说明](../../README.md)

---

## 12. 最终定位

> 路线 C 不把 PWL 改造成克里金，而是把克里金的统计层嫁接到 PWL 上：均值继续由
> 物理蒸馏趋势与稀疏参数化差异项承担——这是当前场景实证最优的组合；方差由只接触
> 校准残差的统计层承担——这是 PWL 进入约束寻优所缺的最后一块接口。混合体成立的
> 判据不是"形式上有方差输出"，而是均值精度不受损、区间覆盖率经独立数据校准、
> 外推预警与包络诊断一致，并且均值/方差两层在文档与评审中始终分开表述。
</content>
