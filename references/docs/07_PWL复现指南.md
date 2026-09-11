# 07 PWL 算法复现指南：细节完整性盘点与缺口补全方案

> **对象文献**：[02_PWL论文解析](legacy/02_PWL论文解析.md)（IEEE T-ASE 2025 正文）+ [03_PWL补充材料解析](legacy/03_PWL补充材料解析.md)（证明）
>
> **本文目的**：评估论文对算法细节的表述完整度，明确"可直接照抄的部分"与"需自行推断补全的部分"，给出可执行的复现方案与验证锚点。

---

## 1. 总体结论

> **算法框架级细节完整，实现级细节不完整。**
> 论文（含补充材料、学位论文第 3 章）完整给出了数学框架：模型式、优化式、BCD 主算法、ADMM 全部近端算子（闭式）、初始化策略、仿真数据生成式、实验协议。但三类关键实现细节缺失：
>
> 1. **基函数 H(x^ph, θ) 与 B(x^pr, x^ph) 的具体形式——未给出**；
> 2. **映射函数 ϕ 的具体形式——未给出**；
> 3. **全部数值型超参数（λ 网格、ω 边界、步长、阈值）——未给出**。

复现 = 论文给定框架 + 本指南 §4 的补全方案；复现正确性用 §5 的锚点校验。

## 2. 已完整表述、可直接复现的内容

### 2.1 模型与优化（正文 §III）

| 内容 | 位置 | 说明 |
|---|---|---|
| 物理模型视角 y = ϕ[η(x^ph,θ)] + ε^disc + ε | 式 (1) | 概念框架 |
| 数据驱动视角 y = f(x^ph,x^pr) + ε | 式 (2) | 概念框架 |
| 联合真值视角 y = u(x^ph,x^pr,θ) + ε | 式 (3) | 建模动机 |
| **联合模型 y = H(x^ph,θ)g + B(x^pr,x^ph)d + ε** | 式 (4) | ★ 核心模型 |
| **优化问题（4 项目标 + θ 盒约束）** | 式 (5) | ★ 核心：拟合项 + 蒸馏项 λ₁‖H₂g−ϕ(η)‖²_F + λ₂‖g‖₁ + λ₃Σ‖g_q‖₂，s.t. ω_l≤θ≤ω_u |
| 矩阵化记号 H₁∈R^{n×p}、B₁∈R^{n×b}、H₂∈R^{N×p} | 式 (7) | n=标记样本数，N=物理样本数 |

### 2.2 算法（正文 §III-D + 补充材料）

| 内容 | 位置 | 完整度 |
|---|---|---|
| **BCD 主循环**（g→d→θ 轮流更新，目标值变化<ε₀ 停止） | Algorithm 1 | ✅ |
| **ADMM 一致性算法**（并行 prox → 平均 → 对偶更新 → 收敛检查） | Algorithm 2 | ✅ |
| **g 更新**：4 个近端算子闭式解 | Proposition 1（证明见补充材料 A 节） | ✅ 全部闭式 |
| **d 更新**：最小二乘闭式解 | Algorithm 1 步骤 2 | ✅ |
| **θ 更新**：3 个近端算子闭式解（含盒约束投影） | Proposition 2（证明见补充材料 B 节） | ✅ 全部闭式 |
| **初始化**：d 取小值（让 g 先学物理模型）；θ 用两步校准结果初始化 | §III-D | ✅ |
| **两步校准法**（物理模型基线 & θ 初始化）：θ̂ = argmin Σ(y−η(x^ph,θ))²，再用 GP 建模差异（Kennedy–O'Hagan） | 式 (12) | ✅ |
| **可辨识性**：spark 条件；n ≥ C·k·log(m/k)；H 未知时 N = (k+1)/β·C(m,k) 子采样下界 | Proposition 3 | ✅ |
| **调参准则**：λ₁、λ₂、λ₃ 由验证集 MSE 最小化确定（网格搜索） | §IV-A | ✅（准则明确，网格未给） |

### 2.3 实验设置（正文 §IV–§V）

| 设置 | 细节 | 位置 |
|---|---|---|
| 仿真数据生成 | η_true（式 9）、差异 δ（式 10）、h(x^pr)（式 11）均有**显式函数式**；SNR=5；q1=3、q2=2、q3=2；θ~U(0,1]；x 来自多元正态 | §IV |
| 样本量实验 | 12 场景：10→120 标记样本；每场景 N=100 物理样本；70/30 训练/验证；200 独立测试样本；20 次随机重复 | §IV-A |
| 物理精度实验 | 相关性 0.85 / 0.7 / 0.5 三档（改 δ 幅度控制）；40 标记 + 100 物理 | §IV-B |
| 标签节省实验 | 场景 3（30 标记+100 物理）对比 30→110 标记的纯监督 | §IV-C |
| 案例 A（点焊温差） | 物理模型 **ΔT = I²·ρ·L·t/(A·m·Cp)**（式 13）；x^ph={电流,时间,质量}、x^pr={压力}、θ={ρ,Cp 等}；35 个 Latin Hypercube 过程样本 + 35 个物理样本（同变量设置）；5 折 CV × 5 次 | §V-A |
| 案例 B（熔核直径） | ANSYS 耦合 PDE → **普通克里金**（高斯相关函数）近似（35 组仿真）；12 设置×10 重复=120 过程样本；噪声方差≈0.2；10 / 100 标记两场景 | §V-B |

## 3. 未表述的缺口（复现障碍清单）

### 缺口 1（最关键）：基函数 H、B 的具体形式

- 论文原话（§III-B）：*"This is a highly flexible framework that can accommodate various types of basis functions. By choosing different basis functions, the model can be tailored to specific problem settings."*
- 全文（期刊版 + 学位论文版）**均未说明实验中实际使用的基函数类型**（多项式/RBF/样条？），也未给出维度 p、b 的取值；
- 式 (9)–(11) 是**数据生成真值**（η_true、δ、h），不是 PWL 拟合用的基——勿混淆。

### 缺口 2：θ 在 H 中的参数化方式（需反推）

- H(x^ph, θ) 中 θ 如何进入基函数，论文未说明；
- **从 Proposition 2 可推断的隐含约束**：θ 更新的近端算子 `prox_{Δf₁}[a] = (I + Δggᵀ)⁻¹(a + Δg(Y−B₁d)ᵀ)` 是最小二乘结构 → **H 必须关于 θ 为仿射（线性）**，即 H(x^ph,θ)g 对固定 g 是 θ 的线性函数。若 θ 非线性进入，闭式 prox 不成立；
- ⚠️ 记号瑕疵提醒：Proposition 2 中 "I 为 p 阶单位阵" 应为 **q₃ 阶**；ggᵀ 项实际是由 g 构造的设计矩阵（推导时需按 `H₁(θ)g = c₀ + G·θ` 整理，G∈R^{n×q₃}），复现前请自行核对维度。

### 缺口 3：映射函数 ϕ 的形式

- 仅说明 ϕ 用于处理"物理输出与质量变量高相关但不同测度"的失配（§III-C 第 2 点）；
- 实验中 ϕ 取恒等、线性还是其他形式，未说明。

### 缺口 4：数值型超参数

| 参数 | 论文表述 | 缺失内容 |
|---|---|---|
| λ₁、λ₂、λ₃ | "验证集 MSE 最小化" | 搜索网格的范围/步长 |
| ω_l、ω_u（θ 边界） | "lower and upper bounds" | 具体数值（案例 A 中 ρ、Cp 的物理区间） |
| ADMM 步长 Δ、阈值 ε | Algorithm 2 输入 "step size Δ, ε" | 取值 |
| BCD 外层 ε₀ | Algorithm 1 停止条件 | 取值、最大迭代上限 |

### 缺口 5：实验次要细节

- 仿真中协方差结构 **Σx 未说明**（对比：RWSL 论文明确用 R_{x,i,j}=a^|i−j|）；
- 物理精度实验中 **δ 的缩放系数**未给出（只给目标相关性 0.85/0.7/0.5）；
- 案例 B 中克里金仿真器**生成的物理样本数 N** 未说明；
- **无公开代码**（IEEE 页面 supplementary 仅含 2 页证明 PDF）。

## 4. 缺口补全方案（建议的复现配置）

> 本节各项选型的**文献依据**详见 [§8 基函数设计的文献依据](#8-基函数设计的文献依据参考文献图谱)。

### 4.1 基函数（缺口 1+2）

**仿真复现**（q1=3, q2=2, q3=2，与式 (9)–(11) 结构相容）：

- H(x^ph, θ)：{1, x₁, x₂, x₃, θ₁, θ₂, x₁x₂, x₁x₃, x₂x₃, x₁θ₁, x₃θ₂, …} —— 低阶单项式 + 交互项；**θ 只以线性（一次）方式出现**（满足缺口 2 的仿射约束）；
- B(x^pr, x^ph)：{1, x₁..x₅, 及二次项/交互项} 的多项式基；
- 分组方式（组 Lasso 的 g_q）：按"同一物理现象的模式"分组，如 {x_i 及其交互} 一组、{θ_j 及其交互} 一组。

**案例 A 复现**：H 由 {电流 I、时间 t、质量 m、ρ、Cp 及其交互} 构成；B 由 {压力 force + x^ph} 构成；同理保持 θ 线性。

### 4.2 映射函数 ϕ（缺口 3）

- 首选**恒等映射**（案例 A 物理输出即 ΔT 估计，相关性 0.884 已较高）；
- 备选：用标记样本对 (η̃, y) 做一维线性最小二乘拟合 ϕ（每轮 BCD 后可更新）。

### 4.3 超参数（缺口 4）

| 参数 | 建议值 |
|---|---|
| λ₁、λ₂、λ₃ | 对数网格 10⁻³~10³（各 7 点），验证集 MSE 网格搜索 |
| ADMM 步长 Δ | 1（一致性 ADMM 常用默认值） |
| ADMM 阈值 ε | 10⁻⁴ ~ 10⁻⁶，最大迭代 10³ 兜底 |
| BCD 外层 ε₀ | 10⁻⁴，最大外层迭代 50 |
| ω_l、ω_u | 按物理先验（如材料手册中 ρ、Cp 的范围）；仿真中取真值 θ~U(0,1] 的支撑 [0,1] |

### 4.4 实验次要细节（缺口 5）

- Σx：沿用 RWSL 论文的 **R_{x,i,j} = a^|i−j|**（如 a=0.5）；
- δ 缩放：对式 (10) 乘标度系数 c，网格搜索 c 使 (y, η) 样本相关性命中 0.85/0.7/0.5；
- 案例 B 的 N：取与过程样本同量级（如 120 或 100），做敏感性分析；
- 初始化：d₀ = 0（或 10⁻³ 量级小值）；θ₀ = 两步法式 (12) 的 θ̂。

## 5. 复现验证锚点（判断实现是否正确）

| 锚点 | 期望值 | 来源 |
|---|---|---|
| 案例 B，100 标记 | **PWL 的 MSE ≈ 0.206**（逼近噪声方差 0.2） | 正文 §V-B |
| 案例 A，物理模型单独使用 | 相关性 ≈ 0.884，RMSE ≈ 72.8（高相关低精度） | 正文 §V-A |
| 仿真样本量实验 | PWL 的 RMSE 在全部 12 场景一致低于 RR/SVR/DT/RF/GP/Physics；**小样本处差距最大** | 正文 §IV-A, Fig. 2 |
| 物理精度实验 | 相关 0.5 时 PWL 仍不显著差于最优基线（λ₁ 自适应调小） | 正文 §IV-B, Table II |
| 标签节省实验 | 30 标记+100 物理 ≈ 纯监督 100 标记（+233.3%） | 正文 §IV-C |
| 算法行为 | 首轮迭代后 g 主要拟合物理模型（因 d₀ 小）；ADMM 残差单调趋稳 | Algorithm 1 设计意图 |

## 6. 复现实施路线图

```
Step 1  数据生成器：按式 (9)–(11) + Σx + SNR=5 实现仿真数据生成（含 η_true、δ、h、带噪声 η）
Step 2  基函数模块：实现 H、B 的多项式基展开（θ 线性）+ 分组定义 + ϕ
Step 3  两步校准：式 (12) 的 θ̂（约束最小二乘）+ GP 差异模型（Physics 基线 & θ 初始化）
Step 4  ADMM 一致性求解器：通用 prox 框架（实现 Proposition 1/2 的 7 个 prox）
Step 5  BCD 主循环：g(ADMM) → d(闭式) → θ(ADMM)，收敛判断
Step 6  调参与评估：λ 网格搜索（验证 MSE）；MSE/RMSE/MAE + 多次重复
Step 7  三个仿真实验：样本量(12 场景) → 物理精度(3 档) → 标签节省
Step 8  案例复现：案例 A（需自采/模拟点焊数据）或仅用仿真验证；案例 B 数据取自 Bayarri et al. (2007) Table III/IV
Step 9  对照 §5 锚点逐一校验
```

## 7. 若需严格对齐原文数值

论文未公开代码，基函数与超参数的唯一权威来源是作者。可联系：
- 通讯作者 **Jing Li**（jli3175@gatech.edu / jing.li@isye.gatech.edu）
- 第一作者 **Dhari F. Alenezi**（dhari@gatch.edu，论文拼写如此；Kuwait University）

---

## 8. 基函数设计的文献依据（参考文献图谱）

> PWL 论文未给出基函数的具体形式，其参考文献仅覆盖理论锚点（蒸馏、字典学习、压缩感知）。但"基函数如何设计"在**计算机试验校准（calibration）、多项式混沌展开（PCE）、稀疏正则化回归**三个成熟领域有大量现成答案。本节按相关度分层整理，作为 §4 补全方案的选型依据。标注 ✅ 的条目已通过 CrossRef 核实引用信息。

### 8.1 基函数设计的依据与目的回顾（论文内部逻辑）

PWL 的联合模型 `y = H(x^ph,θ)g + B(x^pr,x^ph)d + ε`（式 4）在结构上是 **Kennedy–O'Hagan（KOH）校准框架** `reality = ρ·η(x,θ) + δ(x) + ε` 的"基函数化 + 联合优化"版本：

| 设计元素 | 论文中的目的 | 对应的文献领域 |
|---|---|---|
| H 与物理模型共享输入 (x^ph, θ) | 知识蒸馏（缓解标签稀缺） | §8.4 蒸馏、§8.3 PCE |
| B 覆盖 (x^pr, x^ph) | 差异补偿（物理模型不完备） | §8.2 校准/差异建模 |
| θ 显式入基 | 校准参数内生联合优化 | §8.2 校准理论 |
| 稀疏组 Lasso 罚 | 模式选择 + 可辨识性 | §8.3 PCE 稀疏选基、§8.5 稀疏正则化 |
| 基函数类型灵活 | 框架通用性 | §8.5 基展开经典 |

### 8.2 第一层（最直接相关）：校准框架中的"差异函数"设计文献 → 指导 B 基设计

| 文献 | 与基函数设计的关系 |
|---|---|
| Kennedy & O'Hagan (2001), *JRSS-B* 63(3):425–464（PWL 引文 [35]） | 结构原型：emulator + discrepancy 二元分解 |
| Higdon et al. (2004), *SIAM J. Sci. Comput.* 26(2):448–466（PWL 引文 [27]） | KOH 的工程实现范式 |
| **Plumlee (2017), "Bayesian Calibration of Inexact Computer Models", *JASA* 112(519):1274–1285** ✅ | **最重要参考**：discrepancy 应与 emulator 梯度**正交化**，否则 θ 与差异项不可辨识 → 指导 PWL 中 **B 基与 H 基的正交化设计**（论文命题 3 只处理 H 内部可辨识性，H 与 B 之间的混淆需靠此思路解决） |
| Brynjarsdóttir & O'Hagan (2014), "Learning about physical parameters: The importance of model discrepancy", *Inverse Problems* 30(11):114007 ✅ | discrepancy 建模方式对物理参数（θ）估计的影响 |
| Tuo & Wu (2015), *SIAM/ASA JUQ* 3(1):767–795；及 AOAS 相关文 | 校准的参数化、估计与收敛性质理论 |
| Plumlee (2019), *JRSS-B* 81(3):519–539 | 校准的置信与一致性 |

### 8.3 第二层：物理模型响应面的基函数设计——多项式混沌展开（PCE）→ 指导 H 基设计

UQ 领域"用基函数逼近物理模型"的标准方法，与 PWL 的 H 基设计几乎同构：

| 文献 | 内容 |
|---|---|
| Xiu & Karniadakis (2002), "The Wiener–Askey polynomial chaos for stochastic differential equations", *SIAM J. Sci. Comput.* 24(2):619–644 | **按输入分布选正交多项式族**（正态输入→Hermite 基、均匀输入→Legendre 基）——仿真中 θ~U(0,1]、x~正态，可直接套用 |
| **Blatman & Sudret (2011), "Adaptive sparse polynomial chaos expansion based on least angle regression", *JCP* 230(6):2345–2367** ✅ | **稀疏 PCE 工程范式**：先建候选基库（全阶多项式），再用 LARS/L1 筛出活跃基——与 PWL 稀疏组 Lasso 选模式直接对应，是 §4.1 配置的主要模板 |
| Ghanem & Spanos (1991), *Stochastic Finite Elements: A Spectral Approach*（Springer） | PCE 奠基性专著 |
| Sudret (2008), "Global sensitivity analysis using polynomial chaos expansions", *RESS* 93(7):964–979 | 用 PCE 做全局敏感性分析——确定哪些交互项值得入基 |

### 8.4 第三层：知识蒸馏（H 与物理模型输入对齐的依据）

| 文献 | 内容 |
|---|---|
| Hinton, Vinyals & Dean (2015), "Distilling the knowledge in a neural network", arXiv:1503.02531（PWL 引文 [29]） | 蒸馏（教师→学生）原始文献：H·g 为学生、η 为教师 |
| Karniadakis et al. (2021), "Physics-informed machine learning", *Nature Rev. Phys.* 3(6):422–440（引文 [33]） | 物理信息 ML 总纲：物理知识嵌入的三种偏置 |
| Meng et al. (2022), "When physics meets machine learning: A survey of physics-informed machine learning", arXiv:2203.16797（引文 [40]） | 物理信息架构/正则化分类综述 |

### 8.5 第四层：基函数类型与稀疏正则化——统计学习经典

| 文献 | 内容 |
|---|---|
| Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, **第 5 章** "Basis Expansions and Regularization" | 多项式、B 样条、自然三次样条、小波、RBF 基的系统比较与正则化——H、B 选基的总参考手册 |
| Friedman (1991), "Multivariate adaptive regression splines", *Annals of Statistics* 19(1):1–67 | MARS：数据驱动自适应样条基 |
| **Simon, Friedman, Hastie & Tibshirani (2013), "A Sparse-Group Lasso", *JCGS* 22(2):231–245** ✅ | **PWL 罚项 λ₂‖g‖₁+λ₃Σ‖g_q‖₂ 的原始出处**：分组策略与 λ₂/λ₃ 配比设计指南 |
| Yuan & Lin (2006), "Model selection and estimation in regression with grouped variables", *JRSS-B* 68(1):49–67 | 组 Lasso 原始文献 |

### 8.6 第五层：基矩阵可辨识性——字典学习与压缩感知（命题 3 的理论来源）

| 文献 | 内容 |
|---|---|
| **Aharon, Elad & Bruckstein (2006), "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation", *IEEE Trans. Signal Process.* 54(11):4311–4322** ✅ | 字典学习奠基作（基矩阵 + 稀疏系数联合学习） |
| Hillar & Sommer (2015), "When can dictionary learning uniquely recover sparse data from subsamples?", *IEEE Trans. Inf. Theory* 61(11):6290–6297（PWL 引文 [28]） | **命题 3 子采样下界 N=(k+1)/β·C(m,k) 的直接来源** |
| Donoho (2006), "Compressed sensing", *IEEE Trans. Inf. Theory* 52(4):1289–1306（引文 [17]）；Baraniuk et al. (2008), *Constructive Approximation* 28(3):253–263（引文 [1]） | 压缩感知恢复条件（n ≥ Ck·log(m/k)、RIP） |
| Mairal et al. (2009), "Online dictionary learning for sparse coding", ICML | 字典学习的高效算法 |

### 8.7 辅助脉络

- **函数型 ANOVA**（Sobol' 1993；Hoeffding 1948）：把 u(x^ph,x^pr,θ) 分解为主效应 + 交互效应——为 H/B 划分与"按物理现象分组"提供理论语言；
- **代理模型总览**：Santner, Williams & Notz, *The Design and Analysis of Computer Experiments*（Springer，引文 [49] 的专著扩展）；Forrester, Sóbester & Keane (2008), *Engineering Design via Surrogate Modelling*——多项式基 vs 克里金 vs RBF 的选择讨论；
- **两步校准基线的 GP 差异建模**：Bastos & O'Hagan (2009), *Technometrics* 51(4):425–438（引文 [2]）；
- **论文集内部参照**：RWSL / M2WeST 的实用主义选择（"线性映射在应用中效果满意"）。

### 8.8 复现阅读顺序建议

```
① Simon et al. 2013（JCGS）     → 罚项与分组设计（半天）
② Blatman & Sudret 2011（JCP）  → 候选基库 + 稀疏选择的工程范式（1 天）
③ Plumlee 2017（JASA）          → B 基与 H 基正交化，保证可辨识（1 天）
④ ESL 第 5 章                   → 基类型选择（备查手册）
⑤ Xiu & Karniadakis 2002        → 按输入分布选正交基（可选）
```

### 8.9 结论

> **Blatman–Sudret 的稀疏 PCE 范式（候选基库 + L1 筛选）+ Simon 的稀疏组 Lasso 分组 + Plumlee 的正交化原则**，三者组合构成一套有文献支撑、且与论文设计意图（蒸馏 / 补偿 / 校准三目的）完全一致的基函数设计方案，可直接用于 §4.1 的复现配置。

---

## 9. 仿真复现的基函数设计计划（基于式 (9)–(11) 的确认公式）

> 本节先用 PDF 版面分析**精确确认**仿真部分的三个代数公式（此前 OCR 文本存在歧义），再由公式结构推导 H / B 的基函数设计计划。

### 9.1 仿真代数公式的精确确认

**确认方法**：用 PyMuPDF 提取 span 级坐标（上标 y 更小、下标 y 更大，以此区分 x³ 与 x₃），并检测页面绘图对象中的**分数线**位置与长度（脚本：`references/tools/inspect_eq.py`、`references/tools/inspect_lines.py`，中间结果 `references/tools/eq_dump.txt`）。

**关键证据**：式 (9) 区域仅有两条分数线——① 短分数线（y=698.36，x=174.0→188.0，长 13.9）对应指数项的 `1/(2x₃)`；② **一条长分数线（y=698.36，x=216.1→297.2，长 81.2）贯穿整个大分式** → 式 (9) 是**单一大分式**（而非 OCR 文本暗示的"两个分式相加"）。

**确认后的公式**（IEEE 版式 (9)–(11) = 学位论文式 (15)–(17)）：

$$\eta_{true}(x^{ph},\theta) = \Big(1-\exp\big(-\tfrac{1}{2x_3}\big)\Big)\times\frac{10\theta_1x_1^3 + 19x_1 + 60}{9\theta_2x_2^3 + 4x_2 + 20} \tag{9}$$

$$\delta(x^{ph}) = \frac{10x_1^2 + 4x_2^2 + 2x_3^3}{50x_1x_2 + 10} \tag{10}$$

$$h(x^{pr}) = \frac{4x_4^2 + 8x_5^2}{5x_4x_5} \tag{11}$$

**数据流与变量**：
- 真值：`y = η_true(x^ph,θ) + h(x^pr) + ε`，SNR = 5；
- 可观测物理模型：`η_obs = η_true + δ + ε_η`（含差异与不确定性的"不准"版本）；
- x^ph = (x₁,x₂,x₃)（q1=3）、x^pr = (x₄,x₅)（q2=2）、θ = (θ₁,θ₂) ~ U(0,1]²（q3=2）、x ~ MN(0,Σx)。

### 9.2 代数结构对基函数设计的四点约束

| # | 公式事实 | 对基函数设计的约束 |
|---|---|---|
| 1 | 饱和因子 **S(x₃) = 1−exp(−1/(2x₃))** 强非线性 | H 必须包含 S(x₃) 类特征，否则蒸馏项欠拟合 |
| 2 | **θ₂ 在真值分母中 → θ 非线性**；但 ADMM 近端算子要求 H 对 θ **仿射**（缺口 2） | H 只用 θ 一次特征做局部线性近似；近似残差由 B·d 与标记数据吸收——这正是 PWL"物理模型不准也能用"的机制 |
| 3 | δ 只含 x^ph，被 η_obs 带入弱标签 | 蒸馏会把 δ 一起教给 H·g（弱标签有偏）→ 纠偏靠 B·d（x^pr 信息）+ 标记样本 |
| 4 | h 是零阶齐次比 = (4/5)(x₄/x₅) + (8/5)(x₅/x₄) | B 含**比值特征**可精确表示 h；纯多项式只能近似 |

**分组依据**：论文罚项 `λ₃Σ_{q=1}^{q1+q3}‖g_q‖₂` 共 **q1+q3 = 5 组** → 稀疏组 Lasso 的分组应按 **H 的 5 个输入变量**（x₁, x₂, x₃, θ₁, θ₂）划分，即"同一变量派生的模式同进同出"。

### 9.3 H(x^ph, θ) 基函数库设计（蒸馏项，θ 一次仿射，p=25，5 组）

| 组 | 基特征 | 对应真值结构 |
|---|---|---|
| G_x1（4） | 1, x₁, x₁², x₁³ | 式 (9) 分子的 x₁ 通道 |
| G_x2（3） | x₂, x₂², x₂³ | 式 (9) 分母的 x₂ 通道 |
| G_x3（8） | x₃, x₃², x₃³, S(x₃), S(x₃)x₁, S(x₃)x₂, S(x₃)x₁³, S(x₃)x₂³ | 饱和因子及其交互 |
| G_θ1（5） | θ₁, θ₁x₁, θ₁x₁³, θ₁S(x₃), θ₁x₁³S(x₃) | θ₁ 通道（θ 一次，仿射） |
| G_θ2（5） | θ₂, θ₂x₂, θ₂x₂³, θ₂S(x₃), θ₂x₂³S(x₃) | θ₂ 通道（θ 一次，仿射） |

- 记 S(x₃) = 1 − exp(−1/(2x₃))；
- **可选增强**（逼近式 (9) 分母的有理结构，且不破坏对 g/θ 的线性）：固定系数有理特征 1/(20+4x₂)、x₁/(20+4x₂)、S(x₃)/(20+4x₂)；
- 注意：H₁（n 个标记样本处）与 H₂（N 个物理样本处）共用同一基函数库，仅在各自 x^ph 与当前 θ 处求值。

### 9.4 B(x^pr, x^ph) 基函数库设计（补偿项，b=12）

| 部分 | 基特征 | 目标 |
|---|---|---|
| x^pr 通道（8） | 1, x₄, x₅, x₄², x₅², x₄x₅, **x₄/x₅, x₅/x₄** | 精确/近似表示 h（式 11） |
| x^ph 通道（4） | x₁², x₂², x₃³, x₁x₂ | 吸收 δ（式 10）经蒸馏残留的差异 |

- **可辨识性处理（Plumlee 原则）**：x^ph 通道特征与 H 库存在重叠（x₁²、x₂²、x₃³）→ 两个实现版本对照：(a) 直接使用（靠 λ₂/λ₃ 调节）；(b) 对 B 的 x^ph 部分做关于 H 列空间的残差化（正交化）后再入模型。

### 9.5 配套设置

| 项 | 配置 | 依据 |
|---|---|---|
| ϕ | 恒等映射 | 仿真中 η_obs 与 y 同量纲 |
| θ 盒约束 | ω_l=(0,0)，ω_u=(1,1) | 与 θ~U(0,1] 的生成机制一致（可做 [0,1.5] 敏感性分析） |
| 特征标准化 | 每个基特征 z-score 标准化 | 保证 L1/组 Lasso 罚的公平性 |
| Σx | R_{x,i,j} = a^\|i−j\|，a=0.5 | 沿用 RWSL 论文的协方差结构 |
| δ 缩放（精度实验） | 对式 (10) 乘系数 c，网格搜索 c 使 corr(y, η_obs) ≈ 0.85 / 0.7 / 0.5 | 对应论文 §IV-B 的三档物理精度 |
| ε_η | 与 δ 缩放联动微调 | 控制弱标签噪声水平 |

### 9.6 基函数设计计划总览

| 模块 | 设计 | 依据 |
|---|---|---|
| H 库 | 25 个特征、5 组（按输入变量），θ 一次仿射，含 S(x₃) 饱和特征 | 式 (9) 结构 + ADMM 仿射约束 + 论文 5 组罚项 |
| B 库 | 12 个特征：x^pr 比值/多项式 + x^ph 残差特征 | 式 (10)(11) 结构 + 差异补偿目的 |
| 分组 | G_x1, G_x2, G_x3, G_θ1, G_θ2 | λ₃Σ_{q=1}^{5}‖g_q‖₂ 的组结构 |
| 正交化 | B 的 x^ph 部分可选对 H 残差化 | Plumlee (2017) 可辨识性原则（§8.2） |
| 稀疏选择 | λ₂‖g‖₁ + λ₃Σ‖g_q‖₂ 网格搜索 | Blatman–Sudret 稀疏基筛选范式（§8.3） |

---

## 10. 开源代码库排查（GitHub，2026-07 检索）

**结论：PWL（含 RWSL、M2WeST）没有官方或第三方开源实现，必须从头实现。**

### 10.1 直接实现排查（均未找到）

| 检索式 | 结果 | 判定 |
|---|---|---|
| `physics-informed weakly-supervised` | 6 个仓库 | 无一相关（PI-KAN-PointNet、PhysicsInformedPointNet 等为其它论文代码） |
| `"physics-informed weakly-supervised" quality` | 0 | — |
| `M2WeST` / `multi-source multi-task weakly` | 0 | — |
| `RWSL` | 19 个仓库 | 全部无关（缩写撞车，如 rwslib） |
| `ranking-based weakly supervised` | 3 个仓库 | 无关（多标签学习、NLP 关系抽取） |
| 作者排查（`Alenezi`、`Biehler` 用户） | 仅 1 个 2014 年课程项目 | 作者未公开任何代码；论文亦无 code availability 声明 |

唯一"近亲"：**`Elin24/piwsl-pld-perovskite`**（Python）——同为 physics-informed weakly supervised learning 但用于钙钛矿太阳能电池，是不同论文的实现，仅代码组织结构可参考。

### 10.2 组件级可用资源（复现借力）

| 仓库 | 用途 | 与 PWL 复现的关系 |
|---|---|---|
| `nrdg/groupyr`（Python，BSD-3） | 稀疏组 Lasso 求解器 | **对拍基准**：λ₁=0 时 PWL 的 g 更新退化为标准 SGL 回归，可与 groupyr 交叉验证 |
| `mathurinm/celer`（Python，BSD-3） | 快速 L1/Group Lasso 求解 | 同上，参照实现 |
| `rtavenar/SparseGroupLasso`（Notebook） | SGL 教学实现 | 近端算子写法参考 |
| `EugeneNdiaye/GAPSAFE_SGL`（Python） | SGL 安全筛选加速 | 可选性能优化 |
| CRAN `SGL`（Simon et al. 官方 R 包） | SGL 原作者实现 | 结果基准 |

mPower 数据生态（仅 RWSL/M2WeST 相关）：`ResearchKit/mPower`（官方 App）、`Sage-Bionetworks/mPowerSDK`、`wkopp/SpreedictorMPowerParkinson`（DREAM 挑战赛）等。

### 10.3 对复现策略的影响

1. 论文补充材料的 7 个近端算子全部闭式 → ADMM/BCD 用 NumPy/SciPy 即可精确实现，**无需外部优化库**；
2. 采用**对拍策略**验证求解器：先实现 §4 配置，λ₁=0 时与 groupyr/celer 的 SGL 结果比对，再启用完整四项目标；
3. 官方代码只能邮件索取（见 §7）。

---

**关联阅读**：[02_PWL论文解析](legacy/02_PWL论文解析.md)、[03_PWL补充材料解析](legacy/03_PWL补充材料解析.md)、[00_文献整体框架](legacy/00_文献整体框架.md)
