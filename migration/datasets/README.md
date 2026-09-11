# datasets/ — 数据集说明（数据层专用文档）

> 规格：v1.3（1D）| 数据状态：**20 批全部生成**（批次 1–10 为原始批，
> 批次 11–20 于 2026-08-03 由外部管线 N_BATCHES=20 追加生成，协议同源），
> 数据层验收（§5.3，20/20）全部通过
> 生成代码：`comsol_platform/pipeline1d.py`（求解后端 `solver1d.py`，无 COMSOL 依赖）
> 验收代码：`step7_acceptance.py` → `acceptance_report.json`

---

## 1. 目录内容

```
datasets/
├── batch_01 .. batch_20/        # 一维生成，每批 520 样本
│   ├── reference_measurements.csv   # 参考测量集 120（高保真 + 噪声）
│   ├── acceptance_tests.csv         # 独立验收集 200（高保真 + 噪声）
│   ├── engineering_predictions.csv  # 工程预测集 200（低保真，确定性）
│   └── metadata.json                # 批次元数据（k_true/σ/种子/指纹/覆盖性）
├── acceptance_report.json       # 数据层验收报告（§5.3 全部检查项，20 批）
└── legacy/                      # 历史溯源批、生成汇总与旧日志
    ├── batch_01_3d_reference/   # 三维 COMSOL 生成的批次 01
    ├── generation_summary_1d.json
    └── generation.log / .err
```

**规模**：20 批 × 520 样本 = **10,400 条**（参考 2,400 + 验收 4,000 + 工程 4,000），生成耗时秒级。

---

## 2. 物理场景与生成方式

一维双层杆 `z ∈ [0, 0.1 m]`，稳态热传导，输出 **y = 界面两侧平均温度 (T1+T2)/2 [K]**。

| 模型 | 导热系数 | 界面 | 响应 P | 求解器 |
|---|---|---|---|---|
| 高保真（参考/验收） | `k_true ~ U[15,35]` 逐批、`k(T)=k_true·(1+0.002·(T−300))` | 接触热阻 `R_c(P)=3e-2·P^(−0.7)` | **是** | 1D Kirchhoff BVP（毫秒级，无离散误差） |
| 低保真（工程预测） | 恒定 `k_nominal = 25` | 完美贴合 R_c=0 | **否** | 闭式解（微秒级） |

噪声：`y_obs = y_det + ε`，ε ~ N(0, σ²)，σ = 4.494 K（SNR=5，试验综合重复性误差）。

---

## 3. 字段语义

| 字段 | 含义 | 单位 | 出现在 |
|---|---|---|---|
| `Q` | 体积热源 | W/m³ | 全部 |
| `h` | 顶面对流换热系数 | W/(m²·K) | 全部 |
| `T_inf` | 环境温度 | K | 全部 |
| `P` | 界面压强 | MPa | 全部 |
| `y_obs` | 观测温度（确定性输出 + 测量噪声） | K | 参考/验收 |
| `y_pred` | 工程预测温度（确定性，无噪声） | K | 工程预测 |
| `dT_int` | 界面温跃（派生量 = R_c·q_int，P 通道机理诊断） | K | 参考/验收 |

输入范围：`Q ∈ [1e4, 1e5] W/m³`、`h ∈ [3, 50] W/(m²·K)`、`T_inf ∈ [273, 323] K`、`P ∈ [0.5, 10] MPa`。

单位同时声明于每个 CSV 首行 `# units: ...` 注释与 `metadata.json` 中。

---

## 4. 加载示例

```python
import pandas as pd

# 路径自仓库根目录（PWL_Paper/）起算
ref = pd.read_csv("migration/datasets/batch_01/reference_measurements.csv", comment="#")
acc = pd.read_csv("migration/datasets/batch_01/acceptance_tests.csv", comment="#")
eng = pd.read_csv("migration/datasets/batch_01/engineering_predictions.csv", comment="#")

X_ref, y_ref = ref[["Q", "h", "T_inf", "P"]], ref["y_obs"]
X_acc, y_acc = acc[["Q", "h", "T_inf", "P"]], acc["y_obs"]
X_eng, y_eng = eng[["Q", "h", "T_inf", "P"]], eng["y_pred"]
```

---

## 5. 四类集合的研究角色

| 集合 | 规模/批 | 位置 | 角色 | 用法 |
|---|---|---|---|---|
| 参考测量集 | 120 | `reference_measurements.csv` | 模拟昂贵稀缺的高精度实测 | 少量带噪标签，训练/标定多保真融合模型 |
| 工程预测集 | 200 | `engineering_predictions.csv` | 模拟低成本工程模型的大量预测 | 大量廉价弱标签：预训练、偏差学习、自监督 |
| 独立验收集 | 200 | `acceptance_tests.csv` | 最终评估专用，**不得参与任何建模/调参** | 报告泛化误差；误差下限为 σ²≈20.2 K² |
| 标定集（一次性） | 40 | `../COMSOL_Link/reports_out/calibration_pool.csv`（仓库外） | 噪声标定、灵敏度、偏差评估 | 步骤 3/4 使用，非批次数据 |

---

## 6. 数据结构要点（务必阅读）

1. **批次 = 同一批试样**：每批固定一个 `k_true`（见 metadata.json），模拟材料一致性。**跨批次训练时 k_true 是隐含变量**——用于批次泛化研究；单批次内训练则批次内一致。
2. **嵌套参考集**：reference 的 120 行按固定顺序排列，**前 k 行即嵌套子集**（k=10,20,…,120），用于研究样本量递增效应。
3. **噪声**：三集合共用同一 σ。任何方法在验收集上的 MSE 理论下限为 σ²≈20.2 K²（固有属性，非方法缺陷）。工程预测集无噪声。
4. **P 通道（核心结构性偏差）**：工程预测集**不响应 P**（低保真模型无接触热阻，P 列保留但 y_pred 与 P 无关）。真实系统随 P 变化、工程模型不随 P 变化——融合方法若能学到 P 依赖，将在验收集上显著获益。
5. **偏差方向**：高保真 y 系统性高于低保真（均值 +11.4 K，低压强端最高 +54.5 K）。
6. **双保真相关性**：批内 0.76–0.88（设计意图 [0.7, 0.9]）——足够高使迁移学习可行，又足够低使偏差校正有学习空间。
7. **验收集隔离**：验收集种子与参考/工程集互异（独立 LHS），任何建模与调参阶段不可触碰，最终一次性评估。
8. **确定性**：每批全部采样种子入 metadata.json（SEED=20260731，集合种子 = SEED + batch×100 + offset），可逐样本复现；一维求解无离散误差、无失败样本（成功率 100%）。

---

## 7. 典型研究用例

```python
# 用例 1：多保真融合（参考 + 工程 → 预测真实系统）
from sklearn.ensemble import RandomForestRegressor  # 示例

base = RandomForestRegressor(n_estimators=200, random_state=0)
base.fit(X_eng.values, y_eng.values)
clf = RandomForestRegressor(n_estimators=200, random_state=0)
clf.fit(X_ref.values, (y_ref.values - base.predict(X_ref.values)))
y_final = base.predict(X_acc.values) + clf.predict(X_acc.values)

# 用例 2：样本量递增（嵌套子集）
for k in (10, 20, 40, 80, 120):
    sub = ref.iloc[:k]
    # 训练并评估于 acceptance_tests.csv

# 用例 3：批次泛化（跨批次，k_true 各不相同）
# 在 batch_01..05 上训练，在 batch_06..10 的验收集上评估

# 用例 4：P 通道消融（验证融合方法是否学到了界面压强效应）
# 对比含 P 特征与不含 P 特征在验收集上的误差
```

---

## 8. 质量保证与元数据

每批 `metadata.json` 记录：

- `k_true`（批次真实导热系数）、`sigma`（4.4938…）、`R0`（0.03）、`beta_k`（0.002）；
- `solver`（1D Kirchhoff BVP + 闭式解，无离散误差）、`output_definition`（两侧平均）；
- `seeds`（k_true/reference/acceptance/engineering/calibration + 噪声种子，共 8 个）；
- `sizes` 与 `success_rate`（1.0/1.0/1.0）、`failures`（空）；
- `coverage`（工程预测集对参考/验收的最近邻覆盖统计：p95 ≈ 0.246–0.283 < 阈值 0.30，即对角线 15%）；
- `sha256`（三个 CSV 的文件指纹，用于完整性校验）；
- `written_at_utc`、`comsol_version`（N/A，一维求解器未使用 COMSOL）。

**数据层验收结果**（`acceptance_report.json`，全部通过）：

| 检查项 | 结果 |
|---|---|
| 有限性（无 NaN/Inf） | ✅ 20/20 批 |
| sha256 指纹与元数据一致 | ✅ 20/20 批 |
| 覆盖性（95 分位 < 对角线 15%） | ✅ 20/20 批（p95 0.245–0.283） |
| 求解成功率 100% | ✅ 20/20 批 |
| k_true ∈ [15, 35] | ✅ 20/20 批 |
| 噪声标定（残差 ~N(0,σ²)，σ 全局一致） | ✅ 残差均值 \|·\|<1.0 K、std 3.741–5.039 K |
| 双保真相关性 ∈ [0.7, 0.9] | ✅ 0.760–0.881 |
| 验收集隔离（种子互异） | ✅ 20/20 批 |
| 全局 SNR = Var(y_det)/σ² | ✅ 6.618（跨批池化含 k_true 波动，批内设计值 5） |

> 提示：验收集 MSE 下限 σ²≈20.2 K² 应在论文/报告中显式声明；P 通道结构性偏差是数据集的核心学习价值。

---

## 9. 再生成方法

> 注（2026-08-03）：以下命令在外部数据管线仓库（`COMSOL_Link`，本仓库之外）
> 的工作目录中执行，相对路径按该仓库布局书写；本仓库内不含
> `step6b_generate_1d.py`、`verify_1d.py`、`comsol_platform/`。批次 11–20 扩展
> 依赖该外部管线（M4 前置）。

```powershell
# 环境：Python ≥ 3.13（uv 管理），无需 COMSOL
uv sync
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# 1) 一维求解器正确性验证（可选，4 项全过）
.\.venv\Scripts\python.exe ..\verify_1d.py

# 2) 全量生成（秒级；已含 metadata.json + 3 个 CSV 的批次自动跳过，支持断点续跑）
.\.venv\Scripts\python.exe ..\step6b_generate_1d.py

# 3) 数据层验收（重写 acceptance_report.json）
.\.venv\Scripts\python.exe ..\step7_acceptance.py
```

数据生成配置（批次规模、输入范围、σ、R0、种子）统一在 `..\comsol_platform\constants.py` 中定义。

---

## 10. 与三维溯源批的差异

[`legacy/batch_01_3d_reference/`](legacy/README.md) 由 COMSOL 6.4 三维 FEM 生成（细网格 231k 单元），与一维批次**物理等价**（场景全 x-y 对称，严格退化为 1D）：

- 低保真对拍：3D FEM vs 1D 闭式解最大绝对偏差 < 5e-5 K（网格误差量级）；
- 高保真对拍：3D 含噪 y_obs − 1D 确定性预测的残差为纯噪声（均值近 0、std≈σ），已由 `..\cross_validate_3d_vs_1d.py` 验证。

用途：三维原始求解结果的溯源比对；对称性破缺扩展（非均匀热源、局部边界等）时的建模储备。
