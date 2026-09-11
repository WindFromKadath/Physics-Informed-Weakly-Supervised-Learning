# 案例 B 点焊熔核直径数据集（Spot Weld）

> 来源：CRAN 存档 R 包 **SAVE 1.0**（Palomo, Paulo & García-Donato；Bayarri & Berger 署名贡献者）内置数据集
> `spotweldfield` / `spotweldmodel`，即 Bayarri et al. (2007, *Technometrics* 49(2):138–154) 论文 Table III/IV 的官方发布版本。
> PWL 论文（Alenezi et al. 2025, IEEE T-ASE）案例 B 声明使用"Table III of [3] 的 35 组仿真"与"Table IV 的 120 个过程样本"，
> 经核对与本数据集**逐项吻合**（35 组仿真、12 设置 × 10 重复、重复样本合并方差 0.2006 ≈ 论文报告的 0.2）。

## 文件

| 文件 | 内容 | 形状 |
|---|---|---|
| `spotweldfield.csv` | 真实过程样本（field data）：12 个实验设置 × 10 次重复 | 120 × 4 |
| `spotweldmodel.csv` | ANSYS 仿真样本（model data）：含校准参数 tuning | 35 × 5 |
| `legacy/_raw/` | SAVE 1.0 源码包与原始 `.rda` 文件（历史溯源） | — |

## 变量对照（PWL 记号 ↔ 数据列名）

| PWL 记号 | 数据列 | 物理含义 | 取值范围（field） | 取值范围（model） |
|---|---|---|---|---|
| x₁ | `load` | 电极压力 [kN] | 4.0, 5.3（2 水平） | [3.80, 5.50] |
| x₂ | `current` | 焊接电流 [kA] | [21.0, 29.0] | [20.28, 29.44] |
| x₃ | `thickness` | 板厚 gage | 1, 2（2 水平） | 1, 2（2 水平） |
| y | `diameter` | 熔核直径 [mm] | [4.02, 8.37] | [4.36, 7.36] |
| θ | `tuning` | **接触电阻校准参数**（field 数据中无此列） | — | [0.80, 7.712] |

## 关键参数（从参考文献补足，论文未给）

| 参数 | 取值 | 来源 |
|---|---|---|
| θ 盒约束 [ω_l, ω_u] | **[0.8, 8.0]** | SAVE 包官方示例对 t 设均匀先验 `uniform("t", lower=0.8, upper=8.0)` |
| θ 初值参考 | 4.0 | SAVE 示例 `bestguess=list(t=4.0)` |
| θ 的物理意义 | 接触电阻（contact resistance）相关 | SAVE vignette："a calibration input related to contact resistance" |
| 噪声方差 | **0.2006**（设置内重复样本合并方差） | 本数据集实测；论文报告 ≈ 0.2 ✅ |
| 仿真样本数 | 35 | 本数据集；与 PWL 论文"35 simulations"一致 ✅ |
| 代理超参估计 | MLE（DiceKriging，常数趋势 + 高斯协方差） | SAVE 包实现方式（Bayarri 方法论的官方代码） |

## 注意事项

1. 三个可控变量 (load, current, thickness) **全部**是仿真模型的输入 → PWL 案例 B 中 x^pr = ∅，
   补偿项 B 只建立在 x^ph 上，必须按 09 文档 §4.4 做正交化处理；
2. thickness 是二值变量（1/2），基函数设计时其高阶项无意义，按因子/线性项处理；
3. 仿真输入范围略宽于真实实验范围（如 load [3.8,5.5] vs [4.0,5.3]），生成弱标签时应以**真实实验范围**为采样域；
4. Xie & Xu (2018, arXiv:1803.01231) 指出仿真器在 gage=1 时输出质量差，部分后续研究只保留 gage=2 —— 复现时建议先全量使用，再做 gage=2 的敏感性对照。

## 复现提取方式

```powershell
# legacy/_raw/SAVE.tar.gz 为 CRAN Archive 的 SAVE_1.0.tar.gz
# .rda → csv 使用项目 venv 中的 pyreadr（已安装）
.venv/Scripts/python.exe -c "import pyreadr; ..."
```

日常复现直接读取本目录的两个 CSV；只有核验数据来源或重新提取时才需要访问
[`legacy/`](legacy/README.md)。
