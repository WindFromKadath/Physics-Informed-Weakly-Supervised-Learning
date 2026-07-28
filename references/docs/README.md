# PWL_Paper 文献解析文档系列

本目录是对 `references/papers/` 下 4 篇 PDF 文献的系统化解析，构建该文献集的整体知识框架。

## 文献主题

**弱监督学习（Weakly Supervised Learning, WSL）在远程健康监护与制造过程质量预测中的应用**——Georgia Tech ISyE（Jing Li / Jianjun Shi 课题组）Dhari F. Alenezi 的系列研究：

- **RWSL**（2022）：排序弱监督学习 → 帕金森病远程监护
- **PWL**（2024/2025）：物理信息弱监督学习 → 制造过程质量预测
- **M2WeST**（2024）：多源多任务弱监督迁移学习 → 帕金森病远程监护（学位论文独有，未发表）

## 文档导航

| 序号 | 文档 | 说明 |
|---|---|---|
| 00 | [文献整体框架](00_文献整体框架.md) | ⭐ 从这里开始：文献清单、关系图谱、阅读路径 |
| 01 | [RWSL 论文解析](01_RWSL论文解析.md) | IISE 2022 期刊论文 |
| 02 | [PWL 论文解析](02_PWL论文解析.md) | IEEE T-ASE 2025 期刊论文 |
| 03 | [PWL 补充材料解析](03_PWL补充材料解析.md) | 优化与可辨识性证明 |
| 04 | [学位论文解析](04_学位论文解析.md) | 2024 博士论文整体结构与结论 |
| 05 | [M2WeST 方法解析](05_M2WeST方法解析.md) | 学位论文第 4 章（新工作） |
| 06 | [方法演进与对比](06_方法演进与对比.md) | RWSL → PWL → M2WeST 综合分析 |
| 07 | [PWL 复现指南](07_PWL复现指南.md) | 细节完整性盘点、缺口补全、验证锚点、基函数文献依据、仿真公式确认、基函数设计计划、GitHub 代码排查 |
| 08 | [基函数设计逻辑](08_基函数设计逻辑.md) | 基函数 H/B 的设计依据、设计目的、算法约束与实例化方案专题 |

## 可运行复现工程

独立的 Python 复现工程位于 `reproduction/`，入口与已知差异见
[`REPRODUCTION.md`](../../reproduction/REPRODUCTION.md)。建议先运行 `reproduction/configs/smoke.yaml`
完成端到端验证，再运行日常或论文协议配置。

## 文件夹结构

```
references/
├── papers/                      # 4 篇原始文献 PDF
├── docs/                        # ★ 本解析文档系列
├── extracted_text/              # PDF 提取的纯文本（供检索）
└── tools/
    ├── extract_pdf.py           # PDF 文本提取脚本（PyMuPDF）
    ├── inspect_eq.py
    ├── inspect_lines.py         # 仿真公式版面分析脚本
    └── eq_dump.txt
```

## 环境复现

```powershell
uv run python references/tools/extract_pdf.py
```
