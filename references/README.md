# 参考文献与解析材料

本目录只保存论文原文及其派生材料，不包含 PWL 模型训练代码。

```text
references/
├── papers/           # 原始 PDF 文献
├── docs/             # 中文解析、复现依据和基函数设计说明
├── extracted_text/   # 从 PDF 提取的全文文本
└── tools/            # PDF 提取与公式版面检查脚本
```

文献阅读入口是 [`docs/README.md`](docs/README.md)。

## 文献工具

在仓库根目录运行：

```powershell
uv run python references/tools/extract_pdf.py
uv run python references/tools/inspect_eq.py
uv run python references/tools/inspect_lines.py
```

脚本固定从 `references/papers/` 读取 PDF。文本提取结果写入
`references/extracted_text/`，公式检查中间结果写入 `references/tools/eq_dump.txt`。
