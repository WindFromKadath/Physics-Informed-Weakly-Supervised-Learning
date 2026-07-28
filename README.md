# PWL Paper 工作区

当前内容按用途分为两个彼此独立的目录：

```text
PWL_Paper/
├── .venv/          # 整个 PWL_Paper 项目唯一的 uv 环境
├── pyproject.toml  # 根项目依赖与包配置
├── uv.lock         # 根项目依赖锁
├── reproduction/   # PWL 复现代码、配置、测试和实验结果
└── references/     # 论文 PDF、解析文档、提取文本和文献检查脚本
```

## 入口

- 运行和验证算法：[`reproduction/REPRODUCTION.md`](reproduction/REPRODUCTION.md)
- 第IV节实验协议：[`reproduction/SECTION_IV_PROTOCOL.md`](reproduction/SECTION_IV_PROTOCOL.md)
- 阅读论文解析：[`references/docs/README.md`](references/docs/README.md)
- 查看参考材料结构：[`references/README.md`](references/README.md)

`PWL_Paper/` 根目录是唯一的 uv 项目。复现实验代码位于 `reproduction/`；
PDF 提取和公式检查脚本位于 `references/tools/`，两类业务内容不混放。
