# 迁移数据历史归档

本目录保存不参与当前训练与验收的溯源材料：

- `batch_01_3d_reference/`：COMSOL 6.4 三维 FEM 批次 01，用于与当前一维
  模型交叉验证；
- `generation_summary_1d.json`：首次一维批量生成的过程汇总；
- `generation.log`、`generation.err`：早期三维后台生成日志。

当前实验只使用上一级目录的 `batch_01`–`batch_10` 与
`acceptance_report.json`。运行时代码不得默认遍历本目录。
