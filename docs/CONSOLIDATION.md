# 项目整理记录

日期：2026-09-11。分支：`codex/project-consolidation-20260911`，起点为本地 `main` 的 `40eb124`。

## 本次保存范围

接收整理前已存在的暂存和未暂存工作：核心分层、点焊案例 B、热传导迁移、40 批数据、盲测与分层分析、报告及汇报材料。本次未改动模型算法，新增统一状态与汇报索引，并为旧入口补充 S1 更新说明。

原有 `legacy/` 归档和日志移出版本控制的改动一并保存。完整演示和讲稿的上层副本经 SHA-256 确认相同，通过忽略规则避免重复提交；演示检查输出、临时目录、运行结果与缓存不上传。文件保留在本地，历史版本保留用于追溯。

热传导 CSV 的元数据记录原始字节哈希，因此为这些数据关闭 Git 文本换行转换，避免远端检出后验收失配；Office 文件明确标记为二进制。

## 验证结果

| 检查 | 本次结果 |
|---|---|
| pytest | 70 passed |
| 独立仿真数值检查 | 30 项通过 |
| 论文复现 smoke | 成功，24 条指标、1920 条预测 |
| 热传导迁移 smoke | 成功，48 条指标、9600 条预测 |
| 待提交数据字节校验 | 批 1–40 的 120 个 CSV 与元数据 SHA-256 全部一致 |
| 新文档导航 | 本地目标路径全部存在 |

测试最初受 Windows 沙箱临时目录/进程通信权限影响，在正常权限下使用独立临时目录重跑后全部通过。两个 smoke 存在 GP 核参数触及搜索边界的警告，流程正常结束；流程完成不代表全面复现论文数值或通过所有科学质量锚点。

验证命令（仓库根目录）：

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp tmp/pytest-consolidation-20260911
.venv/Scripts/python.exe reproduction/scripts/verify_sim_correctness.py
.venv/Scripts/python.exe reproduction/run_reproduction.py --config reproduction/configs/smoke.yaml --output reproduction/results/consolidation_20260911
.venv/Scripts/python.exe migration/run_migration.py --config migration/configs/heat_smoke.yaml --output migration/results/consolidation_20260911
```

本次未重跑完整 20 批训练或 S1/G0/G1 分析；[当前状态](CURRENT_STATUS.md) 中的科学结论明确引用已有报告。Office 文件只做保留与重复文件哈希核对，未编辑或重新验版。

新分支提交并上传到已有 `origin`；不合并到 `main`。分支上传触发 CI，远端 CI 结果以 GitHub 实际运行结果为准。
