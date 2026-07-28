"""阶段 8.3：批量运行敏感性配置并聚合论文对照结果。

默认行为：不带 --only/--round 时按文件名字母序运行
configs/sensitivity/ 下全部配置（base.yaml 最先，变体随后）。
--round 按方案 §7 分轮：round1={s01,s02,s04,s12}（训练规则类），
round2={s03,s10}（模型结构类），round3={s05,s06,s07,s08,s09,s11}（数据生成类）。

每个变体运行 run_experiments(config, "all") 并 save_artifacts 到
results/sensitivity/<变体名>/；stdout/stderr 同时落到
runs/sensitivity/<变体名>/stdout.log 与 stderr.log。单变体异常不中断，
traceback 写入该变体 stderr.log，汇总中记 status=failed。

--smoke 用于快速端到端验证：repeats=2、sample_sizes=[10,60,120]、
lambda 网格缩到 2x1x1，并把 IV-C 的固定/曲线标签同步缩到
{10}/[60,120] 以满足 mode=all 校验；[ENGINEERING] 同时把 IV-B 校准池
缩到 1000、校准网格缩到 121 点以控制时长。smoke 结果写入
results/sensitivity_smoke/ 与 runs/sensitivity_smoke/，汇总 Markdown 也写在
smoke 结果目录内，不触碰 reports/ 下的正式交付物。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pwl_repro.experiments import run_experiments, save_artifacts

from compare_to_paper import compare_results

CONFIG_DIR = ROOT / "configs" / "sensitivity"
ROUNDS: dict[int, tuple[str, ...]] = {
    1: ("s01", "s02", "s04", "s12"),
    2: ("s03", "s10"),
    3: ("s05", "s06", "s07", "s08", "s09", "s11"),
}


class _Tee:
    """把写入同时转发到原流与日志文件。"""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream = stream
        self._log = log

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()


def _apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """[ENGINEERING] 缩小规模用于端到端烟测，不改变任何默认配置。"""

    config = deepcopy(config)
    experiment = config["experiment"]
    experiment["repeats"] = 2
    experiment["sample_sizes"] = [10, 60, 120]
    # mode=all 校验要求 IV-C 标签集合是 sample_sizes 的子集。
    experiment["label_savings_pwl_n_labeled"] = 10
    experiment["label_savings_sizes"] = [60, 120]
    experiment["accuracy_calibration_size"] = 1000
    experiment["accuracy_scale_log10"] = [-3, 3, 121]
    grid = config["lambda_grid"]
    config["lambda_grid"] = {
        "physics": list(grid["physics"])[:2],
        "l1": list(grid["l1"])[:1],
        "group": list(grid["group"])[:1],
    }
    return config


def _discover_variants(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.yaml"), key=lambda path: path.stem)


def _select_variants(
    paths: list[Path], only: str | None, round_number: int | None
) -> list[Path]:
    if only:
        prefixes = tuple(item.strip() for item in only.split(",") if item.strip())
    elif round_number is not None:
        prefixes = ROUNDS[round_number]
    else:
        return paths
    selected = [
        path for path in paths if path.stem.startswith(prefixes)
    ]
    return selected


def run_variant(
    config_path: Path,
    results_root: Path,
    runs_root: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """运行单个变体；返回汇总行（含 status）。"""

    name = config_path.stem
    output_dir = results_root / name
    log_dir = runs_root / name
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    summary: dict[str, Any] = {"variant": name, "status": "ok"}
    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_log:
        tee_out = _Tee(sys.stdout, stdout_log)
        tee_err = _Tee(sys.stderr, stderr_log)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if smoke:
                    config = _apply_smoke_overrides(config)
                artifacts = run_experiments(config, "all")
                save_artifacts(artifacts, config, output_dir)
            except Exception:
                traceback.print_exc()
                summary["status"] = "failed"
    if summary["status"] == "failed":
        print(f"[sensitivity] {name}: FAILED（见 {stderr_path}）")
    else:
        print(f"[sensitivity] {name}: ok -> {output_dir}")
    return summary


def _summarize_variant(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """用 compare_to_paper 的核心函数填充一个变体的关键指标。"""

    if summary["status"] != "ok":
        return summary
    report = compare_results(output_dir)
    keys = report["key_metrics"]
    iv_a = keys["iv_a_pwl_rmse"]
    iv_b = keys["iv_b_pwl_rmse"]
    summary.update(
        {
            "iv_a_rmse_10": iv_a.get(10),
            "iv_a_rmse_60": iv_a.get(60),
            "iv_a_rmse_120": iv_a.get(120),
            "iv_b_rmse_H": iv_b.get("H"),
            "iv_b_rmse_M": iv_b.get("M"),
            "iv_b_rmse_L": iv_b.get("L"),
            "iv_c_pwl_mse_30": keys["iv_c_pwl_mse_30"],
            "iv_c_crossover_at_labels": keys["iv_c_crossover_at_labels"],
            "iv_c_pwl_below_supervised_at_max": (
                keys["iv_c_pwl_below_supervised_at_max"]
            ),
            "mse_over_noise_variance_at_120": (
                keys["iv_a_mse_over_noise_variance_at_120"]
            ),
            "quality_fail_count": keys["quality_fail_count"],
        }
    )
    return summary


def _markdown_table(frame: pd.DataFrame) -> str:
    """不依赖 tabulate 的简易 Markdown 表。"""

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _write_markdown(rows: list[dict[str, Any]], destination: Path) -> None:
    """按方案 §9.3 归因规则生成表格骨架；结论栏留待人工判读。"""

    frame = pd.DataFrame(rows)
    lines = [
        "# 敏感性实验汇总（阶段 3 归因骨架）",
        "",
        "> 本文件由 run_sensitivity.py 自动生成；结论栏统一留“待人工判读”。",
        "",
        "## 变体关键指标",
        "",
        _markdown_table(frame) if not frame.empty else "（无成功变体）",
        "",
        "## 归因规则对照（方案 §9.3）",
        "",
        "| 观察结果 | 相关变体 | 结论 |",
        "|---|---|---|",
        "| 归一化指标基本不变但绝对值大幅变化 | s05/s06/s11 | 待人工判读 |",
        "| selection_rule 改动后 IV-B 明显改善 | s01 | 待人工判读 |",
        "| d_ridge=0 后 IV-A/IV-B 改善但小样本不稳定 | s02 | 待人工判读 |",
        "| b_profile=expanded 后 IV-B M/L 改善 | s03 | 待人工判读 |",
        "| refit=true 后 IV-C 交点提前 | s04 | 待人工判读 |",
        "| 最接近论文文字组合的整体表现 | s12 | 待人工判读 |",
        "| 所有单因子变化都无法缩小归一化差距 | 全部 | 待人工判读 |",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="逗号分隔的变体名前缀过滤（如 s01,s02 或 base）。",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        choices=sorted(ROUNDS),
        dest="round_number",
        help="按方案 §7 分轮运行（1/2/3）。",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="缩小规模并写入 results/sensitivity_smoke/ 用于快速验证。",
    )
    args = parser.parse_args()

    results_root = ROOT / "results" / ("sensitivity_smoke" if args.smoke else "sensitivity")
    runs_root = ROOT / "runs" / ("sensitivity_smoke" if args.smoke else "sensitivity")
    variants = _select_variants(
        _discover_variants(CONFIG_DIR), args.only, args.round_number
    )
    if not variants:
        raise SystemExit("没有匹配的敏感性配置。")
    print(
        f"[sensitivity] variants={[path.stem for path in variants]} "
        f"smoke={args.smoke} results={results_root}"
    )

    rows: list[dict[str, Any]] = []
    for config_path in variants:
        summary = run_variant(
            config_path, results_root, runs_root, smoke=args.smoke
        )
        rows.append(_summarize_variant(summary, results_root / config_path.stem))

    # [ENGINEERING] Aggregate every variant directory present under the
    # results root (not only the ones run in this invocation) so incremental
    # rounds still produce a complete summary.
    seen = {row["variant"] for row in rows}
    for variant_dir in sorted(results_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name in seen:
            continue
        rows.append(
            _summarize_variant(
                {"variant": variant_dir.name, "status": "ok"}, variant_dir
            )
        )

    frame = pd.DataFrame(rows)
    results_root.mkdir(parents=True, exist_ok=True)
    summary_csv = results_root / "sensitivity_summary.csv"
    frame.to_csv(summary_csv, index=False, encoding="utf-8")
    if args.smoke:
        markdown_path = results_root / "sensitivity_summary.md"
    else:
        markdown_path = ROOT / "reports" / "sensitivity_summary.md"
    _write_markdown(rows, markdown_path)
    print(f"[sensitivity] 汇总: {summary_csv}")
    print(f"[sensitivity] 报告骨架: {markdown_path}")


if __name__ == "__main__":
    main()
