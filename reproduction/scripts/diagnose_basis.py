"""阶段 1：H/B 基函数覆盖诊断。

[INFERRED] 论文未公开 H/B 的具体列。本脚本在大样本上量化当前复现特征库的
覆盖能力：H 对 eta_true 与 eta_true+discrepancy 的最小二乘 R²、compact B 对
式(11)过程项的 R²（自校验，应≈1）、compact/expanded B 对 discrepancy 残差的
吸收能力，以及 H 与 B 列空间的最大典型相关系数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pwl_repro.features import PWLFeatureLibrary
from pwl_repro.simulation import (
    discrepancy,
    eta_true,
    generate_simulation,
    process_contribution,
)

DEFAULT_CONFIG = ROOT / "configs" / "diagnostic.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "sensitivity" / "_basis_diagnosis.json"


def _least_squares_r2(design: np.ndarray, target: np.ndarray) -> float:
    """design 对 target 的最小二乘 R²（含 1 列时等价于通常定义）。"""

    coefficients, residuals, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    total = float(np.sum((target - np.mean(target)) ** 2))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum((target - fitted) ** 2))
    return 1.0 - residual / total


def _max_canonical_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """两矩阵列空间的最大典型相关系数：Q_left.T @ Q_right 的最大奇异值。"""

    q_left, _ = np.linalg.qr(left)
    q_right, _ = np.linalg.qr(right)
    singular_values = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return float(np.max(singular_values))


def _max_canonical_correlation_centered(left: np.ndarray, right: np.ndarray) -> float:
    """去掉截距列后的最大典型相关系数（原始版本因共享常数列恒为 1）。"""

    return _max_canonical_correlation(left[:, 1:], right[:, 1:])


def diagnose_basis(
    config: dict[str, Any],
    seed: int = 0,
    *,
    coverage_size: int = 5000,
) -> dict[str, Any]:
    """计算基函数覆盖诊断并返回可 JSON 序列化的 dict。"""

    simulation = config["simulation"]
    model_config = config.get("model", {})
    stability = simulation.get("stability", {})
    # [ENGINEERING] 覆盖评估用大样本 labeled 池的 x_ph/x_pr；theta 用真值。
    data = generate_simulation(
        n_labeled=coverage_size,
        n_weak=int(simulation.get("n_weak", 100)),
        n_test=int(simulation.get("n_test", 200)),
        seed=seed,
        snr=float(simulation.get("snr", 5.0)),
        correlation=float(simulation.get("correlation", 0.5)),
        discrepancy_scale=float(simulation.get("discrepancy_scale", 1.0)),
        physics_noise_fraction=float(simulation.get("physics_noise_fraction", 0.25)),
        singularity_policy=str(
            simulation.get("singularity_policy", "reject_near_pole")
        ),
        min_abs_x3=float(stability.get("min_abs_x3", 0.15)),
        min_abs_x4=float(stability.get("min_abs_x4", 0.15)),
        min_abs_x5=float(stability.get("min_abs_x5", 0.15)),
        min_abs_eta_denominator=float(
            stability.get("min_abs_eta_denominator", 1.0)
        ),
        min_abs_discrepancy_denominator=float(
            stability.get("min_abs_discrepancy_denominator", 0.75)
        ),
        max_abs_component=float(stability.get("max_abs_component", 25.0)),
        noise_reference_size=int(simulation.get("noise_reference_size", 5000)),
    )
    x_ph = data.x_ph
    x_pr = data.x_pr
    theta = data.theta_true
    standardize = bool(model_config.get("standardize", True))

    def build_library(b_profile: str) -> PWLFeatureLibrary:
        # 与 model.py 中 features_.fit 的调用方式一致：theta_reference 用真值。
        return PWLFeatureLibrary(
            standardize=standardize, b_profile=b_profile
        ).fit(x_ph, x_ph, x_pr, theta)

    library = build_library(str(model_config.get("b_profile", "compact")))
    compact = build_library("compact")
    expanded = build_library("expanded")

    h = library.transform_h(x_ph, theta)
    b_compact = compact.transform_b(x_ph, x_pr)
    b_expanded = expanded.transform_b(x_ph, x_pr)

    physics = eta_true(x_ph, theta)
    scale = float(simulation.get("discrepancy_scale", 1.0))
    biased_physics = physics + scale * discrepancy(x_ph)
    process = process_contribution(x_pr)

    h_r2_eta = _least_squares_r2(h, physics)
    h_r2_biased = _least_squares_r2(h, biased_physics)
    b_compact_r2_process = _least_squares_r2(b_compact, process)

    # discrepancy 减去 H 已解释部分后的残差，检验 B 对物理差异的吸收能力。
    h_coefficients = np.linalg.lstsq(h, biased_physics, rcond=None)[0]
    discrepancy_residual = biased_physics - h @ h_coefficients
    b_compact_r2_residual = _least_squares_r2(b_compact, discrepancy_residual)
    b_expanded_r2_residual = _least_squares_r2(b_expanded, discrepancy_residual)

    return {
        "seed": seed,
        "coverage_size": coverage_size,
        "theta_true": theta.tolist(),
        "standardize": standardize,
        "discrepancy_scale": scale,
        "h_r2_eta_true": h_r2_eta,
        "h_r2_eta_true_plus_discrepancy": h_r2_biased,
        "b_compact_r2_process_contribution": b_compact_r2_process,
        "b_compact_r2_discrepancy_residual": b_compact_r2_residual,
        "b_expanded_r2_discrepancy_residual": b_expanded_r2_residual,
        "max_canonical_correlation_h_b_compact": _max_canonical_correlation(
            h, b_compact
        ),
        "max_canonical_correlation_h_b_expanded": _max_canonical_correlation(
            h, b_expanded
        ),
        "max_canonical_correlation_h_b_compact_centered": (
            _max_canonical_correlation_centered(h, b_compact)
        ),
        "max_canonical_correlation_h_b_expanded_centered": (
            _max_canonical_correlation_centered(h, b_expanded)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = diagnose_basis(config, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"已写出: {output}")


if __name__ == "__main__":
    main()
