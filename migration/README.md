# Heat-Conduction Transfer: Multi-Fidelity Learning and Grey-Box Comparison

[中文](README.zh-CN.md) | English

This directory studies PWL's applicability on one-dimensional steady-state heat-conduction multi-fidelity data, reusing `pwl_repro.core` and the common experiment interfaces. It contains 40 batches of input data, experiment configurations, analysis code, pre-registration, and reports.

**Current evidence: in this scenario, with known mechanism and a low-cost physics model, direct grey-box calibration A4 is more suitable than mechanism-assisted PWL.** This directory preserves PWL's mean advantages, failure boundaries, and ablation evidence, and does not generalize a single scenario's results into universal conclusions.

[Repository home](../README.md) · [Paper reproduction](../reproduction/README.md) · [Current project status](../docs/CURRENT_STATUS.md) · [Full S1 report](reports/迁移后续工作/PWL盲测S1报告.md) (Chinese)

## Scenario and data sources

Inputs are `x_ph=(Q, h, T_inf)`, process variable `x_pr=P`, calibration parameter `theta=1/k`; outputs are the average temperatures on both sides of a two-layer rod interface, in K. The high-fidelity model includes temperature-dependent conductivity and contact resistance; the low-fidelity model uses a closed-form solution with constant conductivity and a perfect interface.

This is **numerically generated research data**, not field measurements. The `reference_measurements` in the CSVs are the high-fidelity solution plus noise. Although historical reports are named after a "COMSOL scenario", the current 1D runs do not invoke COMSOL; the existing CSVs and the in-repo Python physics implementation suffice for the experiments below.

| Batches | Source and purpose | Current status |
|---|---|---|
| 1–20 | Historical external 1D generation pipeline; inputs and acceptance reports committed | Development and frozen v3 baseline |
| 21–40 | In-repo `generate_blind_batches.py`; S1 new-batch confirmation | Converted to development data after S1 unblinding |
| `datasets/legacy/` | Early 3D references and historical material | Not default training input |

Each batch contains a 120-row reference set, a 200-row independent validation set, and a 200-row engineering prediction set — 520 rows per batch, 20,800 across 40 batches. The CSVs' raw bytes are preserved via `.gitattributes` to match the SHA-256 in `metadata.json`.

The [data specification](datasets/README.md) mainly documents batches 1–20; for the generation process and freeze protocol of batches 21–40, see the [S1 pre-registration](reports/迁移后续工作/PWL盲测S1预注册.md) and the [S1 report](reports/迁移后续工作/PWL盲测S1报告.md) (both Chinese). The historical external pipeline is not in this repository; everyday runs read the provided data directly.

## Quick start

Requires Git, uv, and Python ≥3.11, <3.14; this verification used Python 3.11. See the [repository home](../README.md) for how to get the `HeatTest` branch. **All single-line commands below run from the repository root** and work in PowerShell and common Unix shells.

```sh
uv sync --locked --python 3.11 --extra dev
uv run python migration/run_migration.py --help
uv run python migration/run_migration.py --config migration/configs/heat_smoke.yaml --output migration/results/quickstart
uv run pytest migration/tests
```

Smoke uses a small configuration to verify data loading, training, and result saving. The output directory is created automatically; existing directories may be overwritten — use a new directory for formal experiments. "All generic tests pass" and "all scientific quality anchors pass" are different judgments.

## Current results and conclusion boundaries

The following are the batch 21–40 means from the existing S1 report, not re-run for this documentation update; `@120` refers to the labeled-sample level in the configuration, and the train/validation split follows the pre-registration.

| Model | @10 RMSE / K | @60 RMSE / K | @120 RMSE / K |
|---|---:|---:|---:|
| A4 (MechanismAligned) | 7.477 | 4.866 | 4.727 |
| PWL | 7.872 | 5.932 | 5.449 |
| PWL without weak labels | 11.045 | 6.072 | 5.434 |
| GP | 10.785 | 6.039 | 5.612 |
| Physics (calibrated + GP) | 9.327 | 6.559 | 5.554 |

- A4's @120 RMSE is 0.722 K lower than PWL's, with consistent direction in 20/20 batches and Holm-corrected p<1e-4; A4 uses a strong mechanistic structure, and its advantage is confirmed only in this scenario.
- PWL's Holm-corrected p-values against GP/Physics are 0.066/0.266 — insufficient evidence to claim a corrected significant advantage over either.
- Weak-label gains appear mainly at the small-sample end; no weak-label advantage was detected at @120.
- The G0/G1 results on batches 21–25 are development-stage evidence only; see the [stratified report](reports/迁移后续工作/PWL分层G0G1开发测试报告.md) (Chinese). Future changes must be confirmed with a new pre-registration and untouched batches — 21–40 can no longer be called unseen data.
- `theta` has structural non-identifiability in this scenario and should not be interpreted as a reliably recovered true conductivity; see the [criterion revision](reports/迁移后续工作/COMSOL场景theta判据修订.md) (Chinese).

The earlier batches 1–20 results — PWL @120=5.351 K, Physics-GP=5.648 K, and the "12-level mean optimum" — belong to the comparison set of that time, which did not include the later-added A4. See the [20-batch report](reports/迁移后续工作/COMSOL场景20批统计闭环报告.md) and the [information-boundary audit](reports/迁移后续工作/PWL算法真实能力与信息边界审计.md) (both Chinese).

## Configuration choices

| Purpose | Configuration | Evidence location |
|---|---|---|
| Flow check | `configs/heat_smoke.yaml` | Small-scale environment check |
| Historical v3 baseline | `configs/heat_v3_qint_min.yaml` | 10-batch development results |
| v3 extension | `configs/heat_v3_qint_min_b20.yaml` | 20-batch frozen comparison |
| S1 main arm | `configs/heat_blind_s1.yaml` | Batches 21–40, with grey-box comparison |
| S1 weak-label ablation | `configs/heat_blind_s1_no_weak.yaml` | Paired with S1 |
| G0/G1 development | `configs/heat_g0_generic_dev.yaml`, `configs/heat_g1_prior_dev.yaml` | 5-batch directional experiments |

`heat_default.yaml` and early variants were used for historical research and are not the recommended defaults for the latest results. Runtime varies with hardware, batch count, and candidate grids; smoke runtime cannot substitute for a formal experiment budget.

Re-checking the historical baseline uses a new directory:

```sh
uv run python migration/run_migration.py --config migration/configs/heat_v3_qint_min_b20.yaml --output migration/results/v3_b20_recheck
```

### Full S1 recomputation order

A fresh clone contains the input data and manual reports, but **not the training results of the two S1 arms**. The analysis script reads the two directories below, so complete both trainings before analyzing; preserve existing artifacts when running in an existing workspace. This flow reproduces historical S1 — it is not a new blind test.

```sh
uv run python migration/run_migration.py --config migration/configs/heat_blind_s1.yaml --output migration/results/heat_blind_s1
uv run python migration/run_migration.py --config migration/configs/heat_blind_s1_no_weak.yaml --output migration/results/heat_blind_s1_no_weak
uv run python migration/analyze_blind_s1.py
```

The data is provided; no regeneration is needed. `generate_blind_batches.py` exists for generation traceability and refuses to overwrite existing batches; `accept_blind_batches.py --start 21 --end 40` re-runs acceptance and rewrites `datasets/acceptance_report_blind_21_40.json`. Related commands should run as `uv run python migration/<script>.py`.

## Directory and outputs

```text
migration/
├── src/pwl_migration/       # Heat-conduction physics, features, and experiment drivers
├── configs/                 # Baseline, S1, ablation, and development configurations
├── datasets/                # 40 batches of CSVs, metadata, and acceptance reports
├── tests/                   # Physics, feature, generator, and pipeline tests
├── reports/                 # Manual conclusions, pre-registration, and plan documents
├── run_migration.py         # Experiment entry
├── analyze_*.py             # Statistical analysis; some read fixed results subdirectories
└── results/                 # Local run artifacts, not uploaded by default
```

Experiment outputs include `results.csv`, per-sample predictions, configuration and condition records, statistical summaries, figures, and `quality_checks.csv`; the actual configuration and outputs prevail. The latter contains generic checks and scenario anchors that need item-by-item interpretation — method validity should not be judged from the process exit code alone.

Run results, logs, and caches do not enter Git; the [manual baseline card](results/heat_v3_qint_min/BASELINE.md) is the exception. Figures under `results/` referenced by historical reports may not exist in a fresh clone and must be generated by running the corresponding training and plotting scripts. Some Chinese plotting scripts use Microsoft YaHei; other systems need an available Chinese font configured.

## Test scope and feedback

This directory keeps only the actual heat-conduction tests with their data, configurations, and analysis evidence; it contains no other application frameworks or application plans.

When modifying this scenario, follow the [architecture conventions](../docs/ARCHITECTURE.md): reuse the core via `pwl_repro.core` and `experiment_api`; do not introduce scenario dependencies backwards. Feedback should include the commit, command, configuration, batch numbers, Python version, and minimal error output; when citing results, distinguish development from S1 confirmation and trace back to the original paper and data sources.
