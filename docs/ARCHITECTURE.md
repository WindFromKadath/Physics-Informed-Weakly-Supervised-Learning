# Project Architecture

[中文](ARCHITECTURE.zh-CN.md) | English

The repository has only two parts: the paper's Section IV simulation reproduction and the one-dimensional steady-state heat-conduction tests. They share one root-level uv environment and the PWL core; the existence of generic interfaces is not a claim that other applications are implemented.

## Dependency direction

```text
pwl_repro.core                  Feature libraries, PWLRegressor, BCD–ADMM
        ▲
        ├── pwl_repro.scenarios Simulation data and simulation features
        ├── pwl_repro.experiments Tuning, protocols, statistics, and reports
        └── pwl_migration      Heat-conduction tests
                    └── pwl_repro.experiment_api
```

The core does not import specific scenarios. Simulation features are registered at package initialization; heat-conduction features are registered when the migration entry imports the scenario module. Cross-package experiment reuse goes through `experiment_api`. The old `model`, `features`, `optimization`, and `simulation` paths remain as compatibility aliases.

## Directory responsibilities

| Directory | Content |
|---|---|
| `reproduction/` | Section IV simulations, sensitivity configurations, tests, and reproduction notes |
| `migration/` | 1D heat-conduction physics, 40 batches of data, configurations, acceptance, analysis, and reports |
| `docs/` | Current status and architecture of the two parts |
| `references/` | Literature basis and extraction tools for the paper reproduction |

`results/`, `runs/`, caches, and temporary directories do not enter Git, except the manually frozen v3 baseline card. Material kept under `legacy/` serves only historical traceability of these two parts and is not a current implementation entry.

## Running and checking

All commands run from the repository root:

```sh
uv sync --locked --python 3.11 --extra dev
uv run pytest
uv run python reproduction/scripts/verify_sim_correctness.py
```

For experiment commands, see [paper reproduction](../reproduction/README.md) and [heat-conduction tests](../migration/README.md) respectively.
