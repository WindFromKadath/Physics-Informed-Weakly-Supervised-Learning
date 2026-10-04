# PWL: Paper Reproduction and Heat-Conduction Transfer

[中文](README.md) | English

This repository provides an independent research implementation of Physics-Informed Weakly-Supervised Learning (PWL), combining physical weak labels, a small amount of labeled data, and process-discrepancy compensation for quality prediction. It contains only the reproduction of the paper's Section IV simulations and the one-dimensional steady-state heat-conduction test actually used.

**Current working branch: `HeatTest`.** This project is an engineering reproduction of the paper's mathematical structure and optimization methods; it is not the original authors' official implementation, nor does it strictly reproduce all of the paper's numbers.

## Related paper

**Dhari F. Alenezi, Michael Biehler, Jianjun Shi, and Jing Li.**
*Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes.*
**IEEE Transactions on Automation Science and Engineering**, vol. 22, pp. 2006–2019, 2025.
DOI: [10.1109/TASE.2024.3374098](https://doi.org/10.1109/TASE.2024.3374098) · [Paper PDF provided by the author team](https://sites.gatech.edu/jianjun-shi/files/2025/02/Physics-Informed_Weakly-Supervised_Learning_for_Quality_Prediction_of_Manufacturing_Processes.pdf) · [Authors' publication list](https://sites.gatech.edu/jianjun-shi/publications/)

The paper was published online in March 2024 and included in volume 22 of 2025, so the year in the DOI differs from the volume year. This repository cites the formal volume year 2025.

### Research problem and core method

Final quality labels of manufacturing processes usually require expensive or even destructive inspection, so genuinely labeled data is scarce; physics models can provide low-cost predictions but suffer from parameter uncertainty and model bias. The paper proposes PWL, which uses physics-model outputs as weak labels and trains a quality-prediction model together with a small number of true labels.

The prediction structure followed by this project is:

$$
\hat y = H(x^{\mathrm{ph}},\theta)g + B(x^{\mathrm{pr}},x^{\mathrm{ph}})d.
$$

- `H g`: constructs features from physics-related variables and calibration parameters, learning the predictive relationship from physical weak labels.
- `B d`: uses process variables and other information to compensate for discrepancies the physics model cannot fully explain.
- `theta`: physics-model calibration parameters learned jointly; their identifiability must be verified separately.
- Training combines true-label fitting, physics weak-supervision fitting, and L1 and group-sparsity regularization; parameters are updated alternately via block coordinate descent (BCD), with ADMM solving the corresponding subproblems.

For method details, see the original paper; for this project's actual objective function, stabilization terms, and engineering assumptions, see the [reproduction technical notes](reproduction/REPRODUCTION.md#数学实现对应).

### Mapping between the paper and this repository

| Content | Question addressed in the paper | Corresponding location here |
|---|---|---|
| PWL modeling & optimization | Physical weak labels, discrepancy compensation, and joint parameter estimation | `reproduction/src/pwl_repro/core/` |
| Section IV-A | Predictive performance as the number of labeled samples varies | `--mode sample-size` |
| Section IV-B | Effect of physics-model accuracy on learning | `--mode physics-accuracy` |
| Section IV-C | Whether physics weak supervision reduces true-label requirements | `--mode label-savings` |
| 1D steady-state heat-conduction test | This project's independent transfer and applicability check of PWL | `migration/`; not an experiment from the original paper |

Data generation for the paper's simulations follows equations (9)–(11); for the exact splits, correlation targets, and run configuration, see the [Section IV reproduction protocol](reproduction/SECTION_IV_PROTOCOL.md). Details not disclosed in the paper — basis functions, input covariance, numerical handling — are explicitly completed by this project; the advantages reported in the paper must not be taken as results already achieved by this repository. Other experimental cases from the original paper are outside this project's scope.

### Citing the paper

To cite the original method, use the BibTeX below; when citing this repository's reproduction or heat-conduction results, also record the actual commit hash, configuration, and reports used.

```bibtex
@article{alenezi2025pwl,
  author  = {Alenezi, Dhari F. and Biehler, Michael and Shi, Jianjun and Li, Jing},
  title   = {Physics-Informed Weakly-Supervised Learning for Quality Prediction of Manufacturing Processes},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2025},
  volume  = {22},
  pages   = {2006--2019},
  doi     = {10.1109/TASE.2024.3374098}
}
```

## Two experiment lines

| Part | Content | Entry |
|---|---|---|
| Paper reproduction | PWL core, three Section IV simulation groups | [reproduction/README.md](reproduction/README.md) |
| Heat-conduction transfer | 40 batches, PWL/grey-box/GP comparisons, S1 and mechanism stratification | [migration/README.md](migration/README.md) |

## Getting and running

Requires Git, uv, and Python 3.11 (the project declares support for Python ≥3.11, <3.14; this verification used 3.11). Run in a terminal with uv installed; commands use single-line format, suitable for PowerShell and common Unix shells.

```sh
git clone --branch HeatTest https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning.git
cd Physics-Informed-Weakly-Supervised-Learning
uv sync --locked --python 3.11 --extra dev
uv run pytest
uv run python reproduction/run_reproduction.py --config reproduction/configs/smoke.yaml --output reproduction/results/quickstart
uv run python migration/run_migration.py --config migration/configs/heat_smoke.yaml --output migration/results/quickstart
```

Both packages share the root `pyproject.toml` and `uv.lock`; all commands run from the repository root. The included CSVs are sufficient for the flows above; COMSOL is not required. The first environment setup needs to download Python/dependencies.

## Current conclusions

- **Algorithm assets**: reusable feature interfaces, PWLRegressor, BCD–ADMM, and an experiment tuning/statistics pipeline.
- **Reproduction boundary**: only the paper's simulations and optimization structure are reproduced; implementation choices not disclosed in the paper are recorded as engineering assumptions.
- **Heat-conduction S1**: for batches 21–40, the @120 RMSE is **4.727 K** for A4 and **5.449 K** for PWL. In the current scenario, the grey-box A4 with strong mechanistic priors is more suitable; PWL's advantage over GP/Physics after Holm correction is insufficiently supported. See the [S1 report](migration/reports/迁移后续工作/PWL盲测S1报告.md).
- **Heat-conduction stratified tests**: G0/G1 are development-phase evidence and cannot serve as confirmation results on new batches.

For full evidence, historical baselines, and follow-up priorities, see [current useful content and next steps](docs/CURRENT_STATUS.md). Do not generalize the earlier "12-level mean optimum" to the latest comparison set that includes A4.

## Data and reproducibility

- [Heat-conduction data notes](migration/datasets/README.md): specifications and provenance of batches 1–20; for the in-repo generator, acceptance, and status of batches 21–40, see the [migration notes](migration/README.md).
- Code, configs, CSV inputs, acceptance reports, manual reports, and dependency locks are committed to Git; run results, caches, and logs are not uploaded by default — the manually frozen [v3 baseline card](migration/results/heat_v3_qint_min/BASELINE.md) is the exception.
- Figures under `results/` in historical reports must be generated locally; the public repository does not contain complete artifacts of every historical run.

Verification: run both test suites and the independent numerical checks locally; see [current status](docs/CURRENT_STATUS.md). This branch does not configure GitHub Actions automated testing.

## Documentation and feedback

- [Architecture and extension interfaces](docs/ARCHITECTURE.md)
- [Paper-reproduction technical notes](reproduction/REPRODUCTION.md)
- [Reference materials and paper analysis](references/README.md)

When reusing research results, distinguish the original paper, third-party data sources, and this repository's engineering assumptions, and record the branch/commit, configuration, batches, and random seeds. To report problems, open an [Issue](https://github.com/WindFromKadath/Physics-Informed-Weakly-Supervised-Learning/issues) with the command, Python version, configuration, and minimal error message. When modifying models or experiment protocols, update the conclusion boundaries in the related tests and reports accordingly.

## License

The project's original code and original accompanying documentation are licensed under the [MIT License](LICENSE), Copyright (c) 2026 WindFromKadath.

This license does not cover third-party papers, extracted paper text, cited figures, or other third-party materials, including the paper content in `references/papers/`, `references/extracted_text/`, and `references/tools/eq_dump.txt`; rights in those materials belong to the original authors or publishers and must be used under their original terms. Datasets are not covered by this MIT grant; see the [dataset notes](migration/datasets/README.md) for provenance and generation. Third-party dependencies remain under their own licenses. This repository's MIT license does not imply endorsement or authorization of this implementation by the original paper's authors.
