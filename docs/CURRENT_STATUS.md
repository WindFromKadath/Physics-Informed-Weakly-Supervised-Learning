# Current Status: Paper-Simulation Reproduction and Heat-Conduction Tests

[中文](CURRENT_STATUS.zh-CN.md) | English

This repository keeps only two parts: the reproduction of the paper's Section IV simulations, and the one-dimensional steady-state heat-conduction tests actually in use. Code frameworks, configurations, data, and plans for other applications are not implemented content of this project and are not included in the current version.

## Paper-simulation reproduction

- PWL feature libraries, BCD–ADMM, proximal operators, and model fitting.
- Three protocols — sample size, physics accuracy, and label savings — plus related sensitivity analyses.
- The reproduction is structural; H/B, covariance, noise, and singular-point handling contain explicit engineering assumptions.

Entries: [reproduction README](../reproduction/README.md), [technical notes](../reproduction/REPRODUCTION.md), [Section IV protocol](../reproduction/SECTION_IV_PROTOCOL.md).

## Heat-conduction tests

Keeps batches 1–40 data, the v3 baseline, ablations, output alignment, S1 and G0/G1 development tests, and the corresponding configurations and acceptance/analysis scripts. They all belong to the same heat-conduction test line.

- S1 @120 RMSE: A4 4.727 K, PWL 5.449 K, GP 5.612 K, Physics 5.554 K.
- A4 outperforms PWL in the current strong-mechanism-prior scenario; PWL's advantage over GP/Physics after Holm correction is insufficiently supported.
- Weak-label value appears mainly at the small-sample end; G0/G1 remain development-stage evidence.
- Batches 21–40 became development data after S1 unblinding and cannot serve as an unseen blind test again.

Entries: [heat-conduction README](../migration/README.md), [S1 report](../migration/reports/迁移后续工作/PWL盲测S1报告.md) (Chinese), [G0/G1 report](../migration/reports/迁移后续工作/PWL分层G0G1开发测试报告.md) (Chinese), [v3 baseline card](../migration/results/heat_v3_qint_min/BASELINE.md). The numbers above cite existing reports; this scope cleanup did not re-run the full experiments.

## Verification

Verification after the 2026-09-11 scope cleanup:

| Check | Result |
|---|---|
| Retained simulation and heat-conduction tests | 60 passed |
| Independent simulation numerical checks | 30 passed |
| Simulation smoke | Completed; 24 metrics, 1920 predictions |
| Heat-conduction smoke | Completed; 48 metrics, 9600 predictions |

Both smokes raised warnings about GP kernel parameters hitting search boundaries; the flows completed normally — this does not mean all scientific quality anchors pass. Historical test counts that included other scenarios are no longer used as the current version's verification count.

For shared implementation and directory responsibilities, see the [architecture notes](ARCHITECTURE.md). Run results, logs, and caches are not uploaded by default; historical figures must be generated locally by running the corresponding scripts.

### Local verification after the floating-point boundary fix

GitHub Linux tests once produced a maximum canonical correlation of `1.0000000000000002`, tripping the strict upper-bound assertion. The diagnostic function now clamps QR/SVD results to the theoretical range `[0, 1]`, keeps the original test assertions, and adds rounding-overflow and known-angle tests. The full local test result is **64 passed**; this round was not re-run on Linux.

Per project maintenance requirements, `.vscode` and `.github/workflows` have been removed and local test commands retained; this branch no longer configures automatic CI. Deleting the workflows does not erase existing run records on GitHub.
