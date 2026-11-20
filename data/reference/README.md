# Numerical reference inputs and outputs

This directory contains historical sampling inputs and selected derived numerical results. The example drivers construct the Gaussian load and evaluation grids from their settings; FEM workflows also construct their mesh. These new runs require no external experimental dataset and do not read the reference tables.

## Contents

- `theta0_history.csv`, `theta30_history.csv`, `theta60_history.csv`: existing solver exports, preserved byte-for-byte, with 131 rows each. The optional optical replay reads their positive times through 75 s. Their historical `sampled_PV_nm` values predate the matched optical parameters and are not current OPD reference results.
- [`supplement_details/`](supplement_details/): selected numerical tables and JSON provenance. Current sampled/continuous OPD-PV values are in [`pupil_pv.csv`](supplement_details/pupil_pv.csv); finite semi-analytical reference differences are in [`sa_checks.csv`](supplement_details/sa_checks.csv). Other tables record FEM comparisons, optical refinement, and front-load quadrature checks.
- [`optical_update_20261103/`](optical_update_20261103/): selected derived optical arrays and an export manifest. These are not the complete archived FEM nodal fields.

## Optional archived optical replay

[`recompute_paper_optics.py`](../../scripts/cases/recompute_paper_optics.py) retains its existing name and numerical settings. Its `sa` stage uses the three versioned history CSVs to recover the sampling times and installed project dependencies; it does not require a previous `outputs/` archive.

Create a new directory directly under `outputs/optical_parameter_update/` first, then run from the repository root:

```bash
python scripts/cases/recompute_paper_optics.py --run-directory outputs/optical_parameter_update/example-rebuild --angle 60 --stage sa
```

Use 0 or 30 for the other recorded angles. Reusing completed stage directories is rejected. Other stages can require pre-existing SA outputs or archived FEM solutions and settings; they are not stand-alone fresh-run examples. Original source paths in the JSON provenance describe the historical archive layout, not files included in this code distribution. Start new experiments with the [main README examples](../../README.md).

## Preserved input checksums

| Input | SHA-256 |
| --- | --- |
| theta0_history.csv | 27820d4f84da40abc6a841c2754ed552e9974fbe92e4ed035b9f1e564ff52530 |
| theta30_history.csv | 8be0f746da813c29ee8b8b000f787dbf71ce2ef9b38d8089f4fc46144f13b170 |
| theta60_history.csv | 2694b40fb828dd92744e6a3b05cd2e580268aec90fc9ea3444fc005b88ebeb02 |

Moving these numerical files changes their location only. It does not add a new numerical validation or turn historical optical columns into current results.
