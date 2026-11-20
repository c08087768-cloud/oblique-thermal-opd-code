# Thermo-optic response of cylindrical optics under oblique Gaussian heating

This repository computes the steady and transient temperature rise in a finite cylinder subject to oblique Gaussian surface heating and Robin heat exchange. The semi-analytical solver expands the thermal field in Fourier–Bessel modes. A separately assembled finite-element model (FEM) on a prismatic mesh integrates the Gaussian load directly for comparison. The optical path difference (OPD) is the first-order refractive-index contribution integrated along fixed, straight refracted rays that reach the rear face.

**Version:** `0.1.0` ([fixed source tag `v0.1.0`](https://github.com/c08087768-cloud/oblique-thermal-opd-code/tree/v0.1.0)).

## Requirements and installation

Clone this repository and enter its directory first:

```bash
git clone https://github.com/c08087768-cloud/oblique-thermal-opd-code.git
cd oblique-thermal-opd-code
```

Use Python 3.10 or later and Git. The core solver uses NumPy and SciPy. The installation below also provides Matplotlib for plots, scikit-fem for finite-element runs, and pytest for software tests. Run the commands from the repository root. [Dependency specification](pyproject.toml)

**Linux / WSL (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,fem]"
python -m pip check
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,fem]"
.\.venv\Scripts\python.exe -m pip check
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

## Repository structure

| Path | Contents |
| --- | --- |
| [`src/thermal_opd/`](src/thermal_opd/) | Heat-source projection, modal and FEM solvers, OPD integration, and validation |
| [`scripts/`](scripts/) | Single cases, convergence studies, independent benchmarks, diagnostics, and replotting |
| [`run.py`](run.py), [`run_all.py`](run_all.py) | Interactive experiment launcher and batch workflow |
| [`tests/`](tests/) | Software regression tests |
| [`data/reference/`](data/reference/) | Historical sampling inputs, selected numerical tables, derived optical arrays, and provenance |

## Reproducing the numerical experiments

These commands generate new numerical results from the physical parameters and Gaussian heat-source model; they do not require an external experimental dataset. Generated runs are saved under `outputs/`, which is excluded from Git. Replaying selected archived optical comparisons is a separate workflow with additional saved-run requirements; see the [reference-data notes](data/reference/README.md).

### Steady example at 60°

Choose an unused output directory name if rerunning the example.

```bash
python scripts/cases/run_steady_case.py --angle 60 --output-dir outputs/steady_case/example-60deg
python scripts/verification/validate_run.py --results outputs/steady_case/example-60deg
```

The script saves the temperature and OPD arrays in `data/fields.npz` and a plot in `figures/steady_case.png` under the run directory. The sampled peak-to-valley OPD is stored in metres as `opd_pv_m` in `summary.json`. The validator reports saved-file integrity and scientific status separately. [Saved-run validator](scripts/verification/validate_run.py)

Modal source preparation requires positive effective surface heat transfer. Fully insulated cases are rejected because the required axisymmetric radial zero mode is not implemented.

The nominal cylinder has a radius of 12.7 mm and a thickness of 5 mm. The Gaussian beam's 1/e² intensity radius normal to propagation is 0.30 times the cylinder radius. The assigned absorbed power is 110 mW on the full plane. The refractive index and its temperature derivative come from the same Corning 7980 Sellmeier fit at 1064 nm and 300 K. The example retains azimuthal orders 0, 2, 4, and 6 and eight radial modes per order. It uses 400 radial by 32 angular projection samples, 16 ray quadrature points, and a 201 × 201 entrance-plane grid for OPD with an aperture radius of 0.9 times the cylinder radius. [Physical inputs](src/thermal_opd/parameters.py) · [Steady case settings](scripts/cases/run_steady_case.py) · [Sellmeier coefficients, Table 3](https://ntrs.nasa.gov/api/citations/20070018851/downloads/20070018851.pdf#page=6)

### Steady numerical settings

For a standalone check of the steady postprocessing settings, run
`python run.py --experiment numerical-settings --yes --angles 0 30 60`.
The audit reads the production defaults and varies the extrema seeds, OPD batch size,
sidewall samples, and energy quadrature counts. It writes the measured changes and timings to
`outputs/numerical_settings_audit/<run-id>/data/comparison.csv` and saves the
acceptance checks in `verification.json`.
The finest seed grids are numerical references. Batch timing is diagnostic.
Every tested batch size must pass the same pointwise OPD consistency check.

### Batch workflow

```bash
python run_all.py --mode smoke --angles 60
python run_all.py --mode full --angles 0 30 60 --plan
python run_all.py --mode full --angles 0 30 60
```

The smoke batch runs the software tests, the independent uniform-flux FEM benchmark, and a small transient comparison with boundary checks. The full batch adds truncation selection, detailed fields, and FEM refinement studies. Use `--plan` to inspect the stages without starting calculations. The full run can be computationally demanding and may stop if a prerequisite fails. Its `summary.json` reports the aggregate scientific status. For a full batch, exit code `2` means execution finished with incomplete or failed scientific evidence. [Batch implementation](run_all.py)

The experiment drivers generate the heat loads, meshes, and sampling grids from their inputs. The workflows above require no random seed or external input dataset. Individual experiments record settings and provenance in `config.json` and scientific checks in `verification.json`. [Run metadata](src/thermal_opd/run_io.py)

## Numerical reference results and data

The 60° semi-analytical case gives a sampled peak-to-valley OPD of **135.07 nm** over entrance-plane points whose rays reach the rear face. The steady example reproduced the tabulated value with a saved-file integrity status of `PASS` and an aggregate scientific status of `not_run`. Separate finite-reference checks report differences for 0°, 30°, and 60°. [Steady OPD table](data/reference/supplement_details/pupil_pv.csv) · [Semi-analytical checks](data/reference/supplement_details/sa_checks.csv)

The repository includes selected numerical reference tables, derived optical arrays, and provenance under [`data/reference/`](data/reference/). The recorded comparisons include semi-analytical finite-reference checks, cross-method comparisons with FEM, FEM optical refinement on refracted paths, and steady front-load quadrature checks. The refinement and quadrature results reuse archived thermal solutions and quantify numerical sensitivity; they are not experimental validation or a complete standard-loader scientific certificate. [FEM optical refinement evidence](data/reference/supplement_details/fem_optical_refinement_provenance.json) · [Derived optical data](data/reference/optical_update_20261103/)

The three `data/reference/theta*_history.csv` inputs preserve historical time grids and thermal plotting values for the optional optical replay script. Their legacy optical columns are not current matched-material results. See the [input scope and replay requirements](data/reference/README.md).
