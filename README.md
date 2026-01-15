# RICE13_FS
**Fehr–Schmidt Inequality Aversion in the RICE Integrated Assessment Model**

This repository extends a Pyomo implementation of RICE2013 with **Fehr–Schmidt (FS) inequality aversion** to separate **interregional equity concerns** from **intergenerational discounting**. It supports:
- cooperative (social planner) solutions,
- noncooperative Nash equilibria,
- and coalition games with stability diagnostics,
under both CRRA and FS welfare specifications.

This repo is intended to reproduce the results in the paper:

> **Fehr--Schmidt Inequality Aversion in the RICE Integrated Assessment Model**

Replication instructions are described in the paper’s reproduction appendix and implemented here.

---

## Origin and relationship to upstream
This repository is a fork of:

- `white-heomoi/RICE13_pyomo`

The main additions in this fork are:
- **FS preferences** with tunable envy/guilt components,
- a **discounting-adjustment** option to align FS intertemporal weighting to a CRRA benchmark,
- scenario generation for **planner**, **Nash**, and **coalitions**,
- a mandatory SQLite **coalition cache** (single source of truth for coalition exports),
- a `Scenario_explorer.ipynb` notebook to recreate paper figures/tables from exported scenarios,
- and utilities for exporting results to **IAMC/pyam** format.

---

## Repository layout

- `RICE13_FS/` — the Python package
  - `cli.py` — command line entrypoint
  - `pyam_exporter.py` — used by the Scenario_explorer to load Excel scenarios
  - `analysis/` — orchestration (BAU → planners → Nash → coalitions) and Negishi weight computation
  - `core/` — model construction + data loading
  - `solve/` — BAU/planner/Nash/coalition solve routines
  - `output/` — Excel export + caching
- `RICE13_FS/Data/` — RICE2013 input data plus FS parameter files / calibrations
- `RICE13_FS/Results/` — Excel outputs (scenarios, stability sheets, overview tables)
- `RICE13_FS/Cache/` — SQLite cache for coalition solutions (mandatory)
- `RICE13_FS/Diagnostics/` — optional IPOPT logs/plots (when enabled)
- `Scenario_explorer.ipynb` — recreates figures/tables from exported scenarios
- `config.yaml` — run configuration

---

## Installation

### 1) Python dependencies
Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) External solver: IPOPT

This project solves nonlinear programs via Pyomo + IPOPT.

You need an **IPOPT executable** available on your PATH, or set `ipopt_executable:` in `config.yaml`.

> Tip: if you use conda, IPOPT is commonly available via conda-forge.

### 3) Optional: pyam exporter

If you want to run `Scenario_explorer.ipynb` or convert scenario Excel files to IAMC format, install:

```bash
python -m pip install pyam-iamc
```

The import is `import pyam`, but the pip package name is `pyam-iamc`.

If `import pyam` fails or imports the wrong thing, check:

```bash
python -m pip show pyam-iamc
python -c "import pyam; print(pyam.__version__)"
```

---

## Quick start

### Run the model (preferred: as a module)

From the repository root:

```bash
python -m RICE13_FS.cli -c config.yaml --log-level INFO
```

This will:

1. optionally run BAU,
2. optionally compute/load Negishi weights,
3. run cooperative planners (CRRA and/or FS),
4. run Nash equilibria (CRRA and/or FS),
5. and/or run coalition solves with stability diagnostics,
   depending on flags in `config.yaml`.

Outputs are written to the directories set in the config, typically:

* Excel workbooks to `results_dir`
* cache entries to `cache_dir`
* optional logs/plots to `diagnostics_dir`

---

## Configuration overview (`config.yaml`)

The config is designed to support a “scenario pipeline” controlled by booleans and mode switches.

Highlights:

* `T`: horizon in model periods (fixed 10-year grid; data supports up to 59)
* Paths:

  * `data_path`, `results_dir`, `output_dir`, `diagnostics_dir`, `cache_dir`
* Negishi:

  * `negishi_use` and `negishi_source` (`bau`, `file`, `fs_after_disc`)
* FS discounting / alignment (optional):

  * `fs_disc_enabled`, `fs_disc_mode` (`off`, `file`, `one_pass`, `two_pass`)
* Switches:

  * `run_bau`, `run_planner_crra`, `run_planner_fs`, `run_nash_crra`, `run_nash_fs`,
    `run_coalition_crra`, `run_coalition_fs`
* Savings modes (`*_S_mode`):

  * typically `bau`, `optimal`, `file`, or “use results from another regime”
* Coalitions:

  * `coalition` can be `GRAND`, a comma list (`"US,EU,..."`), or a bitmask (`101010101010`)
  * `mega_run` toggles full enumeration where supported (expensive, runs for many hours or days)

* Cache:

  * `cache_namespace` lets you isolate runs when specs/inputs change
  * `cache_allow_mismatch=false` is the safe default

---

## Reproducing the paper

### Option A: Use precomputed scenarios (fastest)

This repo includes a scenarios folder with **precomputed scenario Excel workbooks** and **coalition stability overview tables**.

1. Ensure the folder exists (`scenarios/`), containing the exported Excel results.
2. Open `Scenario_explorer.ipynb`.
3. Set:

   ```python
   scenario_folder = Path("scenarios")  # adjust this if you change the name
   ```
4. Run the notebook to regenerate the figures and tables used in the paper.

### Option B: Recompute scenarios from scratch

1. Edit `RICE13_FS/config.yaml` to select the scenarios you want (planner/Nash/coalitions, CRRA/FS, discounting mode, etc.).
2. Run from the repository root (the directory containing `RICE13_FS/`):

```bash
   python -m RICE13_FS.cli -c RICE13_FS/config.yaml --log-level INFO
```

3. Open `RICE13_FS/Scenario_explorer.ipynb` and point it to the folder containing the exported Excel workbooks:

   * either the `results_dir` configured in `RICE13_FS/config.yaml`, or
   * a curated subset you copied into a dedicated folder (e.g. `RICE13_FS/scenarios/`).

> Coalition runs can be expensive; the SQLite cache (configured via `cache_dir`/`cache_namespace`) is mandatory and is used to avoid recomputation and to standardize exports.

---

## Notebook: `Scenario_explorer.ipynb`

The notebook is meant to recreate paper outputs and additional material, including sections like:

* World temperature
* World inequality
* Carbon taxes / SCC
* Table extraction utilities
* Figures used in the paper
* Nash comparisons

It expects a folder of scenario workbooks and uses the pyam exporter to load them into a single `pyam.IamDataFrame`.

---

## IAMC / pyam export

There is a small helper module to read exported Excel files and build an IAMC-style dataset:

```python
from pathlib import Path
from RICE13_FS.pyam_exporter import build_iamdf

folder = Path("scenarios")   # folder of exported Excel workbooks
iamdf = build_iamdf(folder)
```

You can then filter/plot using `pyam`.

---

## Troubleshooting

### IPOPT not found

* If Pyomo errors with “no executable found”, either:

  * install IPOPT and make sure it’s on PATH, or
  * set `ipopt_executable:` in `config.yaml`.

### Cache mismatch errors

* If `cache_allow_mismatch=false` and you changed inputs/specs, create a new namespace:

  * `cache_namespace: cache2`
* Setting `cache_allow_mismatch=true` is generally not recommended.

### `import pyam` issues

Install the correct package:

```bash
python -m pip install pyam-iamc
```

If you accidentally installed the other `pyam`, uninstall it:

```bash
python -m pip uninstall pyam
```

---

## Acknowledgements

This repository was developed with extensive assistance from OpenAI’s ChatGPT, including iterative design, debugging, refactoring, and documentation support. All code and results were reviewed and validated by the author, who remains responsible for any remaining errors.

## License

License: CC0-1.0. This repository (and the upstream fork base) is dedicated to the public domain under CC0. See LICENSE.
