# pyam_exporter.py
"""
Utilities to collect RICE13_FS Excel outputs into an IAMC-style pyam.IamDataFrame.

Usage (from a Jupyter notebook, for example)
--------------------------------------------
from pathlib import Path
from RICE13_FS.pyam_exporter import build_iamdf

folder = Path("path/to/scenario_excels")
iamdf = build_iamdf(folder)

# Example plots:
iamdf.filter(region="US", variable="Emissions|CO2|Abatement Rate").plot()
iamdf.filter(region="World", variable="Temperature|Global Mean").plot()

Notes
-----
- This reads the *Excel workbooks produced by the RICE13_FS export layer* (see
  `RICE13_FS.output.results`). The workbooks are expected to have:
    * one sheet per region (index = row labels, columns = years)
    * a "global" sheet (same structure)
    * a "config" sheet that contains the run configuration printed as YAML
      (used here to reconstruct `model`, `scenario`, and metadata/tags).
- The output is an IAMC-wide-format table wrapped as `pyam.IamDataFrame`, with
  scenario-level metadata populated from the config sheet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import pyam
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: sheet names & variable dictionary
# ---------------------------------------------------------------------------

# Region sheet names in the RICE13_FS tabular exports.
# These are expected to match the worksheet names written by the Excel exporter.
# (Some codes are historical/legacy RICE region abbreviations; the exporter may
# omit a region sheet in some runs, and we skip those silently.)
REGION_SHEETS: List[str] = [
    "US",
    "EU",
    "JAP",
    "RUS",
    "EUR",
    "CHI",
    "IND",
    "MEST",
    "AFR",
    "LAM",
    "OHI",
    "OTH",
]

GLOBAL_SHEET_NAME = "global"
CONFIG_SHEET_NAME = "config"

# Direct regional variables: one Excel row -> one pyam variable
# (series_transform receives a 1D Series indexed by years and returns a Series)
RegionalDirectSpec = Tuple[str, str, str]  # (excel_row_label, pyam_variable, unit)

# Rows that are optional (no warning if missing)
OPTIONAL_ROW_LABELS = {
    "SCC in regional (k$/tC)",
    "SCC in per capita global (k$/tC)",
    # Inequality diagnostics are new and may be absent in older workbooks
    "Global Gini (C/L, representative agents)",
    "Global Atkinson (ε=1.5, C/L, representative agents)",
    "Global Atkinson (1.5, C/L, representative agents)",
}

# Row-label aliases: map a canonical label used in GLOBAL_DIRECT_VARS /
# REGIONAL_DIRECT_VARS to alternative labels that may appear in older/newer
# workbooks.
ROW_LABEL_ALIASES: Dict[str, List[str]] = {
    # Old patched scenarios: "Global Atkinson (ε=1.5, ...)"
    # New results.py scenarios: "Global Atkinson (1.5, ...)"
    "Global Atkinson (ε=1.5, C/L, representative agents)": [
        "Global Atkinson (1.5, C/L, representative agents)"
    ],
}

REGIONAL_DIRECT_VARS: List[RegionalDirectSpec] = [
    # Consumption per capita
    ("Consumption per capita (C/L)", "Consumption|Per Capita", "USD/person/yr"),
    # Abatement rate μ (share)
    ("Abatement rate (μ)", "Emissions|CO2|Abatement Rate", "1"),
    # Carbon tax (k$/tC) -> Carbon price in USD/tCO2
    ("Carbon tax (k$/tC)", "Price|Carbon", "USD/tCO2"),
    # Regional SCC in local money numéraire (k$/tC) -> USD/tCO2
    #
    # Note: this is the SCC backed out in regional currency units along the
    # cooperative planner path (or any run that populates the corresponding
    # Excel row). We convert k$/tC -> USD/tCO2 using the same factor as the
    # carbon tax:
    #   1 k$ / tC  = 1000 USD / tC
    #   1 tCO2     = (44.01/12.01) tC  ≈ 3.664 tC
    #   => 1 k$/tC ≈ 1000/3.664 USD/tCO2
    ("SCC in regional (k$/tC)", "Price|Carbon|SCC|Regional", "USD/tCO2"),
    # Industrial emissions (GtC/yr) -> GtCO2/yr
    ("Industrial Emissions", "Emissions|CO2|Industrial", "GtCO2/yr"),
]

# Derived regional variables: function of the whole regional sheet
# Callable: df_region -> Series indexed by years
RegionalDerivedSpec = Tuple[str, str, str]  # (name, unit, kind)


def _damage_share(df_region: pd.DataFrame) -> pd.Series:
    """Damage share of potential GDP = D / Q."""
    D = df_region.loc["Climate Damage (D)"]
    Q = df_region.loc["Gross Output (Q)"]
    return D / Q


def _abatement_share(df_region: pd.DataFrame) -> pd.Series:
    """Abatement expenditure share of potential GDP = AB / Q."""
    AB = df_region.loc["Abatement expenditure (AB)"]
    Q = df_region.loc["Gross Output (Q)"]
    return AB / Q


# (pyam_variable, unit, "function-name")
REGIONAL_DERIVED_VARS: Dict[str, Tuple[str, str]] = {
    "Damages|GDP|Share": ("Damages|GDP|Share", "1"),
    "Policy Cost|Abatement Expenditure|Share": (
        "Policy Cost|Abatement Expenditure|Share",
        "1",
    ),
}

# Mapping from derived variable name to the actual function
REGIONAL_DERIVED_FUNCS = {
    "Damages|GDP|Share": _damage_share,
    "Policy Cost|Abatement Expenditure|Share": _abatement_share,
}

# Global variables (direct; one Excel row -> one pyam variable)
GlobalDirectSpec = Tuple[str, str, str]  # (excel_row_label, pyam_variable, unit)


GLOBAL_DIRECT_VARS: List[GlobalDirectSpec] = [
    # Atmospheric temperature increase (°C)
    ("Atmospheric temperature increase", "Temperature|Global Mean", "°C"),
    # Total carbon emissions (GtC/yr) -> GtCO2/yr
    ("Total carbon emissions (GtC per year)", "Emissions|CO2", "GtCO2/yr"),
    # Sea level rise (total), in meters
    ("Sea level rise (total)", "Sea Level Rise", "m"),
    # Global SCC in equal-per-capita money numéraire
    (
        "SCC in per capita global (k$/tC)",
        "Price|Carbon|SCC|GlobalPerCapita",
        "USD/tCO2",
    ),
    # Global inequality diagnostics (between-region, pop-weighted, per-capita consumption)
    (
        "Global Gini (C/L, representative agents)",
        "Inequality|Gini Index|Consumption|Per Capita",
        "1",
    ),
    (
        "Global Atkinson (ε=1.5, C/L, representative agents)",
        "Inequality|Atkinson Index|Consumption|Per Capita",
        "1",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_config_from_workbook(path: Path) -> Dict[str, Any]:
    """
    Read the 'config' sheet and reconstruct the YAML dictionary that was written
    by the Excel export utilities (e.g. `RICE13_FS.output.results.results_to_excel`).

    Returns
    -------
    dict
        Parsed YAML config dict, or `{}` if the sheet/column is missing or YAML
        parsing fails. Callers should treat missing keys as "unknown" and fall
        back to defaults (e.g. scenario = filename stem).
    """
    try:
        df_cfg = pd.read_excel(path, sheet_name=CONFIG_SHEET_NAME)
    except Exception as exc:  # sheet missing or other issue
        logger.warning("Could not read config sheet from %s: %s", path.name, exc)
        return {}

    if "config" not in df_cfg.columns:
        logger.warning("Config sheet in %s has no 'config' column; skipping.", path.name)
        return {}

    lines = [str(x) for x in df_cfg["config"].dropna().tolist()]
    text = "\n".join(lines)
    try:
        cfg = yaml.safe_load(text) or {}
        if not isinstance(cfg, dict):
            logger.warning("Config in %s is not a dict; got %r", path.name, type(cfg))
            return {}
        return cfg
    except Exception as exc:
        logger.warning("Failed to parse YAML config from %s: %s", path.name, exc)
        return {}


def _year_columns(df: pd.DataFrame) -> List[Any]:
    """
    Infer year columns from a sheet.

    The tabular Excel exports use *calendar years* as columns (integers like
    2015, 2025, ...). We treat any column whose string representation is purely
    digits as a "year column" and sort numerically.

    (If you change the exporter to prefix year columns, update this logic.)
    """
    cols = []
    for c in df.columns:
        s = str(c)
        if s.isdigit():
            # Keep original label (int or string), but ensure we can sort later
            cols.append(c)
    # Sort numerically if possible
    try:
        cols_sorted = sorted(cols, key=lambda x: int(str(x)))
    except Exception:
        cols_sorted = cols
    return cols_sorted


def _safe_get_row(
    df: pd.DataFrame, label: str, sheet_name: str, src_file: Path
) -> pd.Series | None:
    """Try to get a row by label; return None if missing.

    - First try the canonical label.
    - If not found, try any aliases from ROW_LABEL_ALIASES.
    - For labels in OPTIONAL_ROW_LABELS we stay quiet at the default INFO level
      and only emit a DEBUG message.
    - Missing non-optional rows still produce a WARNING.
    """
    # Canonical label present?
    if label in df.index:
        return df.loc[label]

    # Try aliases for this label (if any)
    for alt in ROW_LABEL_ALIASES.get(label, []):
        if alt in df.index:
            logger.info(
                "Row %r not found in sheet %s of %s; using alias %r.",
                label,
                sheet_name,
                src_file.name,
                alt,
            )
            return df.loc[alt]

    # No canonical label or alias found: handle as missing
    if label in OPTIONAL_ROW_LABELS:
        logger.debug(
            "Row %r not found in sheet %s of %s; skipping optional variable.",
            label,
            sheet_name,
            src_file.name,
        )
    else:
        logger.warning(
           "Row %r not found in sheet %s of %s; skipping this variable.",
            label,
            sheet_name,
            src_file.name,
        )
    return None


def _series_to_row_dict(
    series: pd.Series,
    years: Iterable[Any],
    *,
    model: str,
    scenario: str,
    region: str,
    variable: str,
    unit: str,
) -> Dict[str, Any]:
    """
    Convert a 1D time series (indexed by year columns) to a single row dict in
    IAMC wide format: model, scenario, region, variable, unit, <years...>.
    """
    row: Dict[str, Any] = {
        "model": model,
        "scenario": scenario,
        "region": region,
        "variable": variable,
        "unit": unit,
    }
    for y in years:
        # Some Excel readers may return NaN or non-numeric; just copy through
        row[int(str(y))] = series.get(y)
    return row


def _collect_from_workbook(path: Path) -> List[Dict[str, Any]]:
    """
    Collect IAMC-style rows from a single Excel workbook produced by the
    RICE13_FS Excel exporter (planner/Nash/coalition outputs).

    Returns a list of dicts (rows) to be concatenated into a DataFrame.
    """
    logger.info("Processing workbook %s", path.name)

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        logger.warning("Could not open %s as an Excel workbook: %s", path.name, exc)
        return []

    cfg = _read_config_from_workbook(path)
    model = str(cfg.get("model", "RICE13_FS_Clone"))
    # Fallback to stem if scenario is missing
    scenario = str(cfg.get("scenario", path.stem))

    rows: List[Dict[str, Any]] = []

    # Define a helper function for unit conversion
    def convert_to_usd_per_tco2(series: pd.Series, pyam_var: str) -> pd.Series:
        """
        Convert carbon-price-like variables from k$/tC (as written in the Excel
        workbook) to USD/tCO2 (IAMC convention).

        Convention:
          - Any pyam variable starting with `Price|Carbon` is assumed to be
            expressed in k$/tC in the workbook (including SCC rows).
          - Conversion uses the molecular weight ratio CO2/C ≈ 44.01/12.01 ≈ 3.664.
        """
        if pyam_var.startswith("Price|Carbon"):
            return series * (1000.0 / 3.664)
        return series
    
    # --- Regional sheets ---
    for region in REGION_SHEETS:
        if region not in xls.sheet_names:
            # Not all workbooks necessarily have all regions; skip silently
            continue
    
        df_reg = pd.read_excel(xls, sheet_name=region, index_col=0)
        years = _year_columns(df_reg)
        if not years:
            logger.warning(
                "No year-like columns found in sheet %s of %s; skipping region.",
                region,
                path.name,
            )
            continue
    
        # Direct regional variables
        for excel_label, pyam_var, unit in REGIONAL_DIRECT_VARS:
            series = _safe_get_row(df_reg, excel_label, region, path)
            if series is None:
                continue
    
            # Transform units where needed
            series = convert_to_usd_per_tco2(series, pyam_var)
    
            row = _series_to_row_dict(
                series,
                years,
                model=model,
                scenario=scenario,
                region=region,
                variable=pyam_var,
                unit=unit,
            )
            rows.append(row)

        # Derived regional variables (damage share, abatement share, ...)
        for derived_name, (pyam_var, unit) in REGIONAL_DERIVED_VARS.items():
            func = REGIONAL_DERIVED_FUNCS[derived_name]
            try:
                series = func(df_reg)
            except KeyError as exc:
                logger.info(
                    "Could not compute derived variable %s for region %s in %s: %s; skipping.",
                    derived_name,
                    region,
                    path.name,
                    exc,
                )
                continue

            row = _series_to_row_dict(
                series,
                years,
                model=model,
                scenario=scenario,
                region=region,
                variable=pyam_var,
                unit=unit,
            )
            rows.append(row)

    # --- Global sheet ---
    if GLOBAL_SHEET_NAME in xls.sheet_names:
        df_glob = pd.read_excel(xls, sheet_name=GLOBAL_SHEET_NAME, index_col=0)
        years = _year_columns(df_glob)
        if not years:
            logger.warning(
                "No year-like columns found in global sheet of %s; skipping global series.",
                path.name,
            )
        else:
            for excel_label, pyam_var, unit in GLOBAL_DIRECT_VARS:
                series = _safe_get_row(df_glob, excel_label, GLOBAL_SHEET_NAME, path)
                if series is None:
                    continue
    
                # Transform units where needed
                series = convert_to_usd_per_tco2(series, pyam_var)
    
                row = _series_to_row_dict(
                    series,
                    years,
                    model=model,
                    scenario=scenario,
                    region="World",
                    variable=pyam_var,
                    unit=unit,
                )
                rows.append(row)
    else:
        logger.warning("Workbook %s has no '%s' sheet; skipping global variables.", path.name, GLOBAL_SHEET_NAME)

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_iamdf(folder: str | Path) -> pyam.IamDataFrame:
    """
    Scan a folder of RICE13_FS Excel outputs and build an IAMC-style pyam.IamDataFrame.

    Parameters
    ----------
    folder : str or Path
        Path to a directory containing Excel files produced by RICE13_FS
        (e.g. planner / Nash / coalition exports). Each file is treated as
        one "run", and the `model` and `scenario` strings are read from the
        YAML config printed on the `config` sheet.

    Returns
    -------
    pyam.IamDataFrame
        An IamDataFrame with columns:
        - model
        - scenario
        - region
        - variable
        - unit
        - <year columns...>

    Notes
    -----
    - This function does not try to guess whether a workbook is BAU, planner,
      Nash, or coalition; it simply reads `model` and `scenario` from the
      config sheet and takes the sheet name (US, EU, ...) as `region`.
    - It is up to the user to decide which Excel files to put into `folder`
      (e.g. only planners, or only a subset of coalition runs).
    - Scenario-level metadata is attached (if the config sheet exists), with
      columns like: run_mode, flavor, negishi, discounting_adjustment,
      discounting_mode (FS-only), cache_namespace (optional), and selected
      user tags from cfg["tags"].
    - Tag-style meta keys are normalized so missing values become the string
      "none" (helps filtering in `pyam` without NaN semantics).
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder_path}")

    all_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    # Track all tag-style meta keys so we can normalize them later
    tag_keys: set[str] = set()

    xlsx_files = sorted(p for p in folder_path.glob("*.xlsx") if p.is_file())
    if not xlsx_files:
        logger.warning("No .xlsx files found in folder %s.", folder_path)

    for wb_path in xlsx_files:
        # Read config once per workbook to build scenario-level meta
        cfg = _read_config_from_workbook(wb_path)
        model = str(cfg.get("model", "RICE13_FS_Clone"))
        scenario = str(cfg.get("scenario", wb_path.stem))

        # ------------------------------------------------------------------
        # Scenario-level meta (one row per (model, scenario))
        # ------------------------------------------------------------------
        run_mode = str(cfg.get("run_mode", "none"))
        flavor = str(cfg.get("flavor", "none"))

        negishi_use = bool(cfg.get("negishi_use", False))
        fs_disc_enabled = bool(cfg.get("fs_disc_enabled", False))
        fs_disc_mode = str(cfg.get("fs_disc_mode", "off")).strip().lower()
        discounting_adjustment = (
            flavor == "fs" and fs_disc_enabled and fs_disc_mode != "off"
        )

        meta_row: Dict[str, Any] = {
            "model": model,
            "scenario": scenario,
            "run_mode": run_mode,
            "flavor": flavor,
            "negishi": negishi_use,
            "discounting_adjustment": discounting_adjustment,
        }
        # Only meaningful to expose a discounting mode for FS runs
        if flavor == "fs":
            meta_row["discounting_mode"] = fs_disc_mode

        # Optional: keep cache_namespace as a debugging / grouping aid
        cache_ns = cfg.get("cache_namespace")
        if cache_ns is not None:
            meta_row["cache_namespace"] = str(cache_ns)

        # Flatten user-specified tags with flavor-aware filtering
        tags = cfg.get("tags") or {}
        if isinstance(tags, dict):

            def _add_tag_if_nonempty(key: str) -> None:
                val = tags.get(key, None)
                if val is None:
                    return
                if isinstance(val, str) and not val.strip():
                    return
                meta_row[key] = val
                tag_keys.add(key)

            if flavor == "fs":
                for key in ("fs_params", "fs_disc_param", "note"):
                    _add_tag_if_nonempty(key)
            elif flavor == "crra":
                for key in ("crra_params", "note"):
                    _add_tag_if_nonempty(key)
            else:  # flavor 'none' (e.g. BAU) or unknown
                _add_tag_if_nonempty("note")

        # Treat FS runs with constant-rho discounting (fs_disc_param like "rho20")
        # as discounting adjustments as well.
        fs_disc_tag = meta_row.get("fs_disc_param")
        if (
            flavor == "fs"
            and isinstance(fs_disc_tag, str)
            and fs_disc_tag.strip().lower().startswith("rho")
        ):
            meta_row["discounting_adjustment"] = True

        meta_rows.append(meta_row)

        # Collect all IAMC rows from this workbook as before
        all_rows.extend(_collect_from_workbook(wb_path))
        
    if not all_rows:
        # Return an empty IamDataFrame rather than failing
        logger.warning("No data harvested from folder %s; returning empty IamDataFrame.", folder_path)
        empty_df = pd.DataFrame(columns=["model", "scenario", "region", "variable", "unit"])
        return pyam.IamDataFrame(empty_df)

    df = pd.DataFrame(all_rows)

    # Ensure year columns are sorted numerically for readability
    non_year_cols = {"model", "scenario", "region", "variable", "unit"}
    year_cols = [c for c in df.columns if c not in non_year_cols]
    year_cols_sorted = sorted(year_cols, key=lambda x: int(str(x)))
    df = df[list(non_year_cols) + year_cols_sorted]

    # Construct the IamDataFrame with scenario-level meta if available
    if meta_rows:
        meta_df = (
            pd.DataFrame(meta_rows)
            .drop_duplicates(subset=["model", "scenario"])
            .set_index(["model", "scenario"])
        )

        # Normalize tag-style meta columns: replace NaN with a sentinel
        # string so that pyam filtering works with simple string matches.
        # This only touches keys coming from cfg["tags"], so it does NOT
        # affect boolean fields like 'negishi' or 'discounting_adjustment'.
        for col in tag_keys:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].astype("object").fillna("none")

        iamdf = pyam.IamDataFrame(df, meta=meta_df)
    else:
        iamdf = pyam.IamDataFrame(df)

    return iamdf
