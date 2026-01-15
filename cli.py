#!/usr/bin/env python3
# cli.py — Command-line entry point for RICE13_FS (also runnable as a module)
#
# Preferred usage (from repo root):
#   python -m RICE13_FS.cli -c config.yaml --log-level DEBUG
#
# Direct script execution is supported for local dev, but module execution is the
# intended/portable entry point because it exercises the package layout.

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import re

import yaml  # PyYAML

# --- Canonical imports (require proper package layout; also validates module execution) ---
from RICE13_FS.analysis.solver import run_analysis
from RICE13_FS.output.results import export_all
from RICE13_FS.common import utils as _U

# -----------------------------
# CLI & config loader
# -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RICE13_FS: run analyses and export results (supports mega_run)."
    )
    p.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: ./config.yaml)."
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)."
    )
    return p.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level config must be a mapping (dict). Got: {type(cfg)}")
    return cfg


# -----------------------------
# Validation helpers (lightweight)
# -----------------------------

def _require_key(cfg: Dict[str, Any], key: str) -> None:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")

def _require_file(path_str: str, key_name: str) -> None:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{key_name} not found: {p}")

def validate_horizon_decadal(config: Dict[str, Any]) -> None:
    """
    Horizon validation + legacy tstep guard.

    - T is limited by available input data (T <= 59).
    - `tstep` is retained for backward compatibility only. The model grid is fixed at 10-year
      steps; if `tstep` is provided, it must equal 10. We also normalize config["tstep"]=10
      for downstream year-mapping helpers / exports.
    """
    T = int(config.get("T", 59))
    if T > 59:
        raise ValueError("Time horizon T must be <= 59 (data only up to 2015+590 years).")
    # legacy-only: if user supplied tstep, it must be 10
    tstep = int(config.get("tstep", 10))
    if tstep != 10:
        raise NotImplementedError(
            "Subdecadal steps are not supported; if provided, tstep must be 10 (legacy-only)."
        )
    config["tstep"] = 10  # normalize downstream


def validate_negishi_config(config: dict) -> None:
    # Nothing to do if not used
    if not config.get("negishi_use", False):
        return
    negishi_source = str(config.get("negishi_source", "bau")).lower()
    if negishi_source not in {"bau", "file", "fs_after_disc"}:
        raise ValueError(f"Invalid 'negishi_source' value: {negishi_source}. Must be 'bau', 'file', or 'fs_after_disc'.")

    if negishi_source == "file":
        # At least one file path must be present (utility-specific paths allowed)
        if not (config.get("negishi_file_crra_path") or config.get("negishi_file_fs_path")):
            raise ValueError("negishi_source='file' but no negishi file path(s) provided.")

def validate_coalition_config_basic(config: Dict[str, Any]) -> None:
    """
    Policy + syntax validation that does NOT depend on regions.
    - mega_run must be present.
    - coalition must be provided (no defaults).
    - Allowed forms: 'all'/'*' (only with mega_run=true), 'GRAND',
      0/1 bitmask (any length), or comma-separated tokens (membership
      validated later in solver after regions are known).
    """
    if "mega_run" not in config:
        raise ValueError("Config must include mega_run (true/false).")
    mr = bool(config["mega_run"])
    raw = config.get("coalition", None)
    s = (str(raw).strip() if raw is not None else "")
    if not s:
        raise ValueError(
            "Missing 'coalition'. Provide 'all' (only with mega_run=true), 'GRAND', "
            "a 0/1 bitmask, or a comma list like 'US,EU'."
        )
    if s.lower() in {"all", "*"} and not mr:
        raise ValueError("coalition='all' is only allowed when mega_run=true.")
    if s.upper() == "GRAND":
        return
    if re.fullmatch(r"[01]+", s):
        # length is checked later once we know len(regions)
        if "1" not in s:
            raise ValueError("Invalid coalition bitmask: must contain at least one '1'.")
        return
    # comma-separated tokens (validate membership later)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if parts:
        return
    raise ValueError(
        "Invalid coalition string. Use 'all', 'GRAND', a 0/1 bitmask, or a comma list like 'US,EU'."
    )

def validate_cache_config_mandatory(config: dict) -> None:
    """
    Cache is mandatory in the new pipeline.
    Required keys:
      - cache_dir (path will be resolved to absolute)
    Optional:
      - cache_namespace (default 'default')
      - cache_allow_mismatch (bool, default False)

    Note: the coalition exporter treats the cache as the single source of truth.
    """
    cache_dir = config.get("cache_dir")
    if not cache_dir:
        raise ValueError("cache_dir is required (cache is mandatory).")
    config["cache_dir"] = str(Path(cache_dir).resolve())
    ns = str(config.get("cache_namespace", "")).strip()
    config["cache_namespace"] = ns or "default"
    config["cache_allow_mismatch"] = bool(config.get("cache_allow_mismatch", False))


def validate_misc(config: dict) -> None:
    """
    Validate small knobs we added:
      - diagnostics_level: boolean (controls *to-disk* diagnostics only; not logging verbosity)
      - stability_eps > 0 (default 1e-6)
      - parallel knob (int workers): 1 = sequential, >1 = processes (clamped to CPU limit)
    """
    if "diagnostics_level" not in config:
        config["diagnostics_level"] = False
    if not isinstance(config["diagnostics_level"], bool):
        raise ValueError("diagnostics_level must be boolean true/false")

    eps = float(config.get("stability_eps", 1e-6))
    if not (eps > 0.0):
        raise ValueError("stability_eps must be > 0.")
    config["stability_eps"] = eps

    # integer workers (safe clamp)
    requested = int(config.get("parallel", 1))
    if requested < 1:
        requested = 1
    cpu_max = max(1, (os.cpu_count() or 2) - 1)
    workers = min(requested, cpu_max)
    if requested != workers:
        logging.warning("parallel workers requested=%d → clamped to %d (cpu limit).", requested, workers)
    config["parallel"] = workers

def validate_fs_discount_config(config: dict) -> None:
    """
    FS discounting (Fehr–Schmidt runs only) – lightweight validation & normalization.
    Keys:
      - fs_disc_enabled: bool
      - fs_disc_mode: 'off' | 'file' | 'one_pass' | 'two_pass'
      - fs_disc_file: path (required iff mode == 'file')
   Behavior:
      - If fs_disc_enabled is false, force mode='off' and return.
      - If enabled and mode == 'file', verify the CSV exists and resolve to abs path.

    Note: this CLI currently requires fs_disc_mode to be present as a key even when disabled
    (strict config style). If you want "omit when disabled" behavior, relax the _require_key.
    """
    enabled = bool(config.get("fs_disc_enabled", False))
    config["fs_disc_enabled"] = enabled

    _require_key(config, "fs_disc_mode")
    mode = str(config.get("fs_disc_mode")).strip().lower()
    if not enabled:
       # If disabled, ignore mode details and hard-set to 'off'
        config["fs_disc_mode"] = "off"
        return

    allowed_modes = {"off", "file", "one_pass", "two_pass"}
    if mode not in allowed_modes:
        raise ValueError(f"fs_disc_mode must be one of {sorted(allowed_modes)}, got: {mode!r}")
    config["fs_disc_mode"] = mode

    if mode == "file":
        _require_key(config, "fs_disc_file")
        p = Path(str(config["fs_disc_file"]))
        if not p.exists():
            raise FileNotFoundError(f"fs_disc_file not found: {p}")
        config["fs_disc_file"] = str(p.resolve())


def _policy_equivalent_for_alignment(s_fs: str, s_crra: str) -> bool:
    """
    Return True if FS and CRRA use the *same underlying S policy* for alignment.
      - exact match
      - CRRA 'optimal' ≡ FS 'crra'
    """
    a = (s_fs or "").strip().lower()
    b = (s_crra or "").strip().lower()
    if a == b:
        return True
    return (a == "crra" and b == "optimal")

def _files_match(a: str, b: str) -> bool:
    if not a and not b:
        return True
    if bool(a) ^ bool(b):
        return False
    return Path(a).resolve() == Path(b).resolve()


def validate_fs_alignment_guards(config: dict) -> None:
    """
    Enforce invariants when using CRRA-aligned FS discounting (one_pass / two_pass):
      A) The CRRA run for the same regime must be enabled (anchor required).
      B) The CRRA and FS runs for that regime must use the same savings policy,
         unless fs_disc_mode == 'file' (external series, independent of anchor).
    Regimes covered independently: planner, nash, coalitions.
    """
    if not config.get("fs_disc_enabled", False):
        return
    mode = config.get("fs_disc_mode", "off")
    if mode not in {"one_pass", "two_pass", "file"}:
        return

    # Helper to check a single regime
    def _check_regime(run_fs_flag: str, run_crra_flag: str,
                      fs_mode_key: str, crra_mode_key: str,
                      fs_file_key: str, crra_file_key: str,
                      regime_name: str) -> None:
        if not bool(config.get(run_fs_flag, False)):
            return
        if mode in {"one_pass", "two_pass"}:
            # (A) CRRA anchor must be runnable
            if not bool(config.get(run_crra_flag, False)):
                raise ValueError(
                    f"FS discount alignment ({mode}) requires a CRRA anchor for {regime_name}. "
                    f"Please set {run_crra_flag}: true."
                )
            # (B) Savings policy must match
            s_fs = config.get(fs_mode_key, "")
            s_crra = config.get(crra_mode_key, "")
            if not _policy_equivalent_for_alignment(s_fs, s_crra):
                raise ValueError(
                    f"FS discount alignment requires the same savings policy as its CRRA anchor "
                    f"for {regime_name}. Found {crra_mode_key}={s_crra!r} vs {fs_mode_key}={s_fs!r}."
                )
            # If both are 'file', ensure the file paths match
            if str(s_fs).strip().lower() == "file" and str(s_crra).strip().lower() == "file":
                # require same file paths
                f_fs = config.get(fs_file_key, "")
                f_crra = config.get(crra_file_key, "")
                if not _files_match(f_fs, f_crra):
                    raise ValueError(
                        f"When using S_mode='file' for {regime_name}, CRRA and FS must point to the same file. "
                        f"Got {crra_file_key}={f_crra!r} vs {fs_file_key}={f_fs!r}."
                    )
        else:
            # mode == 'file' → alignment decoupled from anchors; no S-policy guard.
            return

    # Planner
    _check_regime(
        run_fs_flag="run_planner_fs",
        run_crra_flag="run_planner_crra",
        fs_mode_key="planner_fs_S_mode",
        crra_mode_key="planner_crra_S_mode",
        fs_file_key="planner_fs_S_file",
        crra_file_key="planner_crra_S_file",
        regime_name="planner",
    )
    # Nash
    _check_regime(
        run_fs_flag="run_nash_fs",
        run_crra_flag="run_nash_crra",
        fs_mode_key="nash_fs_S_mode",
        crra_mode_key="nash_crra_S_mode",
        fs_file_key="nash_fs_S_file",
        crra_file_key="nash_crra_S_file",
        regime_name="nash",
    )
    # Coalitions
    _check_regime(
        run_fs_flag="run_coalition_fs",
        run_crra_flag="run_coalition_crra",
        fs_mode_key="coalition_fs_S_mode",
        crra_mode_key="coalition_crra_S_mode",
        fs_file_key="coalition_fs_S_file",
        crra_file_key="coalition_crra_S_file",
        regime_name="coalitions",
    )


def validate_coalition_run_flags(config: Dict[str, Any]) -> None:
    """
    Require explicit coalition run flags; no implicit defaults.
    """
    _require_key(config, "run_coalition_crra")
    _require_key(config, "run_coalition_fs")
    # coerce to bool explicitly (reject non-bool types if you prefer)
    config["run_coalition_crra"] = bool(config["run_coalition_crra"])
    config["run_coalition_fs"] = bool(config["run_coalition_fs"])

def validate_coalition_S_config_strict(config: Dict[str, Any]) -> None:
    """
    Strict per-utility coalition S-policy validation; no shared/fallback keys.
    Requires:
      - coalition_crra_S_mode (and coalition_crra_S_file if 'file')
      - coalition_fs_S_mode   (and coalition_fs_S_file   if 'file')
    If a coalition S mode references planner_* modes, require the corresponding
    planner keys to exist explicitly.
    """
    # Require the keys to exist
    for key in ("coalition_crra_S_mode", "coalition_fs_S_mode"):
        _require_key(config, key)

    allowed = {"optimal", "bau", "file", "planner_crra", "planner_fs"}
    crra_mode = str(config["coalition_crra_S_mode"]).strip().lower()
    fs_mode   = str(config["coalition_fs_S_mode"]).strip().lower()
    if crra_mode not in allowed:
        raise ValueError(f"coalition_crra_S_mode must be one of {sorted(allowed)}, got: {crra_mode!r}")
    if fs_mode not in allowed:
        raise ValueError(f"coalition_fs_S_mode must be one of {sorted(allowed)}, got: {fs_mode!r}")

    # File requirements
    if crra_mode == "file":
        _require_key(config, "coalition_crra_S_file")
        _require_file(str(config["coalition_crra_S_file"]), "coalition_crra_S_file")
        config["coalition_crra_S_file"] = str(Path(config["coalition_crra_S_file"]).resolve())
    if fs_mode == "file":
        _require_key(config, "coalition_fs_S_file")
        _require_file(str(config["coalition_fs_S_file"]), "coalition_fs_S_file")
        config["coalition_fs_S_file"] = str(Path(config["coalition_fs_S_file"]).resolve())

    # If coalition modes reference planner modes, require planner keys explicitly
    def _require_planner_keys(prefix: str) -> None:
        mode_key = f"{prefix}_S_mode"
        _require_key(config, mode_key)
        m = str(config[mode_key]).strip().lower()
        # If planner mode uses 'file', also require the file
        if m == "file":
            file_key = f"{prefix}_S_file"
            _require_key(config, file_key)
            _require_file(str(config[file_key]), file_key)
            config[file_key] = str(Path(config[file_key]).resolve())

    if crra_mode in {"planner_crra", "planner_fs"}:
        _require_planner_keys("planner_crra" if crra_mode == "planner_crra" else "planner_fs")
    if fs_mode in {"planner_crra", "planner_fs"}:
        _require_planner_keys("planner_crra" if fs_mode == "planner_crra" else "planner_fs")



# -----------------------------
# Main
# -----------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load + validate config
    try:
        config = load_config(args.config)
        # policy + syntax (no regions needed)
        validate_coalition_config_basic(config)
        validate_coalition_run_flags(config)
        validate_horizon_decadal(config)
        validate_negishi_config(config)
        validate_cache_config_mandatory(config)
        validate_misc(config)
        validate_fs_discount_config(config)
        validate_coalition_S_config_strict(config)
        validate_fs_alignment_guards(config)
        # flip the global diagnostics switch for all solver modules
        _U.DIAGNOSTICS_ON = bool(config["diagnostics_level"])
        logging.info("Diagnostics-to-disk: %s", "ON" if _U.DIAGNOSTICS_ON else "OFF")
    except Exception as e:
        logging.exception("Configuration error: %s", e)
        sys.exit(2)

    # Resolve output/diagnostics dirs and ensure they exist
    project_root = Path(config.get("project_root", ".")).resolve()
    results_dir = Path(config.get("results_dir")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path(config.get("diagnostics_dir", project_root / "Diagnostics")).resolve()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.get("output_dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coalition_spec = str(config.get("coalition", "")).strip() or "<none>"
    logging.info("Configured: mega_run=%s, coalition=%s", bool(config["mega_run"]), coalition_spec)

    # Run analysis
    try:
        logging.info("Running analysis (coalition=%s).", coalition_spec)
        analysis = run_analysis(config, diagnostics_dir=diagnostics_dir)
    except SystemExit:
        raise
    except Exception as e:
        logging.exception("Analysis failed: %s", e)
        sys.exit(3)

    # Export tabular + coalition workbooks (+ overview for mega runs)
    try:
        meta = analysis.meta or {}
        
        # --- Enrich config for export (display/spec consistency only; does not affect solving) ---
        # We inject BAU S and Negishi weights so export-time spec_id reconstruction matches
        # the spec_ids used during solving/caching.
        export_cfg = dict(config)

        # (1) If coalitions use BAU S at export time, inject BAU S so the exporter
        #     doesn't try to access a missing params object.
        if getattr(analysis, "bau", None) and isinstance(analysis.bau, dict):
            # prefer 'S_exogenous'; fall back to 'S' if present
            _bau_S = analysis.bau.get("S_exogenous")
            if _bau_S is None:
                _bau_S = analysis.bau.get("S")
            if _bau_S is not None:
                export_cfg["bau_saving_rates"] = _bau_S
            else:
                logging.debug(
                    "BAU present but no S_exogenous/S DataFrame found; skipping BAU S injection."
                )

        # (2) If Negishi is enabled, inject weights so spec-ids (with Negishi digest)
        #     match the ones used during solving/caching.
        if export_cfg.get("negishi_use", False):
            try:
                w_crra_path = output_dir / "negishi_weights_crra.csv"
                if w_crra_path.exists():
                    export_cfg["negishi_weights"] = pd.read_csv(w_crra_path, index_col=0)
                else:
                    logging.debug("Negishi CRRA weights CSV not found at %s", w_crra_path)
            except Exception as e:
                logging.warning("Could not load Negishi CRRA weights: %s", e)
            try:
                w_fs_path = output_dir/ "negishi_weights_fs.csv"
                if w_fs_path.exists():
                    export_cfg["negishi_weights_fs"] = pd.read_csv(w_fs_path, index_col=0)
                else:
                    logging.debug("Negishi FS weights CSV not found at %s", w_fs_path)
            except Exception as e:
                logging.warning("Could not load Negishi FS weights: %s", e)    
    
        export_all(
            coop_solution=analysis.coop,
            noncoop_solution=analysis.noncoop,
            coalitions=analysis.coalitions,
            countries=list(meta["countries"]),
            periods=list(meta["periods"]),
            T=max(meta["periods"]),
            output_dir=results_dir,
            L=meta.get("L"),
            tstep=int(meta.get("tstep", config.get("tstep", 10))),
            base_year=int(meta["base_year"]),
            backstop_switch_year=(int(meta["backstop_switch_year"]) if meta.get("backstop_switch_year") is not None else None),
            bau_solution=analysis.bau,
            config=export_cfg,
        )
    except SystemExit:
        raise
    except Exception as e:
        logging.exception("Export failed: %s", e)
        sys.exit(3)

    logging.info(
        "Done. Results: %s | Diagnostics: %s",
        output_dir.resolve(), diagnostics_dir.resolve()
    )


if __name__ == "__main__":
    # Runs when executed as a script *or* as a module via `python -m RICE13_FS.cli`.
    #
    # On Linux, default is "fork" which can deadlock with BLAS/OpenMP/SQLite.
    # Force a clean interpreter per worker and avoid thread over-subscription.
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # or "forkserver"
    except RuntimeError:
        # start method already set elsewhere; that's fine
        pass

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    main()
